// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Lord Boot backend: the kernel side of the Lord Boot bootloader (the `lord-boot` package).
//!
//! Provides the kernel entry point [`_start`], checks the hand-off against the protocol
//! (`protocol.rs`, spec: `docs/19-lxf-boot-protocol.md`) and translates it into the
//! bootloader-independent [`BootInfo`](crate::boot::BootInfo). The application processors
//! are started by [`ap`] (INIT-SIPI), because this bootloader does not park them.

pub mod protocol;

mod ap;

use core::arch::asm;
use core::mem::size_of;

use self::protocol::{self as proto, CpuEntry, KernelBoot, MemEntry};
use super::{ApEntry, BootCpu, BootInfo, BootloaderInfo, Firmware, SmpInfo};
use crate::hal::display::{Channel, FramebufferInfo};
use crate::hal::memory::{Region, RegionKind};
use crate::hal::sync::StaticCell;

/// Capacity of the kernel-owned copy of the memory map (the loader provides up to 1024).
const MAX_REGIONS: usize = 1024;
static REGIONS: StaticCell<[Region; MAX_REGIONS]> =
    StaticCell::new([Region { base: 0, length: 0, kind: RegionKind::Reserved }; MAX_REGIONS]);

const MAX_CPUS_LISTED: usize = 256;
static CPUS: StaticCell<[BootCpu; MAX_CPUS_LISTED]> =
    StaticCell::new([BootCpu { lapic_id: 0, processor_id: 0, handle: 0 }; MAX_CPUS_LISTED]);

/// Stops this CPU after a message on COM1 (the screen is not set up yet).
fn die(msg: &str) -> ! {
    crate::hal::serial::init();
    error!("boot: LXF hand-off rejected: {}", msg);
    crate::hal::cpu::halt_forever()
}

/// Kernel entry point (`ENTRY(_start)` in `linker.ld`). Lord Boot jumps here in 64-bit
/// mode, in the higher half, with interrupts off, `RDI` = the [`KernelBoot`] block and a
/// 64 KiB stack.
#[unsafe(no_mangle)]
pub extern "C" fn _start(info: *const KernelBoot) -> ! {
    if info.is_null() {
        die("no boot info pointer");
    }
    // SAFETY: the loader maps the block (and everything it points to) for the kernel's whole
    // life; it is `BOOTLOADER_RECLAIMABLE`, which the PMM never frees.
    let b: &'static KernelBoot = unsafe { &*info };
    if b.magic != proto::MAGIC {
        die("bad magic");
    }
    if b.version != proto::VERSION {
        die("unsupported protocol version");
    }
    if (b.size as usize) < size_of::<KernelBoot>() {
        die("boot info block is smaller than this kernel expects");
    }
    // The loader keeps a transient identity map of its hand-off stub in PML4 slot 0 (so the CR3
    // switch could happen) and every other low-half slot empty. Drop it: the user half of the
    // address space must start out empty.
    // SAFETY: the PML4 is reachable through the HHDM; we run in the higher half.
    unsafe {
        let cr3: u64;
        asm!("mov {}, cr3", out(reg) cr3, options(nomem, nostack, preserves_flags));
        let pml4 = (b.hhdm_offset + (cr3 & 0x000f_ffff_ffff_f000)) as *mut u64;
        pml4.write_volatile(0);
        asm!("mov cr3, {}", in(reg) cr3, options(nostack)); // flush the TLB
    }

    ap::configure(b.hhdm_offset, b.ap_boot_phys, b.ap_pml4_phys);

    let boot_info = BootInfo {
        bootloader: Some(BootloaderInfo {
            name: proto::ascii(&b.loader_name),
            version: proto::ascii(&b.loader_version),
            protocol: "lxf",
            revision_wanted: Some(u64::from(proto::VERSION)),
            revision_loaded: Some(u64::from(b.version)),
        }),
        firmware: Some(Firmware::Efi64),
        hhdm_offset: Some(b.hhdm_offset),
        memory_map: memory_map(b),
        memory_map_dropped: b.memmap_dropped as usize,
        framebuffer: framebuffer(b),
        rsdp_phys: (b.flags & proto::FLAG_RSDP != 0 && b.rsdp_phys != 0).then_some(b.rsdp_phys),
        smp: smp(b),
    };
    // `_start` never returns, so `boot_info` lives as long as the kernel runs.
    crate::kmain(&boot_info)
}

fn region_kind(raw: u32) -> RegionKind {
    match raw {
        proto::kind::USABLE => RegionKind::Usable,
        proto::kind::RESERVED => RegionKind::Reserved,
        proto::kind::ACPI_RECLAIMABLE => RegionKind::AcpiReclaimable,
        proto::kind::ACPI_NVS => RegionKind::AcpiNvs,
        proto::kind::BAD_MEMORY => RegionKind::BadMemory,
        proto::kind::BOOTLOADER_RECLAIMABLE => RegionKind::BootloaderReclaimable,
        proto::kind::KERNEL_AND_MODULES => RegionKind::KernelAndModules,
        proto::kind::FRAMEBUFFER => RegionKind::Framebuffer,
        proto::kind::RESERVED_MAPPED => RegionKind::ReservedMapped,
        _ => RegionKind::Unknown,
    }
}

/// Copies the loader's memory map into kernel-owned [`Region`]s (once, on the boot CPU).
fn memory_map(b: &KernelBoot) -> &'static [Region] {
    if b.memmap == 0 || b.memmap_len == 0 {
        return &[];
    }
    // SAFETY: the loader guarantees `memmap_len` valid entries at `memmap`.
    let src = unsafe { core::slice::from_raw_parts(b.memmap as *const MemEntry, b.memmap_len as usize) };
    // SAFETY: single boot CPU, called once; nothing else references REGIONS yet.
    let out = unsafe { &mut *REGIONS.get() };
    let n = src.len().min(MAX_REGIONS);
    for (slot, e) in out.iter_mut().zip(src.iter()) {
        *slot = Region { base: e.base, length: e.length, kind: region_kind(e.kind) };
    }
    &out[..n]
}

fn framebuffer(b: &KernelBoot) -> Option<FramebufferInfo> {
    if b.flags & proto::FLAG_FRAMEBUFFER == 0 || b.fb.address == 0 {
        return None;
    }
    let f = &b.fb;
    Some(FramebufferInfo {
        address: f.address as *mut u8,
        width: f.width as usize,
        height: f.height as usize,
        pitch: f.pitch as usize,
        bpp: f.bpp,
        red: Channel { size: f.red_size, shift: f.red_shift },
        green: Channel { size: f.green_size, shift: f.green_shift },
        blue: Channel { size: f.blue_size, shift: f.blue_shift },
    })
}

/// The CPU list comes from the ACPI MADT (read by the loader). Without the AP start-up pages
/// there is no way to wake the others, so report no MP info and run on the boot CPU.
fn smp(b: &KernelBoot) -> Option<SmpInfo> {
    if b.flags & proto::FLAG_SMP == 0 || b.flags & proto::FLAG_AP_BOOT == 0 || b.cpu_count == 0 {
        return None;
    }
    // SAFETY: the loader guarantees `cpu_count` valid entries at `cpus`.
    let src = unsafe { core::slice::from_raw_parts(b.cpus as *const CpuEntry, b.cpu_count as usize) };
    // SAFETY: single boot CPU, called once.
    let out = unsafe { &mut *CPUS.get() };
    let n = src.len().min(MAX_CPUS_LISTED);
    for (i, (slot, c)) in out.iter_mut().zip(src.iter()).enumerate() {
        *slot = BootCpu { lapic_id: c.lapic_id, processor_id: c.processor_id, handle: i };
    }
    Some(SmpInfo {
        bsp_lapic_id: b.bsp_lapic_id,
        x2apic: false,
        cpus: &out[..n],
        start_cpu: start_cpu as fn(&BootCpu, usize, ApEntry),
    })
}

fn start_cpu(cpu: &BootCpu, index: usize, entry: ApEntry) {
    ap::start_cpu(cpu, index, entry)
}
