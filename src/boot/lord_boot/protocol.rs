// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! The **LXF kernel boot protocol**, version 1: what Lord Boot hands to an
//! LXF kernel image. Specification: `docs/19-lxf-boot-protocol.md`.
//!
//! This file exists as two byte-identical copies, one per package: the kernel backend
//! (`src/boot/lord_boot/protocol.rs` in lord-core) and the loader (`src/protocol.rs` in lord-boot).
//! If you change one, change the other and run `tests/check-protocol-sync.sh`
//! so the layout cannot drift. Keep it dependency-free `no_std` and `repr(C)`.
//!
//! State at the kernel entry point (`extern "C" fn _start(info: *const KernelBoot) -> !`):
//! * 64-bit mode, interrupts off, `RDI` = `info` (a higher-half-direct-map address),
//!   a 64 KiB stack in `RSP` (HHDM address), flat GDT, empty IDT.
//! * Paging on, `EFER.NXE` and `CR0.WP` set. The kernel segments are mapped where
//!   `.kseg` says (W^X); all RAM-like memory is mapped read/write, no-execute at
//!   `hhdm_offset + phys`.
//! * PML4 slot 0 still holds a transient identity map of the hand-off stub; the
//!   kernel backend removes it first thing. Every other low-half slot is empty.
//! * Boot services are gone. Memory the loader used stays reserved
//!   (`BOOTLOADER_RECLAIMABLE`); nothing frees it.
#![allow(dead_code)]

pub const MAGIC: u64 = u64::from_le_bytes(*b"LXFKERN\0");
pub const VERSION: u32 = 1;

/// Virtual base of the higher-half direct map the bootloader builds.
pub const HHDM_OFFSET: u64 = 0xffff_8000_0000_0000;

/// `MemEntry::kind` values (same order as the kernel's `hal::memory::RegionKind`).
pub mod kind {
    pub const USABLE: u32 = 0;
    pub const RESERVED: u32 = 1;
    pub const ACPI_RECLAIMABLE: u32 = 2;
    pub const ACPI_NVS: u32 = 3;
    pub const BAD_MEMORY: u32 = 4;
    pub const BOOTLOADER_RECLAIMABLE: u32 = 5;
    pub const KERNEL_AND_MODULES: u32 = 6;
    pub const FRAMEBUFFER: u32 = 7;
    pub const RESERVED_MAPPED: u32 = 8;
}

/// `KernelBoot::flags`
pub const FLAG_FRAMEBUFFER: u64 = 1 << 0;
pub const FLAG_RSDP: u64 = 1 << 1;
pub const FLAG_SMP: u64 = 1 << 2;
pub const FLAG_AP_BOOT: u64 = 1 << 3;

/// One physical memory range (page granular, sorted, non-overlapping, adjacent
/// ranges of the same kind merged).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MemEntry {
    pub base: u64,
    pub length: u64,
    pub kind: u32,
    pub _pad: u32,
}

/// Linear framebuffer (32 bits per pixel). `address` is a **virtual** address.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FbInfo {
    pub address: u64,
    pub width: u32,
    pub height: u32,
    /// Bytes per scanline.
    pub pitch: u32,
    pub bpp: u16,
    pub _pad0: u16,
    pub red_size: u8,
    pub red_shift: u8,
    pub green_size: u8,
    pub green_shift: u8,
    pub blue_size: u8,
    pub blue_shift: u8,
    pub _pad1: [u8; 2],
}

/// One logical CPU from the ACPI MADT.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CpuEntry {
    pub lapic_id: u32,
    pub processor_id: u32,
}

#[repr(C)]
pub struct KernelBoot {
    pub magic: u64,
    pub version: u32,
    /// `size_of::<KernelBoot>()` as the bootloader built it.
    pub size: u32,
    pub flags: u64,
    pub hhdm_offset: u64,
    /// Physical address of the active PML4.
    pub pml4_phys: u64,
    pub kernel_phys: u64,
    pub kernel_size: u64,
    /// Virtual address of `[MemEntry; memmap_len]`.
    pub memmap: u64,
    pub memmap_len: u64,
    /// Ranges that did not fit the loader's buffer.
    pub memmap_dropped: u64,
    pub fb: FbInfo,
    /// Physical address of the ACPI RSDP (`FLAG_RSDP`).
    pub rsdp_phys: u64,
    /// Virtual address of `[CpuEntry; cpu_count]` (`FLAG_SMP`), BSP included.
    pub cpus: u64,
    pub cpu_count: u64,
    pub bsp_lapic_id: u32,
    pub _pad2: u32,
    /// `FLAG_AP_BOOT`: a page below 1 MiB for the AP start-up trampoline, plus three
    /// pages below 4 GiB holding a transient PML4 whose slot 0 identity-maps the first
    /// 2 MiB (PML4 at `ap_pml4_phys`; the kernel copies its higher half into it).
    pub ap_boot_phys: u64,
    pub ap_pml4_phys: u64,
    /// NUL-padded ASCII.
    pub loader_name: [u8; 32],
    pub loader_version: [u8; 16],
}

/// Trailing NUL-padded ASCII -> str.
pub fn ascii(b: &[u8]) -> &str {
    let n = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    core::str::from_utf8(&b[..n]).unwrap_or("?")
}
