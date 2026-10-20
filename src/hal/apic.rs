// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Local APIC (LAPIC) in xAPIC (memory-mapped) mode.
//!
//! Bring-up order, mirrored in [`init_bsp`]:
//!
//! 1. The legacy 8259 PIC has already been masked (`hal::pic`).
//! 2. **Read** the APIC's location and state from `IA32_APIC_BASE`.
//! 3. **Unmount** the bootloader's mapping of the APIC page, if there is one.
//!    The bootloader maps physical memory write-back; mapping device registers
//!    both write-back (bootloader alias) *and* uncacheable (ours) would be
//!    memory-type aliasing, which the architecture leaves undefined.
//! 4. **Mount** the register page uncacheable in the MMIO window and read the
//!    ID / version / LVT registers.
//! 5. **Turn on** the APIC: software-enable bit and spurious vector in `SVR`,
//!    task priority 0, every local vector table entry masked.
//!
//! Application processors call [`init_ap`]: the mapping is shared (one set of
//! page tables), but each CPU has its own `IA32_APIC_BASE` and register file,
//! so each must enable *its own* LAPIC.

use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use super::idt::{self, InterruptFrame};
use super::msr::{self, IA32_APIC_BASE};
use super::{cpu, memory, paging, percpu, MAX_CPUS};

/// Vector delivered for spurious interrupts. Low nibble must be 1s.
pub const SPURIOUS_VECTOR: u8 = 0xFF;
/// Vector used by the self-IPI self-test.
pub const SELF_TEST_VECTOR: u8 = 0x40;

// Register offsets from the LAPIC base (Intel SDM Vol. 3, table 11-1).
pub const REG_ID: usize = 0x020;
pub const REG_VERSION: usize = 0x030;
pub const REG_TPR: usize = 0x080;
pub const REG_EOI: usize = 0x0B0;
pub const REG_LDR: usize = 0x0D0;
pub const REG_DFR: usize = 0x0E0;
pub const REG_SVR: usize = 0x0F0;
pub const REG_ESR: usize = 0x280;
pub const REG_LVT_CMCI: usize = 0x2F0;
pub const REG_ICR_LOW: usize = 0x300;
pub const REG_ICR_HIGH: usize = 0x310;
pub const REG_LVT_TIMER: usize = 0x320;
pub const REG_LVT_THERMAL: usize = 0x330;
pub const REG_LVT_PERF: usize = 0x340;
pub const REG_LVT_LINT0: usize = 0x350;
pub const REG_LVT_LINT1: usize = 0x360;
pub const REG_LVT_ERROR: usize = 0x370;
pub const REG_TIMER_INITIAL: usize = 0x380;
pub const REG_TIMER_CURRENT: usize = 0x390;
pub const REG_TIMER_DIVIDE: usize = 0x3E0;

const LVT_MASKED: u32 = 1 << 16;
const SVR_ENABLE: u32 = 1 << 8;
const ICR_DELIVERY_PENDING: u32 = 1 << 12;
const ICR_LEVEL_ASSERT: u32 = 1 << 14;
const ICR_SHORTHAND_SELF: u32 = 0b01 << 18;

const APIC_BASE_BSP: u64 = 1 << 8;
const APIC_BASE_X2APIC: u64 = 1 << 10;
const APIC_BASE_ENABLE: u64 = 1 << 11;
const APIC_BASE_ADDR_MASK: u64 = 0x000f_ffff_ffff_f000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApicError {
    /// CPUID says this CPU has no local APIC.
    NotSupported,
    /// The CPU is in x2APIC (MSR) mode; this driver is xAPIC/MMIO only.
    X2ApicActive,
    /// `IA32_APIC_BASE` still says "disabled" after we tried to enable it.
    CannotEnable,
    /// An AP tried to start before the BSP mounted the registers.
    NotMounted,
    /// Mapping the register page failed.
    Paging(paging::PagingError),
    /// The software-enable bit did not stick.
    EnableFailed,
    /// The LAPIC ID read from hardware differs from the bootloader's.
    IdMismatch { hardware: u32, expected: u32 },
}

/// Decoded `IA32_APIC_BASE`.
#[derive(Clone, Copy)]
pub struct ApicBase {
    /// Physical address of the register page (normally 0xFEE00000).
    pub phys: u64,
    pub enabled: bool,
    pub bsp: bool,
    pub x2apic: bool,
    raw: u64,
}

pub fn read_base() -> ApicBase {
    // SAFETY: IA32_APIC_BASE exists on every CPU with an APIC.
    let raw = unsafe { msr::rdmsr(IA32_APIC_BASE) };
    ApicBase {
        phys: raw & APIC_BASE_ADDR_MASK,
        enabled: raw & APIC_BASE_ENABLE != 0,
        bsp: raw & APIC_BASE_BSP != 0,
        x2apic: raw & APIC_BASE_X2APIC != 0,
        raw,
    }
}

/// Virtual address of the mounted register page (0 = not mounted yet).
static REGS: AtomicUsize = AtomicUsize::new(0);

pub fn is_mounted() -> bool {
    REGS.load(Ordering::Acquire) != 0
}

/// Reads a 32-bit LAPIC register of the *current* CPU.
pub fn read(reg: usize) -> u32 {
    let base = REGS.load(Ordering::Acquire);
    assert!(base != 0, "LAPIC registers are not mounted");
    // SAFETY: `base` is the uncacheable mapping of the LAPIC page and `reg`
    // is a 16-byte-aligned register offset inside it.
    unsafe { ((base + reg) as *const u32).read_volatile() }
}

/// Writes a 32-bit LAPIC register of the *current* CPU.
pub fn write(reg: usize, value: u32) {
    let base = REGS.load(Ordering::Acquire);
    assert!(base != 0, "LAPIC registers are not mounted");
    // SAFETY: see `read`.
    unsafe { ((base + reg) as *mut u32).write_volatile(value) }
}

/// This CPU's local APIC ID (xAPIC: bits 31..24 of the ID register).
pub fn id() -> u32 {
    read(REG_ID) >> 24
}

pub fn version() -> u32 {
    read(REG_VERSION)
}

/// Index of the last LVT entry (version register bits 23..16).
pub fn max_lvt_entry() -> u32 {
    (version() >> 16) & 0xFF
}

/// Signals end-of-interrupt for the interrupt currently being serviced.
pub fn eoi() {
    write(REG_EOI, 0);
}

fn wait_icr_idle() {
    for _ in 0..1_000_000 {
        if read(REG_ICR_LOW) & ICR_DELIVERY_PENDING == 0 {
            return;
        }
        cpu::pause();
    }
}

/// Sends a fixed-delivery IPI with `vector` to the calling CPU itself.
pub fn send_self_ipi(vector: u8) {
    wait_icr_idle();
    write(REG_ICR_HIGH, 0);
    write(REG_ICR_LOW, vector as u32 | ICR_LEVEL_ASSERT | ICR_SHORTHAND_SELF);
}

// ----------------------------------------------------------------------
// Handlers and self-test
// ----------------------------------------------------------------------

const NO_HITS: AtomicU32 = AtomicU32::new(0);
static SELF_IPI_HITS: [AtomicU32; MAX_CPUS] = [NO_HITS; MAX_CPUS];
static SPURIOUS_HITS: AtomicU32 = AtomicU32::new(0);

fn spurious_handler(_frame: &mut InterruptFrame) {
    // Spurious interrupts must NOT be acknowledged with an EOI.
    SPURIOUS_HITS.fetch_add(1, Ordering::Relaxed);
}

fn self_test_handler(_frame: &mut InterruptFrame) {
    SELF_IPI_HITS[percpu::current_index()].fetch_add(1, Ordering::AcqRel);
    eoi();
}

pub fn spurious_count() -> u32 {
    SPURIOUS_HITS.load(Ordering::Relaxed)
}

/// Proves the LAPIC delivers interrupts: briefly enables interrupts, sends
/// ourselves an IPI and waits for the handler (which sends the EOI) to run.
/// Interrupts are masked again on return.
pub fn self_ipi_test() -> bool {
    let cpu_index = percpu::current_index();
    let before = SELF_IPI_HITS[cpu_index].load(Ordering::Acquire);
    cpu::enable_interrupts();
    send_self_ipi(SELF_TEST_VECTOR);
    let deadline = cpu::rdtsc().saturating_add(2_000_000_000);
    while SELF_IPI_HITS[cpu_index].load(Ordering::Acquire) == before && cpu::rdtsc() < deadline {
        cpu::pause();
    }
    cpu::disable_interrupts();
    SELF_IPI_HITS[cpu_index].load(Ordering::Acquire) > before
}

// ----------------------------------------------------------------------
// Bring-up
// ----------------------------------------------------------------------

/// Hardware-enables this CPU's LAPIC in `IA32_APIC_BASE` if needed.
fn ensure_hardware_enabled() -> Result<ApicBase, ApicError> {
    if !cpu::has_apic() {
        return Err(ApicError::NotSupported);
    }
    let mut base = read_base();
    if base.x2apic {
        return Err(ApicError::X2ApicActive);
    }
    if !base.enabled {
        // SAFETY: setting the global-enable bit, preserving everything else.
        unsafe { msr::wrmsr(IA32_APIC_BASE, base.raw | APIC_BASE_ENABLE) };
        base = read_base();
        if !base.enabled {
            return Err(ApicError::CannotEnable);
        }
    }
    Ok(base)
}

/// Masks every local vector table entry, clears errors, sets TPR to 0 and
/// software-enables the LAPIC with [`SPURIOUS_VECTOR`].
fn enable_local() -> Result<(), ApicError> {
    write(REG_TPR, 0); // accept every priority class

    // Mask the local sources we don't use yet. LINT1 (the NMI pin) is left as
    // the firmware configured it.
    for reg in [REG_LVT_TIMER, REG_LVT_THERMAL, REG_LVT_PERF, REG_LVT_LINT0, REG_LVT_ERROR] {
        write(reg, LVT_MASKED | 0xFF);
    }
    if max_lvt_entry() >= 6 {
        write(REG_LVT_CMCI, LVT_MASKED | 0xFF);
    }

    write(REG_ESR, 0); // writing arms the error status register...
    let _ = read(REG_ESR); // ...and reading latches/clears it

    write(REG_SVR, SVR_ENABLE | SPURIOUS_VECTOR as u32);
    if read(REG_SVR) & SVR_ENABLE == 0 {
        return Err(ApicError::EnableFailed);
    }
    Ok(())
}

/// Brings the boot CPU's LAPIC up. Needs `memory`, the IDT and per-CPU data
/// to be initialised, and must run **before** the other CPUs are started
/// (unmounting an existing mapping is only safe while no other CPU runs).
pub fn init_bsp() -> Result<(), ApicError> {
    // --- read -----------------------------------------------------------
    let base = ensure_hardware_enabled()?;
    debug!(
        "apic: IA32_APIC_BASE .... phys {:#x}, {}, {}, xAPIC mode",
        base.phys,
        if base.enabled { "enabled" } else { "disabled" },
        if base.bsp { "BSP" } else { "AP" }
    );

    // --- unmount ----------------------------------------------------------
    let alias = memory::phys_to_virt(base.phys) as u64;
    match paging::translate(alias) {
        Some(m) => {
            debug!("apic: bootloader mapping . {:#x}: {}", alias, m);
            match paging::unmap_page(alias) {
                Ok(u) => debug!(
                    "apic: unmounted ......... HHDM alias removed ({} large page split(s))",
                    u.splits
                ),
                Err(e) => warn!("apic: could not unmount HHDM alias: {:?}", e),
            }
            if paging::translate(alias).is_some() {
                warn!("apic: HHDM alias still mapped after unmount");
            }
        }
        None => debug!(
            "apic: bootloader mapping . none at {:#x} (HHDM is restrictive), nothing to unmount",
            alias
        ),
    }

    // --- mount (uncacheable) ----------------------------------------------
    let regs = paging::map_mmio(base.phys, 4096).map_err(ApicError::Paging)?;
    REGS.store(regs as usize, Ordering::Release);
    if let Some(m) = paging::translate(regs as u64) {
        debug!("apic: mounted ............ {:#x}: {}", regs as usize, m);
    }

    // --- read the registers (still disabled) ------------------------------
    let ver = version();
    debug!(
        "apic: id / version ...... id {}, version {:#04x} ({}), {} LVT entries",
        id(),
        ver & 0xFF,
        if ver & 0xFF >= 0x10 { "integrated APIC" } else { "82489DX discrete" },
        max_lvt_entry() + 1
    );
    debug!(
        "apic: before enable ..... SVR={:#x} TPR={:#x} LDR={:#x} DFR={:#x}",
        read(REG_SVR),
        read(REG_TPR),
        read(REG_LDR),
        read(REG_DFR)
    );
    debug!(
        "apic: LVT before ........ timer={:#x} lint0={:#x} lint1={:#x} error={:#x}",
        read(REG_LVT_TIMER),
        read(REG_LVT_LINT0),
        read(REG_LVT_LINT1),
        read(REG_LVT_ERROR)
    );

    // --- turn on ------------------------------------------------------------
    idt::register(SPURIOUS_VECTOR, spurious_handler);
    idt::register(SELF_TEST_VECTOR, self_test_handler);
    enable_local()?;
    debug!(
        "apic: enabled ........... SVR={:#x} (spurious vector {:#04x}), TPR=0, LVT masked",
        read(REG_SVR),
        SPURIOUS_VECTOR
    );
    Ok(())
}

/// Enables the calling application processor's own LAPIC. The BSP must have
/// mounted the registers first. Checks the hardware ID against the bootloader's.
pub fn init_ap(expected_lapic_id: u32) -> Result<(), ApicError> {
    if !is_mounted() {
        return Err(ApicError::NotMounted);
    }
    ensure_hardware_enabled()?;
    enable_local()?;
    let hardware = id();
    if hardware != expected_lapic_id {
        return Err(ApicError::IdMismatch { hardware, expected: expected_lapic_id });
    }
    Ok(())
}
