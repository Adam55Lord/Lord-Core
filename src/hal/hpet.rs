// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! High Precision Event Timer (HPET), used as Lord Core's **reference clock**.
//!
//! The HPET is a memory-mapped device with one free-running *main counter*
//! (at least 10 MHz, 32 or 64 bits wide, immune to CPU frequency changes and
//! sleep states) and 3+ *comparators*. Lord Core uses only the main counter: it is the
//! time base against which the LAPIC timer (and later the TSC) is calibrated.
//!
//! Comparator interrupts are deliberately **not** used: without an I/O APIC
//! (roadmap) there is nothing to route them, and the legacy-replacement mode
//! would take over IRQ0/IRQ8 from the PIT/RTC. We therefore switch it off and
//! mask every comparator, whatever state the firmware left them in.
//!
//! Bring-up order, mirrored in [`init`]:
//!
//! 1. **Find** the `HPET` ACPI table (`hal::acpi`) and read the register base.
//! 2. **Unmount** a bootloader alias of that page, if any (as for the LAPIC).
//! 3. **Mount** the 1 KiB register block uncacheable (`paging::map_mmio`).
//! 4. **Read** capabilities: revision, comparator count, counter width, period.
//! 5. **Reset** it: counter halted, legacy routing off, comparator interrupts off,
//!    counter cleared, then **start** the counter and check that it counts.
//!
//! Counter readings are `u64` ticks; convert with [`ticks_to_ns`]. A 32-bit
//! counter wraps after 2^32 ticks (~5 min at 14.3 MHz), so always measure
//! *differences* with [`elapsed_ticks`], which masks correctly.

use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use super::sync::IrqLock;
use super::{acpi, cpu, memory, paging};

// ---- Register offsets (HPET specification 1.0a, section 2.3) ---------------
const REG_CAPS: usize = 0x000;
const REG_CONFIG: usize = 0x010;
const REG_INT_STATUS: usize = 0x020;
const REG_COUNTER: usize = 0x0F0;
const REG_TIMER_BASE: usize = 0x100;
const TIMER_STRIDE: usize = 0x20;

/// Size of the register block we mount.
const REG_BLOCK_LEN: usize = 0x400;

// ---- General capabilities and ID --------------------------------------------
const CAP_COUNT_SIZE_64: u64 = 1 << 13;
const CAP_LEGACY_ROUTE: u64 = 1 << 15;

// ---- General configuration ----------------------------------------------------
const CFG_ENABLE: u64 = 1 << 0;
const CFG_LEGACY_ROUTE: u64 = 1 << 1;

// ---- Per-comparator configuration -----------------------------------------------
const TN_INT_ENABLE: u64 = 1 << 2;
const TN_FSB_ENABLE: u64 = 1 << 14;

/// The specification caps the counter period at 100 ns (10 MHz minimum).
const MAX_PERIOD_FS: u64 = 100_000_000;
const FS_PER_NS: u128 = 1_000_000;

// ---- ACPI HPET table layout (offsets from the start of the table) ---------------
const ACPI_BLOCK_ID: usize = 36;
const ACPI_ADDR_SPACE: usize = 40;
const ACPI_ADDRESS: usize = 44;
const ACPI_NUMBER: usize = 52;
const ACPI_MIN_TICK: usize = 53;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HpetError {
    /// The firmware does not describe an HPET (no ACPI, or no `HPET` table).
    NotPresent,
    /// ACPI lookup failed (bad checksum, mapping error, ...).
    Acpi(acpi::AcpiError),
    /// The table puts the registers somewhere other than system memory.
    UnsupportedAddressSpace(u8),
    /// The table's base address is zero.
    BadAddress,
    /// Mapping the register block failed.
    Paging(paging::PagingError),
    /// The period field is zero or exceeds the 100 ns the spec allows.
    BadPeriod(u64),
    /// The main counter did not advance after being enabled.
    NotCounting,
}

/// What [`init`] learned about the device.
#[derive(Clone, Copy)]
pub struct Info {
    /// Physical register base from the ACPI table.
    pub phys: u64,
    /// Virtual address of the uncacheable mapping.
    pub virt: usize,
    pub revision: u8,
    pub vendor: u16,
    /// Number of comparators (timers).
    pub timers: u32,
    /// 64-bit main counter?
    pub wide: bool,
    /// Can it replace the PIT/RTC interrupts (we keep that off)?
    pub legacy_capable: bool,
    /// Main counter period in femtoseconds.
    pub period_fs: u64,
    /// ACPI "HPET number" and minimum clock tick (periodic mode), for the log.
    pub number: u8,
    pub min_tick: u16,
    /// Did the bootloader map this page in the HHDM (and did we remove it)?
    pub alias_removed: bool,
}

impl Info {
    /// Counter frequency in Hz.
    pub fn frequency_hz(&self) -> u64 {
        1_000_000_000_000_000 / self.period_fs
    }
}

static BASE: AtomicUsize = AtomicUsize::new(0);
static PERIOD_FS: AtomicU64 = AtomicU64::new(0);
static WIDE: AtomicBool = AtomicBool::new(false);
static INFO: IrqLock<Option<Info>> = IrqLock::new(None);

// ----------------------------------------------------------------------
// Register access
// ----------------------------------------------------------------------

fn read64(off: usize) -> u64 {
    let base = BASE.load(Ordering::Acquire);
    assert!(base != 0, "HPET registers are not mounted");
    // SAFETY: `base` is the uncacheable mapping of the HPET block, `off` is a
    // naturally aligned register offset inside the 1 KiB block.
    unsafe { ((base + off) as *const u64).read_volatile() }
}

fn read32(off: usize) -> u32 {
    let base = BASE.load(Ordering::Acquire);
    assert!(base != 0, "HPET registers are not mounted");
    // SAFETY: see `read64`; `off` is 4-byte aligned.
    unsafe { ((base + off) as *const u32).read_volatile() }
}

fn write64(off: usize, value: u64) {
    let base = BASE.load(Ordering::Acquire);
    assert!(base != 0, "HPET registers are not mounted");
    // SAFETY: see `read64`.
    unsafe { ((base + off) as *mut u64).write_volatile(value) }
}

/// True once [`init`] succeeded.
pub fn is_available() -> bool {
    BASE.load(Ordering::Acquire) != 0
}

/// Device description, available after a successful [`init`].
pub fn info() -> Option<Info> {
    *INFO.lock()
}

fn counter_mask() -> u64 {
    if WIDE.load(Ordering::Relaxed) { u64::MAX } else { 0xFFFF_FFFF }
}

/// Current main counter value (ticks since [`init`] started it).
pub fn read_counter() -> u64 {
    if WIDE.load(Ordering::Relaxed) {
        read64(REG_COUNTER)
    } else {
        // A 32-bit counter must be read with a 32-bit access.
        read32(REG_COUNTER) as u64
    }
}

/// Ticks elapsed since `start` (a value from [`read_counter`]), correct across
/// one wrap of a 32-bit counter.
pub fn elapsed_ticks(start: u64) -> u64 {
    read_counter().wrapping_sub(start) & counter_mask()
}

pub fn ticks_to_ns(ticks: u64) -> u64 {
    (ticks as u128 * PERIOD_FS.load(Ordering::Relaxed) as u128 / FS_PER_NS) as u64
}

pub fn ns_to_ticks(ns: u64) -> u64 {
    let period = PERIOD_FS.load(Ordering::Relaxed).max(1) as u128;
    (ns as u128 * FS_PER_NS / period) as u64
}

/// Nanoseconds the counter has been running. Only meaningful across long
/// periods for a 64-bit counter; with a 32-bit one use [`elapsed_ticks`].
pub fn now_ns() -> u64 {
    ticks_to_ns(read_counter())
}

/// Busy-waits at least `ns` nanoseconds and returns the time that *actually*
/// elapsed according to the HPET (always >= `ns` unless the device stalls).
///
/// The loop samples the counter often enough to follow a 32-bit wrap, and
/// gives up after a generous TSC-based bound so a dead counter cannot hang
/// the kernel (the shortfall then shows in the returned value).
pub fn wait_ns(ns: u64) -> u64 {
    let target = ns_to_ticks(ns).max(1);
    let mask = counter_mask();
    let deadline = cpu::rdtsc().saturating_add(4_000_000_000).saturating_add(ns.saturating_mul(10));
    let mut last = read_counter();
    let mut acc = 0u64;
    while acc < target {
        let now = read_counter();
        acc += now.wrapping_sub(last) & mask;
        last = now;
        if cpu::rdtsc() >= deadline {
            break;
        }
        cpu::pause();
    }
    ticks_to_ns(acc)
}

/// Runs `start`, busy-waits until at least `ns` nanoseconds have passed on the
/// counter, runs `stop`, and returns the time from *just before `start`* to
/// *just after `stop`* per the HPET.
///
/// Use this to calibrate another clock: bracketing both of its readings with
/// HPET timestamps means a preemption (SMI, hypervisor) anywhere inside the
/// window lengthens both clocks' view of it, so the error cancels. Valid for
/// windows much shorter than a 32-bit counter wrap.
pub fn measure_ns(ns: u64, start: impl FnOnce(), stop: impl FnOnce()) -> u64 {
    let target = ns_to_ticks(ns).max(1);
    let deadline = cpu::rdtsc().saturating_add(4_000_000_000).saturating_add(ns.saturating_mul(10));
    let t0 = read_counter();
    start();
    while elapsed_ticks(t0) < target {
        if cpu::rdtsc() >= deadline {
            break;
        }
        cpu::pause();
    }
    stop();
    ticks_to_ns(elapsed_ticks(t0))
}

pub fn delay_us(us: u64) {
    wait_ns(us.saturating_mul(1_000));
}

pub fn delay_ms(ms: u64) {
    wait_ns(ms.saturating_mul(1_000_000));
}

// ----------------------------------------------------------------------
// Bring-up
// ----------------------------------------------------------------------

/// Discovers, mounts, resets and starts the HPET main counter.
///
/// Needs `acpi::init` to have succeeded, and must run on the boot CPU before
/// the other CPUs start (it may unmount a bootloader mapping).
pub fn init() -> Result<Info, HpetError> {
    // --- find ---------------------------------------------------------------
    let table = match acpi::table(b"HPET") {
        Ok(t) => t,
        Err(acpi::AcpiError::NotFound | acpi::AcpiError::NotInitialised) => {
            return Err(HpetError::NotPresent)
        }
        Err(e) => return Err(HpetError::Acpi(e)),
    };
    let space = table.u8_at(ACPI_ADDR_SPACE);
    if space != 0 {
        return Err(HpetError::UnsupportedAddressSpace(space));
    }
    let phys = table.u64_at(ACPI_ADDRESS);
    if phys == 0 {
        return Err(HpetError::BadAddress);
    }
    let block_id = table.u32_at(ACPI_BLOCK_ID);
    let number = table.u8_at(ACPI_NUMBER);
    let min_tick = table.u16_at(ACPI_MIN_TICK);
    debug!(
        "hpet: ACPI table ........ phys {:#x}, block id {:#010x}, number {}, min tick {}",
        phys, block_id, number, min_tick
    );

    // --- unmount any bootloader alias -----------------------------------------
    let alias = memory::phys_to_virt(phys & !(paging::PAGE_SIZE - 1)) as u64;
    let mut alias_removed = false;
    if let Some(m) = paging::translate(alias) {
        debug!("hpet: bootloader mapping . {:#x}: {}", alias, m);
        match paging::unmap_page(alias) {
            Ok(u) => {
                alias_removed = true;
                debug!(
                    "hpet: unmounted ......... HHDM alias removed ({} large page split(s))",
                    u.splits
                );
            }
            Err(e) => warn!("hpet: could not unmount HHDM alias: {:?}", e),
        }
    } else {
        debug!("hpet: bootloader mapping . none at {:#x}, nothing to unmount", alias);
    }

    // --- mount (uncacheable) ----------------------------------------------------
    let regs = paging::map_mmio(phys, REG_BLOCK_LEN).map_err(HpetError::Paging)?;
    BASE.store(regs as usize, Ordering::Release);
    if let Some(m) = paging::translate(regs as u64) {
        debug!("hpet: mounted ............ {:#x}: {}", regs as usize, m);
    }

    // --- read capabilities --------------------------------------------------------
    let caps = read64(REG_CAPS);
    let period_fs = caps >> 32;
    if period_fs == 0 || period_fs > MAX_PERIOD_FS {
        BASE.store(0, Ordering::Release);
        return Err(HpetError::BadPeriod(period_fs));
    }
    let wide = caps & CAP_COUNT_SIZE_64 != 0;
    let timers = ((caps >> 8) & 0x1F) as u32 + 1;
    PERIOD_FS.store(period_fs, Ordering::Release);
    WIDE.store(wide, Ordering::Release);

    // --- reset: stop, legacy routing off, comparators silent, counter = 0 ---------
    write64(REG_CONFIG, 0);
    for n in 0..timers as usize {
        let reg = REG_TIMER_BASE + n * TIMER_STRIDE;
        let cfg = read64(reg);
        write64(reg, cfg & !(TN_INT_ENABLE | TN_FSB_ENABLE));
    }
    write64(REG_INT_STATUS, u64::MAX); // clear any latched level-triggered status
    write64(REG_COUNTER, 0);

    // --- start and prove it counts ---------------------------------------------------
    write64(REG_CONFIG, CFG_ENABLE);
    let first = read_counter();
    let deadline = cpu::rdtsc().saturating_add(1_000_000_000);
    while read_counter() == first {
        if cpu::rdtsc() >= deadline {
            write64(REG_CONFIG, 0);
            BASE.store(0, Ordering::Release);
            return Err(HpetError::NotCounting);
        }
        cpu::pause();
    }
    let cfg = read64(REG_CONFIG);

    let info = Info {
        phys,
        virt: regs as usize,
        revision: caps as u8,
        vendor: (caps >> 16) as u16,
        timers,
        wide,
        legacy_capable: caps & CAP_LEGACY_ROUTE != 0,
        period_fs,
        number,
        min_tick,
        alias_removed,
    };
    *INFO.lock() = Some(info);

    debug!(
        "hpet: capabilities ...... rev {}, vendor {:#06x}, {} comparator(s), {}-bit counter, legacy routing {}",
        info.revision,
        info.vendor,
        info.timers,
        if wide { 64 } else { 32 },
        if info.legacy_capable { "capable (kept off)" } else { "not available" }
    );
    let hz = info.frequency_hz();
    debug!(
        "hpet: counter ........... period {} fs = {}.{:03} MHz",
        period_fs,
        hz / 1_000_000,
        (hz % 1_000_000) / 1_000
    );
    debug!(
        "hpet: started ........... CONFIG={:#x} (enabled, legacy routing off), comparator interrupts masked",
        cfg
    );
    Ok(info)
}

/// Checks that the main counter is monotonic over many reads and that it has
/// really advanced. Returns the number of ticks it advanced during the test.
pub fn self_test() -> Result<u64, &'static str> {
    if !is_available() {
        return Err("HPET not initialised");
    }
    let mask = counter_mask();
    let start = read_counter();
    let mut prev = start;
    for _ in 0..10_000 {
        let now = read_counter();
        // A step backwards shows up as an enormous wrapped difference.
        if now.wrapping_sub(prev) & mask > mask / 2 {
            return Err("main counter went backwards");
        }
        prev = now;
    }
    let advanced = prev.wrapping_sub(start) & mask;
    if advanced == 0 {
        return Err("main counter did not advance during 10000 reads");
    }
    Ok(advanced)
}
