// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! The kernel's **reference clock**: one trustworthy time base that other
//! clocks (the LAPIC timer today, the TSC later) are calibrated against.
//!
//! Preference order:
//!
//! 1. **HPET** (`hal::hpet`): memory-mapped, >= 10 MHz, shared by all CPUs,
//!    safe to read from any CPU at any time.
//! 2. **PIT** channel 2 (`hal::pit`): present on every PC-compatible machine
//!    and in every VM, but a single shared device, so only the boot CPU may
//!    use it, and only before the other CPUs start.
//!
//! [`init`] brings up ACPI, the HPET and probes the PIT, and when both exist it
//! cross-checks them against each other (two independent oscillators agreeing
//! within a few percent is strong evidence that neither period is mis-reported).

use core::sync::atomic::{AtomicU8, Ordering};

use super::{acpi, hpet, pit};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Hpet,
    Pit,
}

impl Source {
    pub fn name(self) -> &'static str {
        match self {
            Source::Hpet => "HPET",
            Source::Pit => "PIT",
        }
    }
}

const SRC_NONE: u8 = 0;
const SRC_HPET: u8 = 1;
const SRC_PIT: u8 = 2;
static SOURCE: AtomicU8 = AtomicU8::new(SRC_NONE);

/// Length of one HPET-vs-PIT cross-check window.
const CROSS_CHECK_NS: u64 = 10_000_000;
/// Windows tried; the least-disturbed one counts (interference only ever
/// *adds* time to the HPET side, so the smallest HPET reading is the best).
const CROSS_CHECK_ROUNDS: usize = 5;
/// Largest disagreement (parts per million) before we warn: 10 %. Port I/O
/// to program the PIT costs real time (a lot in an emulator), so this is
/// deliberately loose: it catches a mis-reported period (2x, 1000x), not drift.
const CROSS_CHECK_TOLERANCE_PPM: u64 = 100_000;

/// The active reference clock, if any.
pub fn source() -> Option<Source> {
    match SOURCE.load(Ordering::Acquire) {
        SRC_HPET => Some(Source::Hpet),
        SRC_PIT => Some(Source::Pit),
        _ => None,
    }
}

/// Busy-waits about `ns` nanoseconds on the reference clock and returns how
/// long it *actually* took per that clock, or `None` if there is no usable
/// reference. With the PIT as reference this must only run on the boot CPU.
pub fn wait_ns(ns: u64) -> Option<u64> {
    match source()? {
        Source::Hpet => Some(hpet::wait_ns(ns)),
        Source::Pit => pit::wait_ns(ns),
    }
}

/// Runs `start`, waits about `ns` on the reference clock, runs `stop`, and
/// returns the reference clock's elapsed time covering both calls.
///
/// With the HPET the span is bracketed by timestamps taken just before `start`
/// and just after `stop` (see [`hpet::measure_ns`]), so a preemption inside the
/// window cannot skew a calibration. The PIT can only report its nominal
/// countdown length. Boot CPU only when the reference is the PIT.
pub fn measure(ns: u64, start: impl FnOnce(), stop: impl FnOnce()) -> Option<u64> {
    match source()? {
        Source::Hpet => Some(hpet::measure_ns(ns, start, stop)),
        Source::Pit => {
            start();
            let elapsed = pit::wait_ns(ns);
            stop();
            elapsed
        }
    }
}

/// Busy-waits at least `us` microseconds. Returns `false` without a reference clock.
pub fn delay_us(us: u64) -> bool {
    wait_ns(us.saturating_mul(1_000)).is_some()
}

/// Busy-waits at least `ms` milliseconds. Returns `false` without a reference clock.
pub fn delay_ms(ms: u64) -> bool {
    wait_ns(ms.saturating_mul(1_000_000)).is_some()
}

/// Brings up ACPI table discovery, the HPET and the PIT probe, and picks the
/// reference clock. `rsdp` is the bootloader's RSDP address (if it gave one).
/// Boot CPU only, before the other CPUs start.
pub fn init(rsdp: Option<u64>) -> Option<Source> {
    // --- ACPI -------------------------------------------------------------------
    match rsdp {
        None => debug!("acpi: RSDP ............. none from the bootloader (no ACPI)"),
        Some(addr) => match acpi::init(addr) {
            Ok(()) => {
                let s = acpi::summary();
                debug!(
                    "acpi: RSDP ............. phys {:#x}, revision {}, OEM \"{}\", {} at {:#x}",
                    s.rsdp_phys,
                    s.revision,
                    s.oem(),
                    if s.root_is_xsdt { "XSDT" } else { "RSDT" },
                    s.root_phys
                );
                debug!("acpi: tables ............ {}: {}", s.tables, acpi::TableList);
                debug!(
                    "acpi: table access ...... {}",
                    if acpi::window_maps() == 0 {
                        "through the HHDM (no extra mappings)"
                    } else {
                        "HHDM does not cover them; mounted via the MMIO window"
                    }
                );
                if s.dropped > 0 {
                    warn!("acpi: {} table(s) did not fit the directory ({})", s.dropped, acpi::MAX_TABLES);
                }
            }
            Err(e) => warn!("acpi: initialisation FAILED: {:?}", e),
        },
    }

    // --- HPET ---------------------------------------------------------------------
    let mut hpet_ok = false;
    if acpi::is_ready() {
        match hpet::init() {
            Ok(_) => {
                hpet_ok = true;
                match hpet::self_test() {
                    Ok(ticks) => debug!(
                        "hpet: self-test ......... counter monotonic over 10000 reads (+{} ticks)",
                        ticks
                    ),
                    Err(why) => {
                        error!("hpet: self-test FAILED: {}", why);
                        hpet_ok = false;
                    }
                }
            }
            Err(hpet::HpetError::NotPresent) => debug!("hpet: not present ........ firmware has no HPET table"),
            Err(e) => warn!("hpet: initialisation FAILED: {:?}", e),
        }
    }

    // --- PIT ----------------------------------------------------------------------
    let pit_ok = pit::probe();
    debug!(
        "pit:  8254 channel 2 ... {}",
        if pit_ok { "responds (1.193182 MHz)" } else { "no response" }
    );

    // --- cross-check: two independent oscillators should agree --------------------
    if hpet_ok && pit_ok {
        let mut best: Option<(u64, u64)> = None; // (pit_ns, hpet_ns) of the cleanest round
        for _ in 0..CROSS_CHECK_ROUNDS {
            let start = hpet::read_counter();
            let Some(pit_ns) = pit::wait_ns(CROSS_CHECK_NS) else { break };
            let hpet_ns = hpet::ticks_to_ns(hpet::elapsed_ticks(start));
            if best.is_none_or(|(_, h)| hpet_ns < h) {
                best = Some((pit_ns, hpet_ns));
            }
        }
        if let Some((pit_ns, hpet_ns)) = best {
            let diff = hpet_ns.abs_diff(pit_ns);
            let ppm = (diff as u128 * 1_000_000 / pit_ns.max(1) as u128) as u64;
            let (whole, frac) = (ppm / 10_000, (ppm % 10_000) / 100);
            if ppm <= CROSS_CHECK_TOLERANCE_PPM {
                debug!(
                    "hpet: cross-check ....... PIT {} us vs HPET {} us, differ by {}.{:02}% (best of {})",
                    pit_ns / 1_000,
                    hpet_ns / 1_000,
                    whole,
                    frac,
                    CROSS_CHECK_ROUNDS
                );
            } else {
                warn!(
                    "hpet: cross-check ....... PIT {} us vs HPET {} us differ by {}.{:02}% (> 10%): a period is mis-reported, or the CPU was descheduled in every round",
                    pit_ns / 1_000,
                    hpet_ns / 1_000,
                    whole,
                    frac
                );
            }
        }
    }

    let chosen = if hpet_ok {
        SRC_HPET
    } else if pit_ok {
        SRC_PIT
    } else {
        SRC_NONE
    };
    SOURCE.store(chosen, Ordering::Release);
    let s = source();
    match s {
        Some(src) => debug!("timer: reference clock . {}", src.name()),
        None => warn!("timer: reference clock . none (HPET and PIT unavailable)"),
    }
    s
}
