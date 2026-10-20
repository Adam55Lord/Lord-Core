// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Local APIC timer: a per-CPU down-counter that raises [`TIMER_VECTOR`].
//!
//! Every logical CPU has its own timer, programmed through its own LAPIC
//! registers (`hal::apic::read/write` act on the *calling* CPU). The counter
//! decrements at `bus/core-crystal clock / divider`; that clock has no fixed
//! value, so it must be **calibrated** against the reference clock
//! (`hal::timer`: HPET, or PIT as fallback).
//!
//! Modes (LVT timer register bits 18:17): one-shot (`00`), periodic (`01`) and
//! TSC-deadline (`10`). Lord Core implements one-shot and periodic. TSC-deadline is
//! detected and reported but not used yet.
//!
//! Calibration policy: done **once, on the boot CPU**, and the result is
//! shared. All CPUs of a machine derive their timer from the same clock, and
//! the PIT (the fallback reference) must not be used by several CPUs at once.
//!
//! Bring-up: [`init_bsp`] on the boot CPU (needs the LAPIC enabled and
//! `timer::init` done), [`init_ap`] on every application processor.

use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use super::idt::{self, InterruptFrame};
use super::{apic, cpu, percpu, timer, MAX_CPUS};

/// Interrupt vector of the LAPIC timer (priority class 3, above the remapped PIC range).
pub const TIMER_VECTOR: u8 = 0x30;
/// Default periodic tick rate.
pub const DEFAULT_TICK_HZ: u32 = 100;

const LVT_MASKED: u32 = 1 << 16;
const LVT_MODE_PERIODIC: u32 = 0b01 << 17;
/// Divide Configuration Register value for "divide by 16".
const DIVIDE_BY_16: u32 = 0b0011;
const DIVIDER: u64 = 16;

/// Length of one calibration measurement.
const CAL_WINDOW_NS: u64 = 10_000_000;
const CAL_ROUNDS: usize = 5;
/// Sanity range for the counter rate (after the divider): 100 kHz .. 4 GHz.
const PLAUSIBLE_HZ: core::ops::RangeInclusive<u64> = 100_000..=4_000_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimerError {
    /// The LAPIC registers are not mounted yet.
    ApicNotMounted,
    /// There is no HPET or PIT to calibrate against.
    NoReferenceClock,
    /// The reference clock stopped answering during calibration.
    ReferenceFailed,
    /// The measured rate is outside [`PLAUSIBLE_HZ`] (0 = the timer never counted).
    Implausible(u64),
    /// [`init_bsp`] has not calibrated the timer yet.
    NotCalibrated,
    /// A zero frequency or duration was requested.
    InvalidArgument,
}

/// Result of a calibration.
#[derive(Clone, Copy)]
pub struct Calibration {
    /// Counter decrement rate with the divide-by-16 setting, in Hz.
    pub counter_hz: u64,
    /// Rate of the undivided timer clock (`counter_hz * 16`), in Hz.
    pub base_hz: u64,
    pub source: timer::Source,
    pub rounds: usize,
    /// (max - min) / chosen value over the rounds, in parts per million.
    pub spread_ppm: u64,
}

/// Counter rate with the divide-by-16 setting (0 = not calibrated).
static COUNTER_HZ: AtomicU64 = AtomicU64::new(0);

const NO_TICKS: AtomicU64 = AtomicU64::new(0);
/// Timer interrupts handled, per logical CPU.
static TICKS: [AtomicU64; MAX_CPUS] = [NO_TICKS; MAX_CPUS];

/// Function called after every timer interrupt (the scheduler's clock).
static TICK_HOOK: AtomicUsize = AtomicUsize::new(0);

/// Installs `hook`, called from the timer interrupt with the interrupted
/// context's frame **after** the end-of-interrupt, so the hook may switch to
/// another thread. The HAL does not know what the hook does.
pub fn set_tick_hook(hook: fn(&mut InterruptFrame)) {
    TICK_HOOK.store(hook as usize, Ordering::Release);
}

fn timer_handler(frame: &mut InterruptFrame) {
    TICKS[percpu::current_index()].fetch_add(1, Ordering::Relaxed);
    apic::eoi();
    let raw = TICK_HOOK.load(Ordering::Acquire);
    if raw != 0 {
        // SAFETY: only `set_tick_hook` stores here, and it stores real fn pointers.
        let hook: fn(&mut InterruptFrame) = unsafe { core::mem::transmute::<usize, fn(&mut InterruptFrame)>(raw) };
        hook(frame);
    }
}

// ----------------------------------------------------------------------
// Queries
// ----------------------------------------------------------------------

pub fn is_calibrated() -> bool {
    COUNTER_HZ.load(Ordering::Acquire) != 0
}

/// Counter rate in Hz with the divide-by-16 setting (0 before calibration).
pub fn counter_hz() -> u64 {
    COUNTER_HZ.load(Ordering::Acquire)
}

/// Timer interrupts handled so far by the calling CPU.
pub fn ticks() -> u64 {
    TICKS[percpu::current_index()].load(Ordering::Relaxed)
}

/// Timer interrupts handled so far by logical CPU `cpu_index`.
pub fn ticks_of(cpu_index: usize) -> u64 {
    TICKS.get(cpu_index).map_or(0, |t| t.load(Ordering::Relaxed))
}

/// Current value of this CPU's down-counter.
pub fn current_count() -> u32 {
    apic::read(apic::REG_TIMER_CURRENT)
}

/// Longest one-shot delay the 32-bit counter can express, in nanoseconds.
pub fn max_oneshot_ns() -> u64 {
    let hz = counter_hz();
    if hz == 0 { 0 } else { (u32::MAX as u128 * 1_000_000_000 / hz as u128) as u64 }
}

/// Converts a duration to counter units (clamped to `1..=u32::MAX`).
pub fn ns_to_counts(ns: u64) -> Result<u32, TimerError> {
    let hz = counter_hz();
    if hz == 0 {
        return Err(TimerError::NotCalibrated);
    }
    let counts = (ns as u128 * hz as u128 / 1_000_000_000).clamp(1, u32::MAX as u128);
    Ok(counts as u32)
}

// ----------------------------------------------------------------------
// Control (acts on the calling CPU)
// ----------------------------------------------------------------------

/// Masks and stops this CPU's timer.
pub fn stop() {
    apic::write(apic::REG_LVT_TIMER, LVT_MASKED | TIMER_VECTOR as u32);
    apic::write(apic::REG_TIMER_INITIAL, 0);
}

/// Arms a one-shot: one interrupt after about `ns` nanoseconds.
/// Interrupts must be enabled (`sti`) on this CPU for the handler to run.
pub fn start_oneshot_ns(ns: u64) -> Result<(), TimerError> {
    if ns == 0 {
        return Err(TimerError::InvalidArgument);
    }
    let counts = ns_to_counts(ns)?;
    apic::write(apic::REG_TIMER_DIVIDE, DIVIDE_BY_16);
    apic::write(apic::REG_LVT_TIMER, TIMER_VECTOR as u32); // one-shot, unmasked
    apic::write(apic::REG_TIMER_INITIAL, counts); // writing the count starts it
    Ok(())
}

/// Starts a periodic interrupt at `hz` per second (clamped to what the
/// counter can express). Interrupts must be enabled on this CPU.
pub fn start_periodic_hz(hz: u32) -> Result<(), TimerError> {
    if hz == 0 {
        return Err(TimerError::InvalidArgument);
    }
    let counts = ns_to_counts(1_000_000_000 / hz as u64)?;
    apic::write(apic::REG_TIMER_DIVIDE, DIVIDE_BY_16);
    apic::write(apic::REG_LVT_TIMER, LVT_MODE_PERIODIC | TIMER_VECTOR as u32);
    apic::write(apic::REG_TIMER_INITIAL, counts);
    Ok(())
}

// ----------------------------------------------------------------------
// Calibration
// ----------------------------------------------------------------------

/// Measures the counter rate against the reference clock. Boot CPU only.
fn calibrate() -> Result<Calibration, TimerError> {
    if !apic::is_mounted() {
        return Err(TimerError::ApicNotMounted);
    }
    let source = timer::source().ok_or(TimerError::NoReferenceClock)?;

    apic::write(apic::REG_TIMER_DIVIDE, DIVIDE_BY_16);
    apic::write(apic::REG_LVT_TIMER, LVT_MASKED | TIMER_VECTOR as u32); // one-shot, masked

    let mut rates = [0u64; CAL_ROUNDS];
    for rate in rates.iter_mut() {
        let mut remaining = u32::MAX;
        // The reference brackets start-of-count and the read-back, so a
        // preemption inside the window cancels out of the ratio.
        let elapsed_ns = timer::measure(
            CAL_WINDOW_NS,
            || apic::write(apic::REG_TIMER_INITIAL, u32::MAX),
            || remaining = apic::read(apic::REG_TIMER_CURRENT),
        )
        .ok_or(TimerError::ReferenceFailed)?;
        apic::write(apic::REG_TIMER_INITIAL, 0);
        if elapsed_ns == 0 {
            return Err(TimerError::ReferenceFailed);
        }
        let counted = (u32::MAX - remaining) as u128;
        *rate = (counted * 1_000_000_000 / elapsed_ns as u128) as u64;
    }
    rates.sort_unstable();
    // HPET rounds are bracketed, so noise is symmetric: take the median. A PIT
    // round can only report its nominal length, and any delay after the
    // countdown ends *adds* LAPIC counts, so errors are one-sided (too high):
    // the smallest sample is the least disturbed.
    let median = if source == timer::Source::Pit { rates[0] } else { rates[CAL_ROUNDS / 2] };
    if !PLAUSIBLE_HZ.contains(&median) {
        return Err(TimerError::Implausible(median));
    }
    let spread_ppm = ((rates[CAL_ROUNDS - 1] - rates[0]) as u128 * 1_000_000 / median as u128) as u64;

    COUNTER_HZ.store(median, Ordering::Release);
    Ok(Calibration {
        counter_hz: median,
        base_hz: median * DIVIDER,
        source,
        rounds: CAL_ROUNDS,
        spread_ppm,
    })
}

// ----------------------------------------------------------------------
// Self-tests
// ----------------------------------------------------------------------

/// Arms a 1 ms one-shot on the calling CPU, waits (bounded by the TSC, so it
/// works without a reference clock) for the interrupt, and stops the timer.
/// Enables interrupts briefly; they are masked again on return.
pub fn oneshot_self_test() -> bool {
    let before = ticks();
    if start_oneshot_ns(1_000_000).is_err() {
        return false;
    }
    cpu::enable_interrupts();
    let deadline = cpu::rdtsc().saturating_add(2_000_000_000);
    while ticks() == before && cpu::rdtsc() < deadline {
        cpu::pause();
    }
    cpu::disable_interrupts();
    stop();
    ticks() > before
}

/// Outcome of the boot CPU's full self-test.
#[derive(Clone, Copy)]
pub struct SelfTest {
    /// Periodic rate used for the measurement.
    pub hz: u32,
    /// Window length of the reported attempt, per the reference clock.
    pub window_ns: u64,
    /// Interrupts received in that window.
    pub ticks: u64,
    /// Interrupts a perfect timer would have delivered.
    pub expected: u64,
    /// Attempts made (1..=[`SELF_TEST_ATTEMPTS`]); it stops at the first good one.
    pub attempts: u32,
    /// The timer works but interrupts were lost: the CPU did not service a
    /// 1 ms tick in time (descheduled vCPU, SMI, slow emulation). Not a driver
    /// fault, but worth a warning.
    pub degraded: bool,
}

const SELF_TEST_ATTEMPTS: u32 = 3;

/// One 50 ms periodic window. Returns (interrupts received, window length).
fn periodic_window(hz: u32) -> Result<(u64, u64), &'static str> {
    let before = ticks();
    let window_ns = timer::measure(
        50_000_000,
        || {
            let _ = start_periodic_hz(hz);
            cpu::enable_interrupts();
        },
        || {
            cpu::disable_interrupts();
            stop();
        },
    )
    .ok_or("reference clock failed")?;
    Ok((ticks() - before, window_ns))
}

/// Boot CPU only (uses the reference clock). Checks that
///
/// 1. a one-shot fires, and fires exactly once;
/// 2. a 1 kHz periodic timer keeps re-arming itself and delivers roughly one
///    interrupt per millisecond over a 50 ms window measured on the reference.
///
/// Periodic mode is tried up to [`SELF_TEST_ATTEMPTS`] times and the best
/// attempt counts: an overloaded host can only *lose* ticks (the LAPIC merges
/// a pending vector with one that is already waiting), never invent them.
/// Fewer than two ticks, or clearly too many, is an error; a merely low count
/// is reported as `degraded`.
pub fn self_test() -> Result<SelfTest, &'static str> {
    if !oneshot_self_test() {
        return Err("one-shot timer interrupt was not delivered");
    }
    // A one-shot fires once and only once. Delivery may be *late* (a descheduled
    // vCPU, an SMI), so wait up to 100 ms of reference time for the first
    // interrupt, then make sure no second one follows.
    let before = ticks();
    start_oneshot_ns(1_000_000).map_err(|_| "cannot arm one-shot")?;
    cpu::enable_interrupts();
    let mut waited = 0;
    while ticks() == before && waited < 100 {
        timer::wait_ns(1_000_000).ok_or("reference clock failed")?;
        waited += 1;
    }
    if ticks() == before {
        cpu::disable_interrupts();
        stop();
        return Err("one-shot interrupt was not delivered within 100 ms");
    }
    timer::wait_ns(5_000_000).ok_or("reference clock failed")?;
    cpu::disable_interrupts();
    stop();
    if ticks() != before + 1 {
        return Err("one-shot fired more than once");
    }

    const HZ: u32 = 1_000;
    let mut best: Option<SelfTest> = None;
    let mut made = 0;
    for attempt in 1..=SELF_TEST_ATTEMPTS {
        made = attempt;
        let (got, window_ns) = periodic_window(HZ)?;
        let expected = window_ns * HZ as u64 / 1_000_000_000;
        let ok = got >= expected / 2 && got <= expected + expected / 2;
        let candidate = SelfTest { hz: HZ, window_ns, ticks: got, expected, attempts: 0, degraded: !ok };
        if best.is_none_or(|b| candidate.ticks > b.ticks) {
            best = Some(candidate);
        }
        if ok {
            break;
        }
    }
    let mut result = best.ok_or("no periodic attempt ran")?;
    result.attempts = made;
    if result.ticks < 2 {
        return Err("periodic timer did not re-arm itself");
    }
    if result.ticks > result.expected + result.expected / 2 {
        return Err("periodic timer runs faster than calibrated");
    }
    Ok(result)
}

// ----------------------------------------------------------------------
// Bring-up
// ----------------------------------------------------------------------

fn mhz(hz: u64) -> (u64, u64) {
    (hz / 1_000_000, (hz % 1_000_000) / 1_000)
}

/// Registers the handler, calibrates the timer against the reference clock and
/// leaves it stopped. Boot CPU only; the LAPIC must be enabled and
/// `timer::init` must have run.
pub fn init_bsp() -> Result<Calibration, TimerError> {
    debug!(
        "apic: timer features .... ARAT {}, TSC-deadline {} (not used), invariant TSC {}",
        if cpu::has_arat() { "yes" } else { "no" },
        if cpu::has_tsc_deadline() { "yes" } else { "no" },
        if cpu::has_invariant_tsc() { "yes" } else { "no" }
    );
    idt::register(TIMER_VECTOR, timer_handler);
    let c = calibrate()?;
    stop();
    let base = mhz(c.base_hz);
    debug!(
        "apic: timer calibrated .. counter {} Hz (timer clock {}.{:03} MHz / {}), reference {}, {} rounds, spread {}.{:03}%",
        c.counter_hz,
        base.0,
        base.1,
        DIVIDER,
        c.source.name(),
        c.rounds,
        c.spread_ppm / 10_000,
        (c.spread_ppm % 10_000) / 10
    );
    debug!(
        "apic: timer range ....... one-shot up to {} ms; periodic default {} Hz = {} counts",
        max_oneshot_ns() / 1_000_000,
        DEFAULT_TICK_HZ,
        ns_to_counts(1_000_000_000 / DEFAULT_TICK_HZ as u64).unwrap_or(0)
    );
    Ok(c)
}

/// Prepares the calling application processor's timer (divider, masked LVT).
/// The BSP must have calibrated first; the result is shared.
pub fn init_ap() -> Result<(), TimerError> {
    if !is_calibrated() {
        return Err(TimerError::NotCalibrated);
    }
    stop();
    apic::write(apic::REG_TIMER_DIVIDE, DIVIDE_BY_16);
    Ok(())
}
