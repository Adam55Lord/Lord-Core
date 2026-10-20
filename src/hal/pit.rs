// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Legacy 8254 Programmable Interval Timer, used only as a *reference clock*
//! (fallback when the machine has no HPET). It never raises interrupts here.
//!
//! Channel 2 is the one that can be gated from software without touching the
//! legacy IRQ0 channel: bit 0 of port `0x61` opens the gate, bit 5 reads the
//! channel's output. In mode 0 ("interrupt on terminal count") the output goes
//! high when the programmed count has run down, so polling it gives an exact
//! delay of `count / 1.193182 MHz`.
//!
//! Only **one CPU at a time** may use the PIT (the channel is shared state), so
//! Lord Core uses it on the boot CPU only.

use super::{cpu, port};

/// Input clock of the 8254: 1.193182 MHz (14.31818 MHz / 12).
pub const FREQUENCY_HZ: u64 = 1_193_182;
/// Longest single wait: the counter is 16 bits.
pub const MAX_COUNT: u32 = 0xFFFF;

const CHANNEL2_DATA: u16 = 0x42;
const COMMAND: u16 = 0x43;
const GATE_PORT: u16 = 0x61;
const GATE_ENABLE: u8 = 1 << 0;
const SPEAKER_ENABLE: u8 = 1 << 1;
const OUT2_STATUS: u8 = 1 << 5;
/// Select channel 2, access lobyte/hibyte, mode 0, binary.
const CMD_CH2_MODE0: u8 = 0b10_11_000_0;

/// Upper bound on a single wait, in TSC ticks, so a missing PIT cannot hang us.
const TIMEOUT_TSC: u64 = 2_000_000_000;

/// Converts nanoseconds to PIT ticks (rounded down).
pub fn ns_to_ticks(ns: u64) -> u64 {
    (ns as u128 * FREQUENCY_HZ as u128 / 1_000_000_000) as u64
}

/// Converts PIT ticks to nanoseconds (rounded down).
pub fn ticks_to_ns(ticks: u64) -> u64 {
    (ticks as u128 * 1_000_000_000 / FREQUENCY_HZ as u128) as u64
}

/// Busy-waits for `ticks` PIT ticks (1..=[`MAX_COUNT`]). Returns `false` if
/// the channel's output never went high (no PIT, or it is not counting).
pub fn wait_ticks(ticks: u32) -> bool {
    let ticks = ticks.clamp(1, MAX_COUNT);
    // SAFETY: standard PIT channel-2 one-shot sequence on the legacy ports.
    // The speaker bit is forced off so the countdown is silent.
    unsafe {
        let gate = port::inb(GATE_PORT);
        port::outb(GATE_PORT, gate & !(GATE_ENABLE | SPEAKER_ENABLE)); // gate closed
        port::outb(COMMAND, CMD_CH2_MODE0);
        port::outb(CHANNEL2_DATA, ticks as u8);
        port::outb(CHANNEL2_DATA, (ticks >> 8) as u8);
        port::outb(GATE_PORT, (gate & !SPEAKER_ENABLE) | GATE_ENABLE); // gate open: count
    }
    let deadline = cpu::rdtsc().saturating_add(TIMEOUT_TSC);
    // SAFETY: reading port 0x61 has no side effects.
    while unsafe { port::inb(GATE_PORT) } & OUT2_STATUS == 0 {
        if cpu::rdtsc() >= deadline {
            break;
        }
        cpu::pause();
    }
    // SAFETY: as above.
    let done = unsafe { port::inb(GATE_PORT) } & OUT2_STATUS != 0;
    // SAFETY: close the gate again, leaving the other bits as we found them.
    unsafe { port::outb(GATE_PORT, port::inb(GATE_PORT) & !(GATE_ENABLE | SPEAKER_ENABLE)) };
    done
}

/// Busy-waits about `ns` nanoseconds (any length; chunks of up to ~54 ms).
/// Returns the time that *actually* elapsed according to the PIT, or `None`
/// if the PIT did not respond.
pub fn wait_ns(ns: u64) -> Option<u64> {
    let mut remaining = ns_to_ticks(ns).max(1);
    let mut total = 0u64;
    while remaining > 0 {
        let chunk = remaining.min(MAX_COUNT as u64);
        if !wait_ticks(chunk as u32) {
            return None;
        }
        total += chunk;
        remaining -= chunk;
    }
    Some(ticks_to_ns(total))
}

/// Does the PIT answer? Runs a ~1 ms countdown.
pub fn probe() -> bool {
    wait_ticks(1_193)
}
