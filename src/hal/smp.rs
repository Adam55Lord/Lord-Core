// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Symmetric multiprocessing: starting the application processors (APs).
//!
//! The boot backend finds every logical CPU (each hardware thread counts as
//! one) and knows how to start it ([`SmpInfo::start_cpu`]): Lord Boot leaves the
//! APs idle in firmware, so the backend wakes them with INIT-SIPI; a bootloader
//! that parks its APs can implement the same hook differently. Either way the CPU
//! then runs [`ap_entry`] on a usable stack with interrupts disabled and a
//! temporary GDT. Each AP must set up its *own* GDT/TSS, load the shared IDT
//! and its per-CPU data before doing anything else.
//!
//! Logical CPU numbering: index 0 is always the bootstrap processor (BSP);
//! the APs get 1, 2, ... in the order the bootloader lists them.

use core::sync::atomic::{AtomicUsize, Ordering};

use super::{apic, apic_timer, cpu, gdt, idt, paging, percpu, MAX_CPUS};
use crate::boot::SmpInfo;

/// CPUs that finished their own setup (the BSP counts as one).
static ONLINE: AtomicUsize = AtomicUsize::new(1);
/// CPUs the bootloader reported (including the BSP).
static FOUND: AtomicUsize = AtomicUsize::new(1);

/// Rough upper bound on waiting for APs: this many TSC ticks (~1-4 s).
const START_TIMEOUT_TSC: u64 = 4_000_000_000;

pub struct Report {
    /// Logical CPUs found by the bootloader, BSP included.
    pub found: usize,
    /// APs we asked to start.
    pub requested: usize,
    /// CPUs online now, BSP included.
    pub online: usize,
    /// CPUs left parked because we only support `MAX_CPUS`.
    pub skipped: usize,
}

pub fn found_cpus() -> usize {
    FOUND.load(Ordering::Relaxed)
}

pub fn online_cpus() -> usize {
    ONLINE.load(Ordering::Acquire)
}

/// Starts every application processor and waits (bounded) until they report
/// in. The BSP must already have run `percpu::init(0, ..)`, `gdt::init_cpu(0)`
/// and `idt::init()`.
pub fn start(smp: &SmpInfo) -> Report {
    FOUND.store(smp.cpus.len(), Ordering::Relaxed);

    let mut next_index = 1usize;
    let mut requested = 0usize;
    let mut skipped = 0usize;

    for cpu in smp.cpus {
        if cpu.lapic_id == smp.bsp_lapic_id {
            continue; // that's us
        }
        if next_index >= MAX_CPUS {
            skipped += 1;
            continue;
        }
        (smp.start_cpu)(cpu, next_index, ap_entry);
        next_index += 1;
        requested += 1;
    }

    let deadline = cpu::rdtsc().saturating_add(START_TIMEOUT_TSC);
    while ONLINE.load(Ordering::Acquire) < 1 + requested && cpu::rdtsc() < deadline {
        cpu::pause();
    }

    Report { found: smp.cpus.len(), requested, online: online_cpus(), skipped }
}

/// First kernel code an application processor runs (see `boot::ApEntry`).
extern "C" fn ap_entry(index: usize, lapic_id: u32) -> ! {
    gdt::init_cpu(index); // own GDT + TSS (+ IST stacks), reload segments
    idt::load(); // shared IDT
    percpu::init(index, lapic_id); // GS base -> this CPU's data
    super::syscall::init_cpu(); // `syscall`/`sysret` on this CPU too
    paging::init_pat(); // same PAT layout as the BSP (it is per CPU)

    let topology = cpu::Topology::detect();
    match topology.locate(lapic_id) {
        Some(p) => debug!(
            "smp:  cpu {} online ........ lapic id {} (package {} core {} thread {})",
            index, lapic_id, p.package, p.core, p.thread
        ),
        None => debug!("smp:  cpu {} online ........ lapic id {}", index, lapic_id),
    }

    // Each CPU has its own local APIC: enable ours and prove it works.
    match apic::init_ap(lapic_id) {
        Ok(()) => {
            if apic::self_ipi_test() {
                debug!("apic: cpu {} LAPIC on ..... id {} ok, self-IPI delivered", index, apic::id());
            } else {
                warn!("apic: cpu {} LAPIC self-IPI NOT delivered", index);
            }
        }
        Err(e) => warn!("apic: cpu {} init failed: {:?}", index, e),
    }

    // Its own LAPIC timer too (calibrated once by the BSP, shared).
    if apic_timer::is_calibrated() {
        match apic_timer::init_ap() {
            Ok(()) if apic_timer::oneshot_self_test() => {
                debug!("apic: cpu {} timer ok ..... one-shot interrupt delivered", index)
            }
            Ok(()) => warn!("apic: cpu {} timer one-shot NOT delivered", index),
            Err(e) => warn!("apic: cpu {} timer init failed: {:?}", index, e),
        }
    }

    ONLINE.fetch_add(1, Ordering::AcqRel);
    cpu::halt_forever()
}
