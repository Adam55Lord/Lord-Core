// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Per-CPU data. Each logical CPU gets a [`PerCpu`] block and points its
//! `IA32_GS_BASE` MSR at it, so `gs:[offset]` always reaches *this* CPU's data
//! without knowing which CPU that is.
//!
//! Note: there is no `swapgs` on interrupt or `syscall` entry. That is fine
//! because ring 3 cannot change the GS base (no `wrgsbase`: `CR4.FSGSBASE` is
//! off, and `wrmsr` is privileged), so `IA32_GS_BASE` is *always* this CPU's
//! block, in both rings. Add `swapgs` together with user-mode `FS`/`GS` (TLS).

use core::arch::asm;
use core::mem::size_of;

use super::msr::{self, IA32_GS_BASE};
use super::sync::StaticCell;
use super::MAX_CPUS;

#[repr(C)]
pub struct PerCpu {
    /// Offset 0: pointer to this very struct (the usual `gs:[0]` trick).
    self_ptr: u64,
    /// Offset 8: logical CPU index (0 = bootstrap processor).
    index: u32,
    /// Offset 12: local APIC ID reported by the bootloader.
    lapic_id: u32,
    /// Offset 16: top of the kernel stack of the thread running here (a copy of
    /// TSS `rsp0`): where the `syscall` entry stub switches to.
    kernel_rsp: u64,
    /// Offset 24: scratch for the user `rsp` between `syscall` and the first push.
    user_rsp: u64,
}

/// `gs:[..]` offsets used by the `syscall` entry stub (`hal/syscall.rs`).
pub const OFF_KERNEL_RSP: usize = 16;
pub const OFF_USER_RSP: usize = 24;
const _: () = assert!(core::mem::offset_of!(PerCpu, kernel_rsp) == OFF_KERNEL_RSP);
const _: () = assert!(core::mem::offset_of!(PerCpu, user_rsp) == OFF_USER_RSP);

const PERCPU_INIT: StaticCell<PerCpu> = StaticCell::new(PerCpu { self_ptr: 0, index: 0, lapic_id: 0, kernel_rsp: 0, user_rsp: 0 });
static PERCPU: [StaticCell<PerCpu>; MAX_CPUS] = [PERCPU_INIT; MAX_CPUS];

/// Creates this CPU's block and points GS at it. Call after the GDT is
/// loaded (reloading the GS selector may reset the base on some CPUs).
pub fn init(index: usize, lapic_id: u32) {
    assert!(index < MAX_CPUS, "cpu index out of range");
    let ptr = PERCPU[index].get();
    // SAFETY: each CPU initialises only its own slot, once, before using GS.
    unsafe {
        // Keep a kernel stack that `tss::init` already published for this CPU.
        let kernel_rsp = (*ptr).kernel_rsp;
        ptr.write(PerCpu { self_ptr: ptr as usize as u64, index: index as u32, lapic_id, kernel_rsp, user_rsp: 0 });
        msr::wrmsr(IA32_GS_BASE, ptr as usize as u64);
    }
}

/// Logical index of the CPU running this code. Only valid after [`init`].
pub fn current_index() -> usize {
    let v: u32;
    // SAFETY: GS base points at this CPU's PerCpu once `init` has run.
    unsafe { asm!("mov {0:e}, gs:[8]", out(reg) v, options(nostack, preserves_flags, readonly)) };
    v as usize
}

pub fn current_lapic_id() -> u32 {
    let v: u32;
    // SAFETY: see `current_index`.
    unsafe { asm!("mov {0:e}, gs:[12]", out(reg) v, options(nostack, preserves_flags, readonly)) };
    v
}

/// Like [`current_index`] but safe to call from fault handlers at any time:
/// returns `None` unless GS really points into our per-CPU array.
pub fn try_current_index() -> Option<usize> {
    // SAFETY: IA32_GS_BASE exists on every x86_64 CPU.
    let base = unsafe { msr::rdmsr(IA32_GS_BASE) } as usize;
    let start = PERCPU.as_ptr() as usize;
    let end = start + MAX_CPUS * size_of::<StaticCell<PerCpu>>();
    if base >= start && base < end {
        Some((base - start) / size_of::<StaticCell<PerCpu>>())
    } else {
        None
    }
}

/// Publishes `top` as the kernel stack the `syscall` stub switches to on CPU
/// `cpu` (called by `tss::set_kernel_stack`, so it always equals TSS `rsp0`).
pub fn set_kernel_rsp(cpu: usize, top: u64) {
    if cpu < MAX_CPUS {
        // SAFETY: a plain store to a field of a static; the owning CPU only reads it
        // with interrupts masked, and the scheduler writes it on that same CPU.
        unsafe { (*PERCPU[cpu].get()).kernel_rsp = top };
    }
}
