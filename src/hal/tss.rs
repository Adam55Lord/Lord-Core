// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Task State Segment (TSS) and the Interrupt Stack Table (IST).
//!
//! In 64-bit mode the TSS no longer holds task state. It provides:
//!
//! * `rsp0..rsp2`: the stack the CPU switches to when an interrupt or
//!   exception moves execution from a *less* privileged ring into ring
//!   0/1/2. Ring 3 -> ring 0 transitions use `rsp0`.
//! * `ist1..ist7`: the Interrupt Stack Table. An IDT gate can name one of
//!   these (1..=7); the CPU then *always* switches to that fresh stack for
//!   that vector, even when already in ring 0. This is what makes double
//!   fault / NMI / machine-check handlers survive a corrupted or overflowed
//!   kernel stack.
//! * `iomap_base`: offset of the I/O permission bitmap. We set it to the TSS
//!   size, meaning "no bitmap": ring 3 gets no port access.
//!
//! Layout (Intel SDM Vol. 3, "64-bit TSS"): 104 bytes, 4-byte aligned fields.

use core::mem::size_of;
use core::ptr::addr_of_mut;

use super::sync::StaticCell;
use super::MAX_CPUS;

/// IST slots with a kernel-assigned role. Put these numbers in the IDT gate's
/// `ist` field when you add an IDT.
pub const IST_DOUBLE_FAULT: u8 = 1;
pub const IST_NMI: u8 = 2;
pub const IST_MACHINE_CHECK: u8 = 3;
pub const IST_DEBUG: u8 = 4;
/// Slots 5..=7 are free: assign your own stacks with [`set_ist_stack`].

/// Number of IST entries the hardware provides.
pub const IST_COUNT: usize = 7;

/// Size of each statically allocated IST stack.
pub const IST_STACK_SIZE: usize = 16 * 1024;
/// Size of the ring 3 -> ring 0 entry stack (`rsp0`).
pub const KERNEL_STACK_SIZE: usize = 32 * 1024;

#[repr(C, packed(4))]
pub struct Tss {
    reserved0: u32,
    /// Privilege-level stacks: `rsp[0]` is used when entering ring 0.
    rsp: [u64; 3],
    reserved1: u64,
    /// Interrupt Stack Table: `ist[0]` is IST1, `ist[6]` is IST7.
    ist: [u64; IST_COUNT],
    reserved2: u64,
    reserved3: u16,
    iomap_base: u16,
}

/// Size of the TSS in bytes (104).
pub const TSS_SIZE: usize = size_of::<Tss>();

impl Tss {
    pub(crate) const fn new() -> Self {
        Self {
            reserved0: 0,
            rsp: [0; 3],
            reserved1: 0,
            ist: [0; IST_COUNT],
            reserved2: 0,
            reserved3: 0,
            // An offset >= the TSS limit means "no I/O permission bitmap".
            iomap_base: TSS_SIZE as u16,
        }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct Stack<const N: usize>([u8; N]);

/// Number of IST stacks allocated per CPU (IST1..=IST4).
const IST_STACKS: usize = 4;

/// Everything one CPU needs for privilege/interrupt stack switching.
#[repr(C)]
struct CpuArea {
    tss: Tss,
    kernel_stack: Stack<KERNEL_STACK_SIZE>,
    ist: [Stack<IST_STACK_SIZE>; IST_STACKS],
}

// Zero-initialised and interior-mutable, so they land in writable `.bss`.
// One area per logical CPU: a TSS is marked "busy" once loaded with `ltr`, so
// CPUs cannot share one, and each CPU needs its own emergency stacks.
const AREA_INIT: StaticCell<CpuArea> = StaticCell::new(CpuArea {
    tss: Tss::new(),
    kernel_stack: Stack([0; KERNEL_STACK_SIZE]),
    ist: [Stack([0; IST_STACK_SIZE]); IST_STACKS],
});
static AREAS: [StaticCell<CpuArea>; MAX_CPUS] = [AREA_INIT; MAX_CPUS];

fn area(cpu: usize) -> *mut CpuArea {
    AREAS[cpu].get()
}

/// Fills in CPU `cpu`'s default stacks. Called by `gdt::init_cpu` before `ltr`.
pub fn init(cpu: usize) {
    let a = area(cpu);
    // SAFETY: only taking addresses of fields of a static we own.
    unsafe {
        let ks = addr_of_mut!((*a).kernel_stack) as usize as u64 + KERNEL_STACK_SIZE as u64;
        set_kernel_stack(cpu, ks);
        for i in 0..IST_STACKS {
            // Stacks grow down: the initial RSP is one-past-the-end.
            let top = addr_of_mut!((*a).ist[i]) as usize as u64 + IST_STACK_SIZE as u64;
            let _ = set_ist_stack(cpu, i as u8 + 1, top);
        }
    }
}

/// Sets CPU `cpu`'s `rsp0`: the stack used on a ring 3 -> ring 0 transition.
/// A scheduler should update this on every context switch to the incoming
/// task's kernel stack top.
pub fn set_kernel_stack(cpu: usize, top: u64) {
    // SAFETY: the CPU only reads this field on privilege change.
    unsafe { addr_of_mut!((*area(cpu)).tss.rsp[0]).write_unaligned(top) };
    // The `syscall` instruction does not consult the TSS: keep a copy where its stub finds it.
    super::percpu::set_kernel_rsp(cpu, top);
}

pub fn kernel_stack(cpu: usize) -> u64 {
    // SAFETY: see `set_kernel_stack`.
    unsafe { addr_of_mut!((*area(cpu)).tss.rsp[0]).read_unaligned() }
}

/// Points IST slot `index` (1..=7) of CPU `cpu` at a stack whose *top* is `top`.
pub fn set_ist_stack(cpu: usize, index: u8, top: u64) -> Result<(), &'static str> {
    if !(1..=IST_COUNT as u8).contains(&index) {
        return Err("IST index must be in 1..=7");
    }
    // SAFETY: index checked above.
    unsafe { addr_of_mut!((*area(cpu)).tss.ist[index as usize - 1]).write_unaligned(top) };
    Ok(())
}

/// Top-of-stack currently programmed into IST slot `index`, if valid and set.
pub fn ist_stack(cpu: usize, index: u8) -> Option<u64> {
    if !(1..=IST_COUNT as u8).contains(&index) {
        return None;
    }
    // SAFETY: index checked above.
    let v = unsafe { addr_of_mut!((*area(cpu)).tss.ist[index as usize - 1]).read_unaligned() };
    if v == 0 { None } else { Some(v) }
}

/// (base address, limit) for CPU `cpu`'s TSS descriptor in the GDT.
pub fn base_and_limit(cpu: usize) -> (u64, u32) {
    // SAFETY: only takes the address of a field of a static.
    let base = unsafe { addr_of_mut!((*area(cpu)).tss) } as usize as u64;
    (base, (TSS_SIZE - 1) as u32)
}
