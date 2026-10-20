// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Global Descriptor Table (GDT) with privilege-ring management.
//!
//! x86_64 long mode ignores segment bases and limits, but the GDT still
//! decides the *privilege level* (ring) of running code through the code and
//! data segment descriptors, and it hosts the TSS descriptor.
//!
//! Layout (selector = index * 8 | RPL):
//!
//! | idx | selector | descriptor                  |
//! |-----|----------|-----------------------------|
//! |  0  | `0x00`   | null                        |
//! |  1  | `0x08`   | kernel code, ring 0, 64-bit |
//! |  2  | `0x10`   | kernel data, ring 0         |
//! |  3  | `0x1B`   | user data,   ring 3         |
//! |  4  | `0x23`   | user code,   ring 3, 64-bit |
//! | 5-6 | `0x28`   | 64-bit TSS (16-byte entry)  |
//!
//! User data deliberately comes *before* user code: this is the order the
//! `syscall`/`sysret` instructions require (STAR base 0x10 -> SS=0x1B,
//! CS=0x23), so adding fast system calls later needs no re-layout.
//!
//! Rings 1 and 2 are not used by mainstream kernels (and `syscall`/paging
//! cannot distinguish them well), so only [`Ring::Ring0`] and [`Ring::Ring3`]
//! exist here.

use core::arch::asm;
use core::mem::size_of;
use core::sync::atomic::{AtomicBool, Ordering};

use super::sync::StaticCell;
use super::tss;
use super::MAX_CPUS;

/// CPU privilege level. Ring 0 = kernel, ring 3 = user mode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Ring {
    Ring0 = 0,
    Ring3 = 3,
}

impl Ring {
    pub const fn number(self) -> u8 {
        self as u8
    }

    pub const fn from_number(n: u8) -> Option<Ring> {
        match n {
            0 => Some(Ring::Ring0),
            3 => Some(Ring::Ring3),
            _ => None,
        }
    }
}

/// A segment selector: GDT index (bits 3..), table indicator (bit 2, always
/// GDT here) and requested privilege level (bits 0..1).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SegmentSelector(u16);

impl SegmentSelector {
    pub const fn new(index: u16, rpl: Ring) -> Self {
        Self((index << 3) | rpl as u16)
    }

    pub const fn raw(self) -> u16 {
        self.0
    }

    pub const fn index(self) -> u16 {
        self.0 >> 3
    }

    pub const fn rpl(self) -> u16 {
        self.0 & 3
    }
}

pub const KERNEL_CODE: SegmentSelector = SegmentSelector::new(1, Ring::Ring0);
pub const KERNEL_DATA: SegmentSelector = SegmentSelector::new(2, Ring::Ring0);
pub const USER_DATA: SegmentSelector = SegmentSelector::new(3, Ring::Ring3);
pub const USER_CODE: SegmentSelector = SegmentSelector::new(4, Ring::Ring3);
pub const TSS_SELECTOR: SegmentSelector = SegmentSelector::new(5, Ring::Ring0);

/// The code segment selector for running at `ring`.
pub const fn code_selector(ring: Ring) -> SegmentSelector {
    match ring {
        Ring::Ring0 => KERNEL_CODE,
        Ring::Ring3 => USER_CODE,
    }
}

/// The data/stack segment selector for running at `ring`.
pub const fn data_selector(ring: Ring) -> SegmentSelector {
    match ring {
        Ring::Ring0 => KERNEL_DATA,
        Ring::Ring3 => USER_DATA,
    }
}

// ----------------------------------------------------------------------
// Descriptor encoding
// ----------------------------------------------------------------------

const ACCESS_PRESENT: u8 = 1 << 7;
const ACCESS_NON_SYSTEM: u8 = 1 << 4; // S bit: code/data (not a system segment)
const ACCESS_EXECUTABLE: u8 = 1 << 3;
const ACCESS_RW: u8 = 1 << 1; // readable (code) / writable (data)
const ACCESS_TSS_AVAILABLE: u8 = 0x9; // system segment type: 64-bit TSS (available)

const FLAG_GRANULARITY: u8 = 1 << 3;
const FLAG_SIZE_32: u8 = 1 << 2; // D/B: must be 0 on 64-bit code segments
const FLAG_LONG: u8 = 1 << 1; // L: 64-bit code segment

const fn segment(access: u8, flags: u8) -> u64 {
    // Base 0, limit 0xFFFFF (ignored in long mode, kept conventional).
    0xFFFF | ((access as u64) << 40) | (0xF << 48) | (((flags & 0xF) as u64) << 52)
}

const fn dpl_bits(ring: Ring) -> u8 {
    (ring as u8) << 5
}

const fn code_descriptor(ring: Ring) -> u64 {
    segment(
        ACCESS_PRESENT | dpl_bits(ring) | ACCESS_NON_SYSTEM | ACCESS_EXECUTABLE | ACCESS_RW,
        FLAG_GRANULARITY | FLAG_LONG,
    )
}

const fn data_descriptor(ring: Ring) -> u64 {
    segment(
        ACCESS_PRESENT | dpl_bits(ring) | ACCESS_NON_SYSTEM | ACCESS_RW,
        FLAG_GRANULARITY | FLAG_SIZE_32,
    )
}

/// A 64-bit TSS descriptor is 16 bytes, i.e. two GDT slots.
const fn tss_descriptor(base: u64, limit: u32) -> [u64; 2] {
    let low = (limit as u64 & 0xFFFF)
        | ((base & 0x00FF_FFFF) << 16)
        | (((ACCESS_PRESENT | ACCESS_TSS_AVAILABLE) as u64) << 40)
        | (((limit as u64 >> 16) & 0xF) << 48)
        | (((base >> 24) & 0xFF) << 56);
    [low, base >> 32]
}

/// Number of 8-byte GDT slots (the TSS descriptor takes two).
pub const GDT_ENTRIES: usize = 7;

// Each CPU has its own GDT: the TSS descriptor inside differs per CPU.
const GDT_INIT: StaticCell<[u64; GDT_ENTRIES]> = StaticCell::new([0; GDT_ENTRIES]);
static GDTS: [StaticCell<[u64; GDT_ENTRIES]>; MAX_CPUS] = [GDT_INIT; MAX_CPUS];
const FLAG_INIT: AtomicBool = AtomicBool::new(false);
static INITIALISED: [AtomicBool; MAX_CPUS] = [FLAG_INIT; MAX_CPUS];

#[repr(C, packed)]
struct TablePointer {
    limit: u16,
    base: u64,
}

/// Builds CPU `cpu`'s GDT and TSS, loads them *on the calling CPU*, reloads
/// every segment register and loads the task register. Every CPU must call
/// this for itself, once, early, with interrupts off; repeat calls are ignored
/// (re-running `ltr` on a busy TSS would fault).
pub fn init_cpu(cpu: usize) {
    assert!(cpu < MAX_CPUS, "cpu index out of range");
    if INITIALISED[cpu].swap(true, Ordering::AcqRel) {
        return;
    }
    let gdt = GDTS[cpu].get();

    tss::init(cpu);
    let (tss_base, tss_limit) = tss::base_and_limit(cpu);
    let tss_desc = tss_descriptor(tss_base, tss_limit);

    let table = [
        0,
        code_descriptor(Ring::Ring0),
        data_descriptor(Ring::Ring0),
        data_descriptor(Ring::Ring3),
        code_descriptor(Ring::Ring3),
        tss_desc[0],
        tss_desc[1],
    ];

    // SAFETY: interrupts are off, this CPU is the only user of its own GDT
    // slot, and the bootloader's temporary GDT is what is active until `lgdt` below.
    unsafe {
        gdt.write(table);

        let pointer = TablePointer {
            limit: (size_of::<[u64; GDT_ENTRIES]>() - 1) as u16,
            base: gdt as usize as u64,
        };
        asm!("lgdt [{}]", in(reg) &pointer, options(readonly, nostack, preserves_flags));

        reload_code_segment(KERNEL_CODE);
        reload_data_segments(KERNEL_DATA);
        asm!("ltr {0:x}", in(reg) TSS_SELECTOR.raw(), options(nostack, preserves_flags));
    }
}

/// Reloads CS with a far return (the only way to change CS in long mode
/// without an interrupt).
///
/// # Safety
/// `selector` must name a valid 64-bit code descriptor in the loaded GDT.
unsafe fn reload_code_segment(selector: SegmentSelector) {
    unsafe {
        asm!(
            "push {sel}",
            "lea {tmp}, [2f + rip]",
            "push {tmp}",
            "retfq",
            "2:",
            sel = in(reg) selector.raw() as u64,
            tmp = lateout(reg) _,
            options(preserves_flags),
        );
    }
}

/// Loads DS/ES/SS with `selector` and nulls FS/GS.
///
/// # Safety
/// `selector` must name a valid writable data descriptor in the loaded GDT.
unsafe fn reload_data_segments(selector: SegmentSelector) {
    unsafe {
        asm!(
            "mov ds, {0:x}",
            "mov es, {0:x}",
            "mov ss, {0:x}",
            "mov fs, {1:x}",
            "mov gs, {1:x}",
            in(reg) selector.raw(),
            in(reg) 0u16,
            options(nostack, preserves_flags),
        );
    }
}

// ----------------------------------------------------------------------
// Introspection and self-test
// ----------------------------------------------------------------------

pub fn table_base(cpu: usize) -> u64 {
    GDTS[cpu].get() as usize as u64
}

pub fn current_cs() -> u16 {
    let v: u16;
    // SAFETY: reading a segment register has no side effects.
    unsafe { asm!("mov {0:x}, cs", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

pub fn current_ss() -> u16 {
    let v: u16;
    // SAFETY: reading a segment register has no side effects.
    unsafe { asm!("mov {0:x}, ss", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

pub fn current_tr() -> u16 {
    let v: u16;
    // SAFETY: `str` just stores the task register.
    unsafe { asm!("str {0:x}", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

/// The ring the CPU is executing in right now (low bits of CS).
pub fn current_ring() -> Option<Ring> {
    Ring::from_number((current_cs() & 3) as u8)
}

/// Access-rights dword for `selector` (via `lar`), or `None` if invalid.
/// Bits 13..14 hold the descriptor's DPL, bit 15 the present flag.
pub fn access_rights(selector: SegmentSelector) -> Option<u32> {
    let rights: u32;
    let valid: u8;
    // SAFETY: LAR only reads descriptor info; it reports failure through ZF.
    unsafe {
        asm!(
            "lar {rights:e}, {sel:x}",
            "setz {valid}",
            rights = lateout(reg) rights,
            sel = in(reg) selector.raw(),
            valid = lateout(reg_byte) valid,
            options(nomem, nostack),
        );
    }
    if valid != 0 { Some(rights) } else { None }
}

/// DPL stored in the descriptor behind `selector`.
pub fn descriptor_dpl(selector: SegmentSelector) -> Option<u8> {
    access_rights(selector).map(|r| ((r >> 13) & 3) as u8)
}

/// Checks that the calling CPU's hardware state matches what `init_cpu(cpu)`
/// meant to set up.
pub fn verify(cpu: usize) -> Result<(), &'static str> {
    if current_cs() != KERNEL_CODE.raw() {
        return Err("CS is not the kernel code selector");
    }
    if current_ss() != KERNEL_DATA.raw() {
        return Err("SS is not the kernel data selector");
    }
    if current_tr() != TSS_SELECTOR.raw() {
        return Err("TR is not the TSS selector");
    }

    let mut gdtr = TablePointer { limit: 0, base: 0 };
    // SAFETY: SGDT stores the 10-byte descriptor-table register to memory.
    unsafe { asm!("sgdt [{}]", in(reg) &mut gdtr, options(nostack, preserves_flags)) };
    let (limit, base) = (gdtr.limit, gdtr.base);
    if base != table_base(cpu) || limit as usize != size_of::<[u64; GDT_ENTRIES]>() - 1 {
        return Err("GDTR does not point at our GDT");
    }

    let expect = [
        (KERNEL_CODE, 0u8, "kernel code descriptor DPL != 0"),
        (KERNEL_DATA, 0, "kernel data descriptor DPL != 0"),
        (USER_DATA, 3, "user data descriptor DPL != 3"),
        (USER_CODE, 3, "user code descriptor DPL != 3"),
    ];
    for (sel, dpl, msg) in expect {
        if descriptor_dpl(sel) != Some(dpl) {
            return Err(msg);
        }
    }
    Ok(())
}

// ----------------------------------------------------------------------
// Ring transitions
// ----------------------------------------------------------------------

/// RFLAGS bit 1 is reserved and always reads as 1.
pub const RFLAGS_RESERVED: u64 = 1 << 1;
/// RFLAGS interrupt-enable flag.
pub const RFLAGS_IF: u64 = 1 << 9;

/// Drops to ring 3 with `iretq`: loads CS/SS with the user selectors,
/// `rip = entry`, `rsp = stack_top`, `rflags = rflags`, `rdi = arg` (the first
/// argument of the entry function; every other register is zeroed).
///
/// Requirements (none are checked here):
/// * `entry` and the stack must be mapped *user-accessible* in the page
///   tables (the U/S bit set at every level),
/// * `set RFLAGS_IF` only once an IDT exists to handle interrupts,
/// * `tss::set_kernel_stack` should point at a valid kernel stack so the next
///   interrupt from ring 3 has somewhere to land.
///
/// # Safety
/// Transfers control to arbitrary code and never returns.
pub unsafe fn enter_ring3(entry: u64, stack_top: u64, rflags: u64, arg: u64) -> ! {
    unsafe {
        asm!(
            "push {user_ss}",
            "push {user_sp}",
            "push {user_fl}",
            "push {user_cs}",
            "push {user_ip}",
            // Nothing from the kernel may leak into user registers.
            "xor eax, eax",
            "xor ebx, ebx",
            "xor ecx, ecx",
            "xor edx, edx",
            "xor esi, esi",
            "xor ebp, ebp",
            "xor r8d, r8d",
            "xor r9d, r9d",
            "xor r10d, r10d",
            "xor r11d, r11d",
            "xor r12d, r12d",
            "xor r13d, r13d",
            "xor r14d, r14d",
            "xor r15d, r15d",
            "iretq",
            in("rdi") arg,
            user_ss = in(reg) USER_DATA.raw() as u64,
            user_sp = in(reg) stack_top,
            user_fl = in(reg) rflags | RFLAGS_RESERVED,
            user_cs = in(reg) USER_CODE.raw() as u64,
            user_ip = in(reg) entry,
            options(noreturn),
        );
    }
}
