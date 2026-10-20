// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! The fast system-call gate: the `syscall` / `sysret` instruction pair.
//!
//! `int 0x80` (see `ps/user.rs`) still works and takes the same numbers; this
//! is the same dispatcher reached without the IDT, which is how real user
//! libraries call the kernel.
//!
//! **ABI** (identical to `int 0x80`, plus what the hardware adds):
//!
//! * number in `rax`; arguments in `rdi`, `rsi`, `rdx`, **`r10`**, `r8`, `r9`
//!   (`r10`, not `rcx`: the instruction overwrites `rcx`);
//! * result in `rax` (`>= 0`, or a negative error);
//! * `rcx` and `r11` are **clobbered** (they carry the return `rip` / `rflags`);
//!   every other register is preserved.
//!
//! **How it works.** The CPU does almost nothing on `syscall`: no stack
//! switch, no segment push, interrupts are only masked by `IA32_FMASK`. The
//! entry stub ([`lord_syscall_entry`](#)) therefore
//!
//! 1. parks the user `rsp` in this CPU's `PerCpu` block and loads the kernel stack of the
//!    running thread from there (`hal::percpu`, kept equal to TSS `rsp0` by `tss::set_kernel_stack`),
//! 2. builds the very same [`InterruptFrame`] an interrupt would have built
//!    (vector `0x80`, user `ss`/`cs`, `rip` = `rcx`, `rflags` = `r11`), so the
//!    system calls that block or end the thread, and the scheduler switching
//!    away in the middle of one, need no special case,
//! 3. calls the one Rust dispatcher, and
//! 4. returns with `sysretq` when it is safe (user `cs`, canonical `rip`: the
//!    non-canonical case is the well-known `sysret` privilege-escalation trap)
//!    and with `iretq` otherwise.
//!
//! No `swapgs` is needed: ring 3 cannot change the GS base (see `hal::percpu`).
//!
//! MSRs (set on every CPU by [`init_cpu`]): `STAR[47:32]` = kernel `cs` (`ss` =
//! that + 8), `STAR[63:48]` = `0x10`, so `sysret` loads `ss = 0x18|3` and
//! `cs = 0x20|3`: the user data / user code selectors of the GDT in `hal::gdt`,
//! which was laid out for exactly this. `LSTAR` = the entry stub. `FMASK`
//! clears `IF`, `TF`, `DF`, `NT` and `AC` on entry.

use core::arch::global_asm;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::gdt::{KERNEL_CODE, USER_CODE, USER_DATA};
use super::idt::InterruptFrame;
use super::msr::{self, IA32_EFER, IA32_FMASK, IA32_LSTAR, IA32_STAR};
use super::percpu::{OFF_KERNEL_RSP, OFF_USER_RSP};

const EFER_SCE: u64 = 1;
/// `IF | TF | DF | NT | AC`
const FMASK: u64 = 0x200 | 0x100 | 0x400 | 0x4000 | 0x4_0000;
/// The vector number put in the frame, so the dispatcher sees the same thing as for `int 0x80`.
const FRAME_VECTOR: u64 = 0x80;

static ENABLED: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicU64 = AtomicU64::new(0);

/// System calls that arrived through the `syscall` instruction so far.
pub fn count() -> u64 {
    CALLS.load(Ordering::Relaxed)
}

/// Has at least one CPU enabled the `syscall` gate?
pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Acquire)
}

unsafe extern "C" {
    fn lord_syscall_entry();
}

global_asm!(
    r#"
    .text
    .balign 16
    .global lord_syscall_entry
lord_syscall_entry:
    mov gs:[{user_rsp}], rsp
    mov rsp, gs:[{kernel_rsp}]
    push {user_ss}
    push qword ptr gs:[{user_rsp}]
    push r11
    push {user_cs}
    push rcx
    push 0
    push {vector}
    push rax
    push rcx
    push rdx
    push rbx
    push rbp
    push rsi
    push rdi
    push r8
    push r9
    push r10
    push r11
    push r12
    push r13
    push r14
    push r15
    mov rdi, rsp
    cld
    call {dispatch}
    pop r15
    pop r14
    pop r13
    pop r12
    pop r11
    pop r10
    pop r9
    pop r8
    pop rdi
    pop rsi
    pop rbp
    pop rbx
    pop rdx
    pop rcx
    pop rax
    add rsp, 16
    mov rcx, [rsp]
    cmp qword ptr [rsp + 8], {user_cs}
    jne 2f
    mov r11, rcx
    shr r11, 47
    jnz 2f
    mov r11, [rsp + 16]
    mov rsp, [rsp + 24]
    sysretq
2:
    mov r11, [rsp + 16]
    iretq
"#,
    user_rsp = const OFF_USER_RSP,
    kernel_rsp = const OFF_KERNEL_RSP,
    user_ss = const USER_DATA.raw() as u64,
    user_cs = const USER_CODE.raw() as u64,
    vector = const FRAME_VECTOR,
    dispatch = sym dispatch,
);

extern "C" fn dispatch(f: &mut InterruptFrame) {
    CALLS.fetch_add(1, Ordering::Relaxed);
    crate::ps::dispatch_syscall(f);
}

/// Address of the entry stub (function item -> pointer -> integer, as the lint wants).
fn entry_address() -> u64 {
    lord_syscall_entry as *const () as usize as u64
}

/// Enables `syscall`/`sysret` on the calling CPU. Needs the GDT (selector
/// layout), `percpu::init` and `tss::init` of this CPU to have run.
pub fn init_cpu() {
    // SAFETY: EFER, STAR, LSTAR and FMASK exist on every x86_64 CPU; the values are the
    // documented layout and the address of the stub above.
    unsafe {
        msr::wrmsr(IA32_EFER, msr::rdmsr(IA32_EFER) | EFER_SCE);
        let star = ((USER_DATA.raw() as u64 - 3 - 8) << 48) | ((KERNEL_CODE.raw() as u64) << 32);
        msr::wrmsr(IA32_STAR, star);
        msr::wrmsr(IA32_LSTAR, entry_address());
        msr::wrmsr(IA32_FMASK, FMASK);
    }
    ENABLED.store(true, Ordering::Release);
}

/// Reads the MSRs back and checks they hold what [`init_cpu`] wrote.
pub fn verify() -> Result<(), &'static str> {
    // SAFETY: see `init_cpu`.
    unsafe {
        if msr::rdmsr(IA32_EFER) & EFER_SCE == 0 {
            return Err("EFER.SCE is not set");
        }
        let star = msr::rdmsr(IA32_STAR);
        if (star >> 32) as u16 != KERNEL_CODE.raw() || (star >> 48) as u16 != 0x10 {
            return Err("STAR does not hold the GDT selector bases");
        }
        if msr::rdmsr(IA32_LSTAR) != entry_address() {
            return Err("LSTAR does not point at the entry stub");
        }
        if msr::rdmsr(IA32_FMASK) != FMASK {
            return Err("FMASK changed");
        }
    }
    Ok(())
}
