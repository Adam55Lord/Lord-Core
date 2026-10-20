// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Boot-time self-test of the `syscall` gate and the Executive services. One ring-3
//! program, using only the `syscall` instruction, walks through every call; it ends with
//! `exit(0)` if all of them behaved, or `exit(n)` where `n` names the first step that did not
//! (see [`failure_text`]).

use crate::hal::syscall as gate;
use crate::mm::{pmm, vmm};
use crate::ob::access::SYNCHRONIZE;
use crate::ps::{self, sched, structs::WaitStatus};

/// What [`run`] observed, for the boot log.
#[derive(Clone, Copy, Default)]
pub struct SelfTest {
    /// Calls that went through the `syscall` instruction.
    pub fast_calls: u64,
    pub steps: u32,
}

core::arch::global_asm!(
    r#"
    .section .rodata.lord_user, "a"

    .macro lord_sys nr
    mov eax, \nr
    syscall
    .endm

    # exit(code): the failure path of every check
    .macro lord_fail code
    mov edi, \code
    xor eax, eax
    syscall
    ud2
    .endm

    # result in rax must be >= 0
    .macro lord_ok code
    test rax, rax
    jns 8f
    lord_fail \code
8:
    .endm

    # rax must equal the immediate
    .macro lord_eq value, code
    cmp rax, \value
    je 8f
    lord_fail \code
8:
    .endm

    # rax must be negative (an error)
    .macro lord_err code
    test rax, rax
    js 8f
    lord_fail \code
8:
    .endm

    .global lord_user_systest_start
    .global lord_user_systest_end
lord_user_systest_start:
    sub rsp, 128

    # 1. getpid / gettid / time through `syscall`
    lord_sys 3
    test rax, rax
    jnz 1f
    lord_fail 1
1:
    lord_sys 6
    mov r15, rax
    lord_sys 5
    lord_ok 1

    # 2. sysinfo: 80 bytes, ABI version 1, page size 4096
    mov rdi, rsp
    mov esi, 80
    lord_sys 7
    lord_eq 80, 2
    cmp qword ptr [rsp], 1
    je 2f
    lord_fail 2
2:
    cmp qword ptr [rsp + 64], 4096
    je 3f
    lord_fail 2
3:

    # 3. memory: alloc 2 pages, write both, read back, protect, free
    xor edi, edi
    mov esi, 2
    mov edx, 1
    lord_sys 25
    lord_ok 3
    mov rbx, rax
    mov rax, 0x1122334455667788
    mov [rbx], rax
    mov [rbx + 4096 + 8], rax
    mov rcx, [rbx]
    cmp rcx, rax
    je 4f
    lord_fail 3
4:
    mov rcx, [rbx + 4096 + 8]
    cmp rcx, rax
    je 5f
    lord_fail 3
5:
    mov rdi, rbx
    mov esi, 2
    xor edx, edx
    lord_sys 27
    lord_ok 4
    mov rdi, rbx
    xor esi, esi
    lord_sys 26
    lord_eq 2, 5
    mov rdi, rbx
    xor esi, esi
    lord_sys 26
    lord_err 6
    xor edi, edi
    xor esi, esi
    mov edx, 1
    lord_sys 25
    lord_err 7

    # 4. events: create (notification, clear), time out, set, wait, reset
    xor edi, edi
    xor esi, esi
    lord_sys 22
    lord_ok 8
    mov r13, rax
    mov rdi, r13
    mov esi, 10
    lord_sys 21
    lord_eq 1, 9
    mov rdi, r13
    lord_sys 23
    lord_eq 0, 10
    mov rdi, r13
    mov esi, 10
    lord_sys 21
    lord_eq 0, 11
    mov rdi, r13
    lord_sys 24
    lord_eq 1, 12
    mov rdi, r13
    mov esi, 10
    lord_sys 21
    lord_eq 1, 13

    # 5. a second thread: it signals the event and exits with 77
    xor edi, edi
    mov esi, 2
    mov edx, 1
    lord_sys 25
    lord_ok 14
    lea rsi, [rax + 8192]
    lea rdi, [rip + lord_user_systest_thread]
    mov rdx, r13
    xor r10d, r10d
    lord_sys 19
    lord_ok 14
    mov r14, rax
    mov rdi, r13
    mov esi, 3000
    lord_sys 21
    lord_eq 0, 15
    mov rdi, r14
    mov esi, 3000
    lord_sys 21
    lord_eq 0, 16
    mov rdi, r14
    lord_sys 32
    lord_eq 77, 17

    # 6. a child process built from an image in our own memory: it exits with 5
    lea rdi, [rip + lord_user_systest_name]
    mov esi, 5
    lea rdx, [rip + lord_user_systest_child]
    mov r10d, offset lord_user_systest_child_len
    xor r8d, r8d
    lord_sys 28
    lord_ok 18
    mov rbp, rax
    mov rdi, rbp
    mov esi, 3000
    lord_sys 21
    lord_eq 0, 19
    mov rdi, rbp
    lord_sys 29
    lord_eq 5, 20

    # 7. a child that never stops is terminated from outside, exit code 9
    lea rdi, [rip + lord_user_systest_name]
    mov esi, 5
    lea rdx, [rip + lord_user_systest_spin]
    mov r10d, 2
    xor r8d, r8d
    lord_sys 28
    lord_ok 21
    mov r12, rax
    mov rdi, r12
    lord_sys 29
    lord_err 22
    mov rdi, r12
    mov esi, 9
    lord_sys 30
    lord_eq 0, 23
    mov rdi, r12
    mov esi, 3000
    lord_sys 21
    lord_eq 0, 24
    mov rdi, r12
    lord_sys 29
    lord_eq 9, 25

    # 7b. an LXF program started with lxf_spawn: relocation, imports from ltmath.llf, LibraryMain
    lea rdi, [rip + lord_user_systest_name]
    mov esi, 5
    lea rdx, [rip + lord_user_systest_lxf]
    mov r10d, offset lord_user_systest_lxf_len
    xor r8d, r8d
    lord_sys 33
    lord_ok 33
    mov r12, rax
    mov rdi, r12
    mov esi, 5000
    lord_sys 21
    lord_eq 0, 34
    mov rdi, r12
    lord_sys 29
    lord_eq 1764, 35
    # a damaged image is refused, a library is not a program
    lea rdi, [rip + lord_user_systest_name]
    mov esi, 5
    lea rdx, [rip + lord_user_systest_name]
    mov r10d, 5
    xor r8d, r8d
    lord_sys 33
    lord_err 36

    # 8. bad pointers and unknown numbers fail cleanly instead of hurting the kernel
    mov edi, 0x10
    mov esi, 4
    lord_sys 2
    lord_eq -1, 26
    lord_sys 9999
    lord_eq -1, 27
    mov edi, 0x10
    mov esi, 80
    lord_sys 7
    lord_err 28
    mov rdi, 0xffff800000000000
    mov esi, 80
    lord_sys 7
    lord_err 28
    mov edi, 999
    mov esi, 1
    lord_sys 21
    lord_err 29

    # 9. handles can be closed; a closed one is gone
    mov rdi, r13
    lord_sys 10
    lord_ok 30
    mov rdi, r13
    lord_sys 10
    lord_err 31

    # 10. the thread id is stable
    lord_sys 6
    cmp rax, r15
    je 6f
    lord_fail 32
6:
    xor eax, eax
    xor edi, edi
    syscall
    ud2

lord_user_systest_thread:
    mov r12, rdi
    mov rdi, r12
    mov eax, 23
    syscall
    mov eax, 20
    mov edi, 77
    syscall
    ud2

lord_user_systest_name:
    .ascii "child"

lord_user_systest_child:
    mov eax, 2
    lea rdi, [rip + lord_user_systest_hello]
    mov esi, 39
    syscall
    xor eax, eax
    mov edi, 5
    syscall
    ud2
lord_user_systest_hello:
    .ascii "child process started by process_create"
lord_user_systest_child_end:
    .set lord_user_systest_child_len, lord_user_systest_child_end - lord_user_systest_child

lord_user_systest_lxf:
    .incbin "src/ld/samples/ltcalc.lxf"
lord_user_systest_lxf_end:
    .set lord_user_systest_lxf_len, lord_user_systest_lxf_end - lord_user_systest_lxf

lord_user_systest_spin:
7:
    jmp 7b
lord_user_systest_end:
"#
);

unsafe extern "C" {
    static lord_user_systest_start: u8;
    static lord_user_systest_end: u8;
}

/// What a failure code of the test program means.
pub fn failure_text(code: i32) -> &'static str {
    match code {
        1 => "getpid / gettid / time through `syscall` failed",
        2 => "sysinfo returned the wrong data",
        3 => "vm_alloc failed or the memory is not usable",
        4 => "vm_protect failed",
        5 => "vm_free did not free both pages",
        6 => "vm_free accepted an address twice",
        7 => "vm_alloc accepted zero pages",
        8 | 9 | 10 | 11 | 12 | 13 => "event_create / event_set / event_reset / wait gave a wrong answer",
        14 => "thread_create failed",
        15 => "a user thread did not signal the event",
        16 => "waiting for a user thread did not end",
        17 => "thread_exit_code was wrong",
        18 | 19 | 20 => "process_create / wait on a child / process_exit_code failed",
        21 | 22 | 23 | 24 | 25 => "process_terminate on a running child failed",
        26 => "write with a bad pointer did not fail",
        27 => "an unknown system call number did not fail",
        28 => "a bad user pointer was accepted",
        29 => "wait on an invalid handle did not fail",
        30 | 31 => "close misbehaved",
        32 => "gettid changed",
        33 | 34 | 35 => "lxf_spawn: the LXF program did not start / finish / return 1764",
        36 => "lxf_spawn accepted a damaged image",
        _ if code & (0xC000_0000u32 as i32) == 0xC000_0000u32 as i32 => "the test program faulted in ring 3",
        _ => "the test program failed in an unknown way",
    }
}

pub(crate) fn balance() -> u64 {
    pmm::stats().in_use_frames - vmm::stats().committed_pages
}

/// Runs the self-test. Thread context (needs the scheduler), after `hal::syscall::init_cpu`.
pub fn run() -> Result<SelfTest, &'static str> {
    gate::verify()?;
    let bal0 = balance();
    let fast0 = gate::count();
    // SAFETY: both symbols delimit the read-only blob above.
    let image = unsafe {
        core::slice::from_raw_parts(
            &raw const lord_user_systest_start,
            &raw const lord_user_systest_end as usize - &raw const lord_user_systest_start as usize,
        )
    };
    let p = ps::spawn_user_process("systest", image, ps::structs::priority::NORMAL).map_err(|_| "cannot start the test program")?;
    match ps::wait_for_object(p.untyped(), Some(15_000)) {
        Ok(WaitStatus::Signaled) => {}
        _ => return Err("the test program did not finish in time"),
    }
    let code = p.exit_code();
    if code != 0 {
        return Err(failure_text(code));
    }
    let fast_calls = gate::count() - fast0;
    if fast_calls < 48 {
        return Err("too few calls went through the `syscall` instruction");
    }
    drop(p);
    sched::sleep_ms(60); // the scheduler releases ended threads after the next switch
    if balance() != bal0 {
        return Err("frames leaked");
    }
    let _ = SYNCHRONIZE;
    Ok(SelfTest { fast_calls, steps: 36 })
}
