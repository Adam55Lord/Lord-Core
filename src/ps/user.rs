// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! User mode: processes with their own address space, the `int 0x80`
//! system-call gate and what happens when a user program faults.
//!
//! A user process is an [`AddressSpace`](crate::mm::vmm::AddressSpace) holding
//! a code image (mapped read+execute, written while still read+write: W^X
//! holds) and a stack, and one thread that starts in ring 3. System calls use
//! the `syscall` instruction (`hal/syscall.rs`, the normal way) or the software
//! interrupt `int 0x80` (kept as a debug path; the gate is callable from ring 3), both
//! with the number in `rax` and arguments in `rdi`, `rsi`, `rdx`, `r10`, `r8`, `r9`;
//! the result comes back in `rax`. Both land in [`dispatch_syscall`].
//!
//! | `rax` | call | arguments | result |
//! |---|---|---|---|
//! | 0 | `exit` | `rdi` = exit code | does not return; ends the process |
//! | 1 | `yield` | | 0 |
//! | 2 | `write` | `rdi` = pointer, `rsi` = length (<= 256) | bytes written, or -1 |
//! | 3 | `getpid` | | pid |
//! | 4 | `sleep` | `rdi` = milliseconds | 0 |
//! | 5-7, 19-33 | Executive services | see [`crate::ex`] | value >= 0, or a negative error |
//! | 8-18 | IPC Manager | see [`crate::ipc::syscall`] | value >= 0, or a negative error |
//!
//! Pointers are validated page by page against the caller's address space
//! before the kernel reads them. A CPU exception in ring 3 (page fault,
//! #UD, #GP, ...) ends only that process, with exit code `0xC000_0000 | vector`.

use core::sync::atomic::{AtomicU64, Ordering};

use super::sched;
use super::structs::*;
use super::PsError;
use crate::hal::idt::{self, InterruptFrame};
use crate::hal::memory::phys_to_virt;
use crate::hal::gdt;
use crate::hal::paging::{self, pte};
use crate::mm::vmm::{self, AddressSpace, Prot};
use crate::ob::Obj;

pub const SYS_EXIT: u64 = 0;
pub const SYS_YIELD: u64 = 1;
pub const SYS_WRITE: u64 = 2;
pub const SYS_GETPID: u64 = 3;
pub const SYS_SLEEP: u64 = 4;

const SYSCALL_VECTOR: u8 = 0x80;
const MAX_WRITE: usize = 256;

/// Where program images are mapped.
pub const USER_CODE_BASE: u64 = 0x0000_0000_0040_0000;
/// Exclusive top of the user stack (the page below it is the first stack page).
pub const USER_STACK_TOP: u64 = 0x0000_7FFF_FFFF_F000;
pub const USER_STACK_PAGES: u64 = 4;

static SYSCALLS: AtomicU64 = AtomicU64::new(0);
static USER_WRITES: AtomicU64 = AtomicU64::new(0);

/// System calls handled so far.
pub fn syscall_count() -> u64 {
    SYSCALLS.load(Ordering::Relaxed)
}
/// Successful `write` calls so far.
pub fn user_write_count() -> u64 {
    USER_WRITES.load(Ordering::Relaxed)
}

// ----------------------------------------------------------------------
// Built-in programs (assembled into the kernel image, copied into user space)
// ----------------------------------------------------------------------

core::arch::global_asm!(
    r#"
    .section .rodata.lord_user, "a"

    # prints a line 3 times (yielding in between), then exit(42)
    .global lord_user_hello_start
    .global lord_user_hello_end
lord_user_hello_start:
    mov r12d, 3
2:
    mov eax, 2
    lea rdi, [rip + lord_user_hello_msg]
    mov esi, 30
    int 0x80
    mov eax, 1
    int 0x80
    dec r12d
    jnz 2b
    xor eax, eax
    mov edi, 42
    int 0x80
    ud2
lord_user_hello_msg:
    .ascii "user process running in ring 3"
lord_user_hello_end:

    # writes to address 0: a page fault in ring 3
    .global lord_user_fault_start
    .global lord_user_fault_end
lord_user_fault_start:
    xor eax, eax
    mov dword ptr [rax], 1
    ud2
lord_user_fault_end:

    # never gives up the CPU: only preemption (or termination) stops it
    .global lord_user_spin_start
    .global lord_user_spin_end
lord_user_spin_start:
3:
    jmp 3b
lord_user_spin_end:

    # remembers its pid in a register across a sleep, then exit(pid)
    .global lord_user_pid_start
    .global lord_user_pid_end
lord_user_pid_start:
    mov eax, 3
    int 0x80
    mov r12d, eax
    mov eax, 4
    mov edi, 20
    int 0x80
    mov edi, r12d
    xor eax, eax
    int 0x80
    ud2
lord_user_pid_end:
"#
);

unsafe extern "C" {
    static lord_user_hello_start: u8;
    static lord_user_hello_end: u8;
    static lord_user_fault_start: u8;
    static lord_user_fault_end: u8;
    static lord_user_spin_start: u8;
    static lord_user_spin_end: u8;
    static lord_user_pid_start: u8;
    static lord_user_pid_end: u8;
}

/// A program image built into the kernel.
#[derive(Clone, Copy)]
pub struct UserProgram {
    pub name: &'static str,
    pub image: &'static [u8],
}

fn image(start: *const u8, end: *const u8) -> &'static [u8] {
    // SAFETY: both symbols delimit a read-only blob in the kernel image.
    unsafe { core::slice::from_raw_parts(start, end as usize - start as usize) }
}

/// The built-in test programs: `hello`, `fault`, `spin`, `pid`.
pub fn user_programs() -> [UserProgram; 4] {
    [
        UserProgram { name: "hello", image: image(&raw const lord_user_hello_start, &raw const lord_user_hello_end) },
        UserProgram { name: "fault", image: image(&raw const lord_user_fault_start, &raw const lord_user_fault_end) },
        UserProgram { name: "spin", image: image(&raw const lord_user_spin_start, &raw const lord_user_spin_end) },
        UserProgram { name: "pid", image: image(&raw const lord_user_pid_start, &raw const lord_user_pid_end) },
    ]
}

// ----------------------------------------------------------------------
// Process creation
// ----------------------------------------------------------------------

/// One range of a new process's address space: `bytes` copied to `va`, rounded up to whole
/// zero-filled pages, then protected as `prot`.
pub struct Mapping<'a> {
    pub va: u64,
    pub bytes: &'a [u8],
    pub prot: Prot,
}

/// Creates a user process from ready-made mappings (what the LXF loader, `crate::ld`, produces)
/// and starts its main thread at `rip` with a fresh stack. The ranges must be page-aligned,
/// below the stack, and must not overlap.
pub fn spawn_user_image(name: &str, maps: &[Mapping], rip: u64, prio: u8) -> Result<Obj<Process>, PsError> {
    let stack_base = USER_STACK_TOP - USER_STACK_PAGES * 4096;
    let mut sorted: alloc::vec::Vec<(u64, u64)> = alloc::vec::Vec::new();
    for m in maps {
        let pages = m.bytes.len().div_ceil(4096) as u64;
        let end = pages.checked_mul(4096).and_then(|b| m.va.checked_add(b));
        match end {
            Some(e) if pages > 0 && m.va % 4096 == 0 && e <= stack_base => sorted.push((m.va, e)),
            _ => return Err(PsError::InvalidArgument),
        }
    }
    sorted.sort_unstable();
    if sorted.windows(2).any(|w| w[0].1 > w[1].0) || !sorted.iter().any(|&(a, e)| (a..e).contains(&rip)) {
        return Err(PsError::InvalidArgument);
    }
    let mut space = AddressSpace::new().map_err(PsError::Vm)?;
    for m in maps {
        let pages = m.bytes.len().div_ceil(4096) as u64;
        // Write while the pages are writable, then flip them to their real protection (W^X).
        space.map_new(m.va, pages, Prot::READ_WRITE.user()).map_err(PsError::Vm)?;
        for (i, chunk) in m.bytes.chunks(4096).enumerate() {
            let phys = space.translate(m.va + i as u64 * 4096).ok_or(PsError::NotReady)?;
            // SAFETY: the frame is ours (just mapped, zeroed) and reachable through the HHDM.
            unsafe { core::ptr::copy_nonoverlapping(chunk.as_ptr(), phys_to_virt(phys), chunk.len()) };
        }
        space.protect(m.va, pages, m.prot).map_err(PsError::Vm)?;
    }
    space.map_new(stack_base, USER_STACK_PAGES, Prot::READ_WRITE.user()).map_err(PsError::Vm)?;

    let process = sched::create_process(name, sched::current_pid(), prio, Some(space))?;
    let entry = UserEntry { rip, rsp: USER_STACK_TOP, arg: 0 };
    sched::create_thread(&process, "main", user_thread_entry, 0, prio, Some(entry))?;
    Ok(process)
}

/// Creates a user process running `program` (position-independent code mapped
/// at [`USER_CODE_BASE`]) and starts its main thread.
pub fn spawn_user_process(name: &str, program: &[u8], prio: u8) -> Result<Obj<Process>, PsError> {
    if program.is_empty() || program.len() > 64 * 4096 {
        return Err(PsError::InvalidArgument);
    }
    spawn_user_image(name, &[Mapping { va: USER_CODE_BASE, bytes: program, prot: Prot::READ_EXEC.user() }], USER_CODE_BASE, prio)
}

/// Kernel half of a user thread: drops to ring 3 and never comes back.
pub fn user_thread_entry(_: usize) {
    let entry = sched::with_current(|t| t.user).expect("user thread without an entry");
    // SAFETY: the address space of this thread's process is loaded (the scheduler did
    // that before switching to us) and maps `entry.rip` executable and `entry.rsp` writable.
    unsafe { gdt::enter_ring3(entry.rip, entry.rsp, 0x202, entry.arg) }
}

// ----------------------------------------------------------------------
// System calls and faults
// ----------------------------------------------------------------------

/// Walks the user pages `[va, va + len)` of the calling process and calls
/// `f(frame_virtual_address, offset_in_range, chunk_len)` for each page-sized
/// chunk. Fails if any page is unmapped, not a user page, or (when `write`) not
/// writable: the kernel never writes through a read-only user mapping.
fn walk_user(va: u64, len: usize, write: bool, mut f: impl FnMut(*mut u8, usize, usize)) -> bool {
    if va.checked_add(len as u64).is_none_or(|e| e > vmm::USER_END) {
        return false;
    }
    let root = sched::with_current(|t| t.process.root.load(Ordering::Relaxed));
    if root == 0 {
        return false;
    }
    let mut done = 0usize;
    while done < len {
        let a = va + done as u64;
        let n = (len - done).min(4096 - (a & 0xfff) as usize);
        let Some(m) = paging::translate_in(root, a) else { return false };
        if m.entry & pte::USER == 0 || (write && m.entry & pte::WRITABLE == 0) {
            return false;
        }
        f(phys_to_virt(m.phys + (a & 0xfff)), done, n);
        done += n;
    }
    true
}

/// Copies `out.len()` bytes from user address `va` of the calling process.
pub fn copy_from_user(va: u64, out: &mut [u8]) -> bool {
    let len = out.len();
    let dst = out.as_mut_ptr();
    // SAFETY (closure): `src` is a mapped user frame; chunk `[off, off+n)` lies inside `out`.
    walk_user(va, len, false, |src, off, n| unsafe { core::ptr::copy_nonoverlapping(src, dst.add(off), n) })
}

/// Copies `data` to user address `va` of the calling process (the pages must be user-writable).
pub fn copy_to_user(va: u64, data: &[u8]) -> bool {
    let src = data.as_ptr();
    // SAFETY (closure): `dst` is a mapped, writable user frame; chunk `[off, off+n)` lies inside `data`.
    walk_user(va, data.len(), true, |dst, off, n| unsafe { core::ptr::copy_nonoverlapping(src.add(off), dst, n) })
}

/// Can the calling process read (or, with `write`, write) `[va, va + len)`?
pub fn user_range_ok(va: u64, len: usize, write: bool) -> bool {
    walk_user(va, len, write, |_, _, _| {})
}

fn sys_write(ptr: u64, len: u64) -> u64 {
    if len as usize > MAX_WRITE {
        return u64::MAX;
    }
    let mut buf = [0u8; MAX_WRITE];
    let buf = &mut buf[..len as usize];
    if !copy_from_user(ptr, buf) {
        return u64::MAX;
    }
    USER_WRITES.fetch_add(1, Ordering::Relaxed);
    match core::str::from_utf8(buf) {
        Ok(s) => info!("user: [pid {}] {}", sched::current_pid(), s),
        Err(_) => info!("user: [pid {}] <{} bytes of binary>", sched::current_pid(), len),
    }
    len
}

/// The one system-call dispatcher: reached from `int 0x80` (the IDT handler below) and from
/// the `syscall` instruction (`hal/syscall.rs`). `f.rax` holds the number on entry and the
/// result on return.
pub fn dispatch_syscall(f: &mut InterruptFrame) {
    SYSCALLS.fetch_add(1, Ordering::Relaxed);
    match f.rax {
        SYS_EXIT => sched::exit_current_process(f.rdi as i32),
        SYS_YIELD => {
            f.rax = 0;
            sched::yield_now();
        }
        SYS_WRITE => f.rax = sys_write(f.rdi, f.rsi),
        SYS_GETPID => f.rax = sched::current_pid(),
        SYS_SLEEP => {
            f.rax = 0;
            sched::sleep_ms(f.rdi);
        }
        // IPC Manager (ports and sections): numbers 8 and up.
        n if crate::ipc::syscall::handles(n) => crate::ipc::syscall::dispatch(f),
        // Executive services (time, threads, waiting, events, memory, processes).
        n if crate::ex::handles(n) => crate::ex::dispatch(f),
        _ => f.rax = u64::MAX,
    }
}

fn user_fault(f: &mut InterruptFrame) {
    let code = (0xC000_0000u32 | f.vector as u32) as i32;
    let cr2 = if f.vector == 14 { idt::fault_address() } else { 0 };
    warn!(
        "ps:   pid {} faulted: vector {} error {:#x} at rip={:#x}{}: terminating the process",
        sched::current_pid(),
        f.vector,
        f.error_code,
        f.rip,
        if f.vector == 14 { alloc::format!(", address {:#x}", cr2) } else { alloc::string::String::new() }
    );
    sched::exit_current_process(code)
}

pub(super) fn init() {
    idt::register(SYSCALL_VECTOR, dispatch_syscall);
    idt::allow_from_user_mode(SYSCALL_VECTOR);
    idt::set_user_fault_hook(user_fault);
}
