// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! The **Executive services**: the system calls that let a user program use the
//! kernel beyond the first five (`ps/user.rs`) and IPC (`ipc/syscall.rs`): time,
//! threads, waiting, events, memory and child processes.
//!
//! They are reached like every other system call: the `syscall` instruction
//! (`hal/syscall.rs`) or `int 0x80`, number in `rax`, arguments in `rdi`, `rsi`,
//! `rdx`, `r10`, `r8`, `r9`, result in `rax` (`>= 0`, or a **negative error**, the
//! same codes as [`IpcError::code`](crate::ipc::IpcError::code)). Every object a call
//! names is a **handle** of the caller's process, checked by the Object Manager
//! against the right the call needs. A *timeout* is in milliseconds: `0` polls,
//! `u64::MAX` waits forever.
//!
//! | `rax` | call | arguments | returns |
//! |---|---|---|---|
//! | 5 | `time` | | milliseconds since boot |
//! | 6 | `gettid` | | thread id |
//! | 7 | `sysinfo` | buffer, length | bytes written; ten `u64`: ABI version (1), tick Hz, ticks, uptime ms, pid, tid, free frames, managed frames, page size, CPUs online |
//! | 19 | `thread_create` | entry, stack top (16-aligned), argument (-> `rdi`), priority (0 = normal) | thread handle |
//! | 20 | `thread_exit` | exit code | does not return; the process ends with its last thread |
//! | 21 | `wait` | handle (event, thread or process), timeout | 0 = signaled, 1 = timed out |
//! | 22 | `event_create` | kind (0 notification, 1 synchronization), initially signaled | event handle |
//! | 23 | `event_set` | event | previous state (0/1) |
//! | 24 | `event_reset` | event | previous state (0/1) |
//! | 25 | `vm_alloc` | address hint (0 = any), pages, flags (1 = writable, 2 = executable) | address of zeroed pages |
//! | 26 | `vm_free` | address returned by `vm_alloc`, pages (0 = all of it) | pages freed |
//! | 27 | `vm_protect` | address, pages, flags (as `vm_alloc`, 0 = read only) | 0 |
//! | 28 | `process_create` | name ptr, name len, image ptr, image len, priority (0 = normal) | process handle |
//! | 29 | `process_exit_code` | process handle | exit code; `WouldBlock` while it runs |
//! | 30 | `process_terminate` | process handle, exit code | 0 (ends the caller if it is its own process) |
//! | 31 | `set_priority` | thread handle, priority 1-15 | 0 |
//! | 32 | `thread_exit_code` | thread handle | exit code; `WouldBlock` while it runs |
//! | 33 | `lxf_spawn` | name ptr, name len, LXF image ptr, image len (<= 1 MiB), priority (0 = normal) | process handle (see [`crate::ld`]) |
//!
//! `lxf_spawn` starts an **LXF executable** (`tools/lxf/`) through the kernel's LXF loader
//! ([`crate::ld`]): it relocates it, links it against the registered `LLF` libraries and runs
//! it; the program's return value (or its own `exit`) is the exit code. A bad image fails with
//! `InvalidArgument` (`NotFound` for a missing library or export).
//!
//! `process_create` copies a flat, position-independent program image (at most
//! 64 pages) out of the caller's memory and starts it at `0x400000` with a fresh
//! address space, like the loader does for the built-in programs. User priorities
//! stop at 15 (the real-time levels 16-31 are for the kernel).
//!
//! `vm_alloc` hands out zeroed, user-accessible pages in a window of the address
//! space of its own (`0x2000_0000_0000..0x4000_0000_0000`, a 4 KiB guard gap after
//! each automatic allocation); at most 4096 pages per call and 16384 per process.
//! `vm_free` and `vm_protect` only touch what `vm_alloc` returned, never the
//! program's code or stack. Writable and executable never go together.
//!
//! `wait` parks the object it waits on in the *thread* (not on its stack), so a
//! thread killed while blocked leaks nothing.

pub mod selftest;

use alloc::string::String;
use alloc::vec;
use core::sync::atomic::Ordering;

use crate::hal::idt::InterruptFrame;
use crate::ipc::IpcError;
use crate::mm::vmm::{self, Prot, VmError};
use crate::mm::pmm;
use crate::ob::access::*;
use crate::ob::types::{create_event, event_type, Event, EventKind};
use crate::ob::{Handle, HandleTable, ObRef, ObjectAttributes};
use crate::ps::structs::{priority, process_type, thread_type, Process, Thread, UserEntry, WaitStatus};
use crate::ps::{self, sched, copy_from_user, copy_to_user, current_handles, current_process, user_range_ok, PsError};

pub const SYS_TIME: u64 = 5;
pub const SYS_GETTID: u64 = 6;
pub const SYS_SYSINFO: u64 = 7;
pub const SYS_THREAD_CREATE: u64 = 19;
pub const SYS_THREAD_EXIT: u64 = 20;
pub const SYS_WAIT: u64 = 21;
pub const SYS_EVENT_CREATE: u64 = 22;
pub const SYS_EVENT_SET: u64 = 23;
pub const SYS_EVENT_RESET: u64 = 24;
pub const SYS_VM_ALLOC: u64 = 25;
pub const SYS_VM_FREE: u64 = 26;
pub const SYS_VM_PROTECT: u64 = 27;
pub const SYS_PROCESS_CREATE: u64 = 28;
pub const SYS_PROCESS_EXIT_CODE: u64 = 29;
pub const SYS_PROCESS_TERMINATE: u64 = 30;
pub const SYS_SET_PRIORITY: u64 = 31;
pub const SYS_THREAD_EXIT_CODE: u64 = 32;
pub const SYS_LXF_SPAWN: u64 = 33;

/// Version of the system-call ABI reported by `sysinfo`.
pub const ABI_VERSION: u64 = 1;
/// Words `sysinfo` writes.
pub const SYSINFO_WORDS: usize = 10;

/// Start of the window `vm_alloc` allocates from.
pub const VM_BASE: u64 = 0x0000_2000_0000_0000;
const VM_LIMIT: u64 = 0x0000_4000_0000_0000;
const MAX_ALLOC_PAGES: u64 = 4096;
const MAX_PROCESS_PAGES: u64 = 16384;
const MAX_IMAGE_PAGES: usize = 64;
const MAX_NAME: usize = 32;
const PAGE: u64 = 4096;
const INFINITE: u64 = u64::MAX;

type Res = Result<u64, IpcError>;

/// Is `n` one of ours?
pub fn handles(n: u64) -> bool {
    matches!(n, SYS_TIME..=SYS_SYSINFO | SYS_THREAD_CREATE..=SYS_LXF_SPAWN)
}

fn ps_err(e: PsError) -> IpcError {
    match e {
        PsError::Ob(o) => IpcError::from(o),
        PsError::Vm(v) => vm_err(v),
        PsError::ProcessExited => IpcError::NoProcess,
        PsError::NotAThread | PsError::NotWaitable => IpcError::TypeMismatch,
        PsError::NotReady => IpcError::NotReady,
        _ => IpcError::InvalidArgument,
    }
}

fn vm_err(e: VmError) -> IpcError {
    match e {
        VmError::OutOfFrames | VmError::OutOfVirtualSpace | VmError::TooManyRegions => IpcError::OutOfResources,
        VmError::NotMapped | VmError::NotFound => IpcError::NotFound,
        _ => IpcError::InvalidArgument,
    }
}

fn timeout(v: u64) -> Option<u64> {
    if v == INFINITE { None } else { Some(v) }
}

/// A user priority: 0 means normal, 1-15 are allowed, the real-time levels are not.
fn user_priority(v: u64) -> Result<u8, IpcError> {
    match v {
        0 => Ok(priority::NORMAL),
        1..=15 => Ok(v as u8),
        _ => Err(IpcError::InvalidArgument),
    }
}

/// `1` writable, `2` executable; both is refused (W^X).
fn user_prot(flags: u64) -> Result<Prot, IpcError> {
    match flags {
        0 => Ok(Prot::READ.user()),
        1 => Ok(Prot::READ_WRITE.user()),
        2 => Ok(Prot::READ_EXEC.user()),
        _ => Err(IpcError::InvalidArgument),
    }
}

fn now_ms() -> u64 {
    sched::ticks() * 1000 / ps::TICK_HZ as u64
}

// ----------------------------------------------------------------------
// Information
// ----------------------------------------------------------------------

fn sys_sysinfo(buf: u64, len: u64) -> Res {
    let pm = pmm::stats();
    let words: [u64; SYSINFO_WORDS] = [
        ABI_VERSION,
        ps::TICK_HZ as u64,
        sched::ticks(),
        now_ms(),
        sched::current_pid(),
        sched::current_tid(),
        pm.free_frames,
        pm.managed_frames,
        PAGE,
        crate::hal::smp::online_cpus() as u64,
    ];
    let mut bytes = [0u8; SYSINFO_WORDS * 8];
    for (c, w) in bytes.chunks_exact_mut(8).zip(words) {
        c.copy_from_slice(&w.to_le_bytes());
    }
    let n = (len as usize).min(bytes.len());
    if !copy_to_user(buf, &bytes[..n]) {
        return Err(IpcError::BadAddress);
    }
    Ok(n as u64)
}

// ----------------------------------------------------------------------
// Threads
// ----------------------------------------------------------------------

fn sys_thread_create(table: &HandleTable, entry: u64, stack_top: u64, arg: u64, prio: u64) -> Res {
    let prio = user_priority(prio)?;
    if entry == 0 || stack_top < 16 || stack_top > vmm::USER_END || stack_top & 0xF != 0 {
        return Err(IpcError::InvalidArgument);
    }
    if !user_range_ok(entry, 1, false) || !user_range_ok(stack_top - 8, 8, true) {
        return Err(IpcError::BadAddress);
    }
    let thread = {
        let process = current_process();
        sched::create_thread(&process, "user", ps::user_thread_entry, 0, prio, Some(UserEntry { rip: entry, rsp: stack_top, arg }))
            .map_err(ps_err)?
    };
    match table.create_handle(thread.untyped(), THREAD_ALL, 0) {
        Ok(h) => Ok(h.0 as u64),
        Err(e) => {
            let _ = sched::terminate_thread(thread.untyped(), -1); // do not leave it running unowned
            Err(IpcError::from(e))
        }
    }
}

fn sys_set_priority(table: &HandleTable, h: u64, prio: u64) -> Res {
    if prio == 0 {
        return Err(IpcError::InvalidArgument);
    }
    let prio = user_priority(prio)?;
    let (obj, _) = table.reference_by_handle(Handle(h as usize), THREAD_SET_INFORMATION, Some(thread_type()))?;
    sched::set_priority(&obj, prio).map_err(ps_err)?;
    Ok(0)
}

fn sys_thread_exit_code(table: &HandleTable, h: u64) -> Res {
    let (obj, _) = table.reference_by_handle(Handle(h as usize), THREAD_QUERY, Some(thread_type()))?;
    let t = obj.body::<Thread>(thread_type()).ok_or(IpcError::TypeMismatch)?;
    if !t.exit_event.is_signaled() {
        return Err(IpcError::WouldBlock);
    }
    Ok(t.exit_code() as i64 as u64)
}

// ----------------------------------------------------------------------
// Waiting and events
// ----------------------------------------------------------------------

/// The scheduler event whose signal means "this object is done".
fn waitable_event(obj: &ObRef) -> Option<*const Event> {
    if let Some(e) = obj.body::<Event>(event_type()) {
        Some(e as *const Event)
    } else if let Some(p) = obj.body::<Process>(process_type()) {
        Some(&*p.exit_event as *const Event)
    } else {
        obj.body::<Thread>(thread_type()).map(|t| &*t.exit_event as *const Event)
    }
}

fn sys_wait(table: &HandleTable, h: u64, t: u64) -> Res {
    let (obj, _) = table.reference_by_handle(Handle(h as usize), SYNCHRONIZE, None)?;
    let ev = waitable_event(&obj).ok_or(IpcError::TypeMismatch)?;
    let ticks = timeout(t).map(|ms| if ms == 0 { 0 } else { sched::ms_to_ticks(ms) });
    // The reference moves into the thread: a thread killed while it sleeps drops it with
    // itself instead of leaking it from a stack that is never unwound.
    drop(sched::hold_current(obj));
    // SAFETY: the thread holds a reference, so the event (part of the object) stays alive.
    let status = sched::wait_event(unsafe { &*ev }, ticks);
    drop(sched::release_held());
    Ok((status == WaitStatus::TimedOut) as u64)
}

fn sys_event_create(table: &HandleTable, kind: u64, signaled: u64) -> Res {
    let kind = match kind {
        0 => EventKind::Notification,
        1 => EventKind::Synchronization,
        _ => return Err(IpcError::InvalidArgument),
    };
    let ev = create_event(&ObjectAttributes::unnamed(), kind, signaled != 0)?;
    Ok(table.create_handle(ev.untyped(), EVENT_ALL, 0)?.0 as u64)
}

fn sys_event_change(table: &HandleTable, h: u64, set: bool) -> Res {
    let (obj, _) = table.reference_by_handle(Handle(h as usize), EVENT_MODIFY_STATE, Some(event_type()))?;
    let ev = obj.body::<Event>(event_type()).ok_or(IpcError::TypeMismatch)?;
    Ok((if set { ev.set() } else { ev.reset() }) as u64)
}

// ----------------------------------------------------------------------
// Memory
// ----------------------------------------------------------------------

fn overlaps(allocs: &[(u64, u64)], base: u64, pages: u64) -> bool {
    allocs.iter().any(|&(b, n)| base < b + n * PAGE && b < base + pages * PAGE)
}

fn sys_vm_alloc(hint: u64, pages: u64, flags: u64) -> Res {
    if pages == 0 || pages > MAX_ALLOC_PAGES {
        return Err(IpcError::InvalidArgument);
    }
    let prot = user_prot(flags)?;
    let p = current_process();
    let mut allocs = p.vm_allocs.lock();
    if allocs.iter().map(|a| a.1).sum::<u64>() + pages > MAX_PROCESS_PAGES {
        return Err(IpcError::OutOfResources);
    }
    let base = if hint != 0 {
        if hint % PAGE != 0 || hint < VM_BASE || hint.checked_add(pages * PAGE).is_none_or(|e| e > VM_LIMIT) || overlaps(&allocs, hint, pages) {
            return Err(IpcError::InvalidArgument);
        }
        hint
    } else {
        let span = (pages + 1) * PAGE;
        loop {
            let b = p.vm_cursor.fetch_add(span, Ordering::Relaxed);
            if b + span > VM_LIMIT {
                return Err(IpcError::OutOfResources);
            }
            if !overlaps(&allocs, b, pages) {
                break b;
            }
        }
    };
    p.with_space(|s| s.map_new(base, pages, prot)).ok_or(IpcError::NoProcess)?.map_err(vm_err)?;
    allocs.push((base, pages));
    Ok(base)
}

fn sys_vm_free(addr: u64, pages: u64) -> Res {
    let p = current_process();
    let mut allocs = p.vm_allocs.lock();
    let i = allocs.iter().position(|a| a.0 == addr).ok_or(IpcError::NotFound)?;
    let n = allocs[i].1;
    if pages != 0 && pages != n {
        return Err(IpcError::InvalidArgument);
    }
    allocs.swap_remove(i);
    p.with_space(|s| s.unmap(addr, n)).ok_or(IpcError::NoProcess)?.map_err(vm_err)?;
    Ok(n)
}

fn sys_vm_protect(addr: u64, pages: u64, flags: u64) -> Res {
    let prot = user_prot(flags)?;
    if pages == 0 || addr % PAGE != 0 {
        return Err(IpcError::InvalidArgument);
    }
    let end = pages.checked_mul(PAGE).and_then(|b| addr.checked_add(b)).ok_or(IpcError::InvalidArgument)?;
    let p = current_process();
    let allocs = p.vm_allocs.lock();
    if !allocs.iter().any(|&(b, n)| addr >= b && end <= b + n * PAGE) {
        return Err(IpcError::NotFound);
    }
    p.with_space(|s| s.protect(addr, pages, prot)).ok_or(IpcError::NoProcess)?.map_err(vm_err)?;
    Ok(0)
}

// ----------------------------------------------------------------------
// Processes
// ----------------------------------------------------------------------

fn ld_err(e: crate::ld::LdError) -> IpcError {
    use crate::ld::LdError;
    match e {
        LdError::MissingLibrary | LdError::MissingExport => IpcError::NotFound,
        LdError::TooBig | LdError::TooManyModules => IpcError::OutOfResources,
        LdError::Ps(p) => ps_err(p),
        _ => IpcError::InvalidArgument,
    }
}

fn sys_process_create(table: &HandleTable, a: [u64; 6], lxf: bool) -> Res {
    let [name_ptr, name_len, img_ptr, img_len, prio, _] = a;
    let prio = user_priority(prio)?;
    let max_image = if lxf { crate::ld::MAX_IMAGE_BYTES } else { MAX_IMAGE_PAGES * PAGE as usize };
    if name_len as usize > MAX_NAME || img_len == 0 || img_len as usize > max_image {
        return Err(IpcError::InvalidArgument);
    }
    let mut name = vec![0u8; name_len as usize];
    if !copy_from_user(name_ptr, &mut name) {
        return Err(IpcError::BadAddress);
    }
    let name = String::from_utf8(name).unwrap_or_else(|_| String::from("user"));
    let mut image = vec![0u8; img_len as usize];
    if !copy_from_user(img_ptr, &mut image) {
        return Err(IpcError::BadAddress);
    }
    let name = if name.is_empty() { "user" } else { &name };
    let process = if lxf {
        crate::ld::spawn(name, &image, prio).map_err(ld_err)?
    } else {
        ps::spawn_user_process(name, &image, prio).map_err(ps_err)?
    };
    drop(image);
    match table.create_handle(process.untyped(), PROCESS_ALL, 0) {
        Ok(h) => Ok(h.0 as u64),
        Err(e) => {
            ps::terminate_process(&process, -1);
            Err(IpcError::from(e))
        }
    }
}

fn sys_process_exit_code(table: &HandleTable, h: u64) -> Res {
    let (obj, _) = table.reference_by_handle(Handle(h as usize), PROCESS_QUERY, Some(process_type()))?;
    let p = obj.body::<Process>(process_type()).ok_or(IpcError::TypeMismatch)?;
    if !p.has_exited() {
        return Err(IpcError::WouldBlock);
    }
    Ok(p.exit_code() as i64 as u64)
}

fn sys_process_terminate(table: &HandleTable, h: u64, code: u64) -> Res {
    let (obj, _) = table.reference_by_handle(Handle(h as usize), PROCESS_TERMINATE, Some(process_type()))?;
    let me = sched::current_pid();
    let target = obj.into_typed::<Process>(process_type()).map_err(|_| IpcError::TypeMismatch)?;
    if target.pid == me {
        drop(target); // `exit_current_process` never returns: keep no reference on this stack
        sched::exit_current_process(code as i32);
    }
    ps::terminate_process(&target, code as i32);
    Ok(0)
}

// ----------------------------------------------------------------------
// Dispatch
// ----------------------------------------------------------------------

/// Handles one Executive system call; `f.rax` holds the number on entry and the result on return.
pub fn dispatch(f: &mut InterruptFrame) {
    // SAFETY: the table belongs to the calling thread's process, which the thread keeps alive.
    let table: &HandleTable = unsafe { &*current_handles() };
    let a = [f.rdi, f.rsi, f.rdx, f.r10, f.r8, f.r9];
    let result = match f.rax {
        SYS_TIME => Ok(now_ms()),
        SYS_GETTID => Ok(sched::current_tid()),
        SYS_SYSINFO => sys_sysinfo(a[0], a[1]),
        SYS_THREAD_CREATE => sys_thread_create(table, a[0], a[1], a[2], a[3]),
        SYS_THREAD_EXIT => sched::exit_current(a[0] as i32),
        SYS_WAIT => sys_wait(table, a[0], a[1]),
        SYS_EVENT_CREATE => sys_event_create(table, a[0], a[1]),
        SYS_EVENT_SET => sys_event_change(table, a[0], true),
        SYS_EVENT_RESET => sys_event_change(table, a[0], false),
        SYS_VM_ALLOC => sys_vm_alloc(a[0], a[1], a[2]),
        SYS_VM_FREE => sys_vm_free(a[0], a[1]),
        SYS_VM_PROTECT => sys_vm_protect(a[0], a[1], a[2]),
        SYS_PROCESS_CREATE => sys_process_create(table, a, false),
        SYS_LXF_SPAWN => sys_process_create(table, a, true),
        SYS_PROCESS_EXIT_CODE => sys_process_exit_code(table, a[0]),
        SYS_PROCESS_TERMINATE => sys_process_terminate(table, a[0], a[1]),
        SYS_SET_PRIORITY => sys_set_priority(table, a[0], a[1]),
        SYS_THREAD_EXIT_CODE => sys_thread_exit_code(table, a[0]),
        _ => Err(IpcError::InvalidArgument),
    };
    f.rax = match result {
        Ok(v) => v,
        Err(e) => e.code() as u64,
    };
}
