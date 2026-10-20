// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! The **Process Manager** (Ps): processes, threads, the scheduler and user mode.
//!
//! ```text
//!   Process (EPROCESS)  ── address space (vmm::AddressSpace) + own HandleTable
//!      │ 1..n
//!   Thread (ETHREAD)    ── kernel stack, saved context, priority, quantum, state
//!      │
//!   scheduler           ── 32 priority levels, preemptive, timer-driven (sched.rs)
//! ```
//!
//! Processes and threads are Object Manager objects (`Process`, `Thread`
//! types): they have handles with access rights, are reference counted, can
//! be waited on (a process/thread is "signaled" when it ends) and are destroyed
//! with their last reference. See `docs/17-process-manager.md`.
//!
//! * [`structs`]: `Process`, `Thread`, states, priorities, registries.
//! * [`sched`]: ready queues, `schedule`, the clock tick, sleep/wait, exit and terminate.
//! * `user`: ring-3 processes, the `int 0x80` system-call gate, user faults.

pub mod sched;
mod selftest;
pub mod structs;
mod user;

use core::sync::atomic::Ordering;

#[allow(unused_imports)]
pub use sched::{
    create_process, create_thread, current_handles, current_pid, current_process, current_thread, current_tid, exit_current, exit_current_process, process_infos,
    schedule, set_priority, sleep_ms, sleep_ticks, stats, terminate_process_threads, terminate_thread, thread_infos, ticks,
    yield_now, SchedStats, QUANTUM_TICKS, TICK_HZ,
};
pub use selftest::run_self_test;
#[allow(unused_imports)]
pub use structs::{
    priority, process_type, thread_type, Pid, Process, ProcessInfo, Thread, ThreadInfo, ThreadState, Tid, UserEntry, WaitStatus,
};
#[allow(unused_imports)]
pub use user::{dispatch_syscall, user_thread_entry, copy_from_user, copy_to_user, user_range_ok, spawn_user_image, spawn_user_process, Mapping, USER_CODE_BASE, USER_STACK_TOP, USER_STACK_PAGES, user_programs, UserProgram, SYS_EXIT, SYS_GETPID, SYS_SLEEP, SYS_WRITE, SYS_YIELD};

use crate::hal::{apic_timer, cpu};
use crate::ob::access::*;
use crate::ob::object::{self, ObRef, TypeInit};
use crate::ob::types::{event_type, Event};
use crate::ob::{Handle, HandleTable, Obj, ObError};

/// Pages per kernel stack (32 KiB), plus a guard page on each side.
pub const KSTACK_PAGES: u64 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PsError {
    NotReady,
    AlreadyStarted,
    InvalidArgument,
    ProcessExited,
    NotAThread,
    NotWaitable,
    /// The operation is not allowed on the calling thread.
    IsCurrent,
    AlreadyTerminated,
    Ob(ObError),
    Vm(crate::mm::vmm::VmError),
}

impl From<ObError> for PsError {
    fn from(e: ObError) -> Self {
        PsError::Ob(e)
    }
}

/// Registers the `Process` and `Thread` types, creates the System process (pid 0)
/// and makes the boot context its idle thread. Needs the Object Manager, the
/// VMM and the IDT. Boot CPU, once; scheduling starts with [`start`].
pub fn init() -> Result<(), PsError> {
    if !structs::PROCESS_TYPE.load(Ordering::Acquire).is_null() {
        return Err(PsError::AlreadyStarted);
    }
    let map = |read, write, exec, all| GenericMapping { read, write, execute: exec, all };
    let pt = object::create_type(TypeInit::of::<Process>(
        "Process",
        PROCESS_ALL,
        map(READ_CONTROL | PROCESS_QUERY, READ_CONTROL | PROCESS_CREATE_THREAD, READ_CONTROL | SYNCHRONIZE, PROCESS_ALL),
    ))?;
    structs::PROCESS_TYPE.store(pt as *const _ as *mut _, Ordering::Release);
    let tt = object::create_type(TypeInit::of::<Thread>(
        "Thread",
        THREAD_ALL,
        map(READ_CONTROL | THREAD_QUERY, READ_CONTROL | THREAD_SET_INFORMATION, READ_CONTROL | SYNCHRONIZE, THREAD_ALL),
    ))?;
    structs::THREAD_TYPE.store(tt as *const _ as *mut _, Ordering::Release);
    sched::init_system()?;
    user::init();
    Ok(())
}

/// Starts preemptive scheduling: installs the tick hook, creates the boot
/// "driver" thread running `driver`, starts the periodic timer. Afterwards
/// the caller must enable interrupts and call [`idle_loop`].
pub fn start(driver: fn(usize)) -> Result<(), PsError> {
    if sched::is_started() {
        return Err(PsError::AlreadyStarted);
    }
    let sys = sched::with_current(|t| t.process.clone());
    create_thread(&sys, "boot-driver", driver, 0, priority::NORMAL, None)?;
    drop(sys);
    sched::start_scheduling();
    apic_timer::set_tick_hook(sched::on_tick);
    apic_timer::start_periodic_hz(TICK_HZ).map_err(|_| PsError::NotReady)?;
    Ok(())
}

/// What the boot context does for the rest of time: it is the idle thread
/// (priority 0), so it only ever runs when nothing else can.
pub fn idle_loop() -> ! {
    loop {
        cpu::enable_interrupts();
        cpu::halt();
    }
}

/// Called by `Event::set`.
pub fn event_signaled(ev: &Event) {
    sched::event_signaled(ev);
}

/// A kernel thread in the System process.
pub fn spawn_kernel_thread(name: &str, entry: fn(usize), arg: usize, prio: u8) -> Result<Obj<Thread>, PsError> {
    let sys = sched::with_current(|t| t.process.clone());
    create_thread(&sys, name, entry, arg, prio, None)
}

/// Waits for an object to become signaled: an `Event`, a `Process` (ends) or a
/// `Thread` (ends). `timeout_ms = None` waits forever; `Some(0)` only polls.
pub fn wait_for_object(obj: &ObRef, timeout_ms: Option<u64>) -> Result<WaitStatus, PsError> {
    let timeout = timeout_ms.map(|ms| if ms == 0 { 0 } else { sched::ms_to_ticks(ms) });
    let ev: &Event = if let Some(e) = obj.body::<Event>(event_type()) {
        e
    } else if let Some(p) = obj.body::<Process>(process_type()) {
        &p.exit_event
    } else if let Some(t) = obj.body::<Thread>(thread_type()) {
        &t.exit_event
    } else {
        return Err(PsError::NotWaitable);
    };
    Ok(sched::wait_event(ev, timeout))
}

/// Like [`wait_for_object`] through a handle, which must grant `SYNCHRONIZE`.
pub fn wait_for_handle(table: &HandleTable, h: Handle, timeout_ms: Option<u64>) -> Result<WaitStatus, PsError> {
    let (obj, _) = table.reference_by_handle(h, SYNCHRONIZE, None)?;
    wait_for_object(&obj, timeout_ms)
}

/// Ends all threads of `process` and records `code`. Not for the caller's own process.
pub fn terminate_process(process: &Obj<Process>, code: i32) {
    terminate_process_threads(process, code);
}
