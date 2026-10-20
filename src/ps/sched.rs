// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! The scheduler: ready queues, context switching, preemption, sleeping and waiting.
//!
//! # Model
//!
//! * One CPU schedules (the boot CPU); the APs stay parked until the VMM can
//!   shoot down TLBs (see `docs/17-process-manager.md`). All state is kept
//!   per scheduler, so adding CPUs means adding instances, not redesigning.
//! * **Preemptive, priority-driven round robin.** 32 priority levels, a FIFO
//!   ready queue per level, a bitmap to find the highest non-empty one in O(1).
//!   The highest-priority ready thread runs; equal priorities share the CPU in
//!   quanta of [`QUANTUM_TICKS`] timer ticks.
//! * **The clock** is the LAPIC timer at [`TICK_HZ`]. Its interrupt handler
//!   ([`on_tick`]) wakes sleepers, charges the running thread, and when its
//!   quantum is gone, or something more important became ready, switches
//!   *from inside the interrupt*: the interrupted thread is simply suspended
//!   with its interrupt frame on its own kernel stack and resumes through
//!   `iretq` when it is scheduled again.
//! * **Voluntary switches** (`yield_now`, sleep, wait, exit) call
//!   [`schedule`] directly.
//! * **Boost / decay**: a variable-priority thread that finishes a wait
//!   is boosted one level (max 15); each full quantum it burns decays the
//!   boost. This keeps interactive threads responsive without starving CPU hogs.
//!
//! # Context switch
//!
//! `lord_context_switch(old_slot, new_rsp)` pushes the callee-saved registers
//! (`rbp rbx r12 r13 r14 r15`), stores `rsp` in `*old_slot`, loads `new_rsp`,
//! pops the same registers and returns. Everything else is saved by the
//! compiler (calling convention) or by the interrupt entry code. A new
//! thread's stack is built so that the first `ret` lands in [`thread_start`].
//! The kernel is built soft-float, so there is no FPU/SSE state to switch.
//!
//! # Rules
//!
//! * Every kernel lock masks interrupts (`IrqLock`), so a thread can never
//!   be preempted while holding one. Blocking calls must not be made while
//!   holding a lock.
//! * A terminated thread's stack is freed only after the CPU has left it
//!   ([`finish_switch`] drops the zombie list on the *next* thread's stack).
//! * No `ObRef` is kept in a local across a context switch (a thread that
//!   never resumes would leak it); the scheduler's own structures own the
//!   references.

use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use super::structs::*;
use super::{PsError, KSTACK_PAGES};
use crate::hal::idt::InterruptFrame;
use crate::hal::sync::{self, IrqLock};
use crate::hal::{cpu, paging, tss};
use crate::mm::vmm;
use crate::ob::object::{self, ObRef};
use crate::ob::types::{create_event, Event, EventKind};
use crate::ob::{Obj, ObjectAttributes};

/// Scheduler clock rate.
pub const TICK_HZ: u32 = 100;
/// Timer ticks a thread may run before equal-priority peers get the CPU.
pub const QUANTUM_TICKS: u32 = 2;

pub fn ms_to_ticks(ms: u64) -> u64 {
    (ms * TICK_HZ as u64).div_ceil(1000).max(1)
}

// ----------------------------------------------------------------------
// State
// ----------------------------------------------------------------------

struct Sched {
    ready: [VecDeque<ObRef>; priority::LEVELS],
    bitmap: u32,
    /// (wake tick, tid) -> thread; sleeps and timed waits.
    sleepers: BTreeMap<(u64, Tid), ObRef>,
    /// Threads blocked on an event: (thread, event address).
    waiters: Vec<(ObRef, usize)>,
    /// Terminated threads whose last scheduler reference is dropped after the next switch.
    zombies: Vec<ObRef>,
    current: Option<ObRef>,
    idle: Option<ObRef>,
}

static SCHED: IrqLock<Sched> = IrqLock::new(Sched {
    ready: [const { VecDeque::new() }; priority::LEVELS],
    bitmap: 0,
    sleepers: BTreeMap::new(),
    waiters: Vec::new(),
    zombies: Vec::new(),
    current: None,
    idle: None,
});

static STARTED: AtomicBool = AtomicBool::new(false);
static TICKS: AtomicU64 = AtomicU64::new(0);
static SWITCHES: AtomicU64 = AtomicU64::new(0);
static PREEMPTIONS: AtomicU64 = AtomicU64::new(0);
static NEED_RESCHED: AtomicBool = AtomicBool::new(false);
static WAITER_COUNT: AtomicUsize = AtomicUsize::new(0);
static LOADED_ROOT: AtomicU64 = AtomicU64::new(0);
static KERNEL_ROOT: AtomicU64 = AtomicU64::new(0);

pub fn ticks() -> u64 {
    TICKS.load(Ordering::Relaxed)
}
pub fn switches() -> u64 {
    SWITCHES.load(Ordering::Relaxed)
}
pub fn preemptions() -> u64 {
    PREEMPTIONS.load(Ordering::Relaxed)
}
pub fn is_started() -> bool {
    STARTED.load(Ordering::Acquire)
}

fn thr_ptr(r: &ObRef) -> *const Thread {
    r.body::<Thread>(thread_type()).expect("not a thread object") as *const Thread
}

/// # Safety
/// `p` must point to a live thread body (the caller holds a reference to its object).
unsafe fn th<'a>(p: *const Thread) -> &'a Thread {
    // SAFETY: guaranteed by the caller.
    unsafe { &*p }
}

// ----------------------------------------------------------------------
// Context switch
// ----------------------------------------------------------------------

core::arch::global_asm!(
    r#"
    .global lord_context_switch
lord_context_switch:
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15
    mov [rdi], rsp
    mov rsp, rsi
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
"#
);

unsafe extern "C" {
    /// Saves the callee-saved registers on the current stack, stores the stack
    /// pointer in `*old_slot`, switches to `new_rsp` and restores from there.
    fn lord_context_switch(old_slot: *mut u64, new_rsp: u64);
}

/// Lays out a fresh kernel stack so the first switch to it "returns" into
/// [`thread_start`] with a correctly aligned stack. Returns the initial `rsp`.
///
/// # Safety
/// `top` must be the 16-byte-aligned end of a mapped, writable, unused stack.
pub(super) unsafe fn build_initial_stack(top: u64) -> u64 {
    let ret_slot = top - 16;
    let sp = ret_slot - 48; // six callee-saved registers
    // SAFETY: the caller guarantees the memory is ours and mapped.
    unsafe {
        let p = sp as *mut u64;
        for i in 0..6 {
            p.add(i).write(0);
        }
        (ret_slot as *mut u64).write(thread_start as *const () as usize as u64);
        ((top - 8) as *mut u64).write(0); // fake return address: backtraces stop here
    }
    sp
}

/// First code every new thread runs (reached by the `ret` of the first switch).
extern "C" fn thread_start() -> ! {
    finish_switch();
    cpu::enable_interrupts();
    let (entry, arg) = {
        let s = SCHED.lock();
        // SAFETY: `current` is owned by the scheduler.
        let t = unsafe { th(thr_ptr(s.current.as_ref().expect("no current thread"))) };
        (t.entry, t.arg)
    };
    entry(arg);
    exit_current(0)
}

fn switch_address_space(root: u64) {
    let target = if root == 0 { KERNEL_ROOT.load(Ordering::Relaxed) } else { root };
    if LOADED_ROOT.swap(target, Ordering::AcqRel) != target {
        // SAFETY: `target` is the kernel root or a live process's PML4, both
        // of which map the kernel half (code, stacks, IDT).
        unsafe { paging::load_root(target) };
    }
}

/// Runs on a thread's own stack right after it gets the CPU: releases the
/// terminated threads queued by the previous one (now that we are off their stacks).
fn finish_switch() {
    let dead = {
        let mut s = SCHED.lock();
        if s.zombies.is_empty() {
            return;
        }
        core::mem::take(&mut s.zombies)
    };
    drop(dead);
}

// ----------------------------------------------------------------------
// Ready queues
// ----------------------------------------------------------------------

fn make_ready(s: &mut Sched, t: ObRef) {
    let tp = thr_ptr(&t);
    // SAFETY: `t` is a live thread object.
    let tt = unsafe { th(tp) };
    tt.set_state(ThreadState::Ready);
    let p = tt.priority() as usize;
    if let Some(cur) = &s.current {
        // SAFETY: `current` is owned by the scheduler.
        if tt.priority() > unsafe { th(thr_ptr(cur)) }.priority() {
            NEED_RESCHED.store(true, Ordering::Release);
        }
    }
    s.ready[p].push_back(t);
    s.bitmap |= 1 << p;
}

fn pick_next(s: &mut Sched) -> ObRef {
    if s.bitmap != 0 {
        let p = 31 - s.bitmap.leading_zeros() as usize;
        let q = &mut s.ready[p];
        let t = q.pop_front().expect("bitmap says non-empty");
        if q.is_empty() {
            s.bitmap &= !(1 << p);
        }
        t
    } else {
        s.idle.clone().expect("no idle thread")
    }
}

fn remove_ready(s: &mut Sched, t: &ObRef) -> Option<ObRef> {
    // SAFETY: live thread object.
    let p = unsafe { th(thr_ptr(t)) }.priority() as usize;
    let q = &mut s.ready[p];
    let pos = q.iter().position(|x| x.same_object(t))?;
    let r = q.remove(pos);
    if q.is_empty() {
        s.bitmap &= !(1 << p);
    }
    r
}

fn wake_thread(s: &mut Sched, t: ObRef, result: u8) {
    // SAFETY: live thread object.
    let tt = unsafe { th(thr_ptr(&t)) };
    tt.wait_result.store(result, Ordering::Release);
    tt.wait_event.store(0, Ordering::Release);
    tt.wake_tick.store(0, Ordering::Release);
    if tt.base_priority() < priority::MAX_VARIABLE + 1 {
        let boosted = (tt.base_priority() + 1).min(priority::MAX_VARIABLE);
        if boosted > tt.priority() {
            tt.priority.store(boosted, Ordering::Relaxed);
        }
    }
    tt.quantum_left.store(QUANTUM_TICKS, Ordering::Relaxed);
    make_ready(s, t);
}

fn drop_waiter(s: &mut Sched, addr: usize) -> Option<ObRef> {
    let pos = s.waiters.iter().position(|(w, _)| w.addr() == addr)?;
    WAITER_COUNT.fetch_sub(1, Ordering::AcqRel);
    Some(s.waiters.remove(pos).0)
}

// ----------------------------------------------------------------------
// schedule()
// ----------------------------------------------------------------------

/// Picks the best thread and switches to it (possibly the same one).
/// Callable from thread context and from interrupt handlers; never while
/// holding a lock.
pub fn schedule() {
    if !STARTED.load(Ordering::Acquire) {
        return;
    }
    let was = sync::irq_save();
    let switch = {
        let mut s = SCHED.lock();
        let cur = s.current.clone().expect("no current thread");
        let curp = thr_ptr(&cur);
        // SAFETY: owned by the scheduler (`current`).
        let cur_t = unsafe { th(curp) };
        let cur_is_idle = cur.same_object(s.idle.as_ref().expect("no idle thread"));
        if cur_t.state() == ThreadState::Running {
            cur_t.set_state(ThreadState::Ready);
            if !cur_is_idle {
                let p = cur_t.priority() as usize;
                s.ready[p].push_back(cur.clone());
                s.bitmap |= 1 << p;
            }
        }
        let next = pick_next(&mut s);
        let nextp = thr_ptr(&next);
        // SAFETY: `next` came from the scheduler's own queues.
        let next_t = unsafe { th(nextp) };
        next_t.set_state(ThreadState::Running);
        next_t.quantum_left.store(QUANTUM_TICKS, Ordering::Relaxed);
        if next.same_object(&cur) {
            None
        } else {
            next_t.switches_in.fetch_add(1, Ordering::Relaxed);
            SWITCHES.fetch_add(1, Ordering::Relaxed);
            s.current = Some(next);
            // `cur` and `next` locals die here; the queues / `current` / zombie
            // list own the threads, so the raw pointers below stay valid.
            Some((curp, nextp))
        }
    };

    if let Some((old, new)) = switch {
        // SAFETY: both threads are owned by the scheduler's structures.
        let (old_t, new_t) = unsafe { (th(old), th(new)) };
        tss::set_kernel_stack(0, new_t.rsp0);
        switch_address_space(new_t.process.root.load(Ordering::Relaxed));
        // SAFETY: interrupts are off; `new_t.saved_rsp` was produced by a previous
        // switch away from that thread or by `build_initial_stack`.
        unsafe { lord_context_switch(old_t.saved_rsp.as_ptr(), new_t.saved_rsp.load(Ordering::Acquire)) };
        // We are `old` again (some other thread switched back to us).
        finish_switch();
    }
    sync::irq_restore(was);
}

/// If something more important became ready and it is safe to switch here
/// (interrupts enabled, hence no kernel lock held), switch now; otherwise the
/// next timer tick does it.
pub fn preempt_check() {
    if STARTED.load(Ordering::Acquire) && sync::interrupts_enabled() && NEED_RESCHED.swap(false, Ordering::AcqRel) {
        schedule();
    }
}

/// Gives up the rest of the quantum; equal-priority peers run first.
pub fn yield_now() {
    schedule();
}

// ----------------------------------------------------------------------
// The clock
// ----------------------------------------------------------------------

/// Timer-interrupt hook (installed by [`super::start`]).
pub(super) fn on_tick(_frame: &mut InterruptFrame) {
    if !STARTED.load(Ordering::Acquire) {
        return;
    }
    let now = TICKS.fetch_add(1, Ordering::Relaxed) + 1;
    let mut resched = false;
    {
        let mut s = SCHED.lock();

        // 1. Sleeps and timed waits that expired.
        while let Some(entry) = s.sleepers.first_entry() {
            if entry.key().0 > now {
                break;
            }
            let t = entry.remove();
            // SAFETY: live thread object.
            let was_waiting_on_event = unsafe { th(thr_ptr(&t)) }.wait_event.load(Ordering::Acquire) != 0;
            if was_waiting_on_event {
                drop_waiter(&mut s, t.addr());
                wake_thread(&mut s, t, WAIT_TIMEOUT);
            } else {
                wake_thread(&mut s, t, WAIT_SIGNALED);
            }
        }

        // 2. Charge the running thread.
        let cur = s.current.as_ref().expect("no current thread");
        let is_idle = cur.same_object(s.idle.as_ref().expect("no idle thread"));
        // SAFETY: owned by the scheduler.
        let t = unsafe { th(thr_ptr(cur)) };
        t.cpu_ticks.fetch_add(1, Ordering::Relaxed);
        if is_idle {
            if s.bitmap != 0 {
                resched = true;
            }
        } else if t.quantum_left.fetch_sub(1, Ordering::Relaxed) <= 1 {
            // Quantum used up: decay any boost, let peers run.
            if t.priority() > t.base_priority() {
                t.priority.fetch_sub(1, Ordering::Relaxed);
            }
            resched = true;
        }
        if NEED_RESCHED.swap(false, Ordering::AcqRel) {
            resched = true;
            PREEMPTIONS.fetch_add(1, Ordering::Relaxed);
        }
    }
    if resched {
        schedule();
    }
}

// ----------------------------------------------------------------------
// Sleeping and waiting
// ----------------------------------------------------------------------

/// Blocks the calling thread for at least `ticks` timer ticks.
pub fn sleep_ticks(ticks: u64) {
    let ticks = ticks.max(1);
    let was = sync::irq_save();
    {
        let mut s = SCHED.lock();
        let cur = s.current.clone().expect("no current thread");
        let tp = thr_ptr(&cur);
        // SAFETY: owned by the scheduler (`current`), and by `cur` until moved below.
        let t = unsafe { th(tp) };
        let wake = TICKS.load(Ordering::Relaxed) + ticks;
        t.wake_tick.store(wake, Ordering::Release);
        t.wait_event.store(0, Ordering::Release);
        t.wait_result.store(WAIT_NONE, Ordering::Release);
        t.set_state(ThreadState::Waiting);
        let tid = t.tid;
        s.sleepers.insert((wake, tid), cur);
    }
    schedule();
    sync::irq_restore(was);
}

pub fn sleep_ms(ms: u64) {
    sleep_ticks(ms_to_ticks(ms));
}

/// Waits for `ev` to be signaled (a synchronization event is consumed), for at
/// most `timeout_ticks` ticks if given. Thread context only.
pub fn wait_event(ev: &Event, timeout_ticks: Option<u64>) -> WaitStatus {
    let was = sync::irq_save();
    let status = {
        let mut s = SCHED.lock();
        if ev.try_wait() {
            drop(s);
            sync::irq_restore(was);
            return WaitStatus::Signaled;
        }
        if timeout_ticks == Some(0) {
            drop(s);
            sync::irq_restore(was);
            return WaitStatus::TimedOut;
        }
        let cur = s.current.clone().expect("no current thread");
        let tp = thr_ptr(&cur);
        // SAFETY: owned by the scheduler.
        let t = unsafe { th(tp) };
        t.wait_result.store(WAIT_NONE, Ordering::Release);
        t.wait_event.store(ev as *const Event as usize, Ordering::Release);
        t.set_state(ThreadState::Waiting);
        let tid = t.tid;
        match timeout_ticks {
            Some(n) => {
                let wake = TICKS.load(Ordering::Relaxed) + n;
                t.wake_tick.store(wake, Ordering::Release);
                s.sleepers.insert((wake, tid), cur.clone());
            }
            None => t.wake_tick.store(0, Ordering::Release),
        }
        s.waiters.push((cur, ev as *const Event as usize));
        WAITER_COUNT.fetch_add(1, Ordering::AcqRel);
        tp
    };
    schedule();
    // SAFETY: we are this thread, running; its object is alive.
    let r = unsafe { th(status) }.wait_result.load(Ordering::Acquire);
    sync::irq_restore(was);
    if r == WAIT_TIMEOUT { WaitStatus::TimedOut } else { WaitStatus::Signaled }
}

/// Called by `Event::set`: wakes the threads blocked on `ev`.
pub fn event_signaled(ev: &Event) {
    if !STARTED.load(Ordering::Acquire) || WAITER_COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    let addr = ev as *const Event as usize;
    let was = sync::irq_save();
    {
        let mut s = SCHED.lock();
        let mut i = 0;
        while i < s.waiters.len() {
            if s.waiters[i].1 != addr {
                i += 1;
                continue;
            }
            // Notification events stay signaled; synchronization events are
            // consumed by the thread they release.
            if !ev.try_wait() {
                break;
            }
            let (t, _) = s.waiters.remove(i);
            WAITER_COUNT.fetch_sub(1, Ordering::AcqRel);
            // SAFETY: live thread object.
            let tt = unsafe { th(thr_ptr(&t)) };
            let wake = tt.wake_tick.load(Ordering::Acquire);
            if wake != 0 {
                s.sleepers.remove(&(wake, tt.tid));
            }
            wake_thread(&mut s, t, WAIT_SIGNALED);
            if ev.kind == EventKind::Synchronization {
                break;
            }
        }
    }
    sync::irq_restore(was);
    preempt_check();
}

// ----------------------------------------------------------------------
// Creating processes and threads
// ----------------------------------------------------------------------

fn new_exit_event() -> Result<Obj<Event>, PsError> {
    create_event(&ObjectAttributes::unnamed(), EventKind::Notification, false).map_err(PsError::Ob)
}

/// Makes a process object (no threads yet). `space = None` means a kernel
/// process that runs on the kernel mappings only.
pub fn create_process(
    name: &str,
    parent: Pid,
    base_priority: u8,
    space: Option<vmm::AddressSpace>,
) -> Result<Obj<Process>, PsError> {
    let root = space.as_ref().map_or(0, |s| s.root());
    let pid = NEXT_PID.fetch_add(1, Ordering::Relaxed);
    let p = Process {
        pid,
        name: name.into(),
        parent,
        created_tick: ticks(),
        base_priority,
        space: IrqLock::new(space),
        root: AtomicU64::new(root),
        handles: crate::ob::HandleTable::new(),
        active_threads: AtomicUsize::new(0),
        total_threads: AtomicUsize::new(0),
        exited: core::sync::atomic::AtomicBool::new(false),
        exit_code: core::sync::atomic::AtomicI32::new(0),
        exit_code_set: core::sync::atomic::AtomicBool::new(false),
        vm_cursor: AtomicU64::new(crate::ex::VM_BASE),
        vm_allocs: IrqLock::new(Vec::new()),
        exit_event: new_exit_event()?,
    };
    let obj = object::create(process_type(), &ObjectAttributes::unnamed(), p).map_err(PsError::Ob)?;
    PROCESSES.lock().insert(pid, obj.untyped().addr());
    Ok(obj)
}

fn new_thread_object(
    process: &Obj<Process>,
    name: &str,
    entry: fn(usize),
    arg: usize,
    prio: u8,
    user: Option<UserEntry>,
    stack: Option<vmm::KernelStack>,
    saved_rsp: u64,
    rsp0: u64,
) -> Result<Obj<Thread>, PsError> {
    let tid = NEXT_TID.fetch_add(1, Ordering::Relaxed);
    let t = Thread {
        tid,
        process: process.clone(),
        name: name.into(),
        stack: IrqLock::new(stack),
        saved_rsp: AtomicU64::new(saved_rsp),
        rsp0,
        entry,
        arg,
        user,
        state: core::sync::atomic::AtomicU8::new(ThreadState::Ready as u8),
        base_priority: core::sync::atomic::AtomicU8::new(prio),
        priority: core::sync::atomic::AtomicU8::new(prio),
        quantum_left: core::sync::atomic::AtomicU32::new(QUANTUM_TICKS),
        wake_tick: AtomicU64::new(0),
        wait_result: core::sync::atomic::AtomicU8::new(WAIT_NONE),
        wait_event: AtomicUsize::new(0),
        held: IrqLock::new(None),
        created_tick: ticks(),
        cpu_ticks: AtomicU64::new(0),
        switches_in: AtomicU64::new(0),
        exit_code: core::sync::atomic::AtomicI32::new(0),
        exit_event: new_exit_event()?,
    };
    let obj = object::create(thread_type(), &ObjectAttributes::unnamed(), t).map_err(PsError::Ob)?;
    THREADS.lock().insert(tid, obj.untyped().addr());
    Ok(obj)
}

/// Creates a thread in `process` and makes it ready. A kernel thread runs
/// `entry(arg)` and ends when it returns; a user thread (`user = Some`)
/// enters ring 3 at the given address.
pub fn create_thread(
    process: &Obj<Process>,
    name: &str,
    entry: fn(usize),
    arg: usize,
    prio: u8,
    user: Option<UserEntry>,
) -> Result<Obj<Thread>, PsError> {
    if process.has_exited() {
        return Err(PsError::ProcessExited);
    }
    if prio == priority::IDLE || prio > priority::MAX {
        return Err(PsError::InvalidArgument);
    }
    let stack = vmm::alloc_stack(KSTACK_PAGES).map_err(PsError::Vm)?;
    // SAFETY: `stack` is a fresh, mapped, zeroed kernel stack that nothing uses yet.
    let rsp = unsafe { build_initial_stack(stack.top) };
    let top = stack.top;
    let obj = new_thread_object(process, name, entry, arg, prio, user, Some(stack), rsp, top)?;
    process.active_threads.fetch_add(1, Ordering::AcqRel);
    process.total_threads.fetch_add(1, Ordering::Relaxed);
    {
        let mut s = SCHED.lock();
        make_ready(&mut s, obj.untyped().clone());
    }
    preempt_check();
    Ok(obj)
}

/// Creates the System process (pid 0) and turns the boot context into its
/// idle thread (tid 0, priority 0). Boot CPU, once, before [`start`].
pub(super) fn init_system() -> Result<(), PsError> {
    KERNEL_ROOT.store(vmm::kernel_root(), Ordering::Relaxed);
    LOADED_ROOT.store(vmm::kernel_root(), Ordering::Relaxed);
    let system = create_process("System", 0, priority::NORMAL, None)?;
    // The boot context already has a stack: record it as idle's, with the
    // TSS value in use now.
    let idle = new_thread_object(&system, "Idle", idle_entry, 0, priority::IDLE, None, None, 0, tss::kernel_stack(0))?;
    // SAFETY: just created.
    let t = unsafe { th(thr_ptr(idle.untyped())) };
    t.set_state(ThreadState::Running);
    system.active_threads.fetch_add(1, Ordering::AcqRel);
    system.total_threads.fetch_add(1, Ordering::Relaxed);
    let mut s = SCHED.lock();
    s.current = Some(idle.untyped().clone());
    s.idle = Some(idle.into_untyped());
    // The System process object stays alive through its idle thread.
    Ok(())
}

fn idle_entry(_: usize) {}

/// Turns scheduling on. The caller (the boot context, i.e. the idle thread)
/// should then enable interrupts and idle.
pub(super) fn start_scheduling() {
    STARTED.store(true, Ordering::Release);
}

// ----------------------------------------------------------------------
// Looking at the running thread
// ----------------------------------------------------------------------

/// A new reference to the calling thread.
pub fn current_thread() -> Option<ObRef> {
    SCHED.lock().current.clone()
}

/// Parks `obj` in the calling thread (see `Thread::held`) until [`release_held`]; returns
/// the object it replaced, if any, for the caller to drop outside any lock.
pub fn hold_current(obj: ObRef) -> Option<ObRef> {
    with_current(|t| t.held.lock().replace(obj))
}

/// Takes back what [`hold_current`] parked. Drop the result outside any lock.
pub fn release_held() -> Option<ObRef> {
    with_current(|t| t.held.lock().take())
}

/// The calling thread's TID.
pub fn current_tid() -> Tid {
    let s = SCHED.lock();
    // SAFETY: owned by the scheduler.
    s.current.as_ref().map_or(0, |c| unsafe { th(thr_ptr(c)) }.tid)
}

/// The calling thread's PID.
pub fn current_pid() -> Pid {
    let s = SCHED.lock();
    // SAFETY: owned by the scheduler.
    s.current.as_ref().map_or(0, |c| unsafe { th(thr_ptr(c)) }.process.pid)
}

/// A new reference to the process of the calling thread.
pub fn current_process() -> Obj<Process> {
    with_current(|t| t.process.clone())
}

/// The handle table of the calling thread's process, as a raw pointer.
///
/// Valid for as long as the calling thread lives: a thread holds a strong
/// reference to its process, and the table is a field of it. Unlike
/// [`current_process`] this keeps no reference on the stack, so a thread that
/// is killed while blocked (a blocking system call) leaks nothing.
pub fn current_handles() -> *const crate::ob::HandleTable {
    with_current(|t| &t.process.handles as *const crate::ob::HandleTable)
}

/// Runs `f` with the calling thread's user entry, if it is a user thread.
pub(super) fn with_current<R>(f: impl FnOnce(&Thread) -> R) -> R {
    let s = SCHED.lock();
    // SAFETY: owned by the scheduler.
    f(unsafe { th(thr_ptr(s.current.as_ref().expect("no current thread"))) })
}

// ----------------------------------------------------------------------
// Ending threads and processes
// ----------------------------------------------------------------------

/// Bookkeeping common to every way a thread ends: exit code, exit event,
/// process accounting. The thread must not be in any queue.
fn thread_ended(t: &Thread, code: i32) {
    t.exit_code.store(code, Ordering::Release);
    t.exit_event.set();
    let p = &*t.process;
    if p.active_threads.fetch_sub(1, Ordering::AcqRel) == 1 {
        process_ended(p, code);
    }
}

/// The last thread of a process is gone: release what the process owns.
fn process_ended(p: &Process, last_code: i32) {
    if p.exit_code_set.load(Ordering::Acquire) == false {
        p.exit_code.store(last_code, Ordering::Release);
    }
    p.exited.store(true, Ordering::Release);
    p.handles.close_all();
    let space = p.space.lock().take();
    if space.is_some() {
        // Never destroy a space that is loaded.
        if LOADED_ROOT.load(Ordering::Acquire) == p.root.load(Ordering::Relaxed) {
            switch_address_space(0);
        }
        drop(space); // page tables and every owned user frame go back to the PMM
    }
    // Shared-memory views: the address space is gone, so the sections they pinned can go.
    crate::ipc::process_ended(p.pid);
    p.exit_event.set();
}

/// Ends the calling thread with `code`. Does not return. (Do not hold object
/// references in the caller: they would never be dropped.)
pub fn exit_current(code: i32) -> ! {
    sync::irq_save();
    // SAFETY: the current thread is alive until it is a zombie and we leave its stack.
    let t = unsafe { th(with_current(|t| t as *const Thread)) };
    thread_ended(t, code);
    {
        let mut s = SCHED.lock();
        t.set_state(ThreadState::Terminated);
        let cur = s.current.clone().expect("no current thread");
        s.zombies.push(cur);
    }
    schedule();
    unreachable!("a terminated thread was scheduled again");
}

/// Ends another thread. Fails for the calling thread (use [`exit_current`]).
pub fn terminate_thread(target: &ObRef, code: i32) -> Result<(), PsError> {
    let tp = target.body::<Thread>(thread_type()).ok_or(PsError::NotAThread)? as *const Thread;
    // SAFETY: `target` is a reference to the thread.
    let t = unsafe { th(tp) };
    let mut removed: Vec<ObRef> = Vec::new();
    let was = sync::irq_save();
    {
        let mut s = SCHED.lock();
        if s.current.as_ref().is_some_and(|c| c.same_object(target)) {
            drop(s);
            sync::irq_restore(was);
            return Err(PsError::IsCurrent);
        }
        match t.state() {
            ThreadState::Terminated => {
                drop(s);
                sync::irq_restore(was);
                return Err(PsError::AlreadyTerminated);
            }
            ThreadState::Ready => removed.extend(remove_ready(&mut s, target)),
            ThreadState::Waiting => {
                let wake = t.wake_tick.load(Ordering::Acquire);
                if wake != 0 {
                    removed.extend(s.sleepers.remove(&(wake, t.tid)));
                }
                if t.wait_event.load(Ordering::Acquire) != 0 {
                    removed.extend(drop_waiter(&mut s, target.addr()));
                }
            }
            ThreadState::Running => {} // only the current thread can be running (one CPU)
        }
        t.set_state(ThreadState::Terminated);
    }
    thread_ended(t, code);
    sync::irq_restore(was);
    drop(removed); // the scheduler's references; the thread dies with the last one
    Ok(())
}

/// Ends every thread of `process` other than the caller and records `code`
/// as its exit code. If the caller belongs to the process, call
/// [`exit_current`] afterwards.
pub fn terminate_process_threads(process: &Obj<Process>, code: i32) {
    process.exit_code.store(code, Ordering::Release);
    process.exit_code_set.store(true, Ordering::Release);
    let pid = process.pid;
    let threads: Vec<ObRef> = {
        let reg = THREADS.lock();
        // SAFETY: entries are removed in `Thread::drop` before the memory is freed.
        reg.values().filter_map(|&a| unsafe { ObRef::upgrade(a) }).collect()
    };
    for t in threads {
        let tt = t.body::<Thread>(thread_type()).expect("thread");
        if tt.process.pid == pid {
            let _ = terminate_thread(&t, code); // the caller itself is skipped (IsCurrent)
        }
    }
}

/// Ends the calling thread's whole process (a fault, or `exit` from a
/// single-threaded program). Does not return.
pub fn exit_current_process(code: i32) -> ! {
    let proc = with_current(|t| t.process.clone());
    terminate_process_threads(&proc, code);
    drop(proc);
    exit_current(code)
}

/// Changes a thread's base priority (and drops any boost). Takes effect at once.
pub fn set_priority(thread: &ObRef, prio: u8) -> Result<(), PsError> {
    if prio == priority::IDLE || prio > priority::MAX {
        return Err(PsError::InvalidArgument);
    }
    let tp = thread.body::<Thread>(thread_type()).ok_or(PsError::NotAThread)? as *const Thread;
    // SAFETY: `thread` is a reference.
    let t = unsafe { th(tp) };
    let was = sync::irq_save();
    {
        let mut s = SCHED.lock();
        let requeue = if t.state() == ThreadState::Ready { remove_ready(&mut s, thread) } else { None };
        t.base_priority.store(prio, Ordering::Relaxed);
        t.priority.store(prio, Ordering::Relaxed);
        match requeue {
            Some(r) => make_ready(&mut s, r),
            None => {
                // A running thread that lowered itself may now be outranked.
                NEED_RESCHED.store(true, Ordering::Release);
            }
        }
    }
    sync::irq_restore(was);
    preempt_check();
    Ok(())
}

// ----------------------------------------------------------------------
// Inspection
// ----------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct SchedStats {
    pub ticks: u64,
    pub switches: u64,
    pub preemptions: u64,
    pub ready: usize,
    pub sleeping: usize,
    pub waiting: usize,
    pub processes: usize,
    pub threads: usize,
}

pub fn stats() -> SchedStats {
    let (ready, sleeping, waiting) = {
        let s = SCHED.lock();
        (s.ready.iter().map(|q| q.len()).sum(), s.sleepers.len(), s.waiters.len())
    };
    SchedStats {
        ticks: ticks(),
        switches: switches(),
        preemptions: preemptions(),
        ready,
        sleeping,
        waiting,
        processes: PROCESSES.lock().len(),
        threads: THREADS.lock().len(),
    }
}

pub fn thread_infos() -> Vec<ThreadInfo> {
    let refs: Vec<ObRef> = {
        let reg = THREADS.lock();
        // SAFETY: registry entries are removed before their object is freed.
        reg.values().filter_map(|&a| unsafe { ObRef::upgrade(a) }).collect()
    };
    refs.iter()
        .filter_map(|r| r.body::<Thread>(thread_type()))
        .map(|t| ThreadInfo {
            tid: t.tid,
            pid: t.process.pid,
            name: t.name.clone(),
            state: t.state(),
            priority: t.priority(),
            base_priority: t.base_priority(),
            cpu_ticks: t.cpu_ticks(),
            switches_in: t.switches_in(),
            created_tick: t.created_tick,
            user: t.is_user(),
        })
        .collect()
}

pub fn process_infos() -> Vec<ProcessInfo> {
    let refs: Vec<ObRef> = {
        let reg = PROCESSES.lock();
        // SAFETY: registry entries are removed before their object is freed.
        reg.values().filter_map(|&a| unsafe { ObRef::upgrade(a) }).collect()
    };
    refs.iter()
        .filter_map(|r| r.body::<Process>(process_type()))
        .map(|p| ProcessInfo {
            pid: p.pid,
            parent: p.parent,
            name: p.name.clone(),
            threads: p.thread_count(),
            total_threads: p.total_threads(),
            exited: p.has_exited(),
            exit_code: p.exit_code(),
            base_priority: p.base_priority,
            created_tick: p.created_tick,
            user: p.is_user(),
        })
        .collect()
}
