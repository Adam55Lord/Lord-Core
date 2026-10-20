// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Boot-time self-test of the Process Manager. It runs in the "boot-driver"
//! kernel thread, i.e. under the real scheduler, with real timer interrupts.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::sched;
use super::structs::*;
use super::user::{self, user_programs};
use super::*;
use crate::mm::{pmm, vmm};
use crate::ob::access::*;
use crate::ob::kernel_handles;
use crate::ob::types::{create_event, EventKind};
use crate::ob::ObjectAttributes;

macro_rules! ensure {
    ($c:expr, $m:expr) => {
        if !$c {
            return Err($m);
        }
    };
}

macro_rules! tr {
    ($e:expr, $m:expr) => {
        $e.map_err(|_| $m)?
    };
}

/// What [`run_self_test`] observed, for the boot log.
#[derive(Clone, Copy, Default)]
pub struct SelfTest {
    pub fair_ticks: (u64, u64),
    pub fair_switches: u64,
    pub wake_latency_ticks: u64,
    pub sleep_ticks: u64,
    pub user_syscalls: u64,
    pub stress_threads: u32,
    pub user_processes: u32,
}

fn balance() -> u64 {
    pmm::stats().in_use_frames - vmm::stats().committed_pages
}

fn join(t: &Obj<Thread>) -> Result<(), &'static str> {
    match wait_for_object(t.untyped(), Some(5000)) {
        Ok(WaitStatus::Signaled) => Ok(()),
        _ => Err("a thread did not finish in time"),
    }
}

fn join_process(p: &Obj<Process>) -> Result<(), &'static str> {
    match wait_for_object(p.untyped(), Some(5000)) {
        Ok(WaitStatus::Signaled) => Ok(()),
        _ => Err("a process did not finish in time"),
    }
}

/// Lets terminated threads be released (the scheduler drops them after the next switch).
fn settle() {
    sleep_ms(30);
}

// -- 1: fairness ----------------------------------------------------------

static STOP: AtomicBool = AtomicBool::new(false);
static CNT: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

fn spinner(which: usize) {
    while !STOP.load(Ordering::Relaxed) {
        CNT[which].fetch_add(1, Ordering::Relaxed);
    }
}

// -- 2: priority and wake latency -----------------------------------------

static STOP2: AtomicBool = AtomicBool::new(false);
static LOW_COUNT: AtomicU64 = AtomicU64::new(0);
static LATENCY: AtomicU64 = AtomicU64::new(u64::MAX);
static LOW_AT_WAKE: AtomicU64 = AtomicU64::new(0);

fn low_spinner(_: usize) {
    while !STOP2.load(Ordering::Relaxed) {
        LOW_COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

fn high_sleeper(_: usize) {
    let t0 = ticks();
    sleep_ticks(3);
    let t1 = ticks();
    LATENCY.store(t1.saturating_sub(t0 + 3), Ordering::SeqCst);
    LOW_AT_WAKE.store(LOW_COUNT.load(Ordering::Relaxed), Ordering::SeqCst);
}

// -- 2b: signaling an event preempts the signaler at once -------------------

static HIGH_RAN: AtomicBool = AtomicBool::new(false);
static RAN_BEFORE_SET_RETURNED: AtomicBool = AtomicBool::new(false);

fn high_waiter(arg: usize) {
    // SAFETY: `arg` came from `Arc::into_raw` in the test below.
    let ev = unsafe { Arc::from_raw(arg as *const Obj<crate::ob::types::Event>) };
    if sched::wait_event(&ev, Some(sched::ms_to_ticks(2000))) == WaitStatus::Signaled {
        HIGH_RAN.store(true, Ordering::SeqCst);
    }
}

fn low_signaler(arg: usize) {
    // SAFETY: as above (a second reference for this thread).
    let ev = unsafe { Arc::from_raw(arg as *const Obj<crate::ob::types::Event>) };
    sleep_ms(20); // let the high-priority thread block first
    ev.set();
    RAN_BEFORE_SET_RETURNED.store(HIGH_RAN.load(Ordering::SeqCst), Ordering::SeqCst);
}

// -- 4: events ------------------------------------------------------------

static WOKEN: AtomicU64 = AtomicU64::new(0);

fn event_waiter(arg: usize) {
    // SAFETY: `arg` came from `Arc::into_raw` in the test below, one per waiter.
    let ev = unsafe { Arc::from_raw(arg as *const Obj<crate::ob::types::Event>) };
    if wait_event_forever(&ev) {
        WOKEN.fetch_add(1, Ordering::SeqCst);
    }
}

fn wait_event_forever(ev: &crate::ob::types::Event) -> bool {
    sched::wait_event(ev, Some(sched::ms_to_ticks(2000))) == WaitStatus::Signaled
}

fn sleeper_forever(_: usize) {
    sleep_ms(60_000);
}

fn exits_with_7(_: usize) {
    sleep_ms(20);
    exit_current(7);
}

fn short_lived(_: usize) {
    for _ in 0..3 {
        yield_now();
    }
}

pub fn run_self_test() -> Result<SelfTest, &'static str> {
    let mut out = SelfTest::default();
    settle();
    let bal0 = balance();
    let stacks0 = vmm::stats().regions;
    let threads0 = thread_type().total_objects.load(Ordering::Relaxed);
    let procs0 = process_type().total_objects.load(Ordering::Relaxed);

    // 1) Two CPU-bound threads that never yield share the CPU: only the timer can do that.
    {
        STOP.store(false, Ordering::SeqCst);
        let a = tr!(spawn_kernel_thread("spin-a", spinner, 0, priority::NORMAL), "spawn a");
        let b = tr!(spawn_kernel_thread("spin-b", spinner, 1, priority::NORMAL), "spawn b");
        let sw0 = sched::switches();
        sleep_ms(300);
        STOP.store(true, Ordering::SeqCst);
        join(&a)?;
        join(&b)?;
        let (ta, tb) = (a.cpu_ticks(), b.cpu_ticks());
        ensure!(CNT[0].load(Ordering::Relaxed) > 0 && CNT[1].load(Ordering::Relaxed) > 0, "a spinner never ran");
        ensure!(ta >= 5 && tb >= 5, "a spinner got almost no CPU time");
        ensure!(ta.abs_diff(tb) <= (ta + tb) / 4 + 2, "round robin is not fair");
        out.fair_ticks = (ta, tb);
        out.fair_switches = sched::switches() - sw0;
        ensure!(out.fair_switches >= 10, "too few context switches for two spinners");
    }

    // 2) A high-priority thread that wakes up preempts a low-priority spinner at once.
    {
        STOP2.store(false, Ordering::SeqCst);
        let low = tr!(spawn_kernel_thread("low", low_spinner, 0, priority::LOW), "spawn low");
        sleep_ms(30);
        let high = tr!(spawn_kernel_thread("high", high_sleeper, 0, priority::HIGH), "spawn high");
        join(&high)?;
        STOP2.store(true, Ordering::SeqCst);
        join(&low)?;
        out.wake_latency_ticks = LATENCY.load(Ordering::SeqCst);
        ensure!(out.wake_latency_ticks <= 1, "a woken high-priority thread waited for the low-priority spinner");
        ensure!(LOW_AT_WAKE.load(Ordering::SeqCst) > 0, "the low-priority thread never ran");
    }

    // 2b) Signaling an event that a higher-priority thread waits on switches to it
    //     before `set` returns (the signaler is preempted mid-quantum).
    {
        let ev = Arc::new(tr!(create_event(&ObjectAttributes::unnamed(), EventKind::Notification, false), "event"));
        HIGH_RAN.store(false, Ordering::SeqCst);
        RAN_BEFORE_SET_RETURNED.store(false, Ordering::SeqCst);
        let h = tr!(spawn_kernel_thread("hi-wait", high_waiter, Arc::into_raw(ev.clone()) as usize, priority::HIGH), "spawn");
        let l = tr!(spawn_kernel_thread("lo-signal", low_signaler, Arc::into_raw(ev.clone()) as usize, priority::LOW), "spawn");
        join(&h)?;
        join(&l)?;
        ensure!(RAN_BEFORE_SET_RETURNED.load(Ordering::SeqCst), "the higher-priority waiter did not preempt the signaler");
    }

    // 2c) Lowering a running thread's priority takes effect immediately: the
    //     other (still NORMAL) spinner gets everything, the lowered one nothing.
    {
        STOP.store(false, Ordering::SeqCst);
        CNT[0].store(0, Ordering::SeqCst);
        CNT[1].store(0, Ordering::SeqCst);
        let a = tr!(spawn_kernel_thread("prio-a", spinner, 0, priority::NORMAL), "spawn");
        let b = tr!(spawn_kernel_thread("prio-b", spinner, 1, priority::NORMAL), "spawn");
        sleep_ms(40);
        tr!(set_priority(a.untyped(), priority::LOW), "set priority");
        ensure!(a.base_priority() == priority::LOW && a.priority() == priority::LOW, "priority not changed");
        let (a0, b0) = (a.cpu_ticks(), b.cpu_ticks());
        sleep_ms(200);
        let (da, db) = (a.cpu_ticks() - a0, b.cpu_ticks() - b0);
        STOP.store(true, Ordering::SeqCst);
        join(&a)?;
        join(&b)?;
        ensure!(da <= 1 && db >= 15, "a lowered thread kept running");
        ensure!(set_priority(a.untyped(), 0) == Err(PsError::InvalidArgument), "idle priority accepted");
        ensure!(current_tid() != 0, "the driver thinks it is the idle thread");
    }

    // 3) Sleep accuracy.
    {
        let t0 = ticks();
        sleep_ms(50);
        out.sleep_ticks = ticks() - t0;
        ensure!((5..=8).contains(&out.sleep_ticks), "sleep(50 ms) took the wrong number of ticks");
    }

    // 4) Events: synchronization releases one waiter per signal, notification all of them; timeouts.
    {
        let sync_ev = Arc::new(tr!(create_event(&ObjectAttributes::unnamed(), EventKind::Synchronization, false), "sync event"));
        WOKEN.store(0, Ordering::SeqCst);
        let mut waiters = Vec::new();
        for _ in 0..3 {
            let a = Arc::into_raw(sync_ev.clone()) as usize;
            waiters.push(tr!(spawn_kernel_thread("sync-waiter", event_waiter, a, priority::NORMAL), "spawn waiter"));
        }
        sleep_ms(30);
        ensure!(sched::stats().waiting == 3, "waiters did not block");
        sync_ev.set();
        sleep_ms(30);
        ensure!(WOKEN.load(Ordering::SeqCst) == 1, "a synchronization event released more than one waiter");
        sync_ev.set();
        sync_ev.set();
        sleep_ms(30);
        ensure!(WOKEN.load(Ordering::SeqCst) == 3, "synchronization event lost a signal");
        for w in &waiters {
            join(w)?;
        }

        let note_ev = Arc::new(tr!(create_event(&ObjectAttributes::unnamed(), EventKind::Notification, false), "notification event"));
        WOKEN.store(0, Ordering::SeqCst);
        let mut waiters = Vec::new();
        for _ in 0..3 {
            let a = Arc::into_raw(note_ev.clone()) as usize;
            waiters.push(tr!(spawn_kernel_thread("note-waiter", event_waiter, a, priority::NORMAL), "spawn waiter"));
        }
        sleep_ms(30);
        note_ev.set();
        for w in &waiters {
            join(w)?;
        }
        ensure!(WOKEN.load(Ordering::SeqCst) == 3, "a notification event did not release every waiter");
        ensure!(note_ev.is_signaled(), "a notification event was consumed");

        let never = tr!(create_event(&ObjectAttributes::unnamed(), EventKind::Notification, false), "never event");
        let t0 = ticks();
        ensure!(wait_for_object(never.untyped(), Some(30)) == Ok(WaitStatus::TimedOut), "wait did not time out");
        ensure!(ticks() - t0 >= 3, "wait timed out early");
        ensure!(wait_for_object(never.untyped(), Some(0)) == Ok(WaitStatus::TimedOut), "poll of an unsignaled event");
        never.set();
        ensure!(wait_for_object(never.untyped(), Some(0)) == Ok(WaitStatus::Signaled), "poll of a signaled event");
    }

    // 5) Waiting through handles needs SYNCHRONIZE; thread exit codes.
    {
        let t = tr!(spawn_kernel_thread("exit7", exits_with_7, 0, priority::NORMAL), "spawn");
        let kh = kernel_handles();
        let weak = tr!(kh.create_handle(t.untyped(), THREAD_QUERY, 0), "query handle");
        let strong = tr!(kh.create_handle(t.untyped(), THREAD_QUERY | SYNCHRONIZE, 0), "sync handle");
        ensure!(wait_for_handle(kh, weak, Some(1000)).is_err(), "waited through a handle without SYNCHRONIZE");
        ensure!(wait_for_handle(kh, strong, Some(1000)) == Ok(WaitStatus::Signaled), "wait through a handle");
        ensure!(t.exit_code() == 7 && t.state() == ThreadState::Terminated, "thread exit code / state");
        tr!(kh.close(weak), "close");
        tr!(kh.close(strong), "close");
    }

    // 6) Terminating threads in every state; the caller cannot terminate itself this way.
    {
        STOP.store(false, Ordering::SeqCst);
        let ready = tr!(spawn_kernel_thread("victim-ready", spinner, 0, priority::LOW), "spawn");
        let sleeping = tr!(spawn_kernel_thread("victim-sleep", sleeper_forever, 0, priority::NORMAL), "spawn");
        let ev = tr!(create_event(&ObjectAttributes::unnamed(), EventKind::Notification, false), "event");
        let a = Arc::into_raw(Arc::new(ev)) as usize;
        let waiting = tr!(spawn_kernel_thread("victim-wait", event_waiter, a, priority::NORMAL), "spawn");
        sleep_ms(30);
        ensure!(ready.state() == ThreadState::Ready, "victim not ready");
        ensure!(sleeping.state() == ThreadState::Waiting && waiting.state() == ThreadState::Waiting, "victims not waiting");
        for v in [&ready, &sleeping, &waiting] {
            tr!(terminate_thread(v.untyped(), 99), "terminate");
            ensure!(v.state() == ThreadState::Terminated && v.exit_code() == 99, "terminated thread state");
            ensure!(wait_for_object(v.untyped(), Some(0)) == Ok(WaitStatus::Signaled), "exit event not signaled");
            ensure!(terminate_thread(v.untyped(), 1) == Err(PsError::AlreadyTerminated), "double terminate");
        }
        let me = tr!(current_thread().ok_or(()), "no current thread");
        ensure!(terminate_thread(&me, 1) == Err(PsError::IsCurrent), "terminated itself");
        drop(me);
        let st = sched::stats();
        ensure!(st.sleeping == 0 && st.waiting == 0, "terminated threads left queue entries");
        STOP.store(true, Ordering::SeqCst);
    }

    // 7) A user process: own address space, ring 3, system calls, exit code, full cleanup.
    let progs = user_programs();
    let by = |n: &str| progs.iter().find(|p| p.name == n).copied();
    {
        let hello = tr!(by("hello").ok_or(()), "no hello program");
        let w0 = user::user_write_count();
        let s0 = user::syscall_count();
        let p = tr!(spawn_user_process("hello", hello.image, priority::NORMAL), "spawn user process");
        join_process(&p)?;
        ensure!(p.has_exited() && p.exit_code() == 42, "user process exit code");
        ensure!(user::user_write_count() - w0 == 3, "user process did not write 3 lines");
        out.user_syscalls = user::syscall_count() - s0;
        ensure!(out.user_syscalls == 7, "unexpected number of system calls (3 writes + 3 yields + exit)");
        ensure!(p.space.lock().is_none() && p.thread_count() == 0, "address space not released at exit");
        out.user_processes += 1;
    }

    // 8) Faults kill the process, not the kernel; processes are isolated and preemptible in ring 3.
    {
        let fault = tr!(by("fault").ok_or(()), "no fault program");
        let pf = tr!(spawn_user_process("fault", fault.image, priority::NORMAL), "spawn");
        join_process(&pf)?;
        ensure!(pf.exit_code() == 0xC000_000Eu32 as i32, "user page fault did not end the process with its code");
        out.user_processes += 1;

        let spin = tr!(by("spin").ok_or(()), "no spin program");
        let pid_prog = tr!(by("pid").ok_or(()), "no pid program");
        let spinner = tr!(spawn_user_process("spin", spin.image, priority::NORMAL), "spawn");
        let mut procs = Vec::new();
        for _ in 0..3 {
            procs.push(tr!(spawn_user_process("pid", pid_prog.image, priority::NORMAL), "spawn"));
        }
        let mut roots = Vec::new();
        for p in &procs {
            join_process(p)?;
            ensure!(p.exit_code() as u64 == p.pid, "a process lost its register across a context switch");
            roots.push(p.root.load(Ordering::Relaxed));
        }
        roots.push(spinner.root.load(Ordering::Relaxed));
        roots.sort_unstable();
        roots.dedup();
        ensure!(roots.len() == 4, "processes share an address space");
        ensure!(!spinner.has_exited(), "the spinner ended by itself");
        terminate_process(&spinner, -5);
        join_process(&spinner)?;
        ensure!(spinner.exit_code() == -5 && spinner.space.lock().is_none(), "terminated process state");
        out.user_processes += 4;
    }

    // 9) Many short-lived threads: nothing leaks.
    {
        let mut n = 0;
        for _ in 0..3 {
            let mut batch = Vec::new();
            for _ in 0..50 {
                batch.push(tr!(spawn_kernel_thread("short", short_lived, 0, priority::NORMAL), "spawn"));
            }
            for t in &batch {
                join(t)?;
                n += 1;
            }
        }
        out.stress_threads = n;
    }

    // 10) Everything we created is gone again.
    settle();
    ensure!(thread_type().total_objects.load(Ordering::Relaxed) == threads0, "thread objects leaked");
    ensure!(process_type().total_objects.load(Ordering::Relaxed) == procs0, "process objects leaked");
    ensure!(vmm::stats().regions == stacks0, "kernel stacks leaked");
    ensure!(balance() == bal0, "frames leaked");
    ensure!(vmm::verify().is_ok(), "VMM inconsistent after the run");
    Ok(out)
}
