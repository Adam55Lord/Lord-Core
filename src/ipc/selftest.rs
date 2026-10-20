// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Boot-time self-test of the IPC Manager. It runs in the "boot-driver" kernel
//! thread, under the live scheduler, after the Process Manager's own test:
//! blocking needs real threads and real timer interrupts.
//!
//! Besides kernel threads it launches **ring-3 programs** (assembled into the
//! kernel image, like the Process Manager's): a server and a client that talk
//! through a named port, share a page of memory through a section handle sent
//! in a message, a program that is killed while blocked in `receive`, and one
//! that feeds every system call invalid arguments.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use super::port::MAX_ATTACHMENTS;
use super::section::{self, VIEW_BASE};
use super::*;
use crate::hal::memory::phys_to_virt;
use crate::hal::sync::IrqLock;
use crate::mm::{pmm, vmm};
use crate::ob::namespace;
use crate::ob::types::{create_event, event_type, EventKind};
use crate::ob::{query_event, set_event, Obj};
use crate::ps::{self, priority, ThreadState, WaitStatus};

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

/// What [`self_test`] observed, for the boot log.
#[derive(Clone, Copy, Default)]
pub struct SelfTest {
    pub checks: u32,
    pub messages: u64,
    pub bytes: u64,
    pub requests: u64,
    pub views: u64,
    pub user_syscalls: u64,
    pub stress_messages: u64,
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

fn balance() -> u64 {
    pmm::stats().in_use_frames - vmm::stats().committed_pages
}

fn settle() {
    ps::sleep_ms(30);
}

fn join(t: &Obj<ps::Thread>) -> Result<(), &'static str> {
    match ps::wait_for_object(t.untyped(), Some(5000)) {
        Ok(WaitStatus::Signaled) => Ok(()),
        _ => Err("a thread did not finish in time"),
    }
}

fn join_process(p: &Obj<ps::Process>) -> Result<(), &'static str> {
    match ps::wait_for_object(p.untyped(), Some(5000)) {
        Ok(WaitStatus::Signaled) => Ok(()),
        _ => Err("a process did not finish in time"),
    }
}

fn run_closure<F: FnOnce() + Send + 'static>(arg: usize) {
    // SAFETY: `arg` is the `Box<F>` leaked by `spawn_fn` for exactly this thread.
    let f = unsafe { Box::from_raw(arg as *mut F) };
    (*f)();
}

/// A kernel thread running a closure.
fn spawn_fn<F: FnOnce() + Send + 'static>(name: &str, prio: u8, f: F) -> Result<Obj<ps::Thread>, &'static str> {
    let raw = Box::into_raw(Box::new(f)) as usize;
    match ps::spawn_kernel_thread(name, run_closure::<F>, raw, prio) {
        Ok(t) => Ok(t),
        Err(_) => {
            // SAFETY: the thread was not created, so nobody else owns the box.
            drop(unsafe { Box::from_raw(raw as *mut F) });
            Err("cannot spawn a kernel thread")
        }
    }
}

type Slot<T> = Arc<IrqLock<Option<T>>>;

fn slot<T>() -> Slot<T> {
    Arc::new(IrqLock::new(None))
}

fn put<T>(s: &Slot<T>, v: T) {
    *s.lock() = Some(v);
}

fn take<T>(s: &Slot<T>) -> Option<T> {
    s.lock().take()
}

fn port_info(t: &HandleTable, h: Handle) -> Result<PortInfo, &'static str> {
    Ok(tr!(query(t, h), "query"))
}

fn new_port(t: &HandleTable, cap: usize) -> Result<Handle, &'static str> {
    Ok(tr!(create_port_handle(t, &ObjectAttributes::unnamed(), cap, PORT_ALL), "create port"))
}

fn with_port<R>(t: &HandleTable, h: Handle, f: impl FnOnce(&Port) -> R) -> Result<R, &'static str> {
    let (o, _) = tr!(t.reference_by_handle(h, 0, Some(port_type())), "resolve port");
    Ok(f(o.body::<Port>(port_type()).ok_or("not a port")?))
}

// ----------------------------------------------------------------------
// 1) Queue basics, no blocking
// ----------------------------------------------------------------------

fn test_basics() -> Result<(), &'static str> {
    let t = HandleTable::new();
    let h = new_port(&t, 4)?;
    let mut ids = Vec::new();
    for i in 0..4u8 {
        ids.push(tr!(send(&t, h, &[i; 3], &[], Some(0)), "send into an empty port"));
    }
    ensure!(ids.windows(2).all(|w| w[0] < w[1]), "message ids are not increasing");
    ensure!(send(&t, h, b"x", &[], Some(0)) == Err(IpcError::WouldBlock), "a full port accepted a message");
    ensure!(send(&t, h, &vec![0u8; MAX_MESSAGE + 1], &[], Some(0)) == Err(IpcError::MessageTooBig), "oversized message accepted");
    let i = port_info(&t, h)?;
    ensure!(i.queued == 4 && i.sent == 4 && i.capacity == 4, "port counters after 4 sends");
    for want in 0..4u8 {
        let r = tr!(receive(&t, h, 16, Some(0)), "receive");
        ensure!(r.message.data == [want; 3], "messages out of order");
        ensure!(r.message.kind == MessageKind::Datagram && r.handles.is_empty(), "datagram shape");
        ensure!(r.message.sender_pid == ps::current_pid() && r.message.sender_tid == ps::current_tid(), "sender identity");
    }
    ensure!(receive(&t, h, 16, Some(0)).err() == Some(IpcError::WouldBlock), "empty port did not report WouldBlock");
    let t0 = ps::ticks();
    ensure!(receive(&t, h, 16, Some(30)).err() == Some(IpcError::TimedOut), "empty receive did not time out");
    let waited = ps::ticks() - t0;
    ensure!((2..=20).contains(&waited), "timeout took the wrong time");
    // A message that does not fit the buffer stays queued.
    tr!(send(&t, h, &[7u8; 10], &[], Some(0)), "send 10 bytes");
    ensure!(receive(&t, h, 4, Some(0)).err() == Some(IpcError::BufferTooSmall(10)), "small buffer not reported");
    ensure!(port_info(&t, h)?.queued == 1, "message lost on BufferTooSmall");
    ensure!(tr!(receive(&t, h, 16, Some(0)), "receive after retry").message.data.len() == 10, "retry got the wrong message");
    tr!(t.close(h), "close port handle");
    Ok(())
}

// ----------------------------------------------------------------------
// 2) Names, rights, types
// ----------------------------------------------------------------------

fn test_names_and_rights() -> Result<(), &'static str> {
    let t = HandleTable::new();
    let name = "\\Ports\\SelfTest";
    let ports0 = port_type().total_objects.load(Ordering::Relaxed);
    let hp = tr!(create_port_handle(&t, &ObjectAttributes::named(name), 8, PORT_ALL), "create named port");
    ensure!(create_port(&ObjectAttributes::named("\\ports\\SELFTEST"), 8).err() == Some(IpcError::NameInUse), "name collision (case-insensitive)");
    let hs = tr!(open_port(&t, name, PORT_SEND), "open send-only");
    let hr = tr!(open_port(&t, name, PORT_RECEIVE), "open receive-only");
    let hw = tr!(open_port(&t, name, GENERIC_WRITE), "open with GENERIC_WRITE");
    ensure!(t.info(hw).map(|i| i.access) == Ok(READ_CONTROL | PORT_SEND), "generic write did not map to PORT_SEND");
    ensure!(send(&t, hr, b"x", &[], Some(0)) == Err(IpcError::AccessDenied), "receive-only handle may send");
    ensure!(receive(&t, hs, 8, Some(0)).err() == Some(IpcError::AccessDenied), "send-only handle may receive");
    ensure!(query(&t, hs).err() == Some(IpcError::AccessDenied), "handle without PORT_QUERY may query");
    ensure!(reply(&t, hs, 1, b"x") == Err(IpcError::AccessDenied), "send-only handle may reply");
    tr!(send(&t, hs, b"hello", &[], Some(0)), "send through the send-only handle");
    ensure!(tr!(receive(&t, hr, 8, Some(0)), "receive through the receive-only handle").message.data == b"hello", "payload");
    // Rights can only shrink.
    let hd = tr!(duplicate_reduced(&t, hp, PORT_SEND), "reduced duplicate");
    ensure!(duplicate_reduced(&t, hs, PORT_ALL).err() == Some(IpcError::AccessDenied), "a duplicate gained rights");
    ensure!(send(&t, hd, b"y", &[], Some(0)).is_ok(), "reduced duplicate cannot send");
    // Wrong type, bad handles.
    let ev = tr!(create_event(&ObjectAttributes::unnamed(), EventKind::Notification, false), "event");
    let he = tr!(t.create_handle(ev.untyped(), EVENT_ALL, 0), "event handle");
    ensure!(send(&t, he, b"x", &[], Some(0)) == Err(IpcError::TypeMismatch), "an event was used as a port");
    for bad in [Handle(0), Handle(2), Handle(4 * 9999)] {
        ensure!(send(&t, bad, b"x", &[], Some(0)) == Err(IpcError::InvalidHandle), "bad handle accepted");
    }
    ensure!(port_info(&t, hp)?.queued == 1, "queue length after the rights checks");
    // The name goes with the last handle.
    for h in [hp, hs, hr, hw, hd, he] {
        tr!(t.close(h), "close");
    }
    drop(ev);
    ensure!(namespace::lookup(name, None, true).err() == Some(ObError::NameNotFound), "port name survived its last handle");
    ensure!(port_type().total_objects.load(Ordering::Relaxed) == ports0, "port leaked");
    Ok(())
}

// ----------------------------------------------------------------------
// 3) Blocking: receivers, senders, timeouts, several waiters
// ----------------------------------------------------------------------

fn test_blocking() -> Result<(), &'static str> {
    let t = Arc::new(HandleTable::new());
    let h = new_port(&t, 2)?;

    // a) a blocked receiver wakes as soon as a message arrives
    let res: Slot<(Result<Vec<u8>, IpcError>, u64)> = slot();
    let (t2, r2) = (t.clone(), res.clone());
    let rx = spawn_fn("ipc-rx", priority::HIGH, move || {
        let r = receive(&t2, h, 64, Some(3000)).map(|x| x.message.data);
        put(&r2, (r, ps::ticks()));
    })?;
    ps::sleep_ms(40);
    ensure!(rx.state() == ThreadState::Waiting, "the receiver did not block");
    let sent_at = ps::ticks();
    tr!(send(&t, h, b"wake", &[], None), "send to a blocked receiver");
    join(&rx)?;
    let (r, woke_at) = take(&res).ok_or("receiver left no result")?;
    ensure!(r == Ok(b"wake".to_vec()), "blocked receiver got the wrong message");
    ensure!(woke_at - sent_at <= 1, "blocked receiver woke late");

    // b) a blocked sender continues when room appears
    tr!(send(&t, h, b"1", &[], Some(0)), "fill 1");
    tr!(send(&t, h, b"2", &[], Some(0)), "fill 2");
    let sres: Slot<Result<u64, IpcError>> = slot();
    let (t2, s2) = (t.clone(), sres.clone());
    let tx = spawn_fn("ipc-tx", priority::HIGH, move || put(&s2, send(&t2, h, b"3", &[], Some(3000))))?;
    ps::sleep_ms(40);
    ensure!(tx.state() == ThreadState::Waiting, "the sender did not block on a full port");
    ensure!(tr!(receive(&t, h, 8, Some(0)), "receive 1").message.data == b"1", "first message");
    join(&tx)?;
    ensure!(matches!(take(&sres), Some(Ok(_))), "blocked sender failed");
    ensure!(tr!(receive(&t, h, 8, Some(0)), "receive 2").message.data == b"2", "second message");
    ensure!(tr!(receive(&t, h, 8, Some(0)), "receive 3").message.data == b"3", "third message (the one that was blocked)");

    // c) a send into a full port times out
    tr!(send(&t, h, b"a", &[], Some(0)), "fill a");
    tr!(send(&t, h, b"b", &[], Some(0)), "fill b");
    let t0 = ps::ticks();
    ensure!(send(&t, h, b"c", &[], Some(40)) == Err(IpcError::TimedOut), "full port did not time out");
    ensure!(ps::ticks() - t0 >= 3, "send timeout was too short");
    for _ in 0..2 {
        tr!(receive(&t, h, 8, Some(0)), "drain");
    }

    // d) several blocked receivers: each message goes to exactly one of them
    let got = Arc::new(IrqLock::new(Vec::<u8>::new()));
    let mut rxs = Vec::new();
    for _ in 0..3 {
        let (t2, g) = (t.clone(), got.clone());
        rxs.push(spawn_fn("ipc-rx", priority::HIGH, move || {
            if let Ok(r) = receive(&t2, h, 8, Some(3000)) {
                g.lock().push(r.message.data[0]);
            }
        })?);
    }
    ps::sleep_ms(40);
    ensure!(rxs.iter().all(|r| r.state() == ThreadState::Waiting), "not every receiver blocked");
    for v in [10u8, 11, 12] {
        tr!(send(&t, h, &[v], &[], Some(1000)), "send to three receivers");
    }
    for r in &rxs {
        join(r)?;
    }
    let mut g = got.lock().clone();
    g.sort_unstable();
    ensure!(g == [10, 11, 12], "a message was lost or delivered twice");
    ensure!(port_info(&t, h)?.queued == 0, "messages left in the queue");
    tr!(t.close(h), "close");
    Ok(())
}

// ----------------------------------------------------------------------
// 4) Request / reply
// ----------------------------------------------------------------------

fn echo_server(t: Arc<HandleTable>, h: Handle, served: Arc<AtomicUsize>) {
    loop {
        match receive(&t, h, 256, Some(3000)) {
            Ok(r) if r.message.kind == MessageKind::Request => {
                let d: Vec<u8> = r.message.data.iter().map(|b| b.wrapping_add(1)).collect();
                if reply(&t, h, r.message.id, &d).is_ok() {
                    served.fetch_add(1, Ordering::Relaxed);
                }
            }
            _ => break, // a datagram is the stop signal; an error ends the server too
        }
    }
}

fn test_request_reply() -> Result<(), &'static str> {
    let t = Arc::new(HandleTable::new());
    let h = new_port(&t, 8)?;
    let served = Arc::new(AtomicUsize::new(0));

    // a) four clients, one server
    let (t2, s2) = (t.clone(), served.clone());
    let server = spawn_fn("ipc-server", priority::HIGH, move || echo_server(t2, h, s2))?;
    let ok = Arc::new(AtomicUsize::new(0));
    let mut clients = Vec::new();
    for c in 0..4u8 {
        let hc = tr!(duplicate_reduced(&t, h, PORT_SEND), "client handle");
        let (t2, ok) = (t.clone(), ok.clone());
        clients.push(spawn_fn("ipc-client", priority::NORMAL, move || {
            for i in 0..25u8 {
                let msg = [c, i, 0xFE];
                if request(&t2, hc, &msg, 64, Some(3000)).as_deref() == Ok(&[c.wrapping_add(1), i.wrapping_add(1), 0xFF][..]) {
                    ok.fetch_add(1, Ordering::Relaxed);
                }
            }
        })?);
    }
    for c in &clients {
        join(c)?;
    }
    ensure!(ok.load(Ordering::Relaxed) == 100, "a request got a wrong or no reply");
    tr!(send(&t, h, b"stop", &[], Some(1000)), "stop the server");
    join(&server)?;
    ensure!(served.load(Ordering::Relaxed) == 100, "the server did not serve every request");
    let i = port_info(&t, h)?;
    ensure!(i.requests == 100 && i.replies == 100 && i.calls == 0 && i.queued == 0, "request/reply counters or leftovers");

    // b) the protocol, step by step (the requester runs below us, so we can reply twice before it looks)
    let res: Slot<Result<Vec<u8>, IpcError>> = slot();
    let (t2, r2) = (t.clone(), res.clone());
    let hc = tr!(duplicate_reduced(&t, h, PORT_SEND), "client handle");
    let rq = spawn_fn("ipc-req", priority::LOW, move || put(&r2, request(&t2, hc, b"ping", 16, Some(3000))))?;
    ps::sleep_ms(40);
    let got = tr!(receive(&t, h, 16, Some(1000)), "receive the request");
    ensure!(got.message.kind == MessageKind::Request && got.message.data == b"ping", "request shape");
    let id = got.message.id;
    ensure!(with_port(&t, h, |p| p.reply(id, b"x", 9999))? == Err(IpcError::NotYourRequest), "another process may answer");
    ensure!(with_port(&t, h, |p| p.reply(id + 1000, b"x", ps::current_pid()))? == Err(IpcError::NoSuchRequest), "reply to an unknown id");
    ensure!(reply(&t, h, id, &vec![0u8; MAX_MESSAGE + 1]) == Err(IpcError::MessageTooBig), "oversized reply");
    tr!(reply(&t, h, id, b"pong"), "reply");
    ensure!(reply(&t, h, id, b"again") == Err(IpcError::AlreadyReplied), "double reply");
    join(&rq)?;
    ensure!(take(&res) == Some(Ok(b"pong".to_vec())), "requester got the wrong reply");
    ensure!(reply(&t, h, id, b"late") == Err(IpcError::NoSuchRequest), "reply after the requester collected");

    // c) a reply that does not fit is discarded and reported
    let (t2, r2) = (t.clone(), res.clone());
    let rq = spawn_fn("ipc-req", priority::HIGH, move || put(&r2, request(&t2, hc, b"big", 4, Some(3000))))?;
    let got = tr!(receive(&t, h, 16, Some(1000)), "receive");
    tr!(reply(&t, h, got.message.id, b"0123456789"), "reply");
    join(&rq)?;
    ensure!(take(&res) == Some(Err(IpcError::BufferTooSmall(10))), "oversized reply not reported");

    // d) timeouts withdraw the request
    let t0 = ps::ticks();
    ensure!(request(&t, hc, b"anyone?", 16, Some(40)) == Err(IpcError::TimedOut), "request without a server did not time out");
    ensure!(ps::ticks() - t0 >= 3, "request timeout was too short");
    let i = port_info(&t, h)?;
    ensure!(i.queued == 0 && i.calls == 0, "a timed-out request stayed behind");
    // ...also after a server took it: the late reply finds nobody.
    let (t2, r2) = (t.clone(), res.clone());
    let rq = spawn_fn("ipc-req", priority::HIGH, move || put(&r2, request(&t2, hc, b"slow", 16, Some(50))))?;
    let got = tr!(receive(&t, h, 16, Some(1000)), "receive");
    join(&rq)?;
    ensure!(take(&res) == Some(Err(IpcError::TimedOut)), "requester should have given up");
    ensure!(reply(&t, h, got.message.id, b"too late") == Err(IpcError::NoSuchRequest), "reply to a withdrawn request");
    ensure!(port_info(&t, h)?.calls == 0, "withdrawn call left behind");
    tr!(t.close(hc), "close client handle");
    tr!(t.close(h), "close");
    Ok(())
}

// ----------------------------------------------------------------------
// 5) Disconnecting wakes everybody
// ----------------------------------------------------------------------

fn test_close() -> Result<(), &'static str> {
    let t = Arc::new(HandleTable::new());

    // a) explicit disconnect (a kernel server shutting down): blocked threads see `Closed`
    let (rx_port, tx_port, rq_port) = (new_port(&t, 4)?, new_port(&t, 1)?, new_port(&t, 4)?);
    tr!(send(&t, tx_port, b"full", &[], Some(0)), "fill");
    let rx_res: Slot<IpcError> = slot();
    let tx_res: Slot<IpcError> = slot();
    let rq_res: Slot<IpcError> = slot();
    let mut threads = Vec::new();
    {
        let (t2, r) = (t.clone(), rx_res.clone());
        threads.push(spawn_fn("ipc-rx", priority::HIGH, move || {
            if let Err(e) = receive(&t2, rx_port, 8, Some(3000)) {
                put(&r, e);
            }
        })?);
        let (t2, r) = (t.clone(), tx_res.clone());
        threads.push(spawn_fn("ipc-tx", priority::HIGH, move || {
            if let Err(e) = send(&t2, tx_port, b"more", &[], Some(3000)) {
                put(&r, e);
            }
        })?);
        let (t2, r) = (t.clone(), rq_res.clone());
        threads.push(spawn_fn("ipc-rq", priority::HIGH, move || {
            if let Err(e) = request(&t2, rq_port, b"hello?", 8, Some(3000)) {
                put(&r, e);
            }
        })?);
    }
    ps::sleep_ms(40);
    ensure!(threads.iter().all(|th| th.state() == ThreadState::Waiting), "not every thread blocked");
    let t0 = ps::ticks();
    for p in [rx_port, tx_port, rq_port] {
        with_port(&t, p, |p| p.close())?;
    }
    for th in &threads {
        join(th)?;
    }
    ensure!(ps::ticks() - t0 <= 3, "blocked threads woke late");
    ensure!(take(&rx_res) == Some(IpcError::Closed), "blocked receiver was not told the port closed");
    ensure!(take(&tx_res) == Some(IpcError::Closed), "blocked sender was not told the port closed");
    ensure!(take(&rq_res) == Some(IpcError::Closed), "blocked requester was not told the port closed");
    ensure!(send(&t, rx_port, b"x", &[], Some(0)) == Err(IpcError::Closed), "send to a closed port");
    ensure!(receive(&t, rx_port, 8, Some(0)).err() == Some(IpcError::Closed), "receive from a closed port");
    let i = port_info(&t, tx_port)?;
    ensure!(i.closed && i.queued == 0, "closed port kept its queue");
    for p in [rx_port, tx_port, rq_port] {
        tr!(t.close(p), "close");
    }

    // b) closing the last handle: the waiter's own handle is gone, it must still wake
    let h = new_port(&t, 4)?;
    let res: Slot<IpcError> = slot();
    let (t2, r) = (t.clone(), res.clone());
    let rx = spawn_fn("ipc-rx", priority::HIGH, move || {
        if let Err(e) = receive(&t2, h, 8, Some(3000)) {
            put(&r, e);
        }
    })?;
    ps::sleep_ms(40);
    ensure!(rx.state() == ThreadState::Waiting, "receiver did not block");
    let t0 = ps::ticks();
    tr!(t.close(h), "close last handle");
    join(&rx)?;
    ensure!(ps::ticks() - t0 <= 3, "receiver woke late after the last close");
    ensure!(matches!(take(&res), Some(IpcError::InvalidHandle | IpcError::Closed)), "receiver was not woken with an error");
    Ok(())
}

// ----------------------------------------------------------------------
// 6) Handles travel in messages
// ----------------------------------------------------------------------

fn test_attachments() -> Result<(), &'static str> {
    let ev_objs0 = event_type().total_objects.load(Ordering::Relaxed);
    let ports0 = port_type().total_objects.load(Ordering::Relaxed);
    let (t1, t2) = (HandleTable::new(), HandleTable::new());
    let h1 = new_port(&t1, 4)?;
    let h2 = tr!(t1.duplicate(h1, &t2, DuplicateOptions::same_access()), "give the receiver a handle");

    let ev = tr!(create_event(&ObjectAttributes::unnamed(), EventKind::Notification, false), "event");
    let rights = EVENT_QUERY_STATE | EVENT_MODIFY_STATE | READ_CONTROL;
    let e1 = tr!(t1.create_handle(ev.untyped(), rights, 0), "event handle");
    drop(ev);
    tr!(send(&t1, h1, b"here", &[e1], Some(0)), "send with an attachment");
    let r = tr!(receive(&t2, h2, 16, Some(0)), "receive with an attachment");
    ensure!(r.handles.len() == 1 && r.message.data == b"here", "attachment missing");
    let e2 = r.handles[0];
    let info = tr!(t2.info(e2), "attached handle info");
    ensure!(info.type_name == "Event" && info.access == rights, "attached handle lost its rights");
    ensure!(info.handle_count == 2, "handle count of the passed object");
    ensure!(query_event(&t2, e2) == Ok(false), "attached event state");
    tr!(set_event(&t1, e1), "set through the sender's handle");
    ensure!(query_event(&t2, e2) == Ok(true), "the two handles do not name the same event");

    // limits and errors
    ensure!(send(&t1, h1, b"x", &[e1; MAX_ATTACHMENTS + 1], Some(0)) == Err(IpcError::TooManyAttachments), "too many attachments");
    ensure!(send(&t1, h1, b"x", &[Handle(4 * 777)], Some(0)) == Err(IpcError::InvalidHandle), "bad handle attached");
    ensure!(port_info(&t1, h1)?.queued == 0, "a failed send left a message");

    // an attachment that is never received must not leak: closing the port drops the message
    tr!(send(&t1, h1, b"lost", &[e1], Some(0)), "send again");
    // ...and a port attached to itself is not a reference cycle once it is closed
    tr!(send(&t1, h1, b"self", &[h1], Some(0)), "send the port to itself");
    tr!(t1.close(e1), "close e1");
    tr!(t2.close(e2), "close e2");
    ensure!(event_type().total_objects.load(Ordering::Relaxed) > ev_objs0, "the queued message does not hold the event");
    tr!(t2.close(h2), "close h2");
    tr!(t1.close(h1), "close h1"); // last handle: the port disconnects and drops its queue
    ensure!(event_type().total_objects.load(Ordering::Relaxed) == ev_objs0, "an attached event leaked");
    ensure!(port_type().total_objects.load(Ordering::Relaxed) == ports0, "a port attached to itself leaked");
    Ok(())
}

// ----------------------------------------------------------------------
// 7) Sections: shared memory
// ----------------------------------------------------------------------

fn test_sections(out: &mut SelfTest) -> Result<(), &'static str> {
    let sec0 = section_type().total_objects.load(Ordering::Relaxed);
    let frames0 = pmm::stats().in_use_frames;

    let s = tr!(create_section(&ObjectAttributes::named("\\Sections\\SelfTest"), 3), "create section");
    ensure!(s.pages() == 3 && s.size() == 3 * 4096, "section size");
    ensure!(pmm::stats().in_use_frames == frames0 + 3, "section frames not taken from the PMM");
    ensure!(create_section(&ObjectAttributes::named("\\Sections\\selftest"), 1).err() == Some(IpcError::NameInUse), "section name collision");
    ensure!(create_section(&ObjectAttributes::unnamed(), 0).err() == Some(IpcError::InvalidArgument), "empty section");
    ensure!(create_section(&ObjectAttributes::unnamed(), MAX_SECTION_PAGES + 1).err() == Some(IpcError::InvalidArgument), "huge section");
    // read/write across a page boundary, zero-filled start, bounds
    let mut z = [1u8; 8];
    tr!(s.read(4090, &mut z), "read");
    ensure!(z == [0u8; 8], "a new section is not zeroed");
    tr!(s.write(4090, b"ABCDEFGH"), "write across a page");
    let mut b = [0u8; 8];
    tr!(s.read(4090, &mut b), "read back");
    ensure!(&b == b"ABCDEFGH", "section read-back");
    ensure!(s.write(3 * 4096 - 2, b"xyz") == Err(IpcError::InvalidArgument), "write out of bounds");
    ensure!(s.read(u64::MAX - 1, &mut b) == Err(IpcError::InvalidArgument), "read with an overflowing offset");

    // two processes (no threads needed: only their address spaces matter) map the same frames
    let mk = |name: &str| -> Result<Obj<ps::Process>, &'static str> {
        let space = tr!(vmm::AddressSpace::new(), "address space");
        Ok(tr!(ps::create_process(name, 0, priority::NORMAL, Some(space)), "process"))
    };
    let (pa, pb) = (mk("view-a")?, mk("view-b")?);
    let va = tr!(map_view(&pa, s.untyped(), 0, true), "map view a");
    let vb = tr!(map_view(&pb, s.untyped(), 0, false), "map view b");
    ensure!(va == VIEW_BASE && vb == VIEW_BASE, "first view not at the default base");
    let second = tr!(map_view(&pa, s.untyped(), 0, false), "second view in a");
    ensure!(second == VIEW_BASE + 4 * 4096, "views are not separated by a gap page");
    for page in 0..3u64 {
        let a = pa.with_space(|sp| sp.translate(va + page * 4096)).flatten();
        let bb = pb.with_space(|sp| sp.translate(vb + page * 4096)).flatten();
        ensure!(a.is_some() && a == bb, "the two views do not show the same frame");
    }
    ensure!(pa.with_space(|sp| sp.translate(va + 3 * 4096)).flatten().is_none(), "view is longer than the section");
    let phys = tr!(pa.with_space(|sp| sp.translate(va + 4090)).flatten().ok_or(()), "translate");
    // SAFETY: `phys` is inside one of the section's frames, reachable through the HHDM; 4090 + 6 <= 4096.
    let seen = unsafe { core::slice::from_raw_parts(phys_to_virt(phys), 6) };
    ensure!(seen == b"ABCDEF", "the view does not show what the kernel wrote");
    ensure!(section::view_count(pa.pid) == 2 && section::view_count(pb.pid) == 1, "view registry");
    // chosen addresses
    ensure!(map_view(&pb, s.untyped(), 0x2000_0001, true).err() == Some(IpcError::InvalidArgument), "unaligned address accepted");
    ensure!(map_view(&pb, s.untyped(), vmm::KERNEL_HALF, true).err() == Some(IpcError::InvalidArgument), "kernel address accepted");
    ensure!(map_view(&pb, s.untyped(), vb, true).err() == Some(IpcError::InvalidArgument), "overlapping view accepted");
    ensure!(section::view_count(pb.pid) == 1, "a failed map left a view");
    let vc = tr!(map_view(&pb, s.untyped(), 0x2000_0000, true), "map at a chosen address");
    ensure!(vc == 0x2000_0000, "chosen address ignored");
    tr!(unmap_view(&pb, vc), "unmap chosen");
    ensure!(unmap_view(&pb, 0x123_4000).err() == Some(IpcError::NotFound), "unmap of nothing");
    let ev = tr!(create_event(&ObjectAttributes::unnamed(), EventKind::Notification, false), "event");
    ensure!(map_view(&pa, ev.untyped(), 0, true).err() == Some(IpcError::TypeMismatch), "an event was mapped");
    drop(ev);
    out.views += 5;

    // The views keep the frames alive after the last reference to the section is gone.
    let ht = HandleTable::new();
    let hs = tr!(ht.create_handle(s.untyped(), SECTION_ALL, 0), "section handle");
    drop(s);
    tr!(ht.close(hs), "close the last handle"); // the name goes away, the views remain
    ensure!(namespace::lookup("\\Sections\\SelfTest", None, true).err() == Some(ObError::NameNotFound), "section name survived its last handle");
    ensure!(section_type().total_objects.load(Ordering::Relaxed) == sec0 + 1, "views do not pin the section");
    tr!(unmap_view(&pa, va), "unmap a");
    tr!(unmap_view(&pa, second), "unmap a2");
    ensure!(section_type().total_objects.load(Ordering::Relaxed) == sec0 + 1, "view b should still pin the section");
    // A process that ends releases its views: simulate the Process Manager's callback.
    section::process_ended(pb.pid);
    ensure!(section_type().total_objects.load(Ordering::Relaxed) == sec0, "ending the process did not release the section");
    ensure!(section::view_count(pb.pid) == 0 && section::view_count(pa.pid) == 0, "views left over");
    drop((pa, pb));
    ensure!(pmm::stats().in_use_frames <= frames0 + 1, "section frames or page tables leaked");
    Ok(())
}

// ----------------------------------------------------------------------
// 8) Ring 3
// ----------------------------------------------------------------------

core::arch::global_asm!(
    r#"
    .section .rodata.lord_user, "a"

    # ---- server: creates \Ports\echo, answers one request, then receives a section handle and reads it ----
    .global lord_ipc_server_start
    .global lord_ipc_server_end
lord_ipc_server_start:
    sub rsp, 512                        # [rsp] 256-byte buffer, [rsp+256] 72-byte info block
    mov eax, 8                          # port_create("echo", 4, capacity 8)
    lea rdi, [rip + lord_ipc_server_name]
    mov esi, 4
    mov edx, 8
    int 0x80
    test rax, rax
    js 90f
    mov r12, rax                        # r12 = port
    mov eax, 12                         # receive(port, buf, 256, info, 5000 ms)
    mov rdi, r12
    mov rsi, rsp
    mov edx, 256
    lea r10, [rsp + 256]
    mov r8d, 5000
    int 0x80
    test rax, rax
    js 91f
    mov r13, rax                        # r13 = length
    inc byte ptr [rsp]                  # the service: first byte + 1
    mov eax, 14                         # reply(port, info.id, buf, length)
    mov rdi, r12
    mov rsi, [rsp + 256]
    mov rdx, rsp
    mov r10, r13
    int 0x80
    test rax, rax
    js 92f
    mov eax, 12                         # receive #2: a datagram that carries a section handle
    mov rdi, r12
    mov rsi, rsp
    mov edx, 256
    lea r10, [rsp + 256]
    mov r8d, 5000
    int 0x80
    test rax, rax
    js 93f
    cmp qword ptr [rsp + 256 + 32], 1   # info.nhandles
    jne 94f
    mov rbx, [rsp + 256 + 40]           # info.handles[0]
    mov eax, 16                         # section_map(handle, anywhere, read-only)
    mov rdi, rbx
    xor esi, esi
    xor edx, edx
    int 0x80
    test rax, rax
    js 95f
    movzx edi, byte ptr [rax]           # exit(first byte of the shared page) = what the client wrote
    jmp 99f
90: mov edi, 100
    jmp 99f
91: mov edi, 101
    jmp 99f
92: mov edi, 102
    jmp 99f
93: mov edi, 103
    jmp 99f
94: mov edi, 104
    jmp 99f
95: mov edi, 105
99: xor eax, eax
    int 0x80
    ud2
lord_ipc_server_name:
    .ascii "echo"
lord_ipc_server_end:

    # ---- client: opens \Ports\echo (send-only), sends a request, creates + fills a section, sends it ----
    .global lord_ipc_client_start
    .global lord_ipc_client_end
lord_ipc_client_start:
    sub rsp, 512
    mov r14d, 500                       # the server may not be up yet: retry, yielding
1:  mov eax, 9                          # port_open("echo", PORT_SEND)
    lea rdi, [rip + lord_ipc_client_name]
    mov esi, 4
    mov edx, 1
    int 0x80
    test rax, rax
    jns 2f
    mov eax, 1
    int 0x80
    dec r14d
    jnz 1b
    mov edi, 100
    jmp 99f
2:  mov r12, rax
    mov dword ptr [rsp], 0x00434241     # "ABC"
    mov eax, 13                         # request(port, buf, 3, reply buf, 64, 5000 ms)
    mov rdi, r12
    mov rsi, rsp
    mov edx, 3
    lea r10, [rsp + 64]
    mov r8d, 64
    mov r9d, 5000
    int 0x80
    cmp rax, 3
    jne 91f
    cmp byte ptr [rsp + 64], 0x42       # 'B' = 'A' + 1
    jne 92f
    mov eax, 15                         # section_create(unnamed, 1 page)
    xor edi, edi
    xor esi, esi
    mov edx, 1
    int 0x80
    test rax, rax
    js 93f
    mov r13, rax
    mov eax, 16                         # section_map(section, anywhere, writable)
    mov rdi, r13
    xor esi, esi
    mov edx, 1
    int 0x80
    test rax, rax
    js 94f
    mov byte ptr [rax], 0x5A            # the value the server must see through its own view
    mov [rsp + 128], r13                # array of one handle
    mov eax, 11                         # send(port, buf, 3, handles, 1, 5000 ms)
    mov rdi, r12
    mov rsi, rsp
    mov edx, 3
    lea r10, [rsp + 128]
    mov r8d, 1
    mov r9d, 5000
    int 0x80
    test rax, rax
    js 95f
    xor edi, edi
    jmp 99f
91: mov edi, 101
    jmp 99f
92: mov edi, 102
    jmp 99f
93: mov edi, 103
    jmp 99f
94: mov edi, 104
    jmp 99f
95: mov edi, 105
99: xor eax, eax
    int 0x80
    ud2
lord_ipc_client_name:
    .ascii "echo"
lord_ipc_client_end:

    # ---- block: creates \Ports\Blocker and waits for a message forever ----
    .global lord_ipc_block_start
    .global lord_ipc_block_end
lord_ipc_block_start:
    sub rsp, 128
    mov eax, 8
    lea rdi, [rip + lord_ipc_block_name]
    mov esi, 7
    mov edx, 4
    int 0x80
    test rax, rax
    js 90f
    mov r12, rax
    mov eax, 12                         # receive(port, buf, 64, no info, forever)
    mov rdi, r12
    mov rsi, rsp
    mov edx, 64
    xor r10d, r10d
    mov r8, -1
    int 0x80
    mov edi, 1                          # only reached if something woke it
    jmp 99f
90: mov edi, 2
99: xor eax, eax
    int 0x80
    ud2
lord_ipc_block_name:
    .ascii "Blocker"
lord_ipc_block_end:

    # ---- bad: every kind of invalid argument must fail cleanly ----
    .global lord_ipc_bad_start
    .global lord_ipc_bad_end
lord_ipc_bad_start:
    sub rsp, 128
    mov eax, 8                          # port_create(unnamed)
    xor edi, edi
    xor esi, esi
    mov edx, 4
    int 0x80
    test rax, rax
    js 90f
    mov r12, rax
    mov eax, 12                         # receive into an unmapped buffer: BadAddress (-20)
    mov rdi, r12
    mov esi, 0x10
    mov edx, 16
    xor r10d, r10d
    xor r8d, r8d
    int 0x80
    cmp rax, -20
    jne 91f
    mov eax, 11                         # send from a kernel address: BadAddress
    mov rdi, r12
    movabs rsi, 0xffff800000000000
    mov edx, 4
    xor r10d, r10d
    xor r8d, r8d
    xor r9d, r9d
    int 0x80
    cmp rax, -20
    jne 92f
    mov eax, 9                          # open a port that does not exist: NotFound (-5)
    lea rdi, [rip + lord_ipc_bad_name]
    mov esi, 7
    mov edx, 1
    int 0x80
    cmp rax, -5
    jne 93f
    mov eax, 10                         # close a misaligned handle: InvalidHandle (-2)
    mov edi, 0x7777
    int 0x80
    cmp rax, -2
    jne 94f
    mov eax, 9                          # a name with a backslash: InvalidName (-7)
    lea rdi, [rip + lord_ipc_bad_slash]
    mov esi, 3
    mov edx, 1
    int 0x80
    cmp rax, -7
    jne 95f
    mov eax, 12                         # poll an empty port: WouldBlock (-10)
    mov rdi, r12
    mov rsi, rsp
    mov edx, 16
    xor r10d, r10d
    xor r8d, r8d
    int 0x80
    cmp rax, -10
    jne 96f
    mov eax, 12                         # receive into the read-only code page: the kernel must refuse (-20)
    mov rdi, r12
    mov esi, 0x400000
    mov edx, 16
    xor r10d, r10d
    xor r8d, r8d
    int 0x80
    cmp rax, -20
    jne 97f
    mov eax, 16                         # map a port as if it were a section: TypeMismatch (-3)
    mov rdi, r12
    xor esi, esi
    xor edx, edx
    int 0x80
    cmp rax, -3
    jne 98f
    xor edi, edi
    jmp 99f
90: mov edi, 90
    jmp 99f
91: mov edi, 91
    jmp 99f
92: mov edi, 92
    jmp 99f
93: mov edi, 93
    jmp 99f
94: mov edi, 94
    jmp 99f
95: mov edi, 95
    jmp 99f
96: mov edi, 96
    jmp 99f
97: mov edi, 97
    jmp 99f
98: mov edi, 98
99: xor eax, eax
    int 0x80
    ud2
lord_ipc_bad_name:
    .ascii "Missing"
lord_ipc_bad_slash:
    .ascii "a\\b"
lord_ipc_bad_end:
"#
);

unsafe extern "C" {
    static lord_ipc_server_start: u8;
    static lord_ipc_server_end: u8;
    static lord_ipc_client_start: u8;
    static lord_ipc_client_end: u8;
    static lord_ipc_block_start: u8;
    static lord_ipc_block_end: u8;
    static lord_ipc_bad_start: u8;
    static lord_ipc_bad_end: u8;
}

fn image(start: *const u8, end: *const u8) -> &'static [u8] {
    // SAFETY: both symbols delimit a read-only blob in the kernel image.
    unsafe { core::slice::from_raw_parts(start, end as usize - start as usize) }
}

fn spawn(name: &str, img: &'static [u8]) -> Result<Obj<ps::Process>, &'static str> {
    Ok(tr!(ps::spawn_user_process(name, img, priority::NORMAL), "spawn user process"))
}

fn test_user_mode(out: &mut SelfTest) -> Result<(), &'static str> {
    let sec0 = section_type().total_objects.load(Ordering::Relaxed);
    let ports0 = port_type().total_objects.load(Ordering::Relaxed);
    let calls0 = syscall::count();
    let views0 = section::VIEWS_MAPPED.load(Ordering::Relaxed);

    // a) client and server talk, and share a page through a handle sent in a message
    let server = spawn("ipc-server", image(&raw const lord_ipc_server_start, &raw const lord_ipc_server_end))?;
    let client = spawn("ipc-client", image(&raw const lord_ipc_client_start, &raw const lord_ipc_client_end))?;
    join_process(&client)?;
    join_process(&server)?;
    ensure!(client.exit_code() == 0, "ring-3 client failed (its exit code is the failing step)");
    ensure!(server.exit_code() == 0x5A, "ring-3 server did not see the client's byte through the shared section");
    ensure!(syscall::count() - calls0 >= 8, "too few IPC system calls");
    ensure!(section::VIEWS_MAPPED.load(Ordering::Relaxed) - views0 == 2, "expected one view in each process");
    drop((client, server));
    settle();
    ensure!(section_type().total_objects.load(Ordering::Relaxed) == sec0, "the section outlived both processes");
    ensure!(port_type().total_objects.load(Ordering::Relaxed) == ports0, "the port outlived both processes");
    ensure!(namespace::lookup("\\Ports\\echo", None, true).err() == Some(ObError::NameNotFound), "the port name outlived its processes");

    // b) invalid arguments fail cleanly, nothing crashes
    let bad = spawn("ipc-bad", image(&raw const lord_ipc_bad_start, &raw const lord_ipc_bad_end))?;
    join_process(&bad)?;
    ensure!(bad.exit_code() == 0, "a bad system call was not rejected as expected (exit code = failing step)");
    drop(bad);

    // c) a process blocked in receive is killed: it must leave nothing behind
    let blocker = spawn("ipc-block", image(&raw const lord_ipc_block_start, &raw const lord_ipc_block_end))?;
    let mut tries = 0;
    while namespace::lookup("\\Ports\\Blocker", None, true).is_err() {
        ps::sleep_ms(10);
        tries += 1;
        ensure!(tries < 300, "the blocking program never created its port");
    }
    ps::sleep_ms(50);
    let pid = blocker.pid;
    ensure!(ps::thread_infos().iter().any(|t| t.pid == pid && t.state == ThreadState::Waiting), "the program is not blocked");
    ps::terminate_process(&blocker, 7);
    join_process(&blocker)?;
    ensure!(blocker.exit_code() == 7, "exit code of the killed process");
    drop(blocker);
    settle();
    ensure!(namespace::lookup("\\Ports\\Blocker", None, true).err() == Some(ObError::NameNotFound), "the killed process's port survived");
    ensure!(port_type().total_objects.load(Ordering::Relaxed) == ports0, "killing a blocked process leaked its port");

    out.user_syscalls = syscall::count() - calls0;
    out.views += section::VIEWS_MAPPED.load(Ordering::Relaxed) - views0;
    Ok(())
}

// ----------------------------------------------------------------------
// 9) Stress: many producers, many consumers, one small port
// ----------------------------------------------------------------------

fn test_stress(out: &mut SelfTest) -> Result<(), &'static str> {
    const PRODUCERS: usize = 4;
    const PER: u16 = 200;
    let t = Arc::new(HandleTable::new());
    let h = new_port(&t, 8)?;
    let sums: Arc<[AtomicU64; PRODUCERS]> = Arc::new(core::array::from_fn(|_| AtomicU64::new(0)));
    let counts: Arc<[AtomicU64; PRODUCERS]> = Arc::new(core::array::from_fn(|_| AtomicU64::new(0)));
    let mut consumers = Vec::new();
    for _ in 0..2 {
        let (t2, s, c) = (t.clone(), sums.clone(), counts.clone());
        consumers.push(spawn_fn("ipc-cons", priority::NORMAL, move || {
            while let Ok(r) = receive(&t2, h, 8, Some(5000)) {
                if r.message.data.len() != 3 {
                    break; // the 1-byte poison message
                }
                let (p, seq) = (r.message.data[0] as usize, u16::from_le_bytes([r.message.data[1], r.message.data[2]]));
                s[p].fetch_add(seq as u64, Ordering::Relaxed);
                c[p].fetch_add(1, Ordering::Relaxed);
            }
        })?);
    }
    let mut producers = Vec::new();
    for p in 0..PRODUCERS as u8 {
        let hs = tr!(duplicate_reduced(&t, h, PORT_SEND), "producer handle");
        let t2 = t.clone();
        producers.push(spawn_fn("ipc-prod", priority::NORMAL, move || {
            for seq in 0..PER {
                let b = seq.to_le_bytes();
                let _ = send(&t2, hs, &[p, b[0], b[1]], &[], Some(5000));
            }
        })?);
    }
    for p in &producers {
        join(p)?;
    }
    for _ in 0..2 {
        tr!(send(&t, h, b"!", &[], Some(5000)), "poison");
    }
    for c in &consumers {
        join(c)?;
    }
    let want_sum = PER as u64 * (PER as u64 - 1) / 2;
    for p in 0..PRODUCERS {
        ensure!(counts[p].load(Ordering::Relaxed) == PER as u64, "a message was lost or duplicated");
        ensure!(sums[p].load(Ordering::Relaxed) == want_sum, "a message arrived damaged");
    }
    let i = port_info(&t, h)?;
    ensure!(i.queued == 0 && i.sent == i.received && i.sent == PRODUCERS as u64 * PER as u64 + 2, "stress counters");
    out.stress_messages = i.sent;
    Ok(())
}

// ----------------------------------------------------------------------
// The whole test
// ----------------------------------------------------------------------

pub fn self_test() -> Result<SelfTest, &'static str> {
    let mut out = SelfTest::default();
    settle();
    let bal0 = balance();
    let ports0 = port_type().total_objects.load(Ordering::Relaxed);
    let sections0 = section_type().total_objects.load(Ordering::Relaxed);
    let events0 = event_type().total_objects.load(Ordering::Relaxed);
    let threads0 = ps::thread_type().total_objects.load(Ordering::Relaxed);
    let procs0 = ps::process_type().total_objects.load(Ordering::Relaxed);
    let s0 = stats();

    test_basics()?;
    test_names_and_rights()?;
    test_blocking()?;
    test_request_reply()?;
    test_close()?;
    test_attachments()?;
    test_sections(&mut out)?;
    test_user_mode(&mut out)?;
    test_stress(&mut out)?;
    out.checks = 9;

    settle();
    let s1 = stats();
    out.messages = s1.sent - s0.sent;
    out.bytes = s1.bytes - s0.bytes;
    out.requests = s1.requests - s0.requests;
    ensure!(s1.received - s0.received <= out.messages, "more messages received than sent");
    ensure!(s1.replies - s0.replies <= out.requests, "more replies than requests");
    ensure!(port_type().total_objects.load(Ordering::Relaxed) == ports0, "ports leaked");
    ensure!(section_type().total_objects.load(Ordering::Relaxed) == sections0, "sections leaked");
    ensure!(event_type().total_objects.load(Ordering::Relaxed) == events0, "events leaked");
    ensure!(ps::thread_type().total_objects.load(Ordering::Relaxed) == threads0, "threads leaked");
    ensure!(ps::process_type().total_objects.load(Ordering::Relaxed) == procs0, "processes leaked");
    ensure!(balance() == bal0, "frames leaked");
    Ok(out)
}
