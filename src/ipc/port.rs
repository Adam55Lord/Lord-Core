// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! The `Port` object: a bounded, multi-producer multi-consumer message queue
//! with request/reply correlation.
//!
//! ```text
//!   sender ──push──▶ ┌─────────────────────────────┐ ──pop──▶ receiver
//!   (PORT_SEND)      │ queue (≤ capacity messages) │          (PORT_RECEIVE)
//!                    │ calls: id ─▶ Queued         │
//!                    │               Delivered(pid)│ ◀─reply─ receiver
//!   requester ◀──────│               Replied(data) │
//!                    └─────────────────────────────┘
//! ```
//!
//! # Blocking protocol
//!
//! Every wait is built from the scheduler's [`Event`]s with one rule: **the
//! state lives under the port lock, the event is only a hint**.
//!
//! * `readable` / `writable` are *notification* events (a signal wakes every
//!   waiter). Whoever changes the state signals **after** releasing the lock.
//! * Whoever observes "not satisfied" (queue empty / queue full) resets the
//!   event **under the lock** and only then blocks. A stale signal therefore
//!   costs at most one spurious wake-up, and a signal can never be lost: any
//!   state change that could satisfy the waiter happens after its reset and is
//!   followed by a signal.
//! * Each outstanding request has its own event, signaled by `reply`/`close`.
//!
//! The wait loop ([`drive`]) calls a *step* closure: `Done`, `Fail` or
//! `Block(event)`. A step must not keep an `ObRef` or a half-built message
//! alive when it returns `Block`: a thread that is killed while blocked never
//! resumes, so anything on its stack would leak. The handle-based API in
//! `ipc/mod.rs` honours that (it resolves the handle and rebuilds the message
//! on every attempt). The raw event pointer is valid because the caller's
//! handle (or reference) keeps the port alive, and closing the last handle
//! wakes every waiter before the port can be freed.
//!
//! # Locking
//!
//! One `IrqLock` per port, taken after `ob` locks and before `heap`. Messages,
//! calls and object references are never dropped while it is held (dropping
//! can destroy objects), only collected and dropped afterwards.

use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

use super::{IpcError, PORT_TYPE};
use crate::hal::sync::IrqLock;
use crate::ob::access::AccessMask;
use crate::ob::object::{create, ObRef};
use crate::ob::types::{create_event, Event, EventKind};
use crate::ob::{Obj, ObjectAttributes};
use crate::ps::sched;
use crate::ps::structs::{Pid, Tid, WaitStatus};

/// Messages a port holds when the creator does not say.
pub const DEFAULT_QUEUE: usize = 32;
/// Upper bound for a port's capacity.
pub const MAX_QUEUE: usize = 256;
/// Upper bound for one message (and one reply), in bytes.
pub const MAX_MESSAGE: usize = 4096;
/// Handles one message can carry.
pub const MAX_ATTACHMENTS: usize = 4;

// ----------------------------------------------------------------------
// Messages
// ----------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageKind {
    /// Fire and forget.
    Datagram,
    /// The sender waits for a reply with the same id.
    Request,
}

/// An object travelling inside a message, with the access its new handle gets.
pub struct Attachment {
    pub object: ObRef,
    pub access: AccessMask,
}

pub struct Message {
    /// Unique across the system; the key of a request/reply pair.
    pub id: u64,
    pub kind: MessageKind,
    pub sender_pid: Pid,
    pub sender_tid: Tid,
    pub sent_tick: u64,
    pub data: Vec<u8>,
    pub attachments: Vec<Attachment>,
}

static NEXT_MESSAGE: AtomicU64 = AtomicU64::new(1);

impl Message {
    pub fn new(kind: MessageKind, data: Vec<u8>, attachments: Vec<Attachment>) -> Self {
        Self {
            id: NEXT_MESSAGE.fetch_add(1, Ordering::Relaxed),
            kind,
            sender_pid: sched::current_pid(),
            sender_tid: sched::current_tid(),
            sent_tick: sched::ticks(),
            data,
            attachments,
        }
    }
}

/// A message on its way into a port. For a request, the port creates the
/// reply event when (and only when) the message is accepted.
pub struct Outgoing {
    pub msg: Message,
}

// ----------------------------------------------------------------------
// The wait loop
// ----------------------------------------------------------------------

/// What one attempt of a blocking operation found.
pub enum Step<T> {
    Done(T),
    /// Not possible now: wait on this event, then try again.
    Block(*const Event),
    Fail(IpcError),
}

#[derive(Clone, Copy, Debug)]
pub enum Deadline {
    /// Try once.
    Poll,
    Never,
    /// Absolute scheduler tick.
    At(u64),
}

impl Deadline {
    pub fn from_ms(timeout_ms: Option<u64>) -> Self {
        match timeout_ms {
            None => Deadline::Never,
            Some(0) => Deadline::Poll,
            Some(ms) => Deadline::At(sched::ticks().saturating_add(sched::ms_to_ticks(ms))),
        }
    }
}

/// Runs `step` until it completes, fails or the deadline passes. Thread
/// context only. Blocking needs the scheduler: before it runs, a step that
/// would block fails with `WouldBlock`.
pub fn drive<T>(deadline: Deadline, mut step: impl FnMut() -> Step<T>) -> Result<T, IpcError> {
    loop {
        match step() {
            Step::Done(v) => return Ok(v),
            Step::Fail(e) => return Err(e),
            Step::Block(ev) => {
                if matches!(deadline, Deadline::Poll) || !sched::is_started() {
                    return Err(IpcError::WouldBlock);
                }
                let remaining = match deadline {
                    Deadline::At(t) => {
                        let now = sched::ticks();
                        if now >= t {
                            return Err(IpcError::TimedOut);
                        }
                        Some(t - now)
                    }
                    _ => None,
                };
                // SAFETY: the port (hence the event) is kept alive by the caller's handle
                // or reference; closing the last handle signals this event first, so the
                // scheduler never holds the address of a freed event.
                let status = sched::wait_event(unsafe { &*ev }, remaining);
                if status == WaitStatus::TimedOut {
                    // The message may have raced the timer: one last look.
                    return match step() {
                        Step::Done(v) => Ok(v),
                        Step::Fail(e) => Err(e),
                        Step::Block(_) => Err(IpcError::TimedOut),
                    };
                }
            }
        }
    }
}

// ----------------------------------------------------------------------
// The port
// ----------------------------------------------------------------------

enum CallState {
    /// In the queue, no receiver has it yet.
    Queued,
    /// A receiver of this process took it.
    Delivered(Pid),
    Replied(Vec<u8>),
}

struct Call {
    state: CallState,
    /// Signaled by `reply` and by `close`; the requester waits on it.
    event: Obj<Event>,
}

struct PortState {
    queue: VecDeque<Message>,
    calls: BTreeMap<u64, Call>,
    closed: bool,
}

pub struct Port {
    pub capacity: usize,
    pub max_message: usize,
    state: IrqLock<PortState>,
    /// Level "the queue may hold a message".
    readable: Obj<Event>,
    /// Level "the queue (or the call table) may have room".
    writable: Obj<Event>,
    sent: AtomicU64,
    received: AtomicU64,
    requests: AtomicU64,
    replies: AtomicU64,
}

/// A snapshot for logs and `query`.
#[derive(Clone, Copy, Debug)]
pub struct PortInfo {
    pub capacity: usize,
    pub max_message: usize,
    pub queued: usize,
    /// Requests sent and not yet answered (queued or being served).
    pub calls: usize,
    pub sent: u64,
    pub received: u64,
    pub requests: u64,
    pub replies: u64,
    pub closed: bool,
}

// Machine-wide counters for the boot log.
pub(super) static TOTAL_SENT: AtomicU64 = AtomicU64::new(0);
pub(super) static TOTAL_RECEIVED: AtomicU64 = AtomicU64::new(0);
pub(super) static TOTAL_REQUESTS: AtomicU64 = AtomicU64::new(0);
pub(super) static TOTAL_REPLIES: AtomicU64 = AtomicU64::new(0);
pub(super) static TOTAL_BYTES: AtomicU64 = AtomicU64::new(0);

fn ev_ptr(e: &Obj<Event>) -> *const Event {
    &**e as *const Event
}

impl Port {
    fn new(capacity: usize, max_message: usize) -> Result<Self, IpcError> {
        let un = ObjectAttributes::unnamed();
        let readable = create_event(&un, EventKind::Notification, false)?;
        let writable = create_event(&un, EventKind::Notification, true)?;
        let mut queue = VecDeque::new();
        queue.reserve_exact(capacity);
        Ok(Self {
            capacity,
            max_message,
            state: IrqLock::new(PortState { queue, calls: BTreeMap::new(), closed: false }),
            readable,
            writable,
            sent: AtomicU64::new(0),
            received: AtomicU64::new(0),
            requests: AtomicU64::new(0),
            replies: AtomicU64::new(0),
        })
    }

    pub fn info(&self) -> PortInfo {
        let st = self.state.lock();
        PortInfo {
            capacity: self.capacity,
            max_message: self.max_message,
            queued: st.queue.len(),
            calls: st.calls.len(),
            sent: self.sent.load(Ordering::Relaxed),
            received: self.received.load(Ordering::Relaxed),
            requests: self.requests.load(Ordering::Relaxed),
            replies: self.replies.load(Ordering::Relaxed),
            closed: st.closed,
        }
    }

    pub fn is_closed(&self) -> bool {
        self.state.lock().closed
    }

    /// Checks a message against this port's limits.
    pub fn check_size(&self, len: usize) -> Result<(), IpcError> {
        if len > self.max_message.min(MAX_MESSAGE) {
            Err(IpcError::MessageTooBig)
        } else {
            Ok(())
        }
    }

    // -- sending ------------------------------------------------------------

    /// One attempt to enqueue `*slot` (taken on success, left in place when it
    /// must wait). Returns the message id.
    pub fn push_step(&self, slot: &mut Option<Outgoing>) -> Step<u64> {
        let Some(out) = slot.as_ref() else {
            return Step::Fail(IpcError::InvalidArgument);
        };
        if let Err(e) = self.check_size(out.msg.data.len()) {
            return Step::Fail(e);
        }
        let is_request = out.msg.kind == MessageKind::Request;
        // A request needs its reply event; make it before taking the lock.
        let call_event = if is_request {
            match create_event(&ObjectAttributes::unnamed(), EventKind::Notification, false) {
                Ok(e) => Some(e),
                Err(e) => return Step::Fail(e.into()),
            }
        } else {
            None
        };
        let mut st = self.state.lock();
        if st.closed {
            return Step::Fail(IpcError::Closed);
        }
        if st.queue.len() >= self.capacity || (is_request && st.calls.len() >= self.capacity) {
            self.writable.reset();
            return Step::Block(ev_ptr(&self.writable));
        }
        let out = slot.take().expect("checked above");
        let id = out.msg.id;
        let len = out.msg.data.len() as u64;
        if let Some(event) = call_event {
            st.calls.insert(id, Call { state: CallState::Queued, event });
            self.requests.fetch_add(1, Ordering::Relaxed);
            TOTAL_REQUESTS.fetch_add(1, Ordering::Relaxed);
        }
        st.queue.push_back(out.msg);
        drop(st);
        self.sent.fetch_add(1, Ordering::Relaxed);
        TOTAL_SENT.fetch_add(1, Ordering::Relaxed);
        TOTAL_BYTES.fetch_add(len, Ordering::Relaxed);
        self.readable.set();
        Step::Done(id)
    }

    // -- receiving ----------------------------------------------------------

    /// One attempt to take the oldest message. `receiver` is the calling
    /// process (the only one allowed to reply to a request it takes).
    /// A message longer than `max_len` stays queued and the attempt fails.
    pub fn receive_step(&self, receiver: Pid, max_len: usize) -> Step<Message> {
        let mut garbage: Vec<Message> = Vec::new();
        let result = {
            let mut st = self.state.lock();
            loop {
                let Some(front) = st.queue.front() else {
                    break if st.closed {
                        Step::Fail(IpcError::Closed)
                    } else {
                        self.readable.reset();
                        Step::Block(ev_ptr(&self.readable))
                    };
                };
                let (kind, id, len) = (front.kind, front.id, front.data.len());
                // A request whose sender gave up is dropped unseen.
                if kind == MessageKind::Request && !st.calls.contains_key(&id) {
                    garbage.extend(st.queue.pop_front());
                    continue;
                }
                if len > max_len {
                    break Step::Fail(IpcError::BufferTooSmall(len));
                }
                let m = st.queue.pop_front().expect("front exists");
                if kind == MessageKind::Request {
                    if let Some(c) = st.calls.get_mut(&id) {
                        c.state = CallState::Delivered(receiver);
                    }
                }
                if st.queue.is_empty() {
                    self.readable.reset();
                }
                break Step::Done(m);
            }
        };
        let freed = !garbage.is_empty() || matches!(result, Step::Done(_));
        drop(garbage);
        if let Step::Done(m) = &result {
            self.received.fetch_add(1, Ordering::Relaxed);
            TOTAL_RECEIVED.fetch_add(1, Ordering::Relaxed);
            let _ = m;
        }
        if freed {
            self.writable.set();
        }
        result
    }

    // -- replying -----------------------------------------------------------

    /// Answers request `id`, which `replier` (a process) has received.
    pub fn reply(&self, id: u64, data: &[u8], replier: Pid) -> Result<(), IpcError> {
        self.check_size(data.len())?;
        let payload = data.to_vec();
        let event = {
            let mut st = self.state.lock();
            if st.closed {
                return Err(IpcError::Closed);
            }
            let Some(call) = st.calls.get_mut(&id) else {
                return Err(IpcError::NoSuchRequest);
            };
            match call.state {
                CallState::Queued => return Err(IpcError::NotDelivered),
                CallState::Delivered(pid) if pid != replier => return Err(IpcError::NotYourRequest),
                CallState::Replied(_) => return Err(IpcError::AlreadyReplied),
                CallState::Delivered(_) => {}
            }
            call.state = CallState::Replied(payload);
            call.event.clone()
        };
        self.replies.fetch_add(1, Ordering::Relaxed);
        TOTAL_REPLIES.fetch_add(1, Ordering::Relaxed);
        event.set();
        Ok(())
    }

    /// One attempt to collect the reply to request `id`. A reply longer than
    /// `max_reply` is discarded and reported.
    pub fn call_step(&self, id: u64, max_reply: usize) -> Step<Vec<u8>> {
        let mut st = self.state.lock();
        let replied = match st.calls.get(&id) {
            None => {
                return Step::Fail(if st.closed { IpcError::Closed } else { IpcError::Cancelled });
            }
            Some(c) => matches!(c.state, CallState::Replied(_)),
        };
        if replied || st.closed {
            let call = st.calls.remove(&id).expect("looked up above");
            drop(st);
            self.writable.set();
            return match call.state {
                CallState::Replied(data) if data.len() > max_reply => Step::Fail(IpcError::BufferTooSmall(data.len())),
                CallState::Replied(data) => Step::Done(data),
                _ => Step::Fail(IpcError::Closed),
            };
        }
        let call = st.calls.get(&id).expect("looked up above");
        call.event.reset();
        Step::Block(ev_ptr(&call.event))
    }

    /// The requester gives up (timeout, kill, error): forget the call and take
    /// the request back out of the queue if nobody has received it yet.
    pub fn abandon(&self, id: u64) {
        let (call, msg) = {
            let mut st = self.state.lock();
            let call = st.calls.remove(&id);
            let pos = st.queue.iter().position(|m| m.id == id);
            let msg = pos.and_then(|i| st.queue.remove(i));
            (call, msg)
        };
        drop((call, msg));
        self.writable.set();
    }

    // -- closing ------------------------------------------------------------

    /// Disconnects the port: queued messages are dropped (and with them any
    /// attached objects), every blocked sender, receiver and requester wakes
    /// with `Closed`, new traffic is refused. Idempotent. Runs automatically
    /// when the last handle closes.
    pub fn close(&self) {
        let (messages, events) = {
            let mut st = self.state.lock();
            if st.closed {
                return;
            }
            st.closed = true;
            let messages: Vec<Message> = st.queue.drain(..).collect();
            let events: Vec<Obj<Event>> = st.calls.values().map(|c| c.event.clone()).collect();
            (messages, events)
        };
        self.readable.set();
        self.writable.set();
        for e in &events {
            e.set();
        }
        drop((messages, events));
    }
}

/// `last_close` hook of the `Port` type.
pub(super) fn port_last_close(body: *const u8) {
    // SAFETY: the Object Manager passes the body of a live `Port` object.
    unsafe { &*(body as *const Port) }.close();
}

/// Creates a port. `capacity == 0` means [`DEFAULT_QUEUE`].
pub fn create_port(attrs: &ObjectAttributes<'_>, capacity: usize) -> Result<Obj<Port>, IpcError> {
    let capacity = if capacity == 0 { DEFAULT_QUEUE } else { capacity };
    if capacity > MAX_QUEUE {
        return Err(IpcError::InvalidArgument);
    }
    let port = Port::new(capacity, MAX_MESSAGE)?;
    let ty = PORT_TYPE.load(Ordering::Acquire);
    if ty.is_null() {
        return Err(IpcError::NotReady);
    }
    // SAFETY: set once from an immortal type object.
    Ok(create(unsafe { &*ty }, attrs, port)?)
}
