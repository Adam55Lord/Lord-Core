// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! The **IPC Manager**: how processes (and kernel threads) talk to each other.
//!
//! It is an Executive component built entirely on the Object Manager and the
//! Process Manager: nothing in it is special-cased by the rest of the kernel.
//!
//! ```text
//!   process A                              process B
//!   ┌──────────────┐                       ┌──────────────┐
//!   │ HandleTable  │                       │ HandleTable  │
//!   │  h ─▶ Port   │──── \Ports\Name ─────▶│  h ─▶ Port   │   message passing
//!   │  h ─▶ Section│═════ same frames ════▶│  h ─▶ Section│   shared memory
//!   └──────────────┘                       └──────────────┘
//! ```
//!
//! * **Ports** ([`port`]): bounded message queues with datagrams, blocking
//!   `send`/`receive` with timeouts, synchronous **request/reply**, and
//!   **handle passing** (a message can carry handles to other objects: that is
//!   how a process hands another a section, an event or a port).
//! * **Sections** ([`section`]): shared memory, mapped as views into any number
//!   of address spaces.
//! * **System calls** (`syscall`): the ring-3 interface, `int 0x80` numbers 8-18.
//!
//! Both object types follow the Object Manager's rules: they are named under
//! `\Ports` and `\Sections`, reached through handles that carry **access
//! rights** (`PORT_SEND`, `PORT_RECEIVE`, `SECTION_MAP_WRITE`, ...), reference
//! counted, duplicated with fewer rights, and destroyed with their last
//! reference. A port is *disconnected* when its last handle closes, which wakes
//! everything blocked on it.
//!
//! # Not done yet
//!
//! No security descriptors: any process may open any named port with any
//! rights (the Security reference monitor is the next Executive component).
//! No quotas on memory held in queues or sections. Messages are copied twice
//! (user → kernel → user); large payloads belong in a section.

// An API for subsystems that do not exist yet: not all of it is used today.
#![allow(dead_code)]

pub mod port;
pub mod section;
mod selftest;
pub mod syscall;

use alloc::vec::Vec;
use core::ptr;
use core::sync::atomic::{AtomicPtr, Ordering};

use crate::ob::access::*;
use crate::ob::handle::DuplicateOptions;
use crate::ob::object::{self, ObRef, TypeInit};
use crate::ob::types::{create_directory, event_type};
use crate::ob::{Handle, HandleTable, ObError, ObjectAttributes, ObjectType};
use crate::ps::sched;

#[allow(unused_imports)]
pub use port::{
    create_port, Attachment, Deadline, Message, MessageKind, Port, PortInfo, DEFAULT_QUEUE, MAX_ATTACHMENTS, MAX_MESSAGE, MAX_QUEUE,
};
#[allow(unused_imports)]
pub use section::{create_section, map_view, process_ended, unmap_view, Section, MAX_SECTION_PAGES};
pub use selftest::self_test;

use port::{drive, Outgoing, Step};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IpcError {
    NotReady,
    AlreadyInitialised,
    InvalidHandle,
    /// The handle is not a port / section.
    TypeMismatch,
    /// The handle lacks the needed right (`PORT_SEND`, `PORT_RECEIVE`, ...).
    AccessDenied,
    NotFound,
    NameInUse,
    InvalidName,
    InvalidArgument,
    /// The port was disconnected (its last handle closed).
    Closed,
    /// The operation would block and the caller asked not to wait.
    WouldBlock,
    TimedOut,
    /// The request was withdrawn before it was answered.
    Cancelled,
    MessageTooBig,
    /// The next message / reply is this many bytes; nothing was consumed
    /// (a reply that does not fit is discarded).
    BufferTooSmall(usize),
    TooManyAttachments,
    /// No request with that id is outstanding (answered, withdrawn, never sent).
    NoSuchRequest,
    /// The request is still in the queue: nobody has received it yet.
    NotDelivered,
    /// The request was received by a different process.
    NotYourRequest,
    AlreadyReplied,
    /// A user pointer is not mapped (or not writable where it must be).
    BadAddress,
    OutOfResources,
    /// The target process has ended or has no user address space.
    NoProcess,
    Ob(ObError),
}

impl IpcError {
    /// The negative number a system call returns for this error.
    pub fn code(self) -> i64 {
        match self {
            IpcError::NotReady => -1,
            IpcError::InvalidHandle => -2,
            IpcError::TypeMismatch => -3,
            IpcError::AccessDenied => -4,
            IpcError::NotFound => -5,
            IpcError::NameInUse => -6,
            IpcError::InvalidName => -7,
            IpcError::InvalidArgument => -8,
            IpcError::Closed => -9,
            IpcError::WouldBlock => -10,
            IpcError::TimedOut => -11,
            IpcError::Cancelled => -12,
            IpcError::MessageTooBig => -13,
            IpcError::BufferTooSmall(_) => -14,
            IpcError::TooManyAttachments => -15,
            IpcError::NoSuchRequest => -16,
            IpcError::NotDelivered => -17,
            IpcError::NotYourRequest => -18,
            IpcError::AlreadyReplied => -19,
            IpcError::BadAddress => -20,
            IpcError::OutOfResources => -21,
            IpcError::NoProcess => -22,
            IpcError::Ob(_) => -23,
            IpcError::AlreadyInitialised => -24,
        }
    }
}

impl From<ObError> for IpcError {
    fn from(e: ObError) -> Self {
        match e {
            ObError::InvalidHandle => IpcError::InvalidHandle,
            ObError::TypeMismatch => IpcError::TypeMismatch,
            ObError::AccessDenied | ObError::InvalidAccess => IpcError::AccessDenied,
            ObError::NameNotFound | ObError::PathNotFound => IpcError::NotFound,
            ObError::NameCollision => IpcError::NameInUse,
            ObError::InvalidName | ObError::NotADirectory => IpcError::InvalidName,
            ObError::OutOfResources | ObError::TooManyHandles => IpcError::OutOfResources,
            ObError::NotReady => IpcError::NotReady,
            other => IpcError::Ob(other),
        }
    }
}

// ----------------------------------------------------------------------
// Types
// ----------------------------------------------------------------------

pub(crate) static PORT_TYPE: AtomicPtr<ObjectType> = AtomicPtr::new(ptr::null_mut());
pub(crate) static SECTION_TYPE: AtomicPtr<ObjectType> = AtomicPtr::new(ptr::null_mut());

pub fn port_type() -> &'static ObjectType {
    let p = PORT_TYPE.load(Ordering::Acquire);
    assert!(!p.is_null(), "ipc manager not initialised");
    // SAFETY: set once from an immortal type object.
    unsafe { &*p }
}

pub fn section_type() -> &'static ObjectType {
    let p = SECTION_TYPE.load(Ordering::Acquire);
    assert!(!p.is_null(), "ipc manager not initialised");
    // SAFETY: set once from an immortal type object.
    unsafe { &*p }
}

/// What [`init`] created, for the boot log.
#[derive(Clone, Copy)]
pub struct InitInfo {
    pub types: usize,
    pub directories: usize,
}

/// Registers the `Port` and `Section` types and the `\Ports` and `\Sections`
/// directories. Needs the Object Manager (and the Process Manager for
/// blocking). Boot CPU, once.
pub fn init() -> Result<InitInfo, IpcError> {
    if !PORT_TYPE.load(Ordering::Acquire).is_null() {
        return Err(IpcError::AlreadyInitialised);
    }
    let map = |read, write, exec, all| GenericMapping { read, write, execute: exec, all };
    let pt = object::create_type(
        TypeInit::of::<Port>(
            "Port",
            PORT_ALL,
            map(READ_CONTROL | PORT_RECEIVE | PORT_QUERY, READ_CONTROL | PORT_SEND, READ_CONTROL, PORT_ALL),
        )
        .on_last_close(port::port_last_close),
    )?;
    PORT_TYPE.store(pt as *const _ as *mut _, Ordering::Release);
    let st = object::create_type(TypeInit::of::<Section>(
        "Section",
        SECTION_ALL,
        map(
            READ_CONTROL | SECTION_QUERY | SECTION_MAP_READ,
            READ_CONTROL | SECTION_MAP_WRITE,
            READ_CONTROL,
            SECTION_ALL,
        ),
    ))?;
    SECTION_TYPE.store(st as *const _ as *mut _, Ordering::Release);
    for dir in ["\\Ports", "\\Sections"] {
        drop(create_directory(&ObjectAttributes::named(dir).permanent())?);
    }
    Ok(InitInfo { types: 2, directories: 2 })
}

// ----------------------------------------------------------------------
// Handle-based API (what a process can do with its handles)
// ----------------------------------------------------------------------

/// A message received through a handle: the attached objects already have
/// handles in the receiver's table.
pub struct Received {
    pub message: Message,
    pub handles: Vec<Handle>,
}

fn resolve(table: &HandleTable, h: Handle, access: AccessMask) -> Result<ObRef, IpcError> {
    Ok(table.reference_by_handle(h, access, Some(port_type()))?.0)
}

fn port_of(obj: &ObRef) -> &Port {
    obj.body::<Port>(port_type()).expect("type checked")
}

/// Resolves each handle of `attach` to the object and the access it holds.
pub fn collect_attachments(table: &HandleTable, attach: &[Handle]) -> Result<Vec<Attachment>, IpcError> {
    if attach.len() > MAX_ATTACHMENTS {
        return Err(IpcError::TooManyAttachments);
    }
    let mut v = Vec::with_capacity(attach.len());
    for &h in attach {
        let (object, access) = table.reference_by_handle(h, 0, None)?;
        v.push(Attachment { object, access });
    }
    Ok(v)
}

/// Sends a message built by `build` (called again on every attempt, so a
/// blocked sender holds nothing). Needs `PORT_SEND`. Returns the message id.
pub fn send_with(
    table: &HandleTable,
    h: Handle,
    kind: MessageKind,
    mut build: impl FnMut() -> Result<(Vec<u8>, Vec<Attachment>), IpcError>,
    timeout_ms: Option<u64>,
) -> Result<u64, IpcError> {
    drive(Deadline::from_ms(timeout_ms), || {
        let obj = match resolve(table, h, PORT_SEND) {
            Ok(o) => o,
            Err(e) => return Step::Fail(e),
        };
        let (data, attachments) = match build() {
            Ok(x) => x,
            Err(e) => return Step::Fail(e),
        };
        let mut slot = Some(Outgoing { msg: Message::new(kind, data, attachments) });
        port_of(&obj).push_step(&mut slot)
    })
}

/// Datagram from kernel memory.
pub fn send(table: &HandleTable, h: Handle, data: &[u8], attach: &[Handle], timeout_ms: Option<u64>) -> Result<u64, IpcError> {
    send_with(table, h, MessageKind::Datagram, || Ok((data.to_vec(), collect_attachments(table, attach)?)), timeout_ms)
}

/// Takes the oldest message (`PORT_RECEIVE`). Attached objects get new handles
/// in `table`. A message longer than `max_len` is left queued.
pub fn receive(table: &HandleTable, h: Handle, max_len: usize, timeout_ms: Option<u64>) -> Result<Received, IpcError> {
    let pid = sched::current_pid();
    let mut message = drive(Deadline::from_ms(timeout_ms), || {
        let obj = match resolve(table, h, PORT_RECEIVE) {
            Ok(o) => o,
            Err(e) => return Step::Fail(e),
        };
        port_of(&obj).receive_step(pid, max_len)
    })?;
    let attachments = core::mem::take(&mut message.attachments);
    let mut handles = Vec::with_capacity(attachments.len());
    for a in &attachments {
        match table.create_handle(&a.object, a.access, 0) {
            Ok(nh) => handles.push(nh),
            Err(e) => {
                for nh in handles {
                    let _ = table.close(nh);
                }
                // The message is gone; a request would be left unanswered, so withdraw it by replying nothing is not possible here.
                return Err(e.into());
            }
        }
    }
    Ok(Received { message, handles })
}

/// Sends a request and waits for the reply (`PORT_SEND`). The reply is at most
/// `max_reply` bytes. On timeout, error or kill the request is withdrawn.
pub fn request_with(
    table: &HandleTable,
    h: Handle,
    build: impl FnMut() -> Result<(Vec<u8>, Vec<Attachment>), IpcError>,
    max_reply: usize,
    timeout_ms: Option<u64>,
) -> Result<Vec<u8>, IpcError> {
    let deadline = Deadline::from_ms(timeout_ms);
    let mut build = build;
    let id = drive(deadline, || {
        let obj = match resolve(table, h, PORT_SEND) {
            Ok(o) => o,
            Err(e) => return Step::Fail(e),
        };
        let (data, attachments) = match build() {
            Ok(x) => x,
            Err(e) => return Step::Fail(e),
        };
        let mut slot = Some(Outgoing { msg: Message::new(MessageKind::Request, data, attachments) });
        port_of(&obj).push_step(&mut slot)
    })?;
    let r = drive(deadline, || {
        let obj = match resolve(table, h, 0) {
            Ok(o) => o,
            Err(e) => return Step::Fail(e),
        };
        port_of(&obj).call_step(id, max_reply)
    });
    if let Err(e) = &r {
        // Closed / Cancelled already removed the call; for the rest, withdraw it.
        if !matches!(e, IpcError::Closed | IpcError::Cancelled | IpcError::BufferTooSmall(_)) {
            if let Ok(obj) = resolve(table, h, 0) {
                port_of(&obj).abandon(id);
            }
        }
    }
    r
}

/// Request from kernel memory.
pub fn request(table: &HandleTable, h: Handle, data: &[u8], max_reply: usize, timeout_ms: Option<u64>) -> Result<Vec<u8>, IpcError> {
    request_with(table, h, || Ok((data.to_vec(), Vec::new())), max_reply, timeout_ms)
}

/// Answers request `id`, which this process received through `h` (`PORT_RECEIVE`).
pub fn reply(table: &HandleTable, h: Handle, id: u64, data: &[u8]) -> Result<(), IpcError> {
    let obj = resolve(table, h, PORT_RECEIVE)?;
    port_of(&obj).reply(id, data, sched::current_pid())
}

/// Port statistics through a handle (`PORT_QUERY`).
pub fn query(table: &HandleTable, h: Handle) -> Result<PortInfo, IpcError> {
    let obj = resolve(table, h, PORT_QUERY)?;
    Ok(port_of(&obj).info())
}

/// A new handle to the same object with fewer rights (the way a server hands
/// a client a send-only port).
pub fn duplicate_reduced(table: &HandleTable, h: Handle, access: AccessMask) -> Result<Handle, IpcError> {
    Ok(table.duplicate(h, table, DuplicateOptions { access: Some(access), attributes: 0, close_source: false })?)
}

/// Creates a named or unnamed port and a handle to it with `access`.
pub fn create_port_handle(table: &HandleTable, attrs: &ObjectAttributes<'_>, capacity: usize, access: AccessMask) -> Result<Handle, IpcError> {
    let p = create_port(attrs, capacity)?;
    Ok(table.create_handle(p.untyped(), access, 0)?)
}

/// Opens `\Ports\<name>` (or any absolute path) with `access`.
pub fn open_port(table: &HandleTable, path: &str, access: AccessMask) -> Result<Handle, IpcError> {
    Ok(crate::ob::open_by_name(table, path, access, Some(port_type()), 0)?)
}

// ----------------------------------------------------------------------
// Statistics
// ----------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub ports: usize,
    pub sections: usize,
    pub sent: u64,
    pub received: u64,
    pub requests: u64,
    pub replies: u64,
    pub bytes: u64,
    pub views_mapped: u64,
    pub syscalls: u64,
}

pub fn stats() -> Stats {
    Stats {
        ports: port_type().total_objects.load(Ordering::Relaxed),
        sections: section_type().total_objects.load(Ordering::Relaxed),
        sent: port::TOTAL_SENT.load(Ordering::Relaxed),
        received: port::TOTAL_RECEIVED.load(Ordering::Relaxed),
        requests: port::TOTAL_REQUESTS.load(Ordering::Relaxed),
        replies: port::TOTAL_REPLIES.load(Ordering::Relaxed),
        bytes: port::TOTAL_BYTES.load(Ordering::Relaxed),
        views_mapped: section::VIEWS_MAPPED.load(Ordering::Relaxed),
        syscalls: syscall::count(),
    }
}

/// `Event` is what blocked threads wait on; re-exported for the type check in tests.
pub(crate) fn event_object_type() -> &'static ObjectType {
    event_type()
}
