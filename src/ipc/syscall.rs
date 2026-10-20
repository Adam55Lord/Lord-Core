// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! The IPC system calls, reached through the same `int 0x80` gate as the
//! Process Manager's (`ps/user.rs` forwards numbers 8 and up).
//!
//! **ABI**: number in `rax`; arguments in `rdi`, `rsi`, `rdx`, `r10`, `r8`,
//! `r9` (in that order); result in `rax`: a value `>= 0`, or a **negative error**
//! (see [`IpcError::code`]). Every other register is preserved. Handles are the
//! caller's own (its process's `HandleTable`). A *timeout* argument is in
//! milliseconds: `0` polls, `u64::MAX` waits forever.
//!
//! | `rax` | call | arguments | returns |
//! |---|---|---|---|
//! | 8 | `port_create` | name ptr (0 = unnamed), name len, capacity (0 = default) | handle, all rights |
//! | 9 | `port_open` | name ptr, name len, desired access | handle |
//! | 10 | `close` | handle | 0 |
//! | 11 | `send` | port, buf, len, attach ptr, attach count, timeout | message id |
//! | 12 | `receive` | port, buf, capacity, info ptr (0 = none), timeout | length |
//! | 13 | `request` | port, buf, len, reply buf, reply capacity, timeout | reply length |
//! | 14 | `reply` | port, request id, buf, len | 0 |
//! | 15 | `section_create` | name ptr (0 = unnamed), name len, pages | handle, all rights |
//! | 16 | `section_map` | section, address hint (0 = any), flags (1 = writable) | address |
//! | 17 | `section_unmap` | address | 0 |
//! | 18 | `handle_dup` | handle, access (0 = same) | new handle, fewer rights |
//!
//! Names are leaf names (no `\`, 1-64 bytes): ports live in `\Ports`,
//! sections in `\Sections`; user code cannot name anything else.
//!
//! `receive` fills the 72-byte **info block** (nine `u64`): message id, sender
//! pid, sender tid, kind (0 datagram, 1 request), attachment count, then up to
//! four handles that were created in the receiver's table for the attachments.
//! A message longer than the buffer stays queued and the call fails with
//! `BufferTooSmall`.
//!
//! Blocking calls hold nothing while they sleep: the handle is looked up again
//! after every wake-up and the message is rebuilt from user memory, so killing
//! a blocked process leaks nothing.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

use super::port::{MessageKind, MAX_ATTACHMENTS, MAX_MESSAGE};
use super::*;
use crate::hal::idt::InterruptFrame;
use crate::ps::{copy_from_user, copy_to_user, current_handles, current_process, user_range_ok};

pub const SYS_PORT_CREATE: u64 = 8;
pub const SYS_PORT_OPEN: u64 = 9;
pub const SYS_CLOSE: u64 = 10;
pub const SYS_SEND: u64 = 11;
pub const SYS_RECEIVE: u64 = 12;
pub const SYS_REQUEST: u64 = 13;
pub const SYS_REPLY: u64 = 14;
pub const SYS_SECTION_CREATE: u64 = 15;
pub const SYS_SECTION_MAP: u64 = 16;
pub const SYS_SECTION_UNMAP: u64 = 17;
pub const SYS_HANDLE_DUP: u64 = 18;

/// Timeout value meaning "wait forever".
pub const INFINITE: u64 = u64::MAX;
/// Size of the receive info block in `u64` words.
pub const INFO_WORDS: usize = 9;

static SYSCALLS: AtomicU64 = AtomicU64::new(0);

/// IPC system calls handled so far.
pub fn count() -> u64 {
    SYSCALLS.load(Ordering::Relaxed)
}

/// Is `n` one of ours?
pub fn handles(n: u64) -> bool {
    (SYS_PORT_CREATE..=SYS_HANDLE_DUP).contains(&n)
}

fn timeout(v: u64) -> Option<u64> {
    if v == INFINITE { None } else { Some(v) }
}

fn user_name(ptr: u64, len: u64, prefix: &str) -> Result<String, IpcError> {
    if len == 0 || len > 64 {
        return Err(IpcError::InvalidName);
    }
    let mut buf = [0u8; 64];
    if !copy_from_user(ptr, &mut buf[..len as usize]) {
        return Err(IpcError::BadAddress);
    }
    let s = core::str::from_utf8(&buf[..len as usize]).map_err(|_| IpcError::InvalidName)?;
    if s.contains('\\') {
        return Err(IpcError::InvalidName);
    }
    Ok(format!("{prefix}{s}"))
}

/// Reads `count` handle values from user memory.
fn user_handles(ptr: u64, count: u64) -> Result<Vec<Handle>, IpcError> {
    if count as usize > MAX_ATTACHMENTS {
        return Err(IpcError::TooManyAttachments);
    }
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut raw = [0u8; 8 * MAX_ATTACHMENTS];
    let raw = &mut raw[..count as usize * 8];
    if !copy_from_user(ptr, raw) {
        return Err(IpcError::BadAddress);
    }
    Ok(raw.chunks_exact(8).map(|c| Handle(u64::from_le_bytes(c.try_into().expect("8 bytes")) as usize)).collect())
}

/// Builds the payload of an outgoing message from user memory; called again on
/// every attempt so a blocked sender holds nothing.
fn user_payload(table: &HandleTable, buf: u64, len: u64, attach_ptr: u64, attach_count: u64) -> Result<(Vec<u8>, Vec<Attachment>), IpcError> {
    if len as usize > MAX_MESSAGE {
        return Err(IpcError::MessageTooBig);
    }
    let mut data = vec![0u8; len as usize];
    if !copy_from_user(buf, &mut data) {
        return Err(IpcError::BadAddress);
    }
    let handles = user_handles(attach_ptr, attach_count)?;
    Ok((data, collect_attachments(table, &handles)?))
}

fn port_create(table: &HandleTable, name_ptr: u64, name_len: u64, capacity: u64) -> Result<u64, IpcError> {
    let path = if name_ptr != 0 { Some(user_name(name_ptr, name_len, "\\Ports\\")?) } else { None };
    let attrs = path.as_deref().map_or(ObjectAttributes::unnamed(), ObjectAttributes::named);
    let h = create_port_handle(table, &attrs, capacity.min(MAX_QUEUE as u64 + 1) as usize, PORT_ALL)?;
    Ok(h.0 as u64)
}

fn port_open(table: &HandleTable, name_ptr: u64, name_len: u64, access: u64) -> Result<u64, IpcError> {
    let path = user_name(name_ptr, name_len, "\\Ports\\")?;
    Ok(open_port(table, &path, access as AccessMask)?.0 as u64)
}

fn sys_send(table: &HandleTable, a: [u64; 6]) -> Result<u64, IpcError> {
    let [h, buf, len, attach_ptr, attach_count, t] = a;
    send_with(table, Handle(h as usize), MessageKind::Datagram, || user_payload(table, buf, len, attach_ptr, attach_count), timeout(t))
}

fn sys_receive(table: &HandleTable, a: [u64; 6]) -> Result<u64, IpcError> {
    let [h, buf, cap, info_ptr, t, _] = a;
    let cap = (cap as usize).min(MAX_MESSAGE);
    // Fail before a message is taken, not after.
    if !user_range_ok(buf, cap, true) || (info_ptr != 0 && !user_range_ok(info_ptr, INFO_WORDS * 8, true)) {
        return Err(IpcError::BadAddress);
    }
    let r = receive(table, Handle(h as usize), cap, timeout(t))?;
    let copied = copy_to_user(buf, &r.message.data);
    let mut info = [0u64; INFO_WORDS];
    info[0] = r.message.id;
    info[1] = r.message.sender_pid;
    info[2] = r.message.sender_tid;
    info[3] = (r.message.kind == MessageKind::Request) as u64;
    info[4] = r.handles.len() as u64;
    for (slot, hd) in info[5..].iter_mut().zip(&r.handles) {
        *slot = hd.0 as u64;
    }
    let mut bytes = [0u8; INFO_WORDS * 8];
    for (chunk, w) in bytes.chunks_exact_mut(8).zip(info) {
        chunk.copy_from_slice(&w.to_le_bytes());
    }
    if !copied || (info_ptr != 0 && !copy_to_user(info_ptr, &bytes)) {
        for hd in &r.handles {
            let _ = table.close(*hd); // the user memory vanished under us: do not leak the handles
        }
        return Err(IpcError::BadAddress);
    }
    Ok(r.message.data.len() as u64)
}

fn sys_request(table: &HandleTable, a: [u64; 6]) -> Result<u64, IpcError> {
    let [h, buf, len, reply_buf, reply_cap, t] = a;
    let reply_cap = (reply_cap as usize).min(MAX_MESSAGE);
    if !user_range_ok(reply_buf, reply_cap, true) {
        return Err(IpcError::BadAddress);
    }
    let data = request_with(table, Handle(h as usize), || user_payload(table, buf, len, 0, 0), reply_cap, timeout(t))?;
    if !copy_to_user(reply_buf, &data) {
        return Err(IpcError::BadAddress);
    }
    Ok(data.len() as u64)
}

fn sys_reply(table: &HandleTable, a: [u64; 6]) -> Result<u64, IpcError> {
    let [h, id, buf, len, _, _] = a;
    if len as usize > MAX_MESSAGE {
        return Err(IpcError::MessageTooBig);
    }
    let mut data = vec![0u8; len as usize];
    if !copy_from_user(buf, &mut data) {
        return Err(IpcError::BadAddress);
    }
    reply(table, Handle(h as usize), id, &data)?;
    Ok(0)
}

fn section_create_call(table: &HandleTable, name_ptr: u64, name_len: u64, pages: u64) -> Result<u64, IpcError> {
    let path = if name_ptr != 0 { Some(user_name(name_ptr, name_len, "\\Sections\\")?) } else { None };
    let attrs = path.as_deref().map_or(ObjectAttributes::unnamed(), ObjectAttributes::named);
    let s = create_section(&attrs, pages)?;
    Ok(table.create_handle(s.untyped(), SECTION_ALL, 0)?.0 as u64)
}

fn section_map_call(table: &HandleTable, h: u64, hint: u64, flags: u64) -> Result<u64, IpcError> {
    let write = flags & 1 != 0;
    let need = SECTION_MAP_READ | if write { SECTION_MAP_WRITE } else { 0 };
    let (obj, _) = table.reference_by_handle(Handle(h as usize), need, Some(section_type()))?;
    map_view(&current_process(), &obj, hint, write)
}

/// Handles one IPC system call; `f.rax` holds the number on entry and the result on return.
pub fn dispatch(f: &mut InterruptFrame) {
    SYSCALLS.fetch_add(1, Ordering::Relaxed);
    // SAFETY: the table belongs to the calling thread's process, which the thread keeps alive.
    let table: &HandleTable = unsafe { &*current_handles() };
    let a = [f.rdi, f.rsi, f.rdx, f.r10, f.r8, f.r9];
    let result = match f.rax {
        SYS_PORT_CREATE => port_create(table, a[0], a[1], a[2]),
        SYS_PORT_OPEN => port_open(table, a[0], a[1], a[2]),
        SYS_CLOSE => table.close(Handle(a[0] as usize)).map(|_| 0).map_err(IpcError::from),
        SYS_SEND => sys_send(table, a),
        SYS_RECEIVE => sys_receive(table, a),
        SYS_REQUEST => sys_request(table, a),
        SYS_REPLY => sys_reply(table, a),
        SYS_SECTION_CREATE => section_create_call(table, a[0], a[1], a[2]),
        SYS_SECTION_MAP => section_map_call(table, a[0], a[1], a[2]),
        SYS_SECTION_UNMAP => unmap_view(&current_process(), a[0]).map(|_| 0),
        SYS_HANDLE_DUP => {
            let mask = a[1] as AccessMask;
            let r = if mask == 0 {
                table.duplicate(Handle(a[0] as usize), table, crate::ob::handle::DuplicateOptions::same_access()).map_err(IpcError::from)
            } else {
                duplicate_reduced(table, Handle(a[0] as usize), mask)
            };
            r.map(|h| h.0 as u64)
        }
        _ => Err(IpcError::InvalidArgument),
    };
    f.rax = match result {
        Ok(v) => v,
        Err(e) => e.code() as u64,
    };
}
