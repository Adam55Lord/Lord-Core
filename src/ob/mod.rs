// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! The **Object Manager** (OB): "everything is an object".
//!
//! Every kernel resource that is shared, named, counted or handed to user
//! code is an *object*: a body (a Rust struct) behind an [`ObjectHeader`]
//! that carries the reference counts, the object's [`ObjectType`], its place in
//! the namespace and (later) its security descriptor. The Object Manager is the
//! one place that knows how objects are created, found, referenced, named,
//! handed out as handles and destroyed.
//!
//! ```text
//!   handle ──▶ HandleTable entry ──▶ ┌──────────────┐
//!   (u32*4)    {object, access}      │ ObjectHeader │  pointer_count, handle_count,
//!                                    │              │  type ─▶ ObjectType (itself an object)
//!   "\Device\Serial0" ──namespace──▶ │              │  name  ─▶ parent Directory
//!                                    ├──────────────┤
//!                                    │ body (T)     │  Event, Directory, Device, ...
//!                                    └──────────────┘
//! ```
//!
//! # Pieces
//!
//! * [`object`]: `ObjectHeader`, `ObjectType`, reference counting, creation and
//!   destruction. [`ObRef`] / [`Obj`] are owning smart pointers.
//! * [`namespace`]: the hierarchical, case-insensitive namespace (`\`, directories,
//!   symbolic links, temporary vs permanent names).
//! * [`handle`]: [`HandleTable`] (`HANDLE_TABLE`): per-owner tables mapping handles to
//!   objects with **granted access**, handle attributes, duplication.
//! * [`types`]: the built-in types `Type`, `Directory`, `SymbolicLink`, `Event`,
//!   `Device`, `Processor`, `AddressSpace`.
//! * [`access`]: `ACCESS_MASK` bits and generic mappings.
//!
//! # Counting rules
//!
//! * `pointer_count`: references to the object (Rust [`ObRef`]s, one per open
//!   handle, one extra while the object is *permanent*, one per child for a
//!   directory). At zero the object is unlinked from the namespace, its type's
//!   delete routine runs and the memory is freed.
//! * `handle_count`: open handles. When the last one closes, the type's
//!   `last_close` hook runs and a *temporary* name is removed from the
//!   namespace (the object itself lives on while anything references it).
//!
//! # Locking
//!
//! One namespace lock (interrupts masked) covers every directory and every name;
//! each handle table has its own lock; counts are atomic. Lock order:
//! `ob -> heap -> vmm -> pmm`. No lock is held while a type callback runs or
//! while an object is freed.
//!
//! # Not done yet
//!
//! No security descriptors (the header slot exists), no object waiting
//! (needs a scheduler), no per-process tables until processes exist: today
//! there is one kernel handle table, [`kernel_handles`]. Handles are not
//! generation-tagged, so a closed handle value can be handed out again.

// The Object Manager is an API for code that does not exist yet (processes, drivers):
// not everything in it is used by the kernel today.
#![allow(dead_code)]

pub mod access;
pub mod handle;
pub mod namespace;
pub mod object;
mod selftest;
pub mod types;

use alloc::boxed::Box;
use alloc::string::String;
use core::sync::atomic::{AtomicPtr, Ordering};

pub use access::AccessMask;
pub use handle::{Handle, HandleTable};
#[allow(unused_imports)]
pub use object::{Obj, ObRef, ObjectHeader, ObjectType};
pub use selftest::self_test;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObError {
    NotReady,
    AlreadyInitialised,
    /// Not a handle of this table (zero, misaligned, closed, never issued).
    InvalidHandle,
    /// The object is not of the requested type.
    TypeMismatch,
    /// The handle does not grant the requested rights.
    AccessDenied,
    /// Rights requested that the object type does not define.
    InvalidAccess,
    /// A name is already taken in that directory.
    NameCollision,
    NameNotFound,
    /// An intermediate component of a path does not exist.
    PathNotFound,
    NotADirectory,
    /// Empty/overlong component, control characters, bad root, ...
    InvalidName,
    /// Too many symbolic links while resolving a path.
    LinkLoop,
    /// The handle is protected from close.
    HandleProtected,
    TooManyHandles,
    /// Another kernel resource (frames, address space) ran out.
    OutOfResources,
}

/// How a new object is named and kept. See the builders.
#[derive(Clone, Copy)]
pub struct ObjectAttributes<'a> {
    /// Absolute path (`\Dir\Name`), or relative to `root`. `None`: unnamed.
    pub name: Option<&'a str>,
    pub root: Option<&'a ObRef>,
    /// Keep the object (and its name) alive without any handle or reference.
    pub permanent: bool,
}

impl<'a> ObjectAttributes<'a> {
    pub const fn unnamed() -> Self {
        Self { name: None, root: None, permanent: false }
    }
    pub const fn named(path: &'a str) -> Self {
        Self { name: Some(path), root: None, permanent: false }
    }
    pub const fn permanent(mut self) -> Self {
        self.permanent = true;
        self
    }
    pub const fn relative_to(mut self, dir: &'a ObRef) -> Self {
        self.root = Some(dir);
        self
    }
}

// ----------------------------------------------------------------------
// Initialisation and the kernel handle table
// ----------------------------------------------------------------------

static KERNEL_HANDLES: AtomicPtr<HandleTable> = AtomicPtr::new(core::ptr::null_mut());

/// The handle table of the kernel itself (kernel-mode callers; per-process
/// tables arrive with processes).
pub fn kernel_handles() -> &'static HandleTable {
    let p = KERNEL_HANDLES.load(Ordering::Acquire);
    assert!(!p.is_null(), "object manager not initialised");
    // SAFETY: set once in `init` from a leaked Box; never freed.
    unsafe { &*p }
}

/// What [`init`] created, for the boot log.
#[derive(Clone, Copy)]
pub struct InitInfo {
    pub types: usize,
    pub directories: usize,
}

/// Creates the type system and the root of the namespace. Needs the heap.
/// Boot CPU, once.
pub fn init() -> Result<InitInfo, ObError> {
    if !crate::mm::heap::is_ready() {
        return Err(ObError::NotReady);
    }
    if !KERNEL_HANDLES.load(Ordering::Acquire).is_null() {
        return Err(ObError::AlreadyInitialised);
    }
    types::init()?;
    KERNEL_HANDLES.store(Box::into_raw(Box::new(HandleTable::new())), Ordering::Release);
    Ok(InitInfo { types: object::type_count(), directories: namespace::snapshot().iter().filter(|e| e.type_name == "Directory").count() + 1 })
}

// ----------------------------------------------------------------------
// Convenience layer: the "Nt" calls (handle in, rights checked)
// ----------------------------------------------------------------------

/// Looks `path` up and opens a handle to it in `table` (`ObOpenObjectByName`).
pub fn open_by_name(
    table: &HandleTable,
    path: &str,
    desired: AccessMask,
    expect: Option<&'static ObjectType>,
    attributes: u32,
) -> Result<Handle, ObError> {
    let obj = namespace::lookup(path, expect, true)?;
    table.create_handle(&obj, desired, attributes)
}

/// Gives `obj` a handle in `table` (`ObInsertObject`).
pub fn insert_object(table: &HandleTable, obj: &ObRef, access: AccessMask, attributes: u32) -> Result<Handle, ObError> {
    table.create_handle(obj, access, attributes)
}

/// Sets an event through a handle: needs `EVENT_MODIFY_STATE`. Returns the previous state.
pub fn set_event(table: &HandleTable, h: Handle) -> Result<bool, ObError> {
    let (obj, _) = table.reference_by_handle(h, access::EVENT_MODIFY_STATE, Some(types::event_type()))?;
    Ok(obj.body::<types::Event>(types::event_type()).expect("type checked").set())
}

/// Resets an event through a handle: needs `EVENT_MODIFY_STATE`.
pub fn reset_event(table: &HandleTable, h: Handle) -> Result<bool, ObError> {
    let (obj, _) = table.reference_by_handle(h, access::EVENT_MODIFY_STATE, Some(types::event_type()))?;
    Ok(obj.body::<types::Event>(types::event_type()).expect("type checked").reset())
}

/// Reads an event's state through a handle: needs `EVENT_QUERY_STATE`.
pub fn query_event(table: &HandleTable, h: Handle) -> Result<bool, ObError> {
    let (obj, _) = table.reference_by_handle(h, access::EVENT_QUERY_STATE, Some(types::event_type()))?;
    Ok(obj.body::<types::Event>(types::event_type()).expect("type checked").is_signaled())
}

// ----------------------------------------------------------------------
// Statistics and inspection
// ----------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub types: usize,
    pub objects: usize,
    pub handles: usize,
    pub peak_objects: usize,
}

pub fn stats() -> Stats {
    let mut s = Stats { types: 0, objects: 0, handles: 0, peak_objects: 0 };
    object::for_each_type(|t| {
        s.types += 1;
        s.objects += t.total_objects.load(Ordering::Relaxed);
        s.handles += t.total_handles.load(Ordering::Relaxed);
        s.peak_objects += t.peak_objects.load(Ordering::Relaxed);
    });
    s
}

/// Per-type counters, for the boot log.
#[derive(Clone, Debug)]
pub struct TypeStats {
    pub name: &'static str,
    pub objects: usize,
    pub handles: usize,
    pub peak_objects: usize,
    pub peak_handles: usize,
}

pub fn type_stats() -> alloc::vec::Vec<TypeStats> {
    let mut v = alloc::vec::Vec::new();
    object::for_each_type(|t| {
        v.push(TypeStats {
            name: t.name,
            objects: t.total_objects.load(Ordering::Relaxed),
            handles: t.total_handles.load(Ordering::Relaxed),
            peak_objects: t.peak_objects.load(Ordering::Relaxed),
            peak_handles: t.peak_handles.load(Ordering::Relaxed),
        });
    });
    v
}

/// `\a\b` style path of an object, if it is in the namespace.
pub fn path_of(obj: &ObRef) -> Option<String> {
    namespace::path_of(obj)
}
