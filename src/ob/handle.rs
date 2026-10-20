// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! `HANDLE_TABLE`: handles are the only way code outside the Object Manager
//! names an object, and every handle carries the **access it was granted**.
//!
//! A table is a two-level array of 16-byte entries (pages of 256, grown on
//! demand, up to [`MAX_HANDLES`]). A handle value is `index * 4`; index 0 is
//! never issued, so the value 0 is always invalid, and values that are not
//! multiples of 4 are rejected. Closed entries form a LIFO free list (the
//! most recently closed handle value is reused first).
//!
//! Each open handle owns one pointer reference and one handle count on its
//! object. The table has its own lock; type callbacks and object destruction
//! always run after it is released.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use super::access::{AccessMask, MAXIMUM_ALLOWED};
use super::object::{release_handle, ObRef, ObjectHeader, ObjectType};
use super::ObError;
use crate::hal::sync::IrqLock;

/// Handle attribute: cannot be closed until the flag is cleared.
pub const HANDLE_PROTECT_FROM_CLOSE: u32 = 0x1;
/// Handle attribute: copied into child tables on inheritance (future processes).
pub const HANDLE_INHERIT: u32 = 0x2;

const PAGE_ENTRIES: usize = 256;
/// Upper bound on handles per table (16 M entries).
pub const MAX_HANDLES: usize = 1 << 24;
const NO_FREE: u32 = u32::MAX;

/// An opaque handle value. `Handle::NULL` is never valid.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Handle(pub usize);

impl Handle {
    pub const NULL: Handle = Handle(0);
}

#[derive(Clone, Copy)]
struct Entry {
    /// Header address; 0 = free entry.
    object: usize,
    /// Granted access; for a free entry: index of the next free entry.
    access: u32,
    attrs: u32,
}

const EMPTY: Entry = Entry { object: 0, access: NO_FREE, attrs: 0 };

struct Inner {
    pages: Vec<Box<[Entry]>>,
    next_fresh: u32,
    free_head: u32,
    count: usize,
}

impl Inner {
    fn entry(&mut self, idx: usize) -> &mut Entry {
        &mut self.pages[idx / PAGE_ENTRIES][idx % PAGE_ENTRIES]
    }

    fn slot_for(&mut self, h: Handle) -> Result<usize, ObError> {
        if h.0 == 0 || h.0 % 4 != 0 {
            return Err(ObError::InvalidHandle);
        }
        let idx = h.0 / 4;
        if idx >= self.next_fresh as usize {
            return Err(ObError::InvalidHandle);
        }
        if self.entry(idx).object == 0 {
            return Err(ObError::InvalidHandle);
        }
        Ok(idx)
    }

    fn allocate(&mut self) -> Result<usize, ObError> {
        if self.free_head != NO_FREE {
            let idx = self.free_head as usize;
            self.free_head = self.entry(idx).access;
            return Ok(idx);
        }
        let idx = self.next_fresh as usize;
        if idx >= MAX_HANDLES {
            return Err(ObError::TooManyHandles);
        }
        if idx / PAGE_ENTRIES >= self.pages.len() {
            self.pages.push(vec![EMPTY; PAGE_ENTRIES].into_boxed_slice());
        }
        self.next_fresh += 1;
        Ok(idx)
    }

    fn free(&mut self, idx: usize) {
        let head = self.free_head;
        *self.entry(idx) = Entry { object: 0, access: head, attrs: 0 };
        self.free_head = idx as u32;
        self.count -= 1;
    }
}

/// What a handle looks like from outside.
#[derive(Clone, Copy, Debug)]
pub struct HandleInfo {
    pub access: AccessMask,
    pub attributes: u32,
    pub type_name: &'static str,
    pub handle_count: usize,
    pub pointer_count: usize,
}

/// How [`HandleTable::duplicate`] behaves.
#[derive(Clone, Copy)]
pub struct DuplicateOptions {
    /// Rights for the new handle; `None` = the same as the source. Must be a
    /// subset of the source's rights.
    pub access: Option<AccessMask>,
    pub attributes: u32,
    /// Close the source handle afterwards (a *move*).
    pub close_source: bool,
}

impl DuplicateOptions {
    pub const fn same_access() -> Self {
        Self { access: None, attributes: 0, close_source: false }
    }
}

pub struct HandleTable {
    inner: IrqLock<Inner>,
}

impl Default for HandleTable {
    fn default() -> Self {
        Self::new()
    }
}

impl HandleTable {
    pub fn new() -> Self {
        Self { inner: IrqLock::new(Inner { pages: Vec::new(), next_fresh: 1, free_head: NO_FREE, count: 0 }) }
    }

    /// Open handles.
    pub fn len(&self) -> usize {
        self.inner.lock().count
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Opens a handle to `obj` with `access` (generic rights are mapped through
    /// the type; `MAXIMUM_ALLOWED` means every right the type defines).
    pub fn create_handle(&self, obj: &ObRef, access: AccessMask, attributes: u32) -> Result<Handle, ObError> {
        let ty = obj.object_type();
        let granted = if access & MAXIMUM_ALLOWED != 0 { ty.valid_access } else { ty.generic.map(access) };
        if granted & !ty.valid_access != 0 {
            return Err(ObError::InvalidAccess);
        }
        let mut t = self.inner.lock();
        let idx = t.allocate()?;
        // The handle owns a pointer reference and a handle count.
        obj.header().add_pointer();
        obj.header().add_handle();
        ty.note_handle_opened();
        *t.entry(idx) = Entry { object: obj.as_ptr() as usize, access: granted, attrs: attributes };
        t.count += 1;
        Ok(Handle(idx * 4))
    }

    /// Closes `h`. Fails for a protected handle.
    pub fn close(&self, h: Handle) -> Result<(), ObError> {
        let obj = {
            let mut t = self.inner.lock();
            let idx = t.slot_for(h)?;
            let e = *t.entry(idx);
            if e.attrs & HANDLE_PROTECT_FROM_CLOSE != 0 {
                return Err(ObError::HandleProtected);
            }
            t.free(idx);
            e.object
        };
        // SAFETY: the entry owned this reference; it is now ours to release.
        unsafe { release_handle(obj as *mut ObjectHeader) };
        Ok(())
    }

    /// `ObReferenceObjectByHandle`: resolves `h` to a new reference, checking that
    /// the handle grants `desired` and (if given) that the object has type `expect`.
    /// Returns the object and the access the handle holds.
    pub fn reference_by_handle(
        &self,
        h: Handle,
        desired: AccessMask,
        expect: Option<&'static ObjectType>,
    ) -> Result<(ObRef, AccessMask), ObError> {
        let mut t = self.inner.lock();
        let idx = t.slot_for(h)?;
        let e = *t.entry(idx);
        let p = e.object as *mut ObjectHeader;
        // SAFETY: the table's reference keeps the object alive while we hold the lock.
        let hdr = unsafe { &*p };
        let ty = hdr.object_type();
        if let Some(want) = expect {
            if !core::ptr::eq(ty, want) {
                return Err(ObError::TypeMismatch);
            }
        }
        let need = if desired & MAXIMUM_ALLOWED != 0 { 0 } else { ty.generic.map(desired) };
        if need & !e.access != 0 {
            return Err(ObError::AccessDenied);
        }
        hdr.add_pointer();
        // SAFETY: we just took the reference we hand over.
        Ok((unsafe { ObRef::from_raw_owned(p) }, e.access))
    }

    /// Changes the attributes (protect-from-close, inherit) of `h`.
    pub fn set_attributes(&self, h: Handle, attributes: u32) -> Result<(), ObError> {
        let mut t = self.inner.lock();
        let idx = t.slot_for(h)?;
        t.entry(idx).attrs = attributes;
        Ok(())
    }

    pub fn info(&self, h: Handle) -> Result<HandleInfo, ObError> {
        let mut t = self.inner.lock();
        let idx = t.slot_for(h)?;
        let e = *t.entry(idx);
        // SAFETY: referenced by the entry while the lock is held.
        let hdr = unsafe { &*(e.object as *const ObjectHeader) };
        Ok(HandleInfo {
            access: e.access,
            attributes: e.attrs,
            type_name: hdr.object_type().name,
            handle_count: hdr.handle_count(),
            pointer_count: hdr.pointer_count(),
        })
    }

    /// `ObDuplicateObject`: opens a handle to the same object in `dst`.
    pub fn duplicate(&self, h: Handle, dst: &HandleTable, opts: DuplicateOptions) -> Result<Handle, ObError> {
        let (obj, granted) = self.reference_by_handle(h, MAXIMUM_ALLOWED, None)?;
        let access = match opts.access {
            None => granted,
            Some(a) => {
                let want = obj.object_type().generic.map(a);
                if want & !granted != 0 {
                    return Err(ObError::AccessDenied);
                }
                want
            }
        };
        let new = dst.create_handle(&obj, access, opts.attributes)?;
        if opts.close_source {
            let mut t = self.inner.lock();
            if let Ok(idx) = t.slot_for(h) {
                t.entry(idx).attrs &= !HANDLE_PROTECT_FROM_CLOSE;
            }
            drop(t);
            self.close(h)?;
        }
        Ok(new)
    }

    /// Calls `f(handle, object, access, attributes)` for every open handle (a snapshot).
    pub fn for_each(&self, mut f: impl FnMut(Handle, &ObRef, AccessMask, u32)) {
        let items: Vec<(usize, Entry)> = {
            let mut t = self.inner.lock();
            let mut v = Vec::new();
            for idx in 1..t.next_fresh as usize {
                let e = *t.entry(idx);
                if e.object != 0 {
                    // SAFETY: referenced by the entry under the lock.
                    unsafe { (*(e.object as *const ObjectHeader)).add_pointer() };
                    v.push((idx, e));
                }
            }
            v
        };
        for (idx, e) in items {
            // SAFETY: we took a reference above and hand it to the ObRef.
            let obj = unsafe { ObRef::from_raw_owned(e.object as *mut ObjectHeader) };
            f(Handle(idx * 4), &obj, e.access, e.attrs);
        }
    }
}

impl HandleTable {
    /// Closes every open handle (protected ones too), e.g. when a process ends.
    /// Returns how many were closed. The table stays usable.
    pub fn close_all(&self) -> usize {
        let objects: Vec<usize> = {
            let mut t = self.inner.lock();
            let mut v = Vec::new();
            for idx in 1..t.next_fresh as usize {
                let e = *t.entry(idx);
                if e.object != 0 {
                    v.push(e.object);
                    t.free(idx);
                }
            }
            v
        };
        for o in &objects {
            // SAFETY: each open entry owned a reference and a handle count.
            unsafe { release_handle(*o as *mut ObjectHeader) };
        }
        objects.len()
    }
}

impl Drop for HandleTable {
    fn drop(&mut self) {
        self.close_all();
    }
}
