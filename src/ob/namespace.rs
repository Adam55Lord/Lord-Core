// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! The object namespace: one tree of directories, rooted at `\`.
//!
//! * Names are **case-insensitive** (stored as given, compared in ASCII lower case).
//! * A path is `\Dir\Dir\Name`; components are 1..=255 bytes, no `\`, no control characters.
//! * A **symbolic link** redirects to an absolute path; intermediate links are
//!   always followed, a final one only on request; chains are limited to
//!   [`MAX_LINK_DEPTH`].
//! * A directory entry does **not** own its object. A named object whose last
//!   handle closes loses its name unless it is *permanent*; either way the
//!   name disappears when the object is destroyed. A named object holds a
//!   reference on its parent directory, so a directory outlives its children.
//!
//! All directories share one lock (`NS`), which also protects every object's
//! name information.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::ptr;
use core::sync::atomic::{AtomicBool, AtomicPtr, Ordering};

use super::object::{dereference, ObRef, ObjectHeader};
use super::types::{self, SymbolicLink};
use super::{ObError, ObjectAttributes, ObjectType};
use crate::hal::sync::{IrqLock, IrqLockGuard};

pub const MAX_LINK_DEPTH: u32 = 8;
pub const MAX_COMPONENT: usize = 255;

/// Where an object lives: its leaf name and its (referenced) parent directory.
pub(super) struct NameInfo {
    pub name: String,
    pub parent: *mut ObjectHeader,
}

static NS: IrqLock<()> = IrqLock::new(());
static ROOT: AtomicPtr<ObjectHeader> = AtomicPtr::new(ptr::null_mut());
static READY: AtomicBool = AtomicBool::new(false);

/// Body of a `Directory` object.
pub struct Directory {
    /// lower-case name -> header address; only touched with `NS` held.
    map: UnsafeCell<BTreeMap<String, usize>>,
}

// SAFETY: the map is only accessed while holding the namespace lock.
unsafe impl Sync for Directory {}
// SAFETY: as above.
unsafe impl Send for Directory {}

impl Directory {
    pub fn new() -> Self {
        Self { map: UnsafeCell::new(BTreeMap::new()) }
    }

    #[allow(clippy::mut_from_ref)]
    fn map(&self, _held: &IrqLockGuard<'_, ()>) -> &mut BTreeMap<String, usize> {
        // SAFETY: the namespace lock is held (witnessed by the guard).
        unsafe { &mut *self.map.get() }
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        let g = NS.lock();
        self.map(&g).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for Directory {
    fn default() -> Self {
        Self::new()
    }
}

pub fn is_ready() -> bool {
    READY.load(Ordering::Acquire)
}

pub(super) fn set_root(root: &ObRef) {
    ROOT.store(root.as_ptr(), Ordering::Release);
    READY.store(true, Ordering::Release);
}

/// A new reference to the root directory `\`.
pub fn root() -> ObRef {
    let h = ROOT.load(Ordering::Acquire);
    assert!(!h.is_null(), "object namespace not initialised");
    // SAFETY: the root is permanent and never freed.
    unsafe {
        (*h).add_pointer();
        ObRef::from_raw_owned(h)
    }
}

// ----------------------------------------------------------------------
// Paths
// ----------------------------------------------------------------------

fn valid_component(c: &str) -> bool {
    !c.is_empty() && c.len() <= MAX_COMPONENT && c != "." && c != ".." && c.bytes().all(|b| b >= 0x20 && b != b'\\' && b != 0x7f)
}

/// Splits `a\b\c` (no leading backslash) into validated components.
fn split(rest: &str) -> Result<Vec<&str>, ObError> {
    if rest.is_empty() {
        return Ok(Vec::new());
    }
    let comps: Vec<&str> = rest.split('\\').collect();
    if comps.iter().all(|c| valid_component(c)) { Ok(comps) } else { Err(ObError::InvalidName) }
}

fn start_and_rest<'a>(name: &'a str, root_dir: Option<&ObRef>) -> Result<(ObRef, &'a str), ObError> {
    if let Some(rest) = name.strip_prefix('\\') {
        Ok((root(), rest))
    } else if let Some(r) = root_dir {
        Ok((r.clone(), name))
    } else {
        Err(ObError::InvalidName)
    }
}

fn lookup_child(parent: &ObRef, comp: &str) -> Result<ObRef, ObError> {
    let key = comp.to_ascii_lowercase();
    let g = NS.lock();
    let dir = parent.body::<Directory>(types::directory_type()).ok_or(ObError::NotADirectory)?;
    let p = *dir.map(&g).get(&key).ok_or(ObError::NameNotFound)? as *mut ObjectHeader;
    // SAFETY: entries are removed (under this lock) before their object is freed.
    let h = unsafe { &*p };
    if !h.try_add_pointer() {
        return Err(ObError::NameNotFound); // being destroyed right now
    }
    // SAFETY: we just took a reference for the new ObRef.
    Ok(unsafe { ObRef::from_raw_owned(p) })
}

fn walk(start: ObRef, comps: &[&str], follow_final: bool, depth: u32) -> Result<ObRef, ObError> {
    let mut cur = start;
    for (i, c) in comps.iter().enumerate() {
        let last = i + 1 == comps.len();
        let next = match lookup_child(&cur, c) {
            Err(ObError::NameNotFound) if !last => return Err(ObError::PathNotFound),
            other => other?,
        };
        cur = if types::is_symlink(&next) && (!last || follow_final) {
            if depth >= MAX_LINK_DEPTH {
                return Err(ObError::LinkLoop);
            }
            let target = next.body::<SymbolicLink>(types::symlink_type()).expect("type checked").target.clone();
            let rest = target.strip_prefix('\\').ok_or(ObError::InvalidName)?;
            let comps = split(rest)?;
            walk(root(), &comps, true, depth + 1)?
        } else {
            next
        };
    }
    Ok(cur)
}

/// Finds the object at absolute `path`. If `expect` is given the object must be
/// of that type. A final symbolic link is followed only if `follow_final`.
pub fn lookup(path: &str, expect: Option<&'static ObjectType>, follow_final: bool) -> Result<ObRef, ObError> {
    let rest = path.strip_prefix('\\').ok_or(ObError::InvalidName)?;
    let comps = split(rest)?;
    let obj = walk(root(), &comps, follow_final, 0)?;
    match expect {
        Some(t) if !obj.is_type(t) => Err(ObError::TypeMismatch),
        _ => Ok(obj),
    }
}

/// Like [`lookup`], relative to `dir`.
pub fn lookup_in(dir: &ObRef, relative: &str, expect: Option<&'static ObjectType>) -> Result<ObRef, ObError> {
    let comps = split(relative)?;
    let obj = walk(dir.clone(), &comps, true, 0)?;
    match expect {
        Some(t) if !obj.is_type(t) => Err(ObError::TypeMismatch),
        _ => Ok(obj),
    }
}

// ----------------------------------------------------------------------
// Insert / unlink
// ----------------------------------------------------------------------

/// Gives `obj` the name in `attrs`. The parent directory must exist.
pub(super) fn insert_named(attrs: &ObjectAttributes<'_>, obj: &ObRef) -> Result<(), ObError> {
    let name = attrs.name.ok_or(ObError::InvalidName)?;
    let (start, rest) = start_and_rest(name, attrs.root)?;
    let comps = split(rest)?;
    let (leaf, parents) = comps.split_last().ok_or(ObError::InvalidName)?;
    let parent = walk(start, parents, true, 0).map_err(|e| if e == ObError::NameNotFound { ObError::PathNotFound } else { e })?;
    let key = leaf.to_ascii_lowercase();

    let g = NS.lock();
    let dir = parent.body::<Directory>(types::directory_type()).ok_or(ObError::NotADirectory)?;
    // SAFETY: the namespace lock is held.
    let slot = unsafe { &mut *obj.header().name.get() };
    if !slot.is_null() {
        return Err(ObError::InvalidName); // already named
    }
    let map = dir.map(&g);
    if map.contains_key(&key) {
        return Err(ObError::NameCollision);
    }
    parent.header().add_pointer(); // owned by the NameInfo below
    *slot = Box::into_raw(Box::new(NameInfo { name: leaf.to_string(), parent: parent.as_ptr() }));
    map.insert(key, obj.as_ptr() as usize);
    Ok(())
}

/// Removes `h`'s name (if it has one) and releases its parent directory.
///
/// # Safety
/// `h` must be a live object (the caller holds a reference, or is destroying it).
pub(super) unsafe fn unlink(h: *mut ObjectHeader) {
    let ni = {
        let g = NS.lock();
        // SAFETY: valid object; the name slot is protected by the namespace lock.
        let slot = unsafe { &mut *(*h).name.get() };
        let ni = *slot;
        if ni.is_null() {
            return;
        }
        *slot = ptr::null_mut();
        // SAFETY: a named object's NameInfo and parent are valid while linked.
        let (parent, key) = unsafe { ((*ni).parent, (*ni).name.to_ascii_lowercase()) };
        // SAFETY: the parent is referenced by this NameInfo, so alive and a Directory.
        let dir = unsafe { &*((*parent).body_ptr() as *const Directory) };
        let map = dir.map(&g);
        if map.get(&key).copied() == Some(h as usize) {
            map.remove(&key);
        }
        ni
    };
    // SAFETY: we took the pointer out of the slot under the lock: ours alone now.
    let ni = unsafe { Box::from_raw(ni) };
    let parent = ni.parent;
    drop(ni);
    // SAFETY: gives up the reference the NameInfo held on the parent (outside the lock).
    unsafe { dereference(parent) };
}

// ----------------------------------------------------------------------
// Inspection
// ----------------------------------------------------------------------

/// Full path (`\A\B`) of a named object.
pub fn path_of(obj: &ObRef) -> Option<String> {
    let _g = NS.lock();
    let mut parts: Vec<String> = Vec::new();
    let mut cur = obj.as_ptr();
    loop {
        // SAFETY: the namespace lock is held; linked objects and their parents are alive.
        let ni = unsafe { *(*cur).name.get() };
        if ni.is_null() {
            break;
        }
        // SAFETY: non-null NameInfo of a linked object.
        unsafe {
            parts.push((*ni).name.clone());
            cur = (*ni).parent;
        }
    }
    if parts.is_empty() {
        return if cur == ROOT.load(Ordering::Acquire) && cur == obj.as_ptr() { Some(String::from("\\")) } else { None };
    }
    let mut s = String::new();
    for p in parts.iter().rev() {
        s.push('\\');
        s.push_str(p);
    }
    Some(s)
}

/// One line of [`snapshot`].
#[derive(Clone, Debug)]
pub struct NsEntry {
    pub depth: usize,
    pub name: String,
    pub type_name: &'static str,
    pub handles: usize,
    pub pointers: usize,
    pub permanent: bool,
    pub link_target: Option<String>,
}

/// The whole namespace, depth-first, children in name order.
pub fn snapshot() -> Vec<NsEntry> {
    let mut out = Vec::new();
    let g = NS.lock();
    let root = ROOT.load(Ordering::Acquire);
    if !root.is_null() {
        visit(&g, root, 0, &mut out);
    }
    out
}

fn visit(g: &IrqLockGuard<'_, ()>, dir: *mut ObjectHeader, depth: usize, out: &mut Vec<NsEntry>) {
    // SAFETY: lock held; `dir` is a live directory.
    let d = unsafe { &*((*dir).body_ptr() as *const Directory) };
    let entries: Vec<usize> = d.map(g).values().copied().collect();
    for p in entries {
        let h = p as *mut ObjectHeader;
        // SAFETY: linked objects are alive while the lock is held.
        let hdr = unsafe { &*h };
        let ty = hdr.object_type();
        // SAFETY: linked, so the NameInfo is valid.
        let name = unsafe { (*(*hdr.name.get())).name.clone() };
        let link_target = if ptr::eq(ty, types::symlink_type()) {
            // SAFETY: type checked.
            Some(unsafe { &*(hdr.body_ptr() as *const SymbolicLink) }.target.clone())
        } else {
            None
        };
        out.push(NsEntry {
            depth,
            name,
            type_name: ty.name,
            handles: hdr.handle_count(),
            pointers: hdr.pointer_count(),
            permanent: hdr.is_permanent(),
            link_target,
        });
        if ptr::eq(ty, types::directory_type()) {
            visit(g, h, depth + 1, out);
        }
    }
}

/// Names and types of the entries of `dir`, in name order.
pub fn list_directory(dir: &ObRef) -> Result<Vec<(String, &'static str)>, ObError> {
    let g = NS.lock();
    let d = dir.body::<Directory>(types::directory_type()).ok_or(ObError::NotADirectory)?;
    let mut v = Vec::new();
    for &p in d.map(&g).values() {
        // SAFETY: lock held; linked objects are alive.
        let hdr = unsafe { &*(p as *const ObjectHeader) };
        // SAFETY: linked, so the NameInfo is valid.
        let name = unsafe { (*(*hdr.name.get())).name.clone() };
        v.push((name, hdr.object_type().name));
    }
    Ok(v)
}
