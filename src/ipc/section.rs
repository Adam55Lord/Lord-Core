// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! The `Section` object: shared memory.
//!
//! A section owns a fixed set of physical frames (taken zeroed from the PMM,
//! returned when the object is destroyed). Any number of processes can map it
//! as a **view**; every view shows the same frames, so a write by one process
//! is visible to the others at once. Sections are named, duplicated and passed
//! between processes like every other object (typically attached to a port
//! message), and rights decide who may map it and whether writable.
//!
//! # Lifetime
//!
//! The mapped pages are *not owned* by the address space (`map_phys`), so the
//! VMM never frees them when a view goes away. Instead a view holds a
//! **reference to the section**, recorded in a per-process registry:
//!
//! * `unmap_view` removes the mapping, then drops the reference;
//! * when a process ends, `process_ended` is called by the Process Manager
//!   *after* the address space has been destroyed, and drops every reference.
//!
//! So the frames outlive every mapping of them, and a dying process can never
//! leave its section pinned.
//!
//! # Limits
//!
//! Fixed size (no growing), 4 KiB pages, views are `READ` or `READ_WRITE`
//! (never executable: W^X is not negotiable), no per-process quota.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

use super::{section_type, IpcError, SECTION_TYPE};
use crate::hal::memory::phys_to_virt;
use crate::hal::sync::IrqLock;
use crate::mm::{pmm, vmm};
use crate::ob::object::{create, ObRef};
use crate::ob::{Obj, ObjectAttributes};
use crate::ps::structs::{Pid, Process};

/// Largest section: 4 MiB.
pub const MAX_SECTION_PAGES: u64 = 1024;
/// Where the kernel places a view when the caller does not choose an address.
pub const VIEW_BASE: u64 = 0x0000_1000_0000_0000;

const PAGE: u64 = vmm::PAGE_SIZE;

pub struct Section {
    pages: u64,
    /// Physical addresses of the owned frames, in order.
    frames: Vec<u64>,
}

pub(super) static VIEWS_MAPPED: AtomicU64 = AtomicU64::new(0);

impl Section {
    pub fn pages(&self) -> u64 {
        self.pages
    }

    pub fn size(&self) -> u64 {
        self.pages * PAGE
    }

    fn check(&self, offset: u64, len: usize) -> Result<(), IpcError> {
        match offset.checked_add(len as u64) {
            Some(end) if end <= self.size() => Ok(()),
            _ => Err(IpcError::InvalidArgument),
        }
    }

    /// Copies out of the section through the kernel's own mapping.
    pub fn read(&self, offset: u64, out: &mut [u8]) -> Result<(), IpcError> {
        self.check(offset, out.len())?;
        let mut done = 0usize;
        while done < out.len() {
            let at = offset + done as u64;
            let n = (out.len() - done).min((PAGE - at % PAGE) as usize);
            let frame = self.frames[(at / PAGE) as usize];
            // SAFETY: `frame` is one of our owned frames, reachable through the HHDM; `n` stays in the page.
            unsafe { core::ptr::copy_nonoverlapping(phys_to_virt(frame + at % PAGE), out[done..].as_mut_ptr(), n) };
            done += n;
        }
        Ok(())
    }

    /// Copies into the section through the kernel's own mapping.
    pub fn write(&self, offset: u64, data: &[u8]) -> Result<(), IpcError> {
        self.check(offset, data.len())?;
        let mut done = 0usize;
        while done < data.len() {
            let at = offset + done as u64;
            let n = (data.len() - done).min((PAGE - at % PAGE) as usize);
            let frame = self.frames[(at / PAGE) as usize];
            // SAFETY: as in `read`.
            unsafe { core::ptr::copy_nonoverlapping(data[done..].as_ptr(), phys_to_virt(frame + at % PAGE), n) };
            done += n;
        }
        Ok(())
    }
}

impl Drop for Section {
    fn drop(&mut self) {
        for f in self.frames.drain(..) {
            let _ = pmm::free_frame(f);
        }
    }
}

/// Creates a section of `pages` zeroed pages.
pub fn create_section(attrs: &ObjectAttributes<'_>, pages: u64) -> Result<Obj<Section>, IpcError> {
    if pages == 0 || pages > MAX_SECTION_PAGES {
        return Err(IpcError::InvalidArgument);
    }
    let mut frames = Vec::with_capacity(pages as usize);
    for _ in 0..pages {
        match pmm::alloc_zeroed() {
            Some(f) => frames.push(f),
            None => {
                for f in frames {
                    let _ = pmm::free_frame(f);
                }
                return Err(IpcError::OutOfResources);
            }
        }
    }
    let sec = Section { pages, frames };
    let ty = SECTION_TYPE.load(Ordering::Acquire);
    if ty.is_null() {
        return Err(IpcError::NotReady);
    }
    // SAFETY: set once from an immortal type object. If naming fails, `create`
    // drops the section and its frames go back.
    Ok(create(unsafe { &*ty }, attrs, sec)?)
}

// ----------------------------------------------------------------------
// Views
// ----------------------------------------------------------------------

struct View {
    va: u64,
    pages: u64,
    /// Keeps the frames alive for as long as the mapping exists.
    section: ObRef,
}

struct ProcessViews {
    views: Vec<View>,
    next_va: u64,
}

static VIEWS: IrqLock<BTreeMap<Pid, ProcessViews>> = IrqLock::new(BTreeMap::new());

/// Number of views `pid` has mapped.
pub fn view_count(pid: Pid) -> usize {
    VIEWS.lock().get(&pid).map_or(0, |p| p.views.len())
}

/// Maps `section` into `process` and returns the address. `va == 0` lets the
/// kernel choose; otherwise it must be page aligned, in user space and free.
pub fn map_view(process: &Obj<Process>, section: &ObRef, va: u64, write: bool) -> Result<u64, IpcError> {
    let body = section.body::<Section>(section_type()).ok_or(IpcError::TypeMismatch)?;
    let pages = body.pages;
    let prot = if write { vmm::Prot::READ_WRITE.user() } else { vmm::Prot::READ.user() };
    if va != 0 && (va % PAGE != 0 || va.checked_add(pages * PAGE).is_none_or(|e| e > vmm::USER_END)) {
        return Err(IpcError::InvalidArgument);
    }

    let mut reg = VIEWS.lock();
    let pv = reg.entry(process.pid).or_insert(ProcessViews { views: Vec::new(), next_va: VIEW_BASE });
    let base = if va != 0 { va } else { pv.next_va };
    let mapped = process
        .with_space(|space| {
            for (i, frame) in body.frames.iter().enumerate() {
                if let Err(e) = space.map_phys(base + i as u64 * PAGE, *frame, 1, prot) {
                    let _ = space.unmap(base, i as u64); // not owned: no frame is freed
                    return Err(e);
                }
            }
            Ok(())
        })
        .ok_or(IpcError::NoProcess)?;
    if let Err(e) = mapped {
        let empty = pv.views.is_empty();
        if empty {
            reg.remove(&process.pid);
        }
        return Err(match e {
            vmm::VmError::AlreadyMapped => IpcError::InvalidArgument,
            vmm::VmError::OutOfFrames => IpcError::OutOfResources,
            _ => IpcError::InvalidArgument,
        });
    }
    pv.views.push(View { va: base, pages, section: section.clone() });
    if va == 0 {
        pv.next_va = base + (pages + 1) * PAGE; // one unmapped page between views
    }
    VIEWS_MAPPED.fetch_add(1, Ordering::Relaxed);
    Ok(base)
}

/// Removes the view that starts at `va`.
pub fn unmap_view(process: &Obj<Process>, va: u64) -> Result<(), IpcError> {
    let view = {
        let mut reg = VIEWS.lock();
        let pv = reg.get_mut(&process.pid).ok_or(IpcError::NotFound)?;
        let i = pv.views.iter().position(|v| v.va == va).ok_or(IpcError::NotFound)?;
        let view = pv.views.remove(i);
        if pv.views.is_empty() {
            reg.remove(&process.pid);
        }
        view
    };
    process.with_space(|space| {
        let _ = space.unmap(view.va, view.pages);
    });
    drop(view); // the section may be destroyed now: nothing maps it any more
    Ok(())
}

/// Called by the Process Manager when a process has ended and its address
/// space is gone: drops the references its views held.
pub fn process_ended(pid: Pid) {
    let pv = VIEWS.lock().remove(&pid);
    drop(pv);
}
