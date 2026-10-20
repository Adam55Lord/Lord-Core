// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! The built-in object types.
//!
//! | Type | Body | Purpose |
//! |---|---|---|
//! | `Type` | [`ObjectType`] | describes a kind of object; itself an object (of type `Type`) |
//! | `Directory` | [`Directory`] | namespace node |
//! | `SymbolicLink` | [`SymbolicLink`] | redirects to an absolute path |
//! | `Event` | [`Event`] | a signalable flag (notification or auto-reset) |
//! | `Device` | [`Device`] | a piece of hardware the kernel drives |
//! | `Processor` | [`Processor`] | a logical CPU |
//! | `AddressSpace` | [`AddressSpaceObject`] | a `vmm::AddressSpace` (the future process address space) |
//!
//! Each has a `create_*` function; the types are created once, by [`init`].

use alloc::string::{String, ToString};
use core::ptr;
use core::sync::atomic::{AtomicBool, AtomicPtr, Ordering};

use super::access::*;
use super::namespace::{self, Directory};
use super::object::{self, create, Obj, ObjectType, TypeInit};
use super::{ObError, ObjectAttributes};
use crate::hal::sync::IrqLock;
use crate::mm::vmm;

macro_rules! type_slot {
    ($slot:ident, $getter:ident) => {
        static $slot: AtomicPtr<ObjectType> = AtomicPtr::new(ptr::null_mut());
        pub fn $getter() -> &'static ObjectType {
            let p = $slot.load(Ordering::Acquire);
            assert!(!p.is_null(), "object manager not initialised");
            // SAFETY: set once from an immortal type object.
            unsafe { &*p }
        }
    };
}

type_slot!(TYPE_TYPE, type_type);
type_slot!(DIRECTORY_TYPE, directory_type);
type_slot!(SYMLINK_TYPE, symlink_type);
type_slot!(EVENT_TYPE, event_type);
type_slot!(DEVICE_TYPE, device_type);
type_slot!(PROCESSOR_TYPE, processor_type);
type_slot!(ADDRESS_SPACE_TYPE, address_space_type);

/// Is `obj` a symbolic link? False (not a panic) before the type exists, so
/// path walks work while the type system is still being built.
pub fn is_symlink(obj: &super::ObRef) -> bool {
    let p = SYMLINK_TYPE.load(Ordering::Acquire);
    !p.is_null() && ptr::eq(obj.object_type(), p)
}

fn publish(slot: &AtomicPtr<ObjectType>, ty: &'static ObjectType) {
    slot.store(ty as *const ObjectType as *mut ObjectType, Ordering::Release);
}

const fn mapping(read: u32, write: u32, execute: u32, all: u32) -> GenericMapping {
    GenericMapping { read, write, execute, all }
}

/// Builds the type system and the top of the namespace.
pub(super) fn init() -> Result<(), ObError> {
    // 1. Types exist before any namespace does: `Type` is its own type.
    let tt = object::bootstrap_type_type(TypeInit::of::<ObjectType>(
        "Type",
        TYPE_ALL,
        mapping(READ_CONTROL | TYPE_QUERY, READ_CONTROL, READ_CONTROL, TYPE_ALL),
    ));
    publish(&TYPE_TYPE, tt);
    let dt = object::create_type(TypeInit::of::<Directory>(
        "Directory",
        DIRECTORY_ALL,
        mapping(
            READ_CONTROL | DIRECTORY_QUERY | DIRECTORY_TRAVERSE,
            READ_CONTROL | DIRECTORY_CREATE_OBJECT | DIRECTORY_CREATE_SUBDIRECTORY,
            READ_CONTROL | DIRECTORY_TRAVERSE,
            DIRECTORY_ALL,
        ),
    ))?;
    publish(&DIRECTORY_TYPE, dt);

    // 2. The namespace: `\` and `\ObjectTypes`, then name the types made so far.
    let root = create(dt, &ObjectAttributes::unnamed().permanent(), Directory::new())?;
    namespace::set_root(root.untyped());
    core::mem::forget(root); // immortal
    let ot = create(dt, &ObjectAttributes::named("\\ObjectTypes").permanent(), Directory::new())?;
    drop(ot);
    for h in object::type_headers() {
        // SAFETY: types are immortal; take a reference for the temporary ObRef.
        let r = unsafe {
            (*(h as *const object::ObjectHeader)).add_pointer();
            object::ObRef::from_raw_owned(h as *mut object::ObjectHeader)
        };
        let path = alloc::format!("\\ObjectTypes\\{}", r.body::<ObjectType>(tt).expect("type object").name);
        namespace::insert_named(&ObjectAttributes::named(&path), &r)?;
    }

    // 3. The remaining types (named automatically now).
    publish(
        &SYMLINK_TYPE,
        object::create_type(TypeInit::of::<SymbolicLink>(
            "SymbolicLink",
            SYMLINK_ALL,
            mapping(READ_CONTROL | SYMLINK_QUERY, READ_CONTROL, READ_CONTROL, SYMLINK_ALL),
        ))?,
    );
    publish(
        &EVENT_TYPE,
        object::create_type(TypeInit::of::<Event>(
            "Event",
            EVENT_ALL,
            mapping(
                READ_CONTROL | EVENT_QUERY_STATE,
                READ_CONTROL | EVENT_MODIFY_STATE,
                READ_CONTROL | SYNCHRONIZE,
                EVENT_ALL,
            ),
        ))?,
    );
    publish(
        &DEVICE_TYPE,
        object::create_type(TypeInit::of::<Device>(
            "Device",
            DEVICE_ALL,
            mapping(
                READ_CONTROL | DEVICE_QUERY | DEVICE_READ,
                READ_CONTROL | DEVICE_WRITE,
                READ_CONTROL | SYNCHRONIZE,
                DEVICE_ALL,
            ),
        ))?,
    );
    publish(
        &PROCESSOR_TYPE,
        object::create_type(TypeInit::of::<Processor>(
            "Processor",
            PROCESSOR_ALL,
            mapping(READ_CONTROL | PROCESSOR_QUERY, READ_CONTROL, READ_CONTROL, PROCESSOR_ALL),
        ))?,
    );
    publish(
        &ADDRESS_SPACE_TYPE,
        object::create_type(TypeInit::of::<AddressSpaceObject>(
            "AddressSpace",
            ADDRESS_SPACE_ALL,
            mapping(
                READ_CONTROL | ADDRESS_SPACE_QUERY,
                READ_CONTROL | ADDRESS_SPACE_MAP,
                READ_CONTROL,
                ADDRESS_SPACE_ALL,
            ),
        ))?,
    );

    // 4. The standard directories every other subsystem registers under.
    for dir in ["\\Device", "\\Processors", "\\KernelObjects"] {
        let d = create(dt, &ObjectAttributes::named(dir).permanent(), Directory::new())?;
        drop(d);
    }
    Ok(())
}

// ----------------------------------------------------------------------
// SymbolicLink
// ----------------------------------------------------------------------

pub struct SymbolicLink {
    /// Absolute path this link stands for.
    pub target: String,
}

pub fn create_directory(attrs: &ObjectAttributes<'_>) -> Result<Obj<Directory>, ObError> {
    create(directory_type(), attrs, Directory::new())
}

/// `target` must be an absolute path (`\...`); it is resolved when followed, not now.
pub fn create_symlink(attrs: &ObjectAttributes<'_>, target: &str) -> Result<Obj<SymbolicLink>, ObError> {
    if !target.starts_with('\\') {
        return Err(ObError::InvalidName);
    }
    create(symlink_type(), attrs, SymbolicLink { target: target.to_string() })
}

// ----------------------------------------------------------------------
// Event
// ----------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventKind {
    /// Stays signaled until reset.
    Notification,
    /// Consumed by the first successful `try_wait`.
    Synchronization,
}

pub struct Event {
    pub kind: EventKind,
    signaled: AtomicBool,
}

impl Event {
    /// Signals the event; returns the previous state.
    pub fn set(&self) -> bool {
        let was = self.signaled.swap(true, Ordering::AcqRel);
        // Threads blocked on this event become ready (no-op before the scheduler runs).
        crate::ps::event_signaled(self);
        was
    }
    pub fn reset(&self) -> bool {
        self.signaled.swap(false, Ordering::AcqRel)
    }
    pub fn is_signaled(&self) -> bool {
        self.signaled.load(Ordering::Acquire)
    }
    /// Non-blocking wait (there is no scheduler yet): true if the event was
    /// signaled; a synchronization event is consumed.
    pub fn try_wait(&self) -> bool {
        match self.kind {
            EventKind::Notification => self.is_signaled(),
            EventKind::Synchronization => {
                self.signaled.compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire).is_ok()
            }
        }
    }
}

pub fn create_event(attrs: &ObjectAttributes<'_>, kind: EventKind, signaled: bool) -> Result<Obj<Event>, ObError> {
    create(event_type(), attrs, Event { kind, signaled: AtomicBool::new(signaled) })
}

// ----------------------------------------------------------------------
// Device
// ----------------------------------------------------------------------

/// A device the kernel drives. `base`/`size` describe its registers or buffer.
pub struct Device {
    pub class: &'static str,
    pub description: &'static str,
    pub base: u64,
    pub size: u64,
    /// `base` is an I/O port, not a physical memory address.
    pub io_port: bool,
}

pub fn create_device(attrs: &ObjectAttributes<'_>, device: Device) -> Result<Obj<Device>, ObError> {
    create(device_type(), attrs, device)
}

// ----------------------------------------------------------------------
// Processor
// ----------------------------------------------------------------------

pub struct Processor {
    pub index: usize,
    pub lapic_id: u32,
    pub bsp: bool,
}

pub fn create_processor(attrs: &ObjectAttributes<'_>, cpu: Processor) -> Result<Obj<Processor>, ObError> {
    create(processor_type(), attrs, cpu)
}

// ----------------------------------------------------------------------
// AddressSpace
// ----------------------------------------------------------------------

/// A virtual address space as an object. Destroying the last reference
/// destroys the space (page tables and owned frames go back to the PMM).
pub struct AddressSpaceObject {
    space: IrqLock<vmm::AddressSpace>,
}

impl AddressSpaceObject {
    pub fn root(&self) -> u64 {
        self.space.lock().root()
    }
    pub fn map_new(&self, va: u64, pages: u64, prot: vmm::Prot) -> Result<(), vmm::VmError> {
        self.space.lock().map_new(va, pages, prot)
    }
    pub fn unmap(&self, va: u64, pages: u64) -> Result<u64, vmm::VmError> {
        self.space.lock().unmap(va, pages)
    }
    pub fn translate(&self, va: u64) -> Option<u64> {
        self.space.lock().translate(va)
    }
}

pub fn create_address_space(attrs: &ObjectAttributes<'_>) -> Result<Obj<AddressSpaceObject>, ObError> {
    let space = vmm::AddressSpace::new().map_err(|_| ObError::OutOfResources)?;
    create(address_space_type(), attrs, AddressSpaceObject { space: IrqLock::new(space) })
}
