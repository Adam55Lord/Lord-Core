// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! `OBJECT_HEADER`, `OBJECT_TYPE` and reference counting.
//!
//! Memory layout of every object (one heap allocation, 16-byte aligned):
//!
//! ```text
//!   +0   ObjectHeader (48 bytes)
//!          pointer_count  handle_count  type  flags  name-info  security
//!   +48  body: the Rust value of the type (Event, Directory, ...)
//! ```
//!
//! The header comes first, so from a body pointer the header is a fixed
//! offset back, and from a header the body a fixed offset forward.

use alloc::alloc::{alloc as heap_alloc, dealloc as heap_dealloc, handle_alloc_error, Layout};
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::mem::{align_of, size_of};
use core::ops::Deref;
use core::ptr::{self, NonNull};
use core::sync::atomic::{AtomicPtr, AtomicU32, AtomicUsize, Ordering};

use super::access::{AccessMask, GenericMapping};
use super::namespace::{self, NameInfo};
use super::{types, ObError, ObjectAttributes};
use crate::hal::sync::IrqLock;

/// The object keeps itself (and its name) alive: one extra pointer reference.
pub const OBJ_PERMANENT: u32 = 1 << 0;
const HEADER_ALIGN: usize = 16;

// ----------------------------------------------------------------------
// Object types
// ----------------------------------------------------------------------

/// What a type needs to be created: see [`TypeInit::of`].
pub struct TypeInit {
    pub name: &'static str,
    pub body_size: usize,
    pub valid_access: AccessMask,
    pub generic: GenericMapping,
    pub delete: unsafe fn(*mut u8),
    pub last_close: Option<fn(*const u8)>,
}

unsafe fn drop_body<T>(body: *mut u8) {
    // SAFETY: the caller passes a pointer to a valid, owned `T`.
    unsafe { ptr::drop_in_place(body as *mut T) }
}

impl TypeInit {
    /// A type whose body is `T`; destroying an object drops the `T`.
    pub fn of<T>(name: &'static str, valid_access: AccessMask, generic: GenericMapping) -> Self {
        assert!(align_of::<T>() <= HEADER_ALIGN, "object bodies are limited to 16-byte alignment");
        Self { name, body_size: size_of::<T>(), valid_access, generic, delete: drop_body::<T>, last_close: None }
    }

    /// Runs `f(body)` when the **last handle** to an object of this type closes
    /// (the object may live on through other references).
    pub fn on_last_close(mut self, f: fn(*const u8)) -> Self {
        self.last_close = Some(f);
        self
    }
}

/// `OBJECT_TYPE`: the body of a `Type` object. One per kind of object.
pub struct ObjectType {
    pub name: &'static str,
    pub body_size: usize,
    /// Rights this type defines; a handle can never carry others.
    pub valid_access: AccessMask,
    pub generic: GenericMapping,
    delete: unsafe fn(*mut u8),
    last_close: Option<fn(*const u8)>,
    pub total_objects: AtomicUsize,
    pub peak_objects: AtomicUsize,
    pub total_handles: AtomicUsize,
    pub peak_handles: AtomicUsize,
}

fn bump(cur: &AtomicUsize, peak: &AtomicUsize) {
    let n = cur.fetch_add(1, Ordering::Relaxed) + 1;
    peak.fetch_max(n, Ordering::Relaxed);
}

impl ObjectType {
    pub fn new(init: TypeInit) -> Self {
        Self {
            name: init.name,
            body_size: init.body_size,
            valid_access: init.valid_access,
            generic: init.generic,
            delete: init.delete,
            last_close: init.last_close,
            total_objects: AtomicUsize::new(0),
            peak_objects: AtomicUsize::new(0),
            total_handles: AtomicUsize::new(0),
            peak_handles: AtomicUsize::new(0),
        }
    }

    pub(super) fn note_handle_opened(&self) {
        bump(&self.total_handles, &self.peak_handles);
    }
    pub(super) fn note_handle_closed(&self) {
        self.total_handles.fetch_sub(1, Ordering::Relaxed);
    }
    pub(super) fn run_last_close(&self, body: *const u8) {
        if let Some(f) = self.last_close {
            f(body);
        }
    }
}

/// Every type ever created (types are immortal), as header addresses.
static TYPES: IrqLock<Vec<usize>> = IrqLock::new(Vec::new());

pub fn type_count() -> usize {
    TYPES.lock().len()
}

/// Calls `f` for every registered type (a snapshot; `f` runs unlocked).
pub fn for_each_type(mut f: impl FnMut(&'static ObjectType)) {
    let list: Vec<usize> = TYPES.lock().clone();
    for h in list {
        // SAFETY: type objects are immortal, so the pointer stays valid.
        let hdr = unsafe { &*(h as *const ObjectHeader) };
        // SAFETY: the body of a Type object is an ObjectType.
        f(unsafe { &*(hdr.body_ptr() as *const ObjectType) });
    }
}

/// Header addresses of all types (for naming them once the namespace exists).
pub(super) fn type_headers() -> Vec<usize> {
    TYPES.lock().clone()
}

// ----------------------------------------------------------------------
// The header
// ----------------------------------------------------------------------

/// `OBJECT_HEADER`.
#[repr(C, align(16))]
pub struct ObjectHeader {
    pointer_count: AtomicUsize,
    handle_count: AtomicUsize,
    ty: AtomicPtr<ObjectType>,
    flags: AtomicU32,
    /// Name and parent directory; protected by the namespace lock.
    pub(super) name: UnsafeCell<*mut NameInfo>,
    /// Security descriptor. Reserved: there is no security reference monitor yet.
    pub security: AtomicUsize,
}

// SAFETY: all mutable state is atomic or protected by the namespace lock.
unsafe impl Sync for ObjectHeader {}
// SAFETY: as above.
unsafe impl Send for ObjectHeader {}

pub const HEADER_SIZE: usize = size_of::<ObjectHeader>();

impl ObjectHeader {
    pub fn object_type(&self) -> &'static ObjectType {
        // SAFETY: set at creation to an immortal type object.
        unsafe { &*self.ty.load(Ordering::Acquire) }
    }
    pub fn body_ptr(&self) -> *mut u8 {
        // SAFETY: the body directly follows the header in the same allocation.
        unsafe { (self as *const Self as *mut u8).add(HEADER_SIZE) }
    }
    pub fn pointer_count(&self) -> usize {
        self.pointer_count.load(Ordering::Relaxed)
    }
    pub fn handle_count(&self) -> usize {
        self.handle_count.load(Ordering::Relaxed)
    }
    pub fn is_permanent(&self) -> bool {
        self.flags.load(Ordering::Relaxed) & OBJ_PERMANENT != 0
    }
    pub(super) fn add_pointer(&self) {
        let old = self.pointer_count.fetch_add(1, Ordering::Relaxed);
        debug_assert!(old > 0, "referencing a dead object");
    }
    /// Takes a reference unless the object is already being destroyed.
    pub(super) fn try_add_pointer(&self) -> bool {
        let mut cur = self.pointer_count.load(Ordering::Relaxed);
        loop {
            if cur == 0 {
                return false;
            }
            match self.pointer_count.compare_exchange_weak(cur, cur + 1, Ordering::Acquire, Ordering::Relaxed) {
                Ok(_) => return true,
                Err(c) => cur = c,
            }
        }
    }
    pub(super) fn add_handle(&self) {
        self.handle_count.fetch_add(1, Ordering::Relaxed);
    }
    /// Returns `true` if that was the last handle.
    pub(super) fn remove_handle(&self) -> bool {
        self.handle_count.fetch_sub(1, Ordering::AcqRel) == 1
    }
    pub(super) fn set_permanent(&self) {
        if self.flags.fetch_or(OBJ_PERMANENT, Ordering::AcqRel) & OBJ_PERMANENT == 0 {
            self.add_pointer();
        }
    }
    /// Clears the permanent flag; returns whether it was set (the caller then owns the extra reference to drop).
    pub(super) fn clear_permanent(&self) -> bool {
        self.flags.fetch_and(!OBJ_PERMANENT, Ordering::AcqRel) & OBJ_PERMANENT != 0
    }
}

// ----------------------------------------------------------------------
// Owning references
// ----------------------------------------------------------------------

/// An owning, untyped reference to an object (one `pointer_count`).
pub struct ObRef {
    hdr: NonNull<ObjectHeader>,
}

// SAFETY: the header is Sync and the body type is checked/chosen by the creator.
unsafe impl Send for ObRef {}
// SAFETY: see above.
unsafe impl Sync for ObRef {}

impl ObRef {
    /// # Safety
    /// `h` must be a live object whose reference the caller owns and hands over.
    pub(super) unsafe fn from_raw_owned(h: *mut ObjectHeader) -> Self {
        // SAFETY: the caller guarantees a valid pointer.
        Self { hdr: unsafe { NonNull::new_unchecked(h) } }
    }
    pub fn header(&self) -> &ObjectHeader {
        // SAFETY: we hold a reference, so the object is alive.
        unsafe { self.hdr.as_ref() }
    }
    pub(super) fn as_ptr(&self) -> *mut ObjectHeader {
        self.hdr.as_ptr()
    }
    pub fn object_type(&self) -> &'static ObjectType {
        self.header().object_type()
    }
    /// For a `Type` object: the name of the type it describes.
    pub(super) fn object_type_name(&self) -> &'static str {
        // SAFETY: only called on Type objects, whose body is an ObjectType.
        unsafe { &*(self.header().body_ptr() as *const ObjectType) }.name
    }
    pub fn is_type(&self, ty: &ObjectType) -> bool {
        ptr::eq(self.object_type(), ty)
    }
    /// The body as `&T`, if this object is of type `ty` (whose body is `T`).
    pub fn body<T>(&self, ty: &'static ObjectType) -> Option<&T> {
        if self.is_type(ty) && ty.body_size == size_of::<T>() {
            // SAFETY: type and size match; `T` is the type's body by contract of `create`.
            Some(unsafe { &*(self.header().body_ptr() as *const T) })
        } else {
            None
        }
    }
    /// Is this a Type object whose own type is itself?
    pub(super) fn object_type_is_self(&self) -> bool {
        ptr::eq(self.object_type(), self.header().body_ptr() as *const ObjectType)
    }
    /// Turns the address of an object header into a reference, unless the
    /// object is already being destroyed. For registries that hold objects
    /// *weakly* (they must remove the address before the object is freed).
    ///
    /// # Safety
    /// `addr` must be the address of a header that was valid when registered
    /// and whose owner removes it from the registry before freeing it.
    pub unsafe fn upgrade(addr: usize) -> Option<ObRef> {
        let h = addr as *mut ObjectHeader;
        // SAFETY: guaranteed by the caller.
        if unsafe { &*h }.try_add_pointer() {
            // SAFETY: we just took the reference we hand over.
            Some(unsafe { ObRef::from_raw_owned(h) })
        } else {
            None
        }
    }
    /// Address of the header (identity, for registries).
    pub fn addr(&self) -> usize {
        self.hdr.as_ptr() as usize
    }
    /// Same object? (identity, not equality of bodies)
    pub fn same_object(&self, other: &ObRef) -> bool {
        self.hdr == other.hdr
    }
}

impl Clone for ObRef {
    fn clone(&self) -> Self {
        self.header().add_pointer();
        Self { hdr: self.hdr }
    }
}

impl Drop for ObRef {
    fn drop(&mut self) {
        // SAFETY: we own one reference.
        unsafe { dereference(self.hdr.as_ptr()) };
    }
}

/// An owning reference that knows the body type.
pub struct Obj<T> {
    inner: ObRef,
    _marker: PhantomData<T>,
}

// SAFETY: sharing the reference shares the body; require the body to allow it.
unsafe impl<T: Send + Sync> Send for Obj<T> {}
// SAFETY: see above.
unsafe impl<T: Send + Sync> Sync for Obj<T> {}

impl ObRef {
    /// A typed owning reference, if this object is of type `ty` (whose body is `T`);
    /// otherwise the reference is handed back unchanged.
    pub fn into_typed<T>(self, ty: &'static ObjectType) -> Result<Obj<T>, ObRef> {
        if self.is_type(ty) && ty.body_size == size_of::<T>() {
            Ok(Obj { inner: self, _marker: PhantomData })
        } else {
            Err(self)
        }
    }
}

impl<T> Obj<T> {
    pub fn untyped(&self) -> &ObRef {
        &self.inner
    }
    pub fn into_untyped(self) -> ObRef {
        self.inner
    }
    pub fn header(&self) -> &ObjectHeader {
        self.inner.header()
    }
}

impl<T> Deref for Obj<T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: `create::<T>` wrote a `T` into the body.
        unsafe { &*(self.inner.header().body_ptr() as *const T) }
    }
}

impl<T> Clone for Obj<T> {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone(), _marker: PhantomData }
    }
}

// ----------------------------------------------------------------------
// Create, reference, destroy
// ----------------------------------------------------------------------

fn layout_for(body_size: usize) -> Layout {
    Layout::from_size_align(HEADER_SIZE + body_size, HEADER_ALIGN).expect("object layout")
}

/// Allocates and initialises an object with one reference; no name yet.
fn allocate<T>(ty: *mut ObjectType, body_size: usize, value: T) -> *mut ObjectHeader {
    let layout = layout_for(body_size);
    // SAFETY: non-zero size, valid alignment.
    let p = unsafe { heap_alloc(layout) } as *mut ObjectHeader;
    if p.is_null() {
        handle_alloc_error(layout);
    }
    // SAFETY: `p` is a fresh allocation large enough for header + body.
    unsafe {
        ptr::write(
            p,
            ObjectHeader {
                pointer_count: AtomicUsize::new(1),
                handle_count: AtomicUsize::new(0),
                ty: AtomicPtr::new(ty),
                flags: AtomicU32::new(0),
                name: UnsafeCell::new(ptr::null_mut()),
                security: AtomicUsize::new(0),
            },
        );
        ptr::write((p as *mut u8).add(HEADER_SIZE) as *mut T, value);
    }
    p
}

/// Creates an object of type `ty` whose body is `value`. If `attrs` has a
/// name the object is inserted in the namespace; if it is permanent it stays
/// alive without references. Returns the creator's reference.
pub fn create<T>(ty: &'static ObjectType, attrs: &ObjectAttributes<'_>, value: T) -> Result<Obj<T>, ObError> {
    assert_eq!(size_of::<T>(), ty.body_size, "object body does not match its type");
    let h = allocate(ty as *const ObjectType as *mut ObjectType, ty.body_size, value);
    bump(&ty.total_objects, &ty.peak_objects);
    // SAFETY: we own the only reference; if naming fails the Drop below destroys it.
    let obj: Obj<T> = Obj { inner: unsafe { ObRef::from_raw_owned(h) }, _marker: PhantomData };
    if attrs.name.is_some() {
        namespace::insert_named(attrs, obj.untyped())?;
    }
    if attrs.permanent {
        obj.header().set_permanent();
    }
    Ok(obj)
}

/// Makes the type of types: an object that is its own type.
pub(super) fn bootstrap_type_type(init: TypeInit) -> &'static ObjectType {
    let h = allocate(ptr::null_mut(), size_of::<ObjectType>(), ObjectType::new(init));
    // SAFETY: fresh, exclusively ours.
    let hdr = unsafe { &*h };
    let body = hdr.body_ptr() as *mut ObjectType;
    hdr.ty.store(body, Ordering::Release);
    // SAFETY: `body` was just initialised.
    let ty: &'static ObjectType = unsafe { &*body };
    bump(&ty.total_objects, &ty.peak_objects);
    hdr.set_permanent();
    TYPES.lock().push(h as usize);
    ty
}

/// Creates a new object type. Types are immortal and appear as
/// `\ObjectTypes\<name>`.
pub fn create_type(init: TypeInit) -> Result<&'static ObjectType, ObError> {
    let ty_of_types = types::type_type();
    let path = alloc::format!("\\ObjectTypes\\{}", init.name);
    let attrs = if namespace::is_ready() { ObjectAttributes::named(&path) } else { ObjectAttributes::unnamed() };
    let obj = create::<ObjectType>(ty_of_types, &attrs, ObjectType::new(init))?;
    obj.header().set_permanent();
    TYPES.lock().push(obj.untyped().as_ptr() as usize);
    let body = obj.header().body_ptr() as *const ObjectType;
    core::mem::forget(obj); // immortal: both references stay
    // SAFETY: the object is immortal.
    Ok(unsafe { &*body })
}

/// Drops one pointer reference; destroys the object at zero.
///
/// # Safety
/// The caller owns one reference to `h` and gives it up.
pub(super) unsafe fn dereference(h: *mut ObjectHeader) {
    // SAFETY: the caller owns a reference, so the header is alive.
    let old = unsafe { &*h }.pointer_count.fetch_sub(1, Ordering::AcqRel);
    assert!(old > 0, "object reference count underflow");
    if old == 1 {
        // SAFETY: last reference: nobody else can reach the object.
        unsafe { destroy(h) };
    }
}

unsafe fn destroy(h: *mut ObjectHeader) {
    // SAFETY: no references remain; the header is still valid until freed below.
    let hdr = unsafe { &*h };
    // A name may still point at us (directories hold no reference to children).
    // SAFETY: valid object, last reference.
    unsafe { namespace::unlink(h) };
    let ty = hdr.object_type();
    let size = ty.body_size;
    // SAFETY: the body is a live value of the type's body type.
    unsafe { (ty.delete)(hdr.body_ptr()) };
    ty.total_objects.fetch_sub(1, Ordering::Relaxed);
    // SAFETY: allocated with `layout_for(size)` in `allocate`.
    unsafe { heap_dealloc(h as *mut u8, layout_for(size)) };
}

/// Drops a handle's claim on `h`: counts, last-close hook, temporary-name removal, reference.
///
/// # Safety
/// The caller owns the pointer reference that the handle held.
pub(super) unsafe fn release_handle(h: *mut ObjectHeader) {
    // SAFETY: the handle's reference keeps the object alive.
    let hdr = unsafe { &*h };
    let ty = hdr.object_type();
    ty.note_handle_closed();
    if hdr.remove_handle() {
        ty.run_last_close(hdr.body_ptr());
        if !hdr.is_permanent() {
            // SAFETY: valid object (we still hold a reference).
            unsafe { namespace::unlink(h) };
        }
    }
    // SAFETY: gives up the handle's reference.
    unsafe { dereference(h) };
}

/// `ObMakeTemporaryObject`: a permanent object loses its extra reference and,
/// if it has no handles, its name; it then lives as long as it is referenced.
pub fn make_temporary(obj: &ObRef) {
    let hdr = obj.header();
    if hdr.clear_permanent() {
        if hdr.handle_count() == 0 {
            // SAFETY: we hold a reference.
            unsafe { namespace::unlink(obj.as_ptr()) };
        }
        // SAFETY: drops the reference the permanent flag represented.
        unsafe { dereference(obj.as_ptr()) };
    }
}
