// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Boot-time self-test of the Object Manager.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicUsize, Ordering};

use super::access::*;
use super::handle::{DuplicateOptions, HANDLE_PROTECT_FROM_CLOSE};
use super::object::{self, TypeInit};
use super::types::{self, *};
use super::{kernel_handles, namespace, stats, ObError, ObjectAttributes, ObjectType};
use super::{query_event, reset_event, set_event, HandleTable};
use crate::mm::{heap, pmm, vmm};

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

static DROPS: AtomicUsize = AtomicUsize::new(0);
static LAST_CLOSES: AtomicUsize = AtomicUsize::new(0);

struct Probe(#[allow(dead_code)] u32);

impl Drop for Probe {
    fn drop(&mut self) {
        DROPS.fetch_add(1, Ordering::SeqCst);
    }
}

fn probe_last_close(_body: *const u8) {
    LAST_CLOSES.fetch_add(1, Ordering::SeqCst);
}

/// Frames in use that are *not* VMM-committed pages. The heap grows by whole
/// pages (committed through the VMM) and keeps them, so raw frame counts move
/// legitimately; this balance only moves if page tables or frames leak.
fn balance() -> u64 {
    pmm::stats().in_use_frames - vmm::stats().committed_pages
}

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s >> 33
}

/// What [`self_test`] exercised, for the boot log.
#[derive(Clone, Copy)]
pub struct SelfTest {
    pub checks: u32,
    pub peak_handles: usize,
    pub stress_ops: u32,
}

pub fn self_test() -> Result<SelfTest, &'static str> {
    // The kernel handle table keeps the pages it grows: warm it up first.
    {
        let w = tr!(create_event(&ObjectAttributes::unnamed(), EventKind::Notification, false), "warm-up event");
        let h = tr!(kernel_handles().create_handle(w.untyped(), EVENT_ALL, 0), "warm-up handle");
        tr!(kernel_handles().close(h), "warm-up close");
    }
    let heap0 = heap::stats().live_bytes;
    let bal0 = balance();
    let objs0 = stats().objects;
    let ns0 = namespace::snapshot().len();
    let mut out = SelfTest { checks: 0, peak_handles: 0, stress_ops: 0 };

    // 1) Types are objects in the namespace, found case-insensitively.
    let t = tr!(namespace::lookup("\\ObjectTypes\\Event", Some(types::type_type()), true), "type object not found");
    ensure!(core::ptr::eq(tr!(t.body::<ObjectType>(types::type_type()).ok_or(()), "not a type"), types::event_type()), "wrong type object");
    ensure!(namespace::lookup("\\objecttypes\\EVENT", None, true).is_ok(), "lookup is not case-insensitive");
    ensure!(types::type_type().total_objects.load(Ordering::Relaxed) >= 7, "type objects missing");
    drop(t);
    out.checks += 1;

    // 2) Names, handles, access rights, counts.
    let dir = tr!(types::create_directory(&ObjectAttributes::named("\\SelfTest")), "create directory");
    let ev = tr!(create_event(&ObjectAttributes::named("\\SelfTest\\Ev"), EventKind::Notification, false), "create event");
    ensure!(super::path_of(ev.untyped()).as_deref() == Some("\\SelfTest\\Ev"), "path_of is wrong");
    ensure!(create_event(&ObjectAttributes::named("\\selftest\\EV"), EventKind::Notification, false).err() == Some(ObError::NameCollision), "name collision not detected");
    for bad in ["\\SelfTest\\", "\\SelfTest\\\\X", "NoRoot", "\\SelfTest\\a\u{1}b", "\\SelfTest\\.."] {
        ensure!(create_event(&ObjectAttributes::named(bad), EventKind::Notification, false).err() == Some(ObError::InvalidName), "invalid name accepted");
    }
    ensure!(create_event(&ObjectAttributes::named("\\SelfTest\\Nope\\X"), EventKind::Notification, false).err() == Some(ObError::PathNotFound), "missing parent not reported");
    ensure!(create_event(&ObjectAttributes::named("\\SelfTest\\Ev\\X"), EventKind::Notification, false).err() == Some(ObError::NotADirectory), "event used as directory");

    let table = HandleTable::new();
    let h1 = tr!(table.create_handle(ev.untyped(), EVENT_ALL, 0), "create handle");
    ensure!(h1.0 != 0 && h1.0 % 4 == 0, "handle value shape");
    ensure!(ev.header().handle_count() == 1 && ev.header().pointer_count() == 2, "counts after first handle");
    let r = tr!(namespace::lookup("\\SelfTest\\Ev", Some(event_type()), true), "lookup by name");
    ensure!(r.same_object(ev.untyped()) && ev.header().pointer_count() == 3, "lookup reference");
    drop(r);
    ensure!(ev.header().pointer_count() == 2, "lookup leaked a reference");

    let h2 = tr!(table.duplicate(h1, &table, DuplicateOptions { access: Some(EVENT_QUERY_STATE), attributes: 0, close_source: false }), "duplicate");
    ensure!(ev.header().handle_count() == 2, "duplicate did not count");
    ensure!(set_event(&table, h2) == Err(ObError::AccessDenied), "read-only handle may modify");
    ensure!(query_event(&table, h2) == Ok(false), "query through read-only handle");
    ensure!(set_event(&table, h1) == Ok(false), "set through full handle");
    ensure!(query_event(&table, h2) == Ok(true), "set not visible");
    ensure!(reset_event(&table, h1) == Ok(true) && query_event(&table, h1) == Ok(false), "reset");
    ensure!(
        table.duplicate(h2, &table, DuplicateOptions { access: Some(EVENT_ALL), attributes: 0, close_source: false }) == Err(ObError::AccessDenied),
        "duplicate may add rights"
    );
    ensure!(table.reference_by_handle(h1, 0, Some(directory_type())).err() == Some(ObError::TypeMismatch), "type mismatch not detected");
    ensure!(table.create_handle(ev.untyped(), 0x8000, 0).err() == Some(ObError::InvalidAccess), "rights the type lacks accepted");
    let hg = tr!(table.create_handle(ev.untyped(), GENERIC_READ, 0), "generic handle");
    ensure!(tr!(table.info(hg), "info").access == READ_CONTROL | EVENT_QUERY_STATE, "generic read not mapped");
    tr!(table.close(hg), "close generic");
    for bad in [super::Handle(0), super::Handle(2), super::Handle(4 * 50_000)] {
        ensure!(table.close(bad) == Err(ObError::InvalidHandle) && table.info(bad).err() == Some(ObError::InvalidHandle), "bad handle accepted");
    }
    tr!(table.set_attributes(h2, HANDLE_PROTECT_FROM_CLOSE), "protect");
    ensure!(table.close(h2) == Err(ObError::HandleProtected), "protected handle closed");
    tr!(table.set_attributes(h2, 0), "unprotect");
    tr!(table.close(h2), "close h2");
    ensure!(table.close(h2) == Err(ObError::InvalidHandle), "closed handle still valid");
    drop(ev);
    // Keep a pointer reference while the last handle closes: the *name* must
    // still go (the object itself lives on, unnamed).
    let hold = tr!(namespace::lookup("\\SelfTest\\Ev", None, true), "named object lost with handle open");
    tr!(table.close(h1), "close last handle");
    ensure!(namespace::lookup("\\SelfTest\\Ev", None, true).err() == Some(ObError::NameNotFound), "temporary name survived last close");
    ensure!(super::path_of(&hold).is_none() && hold.header().handle_count() == 0, "unnamed survivor state");
    drop(hold);
    out.checks += 1;

    // 3) Permanent objects keep their name and a reference.
    let p = tr!(create_event(&ObjectAttributes::named("\\SelfTest\\Perm").permanent(), EventKind::Notification, false), "create permanent");
    ensure!(p.header().pointer_count() == 2 && p.header().is_permanent(), "permanent reference");
    let pr = p.untyped().clone();
    drop(p);
    let hh = tr!(table.create_handle(&pr, EVENT_ALL, 0), "perm handle");
    tr!(table.close(hh), "perm close");
    ensure!(namespace::lookup("\\SelfTest\\Perm", None, true).is_ok(), "permanent name removed");
    object::make_temporary(&pr);
    ensure!(namespace::lookup("\\SelfTest\\Perm", None, true).err() == Some(ObError::NameNotFound), "make_temporary kept the name");
    drop(pr);
    out.checks += 1;

    // 4) Symbolic links: chains, middle components, loops.
    let un = ObjectAttributes::unnamed();
    let e2 = tr!(create_event(&ObjectAttributes::named("\\SelfTest\\Ev2"), EventKind::Synchronization, true), "create Ev2");
    let l1 = tr!(create_symlink(&ObjectAttributes::named("\\SelfTest\\L1"), "\\SelfTest\\Ev2"), "link 1");
    let l2 = tr!(create_symlink(&ObjectAttributes::named("\\SelfTest\\L2"), "\\SelfTest\\L1"), "link 2");
    let dl = tr!(create_symlink(&ObjectAttributes::named("\\SelfTest\\DirLink"), "\\SelfTest"), "dir link");
    let via = tr!(namespace::lookup("\\SelfTest\\L2", Some(event_type()), true), "follow chain");
    ensure!(via.same_object(e2.untyped()), "chain resolved to the wrong object");
    let raw = tr!(namespace::lookup("\\SelfTest\\L2", Some(symlink_type()), false), "no-follow");
    ensure!(raw.same_object(l2.untyped()), "no-follow returned the target");
    ensure!(namespace::lookup("\\SelfTest\\L2", Some(directory_type()), true).err() == Some(ObError::TypeMismatch), "link target type");
    let mid = tr!(namespace::lookup("\\SELFTEST\\dirlink\\ev2", Some(event_type()), true), "link in the middle");
    ensure!(mid.same_object(e2.untyped()), "middle link resolved wrongly");
    drop((via, raw, mid));
    ensure!(e2.try_wait() && !e2.try_wait(), "synchronization event not auto-reset");
    let la = tr!(create_symlink(&ObjectAttributes::named("\\SelfTest\\LA"), "\\SelfTest\\LB"), "LA");
    let lb = tr!(create_symlink(&ObjectAttributes::named("\\SelfTest\\LB"), "\\SelfTest\\LA"), "LB");
    ensure!(namespace::lookup("\\SelfTest\\LA", None, true).err() == Some(ObError::LinkLoop), "link loop not detected");
    ensure!(create_symlink(&un, "relative").err() == Some(ObError::InvalidName), "relative link target accepted");
    drop((la, lb, l1, l2, dl, e2));
    out.checks += 1;

    // 5) A directory outlives its children and goes when the last one does.
    let sub = tr!(types::create_directory(&ObjectAttributes::named("\\SelfTest\\Sub")), "create Sub");
    let child = tr!(create_event(&ObjectAttributes::named("Child").relative_to(sub.untyped()), EventKind::Notification, false), "relative create");
    ensure!(super::path_of(child.untyped()).as_deref() == Some("\\SelfTest\\Sub\\Child"), "relative path");
    ensure!(sub.len() == 1, "directory entry count");
    drop(sub);
    ensure!(namespace::lookup("\\SelfTest\\Sub", Some(directory_type()), true).is_ok(), "directory died with children");
    let names = tr!(namespace::list_directory(dir.untyped()), "list");
    ensure!(names.iter().any(|(n, t)| n == "Sub" && *t == "Directory"), "listing misses Sub");
    drop(child);
    ensure!(namespace::lookup("\\SelfTest\\Sub", None, true).err() == Some(ObError::NameNotFound), "directory survived its last child");
    out.checks += 1;

    // 6) Handle table: growth over several pages, reuse, stale handles, drop.
    let ev = tr!(create_event(&un, EventKind::Notification, false), "plain event");
    let t2 = HandleTable::new();
    let mut hs = Vec::new();
    for _ in 0..1000 {
        hs.push(tr!(t2.create_handle(ev.untyped(), EVENT_ALL, 0), "bulk handle"));
    }
    let mut sorted: Vec<usize> = hs.iter().map(|h| h.0).collect();
    sorted.sort_unstable();
    sorted.dedup();
    ensure!(sorted.len() == 1000 && sorted[0] != 0 && sorted.iter().all(|v| v % 4 == 0), "handle values not unique multiples of 4");
    ensure!(ev.header().handle_count() == 1000 && ev.header().pointer_count() == 1001 && t2.len() == 1000, "bulk counts");
    let max_before = sorted[999];
    for i in (1..1000).step_by(2) {
        tr!(t2.close(hs[i]), "bulk close");
    }
    let mut again = Vec::new();
    for _ in 0..500 {
        again.push(tr!(t2.create_handle(ev.untyped(), EVENT_ALL, 0), "reopen"));
    }
    ensure!(again.iter().all(|h| h.0 <= max_before), "closed entries were not reused");
    for i in (0..1000).step_by(2) {
        tr!(t2.close(hs[i]), "bulk close even");
    }
    for h in &again {
        tr!(t2.close(*h), "close reopened");
    }
    ensure!(t2.is_empty() && ev.header().handle_count() == 0 && ev.header().pointer_count() == 1, "table not empty after closing all");
    ensure!(t2.info(hs[0]).err() == Some(ObError::InvalidHandle), "stale handle accepted");
    for _ in 0..10 {
        tr!(t2.create_handle(ev.untyped(), EVENT_ALL, 0), "last batch");
    }
    drop(t2);
    ensure!(ev.header().handle_count() == 0 && ev.header().pointer_count() == 1, "dropping a table leaked handles");
    out.peak_handles = types::event_type().peak_handles.load(Ordering::Relaxed);
    drop((ev, hs, again, sorted));
    out.checks += 1;

    // 7) An address space as an object: destroying the object frees the space.
    let bal1 = balance();
    let used1 = pmm::stats().in_use_frames;
    let asobj = tr!(create_address_space(&un), "create address space");
    let kh = kernel_handles();
    let ha = tr!(kh.create_handle(asobj.untyped(), ADDRESS_SPACE_ALL, 0), "address space handle");
    tr!(asobj.map_new(0x40_0000, 3, vmm::Prot::READ_WRITE.user()).map_err(|_| ObError::OutOfResources), "map in object");
    ensure!(asobj.translate(0x40_0000).is_some() && pmm::stats().in_use_frames > used1, "mapping not backed");
    let hq = tr!(kh.duplicate(ha, kh, DuplicateOptions { access: Some(ADDRESS_SPACE_QUERY), attributes: 0, close_source: false }), "dup query");
    ensure!(kh.reference_by_handle(hq, ADDRESS_SPACE_MAP, Some(address_space_type())).err() == Some(ObError::AccessDenied), "query handle may map");
    tr!(kh.close(hq), "close hq");
    tr!(kh.close(ha), "close ha");
    drop(asobj);
    ensure!(balance() == bal1, "address-space object leaked frames");
    out.checks += 1;

    // 8) Type callbacks: delete runs once per object, last_close once per object.
    let probe_ty: &'static ObjectType = Box::leak(Box::new(ObjectType::new(
        TypeInit::of::<Probe>("SelfTestProbe", STANDARD_RIGHTS_ALL | 0xFF, GenericMapping { read: 0, write: 0, execute: 0, all: 0 })
            .on_last_close(probe_last_close),
    )));
    let (d0, c0) = (DROPS.load(Ordering::SeqCst), LAST_CLOSES.load(Ordering::SeqCst));
    let a = tr!(object::create(probe_ty, &un, Probe(1)), "probe a");
    let b = tr!(object::create(probe_ty, &un, Probe(2)), "probe b");
    let t3 = HandleTable::new();
    let a1 = tr!(t3.create_handle(a.untyped(), STANDARD_RIGHTS_ALL, 0), "a1");
    let a2 = tr!(t3.create_handle(a.untyped(), STANDARD_RIGHTS_ALL, 0), "a2");
    tr!(t3.close(a1), "close a1");
    ensure!(LAST_CLOSES.load(Ordering::SeqCst) == c0, "last_close ran too early");
    tr!(t3.close(a2), "close a2");
    ensure!(LAST_CLOSES.load(Ordering::SeqCst) == c0 + 1, "last_close did not run");
    drop(a);
    ensure!(DROPS.load(Ordering::SeqCst) == d0 + 1, "delete routine did not run");
    tr!(t3.create_handle(b.untyped(), STANDARD_RIGHTS_ALL, 0), "b1");
    drop(t3);
    ensure!(LAST_CLOSES.load(Ordering::SeqCst) == c0 + 2, "dropping a table did not close its handles");
    drop(b);
    ensure!(DROPS.load(Ordering::SeqCst) == d0 + 2 && probe_ty.total_objects.load(Ordering::Relaxed) == 0, "probe objects leaked");
    out.checks += 1;

    // 9) Randomised stress over one table.
    {
        let t = HandleTable::new();
        let mut slots: [Option<super::Handle>; 48] = [None; 48];
        let mut rng = 0xC0FF_EE00_1234_5678u64;
        for _ in 0..6000u32 {
            let i = lcg(&mut rng) as usize % slots.len();
            match slots[i] {
                None => {
                    let o = tr!(create_event(&un, EventKind::Notification, false), "stress create");
                    slots[i] = Some(tr!(t.create_handle(o.untyped(), EVENT_ALL, 0), "stress handle"));
                }
                Some(h) => match lcg(&mut rng) % 3 {
                    0 => {
                        tr!(t.close(h), "stress close");
                        slots[i] = None;
                    }
                    1 => {
                        if let Some(j) = slots.iter().position(|s| s.is_none()) {
                            slots[j] = Some(tr!(t.duplicate(h, &t, DuplicateOptions::same_access()), "stress dup"));
                        }
                    }
                    _ => {
                        ensure!(set_event(&t, h).is_ok() && query_event(&t, h) == Ok(true), "stress event");
                    }
                },
            }
            out.stress_ops += 1;
        }
        for s in slots.iter().flatten() {
            tr!(t.close(*s), "stress final close");
        }
        ensure!(t.is_empty(), "stress left handles");
    }
    out.checks += 1;

    // 10) Everything we made is gone again.
    drop((dir, table, names));
    ensure!(namespace::lookup("\\SelfTest", None, true).err() == Some(ObError::NameNotFound), "test directory survived");
    ensure!(stats().objects == objs0, "object count did not return to baseline");
    ensure!(namespace::snapshot().len() == ns0, "namespace changed");
    ensure!(balance() == bal0, "frames leaked");
    // The only intended survivor is the leaked probe type (types are immortal).
    ensure!(heap::stats().live_bytes == heap0 + core::mem::size_of::<ObjectType>() as u64, "heap memory leaked");
    Ok(out)
}
