// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Virtual Memory Manager (VMM): who owns which **virtual** addresses, and
//! what backs them.
//!
//! The PMM owns physical frames; `hal::paging` knows how to edit page tables.
//! The VMM is the policy layer between them:
//!
//! ```text
//!   heap / drivers / scheduler        (callers)
//!        |  alloc, reserve, commit, map_phys, alloc_stack, AddressSpace
//!        v
//!   mm::vmm   region bookkeeping, W^X, guard pages, frame ownership
//!        |  map_page_in / unmap_page_in / protect_page_in
//!        v
//!   hal::paging   page tables         mm::pmm   physical frames
//! ```
//!
//! # Kernel virtual-address map
//!
//! | range                               | what                              | owner          |
//! |-------------------------------------|-----------------------------------|----------------|
//! | `0x0000_0000_0000_0000..0x0000_8000_0000_0000` | user half (per address space) | [`AddressSpace`] |
//! | `0xFFFF_8000_...` (offset from boot)| HHDM: all RAM, 1:1                | bootloader     |
//! | `0xFFFF_C000_0000_0000` (+1 GiB)    | early MMIO bump window            | `hal::paging`  |
//! | [`ARENA_BASE`] (+[`ARENA_SIZE`] = 512 GiB) | **VMM arena**: heap, stacks, device mappings | this module |
//! | `0xFFFF_FFFF_8000_0000..`           | kernel image                      | bootloader     |
//!
//! # Regions
//!
//! Everything in the arena belongs to a **region**: a page-aligned virtual
//! extent with a kind, a tag (for diagnostics) and optional **guard pages**
//! (never mapped, so an overrun or stack overflow faults instead of silently
//! corrupting the neighbour). Regions live in a fixed, sorted table
//! ([`MAX_REGIONS`]); the VMM needs no heap, which is what lets the heap be
//! built *on* it.
//!
//! A region is used in the Windows way: **reserve** address space first, then
//! **commit** pages inside it when needed. [`alloc`] is reserve + commit-all.
//!
//! # Ownership
//!
//! A leaf mapping made by [`commit`] carries the software bit
//! [`pte::OWNED`]: the frame came from the PMM and goes back to it when the
//! page is unmapped. Mappings of existing memory ([`map_phys`]: device
//! registers, firmware tables) are *not* owned and never free anything.
//!
//! # Rules the VMM enforces
//!
//! * **W^X**: a mapping is never writable and executable at once.
//! * Kernel mappings are never `USER`; user mappings never touch the kernel half.
//! * Commits are all-or-nothing (rolled back on failure); every fresh frame is
//!   **zeroed**, so nothing leaks from a previous owner.
//! * Page tables that become empty are freed (PT and PD level), so a
//!   map/unmap cycle leaves the PMM exactly as it found it.
//!
//! # Concurrency
//!
//! One global [`IrqLock`] (interrupts masked while held), so the VMM and
//! everything above it can be used from interrupt handlers. Lock order is
//! `heap -> vmm -> pmm`. TLB maintenance is local to the executing CPU (see
//! `hal::paging`); the APs are parked after start-up, so no shootdown is
//! needed *yet*. It becomes mandatory the day a second CPU runs kernel code.

use core::fmt;

use super::pmm;
use crate::hal::paging::{self, pte, CacheMode, PagingError};
use crate::hal::sync::IrqLock;

pub const PAGE_SIZE: u64 = 4096;

/// First address of the VMM arena (PML4 slot 416).
pub const ARENA_BASE: u64 = 0xFFFF_D000_0000_0000;
/// Size of the arena: exactly one PML4 slot, 512 GiB.
pub const ARENA_SIZE: u64 = 1 << 39;
const ARENA_END: u64 = ARENA_BASE + ARENA_SIZE;

/// First address of the kernel half of every address space.
pub const KERNEL_HALF: u64 = 0xFFFF_8000_0000_0000;
/// One past the last user-half address.
pub const USER_END: u64 = 0x0000_8000_0000_0000;

/// Capacity of the region table.
pub const MAX_REGIONS: usize = 4096;

// ----------------------------------------------------------------------
// Types
// ----------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VmError {
    NotReady,
    AlreadyInitialised,
    /// Zero size, or size/alignment arithmetic overflowed.
    BadSize,
    /// Alignment not a power of two.
    BadAlignment,
    /// Address not page-aligned, or in the wrong half for this operation.
    BadAddress,
    /// No free virtual extent of that size/alignment in the arena.
    OutOfVirtualSpace,
    /// The PMM has no frame left.
    OutOfFrames,
    /// The region table is full.
    TooManyRegions,
    /// No region starts at / contains that address.
    NotFound,
    /// The range leaves the region (or touches a guard page).
    OutOfRegion,
    AlreadyMapped,
    NotMapped,
    /// Writable + executable was requested.
    WriteExecute,
    /// `USER` requested for a kernel mapping.
    UserInKernel,
    /// The region is of the wrong kind for this operation.
    WrongKind,
    Paging(PagingError),
}

impl From<PagingError> for VmError {
    fn from(e: PagingError) -> Self {
        match e {
            PagingError::OutOfFrames => VmError::OutOfFrames,
            PagingError::AlreadyMapped => VmError::AlreadyMapped,
            PagingError::NotMapped => VmError::NotMapped,
            other => VmError::Paging(other),
        }
    }
}

/// Access rights and memory type of a mapping.
///
/// Reading is always allowed. Build values from the constants:
/// `Prot::READ_WRITE`, `Prot::READ_EXEC.user()`, `Prot::DEVICE`, ...
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Prot {
    pub write: bool,
    pub exec: bool,
    pub user: bool,
    pub cache: CacheMode,
}

impl Prot {
    pub const READ: Prot = Prot { write: false, exec: false, user: false, cache: CacheMode::WriteBack };
    pub const READ_WRITE: Prot = Prot { write: true, ..Prot::READ };
    pub const READ_EXEC: Prot = Prot { exec: true, ..Prot::READ };
    /// Device registers: read/write, uncacheable, never executable.
    pub const DEVICE: Prot = Prot { write: true, cache: CacheMode::Uncached, ..Prot::READ };

    pub const fn user(self) -> Prot {
        Prot { user: true, ..self }
    }

    pub const fn cache(self, cache: CacheMode) -> Prot {
        Prot { cache, ..self }
    }

    fn check(self) -> Result<(), VmError> {
        if self.write && self.exec { Err(VmError::WriteExecute) } else { Ok(()) }
    }

    fn pte_bits(self, nx: bool) -> u64 {
        let mut f = self.cache.pte_bits();
        if self.write {
            f |= pte::WRITABLE;
        }
        if self.user {
            f |= pte::USER;
        }
        if !self.exec && nx {
            f |= pte::NO_EXECUTE;
        }
        f
    }
}

impl fmt::Display for Prot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "r{}{}{}",
            if self.write { 'w' } else { '-' },
            if self.exec { 'x' } else { '-' },
            if self.user { 'u' } else { '-' }
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionKind {
    /// Reserved/committed memory (`alloc`, `reserve`).
    Memory,
    /// Window onto existing physical memory (`map_phys`).
    Physical,
    /// A kernel stack (guard page on both sides).
    Stack,
}

impl RegionKind {
    pub fn name(self) -> &'static str {
        match self {
            RegionKind::Memory => "memory",
            RegionKind::Physical => "physical",
            RegionKind::Stack => "stack",
        }
    }
}

/// Unmapped pages placed around a region's usable part.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Guard {
    None,
    /// One page after the end: catches overruns.
    After,
    /// One page before and one after: catches stack overflow and underrun.
    Both,
}

impl Guard {
    fn lead(self) -> u64 {
        (self == Guard::Both) as u64
    }
    fn trail(self) -> u64 {
        (self != Guard::None) as u64
    }
}

/// A region as reported to callers (see [`for_each_region`]).
#[derive(Clone, Copy, Debug)]
pub struct RegionInfo {
    /// First usable (non-guard) address.
    pub start: u64,
    /// Usable pages.
    pub pages: u64,
    pub committed: u64,
    pub guard_lo: bool,
    pub guard_hi: bool,
    pub kind: RegionKind,
    pub tag: &'static str,
}

#[derive(Clone, Copy)]
struct Region {
    /// First address including the leading guard page.
    start: u64,
    /// Total pages including guards.
    total: u64,
    committed: u64,
    tag: Option<&'static str>, // `None` keeps the empty table all-zero (.bss)
    kind: RegionKind,
    lead: u8,
    trail: u8,
}

impl Region {
    const EMPTY: Region = Region {
        start: 0,
        total: 0,
        committed: 0,
        tag: None,
        kind: RegionKind::Memory,
        lead: 0,
        trail: 0,
    };

    fn end(&self) -> u64 {
        self.start + self.total * PAGE_SIZE
    }
    fn usable_start(&self) -> u64 {
        self.start + self.lead as u64 * PAGE_SIZE
    }
    fn usable_pages(&self) -> u64 {
        self.total - self.lead as u64 - self.trail as u64
    }
    fn usable_end(&self) -> u64 {
        self.usable_start() + self.usable_pages() * PAGE_SIZE
    }
    fn info(&self) -> RegionInfo {
        RegionInfo {
            start: self.usable_start(),
            pages: self.usable_pages(),
            committed: self.committed,
            guard_lo: self.lead != 0,
            guard_hi: self.trail != 0,
            kind: self.kind,
            tag: self.tag.unwrap_or(""),
        }
    }
}

/// A kernel stack: `pages` usable pages with an unmapped guard page below and
/// above. The stack grows down from [`KernelStack::top`].
#[derive(Clone, Copy, Debug)]
pub struct KernelStack {
    pub base: u64,
    pub top: u64,
    pub pages: u64,
}

// ----------------------------------------------------------------------
// State
// ----------------------------------------------------------------------

struct Vmm {
    ready: bool,
    nx: bool,
    root: u64,
    count: usize,
    regions: [Region; MAX_REGIONS],
    reserved_pages: u64,
    committed_pages: u64,
    peak_committed_pages: u64,
    reserve_failures: u64,
    commit_failures: u64,
}

static VMM: IrqLock<Vmm> = IrqLock::new(Vmm {
    ready: false,
    nx: false,
    root: 0,
    count: 0,
    regions: [Region::EMPTY; MAX_REGIONS],
    reserved_pages: 0,
    committed_pages: 0,
    peak_committed_pages: 0,
    reserve_failures: 0,
    commit_failures: 0,
});

fn align_up(v: u64, a: u64) -> Option<u64> {
    v.checked_add(a - 1).map(|x| x & !(a - 1))
}

impl Vmm {
    fn ready(&self) -> Result<(), VmError> {
        if self.ready { Ok(()) } else { Err(VmError::NotReady) }
    }

    /// Index of the region whose *usable* extent contains `addr`.
    fn find_usable(&self, addr: u64) -> Option<usize> {
        // Regions are sorted and disjoint: binary search by start.
        let i = match self.regions[..self.count].binary_search_by(|r| r.start.cmp(&addr)) {
            Ok(i) => i,
            Err(0) => return None,
            Err(i) => i - 1,
        };
        let r = &self.regions[i];
        (addr >= r.usable_start() && addr < r.usable_end()).then_some(i)
    }

    /// Index of the region whose usable part *starts* at `addr`.
    fn find_start(&self, addr: u64) -> Option<usize> {
        let i = self.find_usable(addr)?;
        (self.regions[i].usable_start() == addr).then_some(i)
    }

    /// First-fit search for `total` pages whose *usable* start (after
    /// `lead` guard pages) is aligned to `align` bytes. Returns the region
    /// start and the insertion index.
    fn find_gap(&self, total: u64, lead: u64, align: u64) -> Option<(u64, usize)> {
        let size = total.checked_mul(PAGE_SIZE)?;
        let mut cursor = ARENA_BASE;
        for i in 0..=self.count {
            let limit = if i < self.count { self.regions[i].start } else { ARENA_END };
            let usable = align_up(cursor.checked_add(lead * PAGE_SIZE)?, align)?;
            let start = usable - lead * PAGE_SIZE;
            if start >= cursor && start.checked_add(size).is_some_and(|e| e <= limit) {
                return Some((start, i));
            }
            if i < self.count {
                cursor = self.regions[i].end();
            }
        }
        None
    }

    fn insert(&mut self, at: usize, r: Region) {
        self.regions.copy_within(at..self.count, at + 1);
        self.regions[at] = r;
        self.count += 1;
        self.reserved_pages += r.total;
    }

    fn remove(&mut self, at: usize) {
        self.reserved_pages -= self.regions[at].total;
        self.regions.copy_within(at + 1..self.count, at);
        self.count -= 1;
        self.regions[self.count] = Region::EMPTY;
    }

    fn reserve(
        &mut self,
        pages: u64,
        align_pages: u64,
        kind: RegionKind,
        tag: &'static str,
        guard: Guard,
    ) -> Result<u64, VmError> {
        self.ready()?;
        if pages == 0 {
            return Err(VmError::BadSize);
        }
        if !align_pages.is_power_of_two() {
            return Err(VmError::BadAlignment);
        }
        let total = pages
            .checked_add(guard.lead() + guard.trail())
            .ok_or(VmError::BadSize)?;
        let align = align_pages.checked_mul(PAGE_SIZE).ok_or(VmError::BadAlignment)?;
        if self.count == MAX_REGIONS {
            self.reserve_failures += 1;
            return Err(VmError::TooManyRegions);
        }
        let Some((start, at)) = self.find_gap(total, guard.lead(), align) else {
            self.reserve_failures += 1;
            return Err(VmError::OutOfVirtualSpace);
        };
        let region = Region {
            start,
            total,
            committed: 0,
            tag: Some(tag),
            kind,
            lead: guard.lead() as u8,
            trail: guard.trail() as u8,
        };
        self.insert(at, region);
        Ok(region.usable_start())
    }

    /// Checks `[va, va + pages)` lies inside the usable part of one region.
    fn range_region(&self, va: u64, pages: u64) -> Result<usize, VmError> {
        if va & (PAGE_SIZE - 1) != 0 {
            return Err(VmError::BadAddress);
        }
        if pages == 0 {
            return Err(VmError::BadSize);
        }
        let i = self.find_usable(va).ok_or(VmError::NotFound)?;
        let end = pages
            .checked_mul(PAGE_SIZE)
            .and_then(|b| va.checked_add(b))
            .ok_or(VmError::BadSize)?;
        if end > self.regions[i].usable_end() {
            return Err(VmError::OutOfRegion);
        }
        Ok(i)
    }

    fn unmap_one(&mut self, va: u64) -> Option<bool> {
        match paging::unmap_page_in(self.root, va) {
            Ok(u) => {
                let owned = u.entry & pte::OWNED != 0;
                if owned {
                    // A failure here would mean the PMM and the page tables
                    // disagree; verify() reports the leak, nothing to do now.
                    let _ = pmm::free_frame(u.phys);
                }
                Some(owned)
            }
            Err(_) => None,
        }
    }

    /// Unmaps `[va, va + pages)` (skipping holes), frees owned frames and
    /// empty tables. Returns (pages unmapped, frames freed).
    fn unmap_range(&mut self, va: u64, pages: u64) -> (u64, u64) {
        let (mut unmapped, mut freed) = (0, 0);
        for k in 0..pages {
            if let Some(owned) = self.unmap_one(va + k * PAGE_SIZE) {
                unmapped += 1;
                freed += owned as u64;
            }
        }
        self.reap(va, pages);
        (unmapped, freed)
    }

    /// Frees the page tables left empty by an unmap of `[va, va + pages)`.
    fn reap(&mut self, va: u64, pages: u64) {
        const BLOCK: u64 = 2 << 20;
        let end = va + pages * PAGE_SIZE;
        let mut a = va & !(BLOCK - 1);
        while a < end {
            paging::reap_tables_in(self.root, a);
            a += BLOCK;
        }
    }

    fn commit(&mut self, va: u64, pages: u64, prot: Prot) -> Result<(), VmError> {
        self.ready()?;
        prot.check()?;
        if prot.user {
            return Err(VmError::UserInKernel);
        }
        let at = self.range_region(va, pages)?;
        if self.regions[at].kind == RegionKind::Physical {
            return Err(VmError::WrongKind);
        }
        // Fail fast instead of zeroing frames only to roll them back.
        if pages > pmm::stats().free_frames {
            self.commit_failures += 1;
            return Err(VmError::OutOfFrames);
        }
        let bits = prot.pte_bits(self.nx) | pte::OWNED;
        for k in 0..pages {
            let page = va + k * PAGE_SIZE;
            let Some(frame) = pmm::alloc_zeroed() else {
                self.commit_failures += 1;
                self.unmap_range(va, k);
                return Err(VmError::OutOfFrames);
            };
            if let Err(e) = paging::map_page_in(self.root, page, frame, bits) {
                let _ = pmm::free_frame(frame);
                self.commit_failures += 1;
                self.unmap_range(va, k);
                return Err(e.into());
            }
        }
        self.regions[at].committed += pages;
        self.committed_pages += pages;
        self.peak_committed_pages = self.peak_committed_pages.max(self.committed_pages);
        Ok(())
    }

    fn decommit(&mut self, va: u64, pages: u64) -> Result<u64, VmError> {
        self.ready()?;
        let at = self.range_region(va, pages)?;
        let (unmapped, freed) = self.unmap_range(va, pages);
        self.regions[at].committed -= unmapped;
        self.committed_pages -= unmapped;
        Ok(freed)
    }

    fn release(&mut self, va: u64) -> Result<(), VmError> {
        self.ready()?;
        let at = self.find_start(va).ok_or(VmError::NotFound)?;
        let r = self.regions[at];
        if r.committed != 0 {
            let (unmapped, _) = self.unmap_range(r.usable_start(), r.usable_pages());
            self.committed_pages -= unmapped;
        }
        self.remove(at);
        Ok(())
    }

    fn protect(&mut self, va: u64, pages: u64, prot: Prot) -> Result<(), VmError> {
        self.ready()?;
        prot.check()?;
        if prot.user {
            return Err(VmError::UserInKernel);
        }
        let at = self.range_region(va, pages)?;
        if self.regions[at].kind == RegionKind::Physical && prot.exec {
            return Err(VmError::WrongKind);
        }
        // All-or-nothing: every page must be mapped before any is changed.
        for k in 0..pages {
            if paging::translate_in(self.root, va + k * PAGE_SIZE).is_none() {
                return Err(VmError::NotMapped);
            }
        }
        let bits = prot.pte_bits(self.nx);
        for k in 0..pages {
            paging::protect_page_in(self.root, va + k * PAGE_SIZE, bits)?;
        }
        Ok(())
    }

    fn map_phys(&mut self, phys: u64, bytes: u64, prot: Prot, tag: &'static str) -> Result<*mut u8, VmError> {
        self.ready()?;
        prot.check()?;
        if prot.user {
            return Err(VmError::UserInKernel);
        }
        if bytes == 0 {
            return Err(VmError::BadSize);
        }
        let start = phys & !(PAGE_SIZE - 1);
        let offset = phys - start;
        let pages = offset
            .checked_add(bytes)
            .and_then(|n| n.checked_add(PAGE_SIZE - 1))
            .ok_or(VmError::BadSize)?
            / PAGE_SIZE;
        let va = self.reserve(pages, 1, RegionKind::Physical, tag, Guard::After)?;
        let bits = prot.pte_bits(self.nx); // no OWNED: these frames are not ours
        for k in 0..pages {
            if let Err(e) = paging::map_page_in(self.root, va + k * PAGE_SIZE, start + k * PAGE_SIZE, bits) {
                self.unmap_range(va, k);
                let at = self.find_start(va).unwrap();
                self.remove(at);
                return Err(e.into());
            }
        }
        let at = self.find_start(va).unwrap();
        self.regions[at].committed = pages;
        self.committed_pages += pages;
        self.peak_committed_pages = self.peak_committed_pages.max(self.committed_pages);
        Ok((va + offset) as *mut u8)
    }
}

// ----------------------------------------------------------------------
// Public API: initialisation
// ----------------------------------------------------------------------

/// What [`init`] set up, for the boot log.
#[derive(Clone, Copy)]
pub struct InitInfo {
    pub kernel_root: u64,
    pub arena_base: u64,
    pub arena_size: u64,
    /// Does the CPU enforce no-execute?
    pub nx: bool,
    /// Top-level slots that had to be created so address spaces can share them.
    pub slots_created: u32,
}

/// Starts the VMM on the boot CPU. Needs the PMM and the boot page tables
/// (it adopts the active PML4 as the kernel's). Call before the first
/// [`AddressSpace`] is created, and after `hal::paging::init_pat`.
pub fn init() -> Result<InitInfo, VmError> {
    let mut v = VMM.lock();
    if v.ready {
        return Err(VmError::AlreadyInitialised);
    }
    let root = paging::current_root();
    // Every kernel-half slot we will ever use must exist now: address spaces
    // copy the kernel half of this PML4 and would miss slots added later.
    let mut created = 0;
    for slot_addr in [ARENA_BASE, paging::MMIO_BASE] {
        created += paging::ensure_pml4_slot(root, slot_addr)? as u32;
    }
    v.root = root;
    v.nx = paging::nx_enabled();
    v.ready = true;
    Ok(InitInfo {
        kernel_root: root,
        arena_base: ARENA_BASE,
        arena_size: ARENA_SIZE,
        nx: v.nx,
        slots_created: created,
    })
}

pub fn is_ready() -> bool {
    VMM.lock().ready
}

/// Physical address of the kernel's PML4.
pub fn kernel_root() -> u64 {
    VMM.lock().root
}

// ----------------------------------------------------------------------
// Public API: regions
// ----------------------------------------------------------------------

/// Reserves `pages` of address space (nothing is mapped yet) whose start is a
/// multiple of `align_pages` pages. Returns the first usable address.
pub fn reserve(pages: u64, align_pages: u64, tag: &'static str, guard: Guard) -> Result<u64, VmError> {
    VMM.lock().reserve(pages, align_pages, RegionKind::Memory, tag, guard)
}

/// Backs `[va, va + pages)` of a reserved region with fresh, zeroed frames.
/// All-or-nothing; fails with `AlreadyMapped` if any page is already committed.
pub fn commit(va: u64, pages: u64, prot: Prot) -> Result<(), VmError> {
    VMM.lock().commit(va, pages, prot)
}

/// Returns the frames behind `[va, va + pages)` to the PMM and unmaps the
/// range; the address space stays reserved. Pages that are not committed are
/// skipped. Returns how many frames were freed.
pub fn decommit(va: u64, pages: u64) -> Result<u64, VmError> {
    VMM.lock().decommit(va, pages)
}

/// Gives a whole region back: decommits what is committed and frees the
/// address space. `va` must be the start returned by [`reserve`] & co.
pub fn release(va: u64) -> Result<(), VmError> {
    VMM.lock().release(va)
}

/// Changes the protection of committed pages (all-or-nothing).
pub fn protect(va: u64, pages: u64, prot: Prot) -> Result<(), VmError> {
    VMM.lock().protect(va, pages, prot)
}

/// Reserve + commit in one step, with a guard page after the allocation.
pub fn alloc(pages: u64, prot: Prot, tag: &'static str) -> Result<*mut u8, VmError> {
    alloc_aligned(pages, 1, prot, tag)
}

/// Like [`alloc`], with the start aligned to `align_pages` pages.
pub fn alloc_aligned(pages: u64, align_pages: u64, prot: Prot, tag: &'static str) -> Result<*mut u8, VmError> {
    let mut v = VMM.lock();
    let va = v.reserve(pages, align_pages, RegionKind::Memory, tag, Guard::After)?;
    if let Err(e) = v.commit(va, pages, prot) {
        let _ = v.release(va);
        return Err(e);
    }
    Ok(va as *mut u8)
}

/// Frees an allocation made by [`alloc`] / [`alloc_aligned`]. Returns the
/// number of usable pages it had.
pub fn free(ptr: *mut u8) -> Result<u64, VmError> {
    let mut v = VMM.lock();
    let at = v.find_start(ptr as u64).ok_or(VmError::NotFound)?;
    if v.regions[at].kind != RegionKind::Memory {
        return Err(VmError::WrongKind);
    }
    let pages = v.regions[at].usable_pages();
    v.release(ptr as u64)?;
    Ok(pages)
}

/// Size in pages of the region starting exactly at `ptr`, if there is one.
pub fn region_pages(ptr: *mut u8) -> Option<u64> {
    let v = VMM.lock();
    v.find_start(ptr as u64).map(|i| v.regions[i].usable_pages())
}

/// Allocates a kernel stack of `pages` pages with a guard page below and above.
pub fn alloc_stack(pages: u64) -> Result<KernelStack, VmError> {
    let mut v = VMM.lock();
    let base = v.reserve(pages, 1, RegionKind::Stack, "kstack", Guard::Both)?;
    if let Err(e) = v.commit(base, pages, Prot::READ_WRITE) {
        let _ = v.release(base);
        return Err(e);
    }
    Ok(KernelStack { base, top: base + pages * PAGE_SIZE, pages })
}

pub fn free_stack(stack: KernelStack) -> Result<(), VmError> {
    VMM.lock().release(stack.base)
}

/// Maps `bytes` of existing physical memory (device registers, firmware
/// tables, a framebuffer) at a fresh virtual address, and returns the pointer
/// to the byte at `phys`. The frames are **not** owned: [`unmap_phys`] never
/// frees them. Use [`Prot::DEVICE`] for registers.
pub fn map_phys(phys: u64, bytes: u64, prot: Prot, tag: &'static str) -> Result<*mut u8, VmError> {
    VMM.lock().map_phys(phys, bytes, prot, tag)
}

/// Undoes [`map_phys`] (`ptr` is the pointer it returned).
pub fn unmap_phys(ptr: *mut u8) -> Result<(), VmError> {
    let mut v = VMM.lock();
    let page = ptr as u64 & !(PAGE_SIZE - 1);
    let at = v.find_start(page).ok_or(VmError::NotFound)?;
    if v.regions[at].kind != RegionKind::Physical {
        return Err(VmError::WrongKind);
    }
    v.release(page)
}

/// Physical address behind any mapped kernel virtual address (HHDM, kernel
/// image, arena), or `None` if it is not mapped.
pub fn translate(va: u64) -> Option<u64> {
    let root = VMM.lock().root;
    let m = paging::translate_in(root, va)?;
    Some(m.phys + (va & (PAGE_SIZE - 1)))
}

// ----------------------------------------------------------------------
// Public API: statistics and inspection
// ----------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub regions: usize,
    pub reserved_pages: u64,
    pub committed_pages: u64,
    pub peak_committed_pages: u64,
    pub reserve_failures: u64,
    pub commit_failures: u64,
}

pub fn stats() -> Stats {
    let v = VMM.lock();
    Stats {
        regions: v.count,
        reserved_pages: v.reserved_pages,
        committed_pages: v.committed_pages,
        peak_committed_pages: v.peak_committed_pages,
        reserve_failures: v.reserve_failures,
        commit_failures: v.commit_failures,
    }
}

/// Calls `f` for every region, in address order. `f` runs *without* the VMM
/// lock held (a snapshot is taken region by region), so it may log.
pub fn for_each_region(mut f: impl FnMut(RegionInfo)) {
    let mut i = 0;
    loop {
        let info = {
            let v = VMM.lock();
            if i >= v.count {
                return;
            }
            v.regions[i].info()
        };
        f(info);
        i += 1;
    }
}

/// Consistency check: table sorted, disjoint, inside the arena; committed
/// counters equal the number of pages actually mapped; guard pages unmapped.
pub fn verify() -> Result<(), &'static str> {
    let v = VMM.lock();
    v.ready().map_err(|_| "vmm not initialised")?;
    let mut prev_end = ARENA_BASE;
    let (mut reserved, mut committed) = (0u64, 0u64);
    for r in &v.regions[..v.count] {
        if r.total == 0 || r.start & (PAGE_SIZE - 1) != 0 {
            return Err("empty or misaligned region");
        }
        if r.start < prev_end {
            return Err("regions overlap or are unsorted");
        }
        if r.end() > ARENA_END {
            return Err("region leaves the arena");
        }
        prev_end = r.end();
        reserved += r.total;
        let mut mapped = 0;
        for k in 0..r.usable_pages() {
            if paging::translate_in(v.root, r.usable_start() + k * PAGE_SIZE).is_some() {
                mapped += 1;
            }
        }
        if mapped != r.committed {
            return Err("committed counter does not match the page tables");
        }
        committed += mapped;
        if r.lead != 0 && paging::translate_in(v.root, r.start).is_some() {
            return Err("leading guard page is mapped");
        }
        if r.trail != 0 && paging::translate_in(v.root, r.usable_end()).is_some() {
            return Err("trailing guard page is mapped");
        }
    }
    if reserved != v.reserved_pages || committed != v.committed_pages {
        return Err("global counters do not match the table");
    }
    Ok(())
}

// ----------------------------------------------------------------------
// Address spaces
// ----------------------------------------------------------------------

/// A complete virtual address space: its own user half, plus the kernel half
/// shared with every other space. This is the primitive the process manager
/// will build on; it does page-granular mapping only (placement policy -
/// which address a program's text or stack goes at - belongs to the loader).
///
/// All operations are limited to the user half ([`USER_END`]); kernel
/// mappings go through the global functions above and appear in every space
/// automatically.
pub struct AddressSpace {
    root: u64,
}

impl AddressSpace {
    /// An empty address space.
    pub fn new() -> Result<Self, VmError> {
        let kernel_root = {
            let v = VMM.lock();
            v.ready()?;
            v.root
        };
        Ok(Self { root: paging::create_root(kernel_root)? })
    }

    /// Physical address of this space's PML4 (what goes into CR3).
    pub fn root(&self) -> u64 {
        self.root
    }

    fn check_range(&self, va: u64, pages: u64) -> Result<(), VmError> {
        if va & (PAGE_SIZE - 1) != 0 {
            return Err(VmError::BadAddress);
        }
        if pages == 0 {
            return Err(VmError::BadSize);
        }
        let end = pages
            .checked_mul(PAGE_SIZE)
            .and_then(|b| va.checked_add(b))
            .ok_or(VmError::BadSize)?;
        if end > USER_END {
            return Err(VmError::BadAddress);
        }
        Ok(())
    }

    fn unmap_pages(&mut self, va: u64, pages: u64) -> (u64, u64) {
        let (mut unmapped, mut freed) = (0, 0);
        for k in 0..pages {
            if let Ok(u) = paging::unmap_page_in(self.root, va + k * PAGE_SIZE) {
                unmapped += 1;
                if u.entry & pte::OWNED != 0 {
                    freed += pmm::free_frame(u.phys).is_ok() as u64;
                }
            }
        }
        const BLOCK: u64 = 2 << 20;
        let mut a = va & !(BLOCK - 1);
        while a < va + pages * PAGE_SIZE {
            paging::reap_tables_in(self.root, a);
            a += BLOCK;
        }
        (unmapped, freed)
    }

    /// Maps `pages` fresh, zeroed, owned frames at `va`. All-or-nothing.
    pub fn map_new(&mut self, va: u64, pages: u64, prot: Prot) -> Result<(), VmError> {
        prot.check()?;
        self.check_range(va, pages)?;
        let nx = VMM.lock().nx;
        let bits = prot.pte_bits(nx) | pte::OWNED | pte::USER;
        for k in 0..pages {
            let page = va + k * PAGE_SIZE;
            let Some(frame) = pmm::alloc_zeroed() else {
                self.unmap_pages(va, k);
                return Err(VmError::OutOfFrames);
            };
            if let Err(e) = paging::map_page_in(self.root, page, frame, bits) {
                let _ = pmm::free_frame(frame);
                self.unmap_pages(va, k);
                return Err(e.into());
            }
        }
        Ok(())
    }

    /// Maps existing physical memory (not owned, never freed) at `va`.
    pub fn map_phys(&mut self, va: u64, phys: u64, pages: u64, prot: Prot) -> Result<(), VmError> {
        prot.check()?;
        self.check_range(va, pages)?;
        if phys & (PAGE_SIZE - 1) != 0 {
            return Err(VmError::BadAddress);
        }
        let nx = VMM.lock().nx;
        let bits = prot.pte_bits(nx) | pte::USER;
        for k in 0..pages {
            if let Err(e) = paging::map_page_in(self.root, va + k * PAGE_SIZE, phys + k * PAGE_SIZE, bits) {
                self.unmap_pages(va, k);
                return Err(e.into());
            }
        }
        Ok(())
    }

    /// Unmaps `[va, va + pages)` (holes are skipped), freeing owned frames.
    /// Returns the number of frames freed.
    pub fn unmap(&mut self, va: u64, pages: u64) -> Result<u64, VmError> {
        self.check_range(va, pages)?;
        Ok(self.unmap_pages(va, pages).1)
    }

    /// Changes the protection of mapped pages (all-or-nothing).
    pub fn protect(&mut self, va: u64, pages: u64, prot: Prot) -> Result<(), VmError> {
        prot.check()?;
        self.check_range(va, pages)?;
        for k in 0..pages {
            if paging::translate_in(self.root, va + k * PAGE_SIZE).is_none() {
                return Err(VmError::NotMapped);
            }
        }
        let nx = VMM.lock().nx;
        let bits = prot.pte_bits(nx) | pte::USER;
        for k in 0..pages {
            paging::protect_page_in(self.root, va + k * PAGE_SIZE, bits)?;
        }
        Ok(())
    }

    /// Physical address `va` maps to in this space.
    pub fn translate(&self, va: u64) -> Option<u64> {
        if va >= USER_END {
            return None;
        }
        let m = paging::translate_in(self.root, va)?;
        Some(m.phys + (va & (PAGE_SIZE - 1)))
    }

    /// Makes this the active address space on the calling CPU.
    ///
    /// # Safety
    /// The caller must keep the space alive while it is loaded on any CPU and
    /// switch back before [`AddressSpace::destroy`] / drop.
    pub unsafe fn activate(&self) {
        // SAFETY: the kernel half is shared, so the running code, stack, IDT,
        // GDT and TSS stay mapped; the caller upholds the lifetime rule.
        unsafe { paging::load_root(self.root) };
    }

    /// Is this the space loaded on the calling CPU?
    pub fn is_active(&self) -> bool {
        paging::current_root() == self.root
    }

    /// Frees every page table, every owned frame and the PML4. Returns the
    /// number of frames given back. Panics if the space is still active.
    pub fn destroy(mut self) -> u64 {
        assert!(!self.is_active(), "destroying the active address space");
        let root = core::mem::replace(&mut self.root, 0);
        paging::destroy_root(root)
    }
}

impl Drop for AddressSpace {
    fn drop(&mut self) {
        if self.root != 0 {
            assert!(!self.is_active(), "dropping the active address space");
            paging::destroy_root(self.root);
        }
    }
}

// ----------------------------------------------------------------------
// Boot self-test
// ----------------------------------------------------------------------

/// What [`self_test`] exercised, for the boot log.
#[derive(Clone, Copy)]
pub struct SelfTest {
    pub allocs: u32,
    pub guards_checked: u32,
    pub rejected: u32,
    pub tables_reclaimed: bool,
    pub address_space_frames: u64,
}

pub fn self_test() -> Result<SelfTest, &'static str> {
    let before = pmm::stats().in_use_frames;
    let regions_before = stats().regions;
    let mut t = SelfTest { allocs: 0, guards_checked: 0, rejected: 0, tables_reclaimed: false, address_space_frames: 0 };

    // 1) alloc: zeroed, writable, backed by distinct frames, guard behind it.
    let p = alloc(3, Prot::READ_WRITE, "selftest").map_err(|_| "alloc failed")? as *mut u64;
    t.allocs += 1;
    // SAFETY: three freshly committed pages are ours.
    unsafe {
        for i in 0..(3 * 512) {
            if p.add(i).read_volatile() != 0 {
                return Err("fresh pages are not zero");
            }
        }
        for i in 0..(3 * 512) {
            p.add(i).write_volatile(i as u64 ^ 0xA5A5);
        }
        for i in 0..(3 * 512) {
            if p.add(i).read_volatile() != i as u64 ^ 0xA5A5 {
                return Err("memory does not hold its contents");
            }
        }
    }
    let va = p as u64;
    let (f0, f1, f2) = (translate(va), translate(va + 4096), translate(va + 8192));
    if f0.is_none() || f1.is_none() || f2.is_none() || f0 == f1 || f1 == f2 {
        return Err("pages are not backed by distinct frames");
    }
    if translate(va + 3 * 4096).is_some() {
        return Err("no guard page behind the allocation");
    }
    t.guards_checked += 1;
    if region_pages(p as *mut u8) != Some(3) {
        return Err("region size lookup is wrong");
    }

    // 2) protect: read-only + NX, then back.
    protect(va, 1, Prot::READ).map_err(|_| "protect failed")?;
    let m = paging::translate_in(kernel_root(), va).ok_or("page vanished after protect")?;
    if m.entry & pte::WRITABLE != 0 {
        return Err("page still writable after protect");
    }
    if paging::nx_enabled() && m.entry & pte::NO_EXECUTE == 0 {
        return Err("page not NX after protect");
    }
    if m.entry & pte::OWNED == 0 {
        return Err("protect lost the ownership bit");
    }
    protect(va, 1, Prot::READ_WRITE).map_err(|_| "protect back failed")?;

    // 3) policy: W^X, user bit, overlap, bad free, bad commit.
    let wx = Prot { write: true, exec: true, user: false, cache: CacheMode::WriteBack };
    if protect(va, 1, wx) != Err(VmError::WriteExecute) {
        return Err("W^X not enforced");
    }
    if protect(va, 1, Prot::READ_WRITE.user()) != Err(VmError::UserInKernel) {
        return Err("USER allowed in the kernel half");
    }
    if commit(va, 1, Prot::READ_WRITE) != Err(VmError::AlreadyMapped) {
        return Err("double commit not rejected");
    }
    if commit(va + 2 * 4096, 2, Prot::READ_WRITE) != Err(VmError::OutOfRegion) {
        return Err("commit into the guard page not rejected");
    }
    if free((va + 4096) as *mut u8) != Err(VmError::NotFound) {
        return Err("free of a non-start address not rejected");
    }
    t.rejected += 5;

    // 3b) A commit that fails half-way (page 2 already mapped) rolls back fully.
    let q = reserve(4, 1, "selftest-rollback", Guard::After).map_err(|_| "reserve failed")?;
    commit(q + 2 * 4096, 1, Prot::READ_WRITE).map_err(|_| "commit failed")?;
    let held = pmm::stats().in_use_frames;
    if commit(q, 4, Prot::READ_WRITE) != Err(VmError::AlreadyMapped) {
        return Err("overlapping commit not rejected");
    }
    if pmm::stats().in_use_frames != held || translate(q).is_some()
        || translate(q + 4096).is_some()
        || translate(q + 2 * 4096).is_none()
    {
        return Err("failed commit was not rolled back");
    }
    release(q).map_err(|_| "release failed")?;
    t.rejected += 1;

    // 4) decommit / commit on a partial range keeps the rest intact.
    if decommit(va + 4096, 1) != Ok(1) {
        return Err("decommit did not free exactly one frame");
    }
    if translate(va + 4096).is_some() || translate(va).is_none() || translate(va + 8192).is_none() {
        return Err("decommit hit the wrong page");
    }
    commit(va + 4096, 1, Prot::READ_WRITE).map_err(|_| "re-commit failed")?;
    // SAFETY: page is committed again, and zeroed.
    if unsafe { (va as *const u64).add(512).read_volatile() } != 0 {
        return Err("re-committed page is not zeroed");
    }
    verify()?;

    // 5) free: everything goes back, including the page tables.
    if free(p as *mut u8) != Ok(3) {
        return Err("free reported the wrong size");
    }
    if translate(va).is_some() {
        return Err("memory still mapped after free");
    }
    t.tables_reclaimed = pmm::stats().in_use_frames == before;
    if !t.tables_reclaimed {
        return Err("map/unmap cycle leaked frames (page tables not reclaimed?)");
    }

    // 6) reserve without commit, alignment, then commit later.
    let r = reserve(16, 16, "selftest-reserve", Guard::Both).map_err(|_| "reserve failed")?;
    if r & (16 * 4096 - 1) != 0 {
        return Err("reserve ignored the alignment");
    }
    if translate(r).is_some() {
        return Err("reserve mapped something");
    }
    commit(r, 4, Prot::READ_EXEC).map_err(|_| "commit in reserved region failed")?;
    release(r).map_err(|_| "release failed")?;
    t.allocs += 1;

    // 7) stack with two guard pages.
    let s = alloc_stack(4).map_err(|_| "alloc_stack failed")?;
    if translate(s.base - 4096).is_some() || translate(s.top).is_some() {
        return Err("stack guard pages are mapped");
    }
    if translate(s.base).is_none() || translate(s.top - 1).is_none() {
        return Err("stack is not mapped");
    }
    t.guards_checked += 2;
    free_stack(s).map_err(|_| "free_stack failed")?;

    // 8) map_phys: alias a frame we own; freeing the alias must keep the frame.
    let frame = pmm::alloc_zeroed().ok_or("out of frames")?;
    let alias = map_phys(frame + 0x10, 8, Prot::READ_WRITE, "selftest-alias").map_err(|_| "map_phys failed")?;
    // SAFETY: the alias and the HHDM view address the same private frame.
    unsafe {
        (alias as *mut u64).write_volatile(0xDEAD_BEEF_CAFE_F00D);
        let direct = crate::hal::memory::phys_to_virt(frame + 0x10) as *const u64;
        if direct.read_volatile() != 0xDEAD_BEEF_CAFE_F00D {
            return Err("map_phys alias does not see the frame");
        }
    }
    unmap_phys(alias).map_err(|_| "unmap_phys failed")?;
    if pmm::state(frame) != pmm::FrameState::Used {
        return Err("unmap_phys freed a frame it does not own");
    }
    pmm::free_frame(frame).map_err(|_| "could not return the test frame")?;

    // 9) address spaces: isolation, shared kernel half, full reclamation.
    let space_before = pmm::stats().in_use_frames;
    let mut a = AddressSpace::new().map_err(|_| "address space creation failed")?;
    let mut b = AddressSpace::new().map_err(|_| "second address space creation failed")?;
    const UVA: u64 = 0x0000_0000_0040_0000;
    a.map_new(UVA, 2, Prot::READ_WRITE.user()).map_err(|_| "user map_new failed")?;
    if a.map_new(KERNEL_HALF, 1, Prot::READ_WRITE) != Err(VmError::BadAddress) {
        return Err("user space may map the kernel half");
    }
    t.rejected += 1;
    let frame_a = a.translate(UVA).ok_or("user page not mapped")?;
    if b.translate(UVA).is_some() {
        return Err("address spaces are not isolated");
    }
    b.map_new(UVA, 1, Prot::READ_WRITE.user()).map_err(|_| "map in second space failed")?;
    let frame_b = b.translate(UVA).ok_or("second space lost its page")?;
    if frame_a == frame_b {
        return Err("two spaces share a private frame");
    }
    // SAFETY: both frames are ours, zeroed, reachable through the HHDM.
    unsafe {
        (crate::hal::memory::phys_to_virt(frame_a) as *mut u64).write_volatile(0x1111);
        (crate::hal::memory::phys_to_virt(frame_b) as *mut u64).write_volatile(0x2222);
    }
    let saved = paging::current_root();
    // SAFETY: both spaces outlive the switch, and we return to `saved` before
    // they are dropped. Interrupts stay off: the kernel half is shared, but
    // there is no reason to be preempted while another space is loaded.
    // Running at all while another space is loaded proves the kernel half
    // (code, stack, IDT) is shared.
    let (seen_a, seen_b) = unsafe {
        let irq = IrqLock::new(());
        let _g = irq.lock();
        a.activate();
        let seen_a = (UVA as *const u64).read_volatile();
        b.activate();
        let seen_b = (UVA as *const u64).read_volatile();
        paging::load_root(saved);
        (seen_a, seen_b)
    };
    if seen_a != 0x1111 || seen_b != 0x2222 {
        return Err("a space does not see its own memory after activation");
    }
    // A kernel mapping made *after* both spaces exist must be visible in them.
    let late = alloc(1, Prot::READ_WRITE, "selftest-late").map_err(|_| "late alloc failed")?;
    // SAFETY: one committed page.
    unsafe { (late as *mut u64).write_volatile(0x3333) };
    // SAFETY: as above.
    let late_seen = unsafe {
        a.activate();
        let v = (late as *const u64).read_volatile();
        paging::load_root(saved);
        v
    };
    free(late).map_err(|_| "free of late alloc failed")?;
    if late_seen != 0x3333 {
        return Err("kernel mapping made after clone is not shared");
    }
    a.protect(UVA, 2, Prot::READ.user()).map_err(|_| "user protect failed")?;
    let freed = a.unmap(UVA, 2).map_err(|_| "user unmap failed")?;
    if freed != 2 || a.translate(UVA).is_some() {
        return Err("user unmap did not release its frames");
    }
    a.map_new(UVA, 1, Prot::READ_WRITE.user()).map_err(|_| "remap failed")?;
    t.address_space_frames = a.destroy() + b.destroy();
    if pmm::stats().in_use_frames != space_before {
        return Err("address spaces leaked frames");
    }

    // 10) books balance.
    if pmm::stats().in_use_frames != before || stats().regions != regions_before {
        return Err("self-test left frames or regions behind");
    }
    verify()?;
    Ok(t)
}
