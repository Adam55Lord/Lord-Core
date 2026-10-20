// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Kernel heap: the `#[global_allocator]`, so `Box`, `Vec`, `String`,
//! `BTreeMap` & co. work anywhere in the kernel after [`init`].
//!
//! # Design
//!
//! Two paths, chosen by size:
//!
//! * **Small** (up to [`MAX_SMALL`] = 2048 bytes, alignment up to the same):
//!   eight power-of-two **size classes** (16, 32, ..., 2048). Each class has
//!   a LIFO free list of equally sized blocks, carved from 4 KiB pages taken,
//!   on demand, from one big virtual **heap arena** that was *reserved* up
//!   front ([`ARENA_PAGES`] = 256 MiB) and is *committed* 64 KiB at a time.
//!   Alloc and free are a pop and a push. A block of size `2^k` always starts
//!   at a multiple of `2^k`, so its natural alignment is free.
//! * **Large** (anything bigger, or aligned beyond 2048): its own VMM region,
//!   page-granular, with a guard page behind it; the frames go straight back
//!   to the PMM on free. Fresh frames are zeroed, so `alloc_zeroed` is free
//!   here and no data ever leaks between allocations.
//!
//! # Hardening
//!
//! Every free block carries a canary (`next` pointer + `canary ^ address`):
//!
//! * a **double free** finds the canary and is ignored and counted;
//! * a pointer outside the heap, misaligned for its class, or not owned by
//!   the VMM is ignored and counted (`bad_frees`) instead of corrupting a list;
//! * [`verify`] walks all lists and checks `carved == free + live` per class.
//!
//! # Trade-offs
//!
//! * Small-object pages are **kept** once carved (they are reused, never
//!   returned to the PMM). Large allocations are fully returned.
//! * Internal fragmentation is up to 50 % for small sizes (power-of-two classes).
//! * Realloc is in place only inside one class; otherwise alloc + copy + free.
//!
//! # Concurrency
//!
//! One [`IrqLock`] (interrupts masked while held): allocating from an
//! interrupt handler is allowed, but it can still *fail*, so handlers should
//! not rely on it. Lock order is `heap -> vmm -> pmm`.
//!
//! Until [`init`] succeeds, every allocation returns null (Rust then panics
//! with "memory allocation failed" - a loud failure for allocating too early).

use core::alloc::{GlobalAlloc, Layout};
use core::ptr;

use super::vmm::{self, Guard, Prot, PAGE_SIZE};
use crate::hal::sync::IrqLock;

/// Largest size (and alignment) served by the size classes.
pub const MAX_SMALL: usize = 2048;
const MIN_SHIFT: u32 = 4;
const CLASSES: usize = 8;
/// Virtual pages reserved for the small-object arena (256 MiB).
pub const ARENA_PAGES: u64 = 65536;
/// Pages committed at a time when the arena needs to grow.
const CHUNK_PAGES: u64 = 16;

const CANARY: usize = 0xF4EE_B10C_DEAD_C0DE;

#[repr(C)]
struct FreeBlock {
    next: usize,
    canary: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeapError {
    VmmNotReady,
    AlreadyInitialised,
    Vmm(vmm::VmError),
}

struct Heap {
    ready: bool,
    /// Arena: `[base, end)`; `[base, committed_end)` is mapped,
    /// `[base, carve_next)` has been cut into blocks.
    base: usize,
    end: usize,
    committed_end: usize,
    carve_next: usize,
    free_head: [usize; CLASSES],
    free_count: [u64; CLASSES],
    live_count: [u64; CLASSES],
    carved_count: [u64; CLASSES],
    // statistics
    live_bytes: u64,
    peak_live_bytes: u64,
    allocs: u64,
    frees: u64,
    large_live: u64,
    large_live_bytes: u64,
    failed: u64,
    bad_frees: u64,
    double_frees: u64,
}

static HEAP: IrqLock<Heap> = IrqLock::new(Heap {
    ready: false,
    base: 0,
    end: 0,
    committed_end: 0,
    carve_next: 0,
    free_head: [0; CLASSES],
    free_count: [0; CLASSES],
    live_count: [0; CLASSES],
    carved_count: [0; CLASSES],
    live_bytes: 0,
    peak_live_bytes: 0,
    allocs: 0,
    frees: 0,
    large_live: 0,
    large_live_bytes: 0,
    failed: 0,
    bad_frees: 0,
    double_frees: 0,
});

/// Size class for a request, or `None` if it belongs on the large path.
fn class_of(size: usize, align: usize) -> Option<usize> {
    let n = size.max(align).max(1 << MIN_SHIFT).checked_next_power_of_two()?;
    if n > MAX_SMALL { None } else { Some((n.trailing_zeros() - MIN_SHIFT) as usize) }
}

fn class_size(class: usize) -> usize {
    1 << (class as u32 + MIN_SHIFT)
}

fn canary_for(addr: usize) -> usize {
    CANARY ^ addr
}

impl Heap {
    fn in_arena(&self, addr: usize) -> bool {
        addr >= self.base && addr < self.end
    }

    fn note_alloc(&mut self, bytes: usize) {
        self.allocs += 1;
        self.live_bytes += bytes as u64;
        self.peak_live_bytes = self.peak_live_bytes.max(self.live_bytes);
    }

    /// Makes sure one more page can be carved, committing a chunk if needed.
    fn grow(&mut self) -> bool {
        if self.carve_next < self.committed_end {
            return true;
        }
        let room = ((self.end - self.committed_end) as u64) / PAGE_SIZE;
        for want in [CHUNK_PAGES, 1] {
            let pages = want.min(room);
            if pages == 0 {
                return false;
            }
            if vmm::commit(self.committed_end as u64, pages, Prot::READ_WRITE).is_ok() {
                self.committed_end += (pages * PAGE_SIZE) as usize;
                return true;
            }
        }
        false
    }

    /// Cuts one fresh page into blocks of `class` and queues them.
    fn refill(&mut self, class: usize) -> bool {
        if !self.grow() {
            return false;
        }
        let page = self.carve_next;
        self.carve_next += PAGE_SIZE as usize;
        let size = class_size(class);
        let n = PAGE_SIZE as usize / size;
        // Push in reverse so the list hands out ascending addresses.
        for i in (0..n).rev() {
            let b = page + i * size;
            // SAFETY: `b` is inside a page we just committed; blocks are >= 16 bytes.
            unsafe {
                (b as *mut FreeBlock).write(FreeBlock { next: self.free_head[class], canary: canary_for(b) });
            }
            self.free_head[class] = b;
        }
        self.free_count[class] += n as u64;
        self.carved_count[class] += n as u64;
        true
    }

    fn alloc_small(&mut self, class: usize, bytes: usize) -> *mut u8 {
        if self.free_head[class] == 0 && !self.refill(class) {
            self.failed += 1;
            return ptr::null_mut();
        }
        let b = self.free_head[class];
        // SAFETY: `b` heads the free list: a valid block carved by `refill`.
        let blk = unsafe { &mut *(b as *mut FreeBlock) };
        self.free_head[class] = blk.next;
        blk.canary = 0; // a live block must never look free
        blk.next = 0;
        self.free_count[class] -= 1;
        self.live_count[class] += 1;
        self.note_alloc(bytes);
        b as *mut u8
    }

    fn free_small(&mut self, p: usize, class: usize, bytes: usize) {
        let size = class_size(class);
        if !self.in_arena(p) || p >= self.carve_next || (p - self.base) % size != 0 {
            self.bad_frees += 1;
            return;
        }
        // SAFETY: `p` is a block-aligned address inside carved memory.
        let blk = unsafe { &mut *(p as *mut FreeBlock) };
        if blk.canary == canary_for(p) {
            self.double_frees += 1;
            return;
        }
        blk.next = self.free_head[class];
        blk.canary = canary_for(p);
        self.free_head[class] = p;
        self.free_count[class] += 1;
        self.live_count[class] -= 1;
        self.frees += 1;
        self.live_bytes -= bytes as u64;
    }
}

// ----------------------------------------------------------------------
// Public API
// ----------------------------------------------------------------------

/// What [`init`] set up, for the boot log.
#[derive(Clone, Copy)]
pub struct InitInfo {
    pub arena_base: u64,
    pub arena_pages: u64,
}

/// Reserves the small-object arena. Needs the VMM. After this returns `Ok`
/// the global allocator works. Boot CPU, once.
pub fn init() -> Result<InitInfo, HeapError> {
    if !vmm::is_ready() {
        return Err(HeapError::VmmNotReady);
    }
    let mut h = HEAP.lock();
    if h.ready {
        return Err(HeapError::AlreadyInitialised);
    }
    let base = vmm::reserve(ARENA_PAGES, 1, "heap", Guard::After).map_err(HeapError::Vmm)?;
    h.base = base as usize;
    h.end = h.base + (ARENA_PAGES * PAGE_SIZE) as usize;
    h.committed_end = h.base;
    h.carve_next = h.base;
    h.ready = true;
    Ok(InitInfo { arena_base: base, arena_pages: ARENA_PAGES })
}

pub fn is_ready() -> bool {
    HEAP.lock().ready
}

#[derive(Clone, Copy, Debug)]
pub struct Stats {
    /// Bytes requested and not yet freed (what callers asked for).
    pub live_bytes: u64,
    pub peak_live_bytes: u64,
    pub allocs: u64,
    pub frees: u64,
    /// Pages of the arena committed / carved into blocks.
    pub arena_committed_pages: u64,
    pub arena_carved_pages: u64,
    /// Live large allocations and their requested bytes.
    pub large_live: u64,
    pub large_live_bytes: u64,
    /// Allocations that returned null.
    pub failed: u64,
    /// Frees of pointers that were not ours / not valid (ignored).
    pub bad_frees: u64,
    /// Frees of blocks that were already free (ignored).
    pub double_frees: u64,
    /// Blocks currently on the free lists, per class.
    pub free_blocks: [u64; CLASSES],
    pub live_blocks: [u64; CLASSES],
}

pub fn stats() -> Stats {
    let h = HEAP.lock();
    Stats {
        live_bytes: h.live_bytes,
        peak_live_bytes: h.peak_live_bytes,
        allocs: h.allocs,
        frees: h.frees,
        arena_committed_pages: ((h.committed_end - h.base) as u64) / PAGE_SIZE,
        arena_carved_pages: ((h.carve_next - h.base) as u64) / PAGE_SIZE,
        large_live: h.large_live,
        large_live_bytes: h.large_live_bytes,
        failed: h.failed,
        bad_frees: h.bad_frees,
        double_frees: h.double_frees,
        free_blocks: h.free_count,
        live_blocks: h.live_count,
    }
}

/// Walks every free list and checks the books: nodes inside carved memory,
/// aligned, canary intact, no cycles, and `carved == free + live` per class.
pub fn verify() -> Result<(), &'static str> {
    let h = HEAP.lock();
    if !h.ready {
        return Err("heap not initialised");
    }
    for class in 0..CLASSES {
        let size = class_size(class);
        let mut n = 0u64;
        let mut p = h.free_head[class];
        while p != 0 {
            if !h.in_arena(p) || p >= h.carve_next || (p - h.base) % size != 0 {
                return Err("free-list node outside the arena or misaligned");
            }
            // SAFETY: bounds and alignment were just checked.
            let blk = unsafe { &*(p as *const FreeBlock) };
            if blk.canary != canary_for(p) {
                return Err("free block lost its canary (write after free?)");
            }
            n += 1;
            if n > h.carved_count[class] {
                return Err("free list longer than the blocks carved (cycle?)");
            }
            p = blk.next;
        }
        if n != h.free_count[class] {
            return Err("free-list length does not match its counter");
        }
        if h.carved_count[class] != h.free_count[class] + h.live_count[class] {
            return Err("carved != free + live");
        }
    }
    let pages: u64 = (0..CLASSES).map(|c| h.carved_count[c] * class_size(c) as u64).sum::<u64>() / PAGE_SIZE;
    if pages != ((h.carve_next - h.base) as u64) / PAGE_SIZE {
        return Err("carved blocks do not add up to the carved pages");
    }
    Ok(())
}

// ----------------------------------------------------------------------
// The global allocator
// ----------------------------------------------------------------------

pub struct KernelAllocator;

#[global_allocator]
static ALLOCATOR: KernelAllocator = KernelAllocator;

fn large_pages(layout: &Layout) -> Option<(u64, u64)> {
    let pages = (layout.size() as u64).checked_add(PAGE_SIZE - 1)? / PAGE_SIZE;
    let align_pages = (layout.align() as u64).div_ceil(PAGE_SIZE).max(1);
    Some((pages.max(1), align_pages))
}

impl KernelAllocator {
    fn alloc_inner(&self, layout: Layout) -> *mut u8 {
        let size = layout.size().max(1);
        if let Some(class) = class_of(size, layout.align()) {
            let mut h = HEAP.lock();
            if !h.ready {
                return ptr::null_mut();
            }
            return h.alloc_small(class, layout.size());
        }
        // Large path: no heap lock while the VMM works.
        let ok = HEAP.lock().ready;
        let result = match large_pages(&layout) {
            Some((pages, align_pages)) if ok => vmm::alloc_aligned(pages, align_pages, Prot::READ_WRITE, "heap-large").ok(),
            _ => None,
        };
        let mut h = HEAP.lock();
        match result {
            Some(p) => {
                h.large_live += 1;
                h.large_live_bytes += layout.size() as u64;
                h.note_alloc(layout.size());
                p
            }
            None => {
                if ok {
                    h.failed += 1;
                }
                ptr::null_mut()
            }
        }
    }

    fn dealloc_inner(&self, p: *mut u8, layout: Layout) {
        let size = layout.size().max(1);
        if let Some(class) = class_of(size, layout.align()) {
            HEAP.lock().free_small(p as usize, class, layout.size());
            return;
        }
        {
            // A large block can never live in the small arena.
            let mut h = HEAP.lock();
            if !h.ready || h.in_arena(p as usize) {
                h.bad_frees += 1;
                return;
            }
        }
        let freed = vmm::free(p);
        let mut h = HEAP.lock();
        match freed {
            Ok(_) => {
                h.large_live -= 1;
                h.large_live_bytes -= layout.size() as u64;
                h.frees += 1;
                h.live_bytes -= layout.size() as u64;
            }
            Err(_) => h.bad_frees += 1,
        }
    }
}

// SAFETY: `alloc` returns memory that is unaliased, correctly sized and
// aligned (see the module docs), `dealloc` only recycles blocks it handed
// out, and every shared structure is behind an interrupt-safe lock.
unsafe impl GlobalAlloc for KernelAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.alloc_inner(layout)
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        self.dealloc_inner(p, layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = self.alloc_inner(layout);
        // Large allocations come from freshly zeroed frames already.
        if !p.is_null() && class_of(layout.size().max(1), layout.align()).is_some() {
            // SAFETY: `p` is a live block of at least `layout.size()` bytes.
            unsafe { ptr::write_bytes(p, 0, layout.size()) };
        }
        p
    }

    unsafe fn realloc(&self, p: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller guarantees `new_size` is valid for this alignment.
        let new_layout = unsafe { Layout::from_size_align_unchecked(new_size, layout.align()) };
        let old_class = class_of(layout.size().max(1), layout.align());
        let new_class = class_of(new_size.max(1), layout.align());
        match (old_class, new_class) {
            // Same size class: the block already fits.
            (Some(a), Some(b)) if a == b => {
                let mut h = HEAP.lock();
                h.live_bytes = h.live_bytes - layout.size() as u64 + new_size as u64;
                h.peak_live_bytes = h.peak_live_bytes.max(h.live_bytes);
                return p;
            }
            // Same page count, both large: the mapping already fits.
            (None, None) if large_pages(&layout).map(|x| x.0) == large_pages(&new_layout).map(|x| x.0) => {
                let mut h = HEAP.lock();
                h.live_bytes = h.live_bytes - layout.size() as u64 + new_size as u64;
                h.large_live_bytes = h.large_live_bytes - layout.size() as u64 + new_size as u64;
                h.peak_live_bytes = h.peak_live_bytes.max(h.live_bytes);
                return p;
            }
            _ => {}
        }
        let q = self.alloc_inner(new_layout);
        if !q.is_null() {
            // SAFETY: both blocks are live, distinct, and at least `min` bytes.
            unsafe { ptr::copy_nonoverlapping(p, q, layout.size().min(new_size)) };
            self.dealloc_inner(p, layout);
        }
        q
    }
}

// ----------------------------------------------------------------------
// Boot self-test
// ----------------------------------------------------------------------

/// What [`self_test`] exercised, for the boot log.
#[derive(Clone, Copy)]
pub struct SelfTest {
    pub stress_ops: u32,
    pub largest_vec: usize,
    pub bad_frees_rejected: u32,
}

fn lcg(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state >> 33
}

pub fn self_test() -> Result<SelfTest, &'static str> {
    use alloc::boxed::Box;
    use alloc::collections::BTreeMap;
    use alloc::format;
    use alloc::string::String;
    use alloc::vec::Vec;

    let base = stats();
    let vmm_before = vmm::stats();
    let mut out = SelfTest { stress_ops: 0, largest_vec: 0, bad_frees_rejected: 0 };

    // 1) The basics: Box, String, format!.
    {
        let b = Box::new(0x1122_3344_5566_7788u64);
        if *b != 0x1122_3344_5566_7788 {
            return Err("Box lost its value");
        }
        let s = format!("{}-{}-{}", "Lord", "Core", 7);
        if s != "Lord-Core-7" {
            return Err("format! produced the wrong string");
        }
        let mut t = String::new();
        for i in 0..500 {
            t.push_str(&format!("{i},"));
        }
        if t.len() != 1890 || !t.ends_with("499,") {
            return Err("String growth lost data");
        }
    }

    // 2) Vec growth from tiny, through every size class, to a multi-page
    //    (large-path) buffer, with realloc preserving the contents.
    {
        let mut v: Vec<u64> = Vec::new();
        for i in 0..100_000u64 {
            v.push(i * 3 + 1);
        }
        out.largest_vec = v.capacity() * 8;
        if v.iter().enumerate().any(|(i, &x)| x != i as u64 * 3 + 1) {
            return Err("Vec lost data while growing");
        }
        v.truncate(10);
        v.shrink_to_fit();
        if v.len() != 10 || v[9] != 28 {
            return Err("Vec lost data while shrinking");
        }
    }

    // 3) BTreeMap: many small nodes, inserted and removed out of order.
    {
        let mut m = BTreeMap::new();
        for i in 0..3000u32 {
            m.insert(i.wrapping_mul(2654435761), i);
        }
        for i in (0..3000u32).step_by(2) {
            if m.remove(&i.wrapping_mul(2654435761)) != Some(i) {
                return Err("BTreeMap lost an entry");
            }
        }
        if m.len() != 1500 {
            return Err("BTreeMap has the wrong length");
        }
    }

    // 4) Alignment, small and large, and that memory is usable.
    for &align in &[16usize, 64, 256, 2048, 4096, 8192, 65536] {
        for &size in &[1usize, 100, 2048, 5000] {
            let layout = Layout::from_size_align(size, align).map_err(|_| "bad layout")?;
            // SAFETY: valid non-zero layout; the block is written within bounds and freed.
            unsafe {
                let p = ALLOCATOR.alloc(layout);
                if p.is_null() {
                    return Err("allocation failed");
                }
                if p as usize % align != 0 {
                    return Err("allocation is misaligned");
                }
                ptr::write_bytes(p, 0xAB, size);
                ALLOCATOR.dealloc(p, layout);
            }
        }
    }

    // 5) alloc_zeroed must zero a recycled (dirty) block as well as a large one.
    for &size in &[48usize, 3000, 20000] {
        let layout = Layout::from_size_align(size, 8).map_err(|_| "bad layout")?;
        // SAFETY: valid layout; blocks are used within bounds and freed.
        unsafe {
            let p = ALLOCATOR.alloc(layout);
            ptr::write_bytes(p, 0xFF, size);
            ALLOCATOR.dealloc(p, layout);
            let q = ALLOCATOR.alloc_zeroed(layout);
            if q.is_null() || (0..size).any(|i| q.add(i).read() != 0) {
                return Err("alloc_zeroed returned dirty memory");
            }
            ALLOCATOR.dealloc(q, layout);
        }
    }

    // 6) Invalid frees are rejected, not obeyed.
    {
        let layout = Layout::from_size_align(64, 8).map_err(|_| "bad layout")?;
        let big = Layout::from_size_align(10_000, 8).map_err(|_| "bad layout")?;
        let before = stats();
        // SAFETY: deliberately invalid calls: the allocator must tolerate them.
        unsafe {
            let p = ALLOCATOR.alloc(layout);
            ALLOCATOR.dealloc(p, layout);
            ALLOCATOR.dealloc(p, layout); // double free
            ALLOCATOR.dealloc(p.add(8), layout); // misaligned for its class
            ALLOCATOR.dealloc(0x1000 as *mut u8, layout); // not ours
            ALLOCATOR.dealloc(0x1000 as *mut u8, big); // not ours, large
        }
        let after = stats();
        if after.double_frees != before.double_frees + 1 || after.bad_frees != before.bad_frees + 3 {
            return Err("invalid frees were not detected");
        }
        out.bad_frees_rejected = 4;
        verify()?;
    }

    // 7) Randomised stress: mixed sizes, pattern-checked, 20 000 operations.
    {
        const SLOTS: usize = 192;
        let mut slots: [(usize, usize, u8); SLOTS] = [(0, 0, 0); SLOTS];
        let mut rng = 0x1234_5678_9ABC_DEF0u64;
        for op in 0..20_000u32 {
            let i = lcg(&mut rng) as usize % SLOTS;
            let (p, size, pat) = slots[i];
            if p != 0 {
                let layout = Layout::from_size_align(size, 8).unwrap();
                // SAFETY: slot holds a live block of `size` bytes we filled.
                unsafe {
                    for k in [0, size / 2, size - 1] {
                        if (p as *const u8).add(k).read() != pat {
                            return Err("stress: block contents changed");
                        }
                    }
                    ALLOCATOR.dealloc(p as *mut u8, layout);
                }
                slots[i] = (0, 0, 0);
            } else {
                let r = lcg(&mut rng);
                let size = match r % 16 {
                    0 => 2049 + (lcg(&mut rng) as usize % 70_000),
                    1..=3 => 257 + (lcg(&mut rng) as usize % 1792),
                    _ => 1 + (lcg(&mut rng) as usize % 256),
                };
                let layout = Layout::from_size_align(size, 8).unwrap();
                // SAFETY: valid layout; the whole block is written then recorded.
                unsafe {
                    let q = ALLOCATOR.alloc(layout);
                    if q.is_null() {
                        return Err("stress: allocation failed");
                    }
                    let pat = (op as u8) | 1;
                    ptr::write_bytes(q, pat, size);
                    slots[i] = (q as usize, size, pat);
                }
            }
            out.stress_ops += 1;
        }
        for (p, size, pat) in slots {
            if p != 0 {
                // SAFETY: still-live blocks, verified then freed.
                unsafe {
                    if (p as *const u8).add(size - 1).read() != pat {
                        return Err("stress: tail of a block changed");
                    }
                    ALLOCATOR.dealloc(p as *mut u8, Layout::from_size_align(size, 8).unwrap());
                }
            }
        }
    }

    // 7b) Out of memory: asking for more than the machine has must fail
    //     cleanly (null, nothing leaked), not hang or corrupt anything.
    {
        let frames = crate::mm::pmm::stats().free_frames;
        let in_use = crate::mm::pmm::stats().in_use_frames;
        let layout = Layout::from_size_align(((frames + 1) * PAGE_SIZE) as usize, 8).map_err(|_| "bad layout")?;
        // SAFETY: valid layout; a null result is the expected outcome.
        let p = unsafe { ALLOCATOR.alloc(layout) };
        if !p.is_null() {
            return Err("an impossible allocation succeeded");
        }
        if crate::mm::pmm::stats().in_use_frames != in_use {
            return Err("a failed allocation leaked frames");
        }
    }

    // 8) The books must balance.
    verify()?;
    let end = stats();
    if end.live_bytes != base.live_bytes || end.large_live != base.large_live || end.large_live_bytes != base.large_live_bytes {
        return Err("self-test leaked heap memory");
    }
    let vmm_after = vmm::stats();
    let grown = end.arena_committed_pages - base.arena_committed_pages;
    if vmm_after.committed_pages != vmm_before.committed_pages + grown || vmm_after.regions != vmm_before.regions {
        return Err("large allocations did not return their pages and regions");
    }
    vmm::verify()?;
    // The deliberate bad frees and the impossible allocation above are not
    // real faults: keep them out of the diagnostics counters.
    {
        let mut h = HEAP.lock();
        h.bad_frees = base.bad_frees;
        h.double_frees = base.double_frees;
        h.failed = base.failed;
    }
    Ok(out)
}
