// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Physical Memory Manager (PMM): owns every 4 KiB frame of usable RAM.
//!
//! # Design
//!
//! One **bitmap**, one bit per physical frame: `1` = free, `0` = in use *or
//! not managed at all* (holes, firmware, the kernel image, the first MiB).
//! Starting with "everything used" and then marking the usable ranges free
//! means a frame we never heard of can never be handed out by mistake.
//!
//! * The bitmap costs 1/32768 of the RAM it describes (512 KiB for 16 GiB)
//!   and is carved out of the **end of the highest usable range that fits**,
//!   so low memory (precious for legacy / 32-bit-DMA devices) stays free.
//! * Allocation is **first-fit** from a *low-water hint*: the invariant is
//!   "every frame below `hint` is in use", so scans skip the used prefix.
//!   Addresses therefore come out in ascending order, which is also easy to
//!   read in a debugger.
//! * Contiguous, **aligned** runs and an upper address limit (`max_end`, for
//!   DMA32 / 24-bit-ISA style constraints) are supported by the same search.
//! * Freeing is validated: misaligned addresses, frames outside managed
//!   memory, the PMM's own metadata, and **double frees** are all rejected
//!   with an error and counted instead of corrupting the bitmap.
//!
//! # What it does not manage
//!
//! * Memory below 1 MiB ([`FLOOR`]).
//! * `BootloaderReclaimable` memory: the bootloader's page tables (still in use,
//!   Lord Core edits them in place), stack and response structures live there.
//!   Reclaiming it is a documented future step.
//! * Everything that is not `Usable` in the bootloader memory map.
//!
//! # Concurrency
//!
//! One global [`IrqLock`] (interrupts masked while held, so the preemptive scheduler can never switch away inside it): do not call into
//! the PMM from an interrupt handler only if the interrupted code cannot be
//! the PMM itself (the lock masks interrupts, so this cannot happen).
//!
//! Call [`init`] once on the boot CPU, after `hal::memory::init`. It needs
//! only the HHDM, not paging, so it can run before the IDT exists.

use core::ptr;

use crate::hal::memory::{self, RegionKind};
use crate::hal::sync::IrqLock;

/// Size of a physical frame.
pub const FRAME_SIZE: u64 = 4096;
const FRAME_SHIFT: u32 = 12;
/// Nothing below this physical address is managed (legacy low memory).
pub const FLOOR: u64 = 0x10_0000;
/// `max_end` for devices limited to 24-bit addresses.
pub const LIMIT_16M: u64 = 16 << 20;
/// `max_end` for devices limited to 32-bit addresses (DMA32).
pub const LIMIT_4G: u64 = 4 << 30;
/// `max_end` meaning "anywhere".
pub const NO_LIMIT: u64 = u64::MAX;

/// Capacity of the managed-range table. Real machines report a handful.
const MAX_RANGES: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PmmError {
    /// [`init`] has not completed.
    NotInitialised,
    /// [`init`] was already called.
    AlreadyInitialised,
    /// The memory map has no usable memory above [`FLOOR`].
    NoUsableMemory,
    /// No usable range is large enough to hold the bitmap.
    NoSpaceForMetadata,
    /// Address is not a multiple of [`FRAME_SIZE`].
    Misaligned,
    /// The frame (or part of the run) is not in managed memory.
    OutOfRange,
    /// The run overlaps the PMM's own bitmap.
    Reserved,
    /// A frame in the run is already free.
    DoubleFree,
    /// A zero-length run.
    ZeroSize,
}

/// What [`state`] reports for a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameState {
    /// Managed and available.
    Free,
    /// Managed and handed out.
    Used,
    /// Holds the PMM's own bitmap.
    Reserved,
    /// Not managed (below 1 MiB, firmware, kernel image, holes, ...).
    Unmanaged,
}

/// Half-open range of frame numbers.
#[derive(Clone, Copy)]
struct Range {
    start: u64,
    end: u64,
}

impl Range {
    const EMPTY: Self = Self { start: 0, end: 0 };
}

struct Pmm {
    ready: bool,
    /// HHDM virtual address of the bitmap (kept as an integer so the static is `Send`).
    bitmap: usize,
    words: usize,
    /// The bitmap covers frames `0..frames`.
    frames: u64,
    meta_start: u64,
    meta_frames: u64,
    ranges: [Range; MAX_RANGES],
    range_count: usize,
    ranges_dropped: usize,
    /// Frames in managed ranges (including the bitmap's own).
    managed: u64,
    free: u64,
    /// Every frame below this is in use.
    hint: u64,
    min_free: u64,
    frames_allocated: u64,
    frames_freed: u64,
    alloc_failures: u64,
    free_rejected: u64,
}

static PMM: IrqLock<Pmm> = IrqLock::new(Pmm::new());

/// Bits `b .. b+take` of a word (`take` <= 64 - b).
fn mask(b: u32, take: u64) -> u64 {
    if take >= 64 { !0 } else { ((1u64 << take) - 1) << b }
}

fn align_up(v: u64, align: u64) -> u64 {
    (v + align - 1) & !(align - 1)
}

impl Pmm {
    const fn new() -> Self {
        Self {
            ready: false,
            bitmap: 0,
            words: 0,
            frames: 0,
            meta_start: 0,
            meta_frames: 0,
            ranges: [Range::EMPTY; MAX_RANGES],
            range_count: 0,
            ranges_dropped: 0,
            managed: 0,
            free: 0,
            hint: 0,
            min_free: 0,
            frames_allocated: 0,
            frames_freed: 0,
            alloc_failures: 0,
            free_rejected: 0,
        }
    }

    // ---- raw bitmap access ----------------------------------------------

    fn word(&self, i: usize) -> u64 {
        assert!(i < self.words, "pmm: bitmap index out of range");
        // SAFETY: `bitmap` points to `words` u64s reserved for us in HHDM memory.
        unsafe { (self.bitmap as *const u64).add(i).read() }
    }

    fn set_word(&mut self, i: usize, v: u64) {
        assert!(i < self.words, "pmm: bitmap index out of range");
        // SAFETY: as in `word`; we hold the lock, so no one else writes.
        unsafe { (self.bitmap as *mut u64).add(i).write(v) }
    }

    /// Marks `count` frames from `start` free or used.
    fn set_range(&mut self, start: u64, count: u64, free: bool) {
        let end = start + count;
        let mut f = start;
        while f < end {
            let w = (f >> 6) as usize;
            let b = (f & 63) as u32;
            let take = (64 - b as u64).min(end - f);
            let m = mask(b, take);
            let cur = self.word(w);
            self.set_word(w, if free { cur | m } else { cur & !m });
            f += take;
        }
    }

    /// First free frame in `from..to`.
    fn find_free(&self, from: u64, to: u64) -> Option<u64> {
        let to = to.min(self.frames);
        let mut f = from;
        while f < to {
            let w = (f >> 6) as usize;
            let b = (f & 63) as u32;
            let bits = self.word(w) & (!0u64 << b);
            if bits != 0 {
                let found = (w as u64) * 64 + bits.trailing_zeros() as u64;
                return if found < to { Some(found) } else { None };
            }
            f = (w as u64 + 1) * 64;
        }
        None
    }

    /// First frame in `start..start+count` that is NOT free.
    fn first_used_in(&self, start: u64, count: u64) -> Option<u64> {
        let end = start + count;
        let mut f = start;
        while f < end {
            let w = (f >> 6) as usize;
            let b = (f & 63) as u32;
            let take = (64 - b as u64).min(end - f);
            let used = !self.word(w) & mask(b, take);
            if used != 0 {
                return Some((w as u64) * 64 + used.trailing_zeros() as u64);
            }
            f += take;
        }
        None
    }

    /// Number of free frames in `start..start+count`.
    fn count_free(&self, start: u64, count: u64) -> u64 {
        let end = start + count;
        let mut f = start;
        let mut n = 0;
        while f < end {
            let w = (f >> 6) as usize;
            let b = (f & 63) as u32;
            let take = (64 - b as u64).min(end - f);
            n += (self.word(w) & mask(b, take)).count_ones() as u64;
            f += take;
        }
        n
    }

    // ---- allocation -------------------------------------------------------

    /// First-fit search for `count` free frames starting at a multiple of
    /// `align` frames, entirely below byte address `max_end`.
    fn alloc_run(&mut self, count: u64, align: u64, max_end: u64) -> Option<u64> {
        if !self.ready || count == 0 || align == 0 || !align.is_power_of_two() {
            return None;
        }
        if self.free < count {
            return None;
        }
        // Frame `f` lies below `max_end` iff `f < max_end >> 12`.
        let hi = self.frames.min(max_end >> FRAME_SHIFT);
        let mut start = self.hint;
        let mut first = true;
        loop {
            let f = self.find_free(start, hi)?;
            let s = align_up(f, align);
            if s.checked_add(count)? > hi {
                return None;
            }
            match self.first_used_in(s, count) {
                None => {
                    self.set_range(s, count, false);
                    self.free -= count;
                    self.frames_allocated += count;
                    self.min_free = self.min_free.min(self.free);
                    // `f` was the first free frame at or above the hint, so
                    // everything below `s + count` is used now.
                    if first && s == f {
                        self.hint = s + count;
                    }
                    return Some(s);
                }
                Some(used) => {
                    start = used + 1;
                    first = false;
                }
            }
        }
    }

    // ---- freeing ------------------------------------------------------------

    fn validate_free(&self, addr: u64, count: u64) -> Result<(u64, u64), PmmError> {
        if !self.ready {
            return Err(PmmError::NotInitialised);
        }
        if count == 0 {
            return Err(PmmError::ZeroSize);
        }
        if addr & (FRAME_SIZE - 1) != 0 {
            return Err(PmmError::Misaligned);
        }
        let start = addr >> FRAME_SHIFT;
        let end = start.checked_add(count).ok_or(PmmError::OutOfRange)?;
        let range = self.ranges[..self.range_count]
            .iter()
            .find(|r| start >= r.start && start < r.end)
            .ok_or(PmmError::OutOfRange)?;
        if end > range.end {
            return Err(PmmError::OutOfRange);
        }
        if start < self.meta_start + self.meta_frames && end > self.meta_start {
            return Err(PmmError::Reserved);
        }
        if self.find_free(start, end).is_some() {
            return Err(PmmError::DoubleFree);
        }
        Ok((start, end))
    }

    fn free_run(&mut self, addr: u64, count: u64) -> Result<(), PmmError> {
        match self.validate_free(addr, count) {
            Ok((start, _)) => {
                self.set_range(start, count, true);
                self.free += count;
                self.frames_freed += count;
                self.hint = self.hint.min(start);
                Ok(())
            }
            Err(e) => {
                if e != PmmError::NotInitialised {
                    self.free_rejected += 1;
                }
                Err(e)
            }
        }
    }

    fn state_of(&self, frame: u64) -> FrameState {
        if !self.ready || frame >= self.frames {
            return FrameState::Unmanaged;
        }
        if !self.ranges[..self.range_count].iter().any(|r| frame >= r.start && frame < r.end) {
            return FrameState::Unmanaged;
        }
        if frame >= self.meta_start && frame < self.meta_start + self.meta_frames {
            return FrameState::Reserved;
        }
        if self.word((frame >> 6) as usize) >> (frame & 63) & 1 == 1 {
            FrameState::Free
        } else {
            FrameState::Used
        }
    }
}

// ----------------------------------------------------------------------
// Initialisation
// ----------------------------------------------------------------------

/// What [`init`] set up, for the boot log.
#[derive(Clone, Copy)]
pub struct InitInfo {
    /// Managed ranges after merging adjacent usable regions.
    pub ranges: usize,
    /// Ranges that did not fit the table ([`MAX_RANGES`]) and stay unmanaged.
    pub ranges_dropped: usize,
    /// Lowest managed byte address.
    pub lowest: u64,
    /// One past the highest managed byte address.
    pub highest: u64,
    /// Frames in managed ranges (including the bitmap).
    pub managed_frames: u64,
    pub free_frames: u64,
    /// Physical address and size of the bitmap.
    pub meta_phys: u64,
    pub meta_frames: u64,
    pub meta_bytes: u64,
    /// The bitmap covers frames `0..covered_frames`.
    pub covered_frames: u64,
}

/// Builds the bitmap from the bootloader memory map. Boot CPU, once, after
/// `hal::memory::init`.
pub fn init() -> Result<InitInfo, PmmError> {
    let mut p = PMM.lock();
    if p.ready {
        return Err(PmmError::AlreadyInitialised);
    }

    // 1. Usable regions, clipped to whole frames above FLOOR, adjacent ones merged.
    //    (the bootloader reports regions sorted and non-overlapping; overlap is trimmed anyway.)
    let mut ranges = [Range::EMPTY; MAX_RANGES];
    let mut n = 0usize;
    let mut dropped = 0usize;
    let mut last_end = 0u64;
    memory::for_each_region(|r| {
        if r.kind != RegionKind::Usable {
            return;
        }
        let start_addr = align_up(r.base.max(FLOOR), FRAME_SIZE);
        let end_addr = r.end() & !(FRAME_SIZE - 1);
        let s = (start_addr >> FRAME_SHIFT).max(last_end);
        let e = end_addr >> FRAME_SHIFT;
        if e <= s {
            return;
        }
        last_end = e;
        if n > 0 && ranges[n - 1].end == s {
            ranges[n - 1].end = e;
        } else if n < MAX_RANGES {
            ranges[n] = Range { start: s, end: e };
            n += 1;
        } else {
            dropped += 1;
        }
    });
    if n == 0 {
        return Err(PmmError::NoUsableMemory);
    }

    // 2. Size and place the bitmap: end of the highest range that can hold it.
    let frames = ranges[..n].iter().map(|r| r.end).max().unwrap_or(0);
    let words = frames.div_ceil(64) as usize;
    let meta_bytes = words as u64 * 8;
    let meta_frames = meta_bytes.div_ceil(FRAME_SIZE);
    let host = ranges[..n]
        .iter()
        .rev()
        .find(|r| r.end - r.start >= meta_frames)
        .ok_or(PmmError::NoSpaceForMetadata)?;
    let meta_start = host.end - meta_frames;

    // 3. Everything starts "used"; usable ranges are then marked free.
    p.bitmap = memory::phys_to_virt(meta_start << FRAME_SHIFT) as usize;
    p.words = words;
    p.frames = frames;
    // SAFETY: the bitmap frames are usable RAM reachable through the HHDM and
    // owned by nobody else yet.
    unsafe { ptr::write_bytes(p.bitmap as *mut u8, 0, words * 8) };

    let mut managed = 0;
    for r in &ranges[..n] {
        p.set_range(r.start, r.end - r.start, true);
        managed += r.end - r.start;
    }
    p.set_range(meta_start, meta_frames, false);

    p.ranges = ranges;
    p.range_count = n;
    p.ranges_dropped = dropped;
    p.meta_start = meta_start;
    p.meta_frames = meta_frames;
    p.managed = managed;
    p.free = managed - meta_frames;
    p.min_free = p.free;
    p.hint = ranges[0].start;
    p.ready = true;

    Ok(InitInfo {
        ranges: n,
        ranges_dropped: dropped,
        lowest: ranges[0].start << FRAME_SHIFT,
        highest: frames << FRAME_SHIFT,
        managed_frames: managed,
        free_frames: p.free,
        meta_phys: meta_start << FRAME_SHIFT,
        meta_frames,
        meta_bytes,
        covered_frames: frames,
    })
}

pub fn is_ready() -> bool {
    PMM.lock().ready
}

// ----------------------------------------------------------------------
// Allocation API
// ----------------------------------------------------------------------

/// Allocates one frame; contents are **unspecified**. Returns its physical address.
pub fn alloc_frame() -> Option<u64> {
    alloc_contiguous(1, 1, NO_LIMIT)
}

/// Allocates one frame filled with zeroes (use this for page tables).
pub fn alloc_zeroed() -> Option<u64> {
    alloc_contiguous_zeroed(1, 1, NO_LIMIT)
}

/// Allocates `count` physically contiguous frames whose start address is a
/// multiple of `align_frames` frames (a power of two) and whose last byte is
/// below `max_end` ([`NO_LIMIT`], [`LIMIT_4G`], [`LIMIT_16M`], ...). Contents
/// are unspecified. Returns `None` if no such run exists or the arguments are
/// invalid (zero count, alignment not a power of two).
///
/// Example: a 2 MiB-aligned 2 MiB block is `alloc_contiguous(512, 512, NO_LIMIT)`.
pub fn alloc_contiguous(count: u64, align_frames: u64, max_end: u64) -> Option<u64> {
    let mut p = PMM.lock();
    let r = p.alloc_run(count, align_frames, max_end);
    if r.is_none() && p.ready {
        p.alloc_failures += 1;
    }
    r.map(|f| f << FRAME_SHIFT)
}

/// Like [`alloc_contiguous`], but every byte of the run is zeroed.
pub fn alloc_contiguous_zeroed(count: u64, align_frames: u64, max_end: u64) -> Option<u64> {
    let addr = alloc_contiguous(count, align_frames, max_end)?;
    // SAFETY: the run was just allocated to us and lies in HHDM-mapped usable RAM.
    // Zeroing happens outside the lock: no one else can own these frames.
    unsafe { ptr::write_bytes(memory::phys_to_virt(addr), 0, (count * FRAME_SIZE) as usize) };
    Some(addr)
}

/// Returns one frame to the PMM.
pub fn free_frame(addr: u64) -> Result<(), PmmError> {
    free_contiguous(addr, 1)
}

/// Returns `count` contiguous frames. All-or-nothing: if any frame is
/// invalid or already free, nothing is changed.
pub fn free_contiguous(addr: u64, count: u64) -> Result<(), PmmError> {
    PMM.lock().free_run(addr, count)
}

/// Ownership state of the frame containing `addr`.
pub fn state(addr: u64) -> FrameState {
    PMM.lock().state_of(addr >> FRAME_SHIFT)
}

// ----------------------------------------------------------------------
// Statistics and verification
// ----------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Stats {
    /// Frames in managed ranges, including the bitmap itself.
    pub managed_frames: u64,
    pub free_frames: u64,
    /// Frames holding the PMM's own bitmap.
    pub metadata_frames: u64,
    /// Frames currently handed out (excludes the bitmap).
    pub in_use_frames: u64,
    /// Highest `in_use_frames` ever seen.
    pub peak_in_use_frames: u64,
    /// Cumulative frames allocated / freed.
    pub frames_allocated_total: u64,
    pub frames_freed_total: u64,
    /// Allocation calls that returned `None` (excluding "not initialised").
    pub alloc_failures: u64,
    /// Free calls rejected as invalid or double frees.
    pub free_rejected: u64,
    pub ranges: usize,
    pub ranges_dropped: usize,
}

impl Stats {
    pub fn free_bytes(&self) -> u64 {
        self.free_frames * FRAME_SIZE
    }
}

pub fn stats() -> Stats {
    let p = PMM.lock();
    Stats {
        managed_frames: p.managed,
        free_frames: p.free,
        metadata_frames: p.meta_frames,
        in_use_frames: p.managed - p.free - p.meta_frames,
        peak_in_use_frames: p.managed - p.min_free - p.meta_frames,
        frames_allocated_total: p.frames_allocated,
        frames_freed_total: p.frames_freed,
        alloc_failures: p.alloc_failures,
        free_rejected: p.free_rejected,
        ranges: p.range_count,
        ranges_dropped: p.ranges_dropped,
    }
}

/// Walks the whole bitmap and checks every internal invariant:
///
/// 1. the number of set bits equals the free counter;
/// 2. no frame outside the managed ranges is marked free;
/// 3. the bitmap's own frames are marked used;
/// 4. no free frame exists below the allocation hint;
/// 5. cumulative allocated - freed equals the frames in use.
pub fn verify() -> Result<(), &'static str> {
    let p = PMM.lock();
    if !p.ready {
        return Err("PMM not initialised");
    }
    let mut total = 0u64;
    for i in 0..p.words {
        total += p.word(i).count_ones() as u64;
    }
    if total != p.free {
        return Err("free counter does not match the bitmap");
    }
    let inside: u64 = p.ranges[..p.range_count].iter().map(|r| p.count_free(r.start, r.end - r.start)).sum();
    if inside != total {
        return Err("a frame outside the managed ranges is marked free");
    }
    if p.count_free(p.meta_start, p.meta_frames) != 0 {
        return Err("the bitmap's own frames are marked free");
    }
    if p.find_free(0, p.hint).is_some() {
        return Err("a free frame exists below the allocation hint");
    }
    if p.frames_allocated - p.frames_freed != p.managed - p.free - p.meta_frames {
        return Err("allocation counters do not match the bitmap");
    }
    Ok(())
}

// ----------------------------------------------------------------------
// Self-test
// ----------------------------------------------------------------------

/// What [`self_test`] exercised, for the boot log.
#[derive(Clone, Copy)]
pub struct SelfTest {
    /// Frames allocated (and all returned) during the test.
    pub frames_cycled: u64,
    /// Zero-fill allocations that reused frames previously filled with 0xA5.
    pub dirty_reused: usize,
    /// Invalid / double frees that were correctly rejected.
    pub rejected: u64,
    /// Base of the 64 KiB-aligned contiguous run that was tested.
    pub contiguous_base: u64,
    /// Address of a frame run allocated below 16 MiB, if memory exists there.
    pub below_16m: Option<u64>,
}

fn is_zero_frame(addr: u64) -> bool {
    let p = memory::phys_to_virt(addr) as *const u64;
    for i in 0..(FRAME_SIZE as usize / 8) {
        // SAFETY: `addr` is an allocated frame in the HHDM.
        if unsafe { p.add(i).read() } != 0 {
            return false;
        }
    }
    true
}

/// Allocation, zero-fill, reuse, double-free and invalid-free detection,
/// aligned and limited contiguous runs, failure paths, fragmentation, and
/// the full-bitmap invariant check. Leaves the PMM exactly as it found it.
pub fn self_test() -> Result<SelfTest, &'static str> {
    if !is_ready() {
        return Err("PMM not initialised");
    }
    verify()?;
    let base = stats();
    let meta_phys = {
        let p = PMM.lock();
        p.meta_start << FRAME_SHIFT
    };
    let covered_end = {
        let p = PMM.lock();
        p.frames << FRAME_SHIFT
    };

    // 1. Single frames: aligned, above the floor, distinct, marked used.
    const N: usize = 64;
    let mut held = [0u64; N];
    for i in 0..N {
        let f = alloc_frame().ok_or("alloc_frame returned None")?;
        if f & (FRAME_SIZE - 1) != 0 {
            return Err("allocated frame is not 4 KiB aligned");
        }
        if f < FLOOR {
            return Err("allocated frame is below 1 MiB");
        }
        if state(f) != FrameState::Used {
            return Err("allocated frame is not marked used");
        }
        if held[..i].contains(&f) {
            return Err("the same frame was handed out twice");
        }
        held[i] = f;
    }
    if base.free_frames - stats().free_frames != N as u64 {
        return Err("free counter wrong after allocations");
    }

    // 2. Zero-fill really clears reused memory: dirty everything, free half, re-allocate zeroed.
    for &f in &held {
        // SAFETY: allocated frames in the HHDM.
        unsafe { ptr::write_bytes(memory::phys_to_virt(f), 0xA5, FRAME_SIZE as usize) };
    }
    let mut dirty = [0u64; N / 2];
    for (k, i) in (1..N).step_by(2).enumerate() {
        dirty[k] = held[i];
        free_frame(held[i]).map_err(|_| "free_frame failed")?;
        if state(held[i]) != FrameState::Free {
            return Err("freed frame is not marked free");
        }
    }
    let mut zeroed = [0u64; N / 2];
    let mut dirty_reused = 0usize;
    for z in zeroed.iter_mut() {
        *z = alloc_zeroed().ok_or("alloc_zeroed returned None")?;
        if !is_zero_frame(*z) {
            return Err("alloc_zeroed returned a non-zero frame");
        }
        if dirty.contains(z) {
            dirty_reused += 1;
        }
    }
    if dirty_reused == 0 {
        return Err("freed frames were not reused (zero-fill not exercised)");
    }
    for &z in &zeroed {
        free_frame(z).map_err(|_| "free_frame failed")?;
    }
    for i in (0..N).step_by(2) {
        free_frame(held[i]).map_err(|_| "free_frame failed")?;
    }
    let after = stats();
    if after.free_frames != base.free_frames || after.in_use_frames != base.in_use_frames {
        return Err("frames leaked or double counted by the single-frame test");
    }

    // 3. Invalid and double frees are rejected and change nothing.
    let rejected_before = stats().free_rejected;
    let f = alloc_frame().ok_or("alloc_frame returned None")?;
    free_frame(f).map_err(|_| "free_frame failed")?;
    let free_now = stats().free_frames;
    let checks: [(Result<(), PmmError>, PmmError); 8] = [
        (free_frame(f), PmmError::DoubleFree),
        (free_frame(f + 1), PmmError::Misaligned),
        (free_frame(0), PmmError::OutOfRange),
        (free_frame(u64::MAX & !(FRAME_SIZE - 1)), PmmError::OutOfRange),
        (free_frame(covered_end), PmmError::OutOfRange),
        (free_frame(meta_phys), PmmError::Reserved),
        (free_contiguous(f, 0), PmmError::ZeroSize),
        (free_contiguous(f, u64::MAX), PmmError::OutOfRange),
    ];
    for (got, want) in checks {
        if got != Err(want) {
            return Err("an invalid free was not rejected with the right error");
        }
    }
    let s = stats();
    if s.free_frames != free_now || s.free_rejected != rejected_before + 8 {
        return Err("rejected frees changed the free count or were not counted");
    }

    // 4. Aligned contiguous runs.
    let c1 = alloc_contiguous(16, 16, NO_LIMIT).ok_or("alloc_contiguous(16, 16) failed")?;
    let c2 = alloc_contiguous(16, 16, NO_LIMIT).ok_or("second alloc_contiguous(16, 16) failed")?;
    if c1 % (16 * FRAME_SIZE) != 0 || c2 % (16 * FRAME_SIZE) != 0 {
        return Err("contiguous run is not aligned as requested");
    }
    if c1.abs_diff(c2) < 16 * FRAME_SIZE {
        return Err("two contiguous runs overlap");
    }
    for i in 0..16 {
        if state(c1 + i * FRAME_SIZE) != FrameState::Used || state(c2 + i * FRAME_SIZE) != FrameState::Used {
            return Err("frame inside a contiguous run is not marked used");
        }
    }
    free_contiguous(c1, 16).map_err(|_| "free_contiguous failed")?;
    free_contiguous(c2, 16).map_err(|_| "free_contiguous failed")?;

    // 5. All-or-nothing free: a run with one already-free frame is refused entirely.
    let c = alloc_contiguous(4, 1, NO_LIMIT).ok_or("alloc_contiguous(4) failed")?;
    free_frame(c + FRAME_SIZE).map_err(|_| "free_frame failed")?;
    let before = stats().free_frames;
    if free_contiguous(c, 4) != Err(PmmError::DoubleFree) {
        return Err("partially free run was not rejected");
    }
    if stats().free_frames != before || state(c) != FrameState::Used || state(c + 2 * FRAME_SIZE) != FrameState::Used {
        return Err("rejected run free was not atomic");
    }
    free_frame(c).map_err(|_| "free_frame failed")?;
    free_frame(c + 2 * FRAME_SIZE).map_err(|_| "free_frame failed")?;
    free_frame(c + 3 * FRAME_SIZE).map_err(|_| "free_frame failed")?;

    // 6. Fragmentation: single-frame holes must not satisfy a 2-frame request.
    let mut h = [0u64; 8];
    for x in h.iter_mut() {
        *x = alloc_frame().ok_or("alloc_frame returned None")?;
    }
    for i in (0..8).step_by(2) {
        free_frame(h[i]).map_err(|_| "free_frame failed")?;
    }
    let pair = alloc_contiguous(2, 1, NO_LIMIT).ok_or("alloc_contiguous(2) failed")?;
    for &held_frame in h.iter().skip(1).step_by(2) {
        if (pair..pair + 2 * FRAME_SIZE).contains(&held_frame) {
            return Err("a contiguous run overlaps a frame that is still in use");
        }
    }
    free_contiguous(pair, 2).map_err(|_| "free_contiguous failed")?;
    for i in (1..8).step_by(2) {
        free_frame(h[i]).map_err(|_| "free_frame failed")?;
    }

    // 7. Address limit: honoured, and impossible limits fail cleanly.
    let below_16m = match alloc_contiguous(4, 1, LIMIT_16M) {
        Some(a) => {
            if a + 4 * FRAME_SIZE > LIMIT_16M {
                return Err("allocation exceeds the requested address limit");
            }
            free_contiguous(a, 4).map_err(|_| "free_contiguous failed")?;
            Some(a)
        }
        None => None,
    };

    // 8. Failure paths return None, are counted, and change nothing.
    let s = stats();
    let fails = [
        alloc_contiguous(0, 1, NO_LIMIT),
        alloc_contiguous(1, 3, NO_LIMIT),
        alloc_contiguous(u64::MAX, 1, NO_LIMIT),
        alloc_contiguous(1, 1, FLOOR), // nothing is managed below the floor
    ];
    if fails.iter().any(|r| r.is_some()) {
        return Err("an impossible allocation succeeded");
    }
    let t = stats();
    if t.free_frames != s.free_frames || t.alloc_failures != s.alloc_failures + 4 {
        return Err("failed allocations changed the free count or were not counted");
    }

    // 9. Back to the starting point, and the whole bitmap is consistent.
    let end = stats();
    if end.free_frames != base.free_frames || end.in_use_frames != base.in_use_frames {
        return Err("the test did not restore the free count");
    }
    verify()?;

    Ok(SelfTest {
        frames_cycled: end.frames_allocated_total - base.frames_allocated_total,
        dirty_reused,
        rejected: end.free_rejected - base.free_rejected,
        contiguous_base: c1,
        below_16m,
    })
}
