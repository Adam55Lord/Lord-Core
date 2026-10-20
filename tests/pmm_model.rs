// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Host-side randomized model test for the physical memory manager.
//!
//! Compiles the kernel's real `src/mm/pmm.rs` *unchanged* against a mock
//! memory map and checks every operation against a deliberately naive
//! reference model (one byte per frame). Run it with `tests/run-host-tests.sh`.
//!
//! Each process runs one seed: a random memory layout (holes, adjacent
//! regions, a region straddling 1 MiB, regions above 4 GiB, more regions than
//! the table holds) followed by thousands of random allocations, frees (whole,
//! partial, invalid, double) and queries. The PMM must agree with the model on
//! every result, including *which* address a first-fit allocation returns.

#![allow(dead_code)]

use std::sync::Mutex;

#[path = "../src/hal/sync.rs"]
mod sync_real;

mod hal {
    pub mod sync {
        pub use crate::sync_real::*;
    }
    pub mod memory {
        use std::sync::Mutex;

        #[derive(Clone, Copy, PartialEq, Eq)]
        pub enum RegionKind {
            Usable,
            Reserved,
        }

        #[derive(Clone, Copy)]
        pub struct Region {
            pub base: u64,
            pub length: u64,
            pub kind: RegionKind,
        }

        impl Region {
            pub fn end(&self) -> u64 {
                self.base.saturating_add(self.length)
            }
        }

        /// Regions plus, for each usable one, its offset into the host arena.
        pub static LAYOUT: Mutex<(Vec<Region>, Vec<(u64, u64, usize)>, usize)> =
            Mutex::new((Vec::new(), Vec::new(), 0));

        pub fn for_each_region(mut f: impl FnMut(Region)) {
            let regions = LAYOUT.lock().unwrap().0.clone();
            for r in regions {
                f(r);
            }
        }

        /// Sparse "HHDM": every usable region gets its own slice of one host
        /// allocation, so physical addresses far above the arena size work.
        pub fn phys_to_virt(p: u64) -> *mut u8 {
            let l = LAYOUT.lock().unwrap();
            for &(base, len, off) in &l.1 {
                if p >= base && p < base + len {
                    return (l.2 + off + (p - base) as usize) as *mut u8;
                }
            }
            panic!("mock HHDM: {p:#x} is not in a usable region");
        }
    }
}

#[path = "../src/mm/pmm.rs"]
pub mod pmm_real;

mod mm {
    pub use crate::pmm_real as pmm;
}

use mm::pmm::{self, FrameState, PmmError, FRAME_SIZE};

// ---------------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
}

const FLOOR_FRAME: u64 = 0x10_0000 >> 12;
const MAX_RANGES: usize = 128;

/// Random memory layout: (base_frame, frames, usable) in ascending order.
fn make_layout(rng: &mut Rng, seed: u64) -> Vec<(u64, u64, bool)> {
    let mut out = Vec::new();
    let mut cursor = 0u64;
    let many = seed % 16 == 15; // more usable ranges than the table can hold
    let count = if many { 400 } else { 3 + rng.below(9) };
    for i in 0..count {
        let gap = if many {
            1 + rng.below(3)
        } else if rng.chance(30) {
            0
        } else if rng.chance(15) {
            rng.below(2_000_000) // a huge hole, possibly crossing 4 GiB
        } else {
            rng.below(300)
        };
        cursor += gap;
        let len = if many { 2 + rng.below(6) } else { 1 + rng.below(3000) };
        let usable = many || i == 0 || rng.chance(75);
        out.push((cursor, len, usable));
        cursor += len;
    }
    if seed % 16 == 0 {
        // the classic layout: legacy low memory, a region straddling 1 MiB
        out.clear();
        out.push((0, 0x9F, true));
        out.push((0x80, 0x200, true));
        out.push((0x300, 4000, true));
    }
    if seed % 16 == 14 {
        // only legacy low memory: nothing for the PMM to manage
        out.clear();
        out.push((0, 0x80, true));
    }
    // overlapping inputs are never produced: regions are strictly ascending
    out
}

/// Reference model: one byte per frame (0 unmanaged, 1 free, 2 used, 3 PMM metadata).
struct Model {
    state: Vec<u8>,
    ranges: Vec<(u64, u64)>,
    meta: (u64, u64),
    peak_in_use: u64,
    allocated_total: u64,
    freed_total: u64,
    failures: u64,
    rejected: u64,
}

impl Model {
    fn top(&self) -> u64 {
        self.state.len() as u64
    }
    fn count(&self, s: u8) -> u64 {
        self.state.iter().filter(|&&x| x == s).count() as u64
    }
    /// Lowest valid start of a first-fit run, without allocating it.
    fn find(&self, count: u64, align: u64, max_end: u64) -> Option<u64> {
        if count == 0 || align == 0 || !align.is_power_of_two() {
            return None;
        }
        let hi = self.top().min(max_end >> 12);
        let mut s = 0u64;
        while s.checked_add(count)? <= hi {
            if self.state[s as usize] != 1 {
                s += 1;
                s = (s + align - 1) & !(align - 1);
                continue;
            }
            match (s..s + count).find(|&f| self.state[f as usize] != 1) {
                None => return Some(s),
                Some(u) => s = (u + 1 + align - 1) & !(align - 1),
            }
        }
        None
    }
    fn alloc(&mut self, count: u64, align: u64, max_end: u64) -> Option<u64> {
        let s = self.find(count, align, max_end)?;
        for f in s..s + count {
            self.state[f as usize] = 2;
        }
        self.allocated_total += count;
        let in_use = self.count(2);
        self.peak_in_use = self.peak_in_use.max(in_use);
        Some(s)
    }
    fn expect_free(&self, addr: u64, count: u64) -> Result<(), PmmError> {
        if count == 0 {
            return Err(PmmError::ZeroSize);
        }
        if addr & (FRAME_SIZE - 1) != 0 {
            return Err(PmmError::Misaligned);
        }
        let start = addr >> 12;
        let end = start.checked_add(count).ok_or(PmmError::OutOfRange)?;
        let r = self.ranges.iter().find(|r| start >= r.0 && start < r.1).ok_or(PmmError::OutOfRange)?;
        if end > r.1 {
            return Err(PmmError::OutOfRange);
        }
        if start < self.meta.0 + self.meta.1 && end > self.meta.0 {
            return Err(PmmError::Reserved);
        }
        if (start..end).any(|f| self.state[f as usize] == 1) {
            return Err(PmmError::DoubleFree);
        }
        Ok(())
    }
}

fn fail(seed: u64, step: u64, msg: String) -> ! {
    eprintln!("FAIL seed {seed} step {step}: {msg}");
    std::process::exit(1);
}

fn main() {
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let ops: u64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3000);
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    for _ in 0..8 {
        rng.next();
    }

    // ---- mock machine -----------------------------------------------------
    let layout = make_layout(&mut rng, seed);
    let mut arena_bytes = 0usize;
    let mut slices = Vec::new();
    let mut regions = Vec::new();
    for &(base, len, usable) in &layout {
        regions.push(hal::memory::Region {
            base: base * FRAME_SIZE,
            length: len * FRAME_SIZE,
            kind: if usable { hal::memory::RegionKind::Usable } else { hal::memory::RegionKind::Reserved },
        });
        if usable {
            slices.push((base * FRAME_SIZE, len * FRAME_SIZE, arena_bytes));
            arena_bytes += (len * FRAME_SIZE) as usize;
        }
    }
    let arena: &'static mut [u8] = Box::leak(vec![0u8; arena_bytes.max(1)].into_boxed_slice());
    arena.fill(0xCC); // make "uninitialised" memory recognisable
    {
        let mut l = hal::memory::LAYOUT.lock().unwrap();
        *l = (regions, slices, arena.as_ptr() as usize);
    }

    // ---- expected managed ranges (clip to >= 1 MiB, merge adjacent, 128 max) --
    let mut ranges: Vec<(u64, u64)> = Vec::new();
    for &(base, len, usable) in &layout {
        if !usable {
            continue;
        }
        let s = base.max(FLOOR_FRAME);
        let e = base + len;
        if e <= s {
            continue;
        }
        if matches!(ranges.last(), Some(last) if last.1 == s) {
            ranges.last_mut().unwrap().1 = e;
        } else if ranges.len() < MAX_RANGES {
            ranges.push((s, e));
        }
    }

    let info = match pmm::init() {
        Ok(i) => i,
        Err(e) => {
            if ranges.is_empty() && e == PmmError::NoUsableMemory {
                println!("seed {seed}: no usable memory above 1 MiB, rejected correctly");
                return;
            }
            fail(seed, 0, format!("init failed: {e:?}"));
        }
    };
    if pmm::init().err() != Some(PmmError::AlreadyInitialised) {
        fail(seed, 0, "second init was not rejected".into());
    }

    // ---- build the model from the layout, check what init reported ---------
    let top = ranges.iter().map(|r| r.1).max().unwrap();
    let mut model = Model {
        state: vec![0u8; top as usize],
        ranges: ranges.clone(),
        meta: (info.meta_phys >> 12, info.meta_frames),
        peak_in_use: 0,
        allocated_total: 0,
        freed_total: 0,
        failures: 0,
        rejected: 0,
    };
    for &(s, e) in &ranges {
        for f in s..e {
            model.state[f as usize] = 1;
        }
    }
    let managed: u64 = ranges.iter().map(|r| r.1 - r.0).sum();
    if info.ranges != ranges.len() || info.managed_frames != managed || info.covered_frames != top {
        fail(seed, 0, format!("init info wrong: {} ranges / {} frames / top {} (want {} / {} / {})",
            info.ranges, info.managed_frames, info.covered_frames, ranges.len(), managed, top));
    }
    let need = (top.div_ceil(64) * 8).div_ceil(FRAME_SIZE);
    if info.meta_frames != need {
        fail(seed, 0, format!("bitmap is {} frames, expected {}", info.meta_frames, need));
    }
    // placement: at the end of the highest range that can hold it
    let host = ranges.iter().rev().find(|r| r.1 - r.0 >= need).unwrap();
    if model.meta.0 != host.1 - need {
        fail(seed, 0, format!("bitmap at frame {:#x}, expected {:#x}", model.meta.0, host.1 - need));
    }
    for f in model.meta.0..model.meta.0 + model.meta.1 {
        model.state[f as usize] = 3;
    }
    let free0 = model.count(1);
    if info.free_frames != free0 {
        fail(seed, 0, format!("init free {} != model {}", info.free_frames, free0));
    }
    if let Err(e) = pmm::verify() {
        fail(seed, 0, format!("verify after init: {e}"));
    }

    // ---- random operations ----------------------------------------------------
    let mut live: Vec<(u64, u64)> = Vec::new(); // (addr, frames) currently allocated
    let mut recently_freed: Vec<u64> = Vec::new();
    let span = (top << 12).max(0x20_0000);

    for step in 1..=ops {
        match rng.below(100) {
            // ---- allocate (50 %) ----
            0..=49 => {
                let count = match rng.below(100) {
                    0..=59 => 1,
                    60..=84 => 1 + rng.below(8),
                    85..=94 => 16 << rng.below(5),
                    95..=98 => 512,
                    _ => rng.below(6000),
                };
                let align = if rng.chance(55) { 1 } else if rng.chance(5) { rng.below(10) + 1 } else { 1u64 << rng.below(10) };
                let mut max_end = match rng.below(10) {
                    0..=6 => pmm::NO_LIMIT,
                    7 => pmm::LIMIT_16M,
                    8 => pmm::LIMIT_4G,
                    _ => rng.below(span),
                };
                if rng.chance(25) {
                    // Tight limit: right at, just below, or just above the end of the
                    // lowest run that would fit. Exercises the exact boundary.
                    if let Some(s0) = model.find(count, align, pmm::NO_LIMIT) {
                        let end = (s0 + count) << 12;
                        max_end = [end, end - 1, end - 4095, end - 4096, end + 1, end + 4095][rng.below(6) as usize];
                    }
                }
                let zeroed = count <= 4 && rng.chance(30);
                let want = model.alloc(count, align, max_end);
                if want.is_none() {
                    model.failures += 1;
                }
                let got = if zeroed {
                    pmm::alloc_contiguous_zeroed(count, align, max_end)
                } else {
                    pmm::alloc_contiguous(count, align, max_end)
                };
                if got.map(|a| a >> 12) != want {
                    fail(seed, step, format!(
                        "alloc(count {count}, align {align}, max_end {max_end:#x}): got {:?}, model says {:?}",
                        got.map(|a| a >> 12), want));
                }
                if let Some(a) = got {
                    if zeroed {
                        let p = hal::memory::phys_to_virt(a);
                        for i in 0..(count * FRAME_SIZE) as usize {
                            if unsafe { p.add(i).read() } != 0 {
                                fail(seed, step, format!("zeroed allocation at {a:#x} has a non-zero byte at +{i}"));
                            }
                        }
                    }
                    live.push((a, count));
                }
            }
            // ---- free (30 %): whole run, or a random sub-run of one ----
            50..=79 => {
                if live.is_empty() {
                    continue;
                }
                let k = rng.below(live.len() as u64) as usize;
                let (addr, count) = live[k];
                let (off, len) = if count > 1 && rng.chance(50) {
                    let off = rng.below(count);
                    (off, 1 + rng.below(count - off))
                } else {
                    (0, count)
                };
                let a = addr + off * FRAME_SIZE;
                if rng.chance(30) {
                    // dirty the memory so zero-fill on reuse is meaningful
                    unsafe { std::ptr::write_bytes(hal::memory::phys_to_virt(a), 0xA5, (len * FRAME_SIZE) as usize) };
                }
                if model.expect_free(a, len).is_err() {
                    fail(seed, step, "model rejects freeing a live allocation".into());
                }
                for f in (a >> 12)..(a >> 12) + len {
                    model.state[f as usize] = 1;
                }
                model.freed_total += len;
                if let Err(e) = pmm::free_contiguous(a, len) {
                    fail(seed, step, format!("free_contiguous({a:#x}, {len}) failed: {e:?}"));
                }
                recently_freed.push(a);
                if recently_freed.len() > 32 {
                    recently_freed.remove(0);
                }
                live.swap_remove(k);
                if len < count {
                    if off > 0 {
                        live.push((addr, off));
                    }
                    if off + len < count {
                        live.push((addr + (off + len) * FRAME_SIZE, count - off - len));
                    }
                }
            }
            // ---- invalid / double free (10 %) ----
            80..=89 => {
                let (addr, count) = match rng.below(7) {
                    0 if !recently_freed.is_empty() => (recently_freed[rng.below(recently_freed.len() as u64) as usize], 1 + rng.below(3)),
                    1 => (rng.below(span) | 1, 1),                                   // misaligned
                    2 => (rng.below(span) & !0xFFF, 1 + rng.below(4)),               // random frame
                    3 => (rng.below(0x10_0000) & !0xFFF, 1),                         // below the floor
                    4 => ((top << 12) + (rng.below(8) << 12), 1),                    // at / past the top
                    5 => (model.meta.0 * FRAME_SIZE + rng.below(model.meta.1) * FRAME_SIZE, 1 + rng.below(2)), // bitmap
                    _ => (rng.below(span) & !0xFFF, if rng.chance(50) { 0 } else { u64::MAX - rng.below(4) }),
                };
                let want = model.expect_free(addr, count);
                let before_free = model.count(1);
                match want {
                    Ok(()) => {
                        // it happens to be a valid free (e.g. a live frame): apply it to both
                        for f in (addr >> 12)..(addr >> 12) + count {
                            model.state[f as usize] = 1;
                        }
                        model.freed_total += count;
                        // keep only the parts of live allocations it did not free
                        let (lo, hi) = (addr, addr + count * FRAME_SIZE);
                        let mut kept = Vec::new();
                        for &(a, c) in &live {
                            let end = a + c * FRAME_SIZE;
                            if end <= lo || a >= hi {
                                kept.push((a, c));
                                continue;
                            }
                            if a < lo {
                                kept.push((a, (lo - a) / FRAME_SIZE));
                            }
                            if end > hi {
                                kept.push((hi, (end - hi) / FRAME_SIZE));
                            }
                        }
                        live = kept;
                    }
                    Err(_) => model.rejected += 1,
                }
                let got = pmm::free_contiguous(addr, count);
                if got != want {
                    fail(seed, step, format!("free({addr:#x}, {count}): got {got:?}, model says {want:?}"));
                }
                if want.is_err() && model.count(1) != before_free {
                    fail(seed, step, "model changed on a rejected free".into());
                }
            }
            // ---- state query (10 %) ----
            _ => {
                let addr = if rng.chance(50) { rng.below(span) } else { rng.below(top << 12 | 1) };
                let f = (addr >> 12) as usize;
                let want = match model.state.get(f).copied().unwrap_or(0) {
                    1 => FrameState::Free,
                    2 => FrameState::Used,
                    3 => FrameState::Reserved,
                    _ => FrameState::Unmanaged,
                };
                if pmm::state(addr) != want {
                    fail(seed, step, format!("state({addr:#x}): got {:?}, model says {want:?}", pmm::state(addr)));
                }
            }
        }

        if step % 97 == 0 || step == ops {
            if let Err(e) = pmm::verify() {
                fail(seed, step, format!("verify: {e}"));
            }
            let s = pmm::stats();
            let (free, used) = (model.count(1), model.count(2));
            if s.free_frames != free
                || s.in_use_frames != used
                || s.peak_in_use_frames != model.peak_in_use
                || s.frames_allocated_total != model.allocated_total
                || s.frames_freed_total != model.freed_total
                || s.alloc_failures != model.failures
                || s.free_rejected != model.rejected
            {
                fail(seed, step, format!("stats mismatch: {s:?} vs model free {free} used {used} peak {} alloc {} freed {} fail {} rej {}",
                    model.peak_in_use, model.allocated_total, model.freed_total, model.failures, model.rejected));
            }
        }
    }

    // ---- exhaust the machine, then give everything back ------------------------
    for &(a, c) in &live {
        pmm::free_contiguous(a, c).unwrap_or_else(|e| fail(seed, ops, format!("final free failed: {e:?}")));
        for f in (a >> 12)..(a >> 12) + c {
            model.state[f as usize] = 1;
        }
        model.freed_total += c;
    }
    let mut taken = Vec::new();
    while let Some(a) = pmm::alloc_frame() {
        taken.push(a);
    }
    if taken.len() as u64 != model.count(1) {
        fail(seed, ops, format!("exhaustion handed out {} frames, model has {} free", taken.len(), model.count(1)));
    }
    if pmm::alloc_frame().is_some() || pmm::stats().free_frames != 0 {
        fail(seed, ops, "allocation succeeded on an exhausted PMM".into());
    }
    for a in taken {
        pmm::free_frame(a).unwrap_or_else(|e| fail(seed, ops, format!("free after exhaustion failed: {e:?}")));
    }
    if pmm::stats().free_frames != free0 || pmm::stats().in_use_frames != 0 {
        fail(seed, ops, "frames were lost or invented over the run".into());
    }
    if let Err(e) = pmm::verify() {
        fail(seed, ops, format!("final verify: {e}"));
    }
    let _ = Mutex::new(0); // keep the import used
    println!(
        "seed {seed}: ok  ({} ranges, {} frames managed, top frame {top:#x}, {} ops, peak {} frames in use)",
        ranges.len(), managed, ops, model.peak_in_use
    );
}
