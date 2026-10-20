// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Minimal 4-level paging layer on top of the page tables the bootloader built.
//!
//! * [`translate`]: walk the tables (handles 1 GiB / 2 MiB / 4 KiB pages).
//! * [`map_page`] / [`unmap_page`] / [`protect_page_in`]: edit them. Unmapping
//!   a 4 KiB page that lives inside a large page **splits** the large page
//!   first (1 GiB -> 512 x 2 MiB -> 512 x 4 KiB), preserving every other
//!   mapping and its attributes.
//! * Every operation has an `_in(root, ..)` form that works on *any* address
//!   space (a PML4 physical address), not only the active one; [`create_root`]
//!   and [`destroy_root`] make and tear down such a space. The policy on top
//!   (regions, ownership, W^X) is `mm::vmm`.
//! * [`init_pat`]: programs the PAT so [`CacheMode`] means the same thing no
//!   matter which bootloader ran before us.
//! * [`map_mmio`]: "mount" device registers uncacheable in a dedicated
//!   virtual window (early-boot bump mapper; `mm::vmm::map_device` is the
//!   general, unmappable version).
//!
//! Limitations: TLB invalidation is local (`invlpg`); there is no cross-CPU
//! shootdown, so change *existing* mappings only while other CPUs are not
//! running (today the APs are parked right after start-up) or never touch
//! those addresses. 5-level paging is not supported.

use core::arch::asm;
use core::fmt;
use core::sync::atomic::{AtomicU64, Ordering};

use super::msr::{self, IA32_EFER};
use super::memory;
use crate::mm::pmm;

pub const PAGE_SIZE: u64 = 4096;
const ADDR_MASK: u64 = 0x000f_ffff_ffff_f000;

/// Page-table entry bits.
pub mod pte {
    pub const PRESENT: u64 = 1 << 0;
    pub const WRITABLE: u64 = 1 << 1;
    pub const USER: u64 = 1 << 2;
    /// PWT: with PCD selects the memory type through the PAT.
    pub const WRITE_THROUGH: u64 = 1 << 3;
    /// PCD: cache disable.
    pub const CACHE_DISABLE: u64 = 1 << 4;
    pub const ACCESSED: u64 = 1 << 5;
    pub const DIRTY: u64 = 1 << 6;
    /// PS: this entry maps a large page (PDPT: 1 GiB, PD: 2 MiB).
    pub const HUGE: u64 = 1 << 7;
    /// PAT bit of a 4 KiB page-table entry.
    pub const PAT_4K: u64 = 1 << 7;
    /// PAT bit of a large-page entry (bit 12, inside the address field).
    pub const PAT_HUGE: u64 = 1 << 12;
    pub const GLOBAL: u64 = 1 << 8;
    /// Software bit 9 (ignored by the CPU): the frame behind this leaf was
    /// taken from the PMM for this mapping and goes back to it on unmap.
    pub const OWNED: u64 = 1 << 9;
    pub const NO_EXECUTE: u64 = 1 << 63;
}

/// Memory types a mapping can ask for, as PAT-selected by [`init_pat`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheMode {
    /// Normal RAM.
    WriteBack,
    WriteThrough,
    /// Strongly uncacheable: device registers.
    Uncached,
    /// Write-combining: framebuffers.
    WriteCombining,
}

impl CacheMode {
    /// Leaf-entry bits (4 KiB pages) that select this type under [`init_pat`]'s layout.
    pub const fn pte_bits(self) -> u64 {
        match self {
            CacheMode::WriteBack => 0,
            CacheMode::WriteThrough => pte::WRITE_THROUGH,
            CacheMode::Uncached => pte::CACHE_DISABLE | pte::WRITE_THROUGH,
            CacheMode::WriteCombining => pte::PAT_4K | pte::WRITE_THROUGH,
        }
    }
}

/// Programs IA32_PAT with the layout the rest of the kernel assumes:
/// `0 WB, 1 WT, 2 UC-, 3 UC, 4 WP, 5 WC, 6 UC-, 7 UC` (the power-on default
/// for 0-3, which is also what the bootloader sets). The PAT is **per CPU**: run this
/// on the boot CPU before creating cache-sensitive mappings and on every AP.
pub fn init_pat() {
    const PAT: u64 = 0x0007_0105_0007_0406;
    // SAFETY: IA32_PAT exists on every x86_64 CPU (CPUID.01H:EDX.PAT is
    // mandatory in long mode); the value only uses defined memory types.
    unsafe { msr::wrmsr(msr::IA32_PAT, PAT) };
}

/// Start of the virtual window used for MMIO mappings (PML4 slot 384, which
/// neither the HHDM nor the kernel image uses).
pub const MMIO_BASE: u64 = 0xFFFF_C000_0000_0000;
/// Size of the MMIO window.
pub const MMIO_WINDOW: u64 = 1 << 30;
/// A scratch page address just above the window, used by the self-test.
const SCRATCH_VIRT: u64 = MMIO_BASE + MMIO_WINDOW;

static MMIO_NEXT: AtomicU64 = AtomicU64::new(MMIO_BASE);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PagingError {
    Unaligned,
    NotMapped,
    AlreadyMapped,
    LargePageInTheWay,
    OutOfFrames,
    FiveLevelPaging,
    WindowExhausted,
}

/// What [`translate`] found for an address.
#[derive(Clone, Copy)]
pub struct Mapping {
    /// Physical address of the 4 KiB page containing the queried address.
    pub phys: u64,
    /// Size of the page that maps it: 4 KiB, 2 MiB or 1 GiB.
    pub page_size: u64,
    /// The raw leaf page-table entry.
    pub entry: u64,
}

impl Mapping {
    fn pat_bit(&self) -> bool {
        let bit = if self.page_size == PAGE_SIZE { pte::PAT_4K } else { pte::PAT_HUGE };
        self.entry & bit != 0
    }

    /// Memory type, assuming the PAT layout programmed by [`init_pat`].
    pub fn memory_type(&self) -> &'static str {
        let index = ((self.pat_bit() as u8) << 2)
            | (((self.entry & pte::CACHE_DISABLE != 0) as u8) << 1)
            | ((self.entry & pte::WRITE_THROUGH != 0) as u8);
        match index {
            0 => "WB",
            1 => "WT",
            2 => "UC-",
            3 => "UC",
            4 => "WP",
            5 => "WC",
            _ => "?",
        }
    }
}

impl fmt::Display for Mapping {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let size = match self.page_size {
            0x4000_0000 => "1 GiB",
            0x20_0000 => "2 MiB",
            _ => "4 KiB",
        };
        write!(
            f,
            "{} page -> {:#x}, {}{}{}, {}",
            size,
            self.phys,
            if self.entry & pte::WRITABLE != 0 { "RW" } else { "RO" },
            if self.entry & pte::USER != 0 { " user" } else { "" },
            if self.entry & pte::NO_EXECUTE != 0 { " NX" } else { "" },
            self.memory_type()
        )
    }
}

/// Result of a successful [`unmap_page`].
#[derive(Clone, Copy)]
pub struct Unmapped {
    pub phys: u64,
    pub entry: u64,
    /// Large pages that had to be split to isolate this one page.
    pub splits: u32,
}

// ----------------------------------------------------------------------
// Low-level helpers
// ----------------------------------------------------------------------

pub fn cr3() -> u64 {
    let v: u64;
    // SAFETY: reading CR3 has no side effects.
    unsafe { asm!("mov {}, cr3", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

fn cr4() -> u64 {
    let v: u64;
    // SAFETY: reading CR4 has no side effects.
    unsafe { asm!("mov {}, cr4", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

fn invlpg(virt: u64) {
    // SAFETY: INVLPG only drops cached translations for one address.
    unsafe { asm!("invlpg [{}]", in(reg) virt, options(nostack, preserves_flags)) };
}

/// Is EFER.NXE set (so bit 63 of an entry means "no execute")?
pub fn nx_enabled() -> bool {
    // SAFETY: IA32_EFER exists on every x86_64 CPU.
    let efer = unsafe { msr::rdmsr(IA32_EFER) };
    efer & (1 << 11) != 0
}

fn ensure_4_level() -> Result<(), PagingError> {
    if cr4() & (1 << 12) != 0 {
        Err(PagingError::FiveLevelPaging)
    } else {
        Ok(())
    }
}

fn indices(virt: u64) -> [usize; 4] {
    [
        ((virt >> 39) & 511) as usize,
        ((virt >> 30) & 511) as usize,
        ((virt >> 21) & 511) as usize,
        ((virt >> 12) & 511) as usize,
    ]
}

/// Pointer to entry `index` of the table stored at physical address `table`,
/// reached through the HHDM (page tables live in HHDM-mapped memory).
fn entry_ptr(table: u64, index: usize) -> *mut u64 {
    (memory::phys_to_virt(table) as *mut u64).wrapping_add(index)
}

fn read_entry(p: *mut u64) -> u64 {
    // SAFETY: `p` points into a live page table (see `entry_ptr`).
    unsafe { p.read_volatile() }
}

fn write_entry(p: *mut u64, value: u64) {
    // SAFETY: `p` points into a live page table (see `entry_ptr`).
    unsafe { p.write_volatile(value) }
}

fn alloc_table() -> Result<u64, PagingError> {
    pmm::alloc_zeroed().ok_or(PagingError::OutOfFrames)
}

// ----------------------------------------------------------------------
// translate / map / unmap
// ----------------------------------------------------------------------

/// Looks up how `virt` is currently mapped, if at all.
pub fn translate(virt: u64) -> Option<Mapping> {
    translate_in(cr3() & ADDR_MASK, virt)
}

/// [`translate`] in the address space whose PML4 is at physical address `root`.
pub fn translate_in(root: u64, virt: u64) -> Option<Mapping> {
    ensure_4_level().ok()?;
    let idx = indices(virt);
    let mut table = root;
    for level in 0..4 {
        let e = read_entry(entry_ptr(table, idx[level]));
        if e & pte::PRESENT == 0 {
            return None;
        }
        let large = (level == 1 || level == 2) && e & pte::HUGE != 0;
        if level == 3 || large {
            let size: u64 = match level {
                1 => 1 << 30,
                2 => 1 << 21,
                _ => PAGE_SIZE,
            };
            // Bit 12 is the PAT flag in large-page entries but a plain address
            // bit in 4 KiB entries, so only mask it for large pages.
            let base = if level == 3 {
                e & ADDR_MASK
            } else {
                e & ADDR_MASK & !pte::PAT_HUGE & !(size - 1)
            };
            let phys = base + ((virt & (size - 1)) & !(PAGE_SIZE - 1));
            return Some(Mapping { phys, page_size: size, entry: e });
        }
        table = e & ADDR_MASK;
    }
    None
}

/// Maps the 4 KiB page at `virt` to `phys` with leaf attributes `flags`
/// (`PRESENT` is added for you). Missing intermediate tables are allocated.
pub fn map_page(virt: u64, phys: u64, flags: u64) -> Result<(), PagingError> {
    map_page_in(cr3() & ADDR_MASK, virt, phys, flags)
}

/// [`map_page`] in the address space whose PML4 is at physical address `root`.
pub fn map_page_in(root: u64, virt: u64, phys: u64, flags: u64) -> Result<(), PagingError> {
    if virt & (PAGE_SIZE - 1) != 0 || phys & (PAGE_SIZE - 1) != 0 {
        return Err(PagingError::Unaligned);
    }
    ensure_4_level()?;
    let idx = indices(virt);
    let user = flags & pte::USER;
    let mut table = root;
    for level in 0..3 {
        let ep = entry_ptr(table, idx[level]);
        let e = read_entry(ep);
        if e & pte::PRESENT == 0 {
            let frame = alloc_table()?;
            write_entry(ep, frame | pte::PRESENT | pte::WRITABLE | user);
            table = frame;
        } else if level > 0 && e & pte::HUGE != 0 {
            return Err(PagingError::LargePageInTheWay);
        } else {
            if user != 0 && e & pte::USER == 0 {
                write_entry(ep, e | pte::USER);
            }
            table = e & ADDR_MASK;
        }
    }
    let ep = entry_ptr(table, idx[3]);
    if read_entry(ep) & pte::PRESENT != 0 {
        return Err(PagingError::AlreadyMapped);
    }
    write_entry(ep, phys | flags | pte::PRESENT);
    invlpg(virt);
    Ok(())
}

/// Replaces the large-page entry at `ep` (found at `level` 1 = PDPT or
/// 2 = PD) with a table of next-size-down pages that map exactly the same
/// memory with the same attributes. Returns the new table's physical address.
fn split_large_page(ep: *mut u64, e: u64, level: usize) -> Result<u64, PagingError> {
    let table = alloc_table()?;
    let entries = memory::phys_to_virt(table) as *mut u64;
    let pat = e & pte::PAT_HUGE != 0;
    let attrs = e & !ADDR_MASK; // flag bits (incl. NX) without the address
    if level == 1 {
        // 1 GiB -> 512 x 2 MiB. Entries stay "large", PAT stays at bit 12.
        let base = e & ADDR_MASK & !pte::PAT_HUGE & !((1 << 30) - 1);
        for i in 0..512u64 {
            let v = (base + i * (1 << 21)) | attrs | if pat { pte::PAT_HUGE } else { 0 };
            // SAFETY: `table` is a fresh, zeroed frame reachable via the HHDM.
            unsafe { entries.add(i as usize).write_volatile(v) };
        }
    } else {
        // 2 MiB -> 512 x 4 KiB. The PAT bit moves from bit 12 to bit 7.
        let base = e & ADDR_MASK & !pte::PAT_HUGE & !((1 << 21) - 1);
        for i in 0..512u64 {
            let v = (base + i * PAGE_SIZE)
                | (attrs & !pte::HUGE)
                | if pat { pte::PAT_4K } else { 0 };
            // SAFETY: as above.
            unsafe { entries.add(i as usize).write_volatile(v) };
        }
    }
    // Fully built: now link it. The intermediate entry is permissive; the
    // restrictions live in the leaf entries copied above.
    write_entry(ep, table | pte::PRESENT | pte::WRITABLE | (e & pte::USER));
    Ok(table)
}

/// Unmaps the 4 KiB page at `virt`, splitting any large page that covers it.
pub fn unmap_page(virt: u64) -> Result<Unmapped, PagingError> {
    unmap_page_in(cr3() & ADDR_MASK, virt)
}

/// [`unmap_page`] in the address space whose PML4 is at physical address `root`.
pub fn unmap_page_in(root: u64, virt: u64) -> Result<Unmapped, PagingError> {
    if virt & (PAGE_SIZE - 1) != 0 {
        return Err(PagingError::Unaligned);
    }
    ensure_4_level()?;
    let idx = indices(virt);
    let mut table = root;
    let mut splits = 0u32;
    for level in 0..4 {
        let ep = entry_ptr(table, idx[level]);
        let e = read_entry(ep);
        if e & pte::PRESENT == 0 {
            return Err(PagingError::NotMapped);
        }
        if level == 3 {
            write_entry(ep, 0);
            invlpg(virt);
            return Ok(Unmapped { phys: e & ADDR_MASK, entry: e, splits });
        }
        if level >= 1 && e & pte::HUGE != 0 {
            // Split, then continue one level down inside the new table.
            table = split_large_page(ep, e, level)?;
            splits += 1;
            continue;
        }
        table = e & ADDR_MASK;
    }
    Err(PagingError::NotMapped)
}

/// Changes the attributes of the existing 4 KiB mapping at `virt` to `flags`
/// (`PRESENT` is added; the frame address and the `OWNED` / accessed / dirty
/// bits are kept). Returns the *old* entry. Large pages are refused: split
/// them first with [`unmap_page_in`] on a neighbour, or do not use them.
pub fn protect_page_in(root: u64, virt: u64, flags: u64) -> Result<u64, PagingError> {
    if virt & (PAGE_SIZE - 1) != 0 {
        return Err(PagingError::Unaligned);
    }
    ensure_4_level()?;
    let idx = indices(virt);
    let mut table = root;
    for level in 0..4 {
        let ep = entry_ptr(table, idx[level]);
        let e = read_entry(ep);
        if e & pte::PRESENT == 0 {
            return Err(PagingError::NotMapped);
        }
        if level == 3 {
            let keep = e & (ADDR_MASK | pte::OWNED | pte::ACCESSED | pte::DIRTY);
            let new = keep | (flags & !ADDR_MASK & !pte::OWNED) | pte::PRESENT;
            write_entry(ep, new);
            invlpg(virt);
            return Ok(e);
        }
        if level >= 1 && e & pte::HUGE != 0 {
            return Err(PagingError::LargePageInTheWay);
        }
        table = e & ADDR_MASK;
    }
    Err(PagingError::NotMapped)
}

/// Makes sure the top-level (PML4) slot covering `virt` points to a second-level
/// table, allocating an empty one if needed. Returns whether it had to.
/// `mm::vmm::init` uses it to create every kernel-half slot up front, so that
/// address spaces cloned later (see [`create_root`]) share them.
pub fn ensure_pml4_slot(root: u64, virt: u64) -> Result<bool, PagingError> {
    let ep = entry_ptr(root, indices(virt)[0]);
    if read_entry(ep) & pte::PRESENT != 0 {
        return Ok(false);
    }
    let frame = alloc_table()?;
    write_entry(ep, frame | pte::PRESENT | pte::WRITABLE);
    Ok(true)
}

fn table_is_empty(table: u64) -> bool {
    (0..512).all(|i| read_entry(entry_ptr(table, i)) & pte::PRESENT == 0)
}

/// After pages were unmapped around `virt`: frees the page table (PT) and, if
/// that empties it, the page directory (PD) that covered it. Never frees a
/// PDPT or the PML4, so top-level slots shared between address spaces stay
/// valid. Returns how many frames went back to the PMM.
pub fn reap_tables_in(root: u64, virt: u64) -> u32 {
    let idx = indices(virt);
    let pml4e = read_entry(entry_ptr(root, idx[0]));
    if pml4e & pte::PRESENT == 0 {
        return 0;
    }
    let pdpt = pml4e & ADDR_MASK;
    let pdpte = read_entry(entry_ptr(pdpt, idx[1]));
    if pdpte & pte::PRESENT == 0 || pdpte & pte::HUGE != 0 {
        return 0;
    }
    let pd = pdpte & ADDR_MASK;
    let pde_ptr = entry_ptr(pd, idx[2]);
    let pde = read_entry(pde_ptr);
    if pde & pte::PRESENT == 0 || pde & pte::HUGE != 0 {
        return 0;
    }
    let pt = pde & ADDR_MASK;
    if !table_is_empty(pt) {
        return 0;
    }
    write_entry(pde_ptr, 0);
    let mut freed = pmm::free_frame(pt).is_ok() as u32;
    if table_is_empty(pd) {
        write_entry(entry_ptr(pdpt, idx[1]), 0);
        freed += pmm::free_frame(pd).is_ok() as u32;
    }
    freed
}

/// Makes a new address space: an empty user half (PML4 slots 0..255) and a
/// kernel half (256..511) that **shares** every second-level table with
/// `kernel_root`, so kernel mappings are the same in every space.
///
/// Rule: a kernel-half PML4 slot must be populated *before* the first space is
/// created from it (see `mm::vmm::init`), or the copy will not see it.
pub fn create_root(kernel_root: u64) -> Result<u64, PagingError> {
    let root = alloc_table()?;
    for i in 256..512 {
        write_entry(entry_ptr(root, i), read_entry(entry_ptr(kernel_root, i)));
    }
    Ok(root)
}

/// Frees an address space made by [`create_root`]: every page table of the
/// user half, every leaf frame marked [`pte::OWNED`], and the PML4 itself.
/// The kernel half is shared and is **not** touched. Returns the number of
/// frames given back. `root` must not be loaded on any CPU.
pub fn destroy_root(root: u64) -> u64 {
    let mut freed = 0u64;
    for i4 in 0..256 {
        let e4 = read_entry(entry_ptr(root, i4));
        if e4 & pte::PRESENT == 0 {
            continue;
        }
        let pdpt = e4 & ADDR_MASK;
        for i3 in 0..512 {
            let e3 = read_entry(entry_ptr(pdpt, i3));
            if e3 & pte::PRESENT == 0 || e3 & pte::HUGE != 0 {
                continue; // large pages are never created for user spaces
            }
            let pd = e3 & ADDR_MASK;
            for i2 in 0..512 {
                let e2 = read_entry(entry_ptr(pd, i2));
                if e2 & pte::PRESENT == 0 || e2 & pte::HUGE != 0 {
                    continue;
                }
                let pt = e2 & ADDR_MASK;
                for i1 in 0..512 {
                    let e1 = read_entry(entry_ptr(pt, i1));
                    if e1 & pte::PRESENT != 0 && e1 & pte::OWNED != 0 {
                        freed += pmm::free_frame(e1 & ADDR_MASK).is_ok() as u64;
                    }
                }
                freed += pmm::free_frame(pt).is_ok() as u64;
            }
            freed += pmm::free_frame(pd).is_ok() as u64;
        }
        freed += pmm::free_frame(pdpt).is_ok() as u64;
    }
    freed += pmm::free_frame(root).is_ok() as u64;
    freed
}

/// Loads `root` into CR3 (flushes all non-global TLB entries of this CPU).
///
/// # Safety
/// `root` must be a valid PML4 whose kernel half maps the running code, the
/// current stack and the IDT/GDT/TSS.
pub unsafe fn load_root(root: u64) {
    // SAFETY: forwarded to the caller.
    unsafe { asm!("mov cr3, {}", in(reg) root, options(nostack, preserves_flags)) };
}

/// Physical address of the active PML4.
pub fn current_root() -> u64 {
    cr3() & ADDR_MASK
}

/// Flushes one page from this CPU's TLB.
pub fn flush_page(virt: u64) {
    invlpg(virt);
}

/// "Mounts" `len` bytes of device memory at physical address `phys` into the
/// MMIO window as **uncacheable** (PCD+PWT = PAT3 = UC), no-execute, and
/// returns a pointer to the first byte.
pub fn map_mmio(phys: u64, len: usize) -> Result<*mut u8, PagingError> {
    let start = phys & !(PAGE_SIZE - 1);
    let offset = phys - start;
    let pages = (offset + len as u64 + PAGE_SIZE - 1) / PAGE_SIZE;
    let virt = MMIO_NEXT.fetch_add(pages * PAGE_SIZE, Ordering::AcqRel);
    if virt + pages * PAGE_SIZE > MMIO_BASE + MMIO_WINDOW {
        return Err(PagingError::WindowExhausted);
    }
    let mut flags = pte::WRITABLE | pte::CACHE_DISABLE | pte::WRITE_THROUGH;
    if nx_enabled() {
        flags |= pte::NO_EXECUTE;
    }
    for i in 0..pages {
        map_page(virt + i * PAGE_SIZE, start + i * PAGE_SIZE, flags)?;
    }
    Ok((virt + offset) as *mut u8)
}

/// Exercises map, translate and unmap (including large-page splitting) on a
/// freshly allocated frame. Returns a short description on success.
pub fn self_test() -> Result<u32, &'static str> {
    // Address bit 12 must be set: it doubles as the PAT flag in large-page
    // entries, which is an easy thing to get wrong. An 8 KiB-aligned pair
    // always has exactly one frame with that bit set; use it, return the other.
    let pair = pmm::alloc_contiguous_zeroed(2, 2, pmm::NO_LIMIT).ok_or("out of frames")?;
    let frame = pair + PAGE_SIZE;
    pmm::free_frame(pair).map_err(|_| "could not return the spare frame")?;
    let result = self_test_frame(frame);
    if result.is_ok() {
        // The HHDM alias is whole again; the frame can go back to the PMM.
        pmm::free_frame(frame).map_err(|_| "could not return the test frame")?;
    }
    result
}

fn self_test_frame(frame: u64) -> Result<u32, &'static str> {
    let alias = memory::phys_to_virt(frame) as u64;
    let mut rw = pte::WRITABLE;
    if nx_enabled() {
        rw |= pte::NO_EXECUTE;
    }

    // 1) Map the frame at a scratch address; both views must see the same RAM.
    map_page(SCRATCH_VIRT, frame, rw).map_err(|_| "map_page failed")?;
    // SAFETY: both addresses map the same private, zeroed frame.
    unsafe {
        (SCRATCH_VIRT as *mut u64).write_volatile(0x1234_5678_9ABC_DEF0);
        if (alias as *const u64).read_volatile() != 0x1234_5678_9ABC_DEF0 {
            return Err("scratch mapping does not alias the frame");
        }
    }
    unmap_page(SCRATCH_VIRT).map_err(|_| "unmap_page failed")?;
    if translate(SCRATCH_VIRT).is_some() {
        return Err("scratch page still mapped after unmap");
    }

    // 2) Unmap this frame's page out of the HHDM (splitting whatever large
    //    page covers it), check it is gone, then map it back.
    let before = translate(alias).ok_or("frame missing from HHDM")?;
    let unmapped = unmap_page(alias).map_err(|_| "HHDM unmap failed")?;
    if translate(alias).is_some() {
        return Err("HHDM alias still mapped after unmap");
    }
    // Rebuild the 4 KiB leaf attributes (the PAT bit moves from bit 12 of a
    // large page to bit 7 of a 4 KiB entry).
    let leaf = (before.entry & !ADDR_MASK & !pte::PAT_4K) | if before.pat_bit() { pte::PAT_4K } else { 0 };
    map_page(alias, frame, leaf).map_err(|_| "re-map of HHDM alias failed")?;
    if translate(alias).map(|m| m.phys) != Some(frame) {
        return Err("HHDM alias wrong after re-map");
    }
    Ok(unmapped.splits)
}

/// PML4 slot 385: unused by the bootloader and by us, borrowed by the large-page test.
const LARGE_TEST_VIRT: u64 = 0xFFFF_C080_0000_0000;

/// Builds a synthetic 1 GiB page, then unmaps one 4 KiB page from the middle
/// of it, forcing a 1 GiB -> 2 MiB -> 4 KiB split. Checks that the victim is
/// gone and that every neighbour keeps its physical address and attributes
/// (RW, NX and the PAT-selected memory type). Cleans up after itself.
pub fn self_test_large_pages() -> Result<u32, &'static str> {
    ensure_4_level().map_err(|_| "5-level paging")?;
    let idx = indices(LARGE_TEST_VIRT);
    let pml4_entry = entry_ptr(cr3() & ADDR_MASK, idx[0]);
    if read_entry(pml4_entry) & pte::PRESENT != 0 {
        return Err("test PML4 slot unexpectedly in use");
    }
    let pdpt = alloc_table().map_err(|_| "out of frames")?;
    write_entry(pml4_entry, pdpt | pte::PRESENT | pte::WRITABLE);

    let result = (|| {
        let phys: u64 = 0x4000_0000;
        // RW, NX, and PAT+PWT = PAT index 5 = write-combining (layout of [`init_pat`]).
        let attrs = pte::PRESENT | pte::WRITABLE | pte::HUGE | pte::NO_EXECUTE | pte::PAT_HUGE | pte::WRITE_THROUGH;
        write_entry(entry_ptr(pdpt, idx[1]), phys | attrs);

        let m = translate(LARGE_TEST_VIRT + 0x20_3000).ok_or("1 GiB page not found")?;
        if m.page_size != 1 << 30 || m.phys != phys + 0x20_3000 || m.memory_type() != "WC" {
            return Err("1 GiB translation wrong");
        }

        let victim = LARGE_TEST_VIRT + 0x50_3000;
        let u = unmap_page(victim).map_err(|_| "unmap inside 1 GiB page failed")?;
        if u.splits != 2 || u.phys != phys + 0x50_3000 {
            return Err("split count or unmapped address wrong");
        }
        if translate(victim).is_some() {
            return Err("victim page still mapped");
        }
        for off in [0x0u64, 0x4F_F000, 0x50_2000, 0x50_4000, 0x60_0000, 0x3FFF_F000] {
            let n = translate(LARGE_TEST_VIRT + off).ok_or("neighbour page lost")?;
            if n.phys != phys + off {
                return Err("neighbour physical address changed");
            }
            if n.memory_type() != "WC" || n.entry & pte::WRITABLE == 0 || n.entry & pte::NO_EXECUTE == 0 {
                return Err("neighbour attributes changed by the split");
            }
        }
        if translate(LARGE_TEST_VIRT + 0x50_2000).map(|m| m.page_size) != Some(PAGE_SIZE)
            || translate(LARGE_TEST_VIRT + 0x60_0000).map(|m| m.page_size) != Some(1 << 21)
        {
            return Err("unexpected page sizes after split");
        }
        Ok(u.splits)
    })();

    write_entry(pml4_entry, 0); // remove the synthetic mapping again
    invlpg(LARGE_TEST_VIRT + 0x50_3000);

    // Give every table frame of the synthetic mapping back: the PDPT, the PD
    // the 1 GiB page was split into, and the PT of the 2 MiB page split below it.
    let pd_entry = read_entry(entry_ptr(pdpt, idx[1]));
    if pd_entry & pte::PRESENT != 0 && pd_entry & pte::HUGE == 0 {
        let pd = pd_entry & ADDR_MASK;
        for i in 0..512 {
            let e = read_entry(entry_ptr(pd, i));
            if e & pte::PRESENT != 0 && e & pte::HUGE == 0 {
                let _ = pmm::free_frame(e & ADDR_MASK);
            }
        }
        let _ = pmm::free_frame(pd);
    }
    let _ = pmm::free_frame(pdpt);
    result
}
