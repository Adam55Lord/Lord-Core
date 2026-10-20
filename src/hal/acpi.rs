// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Minimal ACPI table discovery: RSDP -> RSDT/XSDT -> table directory.
//!
//! Lord Core does not interpret AML or the namespace. All it needs today is to *find
//! a table by its 4-character signature* (currently `HPET`; `APIC`/MADT is next
//! on the roadmap) and read plain fields from it.
//!
//! Where the tables live depends on the bootloader:
//!
//! * a restrictive HHDM: the RSDP address is *physical* and ACPI memory is not mapped;
//! * a full HHDM: ACPI memory is mapped, RSDP may be a HHDM address.
//!
//! [`init`] accepts either form. [`map`] uses the HHDM alias when every page of
//! the range is really mapped there and otherwise mounts the range through the
//! uncacheable MMIO window, so the code works in both cases.
//!
//! Call order: after `memory::init` and the paging layer are usable, on the boot
//! CPU before the other CPUs start (the directory is built once and only read
//! afterwards).

use core::fmt;
use core::sync::atomic::{AtomicUsize, Ordering};

use super::sync::IrqLock;
use super::{memory, paging};

/// Size of the standard ACPI System Description Table header.
pub const SDT_HEADER_LEN: usize = 36;
/// Directory capacity. Real machines have ~15-30 tables.
pub const MAX_TABLES: usize = 64;
/// Refuse to map absurdly large tables (corrupt length fields).
const MAX_TABLE_LEN: u32 = 4 << 20;

const PAGE: u64 = 4096;

/// How many [`map`] calls had to fall back to the MMIO window because the
/// HHDM did not cover the range (0 when the bootloader maps ACPI memory in the HHDM).
static WINDOW_MAPS: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcpiError {
    /// The bootloader gave no RSDP (no ACPI on this machine).
    NoRsdp,
    /// Signature or checksum of the RSDP / root table is wrong.
    BadRsdp,
    /// A table's byte sum is not zero.
    BadChecksum,
    /// A length field is implausible.
    BadLength,
    /// [`init`] has not completed.
    NotInitialised,
    /// No table with that signature exists.
    NotFound,
    /// Mapping the table failed.
    Paging(paging::PagingError),
}

/// One entry of the table directory (the header fields we care about).
#[derive(Clone, Copy)]
pub struct TableInfo {
    pub signature: [u8; 4],
    pub phys: u64,
    pub length: u32,
    pub revision: u8,
}

impl TableInfo {
    const EMPTY: Self = Self { signature: [0; 4], phys: 0, length: 0, revision: 0 };

    pub fn name(&self) -> &str {
        core::str::from_utf8(&self.signature).unwrap_or("????")
    }
}

struct Directory {
    ready: bool,
    rsdp_phys: u64,
    rsdp_revision: u8,
    oem: [u8; 6],
    root_phys: u64,
    root_is_xsdt: bool,
    count: usize,
    /// Tables beyond [`MAX_TABLES`] that did not fit.
    dropped: usize,
    tables: [TableInfo; MAX_TABLES],
}

static DIR: IrqLock<Directory> = IrqLock::new(Directory {
    ready: false,
    rsdp_phys: 0,
    rsdp_revision: 0,
    oem: [b' '; 6],
    root_phys: 0,
    root_is_xsdt: false,
    count: 0,
    dropped: 0,
    tables: [TableInfo::EMPTY; MAX_TABLES],
});

// ----------------------------------------------------------------------
// Raw access helpers
// ----------------------------------------------------------------------

/// Makes `len` bytes at physical address `phys` readable and returns a
/// pointer to the first byte.
///
/// Uses the HHDM alias if all pages of the range are present there and map to
/// the expected frames; otherwise mounts the range uncacheable in the MMIO
/// window (a few pages per table at most; never unmapped).
pub fn map(phys: u64, len: usize) -> Result<*const u8, AcpiError> {
    let first = phys & !(PAGE - 1);
    let last = (phys + len as u64 + PAGE - 1) & !(PAGE - 1);
    let mut alias_ok = true;
    let mut p = first;
    while p < last {
        let virt = memory::phys_to_virt(p) as u64;
        match paging::translate(virt) {
            Some(m) if m.phys == p => {}
            _ => {
                alias_ok = false;
                break;
            }
        }
        p += PAGE;
    }
    if alias_ok {
        return Ok(memory::phys_to_virt(phys));
    }
    WINDOW_MAPS.fetch_add(1, Ordering::Relaxed);
    paging::map_mmio(phys, len).map(|p| p as *const u8).map_err(AcpiError::Paging)
}

/// Number of table ranges that were mounted through the MMIO window instead
/// of being read through the HHDM.
pub fn window_maps() -> usize {
    WINDOW_MAPS.load(Ordering::Relaxed)
}

/// Reads a little-endian integer of type `T` at `ptr + off` (unaligned).
macro_rules! rd {
    ($ptr:expr, $off:expr, $t:ty) => {
        // SAFETY: callers bound-check `$off + size_of::<$t>()` against the
        // length they mapped.
        unsafe { ($ptr.add($off) as *const $t).read_unaligned() }
    };
}

fn checksum_ok(ptr: *const u8, len: usize) -> bool {
    let mut sum = 0u8;
    for i in 0..len {
        // SAFETY: `ptr..ptr+len` was mapped by the caller.
        sum = sum.wrapping_add(unsafe { ptr.add(i).read() });
    }
    sum == 0
}

// ----------------------------------------------------------------------
// A mapped, checksum-verified table
// ----------------------------------------------------------------------

/// A whole ACPI table, mapped and checksum-verified. Field accessors are
/// bounds-checked and return 0 past the end, so a truncated table can never
/// cause an out-of-bounds read.
pub struct Table {
    ptr: *const u8,
    pub info: TableInfo,
}

impl Table {
    pub fn len(&self) -> usize {
        self.info.length as usize
    }

    pub fn u8_at(&self, off: usize) -> u8 {
        if off + 1 <= self.len() { rd!(self.ptr, off, u8) } else { 0 }
    }

    pub fn u16_at(&self, off: usize) -> u16 {
        if off + 2 <= self.len() { rd!(self.ptr, off, u16) } else { 0 }
    }

    pub fn u32_at(&self, off: usize) -> u32 {
        if off + 4 <= self.len() { rd!(self.ptr, off, u32) } else { 0 }
    }

    pub fn u64_at(&self, off: usize) -> u64 {
        if off + 8 <= self.len() { rd!(self.ptr, off, u64) } else { 0 }
    }
}

// ----------------------------------------------------------------------
// Public API
// ----------------------------------------------------------------------

/// Validates the RSDP at `rsdp_raw` (physical, or HHDM-virtual) and builds the
/// table directory from the XSDT (ACPI 2+) or RSDT.
pub fn init(rsdp_raw: u64) -> Result<(), AcpiError> {
    let hhdm = memory::hhdm_offset();
    let rsdp_phys = if hhdm != 0 && rsdp_raw >= hhdm { rsdp_raw - hhdm } else { rsdp_raw };

    // --- RSDP -------------------------------------------------------------
    let rsdp = map(rsdp_phys, 36)?;
    let mut sig = [0u8; 8];
    for (i, b) in sig.iter_mut().enumerate() {
        *b = rd!(rsdp, i, u8);
    }
    if &sig != b"RSD PTR " || !checksum_ok(rsdp, 20) {
        return Err(AcpiError::BadRsdp);
    }
    let revision = rd!(rsdp, 15, u8);
    let mut oem = [b' '; 6];
    for (i, b) in oem.iter_mut().enumerate() {
        *b = rd!(rsdp, 9 + i, u8);
    }
    let (root_phys, is_xsdt) = if revision >= 2 {
        let len = rd!(rsdp, 20, u32);
        if !(36..=4096).contains(&len) {
            return Err(AcpiError::BadLength);
        }
        let full = map(rsdp_phys, len as usize)?;
        if !checksum_ok(full, len as usize) {
            return Err(AcpiError::BadRsdp);
        }
        let xsdt = rd!(rsdp, 24, u64);
        if xsdt != 0 { (xsdt, true) } else { (rd!(rsdp, 16, u32) as u64, false) }
    } else {
        (rd!(rsdp, 16, u32) as u64, false)
    };
    if root_phys == 0 {
        return Err(AcpiError::BadRsdp);
    }

    // --- XSDT / RSDT --------------------------------------------------------
    let hdr = map(root_phys, SDT_HEADER_LEN)?;
    let expect: &[u8; 4] = if is_xsdt { b"XSDT" } else { b"RSDT" };
    let mut rsig = [0u8; 4];
    for (i, b) in rsig.iter_mut().enumerate() {
        *b = rd!(hdr, i, u8);
    }
    let root_len = rd!(hdr, 4, u32);
    if &rsig != expect {
        return Err(AcpiError::BadRsdp);
    }
    if root_len < SDT_HEADER_LEN as u32 || root_len > MAX_TABLE_LEN {
        return Err(AcpiError::BadLength);
    }
    let root = map(root_phys, root_len as usize)?;
    if !checksum_ok(root, root_len as usize) {
        return Err(AcpiError::BadChecksum);
    }

    let entry_size = if is_xsdt { 8 } else { 4 };
    let entries = (root_len as usize - SDT_HEADER_LEN) / entry_size;

    let mut dir = DIR.lock();
    dir.rsdp_phys = rsdp_phys;
    dir.rsdp_revision = revision;
    dir.oem = oem;
    dir.root_phys = root_phys;
    dir.root_is_xsdt = is_xsdt;
    dir.count = 0;
    dir.dropped = 0;
    for i in 0..entries {
        let off = SDT_HEADER_LEN + i * entry_size;
        let phys = if is_xsdt { rd!(root, off, u64) } else { rd!(root, off, u32) as u64 };
        if phys == 0 {
            continue;
        }
        let Ok(h) = map(phys, SDT_HEADER_LEN) else {
            warn!("acpi: table at {:#x} cannot be mapped, skipped", phys);
            continue;
        };
        let mut signature = [0u8; 4];
        for (j, b) in signature.iter_mut().enumerate() {
            *b = rd!(h, j, u8);
        }
        let info = TableInfo {
            signature,
            phys,
            length: rd!(h, 4, u32),
            revision: rd!(h, 8, u8),
        };
        if dir.count < MAX_TABLES {
            let n = dir.count;
            dir.tables[n] = info;
            dir.count += 1;
        } else {
            dir.dropped += 1;
        }
    }
    dir.ready = true;
    Ok(())
}

pub fn is_ready() -> bool {
    DIR.lock().ready
}

/// Number of tables in the directory.
pub fn table_count() -> usize {
    DIR.lock().count
}

/// Directory entry `index` (0-based, in XSDT/RSDT order).
pub fn table_info(index: usize) -> Option<TableInfo> {
    let dir = DIR.lock();
    if index < dir.count { Some(dir.tables[index]) } else { None }
}

/// First table with this signature, from the directory (header data only).
pub fn find(signature: &[u8; 4]) -> Option<TableInfo> {
    let dir = DIR.lock();
    dir.tables[..dir.count].iter().find(|t| &t.signature == signature).copied()
}

/// Maps and checksum-verifies the first table with this signature.
pub fn table(signature: &[u8; 4]) -> Result<Table, AcpiError> {
    if !is_ready() {
        return Err(AcpiError::NotInitialised);
    }
    let info = find(signature).ok_or(AcpiError::NotFound)?;
    if info.length < SDT_HEADER_LEN as u32 || info.length > MAX_TABLE_LEN {
        return Err(AcpiError::BadLength);
    }
    let ptr = map(info.phys, info.length as usize)?;
    if !checksum_ok(ptr, info.length as usize) {
        return Err(AcpiError::BadChecksum);
    }
    Ok(Table { ptr, info })
}

/// Summary of the RSDP for the boot log.
pub struct Summary {
    pub rsdp_phys: u64,
    pub revision: u8,
    pub oem: [u8; 6],
    pub root_phys: u64,
    pub root_is_xsdt: bool,
    pub tables: usize,
    pub dropped: usize,
}

impl Summary {
    pub fn oem(&self) -> &str {
        core::str::from_utf8(&self.oem).unwrap_or("?").trim_end()
    }
}

pub fn summary() -> Summary {
    let d = DIR.lock();
    Summary {
        rsdp_phys: d.rsdp_phys,
        revision: d.rsdp_revision,
        oem: d.oem,
        root_phys: d.root_phys,
        root_is_xsdt: d.root_is_xsdt,
        tables: d.count,
        dropped: d.dropped,
    }
}

/// `Display` helper: the directory's signatures separated by spaces.
pub struct TableList;

impl fmt::Display for TableList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dir = DIR.lock();
        for (i, t) in dir.tables[..dir.count].iter().enumerate() {
            if i > 0 {
                f.write_str(" ")?;
            }
            f.write_str(t.name())?;
        }
        Ok(())
    }
}
