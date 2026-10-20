// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Physical memory map and the higher-half direct map (HHDM).
//!
//! This module only reports what the hardware / bootloader says exists
//! (the boot backend converts the bootloader's map into [`Region`]s).
//! Who *owns* each frame is decided by the physical memory manager,
//! `mm::pmm`.

use core::sync::atomic::{AtomicU64, Ordering};

use super::sync::IrqLock;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RegionKind {
    Usable,
    Reserved,
    AcpiReclaimable,
    AcpiNvs,
    BadMemory,
    BootloaderReclaimable,
    KernelAndModules,
    Framebuffer,
    ReservedMapped,
    Unknown,
}

impl RegionKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Usable => "Usable",
            Self::Reserved => "Reserved",
            Self::AcpiReclaimable => "ACPI reclaimable",
            Self::AcpiNvs => "ACPI NVS",
            Self::BadMemory => "Bad memory",
            Self::BootloaderReclaimable => "Bootloader reclaimable",
            Self::KernelAndModules => "Kernel and modules",
            Self::Framebuffer => "Framebuffer",
            Self::ReservedMapped => "Reserved (mapped)",
            Self::Unknown => "Unknown",
        }
    }
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

static HHDM_OFFSET: AtomicU64 = AtomicU64::new(0);
static MAP: IrqLock<Option<&'static [Region]>> = IrqLock::new(None);

/// Records the HHDM offset and the physical memory map (both come from
/// `boot::BootInfo`; this module knows nothing about the bootloader).
pub fn init(hhdm_offset: u64, regions: &'static [Region]) {
    HHDM_OFFSET.store(hhdm_offset, Ordering::Relaxed);
    *MAP.lock() = Some(regions);
}

pub fn hhdm_offset() -> u64 {
    HHDM_OFFSET.load(Ordering::Relaxed)
}

/// Translates a physical address into its HHDM virtual address.
pub fn phys_to_virt(phys: u64) -> *mut u8 {
    (hhdm_offset().wrapping_add(phys)) as *mut u8
}

pub fn for_each_region(mut f: impl FnMut(Region)) {
    let map = *MAP.lock();
    if let Some(regions) = map {
        for r in regions {
            f(*r);
        }
    }
}

pub fn region_count() -> usize {
    (*MAP.lock()).map_or(0, |m| m.len())
}

pub fn usable_bytes() -> u64 {
    let mut total = 0u64;
    for_each_region(|r| {
        if r.kind == RegionKind::Usable {
            total += r.length;
        }
    });
    total
}
