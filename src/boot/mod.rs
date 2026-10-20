// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Boot interface: the **only** thing the kernel knows about how it was loaded.
//!
//! The kernel proper (`kmain` and everything below it) never mentions a
//! bootloader. It receives one [`BootInfo`] made of plain kernel types
//! (memory regions, a framebuffer description, CPU descriptors, ...). Each
//! supported bootloader lives in its own backend module below this one, is
//! selected by a Cargo feature (`boot-<name>`), and has exactly three jobs:
//!
//! 1. provide the kernel entry point `_start` (see `linker.ld`),
//! 2. translate the bootloader's own structures into a [`BootInfo`],
//! 3. call [`crate::kmain`].
//!
//! Replacing the bootloader therefore means adding a sibling of `boot/lord_boot/`, a
//! `boot-<name>` feature and a matching bootloader package - no
//! other kernel file changes. See `docs/20-bootloader-porting.md` for the checklist.
//!
//! ```text
//!   firmware -> bootloader -> boot/<backend>::_start -> BootInfo -> kmain
//! ```

use crate::hal::display::FramebufferInfo;
use crate::hal::memory::Region;

#[cfg(feature = "boot-lord-boot")]
mod lord_boot;

#[cfg(not(feature = "boot-lord-boot"))]
compile_error!(
    "no boot backend selected: enable exactly one `boot-*` feature (default: `boot-lord-boot`). \
     See docs/20-bootloader-porting.md."
);

/// Firmware the machine was booted from.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub enum Firmware {
    Bios,
    Efi32,
    Efi64,
    Other,
}

impl Firmware {
    pub fn name(self) -> &'static str {
        match self {
            Firmware::Bios => "BIOS",
            Firmware::Efi32 => "UEFI (32-bit)",
            Firmware::Efi64 => "UEFI (64-bit)",
            Firmware::Other => "other",
        }
    }
}

/// Who loaded us, for the boot log.
#[derive(Clone, Copy)]
pub struct BootloaderInfo {
    /// e.g. `"Lord Boot"`.
    pub name: &'static str,
    /// e.g. `"0.1.0"`.
    pub version: &'static str,
    /// Boot protocol spoken, e.g. `"lxf"`.
    pub protocol: &'static str,
    /// Protocol revision the kernel asked for, if the protocol has revisions.
    pub revision_wanted: Option<u64>,
    /// Protocol revision the bootloader actually used, if it says.
    pub revision_loaded: Option<u64>,
}

/// One logical CPU as reported by the bootloader.
#[derive(Clone, Copy)]
pub struct BootCpu {
    pub lapic_id: u32,
    pub processor_id: u32,
    /// Opaque to the kernel: the backend uses it to find its own record of
    /// this CPU when asked to start it.
    pub handle: usize,
}

/// Where a started application processor (AP) begins in the kernel.
/// `index` is its logical CPU number (BSP = 0, APs from 1).
pub type ApEntry = extern "C" fn(index: usize, lapic_id: u32) -> !;

/// Everything the kernel needs to know about the machine's logical CPUs.
pub struct SmpInfo {
    pub bsp_lapic_id: u32,
    pub x2apic: bool,
    /// Every logical CPU, BSP included.
    pub cpus: &'static [BootCpu],
    /// Starts `cpu` (never the BSP): it must end up running `entry(index,
    /// lapic_id)` on a usable stack with interrupts off. A bootloader that
    /// does not park APs for us can implement this with INIT-SIPI instead.
    pub start_cpu: fn(cpu: &BootCpu, index: usize, entry: ApEntry),
}

/// Complete, bootloader-independent description of how the kernel was loaded.
pub struct BootInfo {
    pub bootloader: Option<BootloaderInfo>,
    pub firmware: Option<Firmware>,
    /// Offset of the higher-half direct map (all physical RAM at
    /// `hhdm_offset + phys`). Required by the memory managers.
    pub hhdm_offset: Option<u64>,
    /// Physical memory map, sorted by address. Empty if unavailable.
    pub memory_map: &'static [Region],
    /// Regions the backend could not store (map longer than its buffer).
    pub memory_map_dropped: usize,
    pub framebuffer: Option<FramebufferInfo>,
    /// **Physical** address of the ACPI RSDP, if the machine has ACPI.
    pub rsdp_phys: Option<u64>,
    pub smp: Option<SmpInfo>,
}
