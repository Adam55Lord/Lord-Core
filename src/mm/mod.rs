// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Kernel-core memory management.
//!
//! Lord Core is a *hybrid* kernel (see `docs/12-hybrid-kernel.md`, Figure 1): the
//! Executive's managers run in kernel mode. This module is the memory side of
//! the Executive, and it is deliberately separate from `hal`:
//!
//! * `hal::memory` reports what the **hardware / bootloader** says exists
//!   (the physical memory map, the higher-half direct map);
//! * `hal::paging` edits page tables (mechanism);
//! * `mm::pmm` decides who **owns** each physical frame;
//! * `mm::vmm` decides who owns each **virtual** address and what backs it
//!   (regions, guard pages, W^X, address spaces) - the Virtual Memory Manager;
//! * `mm::heap` is the kernel heap (`Box`, `Vec`, ...), built on the VMM.
//!
//! Layering, bottom to top: `pmm` < `hal::paging` < `vmm` < `heap`. Lock order
//! is `heap -> vmm -> pmm`.

#![allow(dead_code)]

pub mod heap;
pub mod pmm;
pub mod vmm;
