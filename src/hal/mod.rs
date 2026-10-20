// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Hardware Abstraction Layer (HAL), first cut.
//!
//! Everything that touches the machine directly lives under here so the rest
//! of the kernel never uses ports, `asm!` or raw framebuffer pointers:
//!
//! | module      | what it abstracts                                   |
//! |-------------|-----------------------------------------------------|
//! | `port`      | x86 port I/O (`in`/`out`)                           |
//! | `serial`    | 16550 UART on COM1 (debug channel to the host)      |
//! | `display`   | framebuffer text console (debug channel on screen)  |
//! | `cpu`       | halting, interrupt flag, CPUID                      |
//! | `gdt`       | GDT: ring 0/ring 3 segments, selectors, `enter_ring3` |
//! | `tss`       | TSS: ring-0 stack (`rsp0`) and Interrupt Stack Table |
//! | `idt`       | IDT: 256 gates, exception dispatch, IST usage       |
//! | `pic`       | legacy 8259 PIC: remapped and masked                |
//! | `msr`       | model-specific registers                            |
//! | `paging`    | page-table walk, map/unmap (splits large pages), MMIO |
//! | `apic`      | Local APIC: unmount bootloader alias, mount UC, enable |
//! | `acpi`      | ACPI table discovery (RSDP -> XSDT/RSDT -> find by signature) |
//! | `hpet`      | High Precision Event Timer: main counter as reference clock |
//! | `pit`       | legacy 8254 channel 2: fallback reference clock      |
//! | `timer`     | reference clock selection (HPET, else PIT) + cross-check |
//! | `apic_timer`| LAPIC timer: calibration, one-shot, periodic, per-CPU ticks |
//! | `percpu`    | per-CPU data via the GS base                        |
//! | `smp`       | start application processors (CPU threads/cores)    |
//! | `memory`    | physical memory map and the HHDM                    |
//! | `sync`      | spin lock for shared HAL state                      |
#![allow(dead_code)]

/// Maximum number of logical CPUs (hardware threads) the kernel supports.
/// Each one costs ~100 KiB of statically allocated stacks.
pub const MAX_CPUS: usize = 16;

pub mod acpi;
pub mod apic;
pub mod apic_timer;
pub mod cpu;
pub mod display;
pub mod font;
pub mod gdt;
pub mod hpet;
pub mod idt;
pub mod memory;
pub mod msr;
pub mod percpu;
pub mod paging;
pub mod pic;
pub mod pit;
pub mod port;
pub mod serial;
pub mod smp;
pub mod sync;
pub mod syscall;
pub mod timer;
pub mod tss;

/// What came up during [`init`].
pub struct HalStatus {
    pub serial: bool,
    pub display: Option<display::Geometry>,
}

/// Brings up the output devices. Call this first: after it returns, the
/// logging macros reach whichever devices are available.
pub fn init(framebuffer: Option<display::FramebufferInfo>) -> HalStatus {
    let serial = serial::init();
    let display = framebuffer.and_then(display::init);
    HalStatus { serial, display }
}
