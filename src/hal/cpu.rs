// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! CPU helpers: halting, interrupt flag, CPUID identification.

use core::arch::asm;
use core::arch::x86_64::CpuidResult;

pub fn halt() {
    // SAFETY: `hlt` just idles the core until the next interrupt.
    unsafe { asm!("hlt", options(nomem, nostack, preserves_flags)) };
}

/// Parks the CPU forever with interrupts masked.
pub fn halt_forever() -> ! {
    disable_interrupts();
    loop {
        halt();
    }
}

pub fn disable_interrupts() {
    // SAFETY: clearing IF is always sound in ring 0.
    unsafe { asm!("cli", options(nomem, nostack)) };
}

pub fn enable_interrupts() {
    // SAFETY: caller is responsible for having a valid IDT first.
    unsafe { asm!("sti", options(nomem, nostack)) };
}

#[allow(unused_unsafe)] // `__cpuid_count` is a safe fn on newer toolchains
fn cpuid(leaf: u32, sub_leaf: u32) -> CpuidResult {
    // SAFETY: CPUID is available on every x86_64 CPU.
    unsafe { core::arch::x86_64::__cpuid_count(leaf, sub_leaf) }
}

pub struct CpuInfo {
    vendor: [u8; 12],
    brand: [u8; 48],
}

impl CpuInfo {
    pub fn detect() -> Self {
        let l0 = cpuid(0, 0);
        let mut vendor = [0u8; 12];
        vendor[0..4].copy_from_slice(&l0.ebx.to_le_bytes());
        vendor[4..8].copy_from_slice(&l0.edx.to_le_bytes());
        vendor[8..12].copy_from_slice(&l0.ecx.to_le_bytes());

        let mut brand = [0u8; 48];
        if cpuid(0x8000_0000, 0).eax >= 0x8000_0004 {
            for i in 0..3u32 {
                let r = cpuid(0x8000_0002 + i, 0);
                let o = (i as usize) * 16;
                brand[o..o + 4].copy_from_slice(&r.eax.to_le_bytes());
                brand[o + 4..o + 8].copy_from_slice(&r.ebx.to_le_bytes());
                brand[o + 8..o + 12].copy_from_slice(&r.ecx.to_le_bytes());
                brand[o + 12..o + 16].copy_from_slice(&r.edx.to_le_bytes());
            }
        }
        Self { vendor, brand }
    }

    pub fn vendor(&self) -> &str {
        core::str::from_utf8(&self.vendor).unwrap_or("?")
    }

    pub fn brand(&self) -> &str {
        core::str::from_utf8(&self.brand)
            .unwrap_or("?")
            .trim_matches(|c: char| c == '\0' || c == ' ')
    }
}

pub fn pause() {
    core::hint::spin_loop();
}

/// Time-stamp counter (monotonic on any modern CPU; ticks at a fixed rate).
pub fn rdtsc() -> u64 {
    // SAFETY: RDTSC is available on every x86_64 CPU.
    unsafe { core::arch::x86_64::_rdtsc() }
}

/// Raises a breakpoint exception (`int3`). Used to test the IDT.
pub fn breakpoint() {
    // SAFETY: `int3` only raises #BP, which the IDT handles.
    unsafe { asm!("int3", options(nomem, nostack)) };
}

/// CPUID.1:EDX bit 9: the CPU has an on-chip local APIC.
pub fn has_apic() -> bool {
    cpuid(1, 0).edx & (1 << 9) != 0
}

/// CPUID.1:ECX bit 24: the LAPIC timer supports TSC-deadline mode.
pub fn has_tsc_deadline() -> bool {
    cpuid(1, 0).ecx & (1 << 24) != 0
}

/// CPUID.6:EAX bit 2 (ARAT): the LAPIC timer keeps running in deep C-states.
pub fn has_arat() -> bool {
    cpuid(0, 0).eax >= 6 && cpuid(6, 0).eax & (1 << 2) != 0
}

/// CPUID.80000007h:EDX bit 8: the TSC runs at a constant rate in all states.
pub fn has_invariant_tsc() -> bool {
    cpuid(0x8000_0000, 0).eax >= 0x8000_0007 && cpuid(0x8000_0007, 0).edx & (1 << 8) != 0
}

/// The initial APIC ID of the CPU executing this (CPUID leaf 1).
pub fn initial_apic_id() -> u32 {
    cpuid(1, 0).ebx >> 24
}

/// Where a logical CPU sits in the physical topology.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CpuPosition {
    pub package: u32,
    pub core: u32,
    /// Hardware thread within the core (0 for the first / only thread).
    pub thread: u32,
}

/// Thread/core topology of the processor package, from CPUID.
#[derive(Clone, Copy)]
pub struct Topology {
    /// CPUID.1:EDX bit 28: the CPU can have multiple logical processors.
    pub htt: bool,
    /// CPUID.1:EBX[23:16]: logical processors per package (legacy field).
    pub legacy_logical_per_package: u32,
    /// From CPUID leaf 0xB, if the CPU implements it.
    pub threads_per_core: Option<u32>,
    pub logical_per_package: Option<u32>,
    smt_shift: u32,
    core_shift: u32,
}

impl Topology {
    pub fn detect() -> Self {
        let l1 = cpuid(1, 0);
        let mut t = Topology {
            htt: l1.edx & (1 << 28) != 0,
            legacy_logical_per_package: (l1.ebx >> 16) & 0xFF,
            threads_per_core: None,
            logical_per_package: None,
            smt_shift: 0,
            core_shift: 0,
        };
        if cpuid(0, 0).eax >= 0xB && cpuid(0xB, 0).ebx & 0xFFFF != 0 {
            for sub in 0..8 {
                let l = cpuid(0xB, sub);
                let count = l.ebx & 0xFFFF;
                let shift = l.eax & 0x1F;
                match (l.ecx >> 8) & 0xFF {
                    1 => {
                        t.threads_per_core = Some(count);
                        t.smt_shift = shift;
                    }
                    2 => {
                        t.logical_per_package = Some(count);
                        t.core_shift = shift;
                    }
                    0 => break,
                    _ => {}
                }
            }
        }
        t
    }

    /// True if topology can be decoded from APIC IDs (leaf 0xB present).
    pub fn can_decode_apic_ids(&self) -> bool {
        self.threads_per_core.is_some() && self.logical_per_package.is_some()
    }

    /// Splits an APIC ID into package / core / thread numbers. Returns `None`
    /// when this CPU doesn't expose CPUID leaf 0xB.
    pub fn locate(&self, apic_id: u32) -> Option<CpuPosition> {
        if !self.can_decode_apic_ids() {
            return None;
        }
        let mask = |bits: u32| if bits >= 32 { u32::MAX } else { (1u32 << bits) - 1 };
        Some(CpuPosition {
            thread: apic_id & mask(self.smt_shift),
            core: (apic_id >> self.smt_shift) & mask(self.core_shift.saturating_sub(self.smt_shift)),
            package: apic_id >> self.core_shift,
        })
    }
}
