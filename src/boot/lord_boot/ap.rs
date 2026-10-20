// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Starting the application processors (APs) without a bootloader that parks them.
//!
//! Under Lord Boot the APs are not parked by the bootloader: under
//! UEFI the APs are idle in firmware, so the kernel wakes them the architectural way,
//! with **INIT + two STARTUP IPIs**, and carries each one from 16-bit real mode to the
//! kernel's [`ApEntry`] itself.
//!
//! ```text
//!   BSP  start_cpu():   copy the trampoline to a page below 1 MiB (given by the loader),
//!                       patch its data block, copy the kernel's higher-half PML4 slots into
//!                       a transient PML4 that also identity-maps the first 2 MiB,
//!                       INIT, 10 ms, SIPI(page), 200 us, SIPI(page), wait for the flag
//!   AP   real mode -> protected mode -> long mode (transient PML4) -> ap_main (kernel,
//!                       higher half) -> mov cr3, <kernel PML4> -> ApEntry(index, lapic id)
//! ```
//!
//! The AP page tables are built by the bootloader (`BootInfo::ap_pml4_phys`, below 4 GiB
//! because the 32-bit stage loads CR3 with 32 bits); the kernel only refreshes the higher
//! half. The trampoline's data block is shared by every AP, so `start_cpu` waits until the
//! AP it just woke has picked its values up before returning.

use core::arch::{asm, global_asm};
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::boot::{ApEntry, BootCpu};
use crate::hal::sync::StaticCell;
use crate::hal::{apic, cpu, timer, MAX_CPUS};

const AP_STACK_SIZE: usize = 16 * 1024;

#[repr(C, align(16))]
struct ApStack([u8; AP_STACK_SIZE]);

/// Boot stacks for the APs (until each installs its own per-CPU stacks via the GDT/TSS).
static AP_STACKS: StaticCell<[ApStack; MAX_CPUS]> = StaticCell::new([const { ApStack([0; AP_STACK_SIZE]) }; MAX_CPUS]);

static HHDM: AtomicU64 = AtomicU64::new(0);
static AP_BOOT_PHYS: AtomicU64 = AtomicU64::new(0);
static AP_PML4_PHYS: AtomicU64 = AtomicU64::new(0);
/// The kernel's address space, loaded by each AP once it is running in the higher half.
static KERNEL_CR3: AtomicU64 = AtomicU64::new(0);
static ENTRY: AtomicUsize = AtomicUsize::new(0);

pub fn configure(hhdm: u64, ap_boot_phys: u64, ap_pml4_phys: u64) {
    HHDM.store(hhdm, Ordering::Relaxed);
    AP_BOOT_PHYS.store(ap_boot_phys, Ordering::Relaxed);
    AP_PML4_PHYS.store(ap_pml4_phys, Ordering::Relaxed);
}

// The trampoline template. It is copied to the low page; everything position dependent
// is computed from label *differences*, or patched into the data block by `start_cpu`.
global_asm!(
    ".section .text.lordboot_ap,\"ax\"",
    ".balign 16",
    ".global lordboot_ap_start",
    "lordboot_ap_start:",
    ".set lordboot_ap_o_gdtr, lordboot_ap_gdtr - lordboot_ap_start",
    ".set lordboot_ap_o_pm_jmp, lordboot_ap_pm_jmp - lordboot_ap_start",
    ".set lordboot_ap_o_lm_jmp, lordboot_ap_lm_jmp - lordboot_ap_start",
    ".set lordboot_ap_o_cr3, lordboot_ap_cr3 - lordboot_ap_start",
    ".set lordboot_ap_o_stack, lordboot_ap_stack - lordboot_ap_start",
    ".set lordboot_ap_o_arg0, lordboot_ap_arg0 - lordboot_ap_start",
    ".set lordboot_ap_o_arg1, lordboot_ap_arg1 - lordboot_ap_start",
    ".set lordboot_ap_o_entry, lordboot_ap_entry - lordboot_ap_start",
    ".set lordboot_ap_o_started, lordboot_ap_started - lordboot_ap_start",
    ".code16",
    "    cli",
    "    cld",
    "    mov ax, cs",
    "    mov ds, ax",
    // ebx = physical base of this page (CS << 4); survives every mode switch below.
    // No call/pop trick: there is no stack yet.
    "    movzx ebx, ax",
    "    shl ebx, 4",
    "    lgdt [lordboot_ap_o_gdtr]",
    "    mov eax, cr0",
    "    or eax, 1",
    "    mov cr0, eax",
    // jmp far fword ptr [lordboot_ap_pm_jmp]  (66 FF /5, modrm [disp16]); target patched by the BSP
    "    .byte 0x66, 0xFF, 0x2E",
    "    .word lordboot_ap_o_pm_jmp",
    ".code32",
    ".global lordboot_ap_pm",
    "lordboot_ap_pm:",
    "    mov ax, 0x10",
    "    mov ds, ax",
    "    mov es, ax",
    "    mov ss, ax",
    "    mov fs, ax",
    "    mov gs, ax",
    "    mov eax, cr4",
    "    or eax, 0x20", // PAE
    "    mov cr4, eax",
    "    mov eax, [ebx + lordboot_ap_o_cr3]",
    "    mov cr3, eax",
    "    mov ecx, 0xC0000080", // EFER
    "    rdmsr",
    "    or eax, 0x900", // LME | NXE
    "    wrmsr",
    "    mov eax, cr0",
    "    and eax, 0x9FFFFFFF", // clear CD | NW (set after INIT: caches would stay off)
    "    or eax, 0x80010020", // PG | WP | NE
    "    mov cr0, eax",
    // jmp far fword ptr [ebx + lordboot_ap_lm_jmp]  (FF /5, modrm [ebx+disp32])
    "    .byte 0xFF, 0xAB",
    "    .long lordboot_ap_o_lm_jmp",
    ".code64",
    ".global lordboot_ap_lm",
    "lordboot_ap_lm:",
    "    mov rsp, [rbx + lordboot_ap_o_stack]",
    "    mov rdi, [rbx + lordboot_ap_o_arg0]",
    "    mov rsi, [rbx + lordboot_ap_o_arg1]",
    "    mov rax, [rbx + lordboot_ap_o_entry]",
    "    mov dword ptr [rbx + lordboot_ap_o_started], 1",
    "    xor ebp, ebp",
    "    jmp rax",
    // ---- data block (patched in the copy) ----
    ".balign 16",
    ".global lordboot_ap_gdtr",
    "lordboot_ap_gdtr:    .word 4*8 - 1",
    "                .long 0", // base: physical address of lordboot_ap_gdt, patched
    ".balign 8",
    ".global lordboot_ap_pm_jmp",
    "lordboot_ap_pm_jmp:  .long 0", // offset: physical address of lordboot_ap_pm, patched
    "                .word 0x08",
    ".balign 8",
    ".global lordboot_ap_lm_jmp",
    "lordboot_ap_lm_jmp:  .long 0", // physical address of lordboot_ap_lm, patched
    "                .word 0x18",
    ".balign 8",
    ".global lordboot_ap_cr3",
    "lordboot_ap_cr3:     .quad 0",
    ".global lordboot_ap_stack",
    "lordboot_ap_stack:   .quad 0",
    ".global lordboot_ap_arg0",
    "lordboot_ap_arg0:    .quad 0",
    ".global lordboot_ap_arg1",
    "lordboot_ap_arg1:    .quad 0",
    ".global lordboot_ap_entry",
    "lordboot_ap_entry:   .quad 0",
    ".global lordboot_ap_started",
    "lordboot_ap_started: .long 0",
    ".balign 16",
    ".global lordboot_ap_gdt",
    "lordboot_ap_gdt:",
    "    .quad 0",
    "    .quad 0x00CF9A000000FFFF", // 0x08 32-bit code
    "    .quad 0x00CF92000000FFFF", // 0x10 32-bit data
    "    .quad 0x00AF9A000000FFFF", // 0x18 64-bit code
    ".global lordboot_ap_end",
    "lordboot_ap_end:",
    ".code64",
    ".text",
);

unsafe extern "C" {
    static lordboot_ap_start: u8;
    static lordboot_ap_end: u8;
    static lordboot_ap_gdtr: u8;
    static lordboot_ap_pm: u8;
    static lordboot_ap_pm_jmp: u8;
    static lordboot_ap_lm: u8;
    static lordboot_ap_lm_jmp: u8;
    static lordboot_ap_cr3: u8;
    static lordboot_ap_stack: u8;
    static lordboot_ap_arg0: u8;
    static lordboot_ap_arg1: u8;
    static lordboot_ap_entry: u8;
    static lordboot_ap_started: u8;
    static lordboot_ap_gdt: u8;
}

fn off(sym: *const u8) -> usize {
    sym as usize - core::ptr::addr_of!(lordboot_ap_start) as usize
}

/// First higher-half code an AP runs: switch from the transient PML4 to the kernel's own,
/// then enter the kernel's AP entry point (see `hal::smp::ap_entry`).
extern "C" fn ap_main(index: usize, lapic_id: u64) -> ! {
    // SAFETY: KERNEL_CR3 is the BSP's live PML4, set before the AP was woken; the kernel half
    // it maps is a superset of what the transient tables map.
    unsafe { asm!("mov cr3, {}", in(reg) KERNEL_CR3.load(Ordering::Acquire), options(nostack)) };
    // SAFETY: ENTRY holds a valid `ApEntry` stored by `start_cpu`.
    let entry = unsafe { core::mem::transmute::<usize, ApEntry>(ENTRY.load(Ordering::Acquire)) };
    entry(index, lapic_id as u32)
}

fn read_cr3() -> u64 {
    let v: u64;
    // SAFETY: reading CR3 has no side effects.
    unsafe { asm!("mov {}, cr3", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v & 0x000f_ffff_ffff_f000
}

fn wait_icr_idle() {
    for _ in 0..100_000 {
        if apic::read(apic::REG_ICR_LOW) & (1 << 12) == 0 {
            return;
        }
        cpu::pause();
    }
}

fn send_ipi(lapic_id: u32, low: u32) {
    wait_icr_idle();
    apic::write(apic::REG_ICR_HIGH, lapic_id << 24);
    apic::write(apic::REG_ICR_LOW, low);
    wait_icr_idle();
}

fn delay_us(us: u64) {
    if !timer::delay_us(us) {
        // No calibrated clock: a crude TSC-free spin is still better than none.
        for _ in 0..us * 200 {
            cpu::pause();
        }
    }
}

/// `SmpInfo::start_cpu`: makes `cpu` run `entry(index, lapic_id)` (never the BSP).
pub fn start_cpu(cpu: &BootCpu, index: usize, entry: ApEntry) {
    let (hhdm, page, ap_pml4) = (
        HHDM.load(Ordering::Relaxed),
        AP_BOOT_PHYS.load(Ordering::Relaxed),
        AP_PML4_PHYS.load(Ordering::Relaxed),
    );
    if page == 0 || ap_pml4 == 0 || page >= 0xA0000 || index >= MAX_CPUS {
        return;
    }
    let kernel_cr3 = read_cr3();
    KERNEL_CR3.store(kernel_cr3, Ordering::Release);
    ENTRY.store(entry as usize, Ordering::Release);

    let start = core::ptr::addr_of!(lordboot_ap_start);
    let len = core::ptr::addr_of!(lordboot_ap_end) as usize - start as usize;
    debug_assert!(len < 4096);
    let dst = (hhdm + page) as *mut u8;

    // SAFETY: `page` is a whole page the loader reserved for this; the transient PML4 is three
    // pages the loader reserved; the kernel PML4 is the live one (readable through the HHDM).
    unsafe {
        core::ptr::copy_nonoverlapping(start, dst, len);
        let p = |sym: *const u8| dst.add(off(sym));
        // GDT base, the two far-jump targets (physical), CR3 for the 32-bit stage.
        (p(core::ptr::addr_of!(lordboot_ap_gdtr)).add(2) as *mut u32)
            .write_unaligned((page + off(core::ptr::addr_of!(lordboot_ap_gdt)) as u64) as u32);
        (p(core::ptr::addr_of!(lordboot_ap_pm_jmp)) as *mut u32)
            .write_unaligned((page + off(core::ptr::addr_of!(lordboot_ap_pm)) as u64) as u32);
        (p(core::ptr::addr_of!(lordboot_ap_lm_jmp)) as *mut u32)
            .write_unaligned((page + off(core::ptr::addr_of!(lordboot_ap_lm)) as u64) as u32);
        (p(core::ptr::addr_of!(lordboot_ap_cr3)) as *mut u64).write_unaligned(ap_pml4);
        let stack_top = (AP_STACKS.get() as *mut ApStack).add(index) as u64 + AP_STACK_SIZE as u64;
        (p(core::ptr::addr_of!(lordboot_ap_stack)) as *mut u64).write_unaligned(stack_top);
        (p(core::ptr::addr_of!(lordboot_ap_arg0)) as *mut u64).write_unaligned(index as u64);
        (p(core::ptr::addr_of!(lordboot_ap_arg1)) as *mut u64).write_unaligned(u64::from(cpu.lapic_id));
        (p(core::ptr::addr_of!(lordboot_ap_entry)) as *mut u64).write_unaligned(ap_main as *const () as usize as u64);
        (p(core::ptr::addr_of!(lordboot_ap_started)) as *mut u32).write_volatile(0);

        // Refresh the kernel half of the transient PML4 (slot 0, the identity map, is the loader's).
        let src = (hhdm + kernel_cr3) as *const u64;
        let tbl = (hhdm + ap_pml4) as *mut u64;
        for i in 256..512 {
            tbl.add(i).write_volatile(src.add(i).read_volatile());
        }
    }
    core::sync::atomic::fence(Ordering::SeqCst);

    // INIT, then two STARTUP IPIs (Intel SDM vol. 3, "MP initialization protocol").
    send_ipi(cpu.lapic_id, 0x0000_4500); // INIT, level assert
    delay_us(10_000);
    let sipi = 0x0000_4600 | (page >> 12) as u32;
    send_ipi(cpu.lapic_id, sipi);
    delay_us(200);
    // A second SIPI is harmless if the AP already started (it ignores it once running).
    let started = (dst as usize + off(core::ptr::addr_of!(lordboot_ap_started))) as *const u32;
    // SAFETY: `started` points into the HHDM copy of the trampoline's data block.
    if unsafe { started.read_volatile() } == 0 {
        send_ipi(cpu.lapic_id, sipi);
    }
    // Wait (bounded, ~1 s) until the AP has loaded its arguments: the data block is shared.
    for _ in 0..10_000 {
        // SAFETY: as above.
        if unsafe { started.read_volatile() } != 0 {
            return;
        }
        delay_us(100);
    }
}
