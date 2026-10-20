// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! x86 port-mapped I/O.

use core::arch::asm;

/// # Safety
/// Writing to an I/O port can have arbitrary hardware side effects.
#[inline]
pub unsafe fn outb(port: u16, value: u8) {
    unsafe {
        asm!("out dx, al", in("dx") port, in("al") value,
             options(nomem, nostack, preserves_flags));
    }
}

/// # Safety
/// Reading an I/O port can have hardware side effects.
#[inline]
pub unsafe fn inb(port: u16) -> u8 {
    let value: u8;
    unsafe {
        asm!("in al, dx", out("al") value, in("dx") port,
             options(nomem, nostack, preserves_flags));
    }
    value
}

/// # Safety
/// See [`outb`].
#[inline]
pub unsafe fn outw(port: u16, value: u16) {
    unsafe {
        asm!("out dx, ax", in("dx") port, in("ax") value,
             options(nomem, nostack, preserves_flags));
    }
}

/// # Safety
/// See [`inb`].
#[inline]
pub unsafe fn inw(port: u16) -> u16 {
    let value: u16;
    unsafe {
        asm!("in ax, dx", out("ax") value, in("dx") port,
             options(nomem, nostack, preserves_flags));
    }
    value
}

/// # Safety
/// See [`outb`].
#[inline]
pub unsafe fn outl(port: u16, value: u32) {
    unsafe {
        asm!("out dx, eax", in("dx") port, in("eax") value,
             options(nomem, nostack, preserves_flags));
    }
}

/// # Safety
/// See [`inb`].
#[inline]
pub unsafe fn inl(port: u16) -> u32 {
    let value: u32;
    unsafe {
        asm!("in eax, dx", out("eax") value, in("dx") port,
             options(nomem, nostack, preserves_flags));
    }
    value
}
