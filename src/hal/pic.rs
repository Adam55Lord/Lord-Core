// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Legacy 8259 PIC. We don't use it: it is remapped away from the CPU
//! exception vectors (0x00-0x1F) and fully masked so it can never deliver a
//! stray IRQ that looks like an exception. The APIC takes over later.

use super::port::{inb, outb};

const PIC1_CMD: u16 = 0x20;
const PIC1_DATA: u16 = 0x21;
const PIC2_CMD: u16 = 0xA0;
const PIC2_DATA: u16 = 0xA1;

pub const PIC1_OFFSET: u8 = 0x20;
pub const PIC2_OFFSET: u8 = 0x28;

fn io_wait() {
    // SAFETY: port 0x80 is the traditional unused "POST code" delay port.
    unsafe { outb(0x80, 0) };
}

/// Remaps both PICs to 0x20..=0x2F and masks every line.
pub fn disable() {
    // SAFETY: standard 8259A initialisation sequence (ICW1-4), then OCW1.
    unsafe {
        let _ = (inb(PIC1_DATA), inb(PIC2_DATA)); // current masks (discarded)
        outb(PIC1_CMD, 0x11); io_wait(); // ICW1: init + ICW4 needed
        outb(PIC2_CMD, 0x11); io_wait();
        outb(PIC1_DATA, PIC1_OFFSET); io_wait(); // ICW2: vector offsets
        outb(PIC2_DATA, PIC2_OFFSET); io_wait();
        outb(PIC1_DATA, 4); io_wait(); // ICW3: slave on IRQ2
        outb(PIC2_DATA, 2); io_wait(); // ICW3: slave identity
        outb(PIC1_DATA, 0x01); io_wait(); // ICW4: 8086 mode
        outb(PIC2_DATA, 0x01); io_wait();
        outb(PIC1_DATA, 0xFF); // mask all
        outb(PIC2_DATA, 0xFF);
    }
}

/// Current interrupt mask registers (master, slave). A set bit = line masked.
pub fn masks() -> (u8, u8) {
    // SAFETY: reading the PIC data ports returns the IMRs without side effects.
    unsafe { (inb(PIC1_DATA), inb(PIC2_DATA)) }
}

/// True once every IRQ line of both PICs is masked.
pub fn is_disabled() -> bool {
    masks() == (0xFF, 0xFF)
}
