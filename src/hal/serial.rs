// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! 16550 UART driver (COM1). In QEMU, `-serial stdio` shows this output in
//! the host terminal, which makes it the primary debug channel.

use core::fmt;

use super::port::{inb, outb};
use super::sync::IrqLock;

pub const COM1_BASE: u16 = 0x3F8;

pub struct SerialPort {
    base: u16,
    ready: bool,
}

impl SerialPort {
    pub const fn new(base: u16) -> Self {
        Self { base, ready: false }
    }

    /// Programs 115200 baud, 8N1, FIFOs on, and runs a loopback self-test.
    /// Returns `false` (and stays silent) if no UART answers.
    pub fn init(&mut self) -> bool {
        let b = self.base;
        // SAFETY: standard 16550 register programming on a legacy COM port.
        unsafe {
            outb(b + 1, 0x00); // disable UART interrupts
            outb(b + 3, 0x80); // DLAB on: expose baud divisor
            outb(b, 0x01); // divisor low  = 1 -> 115200 baud
            outb(b + 1, 0x00); // divisor high = 0
            outb(b + 3, 0x03); // DLAB off, 8 data bits, no parity, 1 stop bit
            outb(b + 2, 0xC7); // enable + clear FIFOs, 14-byte threshold
            outb(b + 4, 0x1E); // loopback mode for self-test
            outb(b, 0xAE);
            if inb(b) != 0xAE {
                return false;
            }
            outb(b + 4, 0x0F); // normal operation, OUT1/OUT2, RTS/DTR
        }
        self.ready = true;
        true
    }

    pub fn is_ready(&self) -> bool {
        self.ready
    }

    pub fn write_byte(&self, byte: u8) {
        if !self.ready {
            return;
        }
        // SAFETY: line status / transmit registers of an initialised UART.
        unsafe {
            // Bounded wait so a dead UART can never hang the kernel.
            let mut spins = 0u32;
            while inb(self.base + 5) & 0x20 == 0 {
                spins += 1;
                if spins > 1_000_000 {
                    return;
                }
                core::hint::spin_loop();
            }
            outb(self.base, byte);
        }
    }
}

impl fmt::Write for SerialPort {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for byte in s.bytes() {
            if byte == b'\n' {
                self.write_byte(b'\r');
            }
            self.write_byte(byte);
        }
        Ok(())
    }
}

pub static COM1: IrqLock<SerialPort> = IrqLock::new(SerialPort::new(COM1_BASE));

/// Initialises COM1. Returns whether a working UART was found.
pub fn init() -> bool {
    COM1.lock().init()
}
