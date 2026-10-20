// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Kernel logging. Every message goes to the serial port (host terminal under
//! `-serial stdio`) and to the on-screen framebuffer console, e.g.:
//!
//! ```text
//! [INFO ] HELLO WORLD!
//! [DEBUG] hhdm offset = 0xffff800000000000
//! ```
//!
//! `trace!` is serial-only, for chatty output such as the full memory map.

use core::fmt::{self, Write};

use crate::hal::display::{Color, CONSOLE};
use crate::hal::serial::COM1;

#[derive(Clone, Copy)]
pub enum Level {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl Level {
    fn label(self) -> &'static str {
        match self {
            Level::Trace => "TRACE",
            Level::Debug => "DEBUG",
            Level::Info => "INFO ",
            Level::Warn => "WARN ",
            Level::Error => "ERROR",
        }
    }

    fn color(self) -> Color {
        match self {
            Level::Trace => Color::GRAY,
            Level::Debug => Color::CYAN,
            Level::Info => Color::GREEN,
            Level::Warn => Color::YELLOW,
            Level::Error => Color::RED,
        }
    }

    fn on_screen(self) -> bool {
        !matches!(self, Level::Trace)
    }
}

pub fn log(level: Level, args: fmt::Arguments) {
    {
        let mut serial = COM1.lock();
        let _ = write!(serial, "[{}] ", level.label());
        let _ = serial.write_fmt(args);
        let _ = serial.write_char('\n');
    }
    if level.on_screen() {
        if let Some(console) = CONSOLE.lock().as_mut() {
            console.set_color(level.color());
            let _ = write!(console, "[{}] ", level.label());
            console.set_color(Color::WHITE);
            let _ = console.write_fmt(args);
            let _ = console.write_char('\n');
        }
    }
}

/// Releases the output locks so the panic handler can always print, even if
/// the panic happened while a log call was in progress.
pub fn unlock_for_panic() {
    // SAFETY: we are about to halt the machine; nothing else will run.
    unsafe {
        COM1.force_unlock();
        CONSOLE.force_unlock();
    }
}

macro_rules! trace { ($($arg:tt)*) => { $crate::log::log($crate::log::Level::Trace, format_args!($($arg)*)) }; }
macro_rules! debug { ($($arg:tt)*) => { $crate::log::log($crate::log::Level::Debug, format_args!($($arg)*)) }; }
macro_rules! info  { ($($arg:tt)*) => { $crate::log::log($crate::log::Level::Info,  format_args!($($arg)*)) }; }
macro_rules! warn  { ($($arg:tt)*) => { $crate::log::log($crate::log::Level::Warn,  format_args!($($arg)*)) }; }
macro_rules! error { ($($arg:tt)*) => { $crate::log::log($crate::log::Level::Error, format_args!($($arg)*)) }; }
