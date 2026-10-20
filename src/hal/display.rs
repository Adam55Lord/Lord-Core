// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Framebuffer text console.
//!
//! Draws 8x8 glyphs (optionally pixel-doubled/tripled on large screens) onto
//! the linear framebuffer the bootloader set up. Supports newline, carriage
//! return, tab, wrapping and scrolling. Only 32-bit-per-pixel framebuffers
//! are supported (what UEFI GOP gives you in practice).

use core::fmt;
use core::ptr;

use super::font::{FONT, GLYPH_H, GLYPH_W};
use super::sync::IrqLock;

#[derive(Clone, Copy)]
pub struct Color(pub u8, pub u8, pub u8);

impl Color {
    pub const BLACK: Color = Color(0x00, 0x00, 0x00);
    pub const WHITE: Color = Color(0xE8, 0xE8, 0xE8);
    pub const GRAY: Color = Color(0x90, 0x90, 0x98);
    pub const GREEN: Color = Color(0x50, 0xE0, 0x70);
    pub const YELLOW: Color = Color(0xF0, 0xD0, 0x40);
    pub const RED: Color = Color(0xFF, 0x55, 0x55);
    pub const CYAN: Color = Color(0x50, 0xD0, 0xE8);
}

/// Position of one colour channel inside a pixel.
#[derive(Clone, Copy)]
pub struct Channel {
    pub size: u8,
    pub shift: u8,
}

/// Everything the console needs to know about the framebuffer.
#[derive(Clone, Copy)]
pub struct FramebufferInfo {
    pub address: *mut u8,
    pub width: usize,
    pub height: usize,
    pub pitch: usize,
    pub bpp: u16,
    pub red: Channel,
    pub green: Channel,
    pub blue: Channel,
}

/// Resulting text-mode geometry, for diagnostics.
#[derive(Clone, Copy)]
pub struct Geometry {
    pub width: usize,
    pub height: usize,
    pub bpp: u16,
    pub cols: usize,
    pub rows: usize,
    pub scale: usize,
}

pub struct Console {
    fb: FramebufferInfo,
    scale: usize,
    cell: usize,
    margin: usize,
    cols: usize,
    rows: usize,
    col: usize,
    row: usize,
    fg: u32,
    bg: u32,
}

// SAFETY: the framebuffer pointer is only used while holding the CONSOLE lock.
unsafe impl Send for Console {}

impl Console {
    pub fn new(fb: FramebufferInfo) -> Option<Self> {
        if fb.address.is_null() || fb.bpp != 32 || fb.width == 0 || fb.height == 0 {
            return None;
        }
        let scale = (fb.width / 640).clamp(1, 4);
        let cell = GLYPH_W * scale;
        let margin = 4 * scale;
        let cols = fb.width.saturating_sub(2 * margin) / cell;
        let rows = fb.height.saturating_sub(2 * margin) / (GLYPH_H * scale);
        if cols == 0 || rows == 0 {
            return None;
        }
        let mut console = Self { fb, scale, cell, margin, cols, rows, col: 0, row: 0, fg: 0, bg: 0 };
        console.fg = console.pack(Color::WHITE);
        console.bg = console.pack(Color(0x08, 0x0A, 0x14));
        Some(console)
    }

    pub fn geometry(&self) -> Geometry {
        Geometry {
            width: self.fb.width,
            height: self.fb.height,
            bpp: self.fb.bpp,
            cols: self.cols,
            rows: self.rows,
            scale: self.scale,
        }
    }

    fn pack(&self, c: Color) -> u32 {
        let channel = |value: u8, ch: Channel| -> u32 {
            let size = ch.size.min(8) as u32;
            ((value as u32) >> (8 - size)) << ch.shift
        };
        channel(c.0, self.fb.red) | channel(c.1, self.fb.green) | channel(c.2, self.fb.blue)
    }

    fn fill_rect(&self, x: usize, y: usize, w: usize, h: usize, color: u32) {
        for yy in y..(y + h).min(self.fb.height) {
            // SAFETY: the row lies inside the framebuffer (clamped above) and
            // the pitch/pixel size come from the bootloader.
            let row = unsafe { self.fb.address.add(yy * self.fb.pitch + x * 4) } as *mut u32;
            for xx in 0..w.min(self.fb.width.saturating_sub(x)) {
                unsafe { row.add(xx).write_volatile(color) };
            }
        }
    }

    pub fn clear(&mut self) {
        self.fill_rect(0, 0, self.fb.width, self.fb.height, self.bg);
        self.col = 0;
        self.row = 0;
    }

    pub fn set_color(&mut self, c: Color) {
        self.fg = self.pack(c);
    }

    fn draw_glyph(&self, col: usize, row: usize, ch: usize) {
        let glyph = &FONT[ch];
        let x0 = self.margin + col * self.cell;
        let y0 = self.margin + row * GLYPH_H * self.scale;
        for (gy, bits) in glyph.iter().enumerate() {
            for gx in 0..GLYPH_W {
                let on = (*bits >> gx) & 1 == 1;
                let color = if on { self.fg } else { self.bg };
                self.fill_rect(x0 + gx * self.scale, y0 + gy * self.scale, self.scale, self.scale, color);
            }
        }
    }

    fn newline(&mut self) {
        self.col = 0;
        if self.row + 1 < self.rows {
            self.row += 1;
        } else {
            self.scroll();
        }
    }

    fn scroll(&mut self) {
        let line_h = GLYPH_H * self.scale;
        let pitch = self.fb.pitch;
        let top = self.margin * pitch;
        // SAFETY: source and destination are both inside the text area of the
        // framebuffer; `ptr::copy` handles the overlap.
        unsafe {
            ptr::copy(
                self.fb.address.add(top + line_h * pitch),
                self.fb.address.add(top),
                (self.rows - 1) * line_h * pitch,
            );
        }
        let last_y = self.margin + (self.rows - 1) * line_h;
        self.fill_rect(0, last_y, self.fb.width, line_h, self.bg);
    }

    pub fn write_char(&mut self, c: char) {
        match c {
            '\n' => self.newline(),
            '\r' => self.col = 0,
            '\t' => {
                let next = (self.col / 4 + 1) * 4;
                while self.col < next.min(self.cols) {
                    self.draw_glyph(self.col, self.row, b' ' as usize);
                    self.col += 1;
                }
            }
            c => {
                if self.col >= self.cols {
                    self.newline();
                }
                let code = c as u32;
                // Anything outside basic ASCII or a control char shows as '?'.
                let idx = if (0x20..0x7F).contains(&code) { code as usize } else { b'?' as usize };
                self.draw_glyph(self.col, self.row, idx);
                self.col += 1;
            }
        }
    }
}

impl fmt::Write for Console {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for c in s.chars() {
            self.write_char(c);
        }
        Ok(())
    }
}

pub static CONSOLE: IrqLock<Option<Console>> = IrqLock::new(None);

/// Brings up the framebuffer console. Returns its geometry on success.
pub fn init(fb: FramebufferInfo) -> Option<Geometry> {
    let mut console = Console::new(fb)?;
    console.clear();
    let geometry = console.geometry();
    *CONSOLE.lock() = Some(console);
    Some(geometry)
}
