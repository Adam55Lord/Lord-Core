// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Boot-time self-test of the LXF loader, with files produced by the real tools
//! (`samples/build-samples.sh` runs `flat2lord.py`): `ltcalc.lxf` has a relocation and imports
//! two functions from `ltmath.llf`, which has a `LibraryMain`. Then a series of damaged copies
//! of it must each be refused for the right reason.

use alloc::vec::Vec;

use super::*;
use crate::ps::structs::{priority, WaitStatus};

const LTCALC: &[u8] = include_bytes!("samples/ltcalc.lxf");
const LTMATH: &[u8] = include_bytes!("samples/ltmath.llf");

/// What [`run`] observed, for the boot log.
pub struct SelfTest {
    pub exit_code: i32,
    pub refused: u32,
}

/// `(offset, size)` of the data of section `name` in `image`.
fn section(image: &[u8], name: &[u8]) -> (usize, usize) {
    for i in 0..u16_at(image, 10) as usize {
        let h = FILE_HEADER + i * SECTION_HEADER;
        let raw = &image[h..h + 8];
        if &raw[..raw.iter().position(|&c| c == 0).unwrap_or(8)] == name {
            return (u32_at(image, h + 8) as usize, u32_at(image, h + 12) as usize);
        }
    }
    panic!("sample without {:?}", name);
}

fn patched(base: &[u8], f: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
    let mut v = base.to_vec();
    f(&mut v);
    v
}

fn put32(v: &mut [u8], at: usize, x: u32) {
    v[at..at + 4].copy_from_slice(&x.to_le_bytes());
}

/// Does `spawn` refuse `image` with exactly `want`?
fn refused(image: &[u8], want: LdError) -> bool {
    matches!(spawn("bad", image, priority::NORMAL), Err(e) if e == want)
}

pub fn run() -> Result<SelfTest, &'static str> {
    let bal0 = super::super::ex::selftest::balance();

    // --- the happy path ---------------------------------------------------
    let p = spawn("ltcalc", LTCALC, priority::NORMAL).map_err(|_| "ltcalc.lxf was not accepted")?;
    match ps::wait_for_object(p.untyped(), Some(10_000)) {
        Ok(WaitStatus::Signaled) => {}
        _ => return Err("ltcalc did not finish"),
    }
    let exit_code = p.exit_code();
    drop(p);
    if exit_code == 1 {
        return Err("relocation failed: ltcalc could not read its own message");
    }
    if exit_code != 1764 {
        return Err("ltcalc returned the wrong value (imports or LibraryMain broken)");
    }

    // --- damaged images ----------------------------------------------------
    let text = section(LTCALC, b".text");
    let idata = section(LTCALC, b".idata");
    let reloc = section(LTCALC, b".reloc");
    let tab = |i: usize| FILE_HEADER + i * SECTION_HEADER; // section header i
    let mut n = 0u32;
    macro_rules! must {
        ($ok:expr, $msg:expr) => {{
            if !$ok {
                return Err($msg);
            }
            n += 1;
        }};
    }
    must!(refused(&LTCALC[..10], LdError::Truncated), "truncated header accepted");
    must!(refused(&LTCALC[..60], LdError::Truncated), "truncated section table accepted");
    must!(refused(&patched(LTCALC, |v| v[0] = b'X'), LdError::BadMagic), "bad magic accepted");
    must!(refused(&patched(LTCALC, |v| v[8] = 2), LdError::BadMagic), "magic/type mismatch accepted");
    must!(refused(&patched(LTCALC, |v| v[4] = 7), LdError::Unsupported("unsupported LXF version")), "bad version accepted");
    must!(refused(&patched(LTCALC, |v| v[6] = 3), LdError::Unsupported("not an x86-64 image")), "wrong architecture accepted");
    must!(refused(&patched(LTCALC, |v| v[16] |= 2), LdError::KernelImage), "kernel image accepted as a program");
    must!(refused(&patched(LTCALC, |v| v[10] = 200), LdError::Malformed("too many sections")), "200 sections accepted");
    must!(refused(&patched(LTCALC, |v| put32(v, tab(0) + 12, 0xFFFF_FFF0)), LdError::Malformed("section overflows")) || refused(&patched(LTCALC, |v| put32(v, tab(0) + 12, 0xFFFF_FFF0)), LdError::Truncated), "huge section size accepted");
    must!(refused(&patched(LTCALC, |v| put32(v, tab(0) + 16, 3)), LdError::Malformed(".text is writable (W^X)")), "writable .text accepted");
    must!(refused(&patched(LTCALC, |v| put32(v, tab(0) + 16, 0)), LdError::Malformed(".text is not executable")), "non-executable .text accepted");
    must!(refused(&patched(LTCALC, |v| put32(v, 12, 0x10000)), LdError::Malformed("entry point is outside .text")), "entry point outside .text accepted");
    must!(refused(&patched(LTCALC, |v| put32(v, reloc.0, text.1 as u32 - 4)), LdError::Malformed("relocation is outside .text")), "relocation outside .text accepted");
    must!(refused(&patched(LTCALC, |v| v[reloc.0 + 4] = 2), LdError::Unsupported("unsupported relocation type")), "unknown relocation type accepted");
    must!(refused(&patched(LTCALC, |v| put32(v, idata.0 + 64, text.1 as u32 - 4)), LdError::Malformed("import slot is outside .text")), "import slot outside .text accepted");
    must!(refused(&patched(LTCALC, |v| v[idata.0] = b'n'), LdError::MissingLibrary), "missing library not detected");
    must!(refused(&patched(LTCALC, |v| v[idata.0 + 32] = b'x'), LdError::MissingExport), "missing export not detected");
    must!(refused(&patched(LTCALC, |v| v[idata.0..idata.0 + 32].fill(b'A')), LdError::Malformed("name is not NUL-terminated")), "unterminated name accepted");
    must!(refused(&patched(LTCALC, |v| put32(v, tab(1) + 12, 67)), LdError::Malformed(".idata has a bad size")), "odd .idata size accepted");
    must!(refused(LTMATH, LdError::NotExecutable), "a library was started as a program");
    must!(register_library("ltx.llf", LTCALC) == Err(LdError::NotLibrary), "a program was registered as a library");
    must!(register_library("ltmath.llf", LTMATH) == Err(LdError::Malformed("library name already registered")), "duplicate library name accepted");
    must!(matches!(parse(&[]), Err(LdError::Truncated)), "empty input accepted");

    // The failed spawns must not have left anything behind.
    crate::ps::sched::sleep_ms(60);
    if super::super::ex::selftest::balance() != bal0 {
        return Err("frames leaked");
    }
    Ok(SelfTest { exit_code, refused: n })
}
