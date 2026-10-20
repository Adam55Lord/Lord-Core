// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! The **LXF loader**: starts user programs stored in the Lord eXecutable Format
//! (`LXF\0` executables, `LLF\0` shared libraries; format and tools in `tools/lxf/`,
//! `lordfmt.py` is the specification) instead of an ELF loader. It is the kernel-side twin of
//! `run_lord.py`, with the same rules, and does what a loader must:
//!
//! 1. **Parse and validate** the file. Everything is bounds- and overflow-checked and the
//!    limits below apply: a malformed or hostile image is refused with an [`LdError`], never
//!    trusted. Kernel images (`FLAG_KERNEL`) are refused: they belong to the bootloader.
//! 2. **Map** the `.text` of the program at [`USER_CODE_BASE`] and of every library it needs
//!    right behind it, each read + execute (W^X: the loader writes the pages first, then
//!    protects them; a section flagged writable is refused). Libraries are found by name in the
//!    kernel's **library registry** ([`register_library`]; there is no file system yet) and
//!    loaded recursively, once per process, like `run_lord.py`'s module cache.
//! 3. **Relocate**: every `.reloc` entry (`ABS64`) gets the module's real base added.
//! 4. **Resolve imports**: every `.idata` entry's IAT slot is filled with the address of the
//!    named export of the named library (`.edata`).
//! 5. **Run it** through a small generated *start stub* (the page after the last module, never
//!    writable): `LibraryMain(1)` of every library that has one (dependencies first), then the
//!    program's entry point, called like a function, then `LibraryMain(0)` in reverse order and
//!    finally `exit(rax)`. So a program may simply `ret` (as the examples in `tools/lxf/` do),
//!    its return value is the process **exit code**; calling `exit` itself works too.
//!
//! Limits: 16 modules, 256 pages (1 MiB) of `.text` in all, 64 pages per module, 256 imports,
//! 65536 relocations, 4096 exports per module. ABI inside the process: the same system calls
//! as everywhere (`syscall`; chapter 21); `write` is number 2, not Linux's 1.
//!
//! User entry points: system call `lxf_spawn` (33, `ex/`), or [`spawn`] from kernel code.

pub mod selftest;

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::hal::sync::IrqLock;
use crate::mm::vmm::Prot;
use crate::ob::Obj;
use crate::ps::{self, Mapping, Process, PsError, USER_CODE_BASE};

// --- format (mirrors tools/lxf/lordfmt.py) ---------------------------------

const MAGIC_LXF: [u8; 4] = *b"LXF\0";
const MAGIC_LLF: [u8; 4] = *b"LLF\0";
const TYPE_EXECUTABLE: u16 = 1;
const TYPE_LIBRARY: u16 = 2;
const ARCH_X86_64: u16 = 1;
const FLAG_HAS_LIBMAIN: u32 = 0x1;
const FLAG_KERNEL: u32 = 0x2;
const SEC_FLAG_EXEC: u32 = 0x1;
const SEC_FLAG_WRITE: u32 = 0x2;
const RELOC_ABS64: u8 = 1;

const FILE_HEADER: usize = 20;
const SECTION_HEADER: usize = 20;
const EXPORT_SIZE: usize = 40;
const IMPORT_SIZE: usize = 68;
const RELOC_SIZE: usize = 8;

// --- limits ----------------------------------------------------------------

pub const MAX_SECTIONS: usize = 16;
pub const MAX_MODULES: usize = 16;
pub const MAX_MODULE_PAGES: usize = 64;
pub const MAX_TOTAL_PAGES: u64 = 256;
pub const MAX_IMAGE_BYTES: usize = 1 << 20;
const MAX_IMPORTS: usize = 256;
const MAX_RELOCS: usize = 65536;
const MAX_EXPORTS: usize = 4096;
const MAX_NAME: usize = 31;

/// Why an image was refused or could not be started.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LdError {
    /// Shorter than its own header says.
    Truncated,
    /// Not an `LXF\0` / `LLF\0` file, or magic and type disagree.
    BadMagic,
    /// A kernel image (`FLAG_KERNEL`): only the bootloader loads those.
    KernelImage,
    /// A library where a program is needed.
    NotExecutable,
    /// An executable where a library is needed.
    NotLibrary,
    /// A feature this loader does not have (other architecture, version, relocation type).
    Unsupported(&'static str),
    /// Structurally wrong: the reason says what.
    Malformed(&'static str),
    /// An imported library is not in the registry.
    MissingLibrary,
    /// A library does not export a name that is imported from it.
    MissingExport,
    /// Over one of the size limits.
    TooBig,
    /// More than [`MAX_MODULES`] modules would be loaded.
    TooManyModules,
    /// The Process Manager refused (out of memory, ...).
    Ps(PsError),
}

impl LdError {
    pub fn reason(&self) -> &'static str {
        match self {
            LdError::Truncated => "file is truncated",
            LdError::BadMagic => "not an LXF/LLF file",
            LdError::KernelImage => "this is a kernel image, not a user program",
            LdError::NotExecutable => "not an executable",
            LdError::NotLibrary => "not a library",
            LdError::Unsupported(w) | LdError::Malformed(w) => w,
            LdError::MissingLibrary => "an imported library is not registered",
            LdError::MissingExport => "a library does not export an imported name",
            LdError::TooBig => "image too big",
            LdError::TooManyModules => "too many modules",
            LdError::Ps(_) => "process creation failed",
        }
    }
}

// --- parsing ---------------------------------------------------------------

struct Import {
    lib: String,
    func: String,
    slot: usize,
}

struct Export {
    name: String,
    offset: usize,
}

/// A validated LXF/LLF file: only what loading needs.
pub struct Module {
    is_library: bool,
    has_libmain: bool,
    entry: usize,
    text: Vec<u8>,
    relocs: Vec<usize>,
    imports: Vec<Import>,
    exports: Vec<Export>,
}

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// A fixed 32-byte, NUL-padded name field.
fn name_at(b: &[u8], o: usize) -> Result<String, LdError> {
    let raw = &b[o..o + 32];
    let n = raw.iter().position(|&c| c == 0).ok_or(LdError::Malformed("name is not NUL-terminated"))?;
    if n == 0 || !raw[..n].iter().all(|c| c.is_ascii_graphic()) {
        return Err(LdError::Malformed("name is empty or not printable ASCII"));
    }
    Ok(String::from_utf8_lossy(&raw[..n]).into_owned())
}

/// Parses and fully validates an LXF/LLF image.
pub fn parse(image: &[u8]) -> Result<Module, LdError> {
    if image.len() < FILE_HEADER {
        return Err(LdError::Truncated);
    }
    if image.len() > MAX_IMAGE_BYTES {
        return Err(LdError::TooBig);
    }
    let magic = [image[0], image[1], image[2], image[3]];
    let (version, arch, ftype, nsec) = (u16_at(image, 4), u16_at(image, 6), u16_at(image, 8), u16_at(image, 10) as usize);
    let (entry, flags) = (u32_at(image, 12) as usize, u32_at(image, 16));
    let is_library = match (magic, ftype) {
        (MAGIC_LXF, TYPE_EXECUTABLE) => false,
        (MAGIC_LLF, TYPE_LIBRARY) => true,
        _ => return Err(LdError::BadMagic),
    };
    if version != 1 {
        return Err(LdError::Unsupported("unsupported LXF version"));
    }
    if arch != ARCH_X86_64 {
        return Err(LdError::Unsupported("not an x86-64 image"));
    }
    if !is_library && flags & FLAG_KERNEL != 0 {
        return Err(LdError::KernelImage);
    }
    if nsec > MAX_SECTIONS {
        return Err(LdError::Malformed("too many sections"));
    }
    if FILE_HEADER + nsec * SECTION_HEADER > image.len() {
        return Err(LdError::Truncated);
    }

    let (mut text, mut idata, mut reloc, mut edata): (Option<(&[u8], u32)>, Option<&[u8]>, Option<&[u8]>, Option<&[u8]>) =
        (None, None, None, None);
    for i in 0..nsec {
        let h = FILE_HEADER + i * SECTION_HEADER;
        let raw = &image[h..h + 8];
        let name = &raw[..raw.iter().position(|&c| c == 0).unwrap_or(8)];
        let (off, size, sflags) = (u32_at(image, h + 8) as usize, u32_at(image, h + 12) as usize, u32_at(image, h + 16));
        let end = off.checked_add(size).ok_or(LdError::Malformed("section overflows"))?;
        if end > image.len() {
            return Err(LdError::Truncated);
        }
        let data = &image[off..end];
        let slot_taken = match name {
            b".text" => text.replace((data, sflags)).is_some(),
            b".idata" => idata.replace(data).is_some(),
            b".reloc" => reloc.replace(data).is_some(),
            b".edata" => edata.replace(data).is_some(),
            _ => false, // unknown sections (including .kseg) are ignored, as the format promises
        };
        if slot_taken {
            return Err(LdError::Malformed("duplicate section"));
        }
    }

    let (text, text_flags) = text.ok_or(LdError::Malformed("no .text section"))?;
    if text.is_empty() {
        return Err(LdError::Malformed(".text is empty"));
    }
    if text.len() > MAX_MODULE_PAGES * 4096 {
        return Err(LdError::TooBig);
    }
    if text_flags & SEC_FLAG_WRITE != 0 {
        return Err(LdError::Malformed(".text is writable (W^X)"));
    }
    if text_flags & SEC_FLAG_EXEC == 0 {
        return Err(LdError::Malformed(".text is not executable"));
    }
    let has_libmain = is_library && flags & FLAG_HAS_LIBMAIN != 0;
    if (!is_library || has_libmain) && entry >= text.len() {
        return Err(LdError::Malformed("entry point is outside .text"));
    }

    let mut imports = Vec::new();
    if let Some(d) = idata {
        if d.len() % IMPORT_SIZE != 0 || d.len() / IMPORT_SIZE > MAX_IMPORTS {
            return Err(LdError::Malformed(".idata has a bad size"));
        }
        for i in 0..d.len() / IMPORT_SIZE {
            let o = i * IMPORT_SIZE;
            let slot = u32_at(d, o + 64) as usize;
            if slot.checked_add(8).is_none_or(|e| e > text.len()) {
                return Err(LdError::Malformed("import slot is outside .text"));
            }
            imports.push(Import { lib: name_at(d, o)?, func: name_at(d, o + 32)?, slot });
        }
    }
    let mut relocs = Vec::new();
    if let Some(d) = reloc {
        if d.len() % RELOC_SIZE != 0 || d.len() / RELOC_SIZE > MAX_RELOCS {
            return Err(LdError::Malformed(".reloc has a bad size"));
        }
        for i in 0..d.len() / RELOC_SIZE {
            let o = i * RELOC_SIZE;
            if d[o + 4] != RELOC_ABS64 {
                return Err(LdError::Unsupported("unsupported relocation type"));
            }
            let at = u32_at(d, o) as usize;
            if at.checked_add(8).is_none_or(|e| e > text.len()) {
                return Err(LdError::Malformed("relocation is outside .text"));
            }
            relocs.push(at);
        }
    }
    let mut exports = Vec::new();
    if let Some(d) = edata.filter(|_| is_library) {
        if d.len() % EXPORT_SIZE != 0 || d.len() / EXPORT_SIZE > MAX_EXPORTS {
            return Err(LdError::Malformed(".edata has a bad size"));
        }
        for i in 0..d.len() / EXPORT_SIZE {
            let o = i * EXPORT_SIZE;
            let offset = u32_at(d, o + 32) as usize;
            if offset >= text.len() {
                return Err(LdError::Malformed("export is outside .text"));
            }
            exports.push(Export { name: name_at(d, o)?, offset });
        }
    }
    Ok(Module { is_library, has_libmain, entry, text: text.to_vec(), relocs, imports, exports })
}

// --- library registry ------------------------------------------------------

static LIBRARIES: IrqLock<Vec<(String, Arc<Module>)>> = IrqLock::new(Vec::new());

/// Makes the `LLF` image `image` importable under `name` (what `.idata` entries name, e.g.
/// `"ltmath.llf"`). Kernel-side only: a user program cannot replace a library others use.
pub fn register_library(name: &str, image: &[u8]) -> Result<(), LdError> {
    if name.is_empty() || name.len() > MAX_NAME || !name.bytes().all(|c| c.is_ascii_graphic()) {
        return Err(LdError::Malformed("bad library name"));
    }
    let module = parse(image)?;
    if !module.is_library {
        return Err(LdError::NotLibrary);
    }
    let mut libs = LIBRARIES.lock();
    if libs.iter().any(|(n, _)| n == name) {
        return Err(LdError::Malformed("library name already registered"));
    }
    libs.push((String::from(name), Arc::new(module)));
    Ok(())
}

/// Number of registered libraries.
pub fn library_count() -> usize {
    LIBRARIES.lock().len()
}

fn find_library(name: &str) -> Option<Arc<Module>> {
    LIBRARIES.lock().iter().find(|(n, _)| n == name).map(|(_, m)| m.clone())
}

/// Registers the libraries the kernel ships with.
pub fn init() -> Result<(), LdError> {
    register_library("ltmath.llf", include_bytes!("samples/ltmath.llf"))
}

// --- loading ---------------------------------------------------------------

struct Placed {
    name: String, // "" for the executable
    module: Arc<Module>,
    base: u64,
}

struct Plan {
    mods: Vec<Placed>,
    /// Indices of libraries in load order: dependencies before their importers.
    order: Vec<usize>,
    next_va: u64,
}

impl Plan {
    fn place(&mut self, name: &str, module: Arc<Module>) -> Result<usize, LdError> {
        if self.mods.len() >= MAX_MODULES {
            return Err(LdError::TooManyModules);
        }
        let pages = module.text.len().div_ceil(4096) as u64;
        let base = self.next_va;
        self.next_va += pages * 4096;
        // +1: the start stub needs a page as well.
        if (self.next_va - USER_CODE_BASE) / 4096 + 1 > MAX_TOTAL_PAGES {
            return Err(LdError::TooBig);
        }
        self.mods.push(Placed { name: String::from(name), module, base });
        Ok(self.mods.len() - 1)
    }

    /// Places every library `idx` imports, recursively (a cache by name, as in `run_lord.py`:
    /// each library once, cycles are fine).
    fn load_imports(&mut self, idx: usize) -> Result<(), LdError> {
        let module = self.mods[idx].module.clone();
        for imp in &module.imports {
            if self.mods.iter().any(|m| m.name == imp.lib) {
                continue;
            }
            let lib = find_library(&imp.lib).ok_or(LdError::MissingLibrary)?;
            let i = self.place(&imp.lib, lib)?;
            self.load_imports(i)?;
            self.order.push(i);
        }
        Ok(())
    }
}

fn call_abs(code: &mut Vec<u8>, target: u64) {
    code.extend_from_slice(&[0x48, 0xB8]); // mov rax, imm64
    code.extend_from_slice(&target.to_le_bytes());
    code.extend_from_slice(&[0xFF, 0xD0]); // call rax
}

/// The code that runs first in ring 3 (see the module docs). Entered with a 16-byte-aligned
/// stack, so every `call` hands its callee a System V-conformant one.
fn start_stub(plan: &Plan) -> Vec<u8> {
    let libmains: Vec<u64> = plan.order.iter().map(|&i| &plan.mods[i]).filter(|m| m.module.has_libmain).map(|m| m.base + m.module.entry as u64).collect();
    let exe = &plan.mods[0];
    let mut c = Vec::new();
    for &a in &libmains {
        c.extend_from_slice(&[0xBF, 1, 0, 0, 0]); // mov edi, 1   (reason: load)
        call_abs(&mut c, a);
    }
    call_abs(&mut c, exe.base + exe.module.entry as u64);
    c.extend_from_slice(&[0x48, 0x89, 0xC3]); // mov rbx, rax  (the program's result; rbx is callee-saved)
    for &a in libmains.iter().rev() {
        c.extend_from_slice(&[0x31, 0xFF]); // xor edi, edi (reason: unload)
        call_abs(&mut c, a);
    }
    c.extend_from_slice(&[0x89, 0xDF, 0x31, 0xC0, 0x0F, 0x05, 0x0F, 0x0B]); // mov edi, ebx; xor eax, eax; syscall (exit); ud2
    c
}

/// Relocates `m` for the base `base` and fills its IAT slots.
fn link(plan: &Plan, m: &Placed) -> Result<Vec<u8>, LdError> {
    let mut text = m.module.text.clone();
    for &at in &m.module.relocs {
        let v = u64::from_le_bytes(text[at..at + 8].try_into().unwrap()).wrapping_add(m.base);
        text[at..at + 8].copy_from_slice(&v.to_le_bytes());
    }
    for imp in &m.module.imports {
        let dep = plan.mods.iter().find(|d| d.name == imp.lib).ok_or(LdError::MissingLibrary)?;
        let ex = dep.module.exports.iter().find(|e| e.name == imp.func).ok_or(LdError::MissingExport)?;
        text[imp.slot..imp.slot + 8].copy_from_slice(&(dep.base + ex.offset as u64).to_le_bytes());
    }
    Ok(text)
}

/// Loads the LXF executable `image`, with its libraries, as a new process (running at priority
/// `prio`) and starts it. The process ends with the program's return value as exit code.
pub fn spawn(name: &str, image: &[u8], prio: u8) -> Result<Obj<Process>, LdError> {
    let exe = parse(image)?;
    if exe.is_library {
        return Err(LdError::NotExecutable);
    }
    let mut plan = Plan { mods: Vec::new(), order: Vec::new(), next_va: USER_CODE_BASE };
    plan.place("", Arc::new(exe))?;
    plan.load_imports(0)?;

    let mut texts: Vec<Vec<u8>> = Vec::new();
    for m in &plan.mods {
        texts.push(link(&plan, m)?);
    }
    let stub = start_stub(&plan);
    let stub_va = plan.next_va;
    let mut maps: Vec<Mapping> = plan.mods.iter().zip(&texts).map(|(m, t)| Mapping { va: m.base, bytes: t, prot: Prot::READ_EXEC.user() }).collect();
    maps.push(Mapping { va: stub_va, bytes: &stub, prot: Prot::READ_EXEC.user() });
    ps::spawn_user_image(name, &maps, stub_va, prio).map_err(LdError::Ps)
}
