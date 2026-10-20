# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Adam55Lord
"""
lordfmt.py -- struct definitions for the LXF (Lord eXecutable Format) and
LLF (Lord Library Format) container, as sketched in the LLF.txt / LXF.txt
diagrams.

Layout (identical skeleton for both; a flag in the file header says which
kind of file it is):

    FileHeader
    SectionHeader[num_sections]
    <section bytes, back to back, each at the offset its SectionHeader says>

Sections used:
  .text    machine code (always present)
  .kseg    (kernel images only, FLAG_KERNEL) load-address table, see below
  .edata   (library only) exported function table: name -> offset/size in .text
  .idata   imported symbols this module needs resolved at load time
  .reloc   locations in .text that must be patched once the real load
           address is known

This is a simplified cousin of the PE/COFF format used by Windows
.exe/.dll files -- one magic, one flat section table, no RVAs/paging.
If you get stuck extending it, the real thing to go skim is the PE/COFF
spec (an .exe and a .dll are literally the same file format with one
flag bit different, and PE's .reloc / import-table / DllMain are exactly
what .reloc / .idata / LibraryMain here are modeled on).
"""
import struct

MAGIC_LXF = b"LXF\0"   # executable
MAGIC_LLF = b"LLF\0"   # shared library ("module")

TYPE_EXECUTABLE = 1
TYPE_LIBRARY = 2

ARCH_X86_64 = 1

FLAG_HAS_LIBMAIN = 0x1  # for TYPE_LIBRARY: entry_offset is a valid LibraryMain() offset
FLAG_KERNEL = 0x2       # for TYPE_EXECUTABLE: an operating-system kernel image, see .kseg below

# FileHeader: magic(4s) version(H) arch(H) filetype(H) num_sections(H)
#             entry_offset(I) flags(I)
FILEHDR_FMT = "<4sHHHHII"
FILEHDR_SIZE = struct.calcsize(FILEHDR_FMT)

# SectionHeader: name(8s) file_offset(I) size(I) flags(I)
SECHDR_FMT = "<8sIII"
SECHDR_SIZE = struct.calcsize(SECHDR_FMT)

SEC_FLAG_EXEC = 0x1
SEC_FLAG_WRITE = 0x2
SEC_FLAG_EXPORTS = 0x4   # this is the .edata table
SEC_FLAG_KSEG = 0x8      # this is the .kseg table (kernel images)

# .edata entry: name(32s) offset(I) size(I)
EXPORT_FMT = "<32sII"
EXPORT_SIZE = struct.calcsize(EXPORT_FMT)

# .idata entry: library_name(32s) func_name(32s) iat_slot_offset(I)
# iat_slot_offset is an offset *into .text* of an 8-byte pointer cell
# ("IAT slot") that the loader fills in with the resolved function's
# runtime address. Importing code calls through that slot, e.g.
#   call qword [rel iat_slot]
IMPORT_FMT = "<32s32sI"
IMPORT_SIZE = struct.calcsize(IMPORT_FMT)

# .reloc entry: offset(I) type(B) + 3 bytes padding
# type 1 = ABS64: the 8 bytes at .text+offset currently hold a link-time
# address computed against a preferred base of 0; the loader adds the
# actual runtime base of .text to that value in place.
RELOC_TYPE_ABS64 = 1
RELOC_FMT = "<IB3x"
RELOC_SIZE = struct.calcsize(RELOC_FMT)


# --- kernel images (FLAG_KERNEL) --------------------------------------------
#
# A kernel is linked for a FIXED virtual address (the x86-64 "kernel" code
# model needs the top 2 GiB), so the "map .text wherever, patch ABS64 relocs"
# scheme above cannot be used.  Instead the image carries a tiny program-header
# table, .kseg, and the loader maps every segment at the virtual address the
# kernel was linked for, with the permissions listed (this keeps W^X):
#
#   .text   the segments' file bytes back to back, each starting on a 4 KiB
#           boundary (a flat blob, exactly as before - old tools can still dump it)
#   .kseg   one KSEG entry per segment:
#               vaddr(Q) offset(I) filesz(I) memsz(I) flags(I)
#           vaddr   virtual address, 4 KiB aligned, in the higher half
#           offset  where the segment's file bytes start inside .text (4 KiB aligned)
#           filesz  bytes taken from .text; memsz >= filesz, the rest (.bss) is zeroed
#           flags   KSEG_R / KSEG_W / KSEG_X (W and X together are rejected)
#   header  entry_offset = offset *inside .text* of the entry point; the loader
#           turns it into a virtual address through the segment that contains it.
#
# A kernel image has no .reloc / .idata (nothing to patch: it runs where it was
# linked).  Files without FLAG_KERNEL ignore .kseg completely, so every existing
# .lxf / .llf is unaffected by this extension.
KSEG_FMT = "<QIIII"
KSEG_SIZE = struct.calcsize(KSEG_FMT)
KSEG_R = 0x1
KSEG_W = 0x2
KSEG_X = 0x4
KERNEL_PAGE = 0x1000


class Segment:
    """One loadable kernel segment (an entry of .kseg)."""

    def __init__(self, vaddr, offset, filesz, memsz, flags):
        self.vaddr, self.offset, self.filesz, self.memsz, self.flags = \
            vaddr, offset, filesz, memsz, flags

    def perms(self):
        return (("r" if self.flags & KSEG_R else "-")
                + ("w" if self.flags & KSEG_W else "-")
                + ("x" if self.flags & KSEG_X else "-"))


def validate_segments(segments, text_size, entry_offset):
    """Raise ValueError unless the segment table is something a loader can trust.
    (The Rust bootloader performs the same checks, src/lxf.rs.)"""
    if not segments:
        raise ValueError("kernel image has no segments")
    for s in segments:
        if s.vaddr % KERNEL_PAGE or s.offset % KERNEL_PAGE:
            raise ValueError(f"segment {s.vaddr:#x}: vaddr/offset must be 4 KiB aligned")
        if s.vaddr < 0xFFFF_8000_0000_0000:
            raise ValueError(f"segment {s.vaddr:#x}: not in the higher half")
        if s.filesz > s.memsz:
            raise ValueError(f"segment {s.vaddr:#x}: filesz > memsz")
        if s.memsz == 0:
            raise ValueError(f"segment {s.vaddr:#x}: empty")
        if s.offset + s.filesz > text_size:
            raise ValueError(f"segment {s.vaddr:#x}: file bytes lie outside .text")
        if (s.flags & KSEG_W) and (s.flags & KSEG_X):
            raise ValueError(f"segment {s.vaddr:#x}: W^X violation (writable and executable)")
        if s.vaddr + s.memsz > 1 << 64:
            raise ValueError(f"segment {s.vaddr:#x}: wraps the address space")
    ordered = sorted(segments, key=lambda s: s.vaddr)
    for a, b in zip(ordered, ordered[1:]):
        if a.vaddr + a.memsz > b.vaddr:
            raise ValueError(f"segments at {a.vaddr:#x} and {b.vaddr:#x} overlap")
    by_off = sorted((s for s in segments if s.filesz), key=lambda s: s.offset)
    for a, b in zip(by_off, by_off[1:]):
        if a.offset + a.filesz > b.offset:
            raise ValueError("segments overlap inside .text")
    for s in segments:
        if (s.flags & KSEG_X) and s.offset <= entry_offset < s.offset + s.filesz:
            return
    raise ValueError("entry point is not inside an executable segment")


class Section:
    def __init__(self, name, data, flags=0):
        if len(name) > 8:
            raise ValueError(f"section name {name!r} too long (max 8 bytes)")
        self.name = name
        self.data = bytearray(data)
        self.flags = flags


class LordFile:
    """In-memory model of a parsed .lxf / .llf file."""

    def __init__(self, filetype, arch, entry_offset, sections,
                 exports=None, imports=None, relocs=None, flags=0, segments=None):
        self.filetype = filetype
        self.arch = arch
        self.entry_offset = entry_offset   # _start offset (exe) or LibraryMain offset (lib, if FLAG_HAS_LIBMAIN)
        self.sections = sections            # list[Section]
        self.exports = exports or {}        # name -> (section_name, offset_in_section, size)
        self.imports = imports or []        # list[(lib_name, func_name, iat_slot_offset)]
        self.relocs = relocs or []          # list[(offset, type)]
        self.flags = flags
        self.segments = segments or []      # list[Segment], FLAG_KERNEL images only

    @property
    def is_kernel(self):
        return self.filetype == TYPE_EXECUTABLE and bool(self.flags & FLAG_KERNEL)

    @property
    def is_library(self):
        return self.filetype == TYPE_LIBRARY

    @property
    def has_libmain(self):
        return self.is_library and bool(self.flags & FLAG_HAS_LIBMAIN)

    def section(self, name):
        for s in self.sections:
            if s.name == name:
                return s
        return None

    def to_bytes(self):
        magic = MAGIC_LLF if self.is_library else MAGIC_LXF
        header = struct.pack(
            FILEHDR_FMT, magic, 1, self.arch, self.filetype,
            len(self.sections), self.entry_offset, self.flags,
        )
        table_size = SECHDR_SIZE * len(self.sections)
        cursor = FILEHDR_SIZE + table_size
        sec_headers = b""
        sec_blobs = b""
        for s in self.sections:
            off = cursor
            sec_headers += struct.pack(SECHDR_FMT, s.name.ljust(8, b"\0"), off, len(s.data), s.flags)
            sec_blobs += bytes(s.data)
            cursor += len(s.data)
        return header + sec_headers + sec_blobs

    @classmethod
    def from_bytes(cls, buf):
        magic, version, arch, filetype, num_sections, entry_offset, flags = \
            struct.unpack_from(FILEHDR_FMT, buf, 0)
        if magic not in (MAGIC_LXF, MAGIC_LLF):
            raise ValueError(f"bad magic {magic!r}, expected LXF\\0 or LLF\\0")
        if (magic == MAGIC_LXF) != (filetype == TYPE_EXECUTABLE):
            raise ValueError("magic / filetype flag mismatch")

        sections = []
        off = FILEHDR_SIZE
        for _ in range(num_sections):
            name, file_offset, size, sflags = struct.unpack_from(SECHDR_FMT, buf, off)
            name = name.rstrip(b"\0")
            data = buf[file_offset:file_offset + size]
            sections.append(Section(name, data, sflags))
            off += SECHDR_SIZE

        f = cls(filetype, arch, entry_offset, sections, flags=flags)

        edata = f.section(b".edata")
        if edata is not None and edata.data:
            n = len(edata.data) // EXPORT_SIZE
            for i in range(n):
                name, eoff, esize = struct.unpack_from(EXPORT_FMT, bytes(edata.data), i * EXPORT_SIZE)
                name = name.rstrip(b"\0").decode()
                f.exports[name] = (b".text", eoff, esize)

        idata = f.section(b".idata")
        if idata is not None and idata.data:
            n = len(idata.data) // IMPORT_SIZE
            for i in range(n):
                lib, func, slot = struct.unpack_from(IMPORT_FMT, bytes(idata.data), i * IMPORT_SIZE)
                f.imports.append((lib.rstrip(b"\0").decode(), func.rstrip(b"\0").decode(), slot))

        reloc = f.section(b".reloc")
        if reloc is not None and reloc.data:
            n = len(reloc.data) // RELOC_SIZE
            for i in range(n):
                roff, rtype = struct.unpack_from(RELOC_FMT, bytes(reloc.data), i * RELOC_SIZE)
                f.relocs.append((roff, rtype))

        kseg = f.section(b".kseg")
        if kseg is not None and kseg.data:
            if len(kseg.data) % KSEG_SIZE:
                raise ValueError(".kseg size is not a multiple of its entry size")
            for i in range(len(kseg.data) // KSEG_SIZE):
                f.segments.append(Segment(*struct.unpack_from(KSEG_FMT, bytes(kseg.data), i * KSEG_SIZE)))

        return f

    def describe(self):
        lines = []
        kind = ("LLF shared library" if self.is_library
                else "LXF kernel image" if self.is_kernel else "LXF executable")
        lines.append(f"{kind}, arch={self.arch}, entry_offset=0x{self.entry_offset:x}"
                      + (" (LibraryMain)" if self.has_libmain else ""))
        for s in self.sections:
            tags = []
            if s.flags & SEC_FLAG_EXEC:
                tags.append("exec")
            if s.flags & SEC_FLAG_WRITE:
                tags.append("write")
            if s.flags & SEC_FLAG_EXPORTS:
                tags.append("exports")
            if s.flags & SEC_FLAG_KSEG:
                tags.append("kseg")
            tagstr = f" [{','.join(tags)}]" if tags else ""
            lines.append(f"  {s.name.decode(errors='replace'):<8} {len(s.data):6d} bytes{tagstr}")
        if self.segments:
            lines.append("  segments (virtual address -> .text offset):")
            for g in self.segments:
                lines.append(f"    {g.vaddr:#018x} {g.perms()}  .text+{g.offset:#08x}  "
                             f"file {g.filesz:#08x}  mem {g.memsz:#08x}")
        if self.exports:
            lines.append("  exports:")
            for name, (sec, off, size) in self.exports.items():
                lines.append(f"    {name} @ {sec.decode()}+0x{off:x} ({size} bytes)")
        if self.imports:
            lines.append("  imports:")
            for lib, func, slot in self.imports:
                lines.append(f"    {func} from {lib} -> IAT slot .text+0x{slot:x}")
        if self.relocs:
            lines.append(f"  relocations: {len(self.relocs)} entr{'y' if len(self.relocs)==1 else 'ies'} in .text")
        return "\n".join(lines)
