#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Adam55Lord
"""
mkkernel.py -- pack the freshly linked Lord Core kernel into an LXF kernel image.

    python3 mkkernel.py target/x86_64-unknown-none/release/lord-core dist/lord-core.lxf

This is the last step of lord-core's ./build.sh, not a general converter: the Rust linker can only
write its result in its own executable format, and this tool reads exactly that one file
(static, x86-64, little endian) and nothing else.  Its output is the only kernel format
Lord Boot loads; the linker output is never installed, shipped or booted.

flat2lord.py wraps a *flat* binary that can run at any address.  A kernel can't:
it is linked for a fixed higher-half address.  So this tool keeps the linked
image's loadable segments apart, as the "kernel image" extension of the
format describes (see lordfmt.py):

  .text   the segments' file bytes, each starting on a 4 KiB boundary
  .kseg   per segment: virtual address, offset in .text, file size, memory size
          (memsz > filesz is the .bss, zeroed by the loader), R/W/X permissions
  header  FLAG_KERNEL, entry_offset = the entry point's offset inside .text

Only what a loader needs is kept: no headers, section table or symbols.
Needs nothing but Python 3 (the linker output is parsed here, no objcopy / pyelftools).
"""
import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lordfmt import (  # noqa: E402
    LordFile, Section, Segment, validate_segments,
    TYPE_EXECUTABLE, ARCH_X86_64, FLAG_KERNEL,
    SEC_FLAG_EXEC, SEC_FLAG_WRITE, SEC_FLAG_KSEG,
    KSEG_FMT, KSEG_R, KSEG_W, KSEG_X, KERNEL_PAGE,
)

ET_EXEC = 2
EM_X86_64 = 62
PT_LOAD = 1
PF_X, PF_W, PF_R = 1, 2, 4


def align_up(n, a=KERNEL_PAGE):
    return (n + a - 1) & ~(a - 1)


def read_linked(path):
    """Return (entry_vaddr, [(vaddr, bytes, memsz, kseg_flags), ...])."""
    with open(path, "rb") as f:
        img = f.read()
    if img[:4] != b"\x7fELF":
        sys.exit(f"{path}: not a statically linked kernel (linker output expected)")
    if img[4] != 2 or img[5] != 1:
        sys.exit(f"{path}: need a 64-bit little-endian kernel")
    (e_type, e_machine, _ver, e_entry, e_phoff, _shoff, _flags, _ehsize,
     e_phentsize, e_phnum) = struct.unpack_from("<HHIQQQIHHH", img, 16)
    if e_machine != EM_X86_64:
        sys.exit(f"{path}: not an x86-64 kernel (machine {e_machine})")
    if e_type != ET_EXEC:
        sys.exit(f"{path}: not a static executable (type {e_type}). A kernel linked as a PIE cannot be "
                 "packed; check .cargo/config.toml (relocation-model=static) and build.rs (--no-pie).")

    segs = []
    for i in range(e_phnum):
        (p_type, p_flags, p_offset, p_vaddr, _paddr, p_filesz, p_memsz, _align) = \
            struct.unpack_from("<IIQQQQQQ", img, e_phoff + i * e_phentsize)
        if p_type != PT_LOAD or p_memsz == 0:
            continue
        if p_offset + p_filesz > len(img):
            sys.exit(f"{path}: segment {i} runs past the end of the file")
        flags = ((KSEG_R if p_flags & PF_R else 0)
                 | (KSEG_W if p_flags & PF_W else 0)
                 | (KSEG_X if p_flags & PF_X else 0))
        segs.append((p_vaddr, img[p_offset:p_offset + p_filesz], p_memsz, flags))
    if not segs:
        sys.exit(f"{path}: no PT_LOAD segments")
    return e_entry, segs


def build(entry_vaddr, segs):
    """Lay the segments out back to back and build the LordFile."""
    segs = sorted(segs, key=lambda s: s[0])
    text = bytearray()
    table = []
    entry_offset = None
    for vaddr, data, memsz, flags in segs:
        # A segment that does not start on a page boundary (vaddr % 4096 != 0)
        # cannot be mapped page-wise without moving its bytes: refuse instead of guessing.
        if vaddr % KERNEL_PAGE:
            sys.exit(f"segment at {vaddr:#x} is not 4 KiB aligned; link with page-aligned segments "
                     "(see linker.ld: ALIGN(CONSTANT(MAXPAGESIZE)))")
        offset = align_up(len(text))
        text.extend(b"\0" * (offset - len(text)))
        text.extend(data)
        table.append(Segment(vaddr, offset, len(data), memsz, flags))
        if (flags & KSEG_X) and vaddr <= entry_vaddr < vaddr + len(data):
            entry_offset = offset + (entry_vaddr - vaddr)
    if entry_offset is None:
        sys.exit(f"entry point {entry_vaddr:#x} is not inside an executable segment")
    text.extend(b"\0" * (align_up(len(text)) - len(text)))   # whole pages

    try:
        validate_segments(table, len(text), entry_offset)
    except ValueError as e:
        sys.exit(f"invalid kernel layout: {e}")

    kseg = b"".join(struct.pack(KSEG_FMT, g.vaddr, g.offset, g.filesz, g.memsz, g.flags) for g in table)
    sections = [
        Section(b".text", bytes(text), flags=SEC_FLAG_EXEC | SEC_FLAG_WRITE),
        Section(b".kseg", kseg, flags=SEC_FLAG_KSEG),
    ]
    return LordFile(TYPE_EXECUTABLE, ARCH_X86_64, entry_offset, sections, flags=FLAG_KERNEL, segments=table)


def main():
    ap = argparse.ArgumentParser(description="Pack the linked Lord Core kernel into an LXF kernel image (.lxf)")
    ap.add_argument("input", help="the statically linked kernel (Cargo's output in target/)")
    ap.add_argument("output", help="output .lxf file")
    args = ap.parse_args()

    entry, segs = read_linked(args.input)
    lf = build(entry, segs)
    blob = lf.to_bytes()

    # Prove the file we wrote round-trips through the same parser the other tools use.
    back = LordFile.from_bytes(blob)
    assert back.is_kernel and back.entry_offset == lf.entry_offset
    assert [(g.vaddr, g.offset, g.filesz, g.memsz, g.flags) for g in back.segments] == \
           [(g.vaddr, g.offset, g.filesz, g.memsz, g.flags) for g in lf.segments]

    with open(args.output, "wb") as f:
        f.write(blob)
    print(f"wrote {args.output} ({len(blob)} bytes) from {args.input}")
    print(f"  entry {entry:#x}")
    print(lf.describe())


if __name__ == "__main__":
    main()
