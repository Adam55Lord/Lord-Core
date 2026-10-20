/* SPDX-License-Identifier: GPL-2.0-only
 * Copyright (C) 2026 Adam55Lord */

/* x86_64 kernel, higher half. The layout is bootloader-neutral.
 * The linker's output is only an intermediate file: tools/lxf/mkkernel.py packs its
 * loadable segments into the LXF image (dist/lord-core.lxf), the only format Lord Boot loads. */
OUTPUT_FORMAT(elf64-x86-64)
/* Every boot backend (src/boot/<name>/) provides `_start`. */
ENTRY(_start)

/* Separate segments so the bootloader can apply correct W^X page permissions. */
PHDRS
{
    text   PT_LOAD;
    rodata PT_LOAD;
    data   PT_LOAD;
}

SECTIONS
{
    /* Topmost 2 GiB (required by the kernel code model). A loader
       that wants another base only needs this one line changed. */
    . = 0xffffffff80000000;

    .text : {
        *(.text .text.*)
    } :text

    . = ALIGN(CONSTANT(MAXPAGESIZE));

    .rodata : {
        *(.rodata .rodata.*)
    } :rodata

    . = ALIGN(CONSTANT(MAXPAGESIZE));

    .data : {
        *(.data .data.*)

    } :data

    /* .bss must be last in :data so the file isn't padded with zeros. */
    .bss : {
        *(.bss .bss.*)
        *(COMMON)
    } :data

    /DISCARD/ : {
        *(.eh_frame*)
        *(.note .note.*)
        *(.comment)
    }
}
