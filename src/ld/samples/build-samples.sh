#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Adam55Lord
# Rebuilds the sample LXF/LLF files embedded by the kernel's loader self-test (needs nasm + python3).
set -euo pipefail
cd "$(dirname "$0")"
# flat2lord.py lives in tools/lxf/; point FLAT2LORD at it if you run this from elsewhere.
T=${FLAT2LORD:-../../../tools/lxf/flat2lord.py}
[ -f "$T" ] || { echo "flat2lord.py not found: set FLAT2LORD=/path/to/flat2lord.py" >&2; exit 1; }
nasm -f bin ltmath.asm -o ltmath.bin
nasm -f bin ltcalc.asm -o ltcalc.bin
python3 $T ltmath.bin ltmath.llf --export add:0:4 --export square:0x10:8 --libmain 0x20
python3 $T ltcalc.bin ltcalc.lxf --entry 0 --reloc 0x100 \
    --import ltmath.llf:add:0x108 --import ltmath.llf:square:0x110
rm -f ltmath.bin ltcalc.bin
ls -l ltmath.llf ltcalc.lxf
