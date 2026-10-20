#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Adam55Lord
# Randomized model test for the physical memory manager, run on the HOST.
#   tests/run-host-tests.sh            200 seeds x 3000 operations
#   tests/run-host-tests.sh 1000 5000  1000 seeds x 5000 operations
# Needs only a stock `rustc` (no kernel toolchain, no QEMU).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SEEDS="${1:-200}"
OPS="${2:-3000}"
OUT="$(mktemp -d)/pmm_model"
rustc --edition 2021 -O -o "$OUT" pmm_model.rs
fail=0
for seed in $(seq 1 "$SEEDS"); do
    "$OUT" "$seed" "$OPS" > /tmp/pmm_model.out 2>&1 || { cat /tmp/pmm_model.out; fail=$((fail+1)); }
done
if [ "$fail" -eq 0 ]; then
    echo "pmm model test: all $SEEDS seeds passed ($OPS operations each)"
else
    echo "pmm model test: $fail of $SEEDS seeds FAILED" >&2
    exit 1
fi
