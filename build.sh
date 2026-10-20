#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Adam55Lord
# Lord Core (kernel): builds the kernel and writes ./dist/lord-core.lxf, the kernel as an LXF image
# (the only format Lord Boot loads).
#   ./build.sh                    release build (default)
#   ./build.sh --debug            debug build (opt-level 1, slower but easier to debug)
#   BOOT_BACKEND=<n> ./build.sh   another boot backend: needs the cargo feature `boot-<n>`
# Needs Rust and python3 (tools/lxf/mkkernel.py packs the linked kernel into LXF).
# To boot it, use the lord-docs-tests package: put dist/lord-core.lxf into its kernel/ folder
# (or unpack lord-core next to it), together with Lord Boot's BOOTX64.EFI.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PROFILE=release; CARGO_PROFILE=(--release)
case "${1:-}" in
    "") ;;
    --debug) PROFILE=debug; CARGO_PROFILE=() ;;
    -h|--help) sed -n '4,10p' "$0"; exit 0 ;;
    *) echo "error: unknown option '$1' (try --help)" >&2; exit 1 ;;
esac
BACKEND="${BOOT_BACKEND:-lord-boot}"

command -v python3 >/dev/null 2>&1 || { echo "error: python3 not found (needed for tools/lxf/mkkernel.py)" >&2; exit 1; }
command -v cargo >/dev/null 2>&1 || { echo "error: cargo not found. Install Rust: https://rustup.rs" >&2; exit 1; }

# Needs Rust 1.85+ (edition 2024). rustup reads rust-toolchain.toml and also
# installs the x86_64-unknown-none target automatically on first build.
echo "==> Building Lord Core ($PROFILE, x86_64-unknown-none, boot backend: $BACKEND)..."
SYSROOT="$(rustc --print sysroot)"
if [ -d "$SYSROOT/lib/rustlib/x86_64-unknown-none" ]; then
    cargo build "${CARGO_PROFILE[@]}" --no-default-features --features "boot-$BACKEND"
else
    # Distro rustc without the bare-metal target: build core/alloc from rust-src (needs rust-src + lld).
    RUSTC_BOOTSTRAP=1 cargo build "${CARGO_PROFILE[@]}" --no-default-features --features "boot-$BACKEND" \
        -Zbuild-std=core,alloc,compiler_builtins -Zbuild-std-features=compiler-builtins-mem
fi

# The linker's output stays in target/ (Cargo's own artifact, useful for gdb). Only the LXF image is installed.
LINKED="target/x86_64-unknown-none/$PROFILE/lord-core"
[ -f "$LINKED" ] || { echo "error: $LINKED missing after build" >&2; exit 1; }
mkdir -p dist
rm -f dist/lord-core
echo "==> Packing the kernel as an LXF image..."
python3 tools/lxf/mkkernel.py "$LINKED" dist/lord-core.lxf
echo "==> Done: $PWD/dist/lord-core.lxf"
echo "    Next: copy it to lord-docs-tests/kernel/, then run ./assemble.sh and ./run.sh there."
