# Third-party material

Lord Core itself is **GPL-2.0-only** (see `LICENSE`). Everything below is compatible with
GPL version 2. The kernel has **no Cargo dependencies** (`Cargo.lock` lists only `lord-core`).

| Material | Where | License | GPL-2.0 compatible |
|---|---|---|---|
| 8x8 bitmap font (`font8x8_basic`, Daniel Hepper, from IBM VGA fonts) | `src/hal/font.rs` | public domain | yes |
| Rust `core`, `alloc` and `compiler_builtins` (linked into every Rust kernel) | compiled from the Rust toolchain, not stored in this repository | MIT OR Apache-2.0 (and, for some `compiler_builtins` parts derived from LLVM compiler-rt, Apache-2.0 WITH LLVM-exception) | **see below** |

## The Rust runtime pieces and Apache-2.0

Apache-2.0 is **not** compatible with GPL-2.0-only (it is with GPL-3.0). This repository contains no
Apache-2.0 source. The only Apache-2.0-licensed code that can end up in a built kernel image is the
Rust toolchain's runtime (`core`, `alloc`, `compiler_builtins`).

* `core` and `alloc` are offered under **MIT OR Apache-2.0**: choose **MIT** and there is no conflict.
* Parts of `compiler_builtins` (the `memcpy`-style and arithmetic helpers) come with
  *Apache-2.0 WITH LLVM-exception* terms. That exception is aimed at exactly this case (object code
  linked into other programs), and the Rust project's own GPL users (e.g. the Linux kernel's Rust support)
  rely on it, but this is a legal question, not a technical one.

This file is not legal advice. If you need certainty before publishing binaries, check the licenses
shipped with your toolchain (`rustc --print sysroot`, then `lib/rustlib/src/rust/COPYRIGHT`) or ask a
lawyer. Publishing the **source** under GPL-2.0-only is not affected.

## Rules for new code

* Allowed to copy in: MIT, BSD-2/3-Clause, 0BSD, ISC, public domain/CC0, and GPL-2.0-compatible code.
  Keep the original copyright notice and add the material to the table above.
* **Not allowed:** Apache-2.0 source, GPL-3.0-only/AGPL, "non-commercial", or license-less code.
* GPL-2.0-or-later code is fine (we use it under version 2).
