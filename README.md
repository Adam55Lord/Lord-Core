# Lord-Core

> An experimental, object-oriented, memory-safe operating system kernel inspired by Windows architecture—built primarily for fun, architectural exploration, and AI-assisted development.

---

## Project Objectives

* **Hybrid Kernel Architecture (Windows-Style):** Combining the performance benefits of a monolithic kernel (fast system calls, integrated subsystems) with the modularity and clean abstractions of a microkernel. Core OS services, device drivers, and executive components are structured around robust, protected subsystems inspired by the Windows NT architecture design.
* **Object-Oriented Architecture & Native Object Manager:** Moving away from the traditional UNIX philosophy of *"everything is a file"* toward a native object-oriented model. Resources, threads, synchronization primitives, and devices are managed uniformly via a centralized Object Manager with strict security descriptors and handle-based access control.
* **Memory Safety:** Leveraging modern safety guarantees to prevent common low-level bugs.
* **AI-Assisted Experimentation:** Exploring how far AI can help design and build a complex operating system kernel (spoiler: I'm not a pro programmer, and I'm learning along the way!).
* **Open Source Kernel:** I want this kernel be an open source for you all.

---

## What is ready and what is missing?

In very shortcut, kernel is booting and has working HAL, Memory, Object Manager and Process Manager but still is missing many stuff like FPU/SSE/AVX support or Multi-CPU scheduling.
More info is in ROADMAP.

## How to build it?

You have to have rust compiler compatible with rust 2024 and Linux.
in linux using ./build.sh you will build this project into folder dist/.

## How to run it?

You will need to build one more project Lord Boot which is custom bootloader because kernel is compiled not in standard format so limine or other bootloader won't know it.
Test tools are in other repo too.
