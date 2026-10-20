# Lord Core: what is missing to become a fully working operating system

Written against the source of this package (about 15,800 lines of Rust, v0.11). Everything under "What exists" was
checked in the code; everything under "What is missing" was searched for and not found. Sizes are rough:
**S** = days, **M** = 1-3 weeks, **L** = 1-2 months, **XL** = several months, for one developer.

Where code goes follows the project's own placement rule (docs/12): *mechanism* in kernel mode,
*policy / APIs / personalities* in user mode.

## 1. Where the project stands

**What exists (and is tested by the boot self-tests):**

| Area | State |
|---|---|
| HAL | serial, framebuffer text console, GDT/TSS, IDT, PIC, PIT, HPET, LAPIC + LAPIC timer, ACPI table lookup, SMP bring-up, paging layer |
| Memory | physical memory manager, virtual memory manager (regions, guard pages, W^X, per-process `AddressSpace`), kernel heap (`Box`, `Vec`, ...) |
| Object Manager | namespace (`\`), handles with access masks, reference counting; types Directory, SymbolicLink, Event, Process, Thread, Port, Section |
| Process Manager | processes and threads, preemptive 32-priority scheduler with boost/decay, sleep/wait, ring 3 via `int 0x80` (exit, yield, write, getpid, sleep) |
| IPC | named ports, datagrams, request/reply, handle passing, shared-memory sections, 11 IPC syscalls |
| Boot | Lord Boot hands over memory map, framebuffer, ACPI RSDP, CPU list |

**What it is not yet:** a system anyone can *use*. It boots, runs four hard-coded assembly test programs
(`ps/user.rs`), prints to the screen and the serial port, and then idles. There is no keyboard, no disk, no file,
no program loaded from anywhere, no shell, no network, and only one CPU does scheduling.

## 2. The shortest path to something usable (milestones)

Do these in order; each one has a visible result in QEMU. Everything in section 3 is the detail behind them.

| # | Milestone | Needs (section 3) | You can see |
|---|---|---|---|
| M1 | **Device interrupts + keyboard** | 3.2, 3.9 | typing on the keyboard echoes on screen |
| M2 | **Real syscalls and a complete user ABI** | 3.3, 3.4, 3.5 | user programs use `syscall`, floats and threads without crashing |
| M3 | **Run programs that are not built into the kernel** | 3.6, 3.15 (RAM disk first) | an `init.lxf` from the bootloader's RAM disk starts in ring 3 |
| M4 | **Disk + filesystem** | 3.10, 3.11, 3.12 | programs and files read from a FAT32 virtio disk |
| M5 | **Shell and core tools** | 3.13, 3.14 | a prompt: `ls`, `cat`, `ps`, run a program |
| M6 | **SMP, security, networking, GUI** | 3.1, 3.7, 3.16, 3.17 | multiple CPUs busy, users/permissions, ping, windows |

M1-M5 is "a tiny but real OS". M6 and the rest is what makes it a *fully working* one.

## 3. What to add

### 3.1 Multi-CPU scheduling (L)
*Now:* application processors start, then park in `hlt`. `hal/paging.rs` has no TLB shootdown; the scheduler runs only on the
boot CPU; locks are global `IrqLock`s; the PMM is documented as not interrupt-safe.
*Add:* per-CPU run queues and idle threads, load balancing and CPU affinity; an IPI API (reschedule, TLB shootdown, panic-stop)
in `hal/apic.rs`; TLB shootdown in the VMM (mandatory before any second CPU runs kernel code, see the note in `mm/vmm.rs`);
per-CPU data reached through `GS` with `swapgs` on every kernel entry (`hal/percpu.rs`); make the PMM interrupt-safe; audit the lock
order (`heap -> vmm -> pmm`) for contention, replace the biggest locks with finer ones.
*Done when:* `./run.sh --smp 8` runs 8 busy threads on 8 CPUs, and a stress test of map/unmap from several CPUs shows no stale TLB.

### 3.2 Interrupt routing and time (M)
*Now:* LAPIC and its timer work; PIC is masked; no I/O APIC, no MSI; x2APIC unsupported (the driver reports an error if firmware
enables it); TSC not calibrated; no wall-clock time.
*Add:* MADT parsing (I/O APIC, interrupt source overrides, NMI) -> an I/O APIC driver and a "route this GSI to that vector on that CPU"
API; x2APIC mode (MSR access) for machines that start in it; MSI/MSI-X (needed by modern PCIe devices); TSC calibration (HPET/PIT) and
use of the invariant TSC as the fast clock; RTC (CMOS, later UEFI/ACPI) for wall-clock time; one `time` module with monotonic time,
wall time and timers (`KeSetTimer`-like), tickless idle later.

### 3.3 Fast system calls and a stable ABI (M)
*Update:* user programs are loaded from LXF (not ELF) by `src/ld/` (chapter 22); `syscall`/`sysret` and calls 5-7, 19-32 (threads, wait, events, `vm_*`, child processes) now exist (`src/hal/syscall.rs`, `src/ex/`). Still open: SMEP/SMAP with `stac/clac`, fault-recovering user copies, a versioned table beyond `sysinfo`'s ABI word.

*Now:* only the software interrupt `int 0x80` (docs say it is a placeholder); numbers 0-4 and 8-18 exist.
*Add:* `syscall`/`sysret` (STAR/LSTAR/SFMASK are already planned in `hal/gdt.rs`) with `swapgs` and a per-CPU kernel stack; a documented,
versioned syscall table; SMEP/SMAP enabled with `stac/clac` around user copies; user-copy with fault recovery (a bad pointer must fail the
call, not the kernel). Keep `int 0x80` only as a debug path or remove it.

### 3.4 CPU state: FPU / SSE / AVX (M)
*Now:* the kernel is soft-float and the context switch saves no FPU/SSE state, so user programs cannot use floating point or SSE at all
(and compiled user code will use SSE even for `memcpy`).
*Add:* enable `CR0.MP/EM`, `CR4.OSFXSR/OSXMMEXCPT/OSXSAVE`, a per-thread `XSAVE` area (size from CPUID 0xD), eager save/restore on switch
(lazy later), `FS`/`GS` base per thread (thread-local storage), and `#NM`/`#XM`/`#XF` handlers. Keep the *kernel* soft-float.

### 3.5 Complete Executive system services (L)
*Now:* user mode can write to the debug console, sleep, and use ports/sections. It cannot create a process or thread, wait on an object,
use an Event, allocate or free memory, or query anything.
*Add (as system calls, each taking handles checked by the Object Manager):* `NtCreateProcess/Thread`, terminate, wait-for-one/many objects,
event/mutex/semaphore/timer objects (only Event exists in `ob/types.rs`), `VirtualAlloc/Free/Protect/Query`, open/close/duplicate handle,
query-information calls, exit codes and `wait` for child processes, `GetTickCount`/time. Put the dispatcher in a new `ex/` module; keep
`ps/user.rs` for the entry/exit path only.

### 3.6 Loading real user programs (L)
*Now:* user programs are four small assembly blobs built into the kernel (`ps/user.rs`, max 256 KiB, one flat image at `0x400000`).
*Add:* a kernel loader (`ps/loader.rs`) for **user LXF executables and LLF libraries** (the format and the host-side `run_lord.py` are in
`tools/lxf`): section mapping with W^X, relocations, imports/IAT, `LibraryMain`, argv/environment on the new stack, ASLR; a toolchain
that *produces* such files from compiled code (today only `nasm` + `flat2lord.py` and the kernel-only `mkkernel.py` exist): either a
`mkkernel.py --user` mode, or a linker script plus tool that emits `.text/.data/.bss/.reloc/.idata/.edata`. Note the current format has a single
`.text` section for code and data; a loader needs separate RX / R / RW segments (extend LXF, see how `.kseg` does it for kernels).

### 3.7 Security (L)
*Now:* the Object Manager has access masks and a reserved `security` field (`ob/types.rs`, "always 0 now"); nothing is enforced.
*Add:* the **Security Reference Monitor** (kernel): security identifiers, access tokens per process, ACLs on objects, privileges, an
access check on every handle open; then the **Security subsystem** (user mode): users, groups, authentication, policy. Without this the
system has exactly one privilege level for everything.

### 3.8 Virtual memory completion (L)
*Now:* user page faults kill the process; memory is committed eagerly; no copy-on-write; bootloader-reclaimable memory (44 MiB in the
default run) and memory below 1 MiB are never reclaimed.
*Add:* a page-fault handler with demand-zero pages, stack growth, copy-on-write, memory-mapped files, protection changes, large pages for the
direct map; reclaim bootloader memory after boot; a physical-memory low-watermark with page reclaim, and optionally swap; per-process memory
limits; `fork`-style duplication only if a POSIX layer is planned (`CreateProcess` is the native model).

### 3.9 Kernel robustness and debugging (M)
*Now:* a panic prints the message and halts; log goes to serial and screen only.
*Add:* register dump and stack backtrace on panic and on fatal exceptions (the stack frames already end in a zero return address);
symbolization from a map file; a kernel log ring buffer readable from user mode (`dmesg`); a GDB stub over serial; kernel stack
overflow and stack-protector checks; a watchdog for hung CPUs; leak and lock-order checkers in debug builds; crash dump to disk once
storage exists.

### 3.10 I/O manager, driver model, PnP, power (XL)
*Now:* absent entirely (docs/12 lists them as planned).
*Add:* Device and Driver object types in the Object namespace (`\Device`, `\Driver`); an I/O request packet (IRP) model with
completion and cancellation; driver loading (first statically linked kernel drivers, later loadable LLF modules); a **PnP manager**
(enumeration, resource assignment, hot-plug); a **Power manager** with ACPI shutdown/reboot (FADT PM1 control, reset register) as the first
deliverable, then sleep states and idle power. ACPI: today only table lookup; add either an AML interpreter or a vendored one (uACPI or
ACPICA) for device resources, battery, power button and embedded controller.

### 3.11 Buses and storage drivers (L)
*Add:* PCI/PCIe enumeration (ECAM from the MCFG table, BARs, capabilities, MSI); a block-device layer (queues, partitions via GPT); drivers in
this order: **virtio-blk** (QEMU, simplest), **AHCI** (SATA), **NVMe**; later USB mass storage. A page/buffer cache between the block layer
and the filesystems.

### 3.12 Filesystems (L)
*Add:* a File object type in the Object namespace and a VFS/IO-manager path (`\Device\Harddisk0\Partition1`, drive/mount names); **FAT32**
first (it is what the ESP already is, so the kernel and tools can read the boot disk), then one real filesystem (ext2 as a read/write
starting point, or a native design with journaling); file syscalls (create/open/read/write/seek/stat/enumerate/rename/delete), memory-mapped
files hooked into 3.8, file-system permissions tied to 3.7.

### 3.13 Input, console and graphics (L)
*Now:* output only, through the serial port and an 8x8 font framebuffer console.
*Add:* input: PS/2 keyboard and mouse (i8042) first, then USB HID (xHCI driver, hubs); key maps; a console/TTY device with line editing for
the shell; scalable fonts and a compositor-friendly framebuffer (share it with user mode through a Section); optional GPU drivers
(virtio-gpu, Bochs/VBE) and mode setting; the window manager and GDI are meant to live in user mode (docs/12), so the kernel only needs the
raw surface, input events and a message path (the IPC manager already provides the last one).

### 3.14 Userland (XL: the bulk of the work)
Nothing of this exists yet:
1. **Native runtime library** ("LORD native API"): syscall stubs, heap, threads, TLS, error codes. Then a **C library** (port musl, or write a minimal
   one) and, if wanted, a **Rust `std` target** for Lord Core (`x86_64-unknown-lord`) so programs can be written in Rust.
2. **init / session manager**: the first user process; starts the subsystems and services in order; restarts them if they die.
3. **Environment subsystem LORD** (user mode, talks to the kernel through ports): the native API that applications program against.
4. **Integral subsystems** from the architecture diagram: Security, Client service, Server service; a logging service; a configuration store
   (registry-like) for system settings.
5. **Console/terminal subsystem, shell, core tools** (`ls cat echo mkdir rm cp mv ps kill mem`), a text editor, a simple package/format for
   installed programs.
6. **GUI** (optional): window manager, compositor, widget toolkit, running as user-mode processes.

### 3.15 Boot-to-userland hand-off (S-M; touches Lord Boot)
Until a disk driver exists, the kernel needs something to start: the bootloader (`lord-boot/ROADMAP.md` section 2.1) must load an **initial RAM
disk** and pass it in `KernelBoot`; the kernel unpacks it into a RAM filesystem and starts `\init`. This is what makes M3 possible *before*
M4. The kernel side: a `boot::modules()` accessor in `src/boot/mod.rs`, a read-only ramdisk filesystem, and starting `init` from
`main.rs` after `ipc::self_test()`.

### 3.16 Networking (XL)
*Add:* NIC drivers (virtio-net, then e1000/e1000e, RTL8139/8168); a stack: Ethernet, ARP, IPv4 (IPv6 later), ICMP, UDP, TCP, DHCP, DNS,
sockets exposed through the environment subsystem. Per the architecture, the protocol stack can run in user mode (Client/Server services) with
only the NIC drivers and packet buffers in the kernel; TLS and higher protocols are libraries.

### 3.17 Hardening, portability and maintenance (ongoing)
KASLR (needs a relocatable kernel image, see `lord-boot/ROADMAP.md` 2.4), SMEP/SMAP/NX everywhere, stack protector, syscall fuzzing, testing on
**real hardware** (so far only QEMU/OVMF), CI that runs `build.sh` and `test.sh` on every commit, sanitizers in host
tests, a coding-standard check (`rustfmt`, `clippy`), `aarch64` or other ports only if the HAL boundary stays clean.

## 4. Known smaller items already noted in the code
* `ps/user.rs` limits user images to 64 pages and has no heap/stack growth; replaced by 3.6/3.8.
* The kernel is built soft-float; user mode will not be (3.4).
* The bootloader leaves 44 MiB of `BOOTLOADER_RECLAIMABLE` memory unused after boot (3.8).
* `src/boot/lord_boot/protocol.rs` and `lord-boot/src/protocol.rs` must stay byte-identical, and `tools/lxf/lordfmt.py`/`mkkernel.py` must match the copies in `lord-docs-tests`; `tests/check-protocol-sync.sh` in `lord-docs-tests` checks them.
