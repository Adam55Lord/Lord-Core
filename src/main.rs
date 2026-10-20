// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Lord Core: a hybrid x86_64 kernel by Adam55Lord.
//!
//! Boot flow:  firmware -> bootloader -> `boot::<backend>::_start` -> [`kmain`].
//! The boot backend (Lord Boot today, see `src/boot/`) loads this kernel (an LXF image, `kernel.lxf`) into the
//! higher half, brings the CPU to 64-bit mode with a stack and a framebuffer,
//! translates what the bootloader knows into a bootloader-independent
//! [`boot::BootInfo`] and calls `kmain`. Nothing below this file names a
//! bootloader.

#![no_std]
#![no_main]

extern crate alloc;

#[macro_use]
mod log;

mod boot;
mod ex;
mod ld;
mod hal;
mod ipc;
mod mm;
mod ob;
mod ps;

use core::panic::PanicInfo;

use boot::BootInfo;

const NAME: &str = "Lord Core";
const VERSION: &str = env!("CARGO_PKG_VERSION");
const AUTHOR: &str = "Adam55Lord";

/// Everything the kernel drives is an object: describe the CPUs and the
/// devices found during boot in the namespace (`\\Processors`, `\\Device`).
fn register_kernel_objects(boot: &BootInfo, bsp_lapic_id: u32) {
    use ob::types::{create_device, create_processor, Device, Processor};
    use ob::ObjectAttributes;

    let mut n_cpu = 0;
    match boot.smp.as_ref() {
        Some(s) => {
            // Logical numbering as in hal::smp: BSP is 0, the others follow in list order.
            let mut next = 1usize;
            for c in s.cpus {
                let (index, bsp) = if c.lapic_id == bsp_lapic_id {
                    (0, true)
                } else {
                    next += 1;
                    (next - 1, false)
                };
                if index >= hal::MAX_CPUS {
                    continue;
                }
                let path = alloc::format!("\\Processors\\CPU{}", index);
                if create_processor(&ObjectAttributes::named(&path).permanent(), Processor { index, lapic_id: c.lapic_id, bsp }).is_ok() {
                    n_cpu += 1;
                }
            }
        }
        None => {
            let p = Processor { index: 0, lapic_id: bsp_lapic_id, bsp: true };
            if create_processor(&ObjectAttributes::named("\\Processors\\CPU0").permanent(), p).is_ok() {
                n_cpu = 1;
            }
        }
    }

    let dev = |path: &str, class, description, base, size, io_port| {
        let d = Device { class, description, base, size, io_port };
        if let Err(e) = create_device(&ObjectAttributes::named(path).permanent(), d) {
            warn!("ob:   could not register {}: {:?}", path, e);
        }
    };
    dev("\\Device\\Serial0", "serial", "16550 UART (COM1), kernel log", 0x3f8, 8, true);
    if let Some(fb) = boot.framebuffer {
        dev("\\Device\\Framebuffer", "display", "linear framebuffer, text console", fb.address as u64, (fb.pitch * fb.height) as u64, false);
    }
    dev("\\Device\\LocalApic", "interrupt", "local APIC of the boot CPU", hal::apic::read_base().phys, 4096, false);
    if let Some(h) = hal::hpet::info() {
        dev("\\Device\\Hpet", "timer", "High Precision Event Timer", h.phys, 1024, false);
    }
    debug!("ob:   registered {} processor object(s) and the boot devices", n_cpu);
}

fn human(bytes: u64) -> (u64, &'static str) {
    match bytes {
        b if b >= 1 << 30 => (b >> 30, "GiB"),
        b if b >= 1 << 20 => (b >> 20, "MiB"),
        b if b >= 1 << 10 => (b >> 10, "KiB"),
        b => (b, "B"),
    }
}

/// Kernel entry, called by the boot backend with a complete [`BootInfo`].
pub fn kmain(boot: &BootInfo) -> ! {
    // 1. HAL first: gives the logger its output devices (serial + screen).
    let status = hal::init(boot.framebuffer);

    // 2. Who we are.
    info!("{} v{} (x86_64)", NAME, VERSION);
    info!("Created by {}", AUTHOR);

    // 3. What the HAL brought up.
    debug!("hal: serial COM1 ........ {}", if status.serial { "ok" } else { "not found" });
    match status.display {
        Some(g) => debug!(
            "hal: framebuffer ....... {}x{}x{} -> {}x{} text cells (x{} scale)",
            g.width, g.height, g.bpp, g.cols, g.rows, g.scale
        ),
        None => warn!("hal: framebuffer ....... unavailable (serial output only)"),
    }

    // 4. What the bootloader told us (already translated by the boot backend).
    match boot.bootloader {
        Some(b) => {
            debug!("boot: bootloader ........ {} {}", b.name, b.version);
            debug!(
                "boot: protocol .......... {}, revision requested {:?}, loaded {:?}",
                b.protocol, b.revision_wanted, b.revision_loaded
            );
        }
        None => warn!("boot: bootloader info missing"),
    }
    match boot.firmware {
        Some(fw) => debug!("boot: firmware .......... {}", fw.name()),
        None => warn!("boot: firmware type missing"),
    }

    // 5. CPU identity and thread/core topology.
    let cpu = hal::cpu::CpuInfo::detect();
    debug!("cpu:  vendor ........... {}", cpu.vendor());
    debug!("cpu:  model ............ {}", cpu.brand());
    debug!("cpu:  ring before GDT .. {:?}", hal::gdt::current_ring());
    let topology = hal::cpu::Topology::detect();
    match (topology.threads_per_core, topology.logical_per_package) {
        (Some(threads), Some(logical)) => debug!(
            "cpu:  topology ......... {} thread(s)/core, {} logical CPU(s)/package (CPUID 0Bh)",
            threads, logical
        ),
        _ => debug!(
            "cpu:  topology ......... HTT={}, {} logical CPU(s)/package (legacy CPUID)",
            topology.htt, topology.legacy_logical_per_package
        ),
    }

    // 5b. Memory: HHDM + physical memory map (paging and the APIC need them).
    match (boot.hhdm_offset, boot.memory_map.is_empty()) {
        (Some(hhdm), false) => {
            hal::memory::init(hhdm, boot.memory_map);
            if boot.memory_map_dropped > 0 {
                warn!("mem:  {} memory-map entries did not fit the boot buffer", boot.memory_map_dropped);
            }
            debug!("mem:  hhdm offset ...... {:#018x}", hhdm);
            let (n, unit) = human(hal::memory::usable_bytes());
            debug!(
                "mem:  usable ........... {} {} in {} regions",
                n,
                unit,
                hal::memory::region_count()
            );
            hal::memory::for_each_region(|r| {
                trace!(
                    "mem:    {:#018x}-{:#018x}  {}",
                    r.base,
                    r.end(),
                    r.kind.name()
                );
            });
        }
        _ => warn!("mem:  HHDM or memory map missing"),
    }

    // 5c. Physical memory manager: owns every usable frame from here on
    // (paging, below, takes its page tables from it). Needs only the HHDM.
    match mm::pmm::init() {
        Ok(i) => {
            let (m, mu) = human(i.managed_frames * mm::pmm::FRAME_SIZE);
            debug!(
                "pmm:  managed .......... {} {} in {} range(s), {} frames (usable, >= 1 MiB; {:#x}-{:#x})",
                m, mu, i.ranges, i.managed_frames, i.lowest, i.highest
            );
            debug!(
                "pmm:  bitmap ........... {} bytes = {} frame(s) at {:#x} (1 bit/frame, covers frames 0..{})",
                i.meta_bytes, i.meta_frames, i.meta_phys, i.covered_frames
            );
            let (f, fu) = human(i.free_frames * mm::pmm::FRAME_SIZE);
            debug!("pmm:  free ............. {} frames ({} {})", i.free_frames, f, fu);
            if i.ranges_dropped > 0 {
                warn!("pmm:  {} range(s) did not fit the table and are unmanaged", i.ranges_dropped);
            }
            match mm::pmm::self_test() {
                Ok(t) => {
                    debug!(
                        "pmm:  self-test ........ {} frames cycled, zero-fill reused {} dirty frame(s), {} bad frees rejected, 64 KiB-aligned run at {:#x}",
                        t.frames_cycled, t.dirty_reused, t.rejected, t.contiguous_base
                    );
                    match t.below_16m {
                        Some(a) => debug!("pmm:  address limit .... a run below 16 MiB was allocated at {:#x}", a),
                        None => debug!("pmm:  address limit .... no usable memory below 16 MiB; limit checked with an impossible one"),
                    }
                }
                Err(why) => error!("pmm:  self-test FAILED: {}", why),
            }
        }
        Err(e) => error!("pmm:  initialisation FAILED: {:?}", e),
    }

    // 6. Boot CPU: GDT + TSS, per-CPU data, IDT, legacy PIC off.
    let bsp_lapic_id = boot
        .smp
        .as_ref()
        .map_or_else(hal::cpu::initial_apic_id, |s| s.bsp_lapic_id);

    hal::paging::init_pat(); // same cache-type layout whichever bootloader ran
    hal::gdt::init_cpu(0);
    hal::percpu::init(0, bsp_lapic_id);
    hal::syscall::init_cpu();
    debug!(
        "gdt:  loaded ............ base {:#x}, {} slots",
        hal::gdt::table_base(0),
        hal::gdt::GDT_ENTRIES
    );
    debug!(
        "gdt:  ring 0 code/data .. {:#06x} / {:#06x}",
        hal::gdt::KERNEL_CODE.raw(),
        hal::gdt::KERNEL_DATA.raw()
    );
    debug!(
        "gdt:  ring 3 code/data .. {:#06x} / {:#06x}",
        hal::gdt::USER_CODE.raw(),
        hal::gdt::USER_DATA.raw()
    );
    debug!(
        "tss:  loaded ............ selector {:#06x}, size {} bytes",
        hal::gdt::current_tr(),
        hal::tss::TSS_SIZE
    );
    debug!("tss:  rsp0 (ring3->0) ... {:#018x}", hal::tss::kernel_stack(0));
    for (index, role) in [
        (hal::tss::IST_DOUBLE_FAULT, "double fault"),
        (hal::tss::IST_NMI, "NMI"),
        (hal::tss::IST_MACHINE_CHECK, "machine check"),
        (hal::tss::IST_DEBUG, "debug"),
    ] {
        match hal::tss::ist_stack(0, index) {
            Some(top) => debug!("tss:  IST{} {:<13} {:#018x}", index, role, top),
            None => warn!("tss:  IST{} {} not set", index, role),
        }
    }
    match hal::gdt::verify(0) {
        Ok(()) => debug!(
            "gdt:  self-test ......... passed (running in ring {})",
            hal::gdt::current_ring().map_or(255, |r| r.number())
        ),
        Err(why) => error!("gdt:  self-test FAILED: {}", why),
    }

    hal::pic::disable();
    let (imr1, imr2) = hal::pic::masks();
    if hal::pic::is_disabled() {
        debug!(
            "pic:  8259 PIC .......... OFF (remapped to 0x20-0x2F, IMR {:#04x}/{:#04x} = all masked)",
            imr1, imr2
        );
    } else {
        error!("pic:  8259 PIC still has unmasked lines: {:#04x}/{:#04x}", imr1, imr2);
    }
    hal::idt::init();
    hal::idt::load();
    debug!(
        "idt:  loaded ............ base {:#x}, {} gates",
        hal::idt::table_base(),
        hal::idt::VECTOR_COUNT
    );
    debug!(
        "idt:  IST gates ......... #DF->IST{} NMI->IST{} #MC->IST{} #DB->IST{}",
        hal::idt::ist_for_vector(hal::idt::VEC_DOUBLE_FAULT as usize),
        hal::idt::ist_for_vector(hal::idt::VEC_NMI as usize),
        hal::idt::ist_for_vector(hal::idt::VEC_MACHINE_CHECK as usize),
        hal::idt::ist_for_vector(hal::idt::VEC_DEBUG as usize)
    );

    // IDT self-test: raise a breakpoint; the CPU must enter our handler and
    // come back here.
    hal::cpu::breakpoint();
    if hal::idt::count(hal::idt::VEC_BREAKPOINT) == 1 {
        debug!("idt:  self-test ......... int3 handled, execution resumed");
    } else {
        error!("idt:  self-test FAILED: breakpoint was not handled");
    }

    // 7. Paging self-test, then the Local APIC (read -> unmount -> mount -> enable).
    match hal::paging::self_test() {
        Ok(splits) => debug!(
            "mmu:  self-test ......... map/translate/unmap ok ({} large page(s) split), {} frame(s) held by paging",
            splits,
            mm::pmm::stats().in_use_frames
        ),
        Err(why) => error!("mmu:  self-test FAILED: {}", why),
    }
    let held_before = mm::pmm::stats().in_use_frames;
    match hal::paging::self_test_large_pages() {
        Ok(splits) => debug!(
            "mmu:  large pages ....... 1GiB->2MiB->4KiB split ok ({} splits), neighbours + attributes intact",
            splits
        ),
        Err(why) => error!("mmu:  large-page self-test FAILED: {}", why),
    }
    let held_after = mm::pmm::stats().in_use_frames;
    if held_after != held_before {
        warn!("mmu:  large-page test leaked {} frame(s)", held_after.wrapping_sub(held_before) as i64);
    }

    // 7a. Virtual memory manager (regions, guard pages, W^X, address spaces),
    // then the kernel heap on top of it. From here on `alloc` works.
    match mm::vmm::init() {
        Ok(i) => {
            let (sz, su) = human(i.arena_size);
            debug!(
                "vmm:  arena ............ {:#018x} + {} {} (one PML4 slot), NX {}, kernel root {:#x}",
                i.arena_base,
                sz,
                su,
                if i.nx { "enforced" } else { "NOT available" },
                i.kernel_root
            );
            debug!("vmm:  shared PML4 slots . {} created so address spaces can share them", i.slots_created);
            match mm::vmm::self_test() {
                Ok(t) => debug!(
                    "vmm:  self-test ......... {} allocs, {} guard pages hold, {} bad requests rejected (W^X, user-in-kernel, overlap, ...), tables reclaimed: {}; 2 address spaces isolated, {} frames reclaimed",
                    t.allocs, t.guards_checked, t.rejected, t.tables_reclaimed, t.address_space_frames
                ),
                Err(why) => error!("vmm:  self-test FAILED: {}", why),
            }
        }
        Err(e) => error!("vmm:  initialisation FAILED: {:?}", e),
    }
    match mm::heap::init() {
        Ok(i) => {
            let (sz, su) = human(i.arena_pages * mm::vmm::PAGE_SIZE);
            debug!(
                "heap: arena ............ {:#018x}, {} {} reserved, committed on demand (8 size classes 16 B..{} B, larger -> VMM)",
                i.arena_base,
                sz,
                su,
                mm::heap::MAX_SMALL
            );
            match mm::heap::self_test() {
                Ok(t) => debug!(
                    "heap: self-test ........ Box/Vec/String/BTreeMap ok (largest Vec {} KiB), {} stress ops, {} invalid frees rejected",
                    t.largest_vec / 1024,
                    t.stress_ops,
                    t.bad_frees_rejected
                ),
                Err(why) => error!("heap: self-test FAILED: {}", why),
            }
        }
        Err(e) => error!("heap: initialisation FAILED: {:?}", e),
    }

    // 7a'. Object Manager: "everything is an object". Types, the namespace
    // (`\`), the kernel handle table.
    match ob::init() {
        Ok(i) => {
            debug!(
                "ob:   object manager ... {} types, root `\\` + {} directories, kernel handle table ready",
                i.types, i.directories
            );
            match ob::self_test() {
                Ok(t) => debug!(
                    "ob:   self-test ......... {} checks (names, rights, counts, symlinks, 1000-handle table, hooks, address-space object), {} stress ops, nothing leaked",
                    t.checks, t.stress_ops
                ),
                Err(why) => error!("ob:   self-test FAILED: {}", why),
            }
        }
        Err(e) => error!("ob:   initialisation FAILED: {:?}", e),
    }

    // 7a''. IPC Manager: ports (message queues, request/reply, handle passing) and
    // sections (shared memory), both Object Manager types.
    match ipc::init() {
        Ok(i) => debug!(
            "ipc:  ipc manager ....... {} types (Port, Section), {} directories (\\Ports, \\Sections), syscalls {}-{}",
            i.types,
            i.directories,
            ipc::syscall::SYS_PORT_CREATE,
            ipc::syscall::SYS_HANDLE_DUP
        ),
        Err(e) => error!("ipc:  initialisation FAILED: {:?}", e),
    }

    let apic_ok = match hal::apic::init_bsp() {
        Ok(()) => {
            if hal::apic::self_ipi_test() {
                debug!(
                    "apic: self-test ......... self-IPI vector {:#04x} delivered and acknowledged (EOI)",
                    hal::apic::SELF_TEST_VECTOR
                );
            } else {
                error!("apic: self-test FAILED: self-IPI was not delivered");
            }
            true
        }
        Err(e) => {
            error!("apic: initialisation FAILED: {:?}", e);
            false
        }
    };

    // 7b. Time: ACPI -> HPET (PIT as fallback) is the reference clock; the
    // LAPIC timer is calibrated against it. Must precede the APs (the PIT is
    // a single shared device and the HPET mapping may unmount an alias).
    match hal::timer::init(boot.rsdp_phys) {
        Some(src) => info!("timer: reference clock is the {}", src.name()),
        None => warn!("timer: no reference clock (HPET and PIT both unavailable)"),
    }
    if apic_ok {
        match hal::apic_timer::init_bsp() {
            Ok(_) => match hal::apic_timer::self_test() {
                Ok(t) if !t.degraded => debug!(
                    "apic: timer self-test ... one-shot fires once; periodic {} Hz: {} ticks in {} us (expected {})",
                    t.hz,
                    t.ticks,
                    t.window_ns / 1_000,
                    t.expected
                ),
                Ok(t) => warn!(
                    "apic: timer self-test ... works, but periodic {} Hz delivered only {} of {} ticks in {} us (best of {}): interrupt latency > period; busy host?",
                    t.hz,
                    t.ticks,
                    t.expected,
                    t.window_ns / 1_000,
                    t.attempts
                ),
                Err(why) => error!("apic: timer self-test FAILED: {}", why),
            },
            Err(e) => error!("apic: timer calibration FAILED: {:?}", e),
        }
    }

    // 8. Other CPUs (hardware threads / cores).
    match boot.smp.as_ref() {
        Some(s) => {
            info!(
                "smp:  {} logical CPU(s) found (BSP lapic id {}, x2APIC {})",
                s.cpus.len(),
                s.bsp_lapic_id,
                s.x2apic
            );
            match topology.locate(bsp_lapic_id) {
                Some(p) => debug!(
                    "smp:  cpu 0 (BSP) ....... lapic id {} (package {} core {} thread {})",
                    bsp_lapic_id, p.package, p.core, p.thread
                ),
                None => debug!("smp:  cpu 0 (BSP) ....... lapic id {}", bsp_lapic_id),
            }
            let report = hal::smp::start(s);
            if report.skipped > 0 {
                warn!(
                    "smp:  {} CPU(s) left parked (MAX_CPUS = {})",
                    report.skipped,
                    hal::MAX_CPUS
                );
            }
            if report.online == report.requested + 1 {
                info!("smp:  {}/{} CPU(s) online", report.online, report.found);
            } else {
                warn!(
                    "smp:  only {} of {} requested CPU(s) came online",
                    report.online - 1,
                    report.requested
                );
            }
        }
        None => warn!("smp:  bootloader gave no MP info; running on one CPU"),
    }

    register_kernel_objects(boot, bsp_lapic_id);

    let s = mm::pmm::stats();
    let (fm, fu) = human(s.free_bytes());
    debug!(
        "pmm:  after boot ....... {} frame(s) in use (peak {}), {} free ({} {}), {} bad free(s) rejected",
        s.in_use_frames, s.peak_in_use_frames, s.free_frames, fm, fu, s.free_rejected
    );
    match mm::pmm::verify() {
        Ok(()) => debug!("pmm:  bitmap check ..... all invariants hold"),
        Err(why) => error!("pmm:  bitmap check FAILED: {}", why),
    }
    let v = mm::vmm::stats();
    debug!(
        "vmm:  after boot ....... {} region(s), {} page(s) reserved, {} committed (peak {})",
        v.regions, v.reserved_pages, v.committed_pages, v.peak_committed_pages
    );
    match mm::vmm::verify() {
        Ok(()) => debug!("vmm:  region check ..... table sorted, counters match the page tables, guards unmapped"),
        Err(why) => error!("vmm:  region check FAILED: {}", why),
    }
    mm::vmm::for_each_region(|r| {
        trace!(
            "vmm:    {:#018x} {:>6} pg ({} committed) {:<8} {}{}{}",
            r.start,
            r.pages,
            r.committed,
            r.kind.name(),
            r.tag,
            if r.guard_lo { " [guard-lo]" } else { "" },
            if r.guard_hi { " [guard-hi]" } else { "" }
        );
    });
    let h = mm::heap::stats();
    debug!(
        "heap: after boot ....... {} B live (peak {} B), {} allocs / {} frees, {} arena page(s) committed, {} failed, {} bad free(s), {} double free(s)",
        h.live_bytes,
        h.peak_live_bytes,
        h.allocs,
        h.frees,
        h.arena_committed_pages,
        h.failed,
        h.bad_frees,
        h.double_frees
    );
    {
        let o = ob::stats();
        debug!(
            "ob:   after boot ........ {} object(s) of {} type(s) (peak {}), {} open handle(s)",
            o.objects, o.types, o.peak_objects, o.handles
        );
        for t in ob::type_stats() {
            trace!("ob:     {:<13} {:>3} object(s), {} handle(s)", t.name, t.objects, t.handles);
        }
        for e in ob::namespace::snapshot() {
            let indent = "  ".repeat(e.depth);
            match &e.link_target {
                Some(t) => debug!("ob:   \\ {}{} [{}] -> {}", indent, e.name, e.type_name, t),
                None => debug!("ob:   \\ {}{} [{}]{}", indent, e.name, e.type_name, if e.permanent { " permanent" } else { "" }),
            }
        }
    }
    match mm::heap::verify() {
        Ok(()) => debug!("heap: free-list check .. canaries intact, carved == free + live for every class"),
        Err(why) => error!("heap: free-list check FAILED: {}", why),
    }

    // 9. Process manager: processes and threads as objects, then the scheduler.
    match ps::init() {
        Ok(()) => debug!(
            "ps:   process manager ... System process (pid 0) + idle thread (tid 0) created; {} priority levels, {} Hz tick, {}-tick quantum",
            ps::structs::priority::LEVELS,
            ps::TICK_HZ,
            ps::QUANTUM_TICKS
        ),
        Err(e) => {
            error!("ps:   initialisation FAILED: {:?}", e);
            info!("Initialisation complete (without a scheduler). Halting CPU.");
            hal::cpu::halt_forever()
        }
    }
    info!("Starting the scheduler: the boot CPU becomes the idle thread, the rest of the boot runs in a thread");
    if let Err(e) = ps::start(boot_driver) {
        error!("ps:   cannot start the scheduler: {:?}", e);
        hal::cpu::halt_forever()
    }
    ps::idle_loop()
}

/// The first real thread: runs the process manager's self-test under the live
/// scheduler, reports, and finishes the boot.
fn boot_driver(_: usize) {
    match ps::run_self_test() {
        Ok(t) => debug!(
            "ps:   self-test ......... 2 spinners shared the CPU ({} and {} ticks, {} context switches), wake latency {} tick(s), sleep(50 ms) = {} ticks, \
             events/timeouts/terminate ok, {} user processes ({} syscalls in the first), {} short threads, nothing leaked",
            t.fair_ticks.0, t.fair_ticks.1, t.fair_switches, t.wake_latency_ticks, t.sleep_ticks, t.user_processes, t.user_syscalls, t.stress_threads
        ),
        Err(why) => error!("ps:   self-test FAILED: {}", why),
    }
    match ipc::self_test() {
        Ok(t) => debug!(
            "ipc:  self-test ......... {} groups: {} messages ({} bytes) through ports, {} request/reply round trips, handles passed in messages, \
             {} shared-memory views (kernel and ring 3), {} IPC syscalls from ring 3, a blocked process killed cleanly, {} messages in the stress run, nothing leaked",
            t.checks, t.messages, t.bytes, t.requests, t.views, t.user_syscalls, t.stress_messages
        ),
        Err(why) => error!("ipc:  self-test FAILED: {}", why),
    }
    if let Err(e) = ld::init() {
        error!("ld:   cannot register the built-in libraries: {}", e.reason());
    }
    match ld::selftest::run() {
        Ok(t) => debug!(
            "ld:   self-test ......... LXF loader: {} libraries registered, program with relocation + imports + LibraryMain ran (exit code {}), {} malformed images refused, nothing leaked",
            ld::library_count(), t.exit_code, t.refused
        ),
        Err(why) => error!("ld:   self-test FAILED: {}", why),
    }
    match ex::selftest::run() {
        Ok(t) => debug!(
            "ex:   self-test ......... `syscall`/`sysret` gate, {} steps in ring 3 (time, sysinfo, vm_alloc/free/protect, events, wait, user threads, child processes, terminate, bad pointers), {} fast calls, nothing leaked",
            t.steps, t.fast_calls
        ),
        Err(why) => error!("ex:   self-test FAILED: {}", why),
    }
    let is = ipc::stats();
    debug!(
        "ipc:  after boot ........ {} port(s), {} section(s) alive; {} messages sent, {} received, {} requests, {} replies, {} view(s) mapped, {} syscalls",
        is.ports, is.sections, is.sent, is.received, is.requests, is.replies, is.views_mapped, is.syscalls
    );
    let st = ps::stats();
    debug!(
        "ps:   after boot ....... {} ticks, {} context switches ({} by preemption), {} process(es), {} thread(s), {} ready",
        st.ticks, st.switches, st.preemptions, st.processes, st.threads, st.ready
    );
    for p in ps::process_infos() {
        debug!(
            "ps:     pid {:>3} (parent {}) {:<8} {} thread(s) of {} ever, prio {}, {}, since tick {}{}",
            p.pid,
            p.parent,
            p.name,
            p.threads,
            p.total_threads,
            p.base_priority,
            if p.exited { "exited" } else { "active" },
            p.created_tick,
            if p.user { ", user mode" } else if p.exited && p.exit_code != 0 { ", failed" } else { "" }
        );
    }
    for t in ps::thread_infos() {
        debug!(
            "ps:     tid {:>3} pid {:>3} {:<12} {:<10} prio {:>2} (base {:>2}) {} cpu ticks, {} switches in, created at tick {}{}",
            t.tid,
            t.pid,
            t.name,
            t.state.name(),
            t.priority,
            t.base_priority,
            t.cpu_ticks,
            t.switches_in,
            t.created_tick,
            if t.user { ", ring 3" } else { "" }
        );
    }
    info!("Initialisation complete. The system is running.");
}

#[panic_handler]
fn panic(panic_info: &PanicInfo) -> ! {
    log::unlock_for_panic();
    error!("*** KERNEL PANIC ***");
    // `PanicInfo`'s Display impl prints the location and the message.
    error!("{}", panic_info);
    hal::cpu::halt_forever()
}
