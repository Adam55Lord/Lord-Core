// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Adam55Lord

//! Access rights (`ACCESS_MASK`), NT style.
//!
//! A 32-bit mask: the low 16 bits are **object-specific** rights (each type
//! defines its own, e.g. `EVENT_MODIFY_STATE`), bits 16-20 are **standard**
//! rights every object has, and the top four bits are **generic** rights that
//! are translated through the type's [`GenericMapping`] into specific ones.

pub type AccessMask = u32;

pub const DELETE: AccessMask = 0x0001_0000;
pub const READ_CONTROL: AccessMask = 0x0002_0000;
pub const WRITE_DAC: AccessMask = 0x0004_0000;
pub const WRITE_OWNER: AccessMask = 0x0008_0000;
pub const SYNCHRONIZE: AccessMask = 0x0010_0000;
pub const STANDARD_RIGHTS_REQUIRED: AccessMask = 0x000F_0000;
pub const STANDARD_RIGHTS_ALL: AccessMask = 0x001F_0000;

pub const MAXIMUM_ALLOWED: AccessMask = 0x0200_0000;
pub const GENERIC_ALL: AccessMask = 0x1000_0000;
pub const GENERIC_EXECUTE: AccessMask = 0x2000_0000;
pub const GENERIC_WRITE: AccessMask = 0x4000_0000;
pub const GENERIC_READ: AccessMask = 0x8000_0000;
const GENERIC_BITS: AccessMask = GENERIC_ALL | GENERIC_EXECUTE | GENERIC_WRITE | GENERIC_READ;

/// What the four generic rights mean for one object type.
#[derive(Clone, Copy, Debug)]
pub struct GenericMapping {
    pub read: AccessMask,
    pub write: AccessMask,
    pub execute: AccessMask,
    pub all: AccessMask,
}

impl GenericMapping {
    /// Replaces the generic bits of `access` by the type-specific rights.
    pub const fn map(&self, access: AccessMask) -> AccessMask {
        let mut out = access & !GENERIC_BITS;
        if access & GENERIC_READ != 0 {
            out |= self.read;
        }
        if access & GENERIC_WRITE != 0 {
            out |= self.write;
        }
        if access & GENERIC_EXECUTE != 0 {
            out |= self.execute;
        }
        if access & GENERIC_ALL != 0 {
            out |= self.all;
        }
        out
    }
}

// --- Directory ---------------------------------------------------------
pub const DIRECTORY_QUERY: AccessMask = 0x0001;
pub const DIRECTORY_TRAVERSE: AccessMask = 0x0002;
pub const DIRECTORY_CREATE_OBJECT: AccessMask = 0x0004;
pub const DIRECTORY_CREATE_SUBDIRECTORY: AccessMask = 0x0008;
pub const DIRECTORY_ALL: AccessMask = STANDARD_RIGHTS_REQUIRED | 0x000F;

// --- Symbolic link -----------------------------------------------------
pub const SYMLINK_QUERY: AccessMask = 0x0001;
pub const SYMLINK_ALL: AccessMask = STANDARD_RIGHTS_REQUIRED | 0x0001;

// --- Event ---------------------------------------------------------------
pub const EVENT_QUERY_STATE: AccessMask = 0x0001;
pub const EVENT_MODIFY_STATE: AccessMask = 0x0002;
pub const EVENT_ALL: AccessMask = STANDARD_RIGHTS_REQUIRED | SYNCHRONIZE | 0x0003;

// --- Device ------------------------------------------------------------
pub const DEVICE_QUERY: AccessMask = 0x0001;
pub const DEVICE_READ: AccessMask = 0x0002;
pub const DEVICE_WRITE: AccessMask = 0x0004;
pub const DEVICE_ALL: AccessMask = STANDARD_RIGHTS_REQUIRED | SYNCHRONIZE | 0x0007;

// --- Processor -----------------------------------------------------------
pub const PROCESSOR_QUERY: AccessMask = 0x0001;
pub const PROCESSOR_ALL: AccessMask = STANDARD_RIGHTS_REQUIRED | 0x0001;

// --- Address space -------------------------------------------------------
pub const ADDRESS_SPACE_QUERY: AccessMask = 0x0001;
pub const ADDRESS_SPACE_MAP: AccessMask = 0x0002;
pub const ADDRESS_SPACE_ALL: AccessMask = STANDARD_RIGHTS_REQUIRED | 0x0003;

// --- Type objects --------------------------------------------------------
pub const TYPE_QUERY: AccessMask = 0x0001;
pub const TYPE_ALL: AccessMask = STANDARD_RIGHTS_REQUIRED | 0x0001;

// --- Process ---------------------------------------------------------------
pub const PROCESS_QUERY: AccessMask = 0x0001;
pub const PROCESS_TERMINATE: AccessMask = 0x0002;
pub const PROCESS_CREATE_THREAD: AccessMask = 0x0004;
pub const PROCESS_ALL: AccessMask = STANDARD_RIGHTS_REQUIRED | SYNCHRONIZE | 0x0007;

// --- Thread ----------------------------------------------------------------
pub const THREAD_QUERY: AccessMask = 0x0001;
pub const THREAD_TERMINATE: AccessMask = 0x0002;
pub const THREAD_SET_INFORMATION: AccessMask = 0x0004;
pub const THREAD_ALL: AccessMask = STANDARD_RIGHTS_REQUIRED | SYNCHRONIZE | 0x0007;

// --- Port (IPC) ---------------------------------------------------------------
pub const PORT_SEND: AccessMask = 0x0001;
pub const PORT_RECEIVE: AccessMask = 0x0002;
pub const PORT_QUERY: AccessMask = 0x0004;
pub const PORT_ALL: AccessMask = STANDARD_RIGHTS_REQUIRED | 0x0007;

// --- Section (shared memory) -----------------------------------------------------
pub const SECTION_QUERY: AccessMask = 0x0001;
pub const SECTION_MAP_READ: AccessMask = 0x0002;
pub const SECTION_MAP_WRITE: AccessMask = 0x0004;
pub const SECTION_ALL: AccessMask = STANDARD_RIGHTS_REQUIRED | 0x0007;
