//! Colibrì's lease contract served by Weightsift's store: the C shim's entry points (`csrc/ws_colibri_store.h`).
//!
//! A C or C++ engine calls them directly; Rust holds the result as an opaque `ColiExpertStore` and closes it with
//! `ColibriStore`'s drop. The contract itself (lookups, releases) is exercised in C, through Colibrì's own inline
//! helpers, by the crate's tests.

use std::ffi::{c_char, c_int, CStr, CString};
use std::path::Path;
use std::ptr::NonNull;

use weightsift_store::ffi::WsStoreOptions;

/// Colibrì's `ColiExpertStore`, opaque to Rust.
#[repr(C)]
pub struct ColiExpertStore {
    _private: [u8; 0],
}

extern "C" {
    pub fn ws_colibri_store_open(
        catalog_path: *const c_char,
        options: *const WsStoreOptions,
        out: *mut *mut ColiExpertStore,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;
    pub fn ws_colibri_store_handle(store: *const ColiExpertStore) -> u64;
    pub fn ws_colibri_store_destroy(store: *mut ColiExpertStore);
}

/// A `ColiExpertStore` served by Weightsift, for handing to a Colibrì engine. Destroyed when dropped (refused, and
/// kept, while leases are alive: Colibrì's contract).
pub struct ColibriStore {
    raw: NonNull<ColiExpertStore>,
}

impl ColibriStore {
    pub fn open(catalog: &Path, options: &WsStoreOptions) -> Result<Self, String> {
        let path = CString::new(catalog.to_str().ok_or("the catalog path is not UTF-8")?)
            .map_err(|_| "the catalog path holds a NUL")?;
        let mut raw = std::ptr::null_mut();
        let mut error = [0 as c_char; 512];
        // SAFETY: a NUL-terminated path, a valid options struct, an out pointer and an error buffer of its length.
        let status =
            unsafe { ws_colibri_store_open(path.as_ptr(), options, &mut raw, error.as_mut_ptr(), error.len()) };
        match NonNull::new(raw) {
            Some(raw) if status == 0 => Ok(ColibriStore { raw }),
            // SAFETY: the shim wrote a NUL-terminated message into the buffer.
            _ => Err(unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy().into_owned()),
        }
    }

    /// The `ColiExpertStore*` to give a Colibrì engine (valid until this value is dropped).
    pub fn as_ptr(&self) -> *mut ColiExpertStore {
        self.raw.as_ptr()
    }

    /// The Weightsift store behind it (for `weightsift_store::ffi` calls: statistics, the catalog).
    pub fn handle(&self) -> u64 {
        // SAFETY: a live store opened by the shim.
        unsafe { ws_colibri_store_handle(self.raw.as_ptr()) }
    }
}

impl Drop for ColibriStore {
    fn drop(&mut self) {
        // SAFETY: a live store opened by the shim, destroyed once.
        unsafe { ws_colibri_store_destroy(self.raw.as_ptr()) }
    }
}
