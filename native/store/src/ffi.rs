//! The C ABI of the expert store (`include/weightsift_store.h`, version 1): the narrow boundary through which C and C++
//! execution engines (Colibrì's, through `weightsift-colibri`) lease experts.
//!
//! Memory safety at the boundary does not rest on the caller: stores and leases are 64-bit handles looked up in tables,
//! never pointers to Rust objects, and never reused, so an unknown, closed, released or doubly released handle is an
//! error code, not undefined behaviour. A lease's bytes are an `Arc` of the cache's entry held in the store's table until
//! the release; a store is an `Arc` in a global registry, so a call in flight keeps it alive even across a racing close.
//! Close takes a write gate that every acquire holds for reading, so no lease is issued by a closing store, and it is
//! refused while leases are alive. Panics are caught at every entry point and become `WS_ERR_INTERNAL`.
//!
//! What remains the caller's contract (the header's): pointers passed in point to what the header says (a NUL-terminated
//! path, `count` keys, structs whose `struct_size` is their allocation), and a lease's bytes are not used after its
//! release. The `unsafe` blocks below are the dereferences of those caller pointers, each stating what it relies on; this
//! is the demonstrated requirement of a C ABI (decision 0014).

use std::collections::HashMap;
use std::ffi::{c_char, CStr};
use std::mem::size_of;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use crate::store::{ExpertKey, ExpertStore, Lease, StoreError, StoreOptions};

pub const WS_STORE_ABI_VERSION: u32 = 1;

pub const WS_OK: i32 = 0;
pub const WS_ERR_INVALID_ARGUMENT: i32 = -1;
pub const WS_ERR_INVALID_HANDLE: i32 = -2;
pub const WS_ERR_INVALID_LEASE: i32 = -3;
pub const WS_ERR_UNAVAILABLE: i32 = -4;
pub const WS_ERR_IO: i32 = -5;
pub const WS_ERR_CATALOG: i32 = -6;
pub const WS_ERR_LEASES_OUTSTANDING: i32 = -7;
pub const WS_ERR_CLOSED: i32 = -8;
pub const WS_ERR_INTERNAL: i32 = -9;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WsStoreOptions {
    pub struct_size: u32,
    pub direct: u32,
    pub host_cache_bytes: u64,
    pub workers: u32,
    pub staging_sets: u32,
    pub staging_slots: u32,
    pub max_prefetches: u32,
    pub staging_slot_bytes: u64,
    pub cache_block_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WsExpertKey {
    pub layer: u32,
    pub expert: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct WsSpan {
    pub data: *const u8,
    pub bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct WsView {
    pub struct_size: u32,
    pub part: u32,
    pub name: *const c_char,
    pub format: *const c_char,
    pub data: *const u8,
    pub bytes: u64,
    pub scale_format: *const c_char,
    pub scales: *const u8,
    pub scale_bytes: u64,
    pub rows: i64,
    pub columns: i64,
    pub block_rows: u32,
    pub block_columns: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WsStoreStats {
    pub struct_size: u32,
    pub abi_version: u32,
    pub acquire_calls: u64,
    pub acquire_failures: u64,
    pub experts: u64,
    pub expert_hits: u64,
    pub expert_waits: u64,
    pub expert_misses: u64,
    pub releases: u64,
    pub active_leases: u64,
    pub leased_bytes: u64,
    pub peak_leased_bytes: u64,
    pub prefetch_calls: u64,
    pub prefetch_refused: u64,
    pub prefetch_experts: u64,
    pub prefetch_rows: u64,
    pub prefetch_used: u64,
    pub prefetch_wasted: u64,
    pub requests: u64,
    pub rows: u64,
    pub logical_bytes: u64,
    pub physical_bytes: u64,
    pub read_calls: u64,
    pub extents: u64,
    pub blocks_4k: u64,
    pub read_ns: u64,
    pub busy_ns: u64,
    pub cache_lookups: u64,
    pub cache_hits: u64,
    pub cache_waits: u64,
    pub cache_misses: u64,
    pub cache_inserts: u64,
    pub cache_evictions: u64,
    pub cache_bypassed: u64,
    pub cache_capacity_bytes: u64,
    pub cache_block_bytes: u64,
    pub cache_resident_bytes: u64,
    pub cache_held_bytes: u64,
    pub cache_peak_held_bytes: u64,
    pub staging_bytes: u64,
}

struct Open {
    store: ExpertStore,
    /// True once closed. An acquire holds the read side, a close the write side.
    closed: RwLock<bool>,
    leases: Mutex<HashMap<u64, Lease>>,
}

fn registry() -> &'static Mutex<HashMap<u64, Arc<Open>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<u64, Arc<Open>>>> = OnceLock::new();
    REGISTRY.get_or_init(Default::default)
}

static NEXT_STORE: AtomicU64 = AtomicU64::new(1);
static NEXT_LEASE: AtomicU64 = AtomicU64::new(1);

type Failure = (i32, String);

fn lookup(store: u64) -> Result<Arc<Open>, Failure> {
    registry()
        .lock()
        .unwrap()
        .get(&store)
        .cloned()
        .ok_or((WS_ERR_INVALID_HANDLE, format!("{store} is not an open store")))
}

fn failure(error: StoreError) -> Failure {
    let status = match &error {
        StoreError::Catalog(_) => WS_ERR_CATALOG,
        StoreError::Invalid(_) => WS_ERR_INVALID_ARGUMENT,
        StoreError::Unavailable(_) => WS_ERR_UNAVAILABLE,
        StoreError::Io(_) => WS_ERR_IO,
        StoreError::Closed => WS_ERR_CLOSED,
        StoreError::Internal(_) => WS_ERR_INTERNAL,
    };
    (status, error.to_string())
}

fn invalid(message: &str) -> Failure {
    (WS_ERR_INVALID_ARGUMENT, message.to_owned())
}

/// Run `body`, turning a failure or a panic into a status (and a message in the caller's buffer).
fn boundary(error: *mut c_char, error_size: usize, body: impl FnOnce() -> Result<(), Failure>) -> i32 {
    let outcome = catch_unwind(AssertUnwindSafe(body)).unwrap_or_else(|panic| {
        let message = panic
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "a panic".into());
        Err((WS_ERR_INTERNAL, format!("internal error of the store: {message}")))
    });
    match outcome {
        Ok(()) => WS_OK,
        Err((status, message)) => {
            if !error.is_null() && error_size > 0 {
                let bytes = message.as_bytes();
                let n = bytes.len().min(error_size - 1);
                // SAFETY: the caller gave `error_size` writable bytes at `error` (the header's contract).
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), error as *mut u8, n);
                    *error.add(n) = 0;
                }
            }
            status
        }
    }
}

/// Write `value` into the caller's versioned struct at `out`: what both know, the rest of the caller's struct zeroed;
/// `struct_size` becomes the size written.
///
/// # Safety
/// `out` points to a struct of the caller's whose first field (`u32`) is its allocation size, writable for that size.
unsafe fn write_versioned<T: Copy>(out: *mut T, value: &T) -> Result<(), Failure> {
    let caller = (out as *const u32).read() as usize;
    if caller < 2 * size_of::<u32>() {
        return Err(invalid("struct_size is too small"));
    }
    let n = caller.min(size_of::<T>());
    std::ptr::write_bytes(out as *mut u8, 0, caller);
    std::ptr::copy_nonoverlapping(value as *const T as *const u8, out as *mut u8, n);
    (out as *mut u32).write(n as u32);
    Ok(())
}

/// Read the caller's versioned input struct (the fields this version knows; it must hold all of them).
///
/// # Safety
/// `input` points to a struct of the caller's whose first field (`u32`) is its size, readable for that size.
unsafe fn read_versioned<T: Copy + Default>(input: *const T) -> Result<T, Failure> {
    let caller = (input as *const u32).read() as usize;
    if caller < size_of::<T>() {
        return Err(invalid("struct_size is smaller than this version's struct"));
    }
    Ok(input.read())
}

fn key(key: WsExpertKey) -> ExpertKey {
    ExpertKey {
        layer: key.layer,
        expert: key.expert,
    }
}

#[no_mangle]
pub extern "C" fn ws_abi_version() -> u32 {
    WS_STORE_ABI_VERSION
}

#[no_mangle]
pub extern "C" fn ws_status_string(status: i32) -> *const c_char {
    let text: &'static CStr = match status {
        WS_OK => c"ok",
        WS_ERR_INVALID_ARGUMENT => c"invalid argument",
        WS_ERR_INVALID_HANDLE => c"not an open store",
        WS_ERR_INVALID_LEASE => c"not a live lease of this store",
        WS_ERR_UNAVAILABLE => c"cannot lease now: release leases and retry",
        WS_ERR_IO => c"I/O error",
        WS_ERR_CATALOG => c"catalog error",
        WS_ERR_LEASES_OUTSTANDING => c"leases are alive",
        WS_ERR_CLOSED => c"the store is closed",
        WS_ERR_INTERNAL => c"internal error",
        _ => c"unknown status",
    };
    text.as_ptr()
}

/// # Safety
/// `options` is NULL or points to a `ws_store_options` whose `struct_size` is its size.
#[no_mangle]
pub unsafe extern "C" fn ws_store_options_default(options: *mut WsStoreOptions) -> i32 {
    boundary(std::ptr::null_mut(), 0, || {
        if options.is_null() {
            return Err(invalid("options is NULL"));
        }
        let defaults = StoreOptions::default();
        let value = WsStoreOptions {
            struct_size: size_of::<WsStoreOptions>() as u32,
            direct: defaults.direct as u32,
            host_cache_bytes: 0,
            workers: defaults.workers as u32,
            staging_sets: defaults.staging_sets as u32,
            staging_slots: defaults.staging_slots as u32,
            max_prefetches: defaults.max_prefetches as u32,
            staging_slot_bytes: defaults.staging_slot_bytes,
            cache_block_bytes: defaults.cache_block_bytes,
        };
        // SAFETY: a caller's options struct, sized by its struct_size (checked non-NULL above).
        unsafe { write_versioned(options, &value) }
    })
}

/// # Safety
/// `catalog_path` is a NUL-terminated UTF-8 path; `options` a `ws_store_options` sized by its `struct_size`; `store` a
/// writable `uint64_t`; `error` NULL or `error_size` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn ws_store_open(
    catalog_path: *const c_char,
    options: *const WsStoreOptions,
    store: *mut u64,
    error: *mut c_char,
    error_size: usize,
) -> i32 {
    boundary(error, error_size, || {
        if catalog_path.is_null() || options.is_null() || store.is_null() {
            return Err(invalid("catalog_path, options and store must not be NULL"));
        }
        // SAFETY: non-NULL pointers the caller guarantees valid (the header's contract).
        let (path, options) = unsafe {
            store.write(0);
            (CStr::from_ptr(catalog_path), read_versioned(options)?)
        };
        let path = path.to_str().map_err(|_| invalid("catalog_path is not UTF-8"))?;
        let options = StoreOptions {
            host_cache_bytes: options.host_cache_bytes,
            direct: options.direct != 0,
            workers: options.workers as usize,
            staging_sets: options.staging_sets as usize,
            staging_slots: options.staging_slots as usize,
            staging_slot_bytes: options.staging_slot_bytes,
            max_prefetches: options.max_prefetches as usize,
            cache_block_bytes: options.cache_block_bytes,
        };
        let opened = ExpertStore::open(Path::new(path), options).map_err(failure)?;
        let id = NEXT_STORE.fetch_add(1, Ordering::AcqRel);
        registry().lock().unwrap().insert(
            id,
            Arc::new(Open {
                store: opened,
                closed: RwLock::new(false),
                leases: Mutex::new(HashMap::new()),
            }),
        );
        // SAFETY: checked non-NULL above.
        unsafe { store.write(id) };
        Ok(())
    })
}

/// # Safety
/// `error` is NULL or `error_size` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn ws_store_close(store: u64, error: *mut c_char, error_size: usize) -> i32 {
    boundary(error, error_size, || {
        let open = lookup(store)?;
        let mut closed = open.closed.write().unwrap();
        if *closed {
            return Err((WS_ERR_INVALID_HANDLE, format!("{store} is not an open store")));
        }
        let alive = open.leases.lock().unwrap().len();
        if alive > 0 {
            return Err((
                WS_ERR_LEASES_OUTSTANDING,
                format!("{alive} leases are alive: release them before closing"),
            ));
        }
        *closed = true;
        registry().lock().unwrap().remove(&store);
        open.store.close();
        Ok(())
    })
}

/// # Safety
/// `layers` points to `capacity` writable `uint32_t` (may be NULL when `capacity` is 0); `count` a writable `size_t`.
#[no_mangle]
pub unsafe extern "C" fn ws_store_layers(store: u64, layers: *mut u32, capacity: usize, count: *mut usize) -> i32 {
    boundary(std::ptr::null_mut(), 0, || {
        if count.is_null() || (capacity > 0 && layers.is_null()) {
            return Err(invalid("count must not be NULL, nor layers when capacity is positive"));
        }
        let open = lookup(store)?;
        let ids = open.store.layer_ids();
        // SAFETY: `capacity` slots at `layers` and a writable `count` (the header's contract; checked non-NULL).
        unsafe {
            for (k, &id) in ids.iter().take(capacity).enumerate() {
                layers.add(k).write(id);
            }
            count.write(ids.len());
        }
        Ok(())
    })
}

/// # Safety
/// `out` points to a `ws_view` whose `struct_size` is its size.
#[no_mangle]
pub unsafe extern "C" fn ws_store_view_info(store: u64, layer: u32, view: u32, out: *mut WsView) -> i32 {
    boundary(std::ptr::null_mut(), 0, || {
        if out.is_null() {
            return Err(invalid("out is NULL"));
        }
        let open = lookup(store)?;
        let layer = open.store.layer(layer).ok_or_else(|| invalid("no such layer"))?;
        let v = layer.views.get(view as usize).ok_or_else(|| invalid("no such view"))?;
        let scales = v.scales.as_ref();
        let value = WsView {
            struct_size: size_of::<WsView>() as u32,
            part: v.part as u32,
            name: v.c_name.as_ptr(),
            format: v.c_format.as_ptr(),
            data: std::ptr::null(),
            bytes: v.bytes,
            scale_format: scales.map_or(std::ptr::null(), |s| s.c_format.as_ptr()),
            scales: std::ptr::null(),
            scale_bytes: scales.map_or(0, |s| s.bytes),
            rows: v.rows,
            columns: v.columns,
            block_rows: v.block_rows,
            block_columns: v.block_columns,
        };
        // SAFETY: a caller's view struct, sized by its struct_size (checked non-NULL above).
        unsafe { write_versioned(out, &value) }
    })
}

/// # Safety
/// `experts`, `parts` and `views` are NULL or writable `uint32_t`.
#[no_mangle]
pub unsafe extern "C" fn ws_store_layer_info(
    store: u64,
    layer: u32,
    experts: *mut u32,
    parts: *mut u32,
    views: *mut u32,
) -> i32 {
    boundary(std::ptr::null_mut(), 0, || {
        let open = lookup(store)?;
        let layer = open.store.layer(layer).ok_or_else(|| invalid("no such layer"))?;
        // SAFETY: each is NULL or a writable u32 (the header's contract).
        unsafe {
            for (out, value) in [
                (experts, layer.experts),
                (parts, layer.parts.len() as u32),
                (views, layer.views.len() as u32),
            ] {
                if !out.is_null() {
                    out.write(value);
                }
            }
        }
        Ok(())
    })
}

/// # Safety
/// `name` is a NUL-terminated string; `index` a writable `uint32_t`.
#[no_mangle]
pub unsafe extern "C" fn ws_store_view_index(store: u64, layer: u32, name: *const c_char, index: *mut u32) -> i32 {
    boundary(std::ptr::null_mut(), 0, || {
        if name.is_null() || index.is_null() {
            return Err(invalid("name and index must not be NULL"));
        }
        let open = lookup(store)?;
        let layer = open.store.layer(layer).ok_or_else(|| invalid("no such layer"))?;
        // SAFETY: a NUL-terminated string (the header's contract).
        let name = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|_| invalid("not UTF-8"))?;
        let found = layer.view_index(name).ok_or_else(|| invalid("no such view"))?;
        // SAFETY: checked non-NULL above.
        unsafe { index.write(found as u32) };
        Ok(())
    })
}

/// # Safety
/// `lease` is a writable `uint64_t`; `error` NULL or `error_size` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn ws_store_acquire(
    store: u64,
    key: WsExpertKey,
    lease: *mut u64,
    error: *mut c_char,
    error_size: usize,
) -> i32 {
    // SAFETY: forwarded under the same contract: one key, one lease slot.
    unsafe { ws_store_acquire_many(store, &key, 1, lease, error, error_size) }
}

/// # Safety
/// `keys` points to `count` keys and `leases` to `count` writable `uint64_t` (both may be NULL when `count` is 0);
/// `error` NULL or `error_size` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn ws_store_acquire_many(
    store: u64,
    keys: *const WsExpertKey,
    count: usize,
    leases: *mut u64,
    error: *mut c_char,
    error_size: usize,
) -> i32 {
    boundary(error, error_size, || {
        if count == 0 {
            return Ok(());
        }
        if keys.is_null() || leases.is_null() {
            return Err(invalid("keys and leases must not be NULL"));
        }
        // SAFETY: `count` keys and `count` lease slots (the header's contract).
        let (keys, out) = unsafe {
            let out = std::slice::from_raw_parts_mut(leases, count);
            out.fill(0);
            (std::slice::from_raw_parts(keys, count), out)
        };
        let keys: Vec<ExpertKey> = keys.iter().map(|&k| key(k)).collect();
        let open = lookup(store)?;
        let closed = open.closed.read().unwrap();
        if *closed {
            return Err(failure(StoreError::Closed));
        }
        let acquired = open.store.acquire_many(&keys).map_err(failure)?;
        let mut table = open.leases.lock().unwrap();
        for (slot, lease) in out.iter_mut().zip(acquired) {
            let id = NEXT_LEASE.fetch_add(1, Ordering::AcqRel);
            table.insert(id, lease);
            *slot = id;
        }
        drop(table);
        drop(closed);
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn ws_store_release(store: u64, lease: u64) -> i32 {
    boundary(std::ptr::null_mut(), 0, || {
        let open = lookup(store)?;
        let removed = open.leases.lock().unwrap().remove(&lease);
        match removed {
            Some(lease) => {
                drop(lease);
                Ok(())
            }
            None => Err((WS_ERR_INVALID_LEASE, format!("{lease} is not a live lease"))),
        }
    })
}

/// Run `body` on a live lease of an open store.
fn with_lease(store: u64, lease: u64, body: impl FnOnce(&Lease) -> Result<(), Failure>) -> Result<(), Failure> {
    let open = lookup(store)?;
    let table = open.leases.lock().unwrap();
    let lease = table
        .get(&lease)
        .ok_or((WS_ERR_INVALID_LEASE, format!("{lease} is not a live lease")))?;
    body(lease)
}

/// # Safety
/// `key` is a writable `ws_expert_key`.
#[no_mangle]
pub unsafe extern "C" fn ws_lease_key(store: u64, lease: u64, key: *mut WsExpertKey) -> i32 {
    boundary(std::ptr::null_mut(), 0, || {
        if key.is_null() {
            return Err(invalid("key is NULL"));
        }
        with_lease(store, lease, |lease| {
            let k = lease.key();
            // SAFETY: checked non-NULL above.
            unsafe {
                key.write(WsExpertKey {
                    layer: k.layer,
                    expert: k.expert,
                })
            };
            Ok(())
        })
    })
}

/// # Safety
/// `span` is a writable `ws_span`.
#[no_mangle]
pub unsafe extern "C" fn ws_lease_part(store: u64, lease: u64, part: u32, span: *mut WsSpan) -> i32 {
    boundary(std::ptr::null_mut(), 0, || {
        if span.is_null() {
            return Err(invalid("span is NULL"));
        }
        with_lease(store, lease, |lease| {
            if part as usize >= lease.parts() {
                return Err(invalid("no such part"));
            }
            let bytes = lease.part(part as usize);
            // SAFETY: checked non-NULL above. The bytes stay valid until the lease's release (the cache keeps the
            // leased entry, which the lease table holds).
            unsafe {
                span.write(WsSpan {
                    data: bytes.as_ptr(),
                    bytes: bytes.len() as u64,
                })
            };
            Ok(())
        })
    })
}

/// # Safety
/// `out` points to a `ws_view` whose `struct_size` is its size.
#[no_mangle]
pub unsafe extern "C" fn ws_lease_view(store: u64, lease: u64, view: u32, out: *mut WsView) -> i32 {
    boundary(std::ptr::null_mut(), 0, || {
        if out.is_null() {
            return Err(invalid("out is NULL"));
        }
        with_lease(store, lease, |lease| {
            let resolved = lease.view(view as usize).ok_or_else(|| invalid("no such view"))?;
            let v = resolved.view;
            let scales = v.scales.as_ref();
            let value = WsView {
                struct_size: size_of::<WsView>() as u32,
                part: v.part as u32,
                name: v.c_name.as_ptr(),
                format: v.c_format.as_ptr(),
                data: resolved.data.as_ptr(),
                bytes: resolved.data.len() as u64,
                scale_format: scales.map_or(std::ptr::null(), |s| s.c_format.as_ptr()),
                scales: resolved.scales.map_or(std::ptr::null(), |s| s.as_ptr()),
                scale_bytes: resolved.scales.map_or(0, |s| s.len() as u64),
                rows: v.rows,
                columns: v.columns,
                block_rows: v.block_rows,
                block_columns: v.block_columns,
            };
            // SAFETY: a caller's view struct, sized by its struct_size (checked non-NULL above).
            unsafe { write_versioned(out, &value) }
        })
    })
}

/// # Safety
/// `keys` points to `count` keys (may be NULL when `count` is 0); `rows` is NULL or a writable `uint64_t`.
#[no_mangle]
pub unsafe extern "C" fn ws_store_prefetch(store: u64, keys: *const WsExpertKey, count: usize, rows: *mut u64) -> i32 {
    boundary(std::ptr::null_mut(), 0, || {
        if count > 0 && keys.is_null() {
            return Err(invalid("keys is NULL"));
        }
        let keys: Vec<ExpertKey> = if count == 0 {
            Vec::new()
        } else {
            // SAFETY: `count` keys (the header's contract).
            unsafe { std::slice::from_raw_parts(keys, count) }
                .iter()
                .map(|&k| key(k))
                .collect()
        };
        let open = lookup(store)?;
        let closed = open.closed.read().unwrap();
        if *closed {
            return Err(failure(StoreError::Closed));
        }
        let admitted = open.store.prefetch(&keys).map_err(failure)?;
        drop(closed);
        if !rows.is_null() {
            // SAFETY: NULL or a writable u64 (the header's contract).
            unsafe { rows.write(admitted) };
        }
        Ok(())
    })
}

/// # Safety
/// `stats` points to a `ws_store_stats` whose `struct_size` is its size.
#[no_mangle]
pub unsafe extern "C" fn ws_store_get_stats(store: u64, stats: *mut WsStoreStats) -> i32 {
    boundary(std::ptr::null_mut(), 0, || {
        if stats.is_null() {
            return Err(invalid("stats is NULL"));
        }
        let open = lookup(store)?;
        let s = open.store.stats();
        let value = WsStoreStats {
            struct_size: size_of::<WsStoreStats>() as u32,
            abi_version: WS_STORE_ABI_VERSION,
            acquire_calls: s.acquire_calls,
            acquire_failures: s.acquire_failures,
            experts: s.experts,
            expert_hits: s.expert_hits,
            expert_waits: s.expert_waits,
            expert_misses: s.expert_misses,
            releases: s.releases,
            active_leases: s.active_leases,
            leased_bytes: s.leased_bytes,
            peak_leased_bytes: s.peak_leased_bytes,
            prefetch_calls: s.prefetch_calls,
            prefetch_refused: s.prefetch_refused,
            prefetch_experts: s.prefetch_experts,
            prefetch_rows: s.prefetch_rows,
            prefetch_used: s.cache.prefetch_used,
            prefetch_wasted: s.cache.prefetch_wasted,
            requests: s.io.requests,
            rows: s.io.rows,
            logical_bytes: s.io.logical_bytes,
            physical_bytes: s.io.physical_bytes,
            read_calls: s.io.read_calls,
            extents: s.io.extents,
            blocks_4k: s.io.blocks_4k,
            read_ns: s.io.read_ns,
            busy_ns: s.io.busy_ns,
            cache_lookups: s.cache.lookups,
            cache_hits: s.cache.hits,
            cache_waits: s.cache.waits,
            cache_misses: s.cache.misses,
            cache_inserts: s.cache.inserts,
            cache_evictions: s.cache.evictions,
            cache_bypassed: s.cache.bypassed,
            cache_capacity_bytes: s.cache_capacity_bytes,
            cache_block_bytes: s.cache_block_bytes,
            cache_resident_bytes: s.cache_resident_bytes,
            cache_held_bytes: s.cache_held_bytes,
            cache_peak_held_bytes: s.cache_peak_held_bytes,
            staging_bytes: s.staging_bytes,
        };
        // SAFETY: a caller's stats struct, sized by its struct_size (checked non-NULL above).
        unsafe { write_versioned(stats, &value) }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versioned_writes_fill_what_both_know_and_zero_the_rest() {
        // A caller compiled against a larger (future) struct: our fields, then zeros; struct_size says what was filled.
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct Larger {
            stats: WsStoreStats,
            future: [u64; 4],
        }
        let mut out = Larger {
            stats: WsStoreStats::default(),
            future: [7; 4],
        };
        out.stats.struct_size = size_of::<Larger>() as u32;
        let value = WsStoreStats {
            struct_size: size_of::<WsStoreStats>() as u32,
            abi_version: 1,
            experts: 5,
            staging_bytes: 9,
            ..WsStoreStats::default()
        };
        unsafe { write_versioned(&mut out as *mut Larger as *mut WsStoreStats, &value) }.unwrap();
        assert_eq!(out.stats.struct_size as usize, size_of::<WsStoreStats>());
        assert_eq!((out.stats.experts, out.stats.staging_bytes), (5, 9));
        assert_eq!(out.future, [0; 4]);
        // A caller compiled against a smaller (older) struct: only its bytes are written.
        let mut small = [0xAAu8; 16];
        small[..4].copy_from_slice(&12u32.to_ne_bytes());
        unsafe { write_versioned(small.as_mut_ptr() as *mut WsStoreStats, &value) }.unwrap();
        assert_eq!(u32::from_ne_bytes(small[..4].try_into().unwrap()), 12);
        assert_eq!(&small[12..], &[0xAA; 4], "nothing past the caller's struct");
        // Too small to hold the header's first two fields.
        let mut tiny = [4u8, 0, 0, 0, 0, 0, 0, 0];
        assert!(unsafe { write_versioned(tiny.as_mut_ptr() as *mut WsStoreStats, &value) }.is_err());
    }

    #[test]
    fn statuses_have_descriptions() {
        for status in [0, -1, -2, -3, -4, -5, -6, -7, -8, -9, 42] {
            let text = unsafe { CStr::from_ptr(ws_status_string(status)) };
            assert!(!text.to_bytes().is_empty());
        }
        assert_eq!(ws_abi_version(), WS_STORE_ABI_VERSION);
    }

    #[test]
    fn unknown_handles_are_errors_not_crashes() {
        assert_eq!(ws_store_release(987_654, 1), WS_ERR_INVALID_HANDLE);
        let mut span = WsSpan {
            data: std::ptr::null(),
            bytes: 0,
        };
        assert_eq!(
            unsafe { ws_lease_part(987_654, 1, 0, &mut span) },
            WS_ERR_INVALID_HANDLE
        );
        let mut error = [0 as c_char; 64];
        assert_eq!(
            unsafe { ws_store_close(987_654, error.as_mut_ptr(), error.len()) },
            WS_ERR_INVALID_HANDLE
        );
        let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_str().unwrap();
        assert!(message.contains("not an open store"), "{message}");
        assert_eq!(
            unsafe { ws_store_options_default(std::ptr::null_mut()) },
            WS_ERR_INVALID_ARGUMENT
        );
    }
}
