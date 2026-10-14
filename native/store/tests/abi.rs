//! The store's C ABI: the header's layout equals the Rust structs'; a C client runs the whole lifecycle; misuse (double
//! release, foreign or stale leases, closed handles, NULL pointers, short structs) is an error code, never a crash; many
//! threads acquire and release through it at once.

mod common;

use std::ffi::{c_char, CStr, CString};
use std::mem::{offset_of, size_of};
use std::sync::Arc;

use common::{fnv1a, Fixture, Lcg, EXPERTS0};
use weightsift_store::ffi::*;

extern "C" {
    fn ws_abi_client_layout(out: *mut u64, count: usize);
    fn ws_abi_client_lifecycle(
        catalog: *const c_char,
        cache_bytes: u64,
        layer: u32,
        expert: u32,
        checksums: *mut u64,
        checksum_count: usize,
        message: *mut c_char,
        message_size: usize,
    ) -> i32;
}

fn options(cache: u64) -> WsStoreOptions {
    let mut options = WsStoreOptions {
        struct_size: size_of::<WsStoreOptions>() as u32,
        ..WsStoreOptions::default()
    };
    assert_eq!(unsafe { ws_store_options_default(&mut options) }, WS_OK);
    options.host_cache_bytes = cache;
    options.staging_slot_bytes = 64 << 10;
    options.workers = 4;
    options
}

fn open(fixture: &Fixture, cache: u64) -> u64 {
    let path = CString::new(fixture.catalog_path().to_str().unwrap()).unwrap();
    let mut store = 0;
    let mut error = [0 as c_char; 256];
    let status = unsafe {
        ws_store_open(
            path.as_ptr(),
            &options(cache),
            &mut store,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    assert_eq!(status, WS_OK, "{:?}", unsafe { CStr::from_ptr(error.as_ptr()) });
    store
}

fn acquire(store: u64, layer: u32, expert: u32) -> Result<u64, i32> {
    let mut lease = 0;
    let status = unsafe {
        ws_store_acquire(
            store,
            WsExpertKey { layer, expert },
            &mut lease,
            std::ptr::null_mut(),
            0,
        )
    };
    if status == WS_OK {
        Ok(lease)
    } else {
        Err(status)
    }
}

fn part(store: u64, lease: u64, part: u32) -> &'static [u8] {
    let mut span = WsSpan {
        data: std::ptr::null(),
        bytes: 0,
    };
    assert_eq!(unsafe { ws_lease_part(store, lease, part, &mut span) }, WS_OK);
    // The test keeps the lease alive while it reads the slice.
    unsafe { std::slice::from_raw_parts(span.data, span.bytes as usize) }
}

fn close(store: u64) -> i32 {
    unsafe { ws_store_close(store, std::ptr::null_mut(), 0) }
}

#[test]
fn the_c_header_matches_the_rust_layout() {
    let mut c = [0u64; 13];
    unsafe { ws_abi_client_layout(c.as_mut_ptr(), c.len()) };
    let rust = [
        size_of::<WsStoreOptions>(),
        size_of::<WsExpertKey>(),
        size_of::<WsSpan>(),
        size_of::<WsView>(),
        size_of::<WsStoreStats>(),
        offset_of!(WsStoreOptions, staging_slot_bytes),
        offset_of!(WsStoreOptions, cache_block_bytes),
        offset_of!(WsView, data),
        offset_of!(WsView, scale_bytes),
        offset_of!(WsView, block_columns),
        offset_of!(WsStoreStats, prefetch_used),
        offset_of!(WsStoreStats, cache_lookups),
        offset_of!(WsStoreStats, staging_bytes),
    ];
    assert_eq!(c.map(|v| v as usize), rust);
}

#[test]
fn a_c_client_runs_the_whole_lifecycle() {
    let fixture = Fixture::new("c-client");
    let path = CString::new(fixture.catalog_path().to_str().unwrap()).unwrap();
    for (layer, expert) in [(0u32, 2u32), (7, 3)] {
        let expected = fixture.expert(layer, expert);
        let mut checksums = vec![0u64; expected.len()];
        let mut message = [0 as c_char; 256];
        let line = unsafe {
            ws_abi_client_lifecycle(
                path.as_ptr(),
                16 << 20,
                layer,
                expert,
                checksums.as_mut_ptr(),
                checksums.len(),
                message.as_mut_ptr(),
                message.len(),
            )
        };
        assert_eq!(line, 0, "{:?}", unsafe { CStr::from_ptr(message.as_ptr()) });
        let wanted: Vec<u64> = expected.iter().map(|p| fnv1a(p)).collect();
        assert_eq!(
            checksums, wanted,
            "layer {layer} expert {expert}: the C client read the file's bytes"
        );
    }
}

#[test]
fn misuse_is_an_error_code_never_a_crash() {
    let fixture = Fixture::new("misuse");
    let (a, b) = (open(&fixture, 16 << 20), open(&fixture, 16 << 20));
    assert_ne!(a, b);
    let lease = acquire(a, 0, 1).unwrap();
    // Another store's lease, a released lease, lease 0.
    assert_eq!(ws_store_release(b, lease), WS_ERR_INVALID_LEASE);
    assert_eq!(close(a), WS_ERR_LEASES_OUTSTANDING);
    assert_eq!(part(a, lease, 1), &fixture.expert(0, 1)[1][..]);
    assert_eq!(ws_store_release(a, lease), WS_OK);
    assert_eq!(ws_store_release(a, lease), WS_ERR_INVALID_LEASE);
    assert_eq!(ws_store_release(a, 0), WS_ERR_INVALID_LEASE);
    // Unknown layers, experts, parts and views.
    assert_eq!(acquire(a, 1, 0), Err(WS_ERR_INVALID_ARGUMENT));
    assert_eq!(acquire(a, 0, EXPERTS0), Err(WS_ERR_INVALID_ARGUMENT));
    let lease = acquire(a, 7, 0).unwrap();
    let mut span = WsSpan {
        data: std::ptr::null(),
        bytes: 0,
    };
    assert_eq!(
        unsafe { ws_lease_part(a, lease, 1, &mut span) },
        WS_ERR_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe { ws_lease_part(a, lease, 0, std::ptr::null_mut()) },
        WS_ERR_INVALID_ARGUMENT
    );
    // A view struct too short for this version's first fields, then a full one.
    let mut short = [0u8; size_of::<WsView>()];
    short[..4].copy_from_slice(&4u32.to_ne_bytes());
    assert_eq!(
        unsafe { ws_lease_view(a, lease, 0, short.as_mut_ptr() as *mut WsView) },
        WS_ERR_INVALID_ARGUMENT
    );
    let mut view: WsView = unsafe { std::mem::zeroed() };
    view.struct_size = size_of::<WsView>() as u32;
    assert_eq!(unsafe { ws_lease_view(a, lease, 2, &mut view) }, WS_OK);
    assert_eq!(unsafe { CStr::from_ptr(view.name) }.to_str().unwrap(), "down");
    assert_eq!(
        unsafe { CStr::from_ptr(view.format) }.to_str().unwrap(),
        "fp4_native_block"
    );
    assert!(view.scales.is_null() && view.scale_format.is_null() && view.scale_bytes == 0);
    assert_eq!(
        unsafe { ws_lease_view(a, lease, 3, &mut view) },
        WS_ERR_INVALID_ARGUMENT
    );
    // NULL pointers.
    let mut store = 0;
    assert_eq!(
        unsafe { ws_store_open(std::ptr::null(), &options(1 << 20), &mut store, std::ptr::null_mut(), 0) },
        WS_ERR_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe { ws_store_acquire_many(a, std::ptr::null(), 2, std::ptr::null_mut(), std::ptr::null_mut(), 0) },
        WS_ERR_INVALID_ARGUMENT
    );
    // An options struct from an older, shorter header is refused.
    let mut old = options(1 << 20);
    old.struct_size = 16;
    let path = CString::new(fixture.catalog_path().to_str().unwrap()).unwrap();
    assert_eq!(
        unsafe { ws_store_open(path.as_ptr(), &old, &mut store, std::ptr::null_mut(), 0) },
        WS_ERR_INVALID_ARGUMENT
    );
    // A catalog that does not exist.
    let missing = CString::new(fixture.dir.join("nope.json").to_str().unwrap()).unwrap();
    let mut error = [0 as c_char; 128];
    assert_eq!(
        unsafe {
            ws_store_open(
                missing.as_ptr(),
                &options(1 << 20),
                &mut store,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        WS_ERR_CATALOG
    );
    assert!(unsafe { CStr::from_ptr(error.as_ptr()) }
        .to_str()
        .unwrap()
        .contains("catalog"));
    // Closing: refused with a lease, then done; the handle is dead everywhere.
    assert_eq!(close(a), WS_ERR_LEASES_OUTSTANDING);
    assert_eq!(ws_store_release(a, lease), WS_OK);
    assert_eq!(close(a), WS_OK);
    assert_eq!(close(a), WS_ERR_INVALID_HANDLE);
    assert_eq!(acquire(a, 0, 0), Err(WS_ERR_INVALID_HANDLE));
    // The other store is untouched.
    let lease = acquire(b, 0, 5).unwrap();
    assert_eq!(part(b, lease, 0), &fixture.expert(0, 5)[0][..]);
    assert_eq!(ws_store_release(b, lease), WS_OK);
    assert_eq!(close(b), WS_OK);
}

#[test]
fn many_threads_acquire_and_release_through_the_abi() {
    let fixture = Arc::new(Fixture::new("abi-threads"));
    let expert = fixture.expert_bytes(0);
    let store = open(&fixture, 4 * expert);
    std::thread::scope(|scope| {
        for t in 0..8u64 {
            let fixture = Arc::clone(&fixture);
            scope.spawn(move || {
                let mut rng = Lcg(900 + t);
                let mut held: Vec<(u64, u32, u32)> = Vec::new();
                for _ in 0..50 {
                    let keys = fixture.keys();
                    let (layer, e) = keys[rng.below(keys.len() as u64) as usize];
                    match acquire(store, layer, e) {
                        Ok(lease) => {
                            assert_eq!(part(store, lease, 0), &fixture.expert(layer, e)[0][..]);
                            held.push((lease, layer, e));
                        }
                        Err(WS_ERR_UNAVAILABLE) => {
                            for (lease, _, _) in held.drain(..) {
                                assert_eq!(ws_store_release(store, lease), WS_OK);
                            }
                        }
                        Err(status) => panic!("status {status}"),
                    }
                    if held.len() > 1 {
                        let (lease, layer, e) = held.remove(0);
                        // Still the file's bytes just before the release.
                        assert_eq!(part(store, lease, 0), &fixture.expert(layer, e)[0][..]);
                        assert_eq!(ws_store_release(store, lease), WS_OK);
                        assert_eq!(ws_store_release(store, lease), WS_ERR_INVALID_LEASE);
                    }
                }
                for (lease, _, _) in held {
                    assert_eq!(ws_store_release(store, lease), WS_OK);
                }
            });
        }
    });
    let mut stats = WsStoreStats {
        struct_size: size_of::<WsStoreStats>() as u32,
        ..WsStoreStats::default()
    };
    assert_eq!(unsafe { ws_store_get_stats(store, &mut stats) }, WS_OK);
    assert_eq!((stats.active_leases, stats.leased_bytes), (0, 0));
    assert_eq!(stats.experts, stats.releases);
    assert!(stats.cache_peak_held_bytes <= stats.cache_capacity_bytes);
    assert_eq!(close(store), WS_OK);
}
