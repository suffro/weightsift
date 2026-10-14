//! The expert store against files on disk: every byte a lease exposes equals the file's and the reference read path's;
//! leases pin their rows; budgets hold; concurrent callers load each row once; errors hold nothing.

mod common;

use std::sync::Arc;

use common::{Fixture, Lcg, EXPERTS0, HALF, SCALES};
use weightsift_io::engine::{Engine, EngineConfig, Request};
use weightsift_io::{FileTable, RawBuffer, Segment};
use weightsift_store::{ExpertKey, ExpertStore, StoreError, StoreOptions};

fn options(cache: u64) -> StoreOptions {
    StoreOptions {
        host_cache_bytes: cache,
        workers: 4,
        staging_slot_bytes: 64 << 10,
        ..StoreOptions::default()
    }
}

fn key(layer: u32, expert: u32) -> ExpertKey {
    ExpertKey { layer, expert }
}

/// The reference read path: the Phase 6A engine's uncached `read_rows` (direct reads, plans equal to Python's).
fn reference_rows(fixture: &Fixture, segment: usize, rows: &[i64]) -> Vec<u8> {
    let catalog = &fixture.catalog;
    let paths = catalog
        .files
        .iter()
        .enumerate()
        .map(|(k, f)| (f.key.clone(), catalog.file_path(k, &fixture.dir)))
        .collect();
    let segments = catalog
        .segments
        .iter()
        .map(|s| match (&s.offset, &s.part_bytes, &s.spans) {
            (Some(offset), _, _) => Segment::plain(&s.name, s.files[0], *offset, s.rows, s.row_bytes).unwrap(),
            (None, Some(p), Some(spans)) => {
                Segment::composed(&s.name, s.files.clone(), s.rows, p.clone(), spans.clone()).unwrap()
            }
            _ => unreachable!(),
        })
        .collect();
    let engine = Engine::new(FileTable::open(paths, true).unwrap(), segments, EngineConfig::default()).unwrap();
    let request = Request {
        segment: segment as u32,
        rows: Some(rows.to_vec()),
        positions: None,
    };
    let mut out = vec![0u8; rows.len() * catalog.segments[segment].row_bytes as usize];
    engine.read_rows(&request, RawBuffer::from_slice(&mut out)).unwrap();
    out
}

#[test]
fn leases_serve_the_bytes_on_disk_and_the_reference_read_path() {
    let fixture = Fixture::new("exact");
    let store = ExpertStore::open(&fixture.catalog_path(), options(16 << 20)).unwrap();
    for pass in 0..2 {
        for (layer, expert) in fixture.keys() {
            let lease = store.acquire(key(layer, expert)).unwrap();
            let expected = fixture.expert(layer, expert);
            assert_eq!(lease.parts(), expected.len());
            for (p, bytes) in expected.iter().enumerate() {
                assert_eq!(
                    lease.part(p),
                    &bytes[..],
                    "layer {layer} expert {expert} part {p} pass {pass}"
                );
            }
            // Views are ranges of the parts, in place.
            let layer_info = store.layer(layer).unwrap();
            for v in 0..layer_info.views.len() {
                let view = lease.view(v).unwrap();
                let part = lease.part(view.view.part);
                let start = view.data.as_ptr() as usize - part.as_ptr() as usize;
                assert_eq!(start as u64, view.view.offset);
                assert_eq!(view.data.len() as u64, view.view.bytes);
            }
            assert!(lease.view(layer_info.views.len()).is_none());
        }
        let stats = store.stats();
        let n = fixture.keys().len() as u64;
        if pass == 0 {
            assert_eq!((stats.experts, stats.expert_misses, stats.expert_hits), (n, n, 0));
        } else {
            assert_eq!((stats.experts, stats.expert_misses, stats.expert_hits), (2 * n, n, n));
        }
    }
    // The same bytes through the reference read path (no cache, no leases).
    let gate_up = reference_rows(&fixture, 0, &[0, 3, 5]);
    for (k, &expert) in [0u32, 3, 5].iter().enumerate() {
        let lease = store.acquire(key(0, expert)).unwrap();
        assert_eq!(
            lease.part(0),
            &gate_up[k * 2 * HALF as usize..(k + 1) * 2 * HALF as usize]
        );
        let gate = lease.view(0).unwrap();
        assert_eq!(gate.data, &lease.part(0)[..HALF as usize]);
        assert_eq!(gate.scales.unwrap(), &lease.part(2)[..SCALES as usize]);
    }
    let composed = reference_rows(&fixture, 3, &[1, 2]);
    let row = fixture.catalog.segments[3].row_bytes as usize;
    let lease = store.acquire(key(7, 2)).unwrap();
    assert_eq!(lease.part(0), &composed[row..2 * row]);
    assert!(lease.view(0).unwrap().scales.is_none());
    // Reads: every row read once (the second pass and the reference checks were hits).
    let stats = store.stats();
    let rows: u64 = 3 * EXPERTS0 as u64 + 4;
    assert_eq!(stats.cache.inserts, rows);
    assert_eq!(stats.io.acquires, stats.acquire_calls);
    assert!(stats.cache_peak_held_bytes <= stats.cache_capacity_bytes);
}

#[test]
fn leased_experts_are_never_evicted_and_a_full_cache_refuses() {
    let fixture = Fixture::new("full");
    let expert = fixture.expert_bytes(0);
    // Room for two experts of layer 0 and a bit.
    let store = ExpertStore::open(&fixture.catalog_path(), options(2 * expert + expert / 2)).unwrap();
    let first = store.acquire(key(0, 0)).unwrap();
    let second = store.acquire(key(0, 1)).unwrap();
    let error = store.acquire(key(0, 2)).err().unwrap();
    assert!(matches!(error, StoreError::Unavailable(_)), "{error:?}");
    let stats = store.stats();
    assert_eq!((stats.acquire_failures, stats.active_leases), (1, 2));
    assert_eq!(stats.leased_bytes, 2 * expert);
    assert!(stats.cache_resident_bytes <= stats.cache_capacity_bytes);
    // The leased bytes did not move or change.
    assert_eq!(first.part(0), &fixture.expert(0, 0)[0][..]);
    drop(first);
    let third = store.acquire(key(0, 2)).unwrap();
    assert_eq!(third.part(1), &fixture.expert(0, 2)[1][..]);
    assert_eq!(second.part(1), &fixture.expert(0, 1)[1][..]);
    // Eviction is per row, least recently used first: the first expert's gate/up row made the room the third needed;
    // its other rows stay cached.
    let stats = store.stats();
    assert_eq!((stats.releases, stats.active_leases), (1, 2));
    assert_eq!((stats.cache.evictions, stats.cache.bypassed), (1, 1), "{stats:?}");
    assert!(stats.cache_peak_held_bytes <= stats.cache_capacity_bytes);
    assert!(stats.peak_leased_bytes <= stats.cache_capacity_bytes);
    let again = store.acquire(key(0, 0)).err().unwrap();
    assert!(
        matches!(again, StoreError::Unavailable(_)),
        "its gate/up row needs room the leases hold"
    );
}

#[test]
fn acquire_many_is_all_or_nothing() {
    let fixture = Fixture::new("many");
    let expert = fixture.expert_bytes(0);
    let store = ExpertStore::open(&fixture.catalog_path(), options(3 * expert)).unwrap();
    let keys: Vec<ExpertKey> = (0..3).map(|e| key(0, e)).collect();
    let leases = store.acquire_many(&keys).unwrap();
    assert_eq!(leases.len(), 3);
    for (lease, e) in leases.iter().zip(0..) {
        assert_eq!(lease.key(), key(0, e));
        assert_eq!(lease.part(0), &fixture.expert(0, e)[0][..]);
    }
    // Four experts do not fit beside three leases: nothing more is held after the failure.
    let more: Vec<ExpertKey> = (3..6).map(|e| key(0, e)).collect();
    assert!(matches!(store.acquire_many(&more), Err(StoreError::Unavailable(_))));
    assert_eq!(store.active_leases(), 3);
    drop(leases);
    assert_eq!(store.acquire_many(&more).unwrap().len(), 3);
    assert!(store.acquire_many(&[]).unwrap().is_empty());
}

#[test]
fn concurrent_callers_load_each_row_once_and_stay_within_the_budget() {
    let fixture = Arc::new(Fixture::new("concurrent"));
    let expert = fixture.expert_bytes(0);
    // Room for about half of layer 0's experts: evictions happen, and some acquires find no room.
    let store = Arc::new(ExpertStore::open(&fixture.catalog_path(), options(4 * expert)).unwrap());
    std::thread::scope(|scope| {
        for t in 0..8u64 {
            let (store, fixture) = (Arc::clone(&store), Arc::clone(&fixture));
            scope.spawn(move || {
                let mut rng = Lcg(500 + t);
                let mut held = Vec::new();
                for _ in 0..60 {
                    let keys = fixture.keys();
                    let (layer, e) = keys[rng.below(keys.len() as u64) as usize];
                    match store.acquire(key(layer, e)) {
                        Ok(lease) => {
                            assert_eq!(lease.part(0), &fixture.expert(layer, e)[0][..]);
                            held.push(lease);
                        }
                        // No room beside the other threads' leases: release ours and go on.
                        Err(StoreError::Unavailable(_)) => held.clear(),
                        Err(error) => panic!("{error:?}"),
                    }
                    if !held.is_empty() && (held.len() > 1 || rng.below(2) == 0) {
                        held.remove(0);
                    }
                    let stats = store.stats();
                    assert!(stats.cache_held_bytes <= stats.cache_capacity_bytes);
                }
            });
        }
    });
    let stats = store.stats();
    assert_eq!(stats.active_leases, 0);
    assert_eq!(stats.leased_bytes, 0);
    assert_eq!(stats.releases, stats.experts);
    assert_eq!(
        stats.experts,
        stats.expert_hits + stats.expert_waits + stats.expert_misses
    );
    assert_eq!(
        stats.cache.lookups,
        stats.cache.hits + stats.cache.waits + stats.cache.misses
    );
    // A call that finds no room for a later part gives back the rows it admitted for the earlier ones (aborted before
    // any read); only failed calls abort. What stays is exactly what was inserted and not evicted.
    assert!(
        stats.cache.aborted_fills == 0 || stats.acquire_failures > 0,
        "{stats:?}"
    );
    assert_eq!(
        stats.cache_resident_bytes,
        stats.cache.insert_bytes - stats.cache.evicted_bytes
    );
    assert!(stats.cache_peak_held_bytes <= stats.cache_capacity_bytes);
}

#[test]
fn prefetched_experts_are_waited_for_or_hit() {
    let fixture = Fixture::new("prefetch");
    let store = ExpertStore::open(&fixture.catalog_path(), options(16 << 20)).unwrap();
    let keys: Vec<ExpertKey> = (0..EXPERTS0).map(|e| key(0, e)).collect();
    let rows = store.prefetch(&keys).unwrap();
    assert_eq!(rows, 3 * EXPERTS0 as u64);
    for &k in &keys {
        let lease = store.acquire(k).unwrap();
        assert_eq!(lease.part(1), &fixture.expert(0, k.expert)[1][..]);
    }
    store.wait_prefetches().unwrap();
    let stats = store.stats();
    assert_eq!(stats.expert_misses, 0, "{stats:?}");
    assert_eq!(stats.cache.prefetch_used, 3 * EXPERTS0 as u64);
    assert_eq!(stats.prefetch_rows, 3 * EXPERTS0 as u64);
    // Cached already: nothing to load.
    assert_eq!(store.prefetch(&keys).unwrap(), 0);
    // A store that allows no prefetch refuses them all (advisory: no error).
    let none = ExpertStore::open(
        &fixture.catalog_path(),
        StoreOptions {
            max_prefetches: 0,
            ..options(16 << 20)
        },
    )
    .unwrap();
    assert!(matches!(none.prefetch(&keys), Err(StoreError::Unavailable(_))));
    let stats = none.stats();
    assert_eq!((stats.prefetch_calls, stats.prefetch_refused), (1, 1));
}

#[test]
fn bad_catalogs_options_and_keys_are_refused() {
    let fixture = Fixture::new("refuse");
    let path = fixture.catalog_path();
    assert!(matches!(
        ExpertStore::open(&path, options(0)),
        Err(StoreError::Invalid(_))
    ));
    let small_block = StoreOptions {
        cache_block_bytes: HALF,
        ..options(16 << 20)
    };
    assert!(matches!(
        ExpertStore::open(&path, small_block),
        Err(StoreError::Invalid(_))
    ));
    let odd_slots = StoreOptions {
        staging_slot_bytes: 5000,
        ..options(16 << 20)
    };
    assert!(matches!(
        ExpertStore::open(&path, odd_slots),
        Err(StoreError::Invalid(_))
    ));
    // A file whose size differs from the catalog's.
    let mut catalog = fixture.catalog.clone();
    catalog.files[1].size = Some(1);
    assert!(matches!(
        ExpertStore::from_catalog(&catalog, &fixture.dir, options(16 << 20)),
        Err(StoreError::Catalog(_))
    ));
    // A missing file.
    let mut catalog = fixture.catalog.clone();
    catalog.files[0].path = "missing.bin".into();
    assert!(matches!(
        ExpertStore::from_catalog(&catalog, &fixture.dir, options(16 << 20)),
        Err(StoreError::Io(_))
    ));
    // A malformed catalog file.
    let broken = fixture.dir.join("broken.json");
    std::fs::write(&broken, "{").unwrap();
    assert!(matches!(
        ExpertStore::open(&broken, options(1 << 20)),
        Err(StoreError::Catalog(_))
    ));
    let store = ExpertStore::open(&path, options(16 << 20)).unwrap();
    assert!(matches!(store.acquire(key(3, 0)), Err(StoreError::Invalid(_))));
    assert!(matches!(store.acquire(key(0, EXPERTS0)), Err(StoreError::Invalid(_))));
    assert_eq!(store.stats().io.physical_bytes, 0);
}

#[test]
fn a_closed_store_leases_nothing_and_its_leases_stay_valid() {
    let fixture = Fixture::new("close");
    let store = ExpertStore::open(&fixture.catalog_path(), options(16 << 20)).unwrap();
    let lease = store.acquire(key(7, 1)).unwrap();
    store.close();
    assert!(matches!(store.acquire(key(7, 2)), Err(StoreError::Closed)));
    assert!(matches!(store.prefetch(&[key(7, 2)]), Err(StoreError::Closed)));
    // A lease outlives the store's closing and even the store.
    drop(store);
    assert_eq!(lease.part(0), &fixture.expert(7, 1)[0][..]);
}

#[test]
fn a_read_error_reaches_the_caller_and_holds_nothing() {
    let fixture = Fixture::new("error");
    let store = ExpertStore::open(&fixture.catalog_path(), options(16 << 20)).unwrap();
    // Truncate the file of layer 0's scales after the store checked it: those reads come back short.
    std::fs::OpenOptions::new()
        .write(true)
        .open(fixture.dir.join("f1.bin"))
        .unwrap()
        .set_len(10)
        .unwrap();
    let error = store.acquire(key(0, 4)).err().unwrap();
    assert!(matches!(error, StoreError::Io(_)), "{error:?}");
    // No lease; the rows of the other file that loaded completely may stay cached (they are exact), nothing else.
    let stats = store.stats();
    assert_eq!(
        (stats.active_leases, stats.leased_bytes, stats.acquire_failures),
        (0, 0, 1)
    );
    assert_eq!(
        stats.cache_resident_bytes,
        stats.cache.insert_bytes - stats.cache.evicted_bytes
    );
    assert_eq!(stats.cache.aborted_fills + stats.cache.inserts, 3, "{stats:?}");
    // The same expert fails again: its scales are still in the truncated file.
    let again = store.acquire(key(0, 4)).err();
    assert!(matches!(again, Some(StoreError::Io(_))), "{again:?}");
}
