//! The expert store on a real checkpoint's experts (Phase 7A, decision 0014); `benchmarks/store_check.py` prepares the
//! inputs (the catalog, and reference digests read through the Python store).
//!
//!     cargo run --release -p weightsift-store --example store_check -- <dir> [--budget-gib 6]
//!
//! Every expert of the catalog's layers is leased through the store:
//!   1. cold: a fresh store, one expert at a time (acquire, release), each lease timed;
//!   2. warm: the same order again on the same store (served by its cache when the budget holds them all);
//!   3. verification: every part of every expert, leased, hashed and compared with the reference digests (untimed);
//!   4. batches: a fresh store, each layer's experts in one `acquire_many`, each call timed.
//!
//! `<dir>/store-check.json` records the timings (a microbenchmark of the store alone on this machine, not an inference
//! result), the store's I/O and cache counters for each pass, and the verification's outcome.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use weightsift_store::{Catalog, ExpertKey, ExpertStore, StoreOptions, StoreStats};

fn percentiles(mut samples: Vec<f64>) -> Value {
    samples.sort_by(|a, b| a.total_cmp(b));
    let at = |q: f64| samples[((samples.len() - 1) as f64 * q).round() as usize];
    json!({
        "count": samples.len(),
        "mean_ms": samples.iter().sum::<f64>() / samples.len() as f64,
        "p50_ms": at(0.50),
        "p90_ms": at(0.90),
        "p99_ms": at(0.99),
        "max_ms": at(1.0),
    })
}

fn counters(stats: &StoreStats) -> Value {
    json!({
        "experts": stats.experts,
        "expert_hits": stats.expert_hits,
        "expert_misses": stats.expert_misses,
        "expert_waits": stats.expert_waits,
        "acquire_failures": stats.acquire_failures,
        "logical_bytes": stats.io.logical_bytes,
        "physical_bytes": stats.io.physical_bytes,
        "read_calls": stats.io.read_calls,
        "extents": stats.io.extents,
        "blocks_4k": stats.io.blocks_4k,
        "busy_ms": stats.io.busy_ns as f64 / 1e6,
        "read_ms": stats.io.read_ns as f64 / 1e6,
        "cache_lookups": stats.cache.lookups,
        "cache_hits": stats.cache.hits,
        "cache_misses": stats.cache.misses,
        "cache_inserts": stats.cache.inserts,
        "cache_evictions": stats.cache.evictions,
        "cache_bypassed": stats.cache.bypassed,
        "cache_capacity_bytes": stats.cache_capacity_bytes,
        "cache_block_bytes": stats.cache_block_bytes,
        "cache_resident_bytes": stats.cache_resident_bytes,
        "cache_peak_held_bytes": stats.cache_peak_held_bytes,
        "staging_bytes": stats.staging_bytes,
        "peak_leased_bytes": stats.peak_leased_bytes,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(args.get(1).ok_or("usage: store_check <dir> [--budget-gib N]")?);
    let budget_gib: f64 = match args.iter().position(|a| a == "--budget-gib") {
        Some(k) => args.get(k + 1).ok_or("--budget-gib needs a value")?.parse()?,
        None => 6.0,
    };
    let catalog_path = dir.join("catalog.json");
    let (catalog, _) = Catalog::from_path(&catalog_path)?;
    let digests: HashMap<String, Vec<String>> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("digests.json"))?)?;
    let options = StoreOptions {
        host_cache_bytes: (budget_gib * (1u64 << 30) as f64) as u64,
        ..StoreOptions::default()
    };
    let keys: Vec<ExpertKey> = catalog
        .layers
        .iter()
        .flat_map(|l| (0..l.experts).map(move |expert| ExpertKey { layer: l.layer, expert }))
        .collect();

    // 1 and 2: one expert at a time, cold then warm.
    let store = ExpertStore::open(&catalog_path, options)?;
    let expert_bytes = store.layer(keys[0].layer).ok_or("no layer")?.expert_bytes();
    let mut passes = Vec::new();
    for pass in ["cold", "warm"] {
        store.reset_stats();
        let mut latencies = Vec::with_capacity(keys.len());
        let started = Instant::now();
        for &key in &keys {
            let at = Instant::now();
            let lease = store.acquire(key)?;
            latencies.push(at.elapsed().as_secs_f64() * 1e3);
            drop(lease);
        }
        let seconds = started.elapsed().as_secs_f64();
        let bytes = keys.len() as u64 * expert_bytes;
        let stats = store.stats();
        passes.push(json!({
            "pass": pass,
            "experts": keys.len(),
            "bytes": bytes,
            "seconds": seconds,
            "gb_per_second": bytes as f64 / seconds / 1e9,
            "lease_latency": percentiles(latencies),
            "store": counters(&stats),
        }));
        eprintln!(
            "{pass}: {} experts, {:.2} GB in {seconds:.2} s",
            keys.len(),
            bytes as f64 / 1e9
        );
    }

    // 3: every part of every expert against the reference.
    let mut checked = 0u64;
    let mut mismatches = Vec::new();
    for &key in &keys {
        let lease = store.acquire(key)?;
        let layer = catalog.layers.iter().find(|l| l.layer == key.layer).ok_or("no layer")?;
        for (p, part) in layer.parts.iter().enumerate() {
            let segment = &catalog.segments[part.segment as usize].name;
            let row = (part.first_row + key.expert as u64) as usize;
            let digest: String = Sha256::digest(lease.part(p))
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            let expected = digests
                .get(segment)
                .and_then(|d| d.get(row))
                .ok_or("no reference digest")?;
            if &digest != expected {
                mismatches.push(format!("{segment} row {row}"));
            }
            checked += 1;
        }
    }
    store.close();
    drop(store);

    // 4: each layer's experts in one call, on a fresh store.
    let store = ExpertStore::open(&catalog_path, options)?;
    let mut latencies = Vec::new();
    let started = Instant::now();
    for layer in &catalog.layers {
        let batch: Vec<ExpertKey> = keys.iter().copied().filter(|k| k.layer == layer.layer).collect();
        let at = Instant::now();
        let leases = store.acquire_many(&batch)?;
        latencies.push(at.elapsed().as_secs_f64() * 1e3);
        drop(leases);
    }
    let seconds = started.elapsed().as_secs_f64();
    let bytes = keys.len() as u64 * expert_bytes;
    passes.push(json!({
        "pass": "batches",
        "experts": keys.len(),
        "bytes": bytes,
        "seconds": seconds,
        "gb_per_second": bytes as f64 / seconds / 1e9,
        "call_latency": percentiles(latencies),
        "store": counters(&store.stats()),
    }));
    store.close();

    let record = json!({
        "catalog": catalog.name,
        "options": {
            "host_cache_bytes": options.host_cache_bytes,
            "direct": options.direct,
            "workers": options.workers,
            "staging_sets": options.staging_sets,
            "staging_slots": options.staging_slots,
            "staging_slot_bytes": options.staging_slot_bytes,
        },
        "expert_bytes": expert_bytes,
        "passes": passes,
        "verification": {"rows": checked, "mismatches": mismatches},
        "kind": "microbenchmark of the expert store alone (no inference)",
    });
    std::fs::write(
        dir.join("store-check.json"),
        serde_json::to_string_pretty(&record)? + "\n",
    )?;
    println!("{}", serde_json::to_string_pretty(&record)?);
    if !record["verification"]["mismatches"]
        .as_array()
        .is_some_and(|m| m.is_empty())
    {
        return Err("leased bytes differ from the reference".into());
    }
    Ok(())
}
