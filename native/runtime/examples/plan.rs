//! The runtime planner on given inputs (Phase 7A, decision 0014): `cargo run -p weightsift-runtime --example plan --
//! <inputs.json>` prints the memory plan as JSON.
//!
//! The inputs: `hardware` (as detected, e.g. by Python's `awpmi.hardware.detect`), `footprint` (the execution
//! backend's needs and the catalog's routed experts) and `policy` (reserves, an optional cap on the expert cache); every
//! field in bytes. The store's options are the defaults. The plan is arithmetic on these inputs, not a measurement.

use serde_json::{json, Value};
use weightsift_runtime::{plan, Device, Footprint, Hardware, Policy};
use weightsift_store::StoreOptions;

fn bytes(value: &Value, key: &str) -> Result<u64, String> {
    value[key]
        .as_u64()
        .ok_or_else(|| format!("{key} must be a number of bytes"))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: plan <inputs.json>")?;
    let inputs: Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    let h = &inputs["hardware"];
    let hardware = Hardware {
        host_total_bytes: bytes(h, "host_total_bytes")?,
        host_available_bytes: bytes(h, "host_available_bytes")?,
        device: match &h["device"] {
            Value::Null => None,
            d => Some(Device {
                name: d["name"].as_str().unwrap_or("").to_owned(),
                total_bytes: bytes(d, "total_bytes")?,
                free_bytes: bytes(d, "free_bytes")?,
            }),
        },
    };
    let f = &inputs["footprint"];
    let footprint = Footprint {
        dense_host_bytes: bytes(f, "dense_host_bytes")?,
        dense_device_bytes: bytes(f, "dense_device_bytes")?,
        state_host_bytes: bytes(f, "state_host_bytes")?,
        state_device_bytes: bytes(f, "state_device_bytes")?,
        workspace_host_bytes: bytes(f, "workspace_host_bytes")?,
        workspace_device_bytes: bytes(f, "workspace_device_bytes")?,
        expert_bytes: bytes(f, "expert_bytes")?,
        experts_per_token: bytes(f, "experts_per_token")?,
        total_experts: bytes(f, "total_experts")?,
    };
    let p = &inputs["policy"];
    let policy = Policy {
        host_reserve_bytes: bytes(p, "host_reserve_bytes")?,
        device_reserve_bytes: bytes(p, "device_reserve_bytes")?,
        store: StoreOptions::default(),
        max_expert_cache_bytes: p["max_expert_cache_bytes"].as_u64(),
    };
    let output = match plan(&hardware, &footprint, &policy) {
        Ok(plan) => json!({
            "feasible": true,
            "host_budget_bytes": plan.host_budget_bytes,
            "host": plan.host.iter().map(|r| json!({"name": r.name, "bytes": r.bytes})).collect::<Vec<_>>(),
            "device_budget_bytes": plan.device_budget_bytes,
            "device": plan.device.iter().map(|r| json!({"name": r.name, "bytes": r.bytes})).collect::<Vec<_>>(),
            "expert_cache_bytes": plan.expert_cache_bytes,
            "cache_experts": plan.cache_experts,
            "working_set_bytes": plan.working_set_bytes,
            "cache_tokens": plan.cache_tokens,
            "resident_fraction": plan.resident_fraction,
            "store_host_cache_bytes": plan.store.host_cache_bytes,
            "notes": plan.notes,
            "kind": "planning arithmetic on the inputs, not a measurement",
        }),
        Err(error) => json!({"feasible": false, "error": error.to_string()}),
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
