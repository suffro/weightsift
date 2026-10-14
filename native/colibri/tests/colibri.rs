//! The bridge to Colibrì, built from its pinned sources: the Rust mirrors of its segment ABI match its header; its own
//! conformance checks pass; its fixture adapters run from Rust through Weightsift's execution interface (continuation
//! after a snapshot is exact, sessions are isolated, bad snapshots and requests are refused); and its expert-store lease
//! contract holds on Colibrì's own inline helpers when the store behind it is Weightsift's, over real files.
//!
//! The fixtures are Colibrì's interface fixtures and a synthetic checkpoint: they prove the seams, nothing about any
//! model. Ignored, with the reason, when the pinned sources are not fetched.

#![cfg_attr(not(colibri_sources), allow(unused))]

use std::ffi::{c_char, c_int, CStr};
use std::io::Write;
use std::mem::{offset_of, size_of};
use std::path::PathBuf;
use std::sync::Once;

#[cfg(colibri_sources)]
use weightsift_colibri::segment::*;
use weightsift_runtime::{ExecutionBackend, ExecutionSession, RunRequest};

const NOT_FETCHED: &str = "Colibri's pinned sources are not fetched: run `python native/fetch_upstream.py`";

#[repr(C)]
struct Fixture {
    family_id: *const c_char,
    display_name: *const c_char,
    state_schema: *const c_char,
    state_description: *const c_char,
    tiny_generator: *const c_char,
    state_kinds: u32,
    state_lanes: u32,
    state_width: u32,
    num_layers: u32,
    max_context_tokens: u32,
    fixture_tag: u32,
}

#[cfg(colibri_sources)]
extern "C" {
    fn ws_colibri_segment_layout(out: *mut u64, count: usize);
    #[allow(clippy::too_many_arguments)]
    fn ws_colibri_contract(
        catalog: *const c_char,
        cache_bytes: u64,
        layer: c_int,
        expert: c_int,
        experts: c_int,
        checksums: *mut u64,
        formats: *mut c_int,
        message: *mut c_char,
        message_size: usize,
    ) -> c_int;
    fn coli_segment_conformance_register_fixtures() -> c_int;
    fn coli_segment_conformance_fixture_count() -> usize;
    fn coli_segment_conformance_fixture_at(index: usize) -> *const Fixture;
    fn coli_segment_conformance_run_fixture(fixture: *const Fixture, error: *mut c_char, error_size: usize) -> c_int;
}

#[cfg(colibri_sources)]
fn fixtures() -> Vec<&'static Fixture> {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| assert_eq!(unsafe { coli_segment_conformance_register_fixtures() }, 0));
    let count = unsafe { coli_segment_conformance_fixture_count() };
    (0..count)
        .map(|k| unsafe { &*coli_segment_conformance_fixture_at(k) })
        .collect()
}

fn text(p: *const c_char) -> String {
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

#[test]
#[cfg_attr(
    not(colibri_sources),
    ignore = "Colibri's pinned sources are not fetched: run `python native/fetch_upstream.py`"
)]
fn colibri_sources_are_built_from_the_pinned_commit() {
    // The build used the fetched checkout only if it verified against the pin; the marker names what was verified.
    let native = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let pins: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(native.join("upstream.json")).unwrap()).unwrap();
    let marker = native.join("third_party/colibri/.weightsift-upstream.json");
    let marker: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(marker).expect(NOT_FETCHED)).unwrap();
    let pin = pins["upstreams"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["name"] == "colibri")
        .unwrap();
    assert_eq!(marker["commit"], pin["commit"]);
    assert_eq!(marker["license"], "Apache-2.0");
    assert_eq!(weightsift_colibri::UPSTREAM_AVAILABLE, marker.is_object());
}

#[cfg(colibri_sources)]
#[test]
fn the_rust_mirrors_match_colibris_segment_abi() {
    let mut c = [0u64; 14];
    unsafe { ws_colibri_segment_layout(c.as_mut_ptr(), c.len()) };
    let rust = [
        size_of::<ColiSegmentCapabilities>(),
        offset_of!(ColiSegmentCapabilities, engine_id),
        offset_of!(ColiSegmentCapabilities, state_dtype),
        offset_of!(ColiSegmentCapabilities, reserved_u64),
        size_of::<ColiSegmentEngineOptions>(),
        offset_of!(ColiSegmentEngineOptions, model_dir),
        offset_of!(ColiSegmentEngineOptions, memory_limit_bytes),
        offset_of!(ColiSegmentEngineOptions, resource_plan_size),
        size_of::<ColiSegmentSessionOptions>(),
        size_of::<ColiSegmentRunRequest>(),
        offset_of!(ColiSegmentRunRequest, token_ids),
        offset_of!(ColiSegmentRunRequest, output_bytes),
        offset_of!(ColiSegmentRunRequest, should_cancel),
        offset_of!(ColiSegmentRunRequest, reserved_u64),
    ];
    assert_eq!(c.map(|v| v as usize), rust);
}

#[cfg(colibri_sources)]
#[test]
fn colibris_conformance_checks_pass_for_every_fixture() {
    let all = fixtures();
    assert!(all.len() >= 7, "Colibri ships a fixture per model family");
    assert!(SegmentEngine::registered() >= all.len());
    for fixture in all {
        let mut error = [0 as c_char; 512];
        let status = unsafe { coli_segment_conformance_run_fixture(fixture, error.as_mut_ptr(), error.len()) };
        assert_eq!(status, 0, "{}: {}", text(fixture.family_id), text(error.as_ptr()));
    }
}

/// Rows of input for a fixture: deterministic values, and one token id per row.
#[cfg(colibri_sources)]
fn rows(width: u32, rows: u32, seed: u32) -> (Vec<u8>, Vec<i32>) {
    let values: Vec<f32> = (0..rows * width).map(|k| ((k * 7 + seed) % 23) as f32 / 8.0).collect();
    let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tokens = (0..rows).map(|r| (r * 31 + seed) as i32).collect();
    (bytes, tokens)
}

#[cfg(colibri_sources)]
fn run(
    session: &mut impl ExecutionSession,
    position: u64,
    input: &[u8],
    tokens: &[i32],
) -> Result<Vec<u8>, weightsift_runtime::ExecutionError> {
    let mut output = vec![0u8; input.len()];
    session.run(RunRequest {
        position,
        rows: tokens.len() as u32,
        token_ids: Some(tokens),
        input,
        output: &mut output,
    })?;
    Ok(output)
}

#[cfg(colibri_sources)]
#[test]
fn fixture_ranges_run_from_rust_through_the_execution_interface() {
    let dir = PathBuf::from("fixture-model");
    for fixture in fixtures() {
        let id = text(fixture.family_id);
        let layers = fixture.num_layers;
        let engine = SegmentEngine::open(&id, &dir, 0..layers, 32, 0).unwrap();
        let caps = engine.capabilities();
        assert_eq!(caps.engine_id, id);
        assert_eq!(caps.state_schema, text(fixture.state_schema));
        assert_eq!((caps.state_width, caps.num_layers), (fixture.state_width, layers));
        assert!(
            caps.token_ids && caps.snapshots && caps.backends.contains(&"cpu"),
            "{caps:?}"
        );
        let width = caps.state_width;
        let (first, first_tokens) = rows(width, 2, 1);
        let (next, next_tokens) = rows(width, 1, 9);
        // A session runs two rows, is snapshotted, runs one more.
        let mut a = engine.session(32, 0).unwrap();
        let out_a = run(&mut a, 0, &first, &first_tokens).unwrap();
        assert_ne!(out_a, first, "{id}: the fixture transforms its input");
        let snapshot = a.snapshot().unwrap();
        let continued = run(&mut a, 2, &next, &next_tokens).unwrap();
        // A second, isolated session restored from the snapshot continues exactly as the first did.
        let mut b = engine.session(32, 0).unwrap();
        b.restore(&snapshot).unwrap();
        assert_eq!(
            run(&mut b, 2, &next, &next_tokens).unwrap(),
            continued,
            "{id}: exact continuation"
        );
        // A third session, run from the start, matches the first: sessions share no state.
        let mut c = engine.session(32, 0).unwrap();
        assert_eq!(
            run(&mut c, 0, &first, &first_tokens).unwrap(),
            out_a,
            "{id}: isolated sessions"
        );
        // A corrupt snapshot is refused and leaves the session as it was.
        let mut corrupt = snapshot.clone();
        corrupt[45] ^= 0xff;
        assert!(c.restore(&corrupt).is_err(), "{id}: a corrupt snapshot is refused");
        assert!(
            c.restore(&snapshot[..snapshot.len() - 1]).is_err(),
            "{id}: a short snapshot is refused"
        );
        assert_eq!(
            run(&mut c, 2, &next, &next_tokens).unwrap(),
            continued,
            "{id}: refused restores change nothing"
        );
        // A snapshot of another range is refused.
        if layers >= 2 {
            let half = SegmentEngine::open(&id, &dir, 0..layers / 2, 32, 0).unwrap();
            let mut h = half.session(32, 0).unwrap();
            assert!(
                h.restore(&snapshot).is_err(),
                "{id}: a snapshot of another range is refused"
            );
        }
        // Requests Colibri's runtime refuses: a position that is not the session's, the wrong buffer size, no tokens.
        let error = run(&mut c, 7, &next, &next_tokens).unwrap_err();
        assert!(error.0.contains("contiguous"), "{id}: {error}");
        let mut short_output = vec![0u8; next.len() - 4];
        let error = c
            .run(RunRequest {
                position: 3,
                rows: 1,
                token_ids: Some(&next_tokens),
                input: &next,
                output: &mut short_output,
            })
            .unwrap_err();
        assert!(error.0.contains("wrong size"), "{id}: {error}");
        let mut output = vec![0u8; next.len()];
        let error = c
            .run(RunRequest {
                position: 3,
                rows: 1,
                token_ids: None,
                input: &next,
                output: &mut output,
            })
            .unwrap_err();
        assert!(error.0.contains("token"), "{id}: {error}");
    }
    // Ranges and engines the runtime refuses.
    let first = fixtures()[0];
    let id = text(first.family_id);
    assert!(SegmentEngine::open(&id, &dir, 0..first.num_layers + 1, 32, 0).is_err());
    assert!(SegmentEngine::open(&id, &dir, 2..2, 32, 0).is_err());
    let error = SegmentEngine::open("no-such-engine", &dir, 0..1, 32, 0).err().unwrap();
    assert!(error.0.contains("not registered"), "{error}");
}

// The lease contract, on a synthetic checkpoint in Colibri's tensor formats: one layer of five experts, gate and up rows
// back to back in one tensor (q4 rows, f32 scales per row), down in another, scales in a third.

const EXPERTS: u32 = 5;
const GATE: u64 = 3_072;
const DOWN: u64 = 3_072;
const SCALES: u64 = 256;

struct Checkpoint {
    dir: PathBuf,
    data: Vec<u8>,
    w13: u64,
    w2: u64,
    scales: u64,
}

impl Checkpoint {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("weightsift-colibri-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut state = 11u64;
        let data: Vec<u8> = (0..200_000)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 33) as u8
            })
            .collect();
        std::fs::File::create(dir.join("experts.bin"))
            .unwrap()
            .write_all(&data)
            .unwrap();
        let (w13, w2, scales) = (512u64 + 7, 60_000u64, 150_000u64);
        let view = |name: &str, part: u32, offset: u64, bytes: u64, scale_offset: u64| {
            serde_json::json!({
                "name": name, "part": part, "offset": offset, "bytes": bytes, "format": "q4_row",
                "rows": 64, "columns": bytes * 2 / 64,
                "scales": {"part": 2, "offset": scale_offset, "bytes": SCALES, "format": "f32"}
            })
        };
        let catalog = serde_json::json!({
            "format": "weightsift-expert-catalog",
            "version": 1,
            "name": "colibri-format fixture",
            "files": [{"key": "experts", "path": "experts.bin", "size": data.len()}],
            "segments": [
                {"name": "w13", "files": [0], "rows": EXPERTS, "row_bytes": 2 * GATE, "offset": w13},
                {"name": "w2", "files": [0], "rows": EXPERTS, "row_bytes": DOWN, "offset": w2},
                {"name": "scales", "files": [0], "rows": EXPERTS, "row_bytes": 3 * SCALES, "offset": scales}
            ],
            "layers": [{
                "layer": 3,
                "experts": EXPERTS,
                "parts": [{"name": "w13", "segment": 0}, {"name": "w2", "segment": 1}, {"name": "scales", "segment": 2}],
                "views": [view("gate", 0, 0, GATE, 0), view("up", 0, GATE, GATE, SCALES), view("down", 1, 0, DOWN, 2 * SCALES)]
            }]
        });
        std::fs::write(
            dir.join("catalog.json"),
            serde_json::to_string_pretty(&catalog).unwrap(),
        )
        .unwrap();
        Checkpoint {
            dir,
            data,
            w13,
            w2,
            scales,
        }
    }

    fn bytes(&self, offset: u64, len: u64) -> &[u8] {
        &self.data[offset as usize..(offset + len) as usize]
    }
}

impl Drop for Checkpoint {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

#[cfg(colibri_sources)]
#[test]
fn colibris_lease_contract_holds_on_weightsifts_store() {
    let checkpoint = Checkpoint::new("contract");
    let catalog = std::ffi::CString::new(checkpoint.dir.join("catalog.json").to_str().unwrap()).unwrap();
    for expert in [0u32, 4] {
        let mut checksums = [0u64; 6];
        let mut formats = [0 as c_int; 3];
        let mut message = [0 as c_char; 512];
        let line = unsafe {
            ws_colibri_contract(
                catalog.as_ptr(),
                1 << 20,
                3,
                expert as c_int,
                EXPERTS as c_int,
                checksums.as_mut_ptr(),
                formats.as_mut_ptr(),
                message.as_mut_ptr(),
                message.len(),
            )
        };
        assert_eq!(line, 0, "{}", text(message.as_ptr()));
        let e = expert as u64;
        let w13 = checkpoint.w13 + e * 2 * GATE;
        let w2 = checkpoint.w2 + e * DOWN;
        let scales = checkpoint.scales + e * 3 * SCALES;
        let expected = [
            fnv1a(checkpoint.bytes(w13, GATE)),
            fnv1a(checkpoint.bytes(w13 + GATE, GATE)),
            fnv1a(checkpoint.bytes(w2, DOWN)),
            fnv1a(checkpoint.bytes(scales, SCALES)),
            fnv1a(checkpoint.bytes(scales + SCALES, SCALES)),
            fnv1a(checkpoint.bytes(scales + 2 * SCALES, SCALES)),
        ];
        assert_eq!(checksums, expected, "Colibri's views point at the file's bytes");
        // COLI_TENSOR_Q4_ROW is the third value of ColiTensorFormat.
        assert_eq!(formats, [2, 2, 2]);
    }
}

#[cfg(colibri_sources)]
#[test]
fn catalogs_colibri_cannot_read_are_refused_when_the_store_opens() {
    use weightsift_colibri::store::ColibriStore;
    use weightsift_store::ffi::{ws_store_options_default, WsStoreOptions};
    let checkpoint = Checkpoint::new("refused");
    let path = checkpoint.dir.join("catalog.json");
    let mut options = WsStoreOptions {
        struct_size: size_of::<WsStoreOptions>() as u32,
        ..WsStoreOptions::default()
    };
    unsafe { ws_store_options_default(&mut options) };
    options.host_cache_bytes = 1 << 20;
    options.staging_slot_bytes = 64 << 10;
    let store = ColibriStore::open(&path, &options).unwrap();
    assert_ne!(store.handle(), 0);
    drop(store);
    // A format Colibri has no tensor type for.
    let json = std::fs::read_to_string(&path)
        .unwrap()
        .replacen("\"q4_row\"", "\"bf16\"", 1);
    std::fs::write(&path, json).unwrap();
    let error = ColibriStore::open(&path, &options).err().unwrap();
    assert!(
        error.contains("bf16") && error.contains("not one Colibri reads"),
        "{error}"
    );
    // A layer without a view the contract needs.
    let json = std::fs::read_to_string(&path)
        .unwrap()
        .replacen("\"bf16\"", "\"q4_row\"", 1)
        .replacen("\"down\"", "\"w_down\"", 1);
    std::fs::write(&path, json).unwrap();
    let error = ColibriStore::open(&path, &options).err().unwrap();
    assert!(error.contains("no view named down"), "{error}");
}
