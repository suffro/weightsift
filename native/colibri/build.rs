//! Builds the bridge to Colibrì from its pinned, fetched sources (decision 0014).
//!
//! Colibrì is never vendored: `native/upstream.json` pins its commit and the sha256 of every file used, and
//! `native/fetch_upstream.py` checks them out into `native/third_party/colibri/` (gitignored). This script builds the C
//! parts only if that checkout carries the pin's commit and every file still has the pin's hash (re-hashed here); then
//! it sets the `colibri_sources` cfg. Without them the crate builds without its C parts, says so, and its tests are
//! ignored with the reason.
//!
//! Compiled: Weightsift's shim (`csrc/ws_colibri_store.c`, Colibrì's lease contract over Weightsift's store ABI, strict
//! warnings) and Colibrì's engine-neutral segment runtime (`c/segment_runtime.c`, as upstream wrote it) into the
//! library; the contract checks (`csrc/ws_colibri_tests.c`) and Colibrì's conformance fixtures into the tests only.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

fn verified(pins: &Path, root: &Path) -> Result<(), String> {
    let pins: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(pins).map_err(|e| format!("cannot read {}: {e}", pins.display()))?,
    )
    .map_err(|e| format!("malformed pins: {e}"))?;
    let pin = pins["upstreams"]
        .as_array()
        .and_then(|all| all.iter().find(|p| p["name"] == "colibri"))
        .ok_or("no pin for colibri")?;
    let marker = root.join(".weightsift-upstream.json");
    let marker: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&marker).map_err(|_| "not fetched".to_string())?)
            .map_err(|e| format!("malformed marker: {e}"))?;
    if marker["commit"] != pin["commit"] {
        return Err(format!("fetched {}, pinned {}", marker["commit"], pin["commit"]));
    }
    let files = pin["files"].as_object().ok_or("the pin lists no files")?;
    for (name, expected) in files {
        let bytes = std::fs::read(root.join(name)).map_err(|_| format!("{name} is missing"))?;
        let digest: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
        if Some(digest.as_str()) != expected.as_str() {
            return Err(format!("{name} differs from the pin"));
        }
    }
    Ok(())
}

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let native = manifest.parent().expect("native/colibri").to_path_buf();
    let pins = native.join("upstream.json");
    let root = native.join("third_party").join("colibri");
    let colibri = root.join("c");
    let store_include = native.join("store").join("include");
    for path in [
        pins.clone(),
        root.join(".weightsift-upstream.json"),
        manifest.join("csrc"),
        store_include.join("weightsift_store.h"),
    ] {
        println!("cargo::rerun-if-changed={}", path.display());
    }
    println!("cargo::rustc-check-cfg=cfg(colibri_sources)");
    if let Err(why) = verified(&pins, &root) {
        println!(
            "cargo::warning=Colibri's pinned sources are not available ({why}): run `python native/fetch_upstream.py`. \
             weightsift-colibri is built without its C parts and its tests are ignored."
        );
        return;
    }
    for name in ["c/segment_runtime.c", "c/tests/segment_conformance_fixtures.c"] {
        println!("cargo::rerun-if-changed={}", root.join(name).display());
    }
    println!("cargo::rustc-cfg=colibri_sources");
    // Weightsift's shim: its own code, held to strict warnings.
    cc::Build::new()
        .file(manifest.join("csrc").join("ws_colibri_store.c"))
        .include(&colibri)
        .include(&store_include)
        .std("c11")
        .warnings(true)
        .extra_warnings(true)
        .warnings_into_errors(true)
        .compile("ws_colibri_store");
    // Colibrì's segment runtime, as upstream wrote it.
    cc::Build::new()
        .file(colibri.join("segment_runtime.c"))
        .include(&colibri)
        .std("c11")
        .compile("colibri_segment_runtime");
    // Tests only: the lease-contract checks and Colibrì's own conformance fixtures; never in the library.
    cc::Build::new()
        .file(manifest.join("csrc").join("ws_colibri_tests.c"))
        .file(colibri.join("tests").join("segment_conformance_fixtures.c"))
        .include(&colibri)
        .include(colibri.join("tests"))
        .include(&store_include)
        .include(manifest.join("csrc"))
        .std("c11")
        .cargo_metadata(false)
        .compile("ws_colibri_tests");
    let out = std::env::var("OUT_DIR").expect("cargo sets OUT_DIR");
    let msvc = std::env::var("CARGO_CFG_TARGET_ENV").is_ok_and(|env| env == "msvc");
    let archive = if msvc {
        "ws_colibri_tests.lib"
    } else {
        "libws_colibri_tests.a"
    };
    println!("cargo::rustc-link-arg-tests={out}/{archive}");
}
