//! Builds Colibri's int4-rans256-g0 codec from its pinned, fetched sources, when they are there.
//!
//! `research/lossless_moe/upstream.json` pins Colibri's commit and the sha256 of every file used;
//! `research/lossless_moe/fetch_upstream.py` checks them out into `research/lossless_moe/third_party/colibri/`
//! (gitignored). This script re-hashes them and, only if all match, compiles `c/tools/rans_ctypes.c` (Colibri's own
//! exported bridge over `c/rans.h`) as upstream wrote it, and sets the `colibri_rans` cfg. Without them the crate builds
//! its own codec alone and says so.

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
    let study = manifest.parent().expect("research/lossless_moe/codec").to_path_buf();
    let pins = study.join("upstream.json");
    let root = study.join("third_party").join("colibri");
    println!("cargo::rerun-if-changed={}", pins.display());
    println!(
        "cargo::rerun-if-changed={}",
        root.join(".weightsift-upstream.json").display()
    );
    println!("cargo::rustc-check-cfg=cfg(colibri_rans)");
    if let Err(why) = verified(&pins, &root) {
        println!(
            "cargo::warning=Colibri's pinned sources are not available ({why}): run \
             `python research/lossless_moe/fetch_upstream.py`. Built without Colibri's codec."
        );
        return;
    }
    for name in ["c/rans.h", "c/tools/rans_ctypes.c"] {
        println!("cargo::rerun-if-changed={}", root.join(name).display());
    }
    println!("cargo::rustc-cfg=colibri_rans");
    // Upstream C, compiled as upstream wrote it (no Weightsift warning policy), optimized like its `make rans`.
    cc::Build::new()
        .file(root.join("c").join("tools").join("rans_ctypes.c"))
        .std("c11")
        .opt_level(3)
        .warnings(false)
        .compile("colibri_rans");
}
