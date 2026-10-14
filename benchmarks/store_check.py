"""Weightsift's expert store on real checkpoint bytes (Phase 7A, decision 0014): the vertical slice of the storage
backend on Moonlight-16B-A3B's routed experts (a local, non-flagship checkpoint: interface-level evidence only, never a
flagship result).

    python -m uv run python benchmarks/store_check.py --output experiments/phase7a/store-moonlight [--layers 1,2,3,4]
    cd native && cargo run --release -p weightsift-store --example store_check -- ../experiments/phase7a/store-moonlight

The first command writes the expert catalog of the chosen layers (from the Phase 4B expert index: composed segments over
the published checkpoint, gate and up rows fused) and the reference digests: the sha256 of every expert row, read
through the Python store (`FileBackedPageStore`, direct I/O), the path validated bit for bit since Phase 4A. The second
leases every expert through the Rust store (cold, then warm, then in batches on a fresh store), checks every leased row
against the digests, and records throughput, per-lease latency and the store's I/O and cache counters
(`store-check.json`). The catalog holds this machine's paths and is not committed; its sha256 is recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from awpmi.hardware import detect
from awpmi.models.catalog import ViewSpec, expert_catalog, write_expert_catalog
from awpmi.storage.pack import open_pack
from awpmi.tracing import native_tree_sha256, source_tree_sha256

ROOT = Path(__file__).resolve().parents[1]
PATTERN = r"^model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$"
# Moonlight (DeepSeek-V3 architecture, transformers' fused experts): gate_up_proj rows are [2 * 1408, 2048], gate first;
# down_proj rows are [2048, 1408]. BF16 as published.
VIEWS = (
    ViewSpec("gate", "gate_up_proj", 0, 1408, "bf16"),
    ViewSpec("up", "gate_up_proj", 1408, 1408, "bf16"),
    ViewSpec("down", "down_proj", 0, 2048, "bf16"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", default=str(ROOT / "packs" / "moonlight-16b-a3b-expert-index"))
    parser.add_argument("--layers", default="1,2,3,4")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pack = open_pack(args.pack, verify="size")
    source = next(iter(pack.manifest["files"].values()))["source"]
    layers = [int(v) for v in args.layers.split(",")]
    catalog = expert_catalog(
        pack,
        name=f"{source['repository']}@{source['revision']} expert index, layers {args.layers}",
        pattern=PATTERN,
        parameters=("gate_up_proj", "down_proj"),
        views=VIEWS,
        layers=layers,
    )
    path = write_expert_catalog(catalog, output / "catalog.json")
    # The reference: every row of every chosen segment through the Python store, with direct reads.
    store = pack.store(direct=True)
    digests: dict[str, list[str]] = {}
    try:
        for segment in catalog["segments"]:
            rows = store.read_rows(segment["name"])
            digests[segment["name"]] = [
                hashlib.sha256(rows[k].numpy().tobytes()).hexdigest() for k in range(segment["rows"])
            ]
    finally:
        store.close()
    (output / "digests.json").write_text(json.dumps(digests, indent=1) + "\n", encoding="utf-8")
    prepare = {
        "pack": str(Path(args.pack).relative_to(ROOT)) if Path(args.pack).is_relative_to(ROOT) else args.pack,
        "pack_manifest_sha256": hashlib.sha256((Path(args.pack) / "manifest.json").read_bytes()).hexdigest(),
        "model": source["repository"],
        "revision": source["revision"],
        "layers": layers,
        "experts": sum(layer["experts"] for layer in catalog["layers"]),
        "expert_bytes": sum(s["row_bytes"] for s in catalog["segments"][:2]),
        "catalog_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reference": "awpmi.storage FileBackedPageStore, direct I/O, sha256 per row",
        "source_tree_sha256": source_tree_sha256(ROOT),
        "native_tree_sha256": native_tree_sha256(ROOT),
        "machine": detect(next(iter(pack.files.values()))),
    }
    (output / "prepare.json").write_text(json.dumps(prepare, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in prepare.items() if k != "machine"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
