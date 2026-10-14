"""Regenerate `headers/` (Phase 7A): the safetensors headers of one MoE layer's routed experts of each flagship, at the
pinned revisions, fetched without any tensor data (HTTP range requests of each shard's first bytes: the 8-byte header
length, then the header). Needs network access; standard library only.

    python research/apexquant/fetch_headers.py

What is downloaded: the index of a checkpoint that has one (MiMo-V2.6 Pro's is 15 MB, Inkling's 0.1 MB) and the headers
of the shards holding layer 10's experts (a few hundred kilobytes at most). Colibri's Inkling container has no index:
its shards' headers are read in order until layer 10's experts are found.
"""

from __future__ import annotations

import json
import struct
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
HF = "https://huggingface.co"
LAYER = 10


def ranged(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=300) as response:
        return json.loads(response.read())


def header(repo: str, revision: str, filename: str) -> dict:
    url = f"{HF}/{repo}/resolve/{revision}/{filename}"
    (length,) = struct.unpack("<Q", ranged(url, 0, 7))
    if length > 64 << 20:
        raise SystemExit(f"{filename}: a header of {length} bytes is implausible")
    return json.loads(ranged(url, 8, 7 + length))


def tensors(meta: dict, filename: str, predicate) -> list[dict]:
    rows = []
    for key in sorted(k for k in meta if k != "__metadata__" and predicate(k)):
        start, end = meta[key]["data_offsets"]
        rows.append({"tensor": key, "dtype": meta[key]["dtype"], "shape": meta[key]["shape"], "bytes": end - start,
                     "file": filename})
    return rows


def record(name: str, repo: str, revision: str, found: list[dict]) -> None:
    out = {"model": name, "repository": repo, "revision": revision, "layer": LAYER, "tensors": found}
    (HERE / "headers").mkdir(exist_ok=True)
    (HERE / "headers" / f"{name}.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"{name}: {[t['tensor'] for t in found]}")


def indexed(name: str, repo: str, revision: str, predicate) -> None:
    index = fetch_json(f"{HF}/{repo}/resolve/{revision}/model.safetensors.index.json")["weight_map"]
    found = []
    for filename in sorted({index[k] for k in index if predicate(k)}):
        found += tensors(header(repo, revision, filename), filename, predicate)
    record(name, repo, revision, found)


def main() -> int:
    mimo = "XiaomiMiMo/MiMo-V2.6-Pro-MOPD", "6a7f202e8dbd0e57b3fe1da03917d08d75abc4b8"
    indexed("mimo-v2.6-pro", *mimo, lambda k: k.startswith(f"model.layers.{LAYER}.mlp.experts.0."))
    inkling = "thinkingmachines/Inkling", "828496eeae4c243ff1a22f7f28ff83694f2f7bc9"
    indexed("inkling-bf16", *inkling, lambda k: f".layers.{LAYER}.mlp.experts." in k)
    repo, revision = "nbeerbower/Inkling-colibri-int4", "3981e658ed68ec681554d48a62f9a29dd9e61543"
    listing = fetch_json(f"{HF}/api/models/{repo}/revision/{revision}")
    found = []
    for filename in sorted(s["rfilename"] for s in listing["siblings"] if s["rfilename"].endswith(".safetensors")):
        found += tensors(header(repo, revision, filename), filename, lambda k: f".layers.{LAYER}.mlp.experts." in k)
        if len(found) >= 4:  # gate_up, its scales, down, its scales
            break
    record("inkling-colibri-int4", repo, revision, found)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
