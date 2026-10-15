"""Fetch the study's quantized-expert sample by HTTP range requests (only after the user's approval).

    python research/lossless_moe/fetch_sample.py --plan            # what would be fetched: files, ranges, bytes (headers only)
    python research/lossless_moe/fetch_sample.py --fetch           # fetch it into experiments/lossless_moe/samples/ (gitignored)

For each source of the configuration's `samples` (a repository at a pinned revision), the routed-expert tensors of the
named layers are located from the safetensors headers (an index when the checkpoint has one; otherwise the shards'
headers in order, each a few kilobytes of range requests), and the sampled experts' byte ranges (weights and their
scales) are fetched exactly: nothing else of the checkpoint is downloaded. Experts are drawn per layer by the corpus's
seeded permutation.

Every slice is written to its own file with its sha256; then two random 64 KiB windows of it are requested again and
compared with what was stored (an independent second read of the same bytes). The manifest records every slice's
source (repository, revision, file, absolute offset, length), tensor, expert, role, dtype, logical shape and sha256.
Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lossless_moe import corpus  # noqa: E402

HF = "https://huggingface.co"
REPO_ROOT = HERE.parents[1]
SAMPLES = REPO_ROOT / "experiments" / "lossless_moe" / "samples"


def ranged(url: str, start: int, end: int, attempts: int = 5) -> bytes:
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            if len(data) != end - start + 1:
                raise IOError(f"short range: {len(data)} of {end - start + 1}")
            return data
        except (urllib.error.URLError, IOError, TimeoutError) as error:
            if attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))
            print(f"    retry {attempt + 1}: {error}", flush=True)
    raise AssertionError("unreachable")


def fetch_json(url: str):
    with urllib.request.urlopen(url, timeout=300) as r:
        return json.loads(r.read())


def header(repo: str, revision: str, filename: str) -> tuple[dict, int]:
    url = f"{HF}/{repo}/resolve/{revision}/{filename}"
    (n,) = struct.unpack("<Q", ranged(url, 0, 7))
    if n > 64 << 20:
        raise SystemExit(f"{filename}: implausible header of {n} bytes")
    return json.loads(ranged(url, 8, 7 + n)), 8 + n


class Locator:
    """Where each tensor of a source lives (file, absolute data offset, length, dtype, shape), from headers only."""

    def __init__(self, repo: str, revision: str) -> None:
        self.repo, self.revision = repo, revision
        listing = fetch_json(f"{HF}/api/models/{repo}/revision/{revision}")
        self.files = sorted(s["rfilename"] for s in listing["siblings"] if s["rfilename"].endswith(".safetensors"))
        self.index = None
        if any(s["rfilename"] == "model.safetensors.index.json" for s in listing["siblings"]):
            self.index = fetch_json(f"{HF}/{repo}/resolve/{revision}/model.safetensors.index.json")["weight_map"]
        self.tensors: dict[str, dict] = {}
        self.scanned: set[str] = set()
        self.header_bytes = 0

    def _scan(self, filename: str) -> None:
        if filename in self.scanned:
            return
        h, start = header(self.repo, self.revision, filename)
        self.header_bytes += start
        self.scanned.add(filename)
        for name, e in h.items():
            if name == "__metadata__":
                continue
            b, en = e["data_offsets"]
            self.tensors[name] = {"file": filename, "offset": start + b, "length": en - b, "dtype": e["dtype"],
                                  "shape": e["shape"]}

    def find(self, names: list[str]) -> dict[str, dict]:
        missing = [n for n in names if n not in self.tensors]
        if self.index is not None:
            for filename in sorted({self.index[n] for n in missing}):
                self._scan(filename)
        else:
            for filename in self.files:
                if all(n in self.tensors for n in names):
                    break
                self._scan(filename)
        lost = [n for n in names if n not in self.tensors]
        if lost:
            raise SystemExit(f"{self.repo}: tensors not found: {lost[:4]}")
        return {n: self.tensors[n] for n in names}


def slices_for(source: dict, locator: Locator, layer: int, expert: int) -> list[dict]:
    """The byte ranges of one expert: per matrix, its weight and its scale, as layout rules define them."""
    layout = source["layout"]
    out = []
    if layout in ("per_expert_colibri_int4", "per_expert_mxfp4", "per_expert_ct_int4"):
        suffix = {"per_expert_colibri_int4": (".weight", ".weight.qs"),
                  "per_expert_mxfp4": (".weight", ".weight_scale"),
                  "per_expert_ct_int4": (".weight_packed", ".weight_scale")}[layout]
        for kind in ("gate", "up", "down"):
            base = f"model.layers.{layer}.mlp.experts.{expert}.{kind}_proj"
            names = [base + suffix[0], base + suffix[1]]
            found = locator.find(names)
            for role, name in zip(("weight", "scale"), names):
                t = found[name]
                out.append({"kind": kind, "role": role, "tensor": name, **t, "slice_offset": t["offset"],
                            "slice_length": t["length"]})
        return out
    if layout == "fused_colibri_int4":  # Inkling: per layer, gate_up_proj [E * 2I * H/2] U8 and .qs [E * 2I] F32
        E = source["experts"]
        for kind, tensor in (("gate_up", "gate_up_proj"), ("down", "down_proj")):
            names = [f"model.layers.{layer}.mlp.experts.{tensor}", f"model.layers.{layer}.mlp.experts.{tensor}.qs"]
            found = locator.find(names)
            for role, name in zip(("weight", "scale"), names):
                t = found[name]
                per = t["length"] // E
                out.append({"kind": kind, "role": role, "tensor": name, **t, "slice_offset": t["offset"] + expert * per,
                            "slice_length": per})
        return out
    if layout == "fused_gptoss_mxfp4":  # gpt-oss: per layer, {gate_up,down}_proj_blocks [E, R, B, 16], _scales [E, R, B]
        for kind, tensor in (("gate_up", "gate_up_proj"), ("down", "down_proj")):
            names = [f"model.layers.{layer}.mlp.experts.{tensor}_blocks", f"model.layers.{layer}.mlp.experts.{tensor}_scales"]
            found = locator.find(names)
            for role, name in zip(("weight", "scale"), names):
                t = found[name]
                E = t["shape"][0]
                per = t["length"] // E
                out.append({"kind": kind, "role": role, "tensor": name, **t, "slice_offset": t["offset"] + expert * per,
                            "slice_length": per})
        return out
    raise SystemExit(f"unknown layout {layout}")


def plan(config: dict) -> list[dict]:
    rows = []
    for name, source in config["samples"].items():
        loc = Locator(source["repository"], source["revision"])
        for layer in source["layers"]:
            for expert in corpus.sample_experts(config["seed"], source.get("sample_as", name), layer, source["experts"],
                                                source["experts_per_layer"]):
                for s in slices_for(source, loc, layer, expert):
                    rows.append({"source": name, "repository": source["repository"], "revision": source["revision"],
                                 "layout": source["layout"], "layer": layer, "expert": expert, **s})
        print(f"{name}: {sum(r['slice_length'] for r in rows if r['source'] == name) / 1e6:.1f} MB in "
              f"{sum(1 for r in rows if r['source'] == name)} slices; headers read {loc.header_bytes / 1e3:.0f} kB",
              flush=True)
    return rows


def fetch(rows: list[dict], out: Path) -> list[dict]:
    rng = random.Random(20261015)
    for i, r in enumerate(rows):
        url = f"{HF}/{r['repository']}/resolve/{r['revision']}/{r['file']}"
        path = out / r["source"] / f"L{r['layer']}_E{r['expert']}_{r['kind']}_{r['role']}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        start, length = r["slice_offset"], r["slice_length"]
        if not path.exists() or path.stat().st_size != length:
            data = bytearray()
            step = 64 << 20
            for b in range(start, start + length, step):
                data += ranged(url, b, min(start + length, b + step) - 1)
            path.write_bytes(bytes(data))
        stored = path.read_bytes()
        digest = hashlib.sha256(stored).hexdigest()
        checks = []
        for _ in range(2):  # an independent second read of two windows of the same bytes
            w = min(65536, length)
            a = rng.randrange(0, length - w + 1)
            again = ranged(url, start + a, start + a + w - 1)
            checks.append({"offset": a, "length": w, "equal": again == stored[a : a + w]})
        if not all(c["equal"] for c in checks):
            raise SystemExit(f"{path}: a re-read window differs from the stored bytes")
        r.update({"path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"), "sha256": digest, "reread": checks})
        print(f"  [{i + 1}/{len(rows)}] {path.name} {length / 1e6:.2f} MB", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "experiments" / "lossless_moe" / "sample-manifest.json")
    args = parser.parse_args()
    config = corpus.load_config(args.config)
    rows = plan(config)
    total = sum(r["slice_length"] for r in rows)
    print(f"total {total / 1e9:.3f} GB in {len(rows)} slices")
    if args.fetch:
        rows = fetch(rows, SAMPLES)
        args.manifest.write_text(json.dumps({"format": "lossless-moe-sample", "version": 1, "total_bytes": total,
                                             "slices": rows}, indent=1), encoding="utf-8")
        print(f"manifest {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
