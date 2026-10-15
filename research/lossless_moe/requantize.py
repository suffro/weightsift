"""The labelled diagnostic: Moonlight's sampled experts quantized with Colibri's per-row int4 recipe, written as a
sample (never a principal result; the brief forbids principal int4 results from re-quantized BF16).

    python research/lossless_moe/requantize.py --manifest experiments/lossless_moe/requant-manifest.json

The recipe is Colibri's `quant_int4_rows` (c/tools/convert_inkling_int4.py at bf244291), reimplemented here in its own
float32 NumPy arithmetic: scale = max |w| / 7 per row (floored at 1e-8), q = clip(rint(w / scale), -8, 7), nibble =
q + 8, low nibble = even column; the scales stored as F32. The experts are the BF16 sample's (`sample_as`), read by
positioned reads and checked against safetensors' own copy. Its output has the layout of Colibri's per-expert
containers, so stage_q reads it like any fetched source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lossless_moe import corpus  # noqa: E402

REPO_ROOT = HERE.parents[1]
SAMPLES = REPO_ROOT / "experiments" / "lossless_moe" / "samples"


def quant_int4_rows(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Colibri's per-row recipe: float32 [O, I] (I even) -> packed u8 [O, I/2], f32 scales [O]."""
    s = np.abs(w).max(axis=1, keepdims=True) / np.float32(7.0)
    s[s < 1e-8] = 1e-8
    q = np.clip(np.rint(w / s), -8, 7).astype(np.int32)
    lo = (q[:, 0::2] + 8).astype(np.uint8)
    hi = (q[:, 1::2] + 8).astype(np.uint8)
    return (lo | (hi << 4)), s[:, 0].astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--source", default="moonlight_requant_int4")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    config = corpus.load_config(args.config)
    spec = config["diagnostic"][args.source]
    model = spec["derived_from"]
    ckpt = corpus.Checkpoint(model, config["corpus"]["bf16"][model])
    slices = []
    out = SAMPLES / args.source
    out.mkdir(parents=True, exist_ok=True)
    for layer in spec["layers"]:
        for expert in corpus.sample_experts(config["seed"], spec["sample_as"], layer, spec["experts"],
                                            spec["experts_per_layer"]):
            for kind in corpus.MATRICES:
                m = ckpt.matrix(layer, expert, kind)
                p = ckpt.read_patterns(m)
                if not np.array_equal(p, ckpt.read_reference(m)):
                    raise SystemExit(f"{m.key}: the two read paths disagree")
                w = torch.from_numpy(p.astype(np.int16)).view(torch.bfloat16).float().numpy()
                packed, scales = quant_int4_rows(w)
                if list(packed.shape) != [spec["shapes"][kind][0], spec["shapes"][kind][1] // 2]:
                    raise SystemExit(f"{m.key}: unexpected shape {packed.shape}")
                for role, data in (("weight", packed.tobytes()), ("scale", scales.astype("<f4").tobytes())):
                    path = out / f"L{layer}_E{expert}_{kind}_{role}.bin"
                    path.write_bytes(data)
                    slices.append({"source": args.source, "layout": spec["layout"], "layer": layer, "expert": expert,
                                   "kind": kind, "role": role, "derived_from": m.key,
                                   "derived_from_sha256": hashlib.sha256(p.tobytes()).hexdigest(),
                                   "recipe": spec["recipe"], "slice_length": len(data),
                                   "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                                   "sha256": hashlib.sha256(data).hexdigest()})
            print(f"  L{layer} E{expert}", flush=True)
    args.manifest.write_text(json.dumps({"format": "lossless-moe-sample", "version": 1, "diagnostic": True,
                                         "total_bytes": sum(s["slice_length"] for s in slices), "slices": slices},
                                        indent=1), encoding="utf-8")
    print(f"{len(slices)} slices, manifest {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
