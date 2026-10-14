"""ApexQuant feasibility screening for routed experts of the flagships (Phase 7A, decision 0014): bytes, not quality.

    python -m uv run python research/apexquant/feasibility.py [--store-check experiments/phase7a/store-moonlight/store-check.json]

No ApexQuant code is used or run (it has no license): its format is costed with the accounting the paper publishes,
P (b + 16 / g) bits per weight for P passes of b bits per coordinate with one fp16 scale per group of g coordinates per
pass (arXiv 2610.07904v1, appendix H; b = 2, g = 512 in every result it reports). The native formats are costed from
the checkpoints' own safetensors headers (`headers/`, fetched without tensor data) and the geometry from their configs
(`geometry.json`). The output says, per flagship: bits per weight with metadata, bytes per expert and per decode token
(every routed expert read, no cache), the byte ratio to the native format, whether the expert matrices' rows split
into whole groups, the upper bound on a decode speedup if decoding were nothing but expert I/O, and the I/O time per
token at the expert store's rate measured on this machine (on Moonlight's experts: an extrapolation to the flagships'
bytes, labelled as such). Writes `feasibility.json` next to this file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# The paper's configuration (every reported result): two bits per coordinate per pass, groups of 512.
BITS_PER_PASS = 2
GROUP = 512
SCREENING_TARGET = 0.20  # the brief's screening target: a 20% reduction of effective bytes
DTYPE_BITS = {"BF16": 16, "F16": 16, "F32": 32, "U8": 8, "I8": 8}


def apexquant_bits_per_weight(passes: int, bits: int = BITS_PER_PASS, group: int = GROUP) -> float:
    """The paper's storage accounting: P (b + 16/g) bits per weight (codes and one fp16 scale per group per pass)."""
    return passes * (bits + 16 / group)


def expert_matrices(model: dict) -> list[tuple[str, int, int]]:
    """(name, rows, columns) of one routed expert's matrices, the columns being each row's input coordinates."""
    hidden, inter = model["hidden_size"], model["moe_intermediate_size"]
    return [("gate", inter, hidden), ("up", inter, hidden), ("down", hidden, inter)]


def native_expert_bytes(record: dict, experts: int) -> tuple[int, int]:
    """Stored bytes of one expert (data and scales) and its weight count, from a layer's tensors in the header record."""
    total = 0
    weights = 0
    for tensor in record["tensors"]:
        per_expert = tensor["bytes"]
        if ".experts.0." not in tensor["tensor"]:  # a fused tensor of the whole layer
            per_expert //= experts
        total += per_expert
        name = tensor["tensor"]
        if tensor["dtype"] in ("BF16", "F16") or (tensor["dtype"] == "U8" and not name.endswith(("scale", ".qs"))):
            if tensor["dtype"] == "U8":
                weights += per_expert * 2  # two 4-bit values per byte
            else:
                weights += per_expert * 8 // DTYPE_BITS[tensor["dtype"]]
    return total, weights


def screen(geometry: dict, headers: dict[str, dict], store_rate: float | None) -> dict:
    out = {}
    for key, model in geometry["models"].items():
        experts, per_token = model["routed_experts"], model["experts_per_token"] * model["moe_layers"]
        matrices = expert_matrices(model)
        weights = sum(rows * cols for _, rows, cols in matrices)
        formats = {}
        for label in (model["native"], model["original"]):
            if label is None:
                continue
            stored, counted = native_expert_bytes(headers[label], experts)
            if counted != weights:
                raise ValueError(f"{label}: the header's tensors hold {counted} weights per expert, not {weights}")
            formats[label] = {"bits_per_weight": 8 * stored / weights, "expert_bytes": stored, "source": "header"}
        for passes in (1, 2, 3):
            bits = apexquant_bits_per_weight(passes)
            formats[f"apexquant-p{passes}"] = {
                "bits_per_weight": bits,
                "expert_bytes": round(weights * bits / 8),
                "source": "the paper's accounting P(b + 16/g), b=2, g=512",
            }
        native = formats[model["native"]]["expert_bytes"]
        for entry in formats.values():
            entry["token_bytes_cold"] = entry["expert_bytes"] * per_token
            entry["ratio_to_native"] = entry["expert_bytes"] / native
            reduction = 1 - entry["ratio_to_native"]
            entry["byte_reduction"] = reduction
            # If a decode token were nothing but expert I/O, a byte reduction r bounds its speedup by 1 / (1 - r).
            entry["io_bound_speedup_bound"] = 1 / entry["ratio_to_native"]
            entry["meets_screening_target"] = reduction >= SCREENING_TARGET
            if store_rate:
                entry["io_seconds_per_token_cold_extrapolated"] = entry["token_bytes_cold"] / store_rate
        out[key] = {
            "name": model["name"],
            "expert_weights": weights,
            "experts_per_token": per_token,
            "matrices": [{"name": n, "rows": r, "columns": c, "whole_groups": c % GROUP == 0} for n, r, c in matrices],
            "groups_fit": all(c % GROUP == 0 for _, _, c in matrices),
            "formats": formats,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store-check", default=str(ROOT / "experiments/phase7a/store-moonlight/store-check.json"))
    args = parser.parse_args()
    geometry = json.loads((HERE / "geometry.json").read_text(encoding="utf-8"))
    headers = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((HERE / "headers").glob("*.json"))}
    store_rate = None
    check = Path(args.store_check)
    if check.is_file():
        record = json.loads(check.read_text(encoding="utf-8"))
        batches = next(p for p in record["passes"] if p["pass"] == "batches")
        store_rate = batches["gb_per_second"] * 1e9
    result = {
        "kind": "screening arithmetic: bytes from headers and the paper's accounting; no ApexQuant code, no weights",
        "screening_target": SCREENING_TARGET,
        "store_rate_bytes_per_second": store_rate,
        "store_rate_source": "experiments/phase7a/store-moonlight (Moonlight-16B experts, batched leases, this "
        "machine's NVMe): measured there, extrapolated to the flagships' bytes" if store_rate else None,
        "models": screen(geometry, headers, store_rate),
    }
    (HERE / "feasibility.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for key, model in result["models"].items():
        print(f"\n{model['name']}: {model['experts_per_token']} routed experts per decode token; rows in whole groups "
              f"of {GROUP}: {model['groups_fit']}")
        print("| format | bits/weight | MB/expert | GB/token (cold) | vs native | I/O-bound speedup bound | s/token I/O |")
        for label, entry in model["formats"].items():
            seconds = entry.get("io_seconds_per_token_cold_extrapolated")
            print(f"| {label} | {entry['bits_per_weight']:.4f} | {entry['expert_bytes'] / 1e6:.2f} | "
                  f"{entry['token_bytes_cold'] / 1e9:.2f} | {entry['ratio_to_native']:.4f} | "
                  f"{entry['io_bound_speedup_bound']:.3f} | {'-' if seconds is None else f'{seconds:.2f}'} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
