"""The study's tables and gate verdicts, from the run records (nothing is re-measured here).

    python research/lossless_moe/report.py --stage-a experiments/lossless_moe/stage-a-bf16-run1 \
        --stage-b experiments/lossless_moe/stage-b-bf16-run1 --stage-q experiments/lossless_moe/stage-q-run1 \
        [--diagnostic experiments/lossless_moe/stage-q-diagnostic-run1] [--stage-d experiments/lossless_moe/stage-d] \
        --output experiments/lossless_moe/report

Context deltas are always computed on matched matrices: a context that exists only for some matrices (the up matrix
given its gate, the down matrix given its neurons, a matrix given the previous sampled expert) is compared with the
order-0 code of exactly those matrices. Three estimators per delta (see `entropy`): in-sample (optimistic: free tables
fitted to the data), adaptive (achievable, every learning cost included) and two-part (achievable static tables per
matrix, their bytes included); held-out tables come from the calibration matrices of the same kind.

The gate (configuration, fixed before measurement) compares the best achievable code length with the strongest
practical static rANS coder's measured bytes, on the same matrices.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lossless_moe import corpus, records  # noqa: E402


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def matched(per_matrix: list[dict], prefix_of) -> dict:
    """{(group, field, context): {est: delta bits/weight, weights}} against 'none' on the same matrices."""
    by = defaultdict(dict)
    for r in per_matrix:
        by[(r["key"], r["field"])][r["context"]] = r
    acc = defaultdict(lambda: defaultdict(float))
    for (key, field), d in by.items():
        if "none" not in d:
            continue
        group = prefix_of(key)
        for ctx, r in d.items():
            n = r["weights"]
            a = acc[(group, field, ctx)]
            a["weights"] += n
            for est in ("plugin", "kt", "two_part"):
                a[est] += (r[est] - d["none"][est]) * n
                a[f"base_{est}"] += d["none"][est] * n
    out = {}
    for k, a in acc.items():
        w = a["weights"]
        out[k] = {"weights": int(w), **{e: a[e] / w for e in ("plugin", "kt", "two_part")},
                  **{f"base_{e}": a[f"base_{e}"] / w for e in ("plugin", "kt", "two_part")}}
    return out


def held_out_best(contexts: list[dict], group_prefix: str, field: str) -> dict:
    """Per held-out group (kind x split kind): the order-0 held-out code and the best context's, by prior-KT(256)."""
    out = {}
    groups = sorted({r["group"] for r in contexts if r["group"].startswith(group_prefix) and "held_out" in r})
    for g in groups:
        rows = {r["context"]: r["held_out"] for r in contexts if r["group"] == g and r["field"] == field and "held_out" in r}
        if "none" not in rows:
            continue
        base = rows["none"]["prior_kt"]["256"]
        best = min(rows.items(), key=lambda kv: kv[1]["prior_kt"]["256"])
        out[g] = {"none_cross": rows["none"]["cross"], "none_prior_kt": base, "best_context": best[0],
                  "best_prior_kt": best[1]["prior_kt"]["256"], "delta": best[1]["prior_kt"]["256"] - base,
                  "weights": rows["none"]["weights"]}
    return out


def ratio_table(rows: list[dict], codecs: dict[str, callable], denominator) -> dict:
    total = sum(denominator(r) for r in rows)
    return {name: sum(fn(r) for r in rows) / total for name, fn in codecs.items()}


# ---------------------------------------------------------------------------------------------------------------------


def bf16_section(stage_a: Path, stage_b: Path) -> tuple[dict, list[str]]:
    blocks = jsonl(stage_a / "blocks.jsonl")
    summary_a = json.loads((stage_a / "summary.json").read_text(encoding="utf-8"))
    per_matrix = jsonl(stage_b / "per_matrix.jsonl")
    contexts = json.loads((stage_b / "contexts.json").read_text(encoding="utf-8"))
    models = sorted({b["model"] for b in blocks})
    res: dict = {"models": {}}
    md = ["## BF16", "", "### Stage A: stored bytes / BF16 bytes (same bytes for every codec)", "",
          "| Model | Matrices | Weights (M) | zstd-3 | zstd-19 | zstd-19 byte split | zstd-19 planes | nvCOMP ANS | "
          "lmc order-0 (15-bit) | lmc order-0 (12-bit) | lmc previous exponent | lmc column class | lmc row class | "
          "lmc order-0 bits/weight |", "| --- | " + "--- | " * 13]
    for m in models:
        rs = [b for b in blocks if b["model"] == m]
        w = sum(b["weights"] for b in rs)
        codecs = {
            "zstd-3": lambda b: b["python"]["zstd-3/raw"]["bytes"],
            "zstd-19": lambda b: b["python"]["zstd-19/raw"]["bytes"],
            "zstd-19 byte split": lambda b: b["python"]["zstd-19/byte_split"]["bytes"],
            "zstd-19 planes": lambda b: b["python"]["zstd-19/planes"]["bytes"],
            "nvCOMP ANS": lambda b: b["nvcomp_ans_float16_1m"]["bytes"],
            **{f"lmc {v}": (lambda v: lambda b: b["lmc"][v]["total_bytes"])(v)
               for v in ("order0", "order0_s12", "left", "col16", "row16")},
        }
        r = ratio_table(rs, codecs, lambda b: b["stored_bytes"])
        sub = [b for b in rs if "lzma-9e/raw" in b["python"]]
        r_sub = ratio_table(sub, {
            "lzma-9e": lambda b: b["python"]["lzma-9e/raw"]["bytes"],
            "lzma-9e byte split": lambda b: b["python"]["lzma-9e/byte_split"]["bytes"],
            "bzip2-9": lambda b: b["python"]["bzip2-9/raw"]["bytes"],
            "zstd-22": lambda b: b["python"]["zstd-22/raw"]["bytes"],
            "zstd-19 planes": lambda b: b["python"]["zstd-19/planes"]["bytes"],
            "lmc order0": lambda b: b["lmc"]["order0"]["total_bytes"]}, lambda b: b["stored_bytes"])
        ev = [b for b in rs if b["split"] == "evaluation"]
        trained = ratio_table(ev, {"order0": lambda b: b["lmc"]["order0"]["total_bytes"],
                                   "trained": lambda b: b["lmc"]["trained"]["total_bytes"]}, lambda b: b["stored_bytes"])
        bits = sum(b["lmc"]["order0"]["total_bytes"] for b in rs) * 8 / w
        res["models"][m] = {"matrices": len(rs), "weights": w, "ratios": r, "subset_ratios": r_sub,
                            "evaluation_trained_vs_per_matrix": trained, "static_rans_bits_per_weight": bits}
        md.append(f"| {m} | {len(rs)} | {w / 1e6:.1f} | " + " | ".join(f"{r[k]:.4f}" for k in codecs) + f" | {bits:.3f} |")
    md += ["", "Subset (the first sampled expert of each layer): LZMA preset 9e, bzip2 9, zstd 22 against lmc order-0 on the "
           "same matrices.", "", "| Model | LZMA 9e | LZMA 9e byte split | bzip2 9 | zstd 22 | zstd-19 planes | lmc order-0 |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for m in models:
        r = res["models"][m]["subset_ratios"]
        md.append(f"| {m} | " + " | ".join(f"{v:.4f}" for v in r.values()) + " |")
    md += ["", "### lmc decode throughput (GB/s of BF16 output, best of repeats, decode tables built in the timed region)", "",
           "| Variant | 1 thread | 6 threads | 12 threads | peak working set (GB) |", "| --- | --- | --- | --- | --- |"]
    res["decode"] = {}
    for v, d in summary_a["lmc"].items():
        speeds = {t["threads"]: d["total_output_bytes"] / t["lmc"]["best_s"] / 1e9 for t in d["timing"]}
        res["decode"][v] = speeds
        md.append(f"| {v} | " + " | ".join(f"{speeds.get(t, float('nan')):.2f}" for t in (1, 6, 12))
                  + f" | {d['peak_working_set_bytes'] / 1e9:.2f} |")
    py = {name: sum(b["stored_bytes"] for b in blocks) / sum(b["python"][name]["decompress_s"] for b in blocks) / 1e9
          for name in ("zstd-3/raw", "zstd-19/raw", "zstd-19/byte_split", "zstd-19/planes")}
    res["python_decode_single_thread_gb_s"] = py
    md += ["", "Python codecs, single thread (GB/s of BF16 out): " + ", ".join(f"{k} {v:.2f}" for k, v in py.items()), ""]
    # Stage B
    deltas = matched(per_matrix, lambda key: key.split("/")[0])
    res["oracle"] = oracle_gains(per_matrix, lambda key: key.split("/")[0], "exponent")
    md += ["### Stage B: bits per weight beyond order-0, matched matrices", "",
           "| Model | Field | Order-0 (adaptive) | Context | In-sample | Adaptive | Two-part |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    res["stage_b"] = {}
    for m in models:
        res["stage_b"][m] = {}
        for field in ("exponent", "mantissa", "mantissa_top3", "sign"):
            rows = {k[2]: v for k, v in deltas.items() if k[0] == m and k[1] == field and k[2] != "none"}
            if not rows:
                continue
            best = sorted(rows.items(), key=lambda kv: kv[1]["kt"])
            res["stage_b"][m][field] = {"base_kt": deltas[(m, field, "none")]["base_kt"], "contexts": rows,
                                        "best_adaptive": best[0][0], "best_adaptive_delta": best[0][1]["kt"],
                                        "best_in_sample_delta": min(v["plugin"] for v in rows.values())}
            for ctx, v in best[:3]:
                md.append(f"| {m} | {field} | {deltas[(m, field, 'none')]['base_kt']:.4f} | {ctx} | {v['plugin']:+.4f} | "
                          f"{v['kt']:+.4f} | {v['two_part']:+.4f} |")
    md += ["", "Held-out (tables from calibration matrices of the same kind, scored on evaluation matrices; prior-KT, "
           "strength 256): order-0 and the best context per group.", "",
           "| Group | Order-0 | Best context | Delta |", "| --- | --- | --- | --- |"]
    res["held_out"] = {}
    for m in models:
        h = held_out_best(contexts, f"{m}/", "exponent")
        res["held_out"][m] = h
        for g, v in h.items():
            md.append(f"| {g} | {v['none_prior_kt']:.4f} | {v['best_context']} | {v['delta']:+.4f} |")
    md.append("")
    return res, md


def quantized_section(stage_q: Path, title: str) -> tuple[dict, list[str]]:
    blocks = jsonl(stage_q / "blocks.jsonl")
    summary = json.loads((stage_q / "summary.json").read_text(encoding="utf-8"))
    per_matrix = jsonl(stage_q / "per_matrix.jsonl")
    contexts = json.loads((stage_q / "contexts.json").read_text(encoding="utf-8"))
    sources = sorted({b["source"] for b in blocks})
    res: dict = {"sources": {}}
    md = [f"## {title}", "", "### Stage A: stored bytes / stored bytes of the container (weights plus scales)", "",
          "| Source | Format | Matrices | Weights (M) | Scales share | zstd-19 | nvCOMP ANS | Colibri g0 | Colibri per-matrix | "
          "lmc order-0 | lmc byte pairs | lmc best context | lmc order-0 bits/weight |",
          "| --- | " + "--- | " * 12]

    def scales_coded(b):
        return b["lmc"]["scales_order0"]["total_bytes"] if "scales_order0" in b.get("lmc", {}) else b["scale_bytes"]

    for s in sources:
        rs = [b for b in blocks if b["source"] == s]
        w = sum(b["weights"] for b in rs)
        stored = lambda b: b["weight_bytes"] + b["scale_bytes"]  # noqa: E731
        codecs = {
            "zstd-19": lambda b: b["python"]["zstd-19/weights"]["bytes"] + b["python"]["zstd-19/scales"]["bytes"],
            "nvCOMP ANS": lambda b: (b["nvcomp_ans_char_1m"]["bytes"] if "nvcomp_ans_char_1m" in b else b["weight_bytes"])
            + b["scale_bytes"],
            "Colibri g0": lambda b: b["lmc"]["colibri_g0"]["colibri_record_bytes"] + b["scale_bytes"],
            "Colibri per-matrix": lambda b: b["lmc"]["colibri_tensor"]["colibri_record_bytes"] + 64 + b["scale_bytes"],
            "lmc order0": lambda b: b["lmc"]["order0"]["total_bytes"] + scales_coded(b),
            "lmc pairs": lambda b: b["lmc"]["pairs"]["total_bytes"] + scales_coded(b),
        }
        r = ratio_table(rs, codecs, stored)
        r["Colibri g0"] += 64 / sum(stored(b) for b in rs)  # its one table per source
        ctx_variants = {v: ratio_table(rs, {v: (lambda v: lambda b: b["lmc"][v]["total_bytes"] + scales_coded(b))(v)},
                                       stored)[v] for v in ("left", "col16", "row16", "scale16")}
        best_ctx = min(ctx_variants.items(), key=lambda kv: kv[1])
        bits = sum(b["lmc"]["order0"]["total_bytes"] + scales_coded(b) for b in rs) * 8 / w
        share = sum(b["scale_bytes"] for b in rs) / sum(stored(b) for b in rs)
        scale_raw = sum(b["scale_bytes"] for b in rs)
        scale_coded = sum(scales_coded(b) for b in rs)
        res["sources"][s] = {"format": rs[0]["format"], "matrices": len(rs), "weights": w, "ratios": r,
                             "context_variants": ctx_variants, "static_rans_bits_per_weight": bits,
                             "scale_share": share, "scale_raw_bytes": scale_raw, "scale_coded_bytes": scale_coded,
                             "zero_level_share": None}
        md.append(f"| {s} | {rs[0]['format']} | {len(rs)} | {w / 1e6:.1f} | {share:.4f} | " +
                  " | ".join(f"{r[k]:.4f}" for k in codecs) + f" | {best_ctx[1]:.4f} ({best_ctx[0]}) | {bits:.4f} |")
    md += ["", "### Decode throughput (GB/s of stored weight bytes out, best of repeats)", "",
           "| Variant | Threads | lmc | Colibri scalar | Colibri branch-free |", "| --- | --- | --- | --- | --- |"]
    res["decode"] = {}
    for v, d in summary["lmc"].items():
        out = d["total_output_bytes"]
        for t in d["timing"]:
            cells = {"lmc": out / t["lmc"]["best_s"] / 1e9}
            for arm in ("colibri_scalar", "colibri_scalar_bf"):
                if arm in t:
                    cells[arm] = out / t[arm]["best_s"] / 1e9
            res["decode"].setdefault(v, {})[t["threads"]] = cells
            md.append(f"| {v} | {t['threads']} | {cells['lmc']:.3f} | " +
                      " | ".join(f"{cells[a]:.3f}" if a in cells else "" for a in ("colibri_scalar", "colibri_scalar_bf")) + " |")
    deltas = matched(per_matrix, lambda key: key.split("/")[0])
    res["oracle"] = oracle_gains(per_matrix, lambda key: key.split("/")[0], "code")
    md += ["", "### Stage B: bits per weight beyond order-0, matched matrices", "",
           "| Source | Field | Order-0 (adaptive) | Context | In-sample | Adaptive | Two-part |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    res["stage_b"] = {}
    for s in sources:
        res["stage_b"][s] = {}
        fields = sorted({k[1] for k in deltas if k[0] == s})
        for field in fields:
            rows = {k[2]: v for k, v in deltas.items() if k[0] == s and k[1] == field and k[2] != "none"}
            if not rows:
                continue
            best = sorted(rows.items(), key=lambda kv: kv[1]["kt"])
            res["stage_b"][s][field] = {"base_kt": deltas[(s, field, "none")]["base_kt"], "contexts": rows,
                                        "best_adaptive": best[0][0], "best_adaptive_delta": best[0][1]["kt"],
                                        "best_in_sample_delta": min(v["plugin"] for v in rows.values())}
            for ctx, v in best[:4]:
                md.append(f"| {s} | {field} | {deltas[(s, field, 'none')]['base_kt']:.4f} | {ctx} | {v['plugin']:+.4f} | "
                          f"{v['kt']:+.4f} | {v['two_part']:+.4f} |")
    md += ["", "Held-out (prior-KT 256): order-0 and the best context per group.", "",
           "| Group | Order-0 | Best context | Delta |", "| --- | --- | --- | --- |"]
    res["held_out"] = {}
    for s in sources:
        h = held_out_best(contexts, f"{s}/", "code")
        res["held_out"][s] = h
        for g, v in h.items():
            md.append(f"| {g} | {v['none_prior_kt']:.4f} | {v['best_context']} | {v['delta']:+.4f} |")
    md.append("")
    return res, md


def oracle_gains(per_matrix: list[dict], prefix_of, primary: str) -> dict:
    """Per group: the gain over order-0, in bits per weight, when every matrix and every field uses its own best
    context chosen in hindsight (signalling the choice costs a few bits per matrix: negligible). Adaptive deltas give the
    achievable gain (learning costs included), in-sample deltas the optimistic one. Fields are disjoint parts of the
    stored bits (mantissa_top3 lies inside the mantissa and is left out); a field's per-symbol deltas are weighted by its
    symbols and divided by the group's weights (the primary field's symbols), so that scale streams (one symbol per 32
    weights) and contexts that exist for some matrices only count for what they cover."""
    by = defaultdict(dict)
    for r in per_matrix:
        by[(r["key"], r["field"])][r["context"]] = r
    acc = defaultdict(lambda: defaultdict(float))
    for (key, field), d in by.items():
        if "none" not in d or field == "mantissa_top3":
            continue
        group = prefix_of(key)
        n = d["none"]["weights"]
        if field == primary:
            acc[group]["weights"] += n
        for est in ("kt", "plugin"):
            best = min(r[est] - d["none"][est] for r in d.values())
            acc[group][est] += min(0.0, best) * n
            acc[group][f"{est}:{field}"] += min(0.0, best) * n
    out = {}
    for group, a in acc.items():
        w = a["weights"]
        out[group] = {"weights": int(w), "achievable_gain_bits": -a["kt"] / w, "optimistic_gain_bits": -a["plugin"] / w,
                      "by_field": {k.split(":", 1)[1]: -v / w for k, v in a.items() if k.startswith("kt:")}}
    return out


def gate(res_q: dict, res_bf16: dict, config: dict) -> tuple[dict, list[str]]:
    """Reduction of the best achievable code against the strongest static rANS bytes, per source and per model."""
    g = config["gates"]
    md = ["## Gate (pre-registered)", "",
          "Best achievable reduction against the strongest practical static rANS coder (lmc order-0 with every header, "
          "table and stream byte, the scales coded order-0 where they are compressible), on the same matrices. Every "
          "matrix and field takes its own best context in hindsight. Achievable: adaptive code lengths (all learning "
          "costs included). Optimistic: in-sample conditional entropies (free tables fitted to the data; biased low).", "",
          "| Corpus | Static rANS (bits/weight) | Achievable gain | Optimistic gain | Achievable reduction | Optimistic reduction | Stage B verdict |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    out = {}

    def verdict(r: float) -> str:
        if r >= g["stage_b_plausible"]:
            return "plausible"
        if r >= g["stage_b_limited_prototype"]:
            return "uncertain: limited prototype"
        return "unsupported"

    for name, sec, kind in [(k, v, "q") for k, v in res_q.get("sources", {}).items()] + \
                           [(k, v, "bf16") for k, v in res_bf16.get("models", {}).items()]:
        o = (res_q if kind == "q" else res_bf16)["oracle"][name]
        ach, opt = -o["achievable_gain_bits"], -o["optimistic_gain_bits"]
        base = sec["static_rans_bits_per_weight"]
        red_a, red_o = -ach / base, -opt / base
        out[name] = {"static_rans_bits": base, "achievable_gain_bits": -ach, "optimistic_gain_bits": -opt,
                     "achievable_reduction": red_a, "optimistic_reduction": red_o, "verdict": verdict(red_o)}
        md.append(f"| {name} | {base:.4f} | {-ach:.4f} | {-opt:.4f} | {red_a:.2%} | {red_o:.2%} | {verdict(red_o)} |")
    md.append("")
    return out, md


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage-a", type=Path, required=True)
    parser.add_argument("--stage-b", type=Path, required=True)
    parser.add_argument("--stage-q", type=Path, default=None)
    parser.add_argument("--diagnostic", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = corpus.load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    result: dict = {}
    md = ["# Lossless MoE study: tables", ""]
    bf16, part = bf16_section(args.stage_a, args.stage_b)
    result["bf16"] = bf16
    md += part
    q: dict = {"sources": {}, "stage_b": {}}
    if args.stage_q:
        q, part = quantized_section(args.stage_q, "Quantized experts (fetched sample)")
        result["quantized"] = q
        md += part
    if args.diagnostic:
        dq, part = quantized_section(args.diagnostic, "Diagnostic: Moonlight re-quantized per-row int4 (not a principal result)")
        result["diagnostic"] = dq
        md += part
    gates, part = gate(q, bf16, config)
    result["gate"] = gates
    md += part
    if args.diagnostic:
        dg, part = gate(result["diagnostic"], {"models": {}, "stage_b": {}, "oracle": {}}, config)
        result["gate_diagnostic"] = dg
        md += ["### Diagnostic gate (not a principal result)", ""] + part[2:]
    records.write_json(args.output / "results.json", result)
    (args.output / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
