"""Stages A and B on quantized experts (int4 per-row, int4 group-32, MXFP4): the fetched sample, every codec on the
same stored bytes, then the context measurements.

    python research/lossless_moe/stage_q.py --output experiments/lossless_moe/stage-q --manifest experiments/lossless_moe/sample-manifest.json

Per matrix (weights and their scales exactly as stored, sha256 checked against the manifest):

  Stage A   zstd 3 and 19 on the weight bytes, zstd 19 on the scale bytes; LZMA 9e and bzip2 9 on a subset; nvCOMP's ANS
            (byte mode, 1 MiB chunks). Colibri's int4-rans256-g0 codec (its own C, 256 streams) with its own table rule
            (one table from the pooled histogram of every matrix of the source: `g0`) and with a table per matrix.
            The study's coder (lmc): the nibbles order-0 per matrix (14-bit tables, the strongest static reference),
            the packed bytes order-0 (two nibbles per decode step, 12-bit tables of 256 symbols: the speed variant),
            or under contexts (previous code; a class per column; a class per row; the class of each weight's group
            scale, derived from the stored scales); the scale stream order-0 or under the previous scale.
  Stage B   the nibbles and the scale streams under every context model (`quantized.nibble_specs`, `scale_specs`):
            in-sample, two-part and adaptive code lengths; tables trained on calibration matrices scored on evaluation
            matrices, held out by expert and by layer.

Every coded matrix is decoded and compared with its stored bytes; lmc and Colibri outputs by sha256 as well.
"""

from __future__ import annotations

import argparse
import bz2
import importlib.util
import json
import lzma
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import zstandard

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lossless_moe import corpus, measure, quantized, records  # noqa: E402

LZMA_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]
COLIBRI_FORMAT = HERE / "third_party" / "colibri" / "c" / "tools" / "rans_format.py"


def colibri_format():
    """Colibri's rans_format.py, unmodified (its frequency quantizer is the table rule of int4-rans256-g0)."""
    spec = importlib.util.spec_from_file_location("colibri_rans_format", COLIBRI_FORMAT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


def python_codecs(m: quantized.QMatrix, subset: bool) -> dict:
    raw = m.weight_bytes.tobytes()
    res = {}

    def run(name, compress, decompress, data):
        packed, cs = timed(compress, data)
        back, ds = timed(decompress, packed)
        if back != data:
            raise SystemExit(f"{m.key} {name}: inexact")
        res[name] = {"bytes": len(packed), "exact": True, "compress_s": cs, "decompress_s": ds}

    d = zstandard.ZstdDecompressor()
    for level in (3, 19):
        run(f"zstd-{level}/weights", zstandard.ZstdCompressor(level=level).compress, d.decompress, raw)
    run("zstd-19/scales", zstandard.ZstdCompressor(level=19).compress, d.decompress, m.scale_bytes.tobytes())
    if subset:
        run("lzma-9e/weights", lambda b: lzma.compress(b, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS),
            lambda b: lzma.decompress(b, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS), raw)
        run("bzip2-9/weights", lambda b: bz2.compress(b, 9), bz2.decompress, raw)
    return res


def nvcomp_ans(m: quantized.QMatrix) -> dict:
    import torch

    from awpmi.streaming.codec import encode_rows

    rows = torch.from_numpy(m.weight_bytes.reshape(1, -1).copy()).cuda()
    chunks = encode_rows("ans", {"data_type": "char"}, rows, 1 << 20)[0]
    return {"bytes": int(sum(c.numel() for c in chunks)) + 8 * len(chunks), "chunks": len(chunks), "exact": True}


def write_inputs(scratch: Path, m: quantized.QMatrix) -> dict:
    base = scratch / m.key.replace("/", "_")
    nib = m.nibbles()
    nib.astype(np.uint8).tofile(f"{base}.nib")
    m.weight_bytes.tofile(f"{base}.packed")
    mag, top, enc = quantized.magnitudes(m, nib)
    col = np.searchsorted(np.quantile(mag.mean(axis=0), np.linspace(0, 1, 17)[1:-1]), mag.mean(axis=0), side="right")
    row = np.searchsorted(np.quantile(mag.mean(axis=1), np.linspace(0, 1, 17)[1:-1]), mag.mean(axis=1), side="right")
    col.astype(np.uint8).tofile(f"{base}.col16")
    row.astype(np.uint8).tofile(f"{base}.row16")
    gsc = quantized.group_scale_classes(m)
    gpr = m.cols // m.group if m.group < m.cols else 1
    np.ascontiguousarray(gsc[:, :: m.group][:, :gpr]).astype(np.uint8).tofile(f"{base}.scale16")
    info = {"base": str(base), "encoding": enc, "nibble_counts": np.bincount(nib.ravel(), minlength=16).tolist()}
    if m.format == "mxfp4":
        m.scale_bytes.tofile(f"{base}.scl")
        info["scale_kind"] = "bytes"
    elif m.format == "int4_g32_bf16":
        p = m.scale_bytes.view("<u2").reshape(m.rows, -1)
        e = ((p >> 7) & 0xFF).astype(np.uint8)
        sm = (((p >> 15) << 7) | (p & 0x7F)).astype(np.uint8)
        e.tofile(f"{base}.scl")
        sm.tofile(f"{base}.sclraw")
        m.scale_bytes.tofile(f"{base}.sclorig")
        info["scale_kind"] = "bf16"
    else:
        info["scale_kind"] = "raw"
    return info


def colibri_tables(blocks: list[dict], fmt) -> dict:
    """Per source, Colibri's g0 table: rans_format.quantize_freq of the pooled nibble histogram of the source."""
    pooled: dict[str, np.ndarray] = {}
    for b in blocks:
        pooled.setdefault(b["source"], np.zeros(16, dtype=np.int64))
        pooled[b["source"]] += np.asarray(b["nibble_counts"], dtype=np.int64)
    return {s: [int(x) for x in fmt.quantize_freq(h)] for s, h in pooled.items()}


def lmc_run(blocks: list[dict], variant: str, out: Path, threads: list[int], repeats: int, g0: dict, fmt) -> dict:
    specs = []
    for b in blocks:
        base = b["base"]
        if variant.startswith("scales"):
            if b["scale_kind"] == "raw":
                continue
            rows, gpr = b["scale_shape"]
            spec = {"name": b["key"], "format": "bytes" if b["scale_kind"] == "bytes" else "bf16",
                    "symbols": f"{base}.scl", "original": f"{base}.scl" if b["scale_kind"] == "bytes" else f"{base}.sclorig",
                    "rows": rows, "cols": gpr, "alphabet": 256, "scale_bits": 12, "streams": 32}
            if b["scale_kind"] == "bf16":
                spec["raw"] = f"{base}.sclraw"
            if variant == "scales_left":
                spec.update(kind=3, n_contexts=257)
            specs.append(spec)
            continue
        spec = {"name": b["key"], "format": "int4", "symbols": f"{base}.nib", "original": f"{base}.packed",
                "rows": b["rows"], "cols": b["cols"], "alphabet": 16, "scale_bits": 14, "streams": 32}
        if variant == "pairs":
            spec = {"name": b["key"], "format": "bytes", "symbols": f"{base}.packed", "original": f"{base}.packed",
                    "rows": b["rows"], "cols": b["cols"] // 2, "alphabet": 256, "scale_bits": 12, "streams": 32}
        if variant in ("colibri_g0", "colibri_tensor"):
            freq = g0[b["source"]] if variant == "colibri_g0" else [int(x) for x in fmt.quantize_freq(np.asarray(b["nibble_counts"]))]
            spec.update(colibri_freq=freq, colibri_scale_bits=14)
        elif variant == "left":
            spec.update(kind=3, n_contexts=17, scale_bits=12)
        elif variant in ("col16", "row16"):
            spec.update(kind=1 if variant == "col16" else 2, side=f"{base}.{variant}", n_contexts=16, scale_bits=12)
        elif variant == "scale16":
            spec.update(kind=4, grid=f"{base}.scale16", grid_group=b["group"] if b["group"] < b["cols"] else b["cols"],
                        n_contexts=16, scale_bits=12)
        specs.append(spec)
    if not specs:
        return {}
    job = {"blocks": specs, "threads": threads, "repeats": repeats}
    path = out.with_suffix(".job.json")
    path.write_text(json.dumps(job), encoding="utf-8")
    done = subprocess.run([str(records.LMC), "bench", str(path), str(out)], capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"lmc {variant}: {done.stderr[-2000:]}")
    return json.loads(out.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--threads", default="1,6,12")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--keep-scratch", action="store_true")
    args = parser.parse_args()
    config = corpus.load_config(args.config)
    config["samples"] = {**config.get("samples", {}), **config.get("diagnostic", {})}
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    scratch = out / "scratch"
    scratch.mkdir(exist_ok=True)
    shutil.copy(args.config, out / "config.yaml")
    shutil.copy(args.manifest, out / "sample-manifest.json")
    used = {sl["source"] for sl in json.loads(args.manifest.read_text(encoding="utf-8"))["slices"]}
    records.write_json(out / "environment.json", records.environment(
        {n: {k: s.get(k) for k in ("repository", "revision", "derived_from", "recipe")} for n, s in config["samples"].items()
         if n in used}))
    fmt = colibri_format()
    matrices = quantized.load(args.manifest, config)
    print(f"{len(matrices)} matrices from {len({m.source for m in matrices})} sources", flush=True)
    acc = measure.Accumulator()
    blocks = []
    first_expert: dict[tuple[str, int], int] = {}
    t0 = time.perf_counter()
    by_expert: dict[tuple[str, int, int], dict[str, quantized.QMatrix]] = {}
    for m in matrices:
        by_expert.setdefault((m.source, m.layer, m.expert), {})[m.kind] = m
    for (source, layer, expert), mats in by_expert.items():
        spec = config["samples"][source]
        layers = list(spec["layers"])
        experts = corpus.sample_experts(config["seed"], spec.get("sample_as", source), layer, spec["experts"],
                                        spec["experts_per_layer"])
        position = experts.index(expert)
        split_e = "calibration" if position % 2 == 0 else "evaluation"
        split_l = "calibration" if layers.index(layer) < len(layers) // 2 else "evaluation"
        subset = first_expert.setdefault((source, layer), expert) == expert
        for kind, m in mats.items():
            nib = m.nibbles()
            extra: dict = {"given": {}}
            if kind == "up" and "gate" in mats:
                extra["given"]["gate"] = mats["gate"].nibbles()
            if kind == "down" and "gate" in mats and "up" in mats:
                extra["neuron_classes"] = quantized.neuron_classes_from(mats["gate"], mats["up"])
            memberships = [(f"{source}/{kind}/experts", split_e), (f"{source}/{kind}/layers", split_l),
                           (f"{source}/all", "all")]
            for s in quantized.nibble_specs(m, nib, extra) + quantized.scale_specs(m):
                acc.add(m.key, memberships, s)
            entry = {"key": m.key, "source": source, "format": m.format, "layer": layer, "expert": expert, "kind": kind,
                     "split_experts": split_e, "split_layers": split_l, "rows": m.rows, "cols": m.cols, "group": m.group,
                     "weights": m.weights, "weight_bytes": int(m.weight_bytes.size), "scale_bytes": int(m.scale_bytes.size),
                     "weight_sha256": m.weight_sha256, "scale_sha256": m.scale_sha256,
                     "scale_shape": [m.rows, m.scale_bytes.size // m.rows // (4 if m.format == "int4_row" else
                                                                              2 if m.format == "int4_g32_bf16" else 1)]}
            entry["python"] = python_codecs(m, subset)
            if not args.no_gpu:
                entry["nvcomp_ans_char_1m"] = nvcomp_ans(m)
            entry.update(write_inputs(scratch, m))
            blocks.append(entry)
        print(f"  {source} L{layer} E{expert} ({time.perf_counter() - t0:.0f} s)", flush=True)
    g0 = colibri_tables(blocks, fmt)
    threads = [int(t) for t in args.threads.split(",")]
    repeats = int(config["timing"]["repeats"])
    lmc = {}
    variants = ["colibri_g0", "colibri_tensor", "order0", "pairs", "left", "col16", "row16", "scale16", "scales_order0",
                "scales_left"]
    for variant in variants:
        result = lmc_run(blocks, variant, out / f"lmc-{variant}.json", threads, repeats, g0, fmt)
        if not result:
            continue
        by = {r["name"]: r for r in result["blocks"]}
        for b in blocks:
            r = by.get(b["key"])
            if r is None:
                continue
            want = b["weight_sha256"] if not variant.startswith("scales") else b["scale_sha256"]
            if not r["exact"] or r["decoded_sha256"] != want:
                raise SystemExit(f"lmc {variant} {b['key']}: not exact")
            rec = {"total_bytes": r["sizes"]["total"] + r["raw_bytes"], "sizes": r["sizes"]}
            if "colibri" in r:
                if not all(r["colibri"]["exact"].values()):
                    raise SystemExit(f"Colibri {variant} {b['key']}: not exact")
                rec["colibri_record_bytes"] = r["colibri"]["record_bytes"]
            b.setdefault("lmc", {})[variant] = rec
        lmc[variant] = {"timing": result["timing"], "total_output_bytes": result["total_output_bytes"],
                        "peak_working_set_bytes": result["peak_working_set_bytes"]}
    names = sorted({k[2:] for k in acc.totals})
    groups = sorted({k[0] for k in acc.totals})
    table = []
    for group in groups:
        for field, context in names:
            row = {"group": group, "field": field, "context": context}
            for split in ("all", "calibration", "evaluation"):
                t = acc.totals.get((group, split, field, context))
                if t is not None:
                    row[split] = t.per_weight()
            held = acc.held_out(group, (field, context))
            if held is not None:
                row["held_out"] = held
            if len(row) > 3:
                table.append(row)
    records.write_json(out / "contexts.json", table)
    with open(out / "per_matrix.jsonl", "w", encoding="utf-8") as handle:
        for r in acc.per_matrix:
            handle.write(json.dumps(r) + "\n")
    with open(out / "blocks.jsonl", "w", encoding="utf-8") as handle:
        for b in blocks:
            handle.write(json.dumps({k: v for k, v in b.items() if k not in ("base", "nibble_counts")}) + "\n")
    summary = {"colibri_g0_tables": g0, "lmc": lmc, "seconds": time.perf_counter() - t0}
    summary["digest"] = records.digest({"contexts": table, "blocks": [{k: v for k, v in b.items() if k != "base"} for b in blocks]})
    records.write_json(out / "summary.json", summary)
    if not args.keep_scratch:
        shutil.rmtree(scratch)
    print(f"digest {summary['digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
