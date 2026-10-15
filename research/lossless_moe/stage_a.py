"""Stage A on BF16: the corpus and the baselines, every codec on exactly the same bytes.

    python research/lossless_moe/stage_a.py --output experiments/lossless_moe/stage-a-bf16

Per model of the configuration (local checkpoints only): the sampled experts' matrices are read twice (positioned reads;
safetensors' own loader) and must agree; the files touched are hashed in full against the publisher's sha256. Then, per
matrix, on its stored bytes:

  general-purpose   zstd 3 and 19 on the raw bytes; zstd 19 after Phase 5C's byte split and bit planes; on a subset (the
                    first sampled expert of each layer) LZMA preset 9e (raw and byte split), bzip2 9, zstd 22
  GPU               nvCOMP's ANS (float16 mode, 1 MiB chunks), Phase 6B's encoded-row codec
  rANS (lmc)        the study's codec: the exponent byte rANS-coded, sign and mantissa stored; tables per matrix
                    (order-0 at 15 bits: the strongest practical static rANS reference; at 12 bits: L1-resident
                    tables), or chosen by a context (the previous
                    exponent, a class per column, a class per row: real coded bytes for Stage B's best contexts), or one
                    table per model and matrix kind trained on calibration matrices and applied to evaluation ones

Every reconstruction is compared with the stored bytes, and every lmc output's sha256 with the safetensors copy's.
Timings are single-threaded for the Python codecs and at 1, 6 and 12 threads for lmc; nothing else runs meanwhile.
"""

from __future__ import annotations

import argparse
import bz2
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
sys.path.insert(0, str(HERE.parent / "expert_deltas"))

from expert_deltas import bits as bits5c  # noqa: E402  (Phase 5C's exact transforms, reused)
from lossless_moe import contexts, corpus, records, symbols  # noqa: E402

LZMA_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]
# lmc variants and their table precision: order-0 at 15 bits (the reference) and 12 bits (L1-resident decode tables);
# context variants at 12 bits (one table per context, built by the decoder per block); the trained table at 15 bits.
SCALE_BITS = {"order0": 15, "order0_s12": 12, "left": 12, "col16": 12, "row16": 12, "trained": 15}


def split_of(index_in_layer: int) -> str:
    return "calibration" if index_in_layer % 2 == 0 else "evaluation"


def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


def python_codecs(p: np.ndarray, subset: bool) -> dict:
    raw = p.tobytes()
    count = p.size
    results = {}

    def run(name, compress, decompress, data, restore):
        packed, cs = timed(compress, data)
        back, ds = timed(decompress, packed)
        exact = bool(np.array_equal(restore(back), p))
        results[name] = {"bytes": len(packed), "exact": exact, "compress_s": cs, "decompress_s": ds}
        if not exact:
            raise SystemExit(f"{name}: inexact reconstruction")

    as_patterns = lambda b: np.frombuffer(b, dtype=np.uint16).reshape(p.shape)  # noqa: E731
    from_split = lambda b: bits5c.byte_merge(b, count).reshape(p.shape)  # noqa: E731
    for level in (3, 19):
        c = zstandard.ZstdCompressor(level=level)
        d = zstandard.ZstdDecompressor()
        run(f"zstd-{level}/raw", c.compress, d.decompress, raw, as_patterns)
    c19, d = zstandard.ZstdCompressor(level=19), zstandard.ZstdDecompressor()
    run("zstd-19/byte_split", c19.compress, d.decompress, bits5c.byte_split(p), from_split)
    planes = bits5c.split_planes(p)
    names = sorted(planes)
    frames, cs = timed(lambda: [c19.compress(planes[n]) for n in names])
    back, ds = timed(lambda: {n: d.decompress(f) for n, f in zip(names, frames)})
    exact = bool(np.array_equal(bits5c.merge_planes(back, count).reshape(p.shape), p))
    results["zstd-19/planes"] = {"bytes": sum(len(f) for f in frames), "exact": exact, "compress_s": cs, "decompress_s": ds}
    if not exact:
        raise SystemExit("planes: inexact")
    if subset:
        run("lzma-9e/raw", lambda b: lzma.compress(b, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS),
            lambda b: lzma.decompress(b, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS), raw, as_patterns)
        run("lzma-9e/byte_split", lambda b: lzma.compress(b, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS),
            lambda b: lzma.decompress(b, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS), bits5c.byte_split(p), from_split)
        run("bzip2-9/raw", lambda b: bz2.compress(b, 9), bz2.decompress, raw, as_patterns)
        c22 = zstandard.ZstdCompressor(level=22)
        run("zstd-22/raw", c22.compress, zstandard.ZstdDecompressor().decompress, raw, as_patterns)
    return results


def nvcomp_ans(p: np.ndarray) -> dict:
    import torch

    from awpmi.streaming.codec import encode_rows

    rows = torch.from_numpy(p.reshape(1, -1).view(np.uint8).copy()).cuda()
    chunks = encode_rows("ans", {"data_type": "float16"}, rows, 1 << 20)[0]  # raises if the GPU round trip differs
    return {"bytes": int(sum(c.numel() for c in chunks)) + 8 * len(chunks), "chunks": len(chunks), "exact": True}


def write_block(scratch: Path, m: corpus.Matrix, p: np.ndarray) -> dict:
    s, e, mant = symbols.bf16_fields(p)
    raw = ((s << 7) | mant).astype(np.uint8)
    base = scratch / m.key.replace("/", "_")
    paths = {"symbols": f"{base}.exp", "raw": f"{base}.sm", "original": f"{base}.bf16"}
    e.tofile(paths["symbols"])
    raw.tofile(paths["raw"])
    p.astype("<u2").tofile(paths["original"])
    col = contexts.column_class(e, 16)
    row = contexts.row_class(e, 16)
    np.ascontiguousarray(col.ids[0]).astype(np.uint8).tofile(f"{base}.col16")
    np.ascontiguousarray(row.ids[:, 0]).astype(np.uint8).tofile(f"{base}.row16")
    return {"paths": paths, "base": str(base), "col_side_bits": col.side_bits, "row_side_bits": row.side_bits,
            "exponent_counts": np.bincount(e.ravel(), minlength=256).tolist()}


def lmc_job(blocks: list[dict], variant: str, out: Path, threads: list[int], repeats: int, tables_dir: Path) -> dict:
    specs = []
    for b in blocks:
        spec = {"name": b["key"], "format": "bf16", **b["paths"], "rows": b["rows"], "cols": b["cols"], "alphabet": 256,
                "scale_bits": SCALE_BITS[variant], "streams": 32}
        if variant == "left":
            spec.update(kind=3, n_contexts=257)
        elif variant in ("col16", "row16"):
            spec.update(kind=1 if variant == "col16" else 2, side=f"{b['base']}.{variant}", n_contexts=16)
        elif variant == "trained":
            spec.update(tables=str(tables_dir / f"{b['model']}_{b['kind']}.u32"))
        specs.append(spec)
    job = {"blocks": specs, "threads": threads, "repeats": repeats}
    job_path = out.with_suffix(".job.json")
    job_path.write_text(json.dumps(job), encoding="utf-8")
    done = subprocess.run([str(records.LMC), "bench", str(job_path), str(out)], capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"lmc {variant} failed: {done.stderr[-2000:]}")
    return json.loads(out.read_text(encoding="utf-8"))


def trained_tables(blocks: list[dict], tables_dir: Path) -> dict:
    """One exponent table per model and matrix kind from the calibration matrices' counts (KT-smoothed over the
    calibration support plus one count for every other symbol, so that held-out symbols stay codable); its cost is
    counted once per model and kind (256 frequencies of 16 bits)."""
    groups: dict[tuple[str, str], np.ndarray] = {}
    for b in blocks:
        if b["split"] == "calibration":
            groups.setdefault((b["model"], b["kind"]), np.zeros(256, dtype=np.int64))
            groups[(b["model"], b["kind"])] += np.asarray(b["exponent_counts"], dtype=np.int64)
    tables_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for (model, kind), counts in groups.items():
        smoothed = counts + 1  # every exponent codable
        m = 1 << 15
        freq = np.maximum(1, np.floor(smoothed / smoothed.sum() * m)).astype(np.int64)
        while freq.sum() > m:
            freq[np.argmax(freq)] -= 1
        while freq.sum() < m:
            freq[np.argmax(smoothed / smoothed.sum() * m - freq)] += 1
        freq.astype("<u4").tofile(tables_dir / f"{model}_{kind}.u32")
        out[f"{model}_{kind}"] = {"table_bits": 256 * 16, "calibration_weights": int(counts.sum())}
        assert int(freq.sum()) == m
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--models", default=None, help="comma-separated subset of the configuration's BF16 models")
    parser.add_argument("--threads", default="1,6,12")
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()
    config = corpus.load_config(args.config)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    scratch = out / "scratch"
    scratch.mkdir(exist_ok=True)
    shutil.copy(args.config, out / "config.yaml")
    if not records.LMC.exists():
        raise SystemExit(f"{records.LMC} is missing: cargo build --release in {records.CODEC}")
    models = config["corpus"]["bf16"]
    selected = args.models.split(",") if args.models else list(models)
    env = records.environment({name: {k: models[name][k] for k in ("repository", "revision")} for name in selected})
    records.write_json(out / "environment.json", env)
    threads = [int(t) for t in args.threads.split(",")]
    repeats = int(config["timing"]["repeats"])
    summary = {"models": {}}
    all_blocks = []
    for name in selected:
        ckpt = corpus.Checkpoint(name, models[name])
        pairs = ckpt.sample(config["seed"])
        per_layer: dict[int, list[int]] = {}
        for layer, expert in pairs:
            per_layer.setdefault(layer, []).append(expert)
        matrices = [(ckpt.matrix(l, e, k), per_layer[l].index(e)) for l, e in pairs for k in corpus.MATRICES]
        files = sorted({m.file for m, _ in matrices})
        print(f"[{name}] {len(pairs)} experts, {len(matrices)} matrices, files {files}", flush=True)
        verification = corpus.verify_files(ckpt, files)
        if not all(v["equal"] for v in verification.values()):
            raise SystemExit(f"{name}: a file differs from the publisher's sha256: {verification}")
        rows_out = []
        for m, position in matrices:
            p = ckpt.read_patterns(m)
            ref = ckpt.read_reference(m)
            if not np.array_equal(p, ref):
                raise SystemExit(f"{m.key}: the two read paths disagree")
            subset = position == 0
            entry = {"key": m.key, "model": name, "layer": m.layer, "expert": m.expert, "kind": m.kind,
                     "split": split_of(position), "rows": m.rows, "cols": m.cols, "weights": int(p.size),
                     "stored_bytes": m.nbytes, "sha256": records.sha256_bytes(ref.tobytes()), "matrix": m.to_json()}
            entry["python"] = python_codecs(p, subset)
            if not args.no_gpu:
                entry["nvcomp_ans_float16_1m"] = nvcomp_ans(p)
            entry.update(write_block(scratch, m, p))
            rows_out.append(entry)
            print(f"  {m.key}: zstd-19 {entry['python']['zstd-19/raw']['bytes'] / m.nbytes:.4f}", flush=True)
        summary["models"][name] = {"repository": ckpt.repository, "revision": ckpt.revision, "files": verification,
                                   "config": corpus.describe(ckpt), "experts": pairs}
        all_blocks += rows_out
    tables = trained_tables(all_blocks, scratch / "tables")
    lmc = {}
    for variant in SCALE_BITS:
        blocks = [b for b in all_blocks if variant != "trained" or b["split"] == "evaluation"]
        result = lmc_job(blocks, variant, out / f"lmc-{variant}.json", threads, repeats, scratch / "tables")
        by_name = {r["name"]: r for r in result["blocks"]}
        for b in blocks:
            r = by_name[b["key"]]
            if not r["exact"] or r["decoded_sha256"] != b["sha256"]:
                raise SystemExit(f"lmc {variant} {b['key']}: not exact against the safetensors copy")
            b.setdefault("lmc", {})[variant] = {"sizes": r["sizes"], "raw_bytes": r["raw_bytes"],
                                                "total_bytes": r["sizes"]["total"] + r["raw_bytes"]}
        lmc[variant] = {"timing": result["timing"], "total_output_bytes": result["total_output_bytes"],
                        "total_stored_bytes": result["total_stored_bytes"],
                        "peak_working_set_bytes": result["peak_working_set_bytes"]}
    with open(out / "blocks.jsonl", "w", encoding="utf-8") as handle:
        for b in all_blocks:
            slim = {k: v for k, v in b.items() if k not in ("paths", "base", "exponent_counts")}
            handle.write(json.dumps(slim) + "\n")
    summary["trained_tables"] = tables
    summary["lmc"] = lmc
    summary["digest"] = records.digest([{k: v for k, v in b.items() if k not in ("paths", "base")} for b in all_blocks])
    records.write_json(out / "summary.json", summary)
    if not args.keep_scratch:
        shutil.rmtree(scratch)
    print(f"digest {summary['digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
