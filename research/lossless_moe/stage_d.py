"""Stage D: the system side of the strongest static coder, on this machine.

    python research/lossless_moe/stage_d.py --output experiments/lossless_moe/stage-d \
        --manifest experiments/lossless_moe/sample-manifest.json

  random access   per source, two packs of the sampled experts on the system drive, each expert contiguous and 4 KiB
                  aligned: the stored bytes (weights and scales as the container holds them) and the lmc order-0 records
                  (scales stored as they are). `lmc random-access` reads every expert once, in a seeded random order,
                  with unbuffered (direct) reads: the compressed record then its decode, and the stored bytes; every
                  decoded matrix is checked against its sha256. Moonlight's BF16 experts too.
  granularity     the same coder with independent blocks of fewer rows (per matrix, 512, 64, 8 rows): what random
                  access at a finer grain costs in tables, headers and stream flushes.
  GPU decode      nvCOMP's ANS (byte mode) over 1 MiB and 64 KiB chunks of every weight byte of a source, decoded in one
                  batched launch on the RTX 4060 Ti, timed with CUDA events after a warm-up, output compared with input.

Nothing else runs meanwhile. Packs live in the run's scratch directory (gitignored) and are deleted at the end.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lossless_moe import corpus, quantized, records, symbols  # noqa: E402

ALIGN = 4096


def lmc(job: dict, path: Path) -> dict:
    path.with_suffix(".job.json").write_text(json.dumps(job), encoding="utf-8")
    done = subprocess.run([str(records.LMC), "bench", str(path.with_suffix(".job.json")), str(path)],
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"lmc bench: {done.stderr[-2000:]}")
    return json.loads(path.read_text(encoding="utf-8"))


def pad(f, align: int = ALIGN) -> None:
    f.write(b"\0" * (-f.tell() % align))


def experts_of(matrices: list) -> dict:
    out: dict = {}
    for m in matrices:
        out.setdefault((m.layer, m.expert), []).append(m)
    return out


def quantized_inputs(scratch: Path, m: quantized.QMatrix) -> dict:
    base = scratch / m.key.replace("/", "_")
    m.nibbles().astype(np.uint8).tofile(f"{base}.nib")
    m.weight_bytes.tofile(f"{base}.packed")
    return {"name": m.key, "format": "int4", "symbols": f"{base}.nib", "original": f"{base}.packed", "rows": m.rows,
            "cols": m.cols, "alphabet": 16, "scale_bits": 14, "streams": 32}


def random_access(name: str, experts: dict, records_dir: Path, scratch: Path, seed: int, stored_parts, record_parts) -> dict:
    """Write the stored pack and the compressed pack, then time cold reads of every expert."""
    stored_path, packed_path = scratch / f"{name}.stored", scratch / f"{name}.lmcpack"
    spec = {"pack": str(packed_path), "experts": [], "order": []}
    with open(stored_path, "wb") as fs, open(packed_path, "wb") as fp:
        for (layer, expert), mats in sorted(experts.items()):
            s0 = fs.tell()
            for part in stored_parts(mats):
                fs.write(part)
            s1 = fs.tell()
            pad(fs)
            p0 = fp.tell()
            entries = []
            for m, record, raw, fmt, out_bytes, sha in record_parts(mats, records_dir):
                r0 = fp.tell() - p0
                fp.write(record)
                w0 = fp.tell() - p0
                fp.write(raw)
                entries.append({"format": fmt, "record_offset": r0, "record_len": len(record), "raw_offset": w0,
                                "raw_len": len(raw), "out_bytes": out_bytes, "original_sha256": sha})
            p1 = fp.tell()
            pad(fp)
            spec["experts"].append({"name": f"{name}/L{layer}/E{expert}", "offset": p0, "length": p1 - p0,
                                    "matrices": entries,
                                    "original": [{"file": str(stored_path), "offset": s0, "length": s1 - s0}]})
    order = list(range(len(spec["experts"])))
    random.Random(seed).shuffle(order)
    spec["order"] = order
    time.sleep(20)  # let the OS close its cache maps of the freshly written packs before unbuffered reads (decision 0006)
    spec_path = scratch / f"{name}.ra.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    out = scratch / f"{name}.ra.out.json"
    done = subprocess.run([str(records.LMC), "random-access", str(spec_path), str(out)], capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"random access {name}: {done.stderr[-2000:]}")
    result = json.loads(out.read_text(encoding="utf-8"))
    reads = result["reads"]
    summary = {"experts": len(reads), "exact": result["exact"]}
    for k in ("record_bytes", "read_bytes", "original_read_bytes"):
        summary[k] = float(np.mean([r[k] for r in reads]))
    for k in ("read_s", "decode_s", "total_s", "original_read_s"):
        v = np.array([r[k] for r in reads])
        summary[k] = {"median": float(np.median(v)), "p90": float(np.quantile(v, 0.9)), "mean": float(v.mean())}
    summary["reads"] = reads
    return summary


def granularity(name: str, mats: list[quantized.QMatrix], scratch: Path) -> dict:
    """lmc order-0 with independent blocks of `rows` rows: total bytes over stored weight bytes."""
    out = {}
    for rows in (0, 512, 64, 8):
        specs = []
        for m in mats:
            nib = m.nibbles().astype(np.uint8)
            step = m.rows if rows == 0 else rows
            for r0 in range(0, m.rows, step):
                r1 = min(m.rows, r0 + step)
                base = scratch / f"g_{m.key.replace('/', '_')}_{r0}"
                nib[r0:r1].tofile(f"{base}.nib")
                m.weight_bytes[r0:r1].tofile(f"{base}.packed")
                specs.append({"name": f"{m.key}/{r0}", "format": "int4", "symbols": f"{base}.nib",
                              "original": f"{base}.packed", "rows": r1 - r0, "cols": m.cols, "alphabet": 16,
                              "scale_bits": 14, "streams": min(32, r1 - r0)})
        res = lmc({"blocks": specs, "threads": [6], "repeats": 1}, scratch / f"gran_{name}_{rows}.json")
        if not all(b["exact"] for b in res["blocks"]):
            raise SystemExit(f"granularity {name} {rows}: inexact")
        total = sum(b["sizes"]["total"] for b in res["blocks"])
        stored = sum(int(m.weight_bytes.size) for m in mats)
        out[str(rows or "matrix")] = {"blocks": len(specs), "ratio": total / stored,
                                      "overhead_bytes": sum(b["sizes"]["total"] - b["sizes"]["payload"] for b in res["blocks"])}
        for f in scratch.glob("g_*"):
            f.unlink()
    return out


def gpu_decode(name: str, mats: list[quantized.QMatrix], chunk: int, repeats: int = 10) -> dict:
    import torch

    from awpmi.streaming import nvcomp

    data = torch.from_numpy(np.concatenate([m.weight_bytes.reshape(-1) for m in mats])).cuda()
    chunks = [data[o : o + chunk] for o in range(0, data.numel(), chunk)]
    copts = nvcomp.compress_options("ans", type=0, data_type=nvcomp.DATA_TYPES["char"])
    dopts = nvcomp.decompress_options("ans", backend=0, data_type=nvcomp.DATA_TYPES["char"])
    packed = nvcomp.compress("ans", copts, chunks)
    out = torch.empty_like(data)
    arrays = torch.tensor([[p.data_ptr() for p in packed], [p.numel() for p in packed],
                           [out.data_ptr() + k * chunk for k in range(len(chunks))], [c.numel() for c in chunks]],
                          dtype=torch.int64, device="cuda")
    temp = nvcomp.decompress_temp_bytes("ans", dopts, len(chunks), chunk, data.numel())
    scratch = torch.empty(max(temp, 1), dtype=torch.uint8, device="cuda")
    actual = torch.empty(len(chunks), dtype=torch.int64, device="cuda")
    statuses = torch.empty(len(chunks), dtype=torch.int32, device="cuda")
    nvcomp.decompress("ans", dopts, arrays, scratch, actual, statuses)  # warm-up
    torch.cuda.synchronize()
    if bool((statuses != 0).any()) or not torch.equal(out, data):
        raise SystemExit(f"GPU decode {name}: inexact")
    times = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        nvcomp.decompress("ans", dopts, arrays, scratch, actual, statuses)
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) / 1e3)
    if not torch.equal(out, data):
        raise SystemExit(f"GPU decode {name}: inexact after timing")
    stored = sum(p.numel() for p in packed) + 8 * len(packed)
    return {"chunk_bytes": chunk, "chunks": len(chunks), "bytes_out": int(data.numel()), "ratio": stored / data.numel(),
            "best_s": min(times), "median_s": float(np.median(times)), "gb_s_best": data.numel() / min(times) / 1e9}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--sources", default="glm52_int4_row,inkling_int4_row,mimo_v26_pro_mxfp4,kimi_k2_int4_g32,gptoss20b_mxfp4")
    parser.add_argument("--bf16", default="moonlight")
    args = parser.parse_args()
    config = corpus.load_config(args.config)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    scratch = out / "scratch"
    scratch.mkdir(exist_ok=True)
    shutil.copy(args.config, out / "config.yaml")
    records.write_json(out / "environment.json", records.environment())
    seed = int(config["seed"])
    matrices = quantized.load(args.manifest, config)
    wanted = args.sources.split(",")
    result: dict = {"random_access": {}, "granularity": {}, "gpu_decode": {}}
    for name in wanted:
        mats = [m for m in matrices if m.source == name]
        if not mats:
            continue
        result["gpu_decode"][name] = [gpu_decode(name, mats, c) for c in (1 << 20, 1 << 16)]
        print(f"[{name}] GPU decode {[round(g['gb_s_best'], 1) for g in result['gpu_decode'][name]]} GB/s", flush=True)
        result["granularity"][name] = granularity(name, mats, scratch)
        print(f"[{name}] granularity {json.dumps({k: round(v['ratio'], 4) for k, v in result['granularity'][name].items()})}",
              flush=True)
        specs = [quantized_inputs(scratch, m) for m in mats]
        res = lmc({"blocks": specs, "threads": [1], "repeats": 1, "containers": str(scratch / f"{name}-records")},
                  scratch / f"rec_{name}.json")
        if not all(b["exact"] for b in res["blocks"]):
            raise SystemExit(f"{name}: records inexact")
        records_dir = scratch / f"{name}-records"

        def stored_parts(ms):
            for m in ms:
                yield m.weight_bytes.tobytes()
                yield m.scale_bytes.tobytes()

        def record_parts(ms, rd):
            for m in ms:
                rec = (rd / f"{m.key.replace('/', '_')}.lmc").read_bytes()
                yield m, rec, m.scale_bytes.tobytes(), "int4", int(m.weight_bytes.size), m.weight_sha256

        result["random_access"][name] = random_access(name, experts_of(mats), records_dir, scratch, seed, stored_parts,
                                                      record_parts)
        ra = result["random_access"][name]
        print(f"[{name}] random access: read {ra['read_s']['median'] * 1e3:.1f} ms + decode {ra['decode_s']['median'] * 1e3:.1f} ms"
              f" vs stored {ra['original_read_s']['median'] * 1e3:.1f} ms", flush=True)
        for f in scratch.glob(f"{name}*"):
            if f.is_file():
                f.unlink()
        shutil.rmtree(records_dir, ignore_errors=True)
    if args.bf16:
        model = args.bf16
        ckpt = corpus.Checkpoint(model, config["corpus"]["bf16"][model])
        pairs = ckpt.sample(seed)
        experts: dict = {}
        specs = []
        shas = {}
        for layer, expert in pairs:
            for kind in corpus.MATRICES:
                m = ckpt.matrix(layer, expert, kind)
                p = ckpt.read_patterns(m)
                if not np.array_equal(p, ckpt.read_reference(m)):
                    raise SystemExit(f"{m.key}: the two read paths disagree")
                s, e, mant = symbols.bf16_fields(p)
                base = scratch / m.key.replace("/", "_")
                e.tofile(f"{base}.exp")
                ((s << 7) | mant).astype(np.uint8).tofile(f"{base}.sm")
                p.astype("<u2").tofile(f"{base}.bf16")
                specs.append({"name": m.key, "format": "bf16", "symbols": f"{base}.exp", "raw": f"{base}.sm",
                              "original": f"{base}.bf16", "rows": m.rows, "cols": m.cols, "alphabet": 256,
                              "scale_bits": 15, "streams": 32})
                shas[m.key] = records.sha256_bytes(p.tobytes())
                experts.setdefault((layer, expert), []).append(m)
        res = lmc({"blocks": specs, "threads": [1], "repeats": 1, "containers": str(scratch / f"{model}-records")},
                  scratch / f"rec_{model}.json")
        if not all(b["exact"] and b["decoded_sha256"] == shas[b["name"]] for b in res["blocks"]):
            raise SystemExit(f"{model}: records inexact")

        def stored_parts(ms):
            for m in ms:
                yield (scratch / f"{m.key.replace('/', '_')}.bf16").read_bytes()

        def record_parts(ms, rd):
            for m in ms:
                rec = (rd / f"{m.key.replace('/', '_')}.lmc").read_bytes()
                raw = (scratch / f"{m.key.replace('/', '_')}.sm").read_bytes()
                yield m, rec, raw, "bf16", m.nbytes, shas[m.key]

        result["random_access"][model] = random_access(model, experts, scratch / f"{model}-records", scratch, seed,
                                                       stored_parts, record_parts)
        ra = result["random_access"][model]
        print(f"[{model}] random access: read {ra['read_s']['median'] * 1e3:.1f} ms + decode {ra['decode_s']['median'] * 1e3:.1f} ms"
              f" vs stored {ra['original_read_s']['median'] * 1e3:.1f} ms", flush=True)
    records.write_json(out / "stage_d.json", result)
    shutil.rmtree(scratch, ignore_errors=True)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
