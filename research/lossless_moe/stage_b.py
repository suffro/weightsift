"""Stage B on BF16: how much information is left beyond a static order-0 coder?

    python research/lossless_moe/stage_b.py --output experiments/lossless_moe/stage-b-bf16 \
        [--stage-a experiments/lossless_moe/stage-a-bf16]

The same sample as Stage A (same configuration and seed). Every matrix is read by positioned reads and, when a Stage A
run is named, its sha256 must equal the one Stage A recorded against safetensors' own copy (the same bytes). Per matrix,
its exponent, mantissa and sign are measured under every context model of `measure.bf16_specs` (in-sample, two-part,
adaptive), with these cross-matrix contexts:

  given_gate        the up matrix's exponents and signs given the same expert's gate (decoded first)
  given_prev        a matrix given the previous sampled expert's same matrix in the same layer (cross-expert)
  neuron_class16    the down matrix's columns classed by the same expert's gate and up rows (decoded first)

Counts are pooled per model and matrix kind, split two ways (held-out experts within layers; held-out layers), and
tables trained on calibration matrices are scored on evaluation matrices. Linear diagnostics (adjacent correlations,
low rank against a permuted null, gate-up correlation) run on the first sampled expert of each layer.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lossless_moe import corpus, linear, measure, records, symbols  # noqa: E402


def bf16_values(p: np.ndarray) -> np.ndarray:
    return torch.from_numpy(p.astype(np.int16)).view(torch.bfloat16).float().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--models", default=None)
    parser.add_argument("--stage-a", type=Path, default=None, help="a Stage A run whose sha256 the bytes must match")
    args = parser.parse_args()
    config = corpus.load_config(args.config)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / "config.yaml")
    models = config["corpus"]["bf16"]
    selected = args.models.split(",") if args.models else list(models)
    records.write_json(out / "environment.json",
                       records.environment({n: {k: models[n][k] for k in ("repository", "revision")} for n in selected}))
    expected = {}
    if args.stage_a:
        for line in open(args.stage_a / "blocks.jsonl", encoding="utf-8"):
            row = json.loads(line)
            expected[row["key"]] = row["sha256"]
    acc = measure.Accumulator()
    diagnostics = []
    summary = {"models": {}}
    t_start = time.perf_counter()
    for name in selected:
        ckpt = corpus.Checkpoint(name, models[name])
        pairs = ckpt.sample(config["seed"])
        layers = list(models[name]["layers"])
        half = len(layers) // 2
        per_layer: dict[int, list[int]] = {}
        for layer, expert in pairs:
            per_layer.setdefault(layer, []).append(expert)
        # Pass 1: the model's exponent support (declared once per model).
        support = set()
        for layer, expert in pairs:
            for kind in corpus.MATRICES:
                support |= set(np.unique(symbols.bf16_fields(ckpt.read_patterns(ckpt.matrix(layer, expert, kind)))[1]).tolist())
        support = np.array(sorted(support), dtype=np.int64)
        print(f"[{name}] {len(pairs)} experts; exponent support {support.min()}..{support.max()} ({support.size} values)",
              flush=True)
        prev: dict[tuple[int, str], np.ndarray] = {}
        prev_sign: dict[tuple[int, str], np.ndarray] = {}
        for layer, expert in pairs:
            position = per_layer[layer].index(expert)
            split_e = "calibration" if position % 2 == 0 else "evaluation"
            split_l = "calibration" if layers.index(layer) < half else "evaluation"
            fields = {}
            for kind in corpus.MATRICES:
                m = ckpt.matrix(layer, expert, kind)
                p = ckpt.read_patterns(m)
                if expected and records.sha256_bytes(p.tobytes()) != expected[m.key]:
                    raise SystemExit(f"{m.key}: not the bytes Stage A verified")
                fields[kind] = (m, p, *symbols.bf16_fields(p))
            for kind in corpus.MATRICES:
                m, p, s, e, mant = fields[kind]
                extra: dict = {"given_exponent": {}, "given_sign": {}}
                if kind == "up":
                    extra["given_exponent"]["gate"] = fields["gate"][3]
                    extra["given_sign"]["gate"] = fields["gate"][2]
                if (layer, kind) in prev:
                    extra["given_exponent"]["prev"] = prev[(layer, kind)]
                    extra["given_sign"]["prev"] = prev_sign[(layer, kind)]
                if kind == "down":
                    extra["neuron_classes"] = measure.neuron_classes(fields["gate"][3].astype(np.float64),
                                                                     fields["up"][3].astype(np.float64))
                memberships = [(f"{name}/{kind}/experts", split_e), (f"{name}/{kind}/layers", split_l),
                               (f"{name}/all", "all")]
                for spec in measure.bf16_specs(e, s, mant, support, extra):
                    acc.add(m.key, memberships, spec)
                prev[(layer, kind)] = e
                prev_sign[(layer, kind)] = s
            if position == 0:
                values = {kind: bf16_values(fields[kind][1]) for kind in corpus.MATRICES}
                d = {"key": f"{name}/L{layer}/E{expert}"}
                for kind in corpus.MATRICES:
                    d[kind] = {"adjacent": linear.adjacent(values[kind]), "low_rank": linear.low_rank(values[kind])}
                d["gate_up"] = linear.paired(values["gate"], values["up"])
                diagnostics.append(d)
            print(f"  L{layer} E{expert} done ({time.perf_counter() - t_start:.0f} s)", flush=True)
        summary["models"][name] = {"repository": ckpt.repository, "revision": ckpt.revision,
                                   "exponent_support": support.tolist(), "experts": pairs}
    # Totals and held-out scores per group and (field, context)
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
    records.write_json(out / "linear.json", diagnostics)
    summary["digest"] = records.digest({"contexts": table, "linear": diagnostics})
    summary["seconds"] = time.perf_counter() - t_start
    records.write_json(out / "summary.json", summary)
    print(f"digest {summary['digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
