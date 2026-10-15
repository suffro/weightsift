# Lossless MoE weight compression: a feasibility study

Research code, not a runtime. It asks whether routed-expert tensors of MoE models hold enough exploitable statistical
structure, beyond what a static order-0 entropy coder (rANS with fixed tables) already takes, for a new lossless codec to
store them in substantially fewer bytes: the brief of 2026-10-15 asks for at least 25% fewer bytes than the strongest
practical rANS baseline on held-out real int4 data. Every candidate must rebuild the stored bytes exactly.

Report: `.context/history/2026-10-15-weightsift-lossless-moe-compression-feasibility.md`. Results:
`experiments/lossless_moe/`. Configuration, samples and gates (fixed before any measurement): `config.yaml`. Nothing in
`src/` imports this directory, and nothing here changes Weightsift's runtime.

## Environment

The Weightsift environment (its `research` group: zstandard, SciPy; PyTorch and nvCOMP for the GPU baseline), the Rust
toolchain pinned by `codec/rust-toolchain.toml`, and Colibri's codec from its pinned sources:

```bash
python research/lossless_moe/fetch_upstream.py                   # Colibri's rans.h, rans_ctypes.c, rans_format.py (pinned, verified)
(cd research/lossless_moe/codec && cargo build --release && cargo test --release)
python -m uv run python -m pytest research/lossless_moe/tests    # the Python tests
```

## Pipeline

```bash
export PYTHONIOENCODING=utf-8
# Stage A and B on BF16 (local checkpoints only), twice with different hash seeds; the digests must agree
for n in 1 2; do
  PYTHONHASHSEED=$n python research/lossless_moe/stage_a.py --output experiments/lossless_moe/stage-a-bf16-run$n
  PYTHONHASHSEED=$n python research/lossless_moe/stage_b.py --output experiments/lossless_moe/stage-b-bf16-run$n \
      --stage-a experiments/lossless_moe/stage-a-bf16-run$n
done
# The labelled diagnostic: Moonlight re-quantized with Colibri's per-row int4 recipe (never a principal result)
python research/lossless_moe/requantize.py --manifest experiments/lossless_moe/requant-manifest.json
# The quantized sample (int4 per-row, int4 group-32, MXFP4) by range requests -- only with the user's approval
python research/lossless_moe/fetch_sample.py --plan               # files, ranges, bytes; headers only
python research/lossless_moe/fetch_sample.py --fetch              # 1.685 GB into experiments/lossless_moe/samples/ (gitignored)
# Stages A and B on quantized experts, twice; the diagnostic once more
for n in 1 2; do
  PYTHONHASHSEED=$n python research/lossless_moe/stage_q.py --output experiments/lossless_moe/stage-q-run$n \
      --manifest experiments/lossless_moe/sample-manifest.json
done
python research/lossless_moe/stage_q.py --output experiments/lossless_moe/stage-q-diagnostic \
    --manifest experiments/lossless_moe/requant-manifest.json
# Stage D: cold random access, block granularity, GPU decode
python research/lossless_moe/stage_d.py --output experiments/lossless_moe/stage-d --manifest experiments/lossless_moe/sample-manifest.json
# Tables and gate verdicts
python research/lossless_moe/report.py --stage-a experiments/lossless_moe/stage-a-bf16-run1 \
    --stage-b experiments/lossless_moe/stage-b-bf16-run1 --stage-q experiments/lossless_moe/stage-q-run1 \
    --diagnostic experiments/lossless_moe/stage-q-diagnostic --output experiments/lossless_moe/report
```

Timings are measured only when nothing else runs on the machine; digests leave them out.

## Modules

| File | Role |
| --- | --- |
| `lossless_moe/symbols.py` | Exact field splits and their inverses: BF16 (sign, exponent, mantissa; high and low bytes), int4 nibbles (low first; offset-8 or two's complement), E2M1 (sign, magnitude) |
| `lossless_moe/entropy.py` | Code lengths in bits: in-sample conditional entropy (optimistic), two-part static code (tables paid), adaptive Krichevsky-Trofimov code (achievable, no side information), held-out cross-entropy and prior-adaptive codes |
| `lossless_moe/contexts.py` | Context models: causal (previous symbols, row above, running row and column statistics), side information with its bits charged (classes per row, column, rank-one), other tensors the decoder holds |
| `lossless_moe/measure.py` | Stage B's measurements per (field, context), pooled per group and split for held-out scoring; the BF16 context sets |
| `lossless_moe/quantized.py` | The fetched sample's matrices (stored bytes, sha256 checked), scales per format, the quantized formats' contexts (scale class, block maximum) |
| `lossless_moe/linear.py` | Linear diagnostics: adjacent correlations, low rank against a permuted null, gate-up correlation |
| `lossless_moe/corpus.py` | Local checkpoints, the seeded sample, two independent read paths, publisher sha256 |
| `lossless_moe/records.py` | Environment, study tree hash, digests without timings |
| `stage_a.py`, `stage_b.py` | BF16: every codec on the same bytes; the context measurements |
| `requantize.py` | The labelled diagnostic (Colibri's per-row int4 recipe on Moonlight's sample) |
| `fetch_sample.py` | Range-request sample from safetensors headers, slices hashed, two windows of each re-read and compared |
| `stage_q.py` | Quantized experts: codecs (Colibri's own included) and context measurements |
| `stage_d.py` | Cold random access, block granularity, GPU decode (nvCOMP ANS) |
| `report.py` | Tables, matched context deltas, gate verdicts |
| `codec/` | `lmc`, a Rust prototype: static rANS (32-bit state, byte renormalization) with one table per context, streams by rows, a lockstep decoder whose stream ends are checked; `LMC1` block records with every byte counted; Colibri's int4-rans256-g0 compiled from its pinned C and called through its own bridge; the bench and random-access drivers |

## What is reused, what is written here

Reused as published: zstandard (libzstd), CPython's lzma and bz2, nvCOMP 5.3's ANS (through Weightsift's binding,
`awpmi.streaming`), Colibri's `c/rans.h` and `c/tools/rans_ctypes.c` (compiled unmodified) and `rans_format.py` (its
table rule, imported unmodified), Phase 5C's byte split and bit planes (`research/expert_deltas/expert_deltas/bits.py`),
SciPy's `gammaln`, safetensors' loader as the independent read path.

Written here: the estimators, the context models, the `lmc` coder and container (an ordinary static rANS; its novelty is
nil by design: it is the reference, plus context-selected tables), the drivers.

## Invariants (tested)

- Every field split restores every pattern; every coded matrix is decoded and compared with its stored bytes, and with
  the independent read's sha256 (Stage A, Stage D, the quantized stage).
- The adaptive estimator equals a sequential adaptive coder's code length, in any symbol order; the in-sample estimator
  equals its definition; held-out estimators equal direct sums.
- A causal context never depends on the symbol it codes or a later one (with a negative control); side information is
  charged; a cross-matrix context is compared only on the matrices that have it.
- `lmc`: every context kind round-trips, including degenerate shapes and alphabets; framing errors are refused; every
  single-bit flip of a payload is either refused by the checked decoder or changes the output (caught by the sha256);
  Colibri's codec round-trips through its own bridge with both its arms.
