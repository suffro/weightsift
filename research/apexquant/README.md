# ApexQuant feasibility for flagship routed experts (Phase 7A, decision 0014)

**Verdict: BLOCKED.** ApexQuant's code has no license, so Weightsift may not copy, vendor, run or depend on it; and no
flagship expert tensors are local or approved for download, so nothing about its error or quality on them could be
measured. The screening that could be done (bytes, from the checkpoints' headers and the paper's own accounting) also
says that, unblocked, its four-bit class would not pay: about 4.4% fewer bytes than MiMo's MXFP4 experts and 1.4% more
than Colibrì's int4 container of Inkling, against a 20% screening target. Only its two-bit prefix meets the target on
bytes, at a precision the paper reports as poor on dense 7B models and nobody has measured on MoE or at flagship scale.

Everything here was obtained without ApexQuant's code and without any tensor data. What is measured is labelled
measured; what is extrapolated or analysed is labelled so.

## 1. Licensing gate (C.1)

Inspected on 2026-10-12: `afristrup/ApexQuant` at `42b6efd532c5c1cf4df730ffd969d1bfbe8016c5` (2026-10-05).

| Where | Found |
| --- | --- |
| Repository root and every directory | no `LICENSE`, `COPYING` or `NOTICE` file |
| `apexquant/pyproject.toml` (the package) | no `license` field or classifier; authors "apexquant contributors" |
| Source files (`grep` for license, copyright, SPDX) | no header |
| GitHub's license detection (API) | none (`license: null`) |
| Package index | `apexquant` is not on PyPI (404) |
| The paper, arXiv 2610.07904v1 | arXiv's perpetual non-exclusive license: arXiv may distribute the paper; it grants nothing about the code |
| Third-party code inside it | `kernels/gemlite_splitk_nonuniform.py` is a fork of gemlite's GEMM_SPLITK, which a future license would also have to cover |

Without a license the authors keep every right: Weightsift does not copy, vendor, execute or depend on the code. What
remains allowed, and is all this study does: reading the paper and the published design to analyse them, and costing
the format with the accounting the paper publishes. Unblocking needs a license from the authors (Apache-2.0 or MIT, for
example) at a commit that can be pinned.

## 2. Technical study (C.2), from the paper and the published design

- **Residual re-isotropization.** Each pass rotates the current residual by a fresh randomized Hadamard transform (per
  group of g = 512 coordinates), normalizes it onto the sphere, rounds it with a fixed stage, and subtracts the
  reconstruction. For Haar rotations the residual's marginal is the same at every pass, so one codebook stays optimal
  and the relative error decays as ρ^P (proved in Lean); the deployed Hadamard transform is within 0.2% of that on
  real weights (measured by the authors, not a bound).
- **Stages** (b = 2 bits per coordinate per pass): scalar Lloyd–Max (ρ² = 0.1171), E8 lattice with QuIP#'s E8P codebook
  (0.0906), QTIP's bitshift trellis with L = 14 (0.0720), against the rate–distortion bound 0.0625. A greedy pass pays
  its stage's rate loss Δ₁ every pass (0.45, 0.27, 0.10 bits), so one wider pass beats several narrow ones at equal
  storage; the nested format's value is operational, not rate–distortion.
- **Nested prefixes.** The first P′ passes of a P-pass artifact are byte for byte the P′-pass artifact (checked by the
  authors on their published Llama-3.1-70B, Qwen3.5-35B-A3B and Gemma-3-12B tiers). A row's passes are stored back to
  back, so a lower precision is a contiguous prefix of a row: a format a storage engine can read partially.
- **Storage.** Codes `P × rows × groups × C` (b bits per weight per pass) and one fp16 scale per row, group and pass:
  P (b + 16/g) bits per weight, 2.03 / 4.06 / 6.09 at P = 1, 2, 3. Rotations come from seeds (seed + p); nothing else
  is stored.
- **Kernels.** Triton GPU kernels: per pass, the activation is rotated once and multiplied against codebook values
  gathered from the codes (no dequantized BF16 weight); a split-K GEMM forked from gemlite for mid-size batches. Batch-1
  decode is memory-bound (their three-pass token at 85% of the weight-traffic roofline on an RTX 4090, kernels alone);
  prefill is compute-bound and pays P times the dequantize-multiply work (not measured by the authors). There are no
  CPU kernels.
- **Sensitivity-based pass allocation** (one leave-one-out calibration pass): a Pareto gain in embedding error on a
  vision encoder; "does not transfer to language models" (appendix I).
- **Quality evidence.** Dense models up to 7–8B (TinyLlama, Phi-1.5, Mistral-7B, Qwen2.5-7B, LLaMA2-7B, LLaMA3-8B;
  Gemma-3-12B in an appendix). At four bits it is close to full precision (LLaMA2-7B perplexity 5.61 against 5.50; KL
  0.019 nats on Mistral-7B). At two bits it is far from usable on several models (LLaMA2-7B perplexity 71.8; KL 1.90 nats
  on Mistral-7B), where a calibrated trellis (QTIP) reaches 5.86. The authors state that the decay law has not been
  verified at frontier scale. There is no quality result on a mixture-of-experts model.

**Can it serve routed expert matrices?** Structurally, yes. The code handles fused expert tensors `[E, rows, columns]`
(a regression test on Qwen3.5-35B-A3B's layout). Quantization is per row and group, so experts stay independent. Both
flagships' expert rows split into whole groups of 512 (measured below). A pass of an expert row is a byte range, so
Weightsift's store could serve a precision as a prefix. What does not exist:

- CPU kernels. Colibrì computes Inkling's routed experts on the CPU, about 90% of its time at high hit rates per its
  documentation.
- An integration with a native-format MoE engine.
- Any quality measurement on MoE.

## 3. Measurements (C.3): what could safely be run

Not measured, and why:

- Error of real expert tensors, conversion time and peak memory, GEMV cost per prefix, throughput against MXFP4 or INT4:
  there are no flagship tensors here, their download is not approved, and ApexQuant's code may not be run.
- Quality: the Quality Budget needs a flagship and an engine.

Measured from metadata: each checkpoint's safetensors headers for one MoE layer, fetched without any tensor data by
`fetch_headers.py` and recorded in `headers/`, plus the geometry in `geometry.json`:

| Flagship | Format | Bits per weight (with scales) | MB per expert | Routed experts per decode token | GB per decode token (no cache) |
| --- | --- | --- | --- | --- | --- |
| MiMo-V2.6 Pro (`6a7f202e`) | MXFP4 as released: 4-bit values, one 8-bit scale per 32 | 4.2500 | 20.05 | 8 × 69 = 552 | 11.07 |
| Inkling (`828496ee`) | BF16, the original | 16.0000 | 113.25 | 6 × 64 = 384 | 43.49 |
| Inkling, Colibrì's int4 container (`3981e658`) | 4-bit values, one f32 scale per row | 4.0069 | 28.36 | 384 | 10.89 |

Measured on this machine (`experiments/phase7a/store-moonlight`, two runs): Weightsift's expert store leases
Moonlight-16B's real experts from the NVMe at 3.02 GB/s in batches (2.20 to 2.51 GB/s one expert at a time).
**Extrapolated**, not run, at that rate: a decode token that reads every routed expert from storage spends 3.6 s
(Inkling, int4 container) to 3.7 s (MiMo Pro, MXFP4) on I/O.

## 4. Screening (C.4)

Arithmetic from the measured bytes and the paper's accounting (`feasibility.py`, `feasibility.json`). "Bound" is the
decode speedup if a token were nothing but expert I/O, 1 / (bytes ratio): an upper bound, which any compute share
lowers.

| Flagship | ApexQuant | Bits per weight | Bytes against the native format | Bound | Meets the 20% target |
| --- | --- | --- | --- | --- | --- |
| MiMo-V2.6 Pro (native MXFP4) | P = 1 | 2.03 | 0.478 | 2.09 | yes, on bytes only |
| | P = 2 | 4.06 | 0.956 | 1.046 | no |
| | P = 3 | 6.09 | 1.434 | 0.70 | no |
| Inkling (native: Colibrì int4) | P = 1 | 2.03 | 0.507 | 1.97 | yes, on bytes only |
| | P = 2 | 4.06 | 1.014 | 0.99 | no |
| | P = 3 | 6.09 | 1.521 | 0.66 | no |

- **The four-bit class does not pay.** Against MXFP4, P = 2 saves 4.4% of the bytes (a 1.046 bound); against Colibrì's
  int4 container it costs 1.4% more. The optimistic end-to-end bound is negligible, so the track stops there.
- **Only the two-bit prefix meets the byte target**, alone or mixed with P = 2 (for example, a third of the experts at
  P = 1 averages 3.39 bits per weight, 20.3% below MXFP4). Its quality is the open question. The paper's evidence at two bits is poor
  on dense 7B models, and there is none on MoE or at a trillion parameters. It would be a different model, to be judged
  by the Quality Budget, not by weight error.
- **Native MXFP4 compounds errors.** MiMo's experts are released in MXFP4, so its quantization error against the
  unreleased higher-precision weights cannot be measured. ApexQuant at any rate would add its own error on top
  (re-quantizing quantized weights). For Inkling the BF16 original exists (1.91 TB), so its experts' errors can be
  separated: Colibrì's int4 container's against BF16, and ApexQuant's against BF16.
- **Kernel overhead (analysis).** P = 2 reads the same four bits per weight as a four-bit format, but performs two
  codebook passes and a Hadamard rotation of the activation per pass, about twice the gather-and-multiply work of one
  four-bit lookup pass. On a CPU-bound expert path (Colibrì's) that is an extra cost, not a saving. On the GPU,
  ApexQuant's own measurement puts batch-1 decode at the weight-traffic roofline, which helps only when the bytes are
  fewer.
- **Geometry.** No fundamental incompatibility: every expert matrix row of both flagships (6144, 3072 or 2048
  coordinates) splits into whole groups of 512.

## 5. What would justify a larger experiment

In order:

1. **A license** for ApexQuant's code, at a pinned commit (gemlite's terms included).
2. **The user's approval** to fetch a bounded sample of Inkling's BF16 experts by range requests. For example, 16
   experts of one layer: about 1.8 GB. One whole layer is 29 GB.
3. **A bounded probe:** per-pass relative error of P = 1 and P = 2 against BF16, beside the int4 container's own error
   against BF16, on those experts. This compares like with like, on real weights.
4. **Only if a P = 1 / P = 2 mix is close to int4's error**: a quality run of that mix under the Quality Budget, with
   Colibrì as the execution engine (which first needs CPU kernels for the format, or a GPU path).

## Reproduce

```bash
python research/apexquant/fetch_headers.py                        # network: indexes and headers only, no tensor data
python -m uv run python research/apexquant/feasibility.py         # the screening table, feasibility.json
python -m uv run python -m pytest research/apexquant/tests        # its arithmetic
```
