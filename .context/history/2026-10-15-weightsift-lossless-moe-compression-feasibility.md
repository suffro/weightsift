# Weightsift — lossless compression of MoE weights: a feasibility study

Date: 2026-10-15 · Follows: `history/2026-10-15-weightsift-phase7b-review.md` (Phase 7B) · Brief:
`state/lossless_moe_brief.md` (the user's of 2026-10-15) · Branch: `research/lossless-moe-compression`, from `06b7fde`,
not merged · Code: `research/lossless_moe/` · Results: `experiments/lossless_moe/` · Status: **complete. NO-GO.** On held-out real
MoE experts, no statistical model finds more than **0.24–0.31%** below the strongest static rANS coder on per-row int4,
**1.4%** on group-scaled int4 and **3.1–3.3%** on MXFP4. The brief's target is 25%. The early gate stopped Stage C, as
the brief requires. Weightsift's runtime roadmap stands.

## How to read this report

Labels, as the brief defines them:

- **[Verified]**: measured in this study (with its run directory), read in source code, or a published result read for it.
- **[Inference]**: reasoned from identified evidence.
- **[Hypothesis]**: plausible, untested.
- **[Speculation]**: exploratory, little evidence.
- **[Unknown]**: not enough information.

A number from another project or paper is that project's claim on its own machines. It is [Verified] as a claim, never as
a reproduction. "Stored bytes" are the bytes a checkpoint or container holds for a matrix: weights plus their scales. A
**ratio** is compressed bytes over stored bytes. A **reduction** is relative to the strongest practical rANS coder on the
same bytes, as the brief defines its targets.

## 1. Executive conclusion

**The answer is no.** The measured data do not contain the information a substantially better lossless codec would need.

- **The reference.** The strongest practical rANS coder is static order-0 tables per matrix, with the scales coded too
  when they are compressible (Section 4). It stores per-row int4 experts in **2.94–2.95 bits per weight**: 0.732–0.737
  of their bytes. That is within 0.01% of their order-0 entropy, every header, table and stream flush included
  [Verified]. Colibrì's own codec, run from its pinned C, reaches 0.733–0.738 on the same bytes.
- **What a 25% reduction needs.** About 2.2 bits per weight on GLM-5.2's container: 0.74 bits below the reference.
- **What the data hold.** Every context model measured, with each matrix picking its best context in hindsight, takes
  **0.007–0.009 bits per weight** on per-row int4 [Verified]. The models tested: neighbours, rows, columns,
  rank-one variance, the stored scales, gate given up, down given its neurons, the previous expert, and held-out tables.
  - That is 1% of the gap, and in-sample estimates (free tables fitted to the data) agree to within 0.0002 bits.
  - Group-scaled formats keep a little more: 0.055 bits on Kimi-K2's int4 (1.4%), 0.12–0.13 bits on MXFP4 (3.1–3.3%).
  - BF16 keeps 0.05 bits of 10.55 (0.5%).
- **Why.** Trained weights are close to independent draws once their scale is known [Inference from Sections 5.2–5.4].
  - Adjacent weights correlate at |ρ| < 0.02 (one Granite matrix: 0.07, down its columns).
  - Gate and up projections are uncorrelated (|ρ| < 0.015).
  - Experts share nothing, as Phase 5C found.
  - A low-rank predictor never pays for its factors.
  - What remains is per-row and per-column variance and quantizer artifacts. Their information is a few thousandths of a
    bit on per-row int4 and about a tenth of a bit on MXFP4.
- **Decoder cost runs the wrong way.** On this machine's 6-core AVX2 CPU, static rANS decodes int4 at **0.16–0.35 GB/s
  per thread**, 0.8–1.9 GB/s on all cores. Colibrì's arms decode at 0.09 per thread and 0.47–0.52 on all cores. A
  context model decodes 19–31% slower [Verified].
  - Decoding a cold int4 or MXFP4 expert on one thread takes 14–15 times as long as reading it from the drive.
  - Only the GPU decodes fast enough: nvCOMP's ANS at 122–139 GB/s [Verified].
- **For Colibrì and DS4.** At most 0.3% fewer drive bytes on per-row int4 and 3.5% on MXFP4, even with free decoding
  [Inference]. Lossless coding's useful lever is already the static coder, worth about 1.36× on the drive term
  (Phase 7B). It pays only where the decoder outruns the drive by about 4× (Colibrì's own crossover).
- **Recommendation: NO-GO** (Section 14). Keep the runtime roadmap.

| Corpus (held out) | Format | Reference (bits/weight) | Achievable beyond it | Optimistic beyond it | Reduction | Stage B verdict |
| --- | --- | --- | --- | --- | --- | --- |
| GLM-5.2 (Colibrì's container) | int4, per-row scales | 2.9540 | 0.0072 | 0.0073 | 0.24% | unsupported |
| Inkling (Colibrì's container) | int4, per-row scales | 2.9348 | 0.0092 | 0.0093 | 0.31% | unsupported |
| Kimi-K2-Thinking (publisher) | int4, BF16 scale per 32 | 3.9006 | 0.0550 | 0.0555 | 1.41% | unsupported |
| MiMo-V2.6 Pro (publisher) | MXFP4 | 3.7674 | 0.1260 | 0.1265 | 3.34% | unsupported |
| gpt-oss-20b (publisher) | MXFP4 | 3.9189 | 0.1198 | 0.1206 | 3.06% | unsupported |
| Moonlight-16B-A3B | BF16 | 10.5497 | 0.0497 | 0.0535 | 0.47% | unsupported |
| OLMoE-1B-7B | BF16 | 10.5640 | 0.0528 | 0.0577 | 0.50% | unsupported |
| Granite-3.1-1B-A400M | BF16 | 10.6021 | 0.0534 | 0.0697 | 0.50% | unsupported |
| *Moonlight re-quantized (diagnostic)* | *int4, per-row* | *3.0549* | *0.0074* | *0.0079* | *0.24%* | *unsupported* |

The pre-registered gate (`research/lossless_moe/config.yaml`, fixed before any measurement):

- **Plausible** at a reduction of 25% or more.
- **Uncertain, worth a limited prototype** at 10–25%.
- **Unsupported** below 10%.

"Achievable" is the adaptive code length, with every learning cost included. "Optimistic" is the in-sample conditional
entropy, which is biased low. The reduction column uses the optimistic gain, the generous choice. Source:
`experiments/lossless_moe/report/tables.md` [Verified].

## 2. Datasets and checkpoints inspected

**Weightsift** at `06b7fde` (Phase 7B): `.context` (state, guide, decisions 0011–0014, the 5C, 6B and 7B reports), Phase
5C's codecs (`research/expert_deltas`), Phase 6B's encoded rows and GPU probe, Phase 7A's header probes.

**Colibrì** (Apache-2.0), cloned read-only:

- `main` at `bf244291` (the pin) and `dev` at `37127ed` (2026-10-15, 110 commits later). The rANS files are identical on
  both.
- Read: the format document, `c/rans.h`, `c/tools/rans_ctypes.c`, `rans_format.py`, `repack_rans.py`, `rans_verify.py`,
  `analyze_split_entropy.py`, the per-row int4 converters, and issues #594 and #109.
- Pinned for this study in `research/lossless_moe/upstream.json` (six files, sha256 verified). Its codec is compiled
  from them, unmodified.

**DS4** (antirez/ds4, MIT) at `fc80bd69` (Phase 7A's pin, still its head): README and its SSD-streaming notes.

**The corpus** [Verified]: every matrix read as stored, with exactly these selection rules.

| Corpus | Repository at revision | Layers sampled | Experts per layer | Matrices | Weights | How obtained |
| --- | --- | --- | --- | --- | --- | --- |
| Moonlight-16B-A3B, BF16 | `moonshotai/Moonlight-16B-A3B` @ `476b36a4` | 1, 6, 11, 16, 21, 26 | 8 | 144 | 415.2 M | local cache |
| OLMoE-1B-7B-0924, BF16 | `allenai/OLMoE-1B-7B-0924` @ `6d84c485` | 0, 5, 10, 15 | 8 | 96 | 201.3 M | local cache |
| Granite-3.1-1B-A400M, BF16 | `ibm-granite/granite-3.1-1b-a400m-instruct` @ `0da7a48b` | 0, 8, 16, 23 | 8 | 96 | 50.3 M | local cache |
| GLM-5.2, Colibrì's per-row int4 | `jlnsrk/GLM-5.2-colibri-int4` @ `cce7e760` | 10, 30, 50, 76 | 4 | 48 | 604.0 M | range requests |
| Inkling, Colibrì's per-row int4 | `nbeerbower/Inkling-colibri-int4` @ `3981e658` | 10, 25, 40, 60 | 4 | 48 | 906.0 M | range requests |
| MiMo-V2.6 Pro, MXFP4 | `XiaomiMiMo/MiMo-V2.6-Pro-MOPD` @ `6a7f202e` | 10, 30, 50, 68 | 4 | 48 | 604.0 M | range requests |
| Kimi-K2-Thinking, int4 group 32 (BF16 scales) | `moonshotai/Kimi-K2-Thinking` @ `a51ccc05` | 10, 25, 40, 58 | 4 | 48 | 704.6 M | range requests |
| gpt-oss-20b, MXFP4 | `openai/gpt-oss-20b` @ `6cee5e81` | 2, 8, 14, 22 | 4 | 32 | 398.1 M | range requests |
| *Diagnostic: Moonlight re-quantized* | *Colibrì's per-row recipe on the Moonlight sample* | *as Moonlight* | *8* | *144* | *415.2 M* | *derived locally* |

**How the corpus was built.**

- **Expert choice.** Within a layer, experts come from a seeded permutation
  (`numpy.random.default_rng([20261015, crc32(model), layer])`, `corpus.sample_experts`).
- **Matrices.** Every projection of a sampled expert is used: gate, up and down, or gpt-oss's fused gate_up and down.
- **The fetched sample, with your approval.** 1.685 GB in 416 slices of the sampled experts' weights and scales, fetched
  by HTTP range requests (`fetch_sample.py`).
  - Tensors were located from safetensors headers only (13.3 MB of headers for GLM-5.2's 144 unindexed shards).
  - Each slice was hashed, and two random 64 KiB windows of each were requested again and compared; all equal.
  - Manifest: `experiments/lossless_moe/sample-manifest.json`. The slices are gitignored.
- **Local BF16 files.** The files touched were hashed in full against the publisher's sha256; all equal.
- **Not included.** Colibrì's g64 containers were left out because its rANS format refuses them by design. DeepSeek-V2-Lite
  is only a config in the local cache.

**Format checks on the fetched sample** [Verified, `experiments/lossless_moe/fetch-sample.log`]:

- **Per-row containers** (GLM, Inkling) store level + 8, as Colibrì's converters write it. Level −8 never occurs, and
  every row holds a ±7.
- **Kimi-K2** (compressed-tensors `pack-quantized`) packs eight level + 8 nibbles per int32, low nibble first. Every group
  of 32 holds a ±7.
- **MXFP4 blocks** have their maximum at E2M1 4 or 6, as the OCP scale rule implies. MiMo stores every zero as −0.

**The labelled diagnostic** (`requantize.py`) applies Colibrì's `quant_int4_rows` to the Moonlight sample. It is never a
principal result, as the brief forbids. It isolates what a per-row int4 container does to structure measured in BF16, on
the same weights.

**Machine** [Verified]:

- Intel i7-8700K (6 cores; the OS reports 6 logical processors; AVX2, no AVX-512) and 32 GB of RAM.
- RTX 4060 Ti 8 GB, and a Samsung 990 PRO on PCIe 3.0 x4.
- Windows 11 Pro 10.0.26200.
- Toolchain: MSVC 14.44, Rust 1.95.0, CPython 3.13.3, zstandard 0.25 (libzstd 1.5.7), nvCOMP 5.3.

## 3. Existing lossless weight compression

These works were read through their abstracts or project pages; no full text was read unless named. Every ratio and speed
is the authors' own, on their hardware [Verified as their claims, not reproduced]. "Transfer" says whether a result is
evidence about this study's int4 and MXFP4 experts [Inference].

### 3.1 Entropy coders and general-purpose codecs

| Work | What it is | Decoder speed (their claims) | Random access | Availability | Relevance |
| --- | --- | --- | --- | --- | --- |
| rANS and tANS (Duda, 2013–14) | Asymmetric numeral systems: arithmetic-coding ratios at table-lookup cost. rANS divides; tANS (FSE) uses finite-state tables | Depends on the implementation (below) | Per stream | Published designs | The coder of every baseline here |
| [Interleaved entropy coders](https://arxiv.org/abs/1402.3392) (Giesen, 2014), ryg_rans | Several rANS states interleaved in one stream; SIMD decoding | SSE 4.1, 8-way: about 195–268 MB/s on its test files (2014 hardware) | Per stream | GitHub rygorous/ryg_rans (license not detected by GitHub) | The construction Colibrì's `rans.h` and this study's `lmc` use |
| [Recoil](https://arxiv.org/abs/2306.12141) (Lin et al., 2023) | Parallel decoding of one rANS bitstream from stored intermediate states | 90+ GB/s on a Turing GPU; AVX2 and AVX-512 CPU variants | Splits chosen at decode time | Paper | Splits streams after encoding; leaves the ratio unchanged |
| htscodecs rANS 4x16 and 32x16 (Bonfield) | SIMD rANS with order-0 and order-1 models, used in CRAM genomics | AVX2 and AVX-512 decoders; speeds not checked here [Unknown] | Per block | GitHub jkbonfield/htscodecs (license not detected by GitHub) | Order-1 rANS models are routine; the model, not the coder, sets the ratio |
| [DietGPU](https://github.com/facebookresearch/dietgpu) (Johnson, Meta) | Byte-oriented rANS on the GPU, and a float codec that splits exponents | 250–410 GB/s (ANS) and 250–600 GB/s (float) on an A100 | Batched chunks | MIT, archived | Colibrì's CUDA ANS VRAM tier uses it: +13.9% capacity, +4.5% median decode on 6× RTX 5090 |
| nvCOMP ANS (NVIDIA) | Batched GPU rANS (byte and float16 modes) | Measured here: Section 8.3 | 1 MiB or 64 KiB chunks | NVIDIA SDK license | Phase 6B's encoded rows (0.6747 of BF16 on Moonlight) |
| zstd, LZ4, LZMA, bzip2, Brotli | LZ77 or BWT, followed by entropy stages | zstd decodes at 0.4–0.7 GB/s on one thread here (Section 4.1) | Per frame | Open source | LZ matching finds nothing in weights (Section 4); their entropy stages are order-0 |
| Bitshuffle and byte shuffle (Masui et al., 2015; Blosc) | Reversible transposition of bits or bytes before an LZ codec | Memory-bound | Per block | Open source | Phase 5C's bit planes and byte split are these transforms; they bring LZ codecs up to order-0, not beyond |

### 3.2 Lossless compression of neural network weights

| Work | Input | Lossless | Reported ratio | Models | Decoder | Random access | Availability | Transfer to MoE int4 and MXFP4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [ZipNN](https://arxiv.org/abs/2411.05239) (Hershcovitch et al., 2024) | BF16, FP16, FP32 | Yes | "Often" 33% saved, sometimes over 50% | Hugging Face models, Llama 3 among them | CPU, Huffman on exponent bytes | Chunks | GitHub zipnn/zipnn (license not detected by GitHub) | None for int4: its redundancy is the exponent field |
| [DFloat11](https://arxiv.org/abs/2504.11651) (Zhang et al., 2025) | BF16 | Yes, bit-exact | About 70% of BF16 (11 bits per weight) | Llama 3.1/3.3, Qwen 3, Mistral 3, FLUX.1 | GPU Huffman with hierarchical look-up tables | Transformer block | GitHub LeanModels/DFloat11, Apache-2.0 | None for int4. Same mechanism as Phase 6B's rANS rows and this study's BF16 reference |
| [NeuZip](https://arxiv.org/abs/2410.20650) (Hao et al., 2024) | BF16 exponents (training); truncated mantissas (inference, lossy) | Training path only | Llama-3 8B training memory from 31 GB to under 16 GB | Llama 3 | nvCOMP ANS | Per layer | GitHub BorealisAI/neuzip (license not detected by GitHub) | Only the lossless exponent path is comparable |
| [Huff-LLM](https://arxiv.org/abs/2502.00922) (2025) | Weight subfields | Yes | Up to 31% lower latency (hardware) | Llama and others | Hardware Huffman | — | Paper | Hardware, oriented to BF16 |
| [ECF8](https://arxiv.org/abs/2510.02676) (Yang et al., ICLR 2026) | FP8 | Yes | Up to 26.9% of memory saved | LLMs and DiTs up to 671B | GPU | — | GitHub zeyuyang8/ecf8 (no license file) | FP8 exponent concentration; not int4 |
| [ZipServ](https://arxiv.org/abs/2603.17435) (Fan et al., 2026) | BF16 | Yes | Up to 30% smaller | LLMs | Decompression fused into the GEMM (bitmap encoding) | Tile | Paper | BF16 only |
| [ZipMoE](https://arxiv.org/abs/2601.21198) (2026) | BF16 MoE experts | Yes | 68% with zstd on split fields (order-0 entropy 66.0–66.5%) | DeepSeek-V2-Lite, Qwen1.5-MoE, Switch-Large | CPU (Jetson Orin) | Tensor | Paper | BF16; agrees with Section 4.1 (MoE BF16 at 0.66) |
| [FluxMoE](https://arxiv.org/abs/2604.02715) (2026) | BF16 experts | Yes | Not extracted [Unknown] | GLM-4.5, Mixtral | Selective exponent Huffman | — | Paper | BF16 |
| [Approaching Shannon Bound](https://arxiv.org/abs/2606.15789) (2026) | BF16, FP8, INT8, FP4, INT4, SQ8, AWQ4 | Yes | Order-0 entropy per tensor; "int4 retains only 0.6–1.0 bits" | 1.5B–405B, Mixtral-176B | GPU ANS per GEMM tile, "within 0.01–0.1 bits of the Shannon limit" | Tile | Not released | Its int4 figure cannot describe a deployed int4 container, which measures 2.9–3.9 bits (Section 4). The paper names no quantizer or granularity; per-tensor scaling would make most codes zero [Inference] |
| [EntroLLM](https://arxiv.org/abs/2505.02380) (2025) | uint8 and uint4 from its own tensor-level quantization | Yes, of its codes | 30% over uint8, 65% over uint4 | smolLM-1.7B, phi3-mini, mistral-7B | Parallel Huffman, Jetson | — | Paper | The gain comes from a quantizer that concentrates codes: a property of the quantizer, not of the coder [Inference] |
| [Brevis](https://arxiv.org/abs/2608.02162) (Shi et al., 2026) | Checkpoint tensors | Yes | 2.13 TB to 1.41 TB over 10 checkpoints; 30.87% smaller than zstd and gzip | 10 public checkpoints | 6.61 GB/s | Program per tensor | Paper | A search over reversible operators (repeats, float fields); no evidence on int4 experts |
| [DeepCABAC](https://arxiv.org/abs/1907.11900) and MPEG NNC (ISO/IEC 15938-17) | Integers from its own rate-distortion quantization | Its arithmetic coding is lossless; the pipeline is lossy | VGG16 × 63.6 (whole pipeline) | CNNs | CABAC (sequential) | Limited | Reference software | Its context models for quantized weights (significance and sign from neighbours) are the closest prior art to Candidate A. Its published ratios include its quantization and do not transfer |

### 3.3 MoE int4 containers in practice: Colibrì and its community

| Measurement | Format | Result | Source |
| --- | --- | --- | --- |
| GLM-5.2, full-corpus census | Colibrì per-row int4 | 2.9147 bits per weight order-0 (72.87%); per layer 2.77–2.95 | Colibrì [docs/experiments/glm52-split-entropy-2026-07-31.md](https://github.com/JustVugg/colibri/blob/bf2442915d6e3dd4cdfd2eb9c2a3d2aa44a25850/docs/experiments/glm52-split-entropy-2026-07-31.md) |
| int4-rans256-g0 | Same | About 0.76 achieved; one static table per shard, 256 streams | Colibrì [docs/int4-rans256-g0.md](https://github.com/JustVugg/colibri/blob/bf2442915d6e3dd4cdfd2eb9c2a3d2aa44a25850/docs/int4-rans256-g0.md) |
| A 340 MB GLM-5.2 sample | Same | 0.7376 with one static table; per-expert chunks cost about 0.06% | Colibrì [issue #594](https://github.com/JustVugg/colibri/issues/594) |
| In-engine A/B: Zen 5 with AVX-512, NVMe at 5–6 GB/s | Same | Read time halved; decode 0.99 GB/s across about 11 threads; end to end 4–5× slower; break-even drive about 140–260 MB/s | Colibrì [issue #594](https://github.com/JustVugg/colibri/issues/594) |
| Qwen3.6-35B-A3B, 320 experts | Per-row int4 from another converter | 3.037 bits global, 3.022 per expert, 2.993 per row ("per-row tables would add one point"); zstd-19 76.2%, xz 76.5% | Colibrì [issue #594](https://github.com/JustVugg/colibri/issues/594) |
| MiMo-V2.5 311B | int4 container, one zstd-1 frame per tensor | 71.8% of the full container; 5.89 bits per byte order-0; engine 18% slower when streaming compressed | Colibrì [issue #109](https://github.com/JustVugg/colibri/issues/109) |
| GLM-5.2 REAP-504B | Per-row int4 against E8/IQ3 | zlib: 76.7% against 99.1% | Colibrì [issue #594](https://github.com/JustVugg/colibri/issues/594) |
| VQ3R residual-VQ indices | 3 bits per weight, codebook | 7.99 of 8 bits per byte; mutual information between stages 0.16–0.27 bits (2.8% at most) | Colibrì [issue #594](https://github.com/JustVugg/colibri/issues/594) (sqliteai/warp) |
| GPU ANS VRAM tier | Per-row int4, DietGPU | +13.9% expert capacity; median decode +4.5% (6× RTX 5090, every expert resident) | Colibrì [issue #594](https://github.com/JustVugg/colibri/issues/594), PR #651 |

### 3.4 What the literature says and leaves open

- **Every lossless weight codec found uses an order-0 model per field or per tensor.** Some first split a field (the
  exponent apart from sign and mantissa) or transpose bits.
  - None reports a context model beating order-0 by a meaningful margin on trained weights.
  - The only systematic context modeling of quantized weights is DeepCABAC/NNC, whose published gains cannot be
    separated from its own quantizer [Verified as stated; the separation is Inference].
- **The large ratios in the literature have two sources.** One is the exponent field of floating-point weights (BF16,
  FP8); the other is quantizers that leave codes concentrated, such as tensor-level scales. Neither applies to per-row or
  group int4 containers or to MXFP4, whose quantizers already spend most of their codes [Inference].
- **Codebook formats are incompressible by construction.** E8/IQ3 and VQ3R compress to about 99–100% [Verified as Colibrì's
  community measurements].
- **What was open before this study:** the conditional entropy of real MoE int4 and MXFP4 experts under rich context
  models, held out and with every model cost charged. Section 5 measures it.
- **A new entropy coder or a new statistical model.** A static rANS coder already lands within 0.01% of order-0 entropy
  here (Section 4.4). Any gain must come from a better model, and an existing coder (rANS with context-selected tables)
  carries any model. This study's candidates are therefore models.

## 4. Baselines (Stage A)

Every codec ran on exactly the same stored bytes. Every reconstruction was decoded and compared with them, and every
`lmc` and Colibrì output was also checked by sha256 against safetensors' own read. The ratios include every header,
table, offset, stream flush and scale byte.

**The codecs.**

- **General-purpose:** zstd 3, 19 and 22, LZMA preset 9e, and bzip2 9, on the raw bytes. Phase 5C's byte split and bit
  planes were run with zstd 19.
- **nvCOMP's ANS:** Phase 6B's float16 mode for BF16, byte mode for int4; 1 MiB chunks, 8 bytes of index per chunk.
- **Colibrì's int4-rans256-g0, its own C:**
  - **g0:** one table from the pooled histogram of every matrix of a source, which is its rule; 256 streams; 64 bytes of
    table counted once.
  - **per matrix:** the same codec with a table per matrix.
- **`lmc`, this study's coder:**
  - A static rANS coder: 32-bit state, byte renormalization, 32 streams by rows, 14-bit tables for nibbles and 15-bit
    tables for BF16 exponents.
  - For BF16 it codes the exponent byte and stores sign and mantissa as one raw byte.
  - For group formats it also codes the scale stream: E8M0 bytes order-0, or a BF16 scale's exponent with its sign and
    mantissa stored.
  - This is the **strongest practical rANS reference**: order-0 tables per matrix, every byte counted.

The LZMA, bzip2 and zstd 22 rows cover the first sampled expert of each layer (Section 4.1).

### 4.1 BF16 (stored bytes over BF16 bytes)

| Model | zstd-3 | zstd-19 | zstd-19 byte split | zstd-19 bit planes (5C's best) | nvCOMP ANS (6B) | lmc order-0 | lmc, bits per weight |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Moonlight | 0.7828 | 0.7762 | 0.6724 | 0.6610 | 0.6742 | **0.6594** | 10.550 |
| OLMoE | 0.7781 | 0.7729 | 0.6686 | 0.6623 | 0.6692 | **0.6603** | 10.564 |
| Granite | 0.7798 | 0.7773 | 0.6713 | 0.6650 | 0.6713 | **0.6626** | 10.602 |

On the subset of each layer's first expert:

| Model | LZMA 9e | LZMA 9e, byte split | bzip2 9 | zstd 22 | lmc order-0 |
| --- | --- | --- | --- | --- | --- |
| Moonlight | 0.7126 | 0.6877 | 0.6898 | 0.7764 | 0.6592 |
| OLMoE | 0.7142 | 0.6840 | 0.6935 | 0.7725 | 0.6601 |
| Granite | 0.7234 | 0.6867 | 0.6969 | 0.7773 | 0.6631 |

**Findings** [Verified, `experiments/lossless_moe/stage-a-bf16-run1`]:

- **The static coder is the best codec on every model.** zstd 19 plus bit planes, Phase 5C's best, comes within 0.2–0.4%.
- **LZ matching finds nothing.** zstd 22 equals zstd 19, and LZMA 9e beats zstd only through its better entropy stage.
- **Context-selected tables lose on real bytes,** because their tables cost more than they save. On Moonlight:
  - the previous exponent: 0.6617;
  - a class per column: 0.6610;
  - a class per row: 0.6608;
  - order-0: 0.6594.
- **A shared table trained on calibration matrices** costs 0.6602 on held-out matrices, against 0.6593 with tables per
  matrix (plus 512 bytes per model and kind).
- **12-bit tables** cost 0.0002 against 15-bit ones (0.6595).

### 4.2 Quantized experts (stored bytes: weights plus scales)

| Source | Scales' share of stored bytes | zstd-19 (weights and scales) | nvCOMP ANS (scales raw) | Colibrì g0 (scales raw) | Colibrì per matrix | **lmc order-0 (reference)** | lmc byte pairs | Reference, bits per weight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GLM-5.2, per-row int4 | 0.22% | 0.7460 | 0.7588 | 0.7376 | 0.7372 | **0.7369** | 0.7398 | 2.9540 |
| Inkling, per-row int4 | 0.17% | 0.7421 | 0.7547 | 0.7328 | 0.7326 | **0.7324** | 0.7355 | 2.9348 |
| Kimi-K2, int4 g32 | 11.11% | 0.8665 | 0.9224 | 0.9163 | 0.9163 | **0.8668** | 0.8668 | 3.9006 |
| MiMo-V2.6 Pro, MXFP4 | 5.88% | 0.8880 | 0.9440 | 0.9478 | 0.9384 | **0.8865** | 0.8860 | 3.7674 |
| gpt-oss-20b, MXFP4 | 5.88% | 0.9199 | 0.9722 | 0.9681 | 0.9677 | **0.9221** | 0.9204 | 3.9189 |
| *Moonlight re-quantized (diagnostic)* | *0.45%* | *0.7668* | *0.7808* | *0.7615* | *0.7615* | ***0.7603*** | *0.7627* | *3.0549* |

On the subset of each layer's first expert, weights only: GLM's LZMA 9e reaches 0.7492, bzip2 0.7989 and zstd-19
0.7470, against lmc order-0 at 0.7366. The other sources follow the same order [Verified,
`experiments/lossless_moe/stage-q-run1/blocks.jsonl`].

**Findings** [Verified, `experiments/lossless_moe/stage-q-run1`]:

- **Colibrì's figures reproduce.** Its codec on GLM-5.2's container gives **0.7376**, the ratio its maintainers reported
  for their 340 MB sample. The per-matrix order-0 entropy is 2.90–2.92 bits for gate and up and 2.99–3.02 for down; per
  layer, 2.94–2.95 at layers 10, 30, 50 and 76. That agrees with its census (2.94–2.95 at those layers; 2.9147 pooled
  with layer 3 at 2.77). Its document's "about 0.76" is a full-corpus figure this study did not reproduce [Unknown].
- **The scale stream is where Colibrì's format leaves bytes on group formats.**
  - Coding the scales (and using tables per matrix) beats Colibrì's codec with raw scales by **5.4% on Kimi-K2, 6.5% on
    MiMo and 4.7% on gpt-oss**. That is a modeling choice inside static rANS, not a new codec.
  - On per-row int4, scales are 0.2% of the bytes, and lmc beats Colibrì's codec by 0.05–0.1%.
- **One table per shard costs 1.0% on MiMo** (0.9478 against 0.9384 with tables per matrix), because its layers differ
  (Section 10). Elsewhere it costs 0.05% or less.
- **General-purpose codecs stay behind:**
  - zstd reaches order-0 only on the scale streams and on nibbles that are already nearly uniform;
  - nvCOMP's byte-mode ANS sits 0.7–3% behind Colibrì's codec (both with raw scales; nvCOMP's tables are per chunk);
  - LZMA and bzip2 sit further behind.
- **Byte pairs** (two nibbles per decode step, 12-bit tables of 256 symbols) cost 0.4% more on per-row int4. They gain
  0.06–0.18% on MXFP4, where adjacent codes depend on each other slightly (Section 5.3).

### 4.3 The trained-table question

A table trained on calibration matrices and applied to held-out matrices compares with tables per matrix as follows
[Verified, held-out groups in `contexts.json`]:

| Corpus | Penalty of a trained table, bits per code |
| --- | --- |
| GLM-5.2, Inkling, Kimi-K2 | 0.0000–0.0008 |
| gpt-oss | 0.0006–0.0053 |
| BF16 exponents | 0.002–0.022 |
| MiMo | 0.009–0.11 (its layers differ) |

Tables per matrix cost 32 bytes each, a few millionths of a bit per weight. They are the right default; a trained or
shared table saves nothing worth having.

### 4.4 The coder is not where the gains are

`lmc`'s order-0 bytes stand against the order-0 entropy of the same matrices as follows [Verified]:

- **int4 and MXFP4: +0.004% to +0.007%**, with headers, tables, offsets and stream flushes counted.
- **Colibrì's g0:** +0.03% to +0.10%, and +1.1% on MiMo, where its pooled table does not fit the layers.
- **BF16 exponent records:** +0.09% to +0.46%. The 256-symbol tables weigh most on Granite's small matrices.

A new entropy coder therefore has nothing to gain at order-0. The question is entirely one of the model (Section 5).

## 5. Entropy and conditional entropy (Stage B)

### 5.1 Method

Each matrix's symbols are measured under every context model.

- **Symbols:** BF16 exponent, mantissa and sign; int4 and E2M1 nibbles; scale streams.
- **Estimators, from optimistic to achievable** (`lossless_moe/entropy.py`; tested against their definitions):
  - **in-sample**: the plug-in conditional entropy, with tables free and fitted to the data they code. Biased low; the
    bias grows with the number of contexts.
  - **two-part**: the same, plus every used table at 16 bits per entry. Achievable by a static coder.
  - **adaptive**: the code length of an adaptive arithmetic coder with Krichevsky–Trofimov counts per context. Achievable,
    with every learning cost included and no side information.
  - **held out**: tables trained on calibration matrices and scored on evaluation matrices, as cross-entropy and as a
    calibration-prior adaptive code. Neither set shares a matrix with the other.
- **Splits:** held-out experts (alternate experts of each layer) and held-out layers (the first half of the layers
  calibrates, the second evaluates).
- **Side information is charged:** classes per column or row at their bits, rank-one terms at 8 bits each, quantile
  edges at 32 bits. Contexts a decoder derives from what it already holds are free: the stored scales, the same expert's
  gate before up, gate and up before down.
- **Matched comparison:** a context available only for some matrices (up given gate, down given its neurons, a matrix
  given the previous expert) is compared with order-0 on exactly those matrices.
- **Assumptions and reliability:**
  - An empirical conditional entropy is not a lower bound on what any model could do. It bounds what these models do,
    estimated on 0.2–0.9 G weights per corpus.
  - The adaptive and held-out figures are achievable code lengths, not estimates of an unknown quantity.
  - The in-sample figures exceed the adaptive ones by at most 0.0008 bits per weight on quantized experts and 0.016 on
    BF16, so estimation bias does not hide anything (Section 5.5).

**Context families.** All contexts for BF16 apply to the exponent unless noted; for quantized formats they apply to the
code, the scale classes being derived from the stored scales.

| Family | BF16 contexts | Quantized-format contexts |
| --- | --- | --- |
| Neighbours | previous one or two symbols; the row above; both | the same |
| Running statistics (causal) | running row statistics (window 8 or 64); running column mean | running row statistics (window 8, 64 or 512); running column mean |
| Side-information classes | class per column, per row, or rank-one; combinations | the same |
| Scale-derived classes | — | each group's scale relative to its row (per-row formats: the row scale relative to the matrix); the block-maximum state |
| Cross-matrix | gate (for up), the previous expert, neuron classes (for down) | the same |
| Inside a symbol or scale | mantissa given the exponent | scales given their neighbours, row or column class; a BF16 scale's mantissa given its exponent |

### 5.2 BF16

Order-0 per field, in bits per weight [Verified, `experiments/lossless_moe/stage-b-bf16-run1`]:

| Model | Exponent | Mantissa | Sign | Total |
| --- | --- | --- | --- | --- |
| Moonlight | 2.547 | 6.973 | 1.000 | 10.520 |
| OLMoE | 2.561 | 6.973 | 1.000 | 10.534 |
| Granite | 2.591 | 6.974 | 1.000 | 10.564 |

Best contexts per field, matched, in bits per weight:

| Field | Best context | Adaptive gain | In-sample gain |
| --- | --- | --- | --- |
| Exponent | Row class (Granite) | 0.0113 | 0.0152 |
| Exponent | Moonlight and OLMoE, any context | ≤ 0.0038 | ≤ 0.0050 |
| Mantissa | Given the exponent: the within-binade slope, almost all of it in the top three bits | 0.037–0.047 | 0.049–0.051 |
| Sign | Any context | 0.0000 | 0.0001 |
| Exponent | Up given gate (cross-projection) | −0.0005 (a loss, Moonlight) to 0.0015 (Granite) | 0.0002–0.0047 |
| Exponent | Given the previous expert | −0.0007 to −0.0026 (a loss on every model) | 0.0001–0.0006 |

Held out, the best exponent context saves 0.0006–0.0161 bits.

**Linear structure** (`linear.json`) [Verified]:

- **Adjacent weights:** |ρ| ≤ 0.005 along rows and ≤ 0.073 down columns (one Granite matrix; ≤ 0.017 on Moonlight).
- **Gate and up:** |ρ| ≤ 0.015, element by element.
- **Low rank:** rank 16 captures a median 4.6% of Moonlight's energy, against 3.6% for the same matrix permuted at random.
  Paying the factors at 16 bits makes every rank lose (−0.005 bits at rank 1, −0.25 at rank 16). One Granite matrix gains
  0.043 bits at rank 1.

The upper bound of a linear predictor's gain, 0.5·log2(1/(1 − ρ²)), is below 0.0003 bits at |ρ| ≤ 0.02, and 0.004 bits
for the one Granite matrix at 0.073.

### 5.3 int4 and MXFP4

Order-0 entropy of the codes, adaptive, bits per code, per matrix [Verified, `experiments/lossless_moe/stage-q-run1`]:

| Source | Range | Note |
| --- | --- | --- |
| GLM-5.2 | 2.90–3.02 | |
| Inkling | 2.84–2.98 | |
| Kimi-K2 | 3.60–3.62 | |
| MiMo | 2.98–3.83 | Layer 10 at 2.98–3.53, other layers 3.80–3.83 |
| gpt-oss | 3.80–3.88 | Nearly uniform |
| Diagnostic | about 3.04 | Comparable with Colibrì's community measurement on Qwen3.6 |

Gains beyond order-0 on the codes, matched, adaptive, in bits per code [Verified]:

| Context | GLM-5.2 | Inkling | Kimi-K2 | MiMo | gpt-oss | Diagnostic |
| --- | --- | --- | --- | --- | --- | --- |
| Previous code | 0.0000 | 0.0000 | 0.0009 | 0.0042 | 0.0145 | −0.0003 |
| Previous two codes | −0.0008 | −0.0005 | 0.0006 | 0.0068 | 0.0273 | −0.0035 |
| Row above | −0.0001 | 0.0000 | −0.0001 | 0.0248 | 0.0026 | −0.0004 |
| Running row mean, 512 | 0.0051 | 0.0056 | 0.0001 | 0.0065 | 0.0026 | 0.0063 |
| Running column mean | 0.0006 | 0.0023 | 0.0003 | 0.0659 | 0.0323 | 0.0000 |
| Class per column | −0.0006 | 0.0015 | −0.0008 | 0.0654 | 0.0300 | −0.0021 |
| Class per row | **0.0068** | 0.0073 | −0.0008 | 0.0064 | 0.0015 | **0.0071** |
| Rank-one class | 0.0039 | **0.0075** | −0.0042 | 0.0769 | 0.0309 | 0.0005 |
| Scale class (from the stored scales) | 0.0056 | 0.0051 | 0.0177 | 0.0456 | 0.0579 | 0.0065 |
| Block-maximum state | — | — | 0.0165 | 0.0039 | 0.0066 | — |
| Block maximum and scale class | — | — | **0.0356** | 0.0504 | 0.0632 | — |
| Scale class and column class | 0.0046 | 0.0063 | 0.0166 | **0.1126** | **0.0916** | 0.0031 |
| Up given gate | 0.0014 | 0.0001 | 0.0003 | 0.0320 | — | −0.0003 |
| Down given neuron classes | −0.0001 | 0.0000 | −0.0001 | 0.0210 | — | −0.0005 |

A negative entry is a loss. The in-sample gains are the same to within 0.0012, except for the two-symbol neighbour
contexts, whose larger tables cost up to 0.0045 bits to learn.

**Scale streams** [Verified], in bits per scale:

- **MXFP4 E8M0.** Order-0 is 0.95 (MiMo) and 1.82 (gpt-oss) of 8 bits; it is already in the reference. A class per row
  saves 0.13 and 0.90 more, the previous scale 0.07 and 0.66. Per weight, that is 0.004 and 0.028 bits.
- **Kimi-K2 BF16 scales.** The exponent takes 0.89 bits, and the mantissa given its exponent saves 0.62 bits per scale:
  0.019 per weight.

**What the measurements say** [Inference]:

- **Per-row int4 (GLM, Inkling, the diagnostic).** The only structure is each row's spread relative to its own maximum,
  which a row class, a running row mean or the stored scale captures. It is worth **0.006–0.009 bits**.
  - That is consistent with Colibrì's community measurement on Qwen3.6, where per-row tables measured 1% under
    per-expert ones, in-sample and before their own cost.
  - A table per 2,048-symbol row would cost more than that 1%.
  - Neighbours, columns, other projections and other experts carry nothing measurable.
  - The re-quantized diagnostic gives the same figures as the real containers, so a per-row container adds no structure
    to what BF16 weights hold.
- **Group-scaled int4 (Kimi-K2, quantization-aware training).** The quantizer itself leaves structure:
  - every group of 32 holds a ±7 (the block-maximum context);
  - the group's scale relative to its row predicts the spread of the other 31 codes.

  Together they are worth 0.036 bits per code, and the scale mantissa given its exponent another 0.019 per weight.
- **MXFP4 (MiMo, gpt-oss).** Three effects add up to **0.12–0.13 bits per weight**:
  - per-column variance survives the 32-wide blocks;
  - the block scale relative to its row helps, as above;
  - MiMo's layers and its −0 artifact differ.

  It is the largest effect found, and it is 3% of the reference.

### 5.4 Reversible transformations

- **Byte split and bit planes (BF16).** They raise LZ codecs to order-0 (0.661) and not beyond, as Phase 5C found.
- **Byte pairs (int4).** They code the joint of two adjacent nibbles: worse on per-row int4, 0.2% better on MXFP4
  (Section 4.2).
- **Deltas.** Delta coding along rows or columns is a special case of the neighbour contexts above, which find nothing.
  Phase 5C showed XOR and modular deltas between experts cost more than the experts alone.
- **Permutations.** A permutation of rows or columns must itself be stored, about n·log2 n bits. It cannot reveal
  structure the class contexts do not already see, and the class contexts gain at most 0.11 bits [Inference].

### 5.5 Gate (pre-registered)

The best achievable gain comes from letting every matrix and field use its own best context, chosen in hindsight
(signalling the choice costs a few bits per matrix).

- **Reductions** against the reference are 0.24–3.3% on quantized experts and 0.5% on BF16 (the table in Section 1).
- **Optimistic in-sample figures** exceed the achievable ones by at most 0.0008 bits on quantized experts and 0.016 on
  BF16.
- **Held out by expert and by layer,** the best context's gain is about the same or smaller (`experiments/lossless_moe/report/tables.md`).

**Verdict on every corpus: a large improvement is unsupported by the evidence.** Stage C stops here, as the brief
requires. No candidate was built beyond the context-selected tables `lmc` already carries (Section 6).

## 6. Candidate mechanisms

Stage B's gate stopped Stage C. Each candidate below is described as the brief asks, with its measured ceiling in place
of a projection.

| Candidate | Mechanism (encode, decode) | Redundancy it targets | Why order-0 rANS misses it | Metadata | Decoder complexity and memory | Random access | Measured ceiling | Fails when |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A. Conditional model | rANS with one table per context; the decoder recomputes the context | Neighbours, rows, columns, scale classes, cross-projection | Static tables code every symbol with one distribution | Tables per context (16 to 257 per matrix), side classes | One table lookup per symbol; tables of 20–160 KB per context; 19–31% slower | Per matrix or row block | 0.007–0.009 bits (per-row int4), 0.13 (MXFP4) | Context information is below its table cost: true on every corpus |
| B. Predictive residual | Predict each weight (neighbours, low rank, gate for up); code the residual | Linear dependence | No predictor | Predictor parameters (low rank: k·(m+n) values) | A multiply-add per weight, plus the factors | Per matrix | |ρ| < 0.02; low rank loses at every rank | Correlations are tiny: measured |
| C. Hierarchical blocks | Independent blocks, each with its own table | Local changes of distribution | One table per matrix | A table per block | As order-0; more tables | Finer | Finer blocks cost more: from per matrix to 512, 64 and 8 rows, 0.04%, 0.3% and 1.0% more bytes (Section 8.4) | Distributions are stationary within a matrix: measured |
| D. Structured blocks | Block distributions, bit layouts, permutations, shared dictionaries | Repeats, cross-expert reuse | No sharing | Dictionaries, permutations | LZ-like | Shared state | LZ matching finds nothing; Phase 5C: 0 of 1,536 shared bases pay; a permutation costs n·log2 n bits | Trained experts share nothing: measured twice |
| E. Format-aware coding (original direction) | Code a group format's scale stream with a context (previous scale, row class) and its codes given (block-maximum state, scale class) | Quantizer artifacts: a maximal code per group, scale spread, scale stream | Colibrì's format leaves scales raw; order-0 ignores the group's maximum | None: classes derive from the stored scales | One more lookup per symbol | Per matrix | Scales coded order-0 save 4.7–6.5% against Colibrì's format (adopted into the reference); contexts add 1.4% (int4 g32) and 3.1–3.3% (MXFP4) | Per-row formats have neither artifact; MXFP4 codes are nearly uniform |

**Candidate E, in brief.** It targets the one place where measurable bytes exist: the scale streams and maximal codes of
group formats.

- **The scale stream.** Its order-0 part is real, about 5–6.5% against Colibrì's format, but it is a modeling choice
  inside a static coder. The study counts it in the reference rather than as a new codec's gain.
- **The context part.** It adds the 1.4–3.3% above.
- **Novelty.** Modest at best: context coding of scales and maximal-code constraints follow well-known practice
  (DeepCABAC's significance flags; the scale coding of image codecs) [Inference].

## 7. Implementation

All of it is research code on the research branch (`research/lossless_moe/`). Nothing in `src/`, `benchmarks/`,
`configs/` or `native/` changed, so Weightsift's runtime is untouched.

- **Python harness** (Weightsift's environment, no new dependency):
  - `lossless_moe/symbols.py`: exact field splits and their inverses.
  - `entropy.py`: the estimators. `contexts.py` and `measure.py`: the context models and their accounting.
  - `quantized.py`: formats, scales, block contexts. `linear.py`: linear diagnostics.
  - `corpus.py`: the sample and two read paths. `records.py`: environment and digests.
  - Drivers: `stage_a.py`, `stage_b.py`, `stage_q.py`, `stage_d.py`, `requantize.py`, `fetch_sample.py`, `report.py`.
  - 20 tests (`research/lossless_moe/tests`).
- **`lmc`, a Rust prototype** (`research/lossless_moe/codec`, Rust 1.95.0, about 2,100 lines with tests):
  - **The coder.** Static rANS (32-bit state, L = 2^23, byte renormalization; scale bits 12–15) with one table per
    context. Context kinds: none, a class per column, a class per row, the previous symbol, and a class per (row, group)
    derived from scales.
  - **Streams and decoding.** Rows go round-robin to 32 streams. The decoder runs 8 streams in lockstep with a
    branch-free refill, and checks each stream's end: exact consumption and final state equal to L.
  - **Records.** `LMC1` block records with every byte counted; the parser refuses inconsistencies by name.
  - **Sinks.** Output goes straight into BF16 patterns, packed nibbles or bytes.
  - **Colibrì's codec** is compiled by `build.rs` from its pinned, hash-verified C (`c/tools/rans_ctypes.c` over
    `c/rans.h`, MSVC, `/O2`) and called through its exported bridge with both its arms.
  - **Drivers:** `lmc bench` (encode, verify, timed decode at 1, 6 and 12 threads) and `lmc random-access` (cold
    unbuffered reads of single experts).
  - **Tests:** 9 (Section 9). `cargo clippy -D warnings` is clean.
- **Reused as published:** zstandard, CPython's lzma and bz2, nvCOMP through `awpmi.streaming`, Colibrì's `rans.h`,
  `rans_ctypes.c` and `rans_format.py` (its table rule, imported unmodified), Phase 5C's byte split and bit planes,
  SciPy's `gammaln`, and safetensors' loader as the independent read.

## 8. Compression and decompression benchmarks (Stage D)

Every timing ran with nothing else on the machine, best of 5 repeats. Decode tables are built inside the timed region.
Output rates are in GB/s of reconstructed stored bytes.

### 8.1 CPU decode, all experts of the sample (i7-8700K, AVX2)

| Decoder | 1 thread | 6 threads | 12 threads | CPU use per thread |
| --- | --- | --- | --- | --- |
| lmc order-0, int4 nibbles (14-bit tables) | 0.156 | 0.821 | 0.822 | 0.99–1.00 |
| lmc byte pairs (12-bit, 256 symbols) | 0.354 | 1.909 | 1.918 | 0.98–1.00 |
| Colibrì scalar arm (its C) | 0.095 | 0.520 | 0.523 | 0.99–1.00 |
| Colibrì branch-free arm (its C) | 0.086 | 0.469 | 0.469 | 0.98–1.00 |
| lmc, previous-code context | 0.115 | 0.623 | 0.624 | |
| lmc, class per column | 0.108 | 0.579 | 0.585 | |
| lmc, class per row | 0.127 | 0.671 | 0.678 | |
| lmc, scale class | 0.110 | 0.580 | 0.581 | |
| lmc order-0, BF16 (15-bit) | 0.52 | 2.87 | 2.88 | 0.96–1.00 |
| lmc order-0, BF16 (12-bit) | 0.60 | 3.28 | 3.28 | |
| zstd-19 decoding bit planes, BF16 (Python) | 1.76 | — | — | |

[Verified, `stage-q-run1/summary.json`, `stage-a-bf16-run1/summary.json`]

- **Scaling.** Six threads give about 5.3×, and twelve add nothing: the OS reports 6 logical processors.
- **Colibrì's AVX-512 arm** is unavailable on this CPU.
- **Context models cost 19–31% of decode speed.**
- **Memory.** The bench process peaked at 3.0–4.4 GB, holding every block, record and output. A decoder itself needs one
  table per context (20–160 KB) and 4 bytes of state per stream.
- **Encoding.** lmc encodes at 53 MB/s per thread for int4 and 210 MB/s for BF16; Colibrì's encoder at 46 MB/s.

### 8.2 Cold random access to one expert (unbuffered reads, system NVMe)

Every expert was read once in a seeded random order, from two 4 KiB-aligned packs. One pack holds the stored bytes; the
other holds the lmc records, with scales stored raw. Each decoded expert was checked against the sha256 of its stored
bytes. All were exact [Verified, `experiments/lossless_moe/stage-d/stage_d.json`].

| Source | Experts | Record (MB) | Stored (MB) | Read the record, median (ms) | Read stored, median (ms) | Decode, one thread, median (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| GLM-5.2 | 16 | 13.94 | 18.92 | 5.6 | 8.5 | 128.9 |
| Inkling | 16 | 20.77 | 28.36 | 8.9 | 12.7 | 193.8 |
| MiMo | 16 | 18.81 | 20.05 | 7.9 | 9.1 | 134.1 |
| Kimi-K2 | 16 | 22.69 | 24.77 | 9.3 | 11.2 | 160.0 |
| gpt-oss | 16 | 12.79 | 13.22 | 5.2 | 6.0 | 87.5 |
| Moonlight (BF16) | 48 | 11.41 | 17.30 | 4.8 | 7.8 | 42.8 |

- **Bytes physically read** are each record rounded up to 4 KiB: amplification under 0.03%.
- **Compression shortens the read in proportion to the bytes,** but one thread's decode takes 14–15 times as long as
  reading the stored bytes (5.5 times for BF16). Even spread over 6 threads (Section 8.1), decode stays above the read
  time it saves.
- **p90 times** lie within 5% of the medians, except Inkling's stored reads (14.9 ms against 12.7).

### 8.3 GPU decode (nvCOMP ANS, byte mode, RTX 4060 Ti)

| Source | 1 MiB chunks: ratio of weights | 1 MiB chunks: GB/s | 64 KiB chunks: ratio | 64 KiB chunks: GB/s |
| --- | --- | --- | --- | --- |
| GLM-5.2 | 0.7536 | 138.8 | 0.7611 | 135.5 |
| Inkling | 0.7492 | 136.5 | 0.7567 | 136.8 |
| MiMo | 0.9361 | 124.2 | 0.9481 | 122.7 |
| Kimi-K2 | 0.9082 | 128.1 | 0.9205 | 125.6 |
| gpt-oss | 0.9667 | 125.9 | 0.9782 | 124.0 |

One batched launch, timed with CUDA events after a warm-up; the output was compared with the input [Verified]. The GPU
decodes about 40 times faster than the drive delivers. Its ratio is 1–2.5% worse than lmc's because of chunked tables.

### 8.4 Block granularity

lmc order-0 with independent blocks of fewer rows, weights only [Verified]:

| Source | Per matrix | 512 rows | 64 rows | 8 rows |
| --- | --- | --- | --- | --- |
| GLM-5.2 | 0.7363 | 0.7366 | 0.7388 | 0.7438 |
| Inkling | 0.7320 | 0.7322 | 0.7339 | 0.7378 |
| MiMo | 0.9343 | 0.9346 | 0.9367 | 0.9417 |
| Kimi-K2 | 0.9056 | 0.9059 | 0.9079 | 0.9128 |
| gpt-oss | 0.9654 | 0.9658 | 0.9685 | 0.9751 |

Per-expert random access, and row blocks down to 512 rows, cost nothing measurable. Blocks of 8 rows cost about 1%.

### 8.5 The brief's ten metrics, for the best demonstrated variant

The best demonstrated variant is lmc with a class per column, on MiMo [Verified]:

1. **Total bytes:** 0.8715 of stored, against 0.8865 for the reference.
2. **Bits per weight:** 3.704, against 3.767.
3. **Ratio:** 0.8715.
4. **Improvement over rANS:** 1.69%.
5. **Decode throughput:** 0.108 GB/s per thread, against 0.156 for the reference.
6. **CPU use and memory:** about 1.0 per thread; tables of 20 KB per class.
7. **Cold latency:** about 30% more decode time than the reference [Inference from the throughput; random access was
   timed for the reference's records only].
8. **Disk bytes per expert:** about 1.7% fewer [Inference from the ratio].
9. **Encode speed:** 53 MB/s per thread, like the reference.
10. **Independent decoding:** yes, per matrix or row block.

On per-row int4, the best variant (a class per row) saves 0.19–0.20% at about 20% slower decoding.

## 9. Exact reconstruction

[Verified]:

- **Every codec on every matrix** was decoded and compared with the stored bytes in every run: zstd, LZMA, bzip2, nvCOMP,
  5C's transforms, Colibrì's two arms, and every lmc variant. lmc and Colibrì outputs were also checked by sha256 against
  safetensors' own read (BF16), or against the fetched slice (quantized). The timed decodes were compared again after
  timing.
- **Independent reads.**
  - BF16 matrices were read by positioned reads and by `safe_open`, and the two were compared.
  - Local files were hashed in full against the publisher's sha256.
  - Fetched slices were hashed, and two windows of each were re-requested and compared.
  - Every quantized-stage load re-hashes each slice against the manifest.
- **No hidden reliance on the original.**
  - lmc's decoder gets only the record. External tables and derived classes are passed explicitly and counted.
  - The derived classes are computed from the stored scales, which the record's consumer holds.
  - The random-access benchmark reads records from a pack and checks decoded outputs only against sha256 values.
- **Edge cases and corruption** (`codec/tests/roundtrip.rs`):
  - every context kind;
  - degenerate shapes: 1×1, a single row or column, more streams than rows;
  - alphabets of one symbol, of 16 and of 256;
  - external tables required, and refused when wrong;
  - framing refusals: truncation, a trailing byte, bad magic, tables that do not sum;
  - **every single-bit flip of a payload** is either refused by the checked decoder or changes the output, which the
    sha256 then catches;
  - Colibrì's codec round-trips through its own bridge.
- **Python tests** (`research/lossless_moe/tests`):
  - every field split restores all 65,536 BF16 patterns;
  - the estimators equal their definitions;
  - causal contexts never look ahead, with a negative control;
  - compressed-tensors' int32 packing equals low-nibble-first bytes;
  - Colibrì's per-row recipe produces ±7 maxima and never −8.
- **Determinism.** The BF16 and quantized stages ran twice each, with `PYTHONHASHSEED` 1 and 2 on the same study tree.
  The digests are identical: Stage A `1c7551b6…`, Stage B `8af0ae75…`, quantized `bfc0e112…`.

## 10. Generalization across experts, layers and formats

- **Held out by expert and by layer** [Verified]:
  - on per-row int4 and Kimi-K2, a context's gain on evaluation matrices matches its calibration gain to a median
    0.0000–0.0004 bits (largest difference 0.011, Inkling's down matrices held out by layer);
  - the best held-out context saves 0.0065–0.0098 bits on GLM and Inkling and 0.036 on Kimi-K2;
  - on MXFP4 it saves 0.04–0.24 bits, the 0.24 being one MiMo split (gate matrices, held out by expert). There, gains do
    not transfer reliably: MiMo's rank-one class gains 0.24 bits on its calibration layers, which include layer 10, and
    nothing on the evaluation layers.
- **Across layers** [Verified]:
  - per-row int4 entropy varies by 0.01–0.04 bits between layers (GLM's layer 76 is lowest for down);
  - Kimi-K2 varies by under 0.003;
  - **MiMo's layer 10 is the exception**: its codes have 0.1 (down) to 0.8 bits (some gate matrices) less entropy than
    layers 30, 50 and 68. Tables per matrix absorb it.
- **Across experts:** within a layer and kind, order-0 entropy varies by up to 0.02 bits on GLM and Kimi-K2, 0.10 on
  Inkling, 0.06 on gpt-oss and 0.61 on MiMo [Verified].
- **Across projections:** down matrices carry about 0.1 bit more per code than gate and up on per-row int4
  (2.99–3.02 against 2.90–2.92), because a row of down spans the expert's narrow dimension [Inference].
- **Across architectures:** Moonlight (DeepSeek-V3 type), OLMoE, Granite-MoE, GLM-5.2, Inkling, MiMo, Kimi-K2 and gpt-oss
  all show the same picture [Verified]. Order-0 takes nearly everything; per-row int4 leaves under 0.01 bits; group
  formats leave 0.05–0.13 bits through their quantizer's artifacts.
- **Across formats:** BF16 0.5%, per-row int4 0.24–0.31%, group int4 1.4%, MXFP4 3.1–3.3% [Verified]. No format comes
  within a factor of three of the 10% "limited prototype" threshold.

## 11. Practical implications for Colibrì and DS4

The figures below are calculations on the Phase 7B model of Inkling on this machine (its § 5), not measured engine
speedups. A decode token reads E·b·(1 − h) = 10.89 GB × 0.85 = 9.26 GB of int4 experts from a 3.2 GB/s drive: 2.89 s
[Inference, 7B].

| Scenario | Drive bytes per token | Drive time | Decode cost per token on this machine | Net effect on a disk-bound token |
| --- | --- | --- | --- | --- |
| Stored int4 (Colibrì today) | 9.26 GB | 2.89 s | 0 | — |
| Static rANS (measured e = 0.7324 on Inkling) | 6.78 GB | 2.12 s | CPU: 4.8 s (byte pairs, 6 threads), 11.3 s (lmc order-0), 17.8 s (Colibrì's scalar arm). GPU: about 0.07 s | CPU: a net loss of 2–17 s (perfect overlap to serial). GPU: up to 1.36× on the drive term, only if the GPU consumes the experts |
| Best context model (+0.3% beyond static) | 6.76 GB | 2.11 s | CPU 20–30% slower than static | At most +0.3% even with free decoding |
| MXFP4 container, best context (+3.3%) | — | −3.3% of the static coder's | as above | At most +3.5% even with free decoding |

- **Where a decoder would fit in Colibrì.** At cache fill, between the expert read (`slot_fill` and its siblings) and the
  slot the kernels read; on the GPU, between the PCIe copy and the expert kernel. This is Colibrì's issue #594 boundary
  ladder [Verified as its design]. Its CPU arm was measured, and rejected, by its own maintainers on Zen 5 and NVMe: end
  to end 4–5× slower, break-even at a drive of about 140–260 MB/s.
- **This machine reaches the same verdict.** The best CPU decoder measured, 1.9 GB/s on all 6 cores, decodes 9.26 GB in
  4.8 s. That is more than the 0.77 s of drive time static coding saves, and it uses the cores the expert arithmetic needs
  [Inference from measured rates].
  - Even with perfect overlap, breaking even needs at least 3.2 GB/s of int4 output on cores otherwise idle, and the
    full gain 4.4 GB/s.
  - That is 1.7–2.3× the byte-pair decoder on every core. An AVX2 SIMD decoder might approach it [Hypothesis, not
    built].
- **The GPU path is the only one that clears the drive** (122–139 GB/s, Section 8.3). It helps only where experts are
  computed on the GPU (Phase 7B's H3) or held compressed in VRAM (Colibrì's ANS tier: +13.9% capacity, +4.5% decode on
  its host).
  - Decoding on the GPU and copying back to the CPU costs two PCIe crossings, about 2.7 s per token here: no gain
    [Inference].
- **Cache capacity.** A compressed host cache holds 1/e times more experts: 1.37× for per-row int4 with the static
  coder, 1.08–1.13× for MXFP4. A context model adds ×1.003 (per-row int4) or ×1.035 (MXFP4) [Inference]. Every hit then pays a
  decode.
- **RAM and VRAM minima are unchanged.** The minimum to run is the dense set, the KV state and working buffers; routed
  experts stream. Lossless expert coding shrinks the checkpoint on disk and the cache's footprint per expert, not the
  minimum memory [Inference]. Inkling's dense set still needs its lossy int4 conversion to fit in 32 GB (7B § 4.1).
- **Cold and warm.** Cold prefill reads every touched expert once, so its bytes scale with e (static coding: about
  −27%). Warm decode scales with e·(1 − h). Each saving is lost again when decoding cannot keep up.
- **Throughput at 25%, 50% or 100% more.** Not from a better lossless codec: measured headroom beyond static rANS is
  0.3–3.5%. The static coder's own 1.36× ceiling on the drive term (7B § 5.6) exists whatever codec carries it, and on
  this machine it pays only through a GPU.
- **DS4** (antirez/ds4 at `fc80bd69`). It streams GGUF experts through a bounded cache, with no entropy coding.
  - Its DeepSeek V4.1 files use IQ2_XXS for gate and up: codebook indices, which the E8/IQ3 and VQ3R measurements show to
    be near-incompressible.
  - Its Q2_K down projections and Q4_K variants are block quantizations like Kimi-K2's group int4. Expect an order-0 gain
    from coding codes and scales, and contexts worth about 1–3% more [Inference, not measured].
  - Its main targets are unified-memory Macs, where decoding competes with GPU compute for the same memory bandwidth
    [Inference].
- **An independent Rust library with a C ABI.** `lmc` is small, has a checked record format and could become one in days
  [Inference].
  - Its gain over Colibrì's `rans.h` on per-row int4 is 0.05–0.1%, too little to justify a library.
  - The worthwhile pieces are a format extension and existing decoders: per-matrix tables plus coded scales for group
    formats (4.7–6.5% against Colibrì's format) belong in Colibrì's format, and the GPU decoders already exist (nvCOMP,
    DietGPU).

## 12. Originality

- **The measurement is the contribution.** No published work found measures the conditional entropy of real MoE int4 and
  MXFP4 experts under rich context models with held-out validation and every cost charged [Verified by the review in
  Section 3; its coverage is limited to what was found]. Colibrì's community measured order-0 and per-row tables; papers
  measure order-0.
- **No new codec is claimed.** `lmc` is a textbook static rANS coder with context-selected tables. Candidate E's pieces
  are known practice: scale coding, and maximal-code constraints after DeepCABAC's significance modeling [Inference].
- **The negative result is not new in kind** (Colibrì's thread: "a static frequency table is all the format needs").
  It is new in breadth (five real quantized MoE containers and three BF16 MoE models) and in rigour: matched contexts,
  held-out splits, two seeds, Colibrì's own codec reproduced.

## 13. Limitations and negative results

- **Sample size.** 16 experts per quantized source and 32–48 per BF16 model, in 4–6 layers. Per-row int4 varies too
  little for a larger sample to move a 0.3% result toward 10%. MiMo's layer heterogeneity suggests other layers could
  differ, but its best case would need a 30× larger effect to reach 10% [Inference].
- **Model families.**
  - No learned predictor (a neural model of weights) was tried. The measured linear and contextual structure leaves
    nothing for one to learn cheaply, and a learned decoder would cost far more per symbol [Inference].
  - Context mixing (as in PAQ-class compressors) is bounded above by the in-sample conditional entropies measured, which
    sit at the adaptive figures [Inference].
- **Not every local structure was enumerated.** Contexts were chosen from the brief's list and the formats' known
  artifacts. A structure no tested context correlates with cannot be excluded. Every general-purpose codec tested (LZMA,
  bzip2, zstd 22) also finds less than order-0 [Verified].
- **Decoder speed.** Only scalar decoders were measured (lmc's and Colibrì's). An AVX2 SIMD rANS decoder might be 2–4×
  faster [Hypothesis]. Even then, decoding cannot sustain the drive's rate on this CPU without taking the expert
  arithmetic's cores (Section 11).
- **No engine measurement.** No Colibrì or DS4 engine ran. Every system figure in Section 11 is a calculation, labelled
  as such.
- **Colibrì's g64 containers and DS4's GGUF files were not measured.** Their conclusions are Inferences from the group
  int4 and MXFP4 results and from the published IQ3 and VQ3R figures.
- **Negative results, all [Verified]:**
  - shared and trained tables (no gain over tables per matrix);
  - neighbour contexts on per-row int4 (nothing);
  - cross-projection and cross-expert contexts (nothing; a loss when charged);
  - low-rank prediction (a loss at every rank);
  - LZ-class codecs (below order-0);
  - finer blocks (cost only);
  - context-selected tables on BF16 (a loss on real bytes).

## 14. Recommendation

**NO-GO.**

- The data do not contain the information a substantially better lossless MoE codec would need. Measured headroom beyond
  static rANS is 0.24–0.31% on the per-row int4 containers that motivated the brief, 1.4% on group int4, 3.1–3.3% on
  MXFP4 and 0.5% on BF16. The brief's thresholds (25% strong, 10% partial) are out of reach by a factor of 3–100.
- Keep Weightsift's existing runtime roadmap: the direction Phase 7B recommended, which remains your choice.
- **Two small, measured pieces are worth passing on.** Neither is a project:
  - For group-scaled containers (MXFP4, group int4), code the scale stream and use tables per matrix. That is 4.7–6.5%
    fewer bytes than Colibrì's int4-rans256-g0 applied to the codes alone [Verified].
  - Lossless coding of streamed experts pays on this machine only with GPU decoding (Phase 7B's H2 and H3) [Inference].

## 15. Reproduction and raw data

The commands are in `research/lossless_moe/README.md`. In order:

```bash
python research/lossless_moe/fetch_upstream.py                       # Colibrì's pinned files (verified)
(cd research/lossless_moe/codec && cargo build --release && cargo test --release)
python -m uv run python -m pytest research/lossless_moe/tests        # 20 tests
for n in 1 2; do                                                     # BF16 Stage A and B (about 30 min per run)
  PYTHONHASHSEED=$n python research/lossless_moe/stage_a.py --output experiments/lossless_moe/stage-a-bf16-run$n
  PYTHONHASHSEED=$n python research/lossless_moe/stage_b.py --output experiments/lossless_moe/stage-b-bf16-run$n \
      --stage-a experiments/lossless_moe/stage-a-bf16-run$n
done
python research/lossless_moe/requantize.py --manifest experiments/lossless_moe/requant-manifest.json
python research/lossless_moe/fetch_sample.py --fetch                 # 1.685 GB, with approval
for n in 1 2; do                                                     # quantized stage (about 65 min per run)
  PYTHONHASHSEED=$n python research/lossless_moe/stage_q.py --output experiments/lossless_moe/stage-q-run$n \
      --manifest experiments/lossless_moe/sample-manifest.json
done
python research/lossless_moe/stage_q.py --output experiments/lossless_moe/stage-q-diagnostic --manifest experiments/lossless_moe/requant-manifest.json
python research/lossless_moe/stage_d.py --output experiments/lossless_moe/stage-d --manifest experiments/lossless_moe/sample-manifest.json
python research/lossless_moe/report.py --stage-a experiments/lossless_moe/stage-a-bf16-run1 --stage-b experiments/lossless_moe/stage-b-bf16-run1 \
    --stage-q experiments/lossless_moe/stage-q-run1 --diagnostic experiments/lossless_moe/stage-q-diagnostic --output experiments/lossless_moe/report
```

**Raw data** (`experiments/lossless_moe/`):

- every run's `config.yaml` and `environment.json` (versions, study and source tree hashes, hash seed);
- per-matrix records (`blocks.jsonl`, `per_matrix.jsonl`), context tables (`contexts.json`), lmc results with timings
  (`lmc-*.json`), linear diagnostics (`linear.json`);
- Stage D (`stage-d/stage_d.json`), the fetch log and manifests, and the logs;
- the generated tables (`experiments/lossless_moe/report/tables.md`, `experiments/lossless_moe/report/results.json`).

The fetched slices (`samples/`) are gitignored and rebuilt by `fetch_sample.py` from the manifest's sources.

**Trees.** Each pair of compared runs ran on one frozen study tree: BF16 `1edb1ab9…`, quantized `e2ca8333…`. The committed
tree (`16958d0a…`) differs from both in two ways only:

- code those runs do not use: the quantized stage, Stage D, and the codec's grid context, which leaves the byte format of
  the other kinds unchanged, for the BF16 runs;
- the report generator, which is analysis only. After the quantized runs it was corrected to weight scale-stream gains per
  weight, and partial contexts by the matrices they cover.

Every run directory keeps the configuration it used.

## 16. Answers

**A. Is a substantially better lossless MoE codec technically plausible, based on the measured data?**

No [Verified]. On every corpus the measured information beyond a static order-0 coder is under 3.5% of its bytes. On the
per-row int4 containers it is under 0.35%.

**B. How much extra compressible information was actually found beyond conventional rANS modeling?**

Achievable, held out, every cost charged, each matrix taking its best context in hindsight [Verified]:

| Format | Bits per weight | Share of the reference |
| --- | --- | --- |
| Per-row int4 (GLM-5.2, Inkling) | 0.0072–0.0092 | 0.24–0.31% |
| Group int4 (Kimi-K2) | 0.055 | 1.41% |
| MXFP4 (MiMo, gpt-oss) | 0.120–0.126 | 3.06–3.34% |
| BF16 | 0.050–0.053 | 0.47–0.50% |

In-sample estimates are within 0.0008 bits of these on quantized experts and 0.016 bits on BF16.

The "conventional rANS modeling" here is the strongest one: tables per matrix, with scales coded. Against Colibrì's
format as shipped (one table per shard, scales raw), the static reference itself saves 0.05–0.1% on per-row int4 and
4.7–6.5% on group formats.

**C. What is the best experimentally demonstrated improvement over rANS, including metadata and decoder cost?**

1.69% fewer bytes on MiMo's MXFP4 experts, every side-information and table byte included (0.8865 to 0.8715 of stored),
at 31% lower CPU decode throughput (0.156 to 0.108 GB/s per thread) [Verified]. On per-row int4 it is 0.19–0.20% (GLM
0.7369 to 0.7355; Inkling 0.7324 to 0.7309), at about 20% lower throughput.

**D. How much would that improvement realistically help Colibrì or DS4?**

Negligibly [Inference from measured rates].

- At most 0.3% (per-row int4) or 3.5% (MXFP4) fewer drive bytes, even if decoding were free.
- On this machine every CPU rANS decoder measured costs more time than the drive read it saves, before any context
  model.
- GPU decoding (122–139 GB/s) makes the static coder's ~1.36× on the drive term reachable where the GPU consumes the
  experts. A better model adds nothing material to that.

**E. Does this justify changing Weightsift from an inference runtime project into a dedicated MoE compression research
project?**

No. **NO-GO.** Keep the existing runtime roadmap.
