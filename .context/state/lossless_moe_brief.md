# Weightsift — Deep Research & Experimental Feasibility Study: Novel Lossless Compression for MoE Weights

*The user's brief of 2026-10-15, pasted in the session and kept here verbatim, except two path references rewritten
so that link checks pass: Colibrì's format document as a link at its pinned commit, and the report's file-name template
without code formatting. Report: `history/2026-10-15-weightsift-lossless-moe-compression-feasibility.md`.*

You are acting as a senior research engineer specializing in information theory, lossless compression, Mixture-of-Experts models, low-level systems programming, and memory-constrained LLM inference.

Your task is to **conduct a rigorous research investigation, including real experimental work**, to determine whether a new lossless compression algorithm specifically designed for MoE weights could significantly outperform existing rANS-based approaches.

This is not a request to implement another inference engine.

**The goal is to determine whether a genuinely valuable MoE-specific lossless codec is technically achievable.**

Do not assume the answer is yes. A convincing negative result is preferable to an unsupported optimistic conclusion.

## 1. Project context

Repository: `https://github.com/suffro/weightsift`

Development branch: `weightsift-and-AWPMI-implementation`

Starting reference: Phase 7B, commit `06b7fde`.

Before starting, inspect:

- `.context/index.md`
- `.context/state/current.md`
- `.context/truth/architecture.md`
- `.context/state/Weightsift_Implementation_Guide.md`
- `.context/history/2026-10-15-weightsift-phase7b-review.md`
- The Phase 5C and 6B reports, experiments, codec implementations, and decisions
- Existing Moonlight expert tensors, indexes, routing traces, and compression benchmarks, where available

Also inspect COLIBRI:

`https://github.com/JustVugg/colibri`

Particularly:

- [docs/int4-rans256-g0.md](https://github.com/JustVugg/colibri/blob/bf2442915d6e3dd4cdfd2eb9c2a3d2aa44a25850/docs/int4-rans256-g0.md)
- `c/rans.h`
- `c/tools/repack_rans.py`
- `c/tools/rans_verify.py`
- Relevant expert storage, quantized container, and decoding code

Start from the pinned COLIBRI commit `bf244291`, and separately inspect relevant newer upstream changes.

Reuse existing tools and data. Do not repeat previous experiments without a specific reason.

This research must be conducted on an isolated research branch. Do not change the established Weightsift architecture, replace its runtime roadmap, or merge experimental code into production.

## 2. Absolute scope: lossless compression only

This is critical.

We want **strictly lossless compression of existing weight representations**.

If the input consists of BF16 weights, decoding must reconstruct their exact original bytes.

If the input consists of already-quantized int4 or MXFP4 weights, decoding must reconstruct that exact stored representation, including all necessary scales and metadata.

Allowed:

- Entropy coding
- Reversible transforms
- Predictive coding
- Context modeling
- Conditional probability models
- Reversible structural transformations
- Shared dictionaries, if their costs are fully accounted for
- Novel codecs or combinations of the above

Not allowed:

- Additional quantization
- Dropping precision
- Approximate reconstruction
- Pruning
- Changing routing
- Retraining
- Accepting token-level agreement as a substitute for byte-level equality
- Comparing codecs on different input representations

**Every candidate must reconstruct the original bytes exactly, independently verified against the source checkpoint.**

The objective is compression of information, not accepting information loss.

## 3. The fundamental research question

COLIBRI reports approximately 0.76 compression ratio with rANS on a GLM-5.2 per-row int4 container.

Its measured marginal entropy is approximately 2.91 bits per weight.

These measurements are not universal constants for all MoE models.

We need to determine:

**Is there additional, exploitable statistical structure inside modern MoE weight tensors that a conventional rANS implementation with a mostly static symbol model does not capture?**

Investigate:

1. Dependencies between adjacent weights.
2. Correlations along rows and columns.
3. Correlations between gate, up, and down projections.
4. Statistical differences among layers and experts.
5. Conditional distributions depending on local context.
6. Reversible transformations that improve predictability.
7. Repeated patterns, shared dictionaries, and cross-expert redundancy.
8. Structure introduced by existing quantization formats.
9. Whether local patterns remain useful across different MoE architectures.
10. Whether the information required to exploit these patterns costs more bytes than it saves.

Importantly, Phase 5C already found poor exact structural reuse between Moonlight experts. Treat that as negative prior evidence; do not assume shared expert bases are promising.

## 4. Research the existing state of the art

Conduct a technical literature and implementation review covering:

- rANS, tANS/FSE, arithmetic coding, context-adaptive entropy coding
- Zstandard, LZMA, Brotli, LZ4 and other appropriate lossless baselines
- Bitshuffle, byte shuffle, predictive residuals and reversible integer transforms
- ZipNN, DFloat11, EntroLLM, and other neural-weight compression methods
- Compression of low-bit packed tensors
- GPU-friendly and SIMD-friendly lossless decoding
- Independently decompressible blocks and random-access compressed storage
- Other relevant research discovered during investigation

For each relevant approach, identify:

- Input format
- Whether it is actually lossless
- Achieved compression ratio
- Model and dataset
- Decoder performance
- Random-access compatibility
- Licensing and implementation availability
- Whether its results are transferable to modern MoE expert tensors

Distinguish **a new entropy coder** from **a new statistical model using an existing entropy coder**.

A new MoE-specific predictor combined with rANS is acceptable if the complete codec significantly improves the outcome.

Prioritize primary papers, repositories and reproducible experiments over marketing claims.

## 5. Experimental Stage A — Corpus and baseline

First, inventory what is already available locally.

Preferred corpus:

- Moonlight-16B-A3B BF16 experts: local research and regression baseline.
- Existing genuinely quantized MoE expert tensors: int4 and MXFP4 where available.
- COLIBRI's GLM-5.2 int4 format: reference comparison.
- Inkling and MiMo expert samples: only after explicit download approval.

Do not re-quantize BF16 tensors to generate the principal int4 results. Synthetic fixtures may be used for correctness tests, but cannot establish real-model compression performance.

**Do not download a full flagship checkpoint.**

If no representative int4/MXFP4 tensors are locally available, complete the local investigation first and request approval for a small, clearly identified range-request sample. The Phase 7B report describes a potential approximately 1.8 GB Inkling sampling experiment.

Run comparable baseline codecs on the **same exact bytes**:

- Original uncompressed representation
- COLIBRI's actual rANS codec, when compatible
- A well-tuned rANS reference with appropriate modeling choices
- zstd at practical and high-compression settings
- LZMA or other strong offline comparison
- Existing Weightsift lossless codecs
- Other relevant competitive baselines

Report separately for BF16, int4 and MXFP4.

Record all codec versions, configurations, input hashes, sample selection rules, hardware and timings.

Include format headers, codebooks, sidecars, alignment and indexes in the storage accounting.

## 6. Experimental Stage B — Measure the remaining entropy

This is the most important stage.

Before designing a large new compressor, establish how much information may actually remain compressible.

Measure:

- Marginal entropy, H(X)
- Conditional entropy estimates, H(X | context)
- Row-conditional and column-conditional predictability
- Layer-specific and projection-specific symbol distributions
- Entropy after reversible transformations
- Compression gains from trained versus static probability tables
- Amortized costs of dictionaries and prediction metadata
- Model generalization to held-out experts and layers

Compare simple contexts before exploring expensive models.

For int4 data, investigate whether conditional modeling can beat the approximately 2.91 bits per weight measured on GLM-5.2.

If the actual baseline produces approximately 3.04 bits per weight, achieving 25% fewer bytes would require approximately 2.28 bits per weight, including overhead.

That is a substantial reduction.

**Determine whether measurements support anything close to that target.**

An empirical conditional-entropy estimate is not a universal lower bound. Explain its assumptions, the model's coding overhead, and its reliability on held-out data.

Use training/calibration and evaluation partitions that do not overlap.

### Early feasibility gate

If realistic contextual models reveal only marginal additional compressibility, stop rather than inventing a complicated codec that cannot meet the target.

Explicitly report whether a large improvement appears:

- Plausible
- Uncertain but worth a limited prototype
- Unsupported by the evidence

## 7. Experimental Stage C — Candidate algorithms

Only if Stage B demonstrates meaningful headroom, investigate multiple genuinely different candidate designs.

Suggested families:

**Candidate A — MoE-aware conditional entropy model**

Use context-dependent symbol probabilities based on reversible local structure, projection type and tensor statistics.

**Candidate B — Reversible predictive transformation**

Predict weights or packed symbols from nearby information, encode a reversible residual, then entropy-code that residual.

**Candidate C — Adaptive hierarchical coding**

Partition weights into independently accessible regions and choose local statistical models without excessive overhead.

**Candidate D — Structured block coding**

Explore block-specific distributions, reversible permutations, bit layouts, and lightweight shared dictionaries.

**Candidate E — Original research direction**

Propose at least one additional mechanism motivated by the measured structure, rather than merely rebranding existing codecs.

For each candidate explain:

- Exact encoding and decoding mechanism
- What statistical redundancy it exploits
- Why conventional rANS does not already exploit it
- Required metadata
- Decoder complexity
- Memory requirements
- Random-access cost
- Expected compression advantage
- Failure conditions

Do not implement all five automatically.

Select the most promising candidates based on measured evidence. Prefer a small number of rigorous prototypes over many superficial ones.

Python is acceptable for statistical analysis. Use existing native implementations where possible; Rust is preferred for any new reusable systems implementation.

## 8. Experimental Stage D — Real benchmarks

Benchmark promising implementations against the strongest applicable rANS baseline.

Primary metrics:

1. Total compressed bytes, including metadata.
2. Effective bits per original weight.
3. Compression ratio.
4. Percentage improvement over rANS.
5. Sustained decompression throughput, GB/s of reconstructed output.
6. Decoder CPU utilization and peak RAM.
7. Cold random-access latency for a single expert.
8. Bytes physically read from disk per expert.
9. Compression throughput and offline preparation cost.
10. Whether blocks can be decompressed independently.

Test various expert, projection, layer and block sizes.

Measure performance on the actual available hardware, not merely theoretical throughput.

Where possible, reuse COLIBRI's loader or an instrumented equivalent for realistic randomly routed expert access.

A standalone codec microbenchmark is not evidence of end-to-end token/s improvements.

If no full flagship is available, provide conservative system-level projections with clearly stated assumptions.

Never claim an integrated COLIBRI speedup that was not measured.

### Correctness

Every result requires:

- Exact reconstruction checked against independently read source bytes
- SHA-256 validation
- Edge-case and corruption tests
- Deterministic, repeatable measurements
- No hidden reliance on the original tensor or unavailable decoder state

All models, tables and decoder dependencies must be counted.

## 9. Success criteria

The desired result must be **substantial**, not a marginal improvement.

Use these research targets:

**Strong success:**

- At least **25% fewer total compressed bytes than the strongest practical rANS baseline**, on held-out real MoE int4 data.
- Exact reconstruction.
- Decoder throughput compatible with the SSD streaming workload.
- Efficient per-expert random access.
- Reasonable decoder memory consumption.
- Evidence that the result is not confined to one cherry-picked expert or layer.

**Exceptional success:**

- At least 40% fewer bytes than rANS under those same constraints.

**Partial success:**

- A reproducible 10–25% improvement with acceptable decoder costs. Record it as scientifically interesting, but insufficient to justify the intended project pivot without further evidence.

**Negative result:**

- Less than 10% improvement, or gains erased by decoder overhead, metadata, poor generalization or random-access amplification.

These percentages refer to the **compressed size relative to rANS**, not the original uncompressed checkpoint.

If the data show strong storage savings but unacceptable decoder costs, report the tradeoff honestly.

Do not relax these thresholds after observing results.

## 10. Feasibility for COLIBRI and DS4

Determine whether the best successful codec could realistically improve existing MoE inference engines.

Evaluate:

- Where the decoder would fit in COLIBRI's expert-loading path
- Compatibility with SSD-to-RAM and SSD-to-GPU streaming
- CPU decoding on AVX2 hardware
- Potential GPU decoder feasibility
- Effects on cache capacity
- Effects on RAM and VRAM minimum requirements
- Effects on cold and warm inference
- Whether it could become an independent Rust library with a stable C ABI

Do not integrate it into COLIBRI or DS4 during this study unless a tiny, isolated benchmarking adapter is strictly required.

Do not confuse reducing checkpoint disk size with reducing the minimum RAM required to run the model.

Assess realistically whether the codec could improve throughput by 25%, 50%, 100%, or more on a disk-bound target. Distinguish calculations from measured results.

## 11. Execution discipline

This study must not trigger another uncontrolled change of direction.

- Use a dedicated research branch.
- Keep production Weightsift inference architecture unchanged.
- Reuse previous Phase 5C/6B implementations.
- Run genuine experiments; do not stop at a literature survey.
- Build the smallest viable experimental harness first.
- Use reproducible configuration files and deterministic sampling.
- Preserve raw outputs and benchmark commands.
- Apply early stop gates.
- Do not download large models without explicit approval.
- Do not install large dependencies, make significant hardware changes, or use paid compute without approval.
- Do not introduce new model quantization or Quality Budget experiments.
- Do not claim novelty without checking the literature.
- Do not manufacture performance results or theoretical bounds.
- Use `[Verified]`, `[Inference]`, `[Hypothesis]`, `[Speculation]`, and `[Unknown]` consistently.

If a sample is unavailable, distinguish what was tested locally from what still needs flagship validation.

## 12. Deliverables

Place the final research report in:

.context/history/YYYY-MM-DD-weightsift-lossless-moe-compression-feasibility.md

Keep reproducible experimental code in a dedicated location such as:

`research/lossless_moe/`

Store measured artifacts under:

`experiments/lossless_moe/`

Follow the repository's existing conventions and avoid committing large model files or bulky generated data.

The report must include:

1. Executive conclusion.
2. Exact datasets and checkpoints inspected.
3. Review of existing lossless weight compression.
4. Original rANS and alternative baseline measurements.
5. Entropy and conditional-entropy analysis.
6. Candidate compression mechanisms.
7. Experimental implementation details.
8. Compression and decompression benchmarks.
9. Exact reconstruction verification.
10. Generalization across experts, layers and formats.
11. Practical implications for COLIBRI and DS4.
12. Originality assessment.
13. Limitations and negative results.
14. Final go/no-go recommendation.
15. Reproducibility instructions and raw-data references.

Include comparison tables and plots where they improve the analysis.

Update `.context/state/current.md` and `.context/index.md` only to reflect verified completed work. Do not overwrite the existing project direction or create a new architectural decision without approval.

Commit the research code and documentation on the dedicated branch. Do not merge it into the development branch.

## 13. Final decision required

End with explicit answers to these questions.

**A. Is a substantially better lossless MoE codec technically plausible, based on the measured data?**

**B. How much extra compressible information was actually found beyond conventional rANS modeling?**

**C. What is the best experimentally demonstrated improvement over rANS, including metadata and decoder cost?**

**D. How much would that improvement realistically help COLIBRI or DS4?**

**E. Does this justify changing Weightsift from an inference runtime project into a dedicated MoE compression research project?**

Choose exactly one recommendation:

- **GO:** compelling experimental evidence of a substantial, practically usable advantage.
- **CONTINUE LIMITED RESEARCH:** meaningful evidence, but important limitations remain.
- **NO-GO:** the data do not support a sufficiently large advantage.

If the answer is NO-GO, say so explicitly and recommend retaining Weightsift's existing runtime roadmap.

---

**Core principle: We are not trying to prove that a new algorithm can beat rANS. We are trying to discover whether the data contain enough exploitable information to make that possible.**

Perform the investigation, run the experiments, and let the results determine the conclusion.
