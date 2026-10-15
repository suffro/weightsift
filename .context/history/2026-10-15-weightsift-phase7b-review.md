# Weightsift Phase 7B — an independent technical review against Colibrì

Date: 2026-10-15 · Follows: `history/2026-10-12-weightsift-phase7a-report.md` (Phase 7A) · Brief: `state/phase_7B.md` (the
user's of 2026-10-14) ·
Status: **review complete. Nothing was built, run or measured in this phase.** The brief forbids writing code or
modifying repositories during the assessment, so this report and the `.context` updates that carry it are the only
changes. Every number below comes from Weightsift's committed runs, from Colibrì's repository at a stated commit, or
from a cited source, and says which.

## How to read this report

Labels, as the brief defines them:

- **[Verified]**: read in source code, in a committed measurement, or in a published result consulted for this review.
- **[Inference]**: reasoned from identified evidence.
- **[Hypothesis]**: technically plausible, untested.
- **[Speculation]**: exploratory, little evidence.
- **[Unknown]**: not enough information.

A number taken from Colibrì's documentation is Colibrì's measurement on Colibrì's machines. It is [Verified] as a
published claim, never as a reproduction.

**What was inspected.**

- **Weightsift** at `54c31eb` on `weightsift-and-AWPMI-implementation`, the latest commit:
  - `.context`: state, the implementation guide, decisions 0001–0014, the 13 phase reports;
  - `native/`: Rust core, store, runtime and the Colibrì bridge, 11.5k tracked lines of Rust and C;
  - `src/awpmi` (13.1k lines of Python), `benchmarks` (12.1k), tracked `research` (9.4k), tests (6.7k);
  - `experiments/`: routing traces, cache replays, profiles.
- **Colibrì** (Apache-2.0), cloned read-only into a scratch directory:
  - `main` at `bf244291` (2026-10-06, release v2.0.0, Weightsift's pin), and `dev` at `2fc033dc` (2026-10-15, 97
    commits later);
  - its source: about 157k lines of C under `c/`, plus CUDA, Vulkan and Metal sources, 511 test files and 158 tools;
  - its 57 documents, its decode failure ledger and its paper test matrix;
  - its release assets, through the GitHub API. Nothing of Colibrì was built or run.
- **External sources:** about 40 papers and project pages, plus Hugging Face model cards (Inkling, its NVFP4 release,
  Colibrì's Inkling container). They were read through arXiv abstracts and project pages; full texts were not read
  unless stated.

**Limits.**

- No measurement was possible: no flagship checkpoint is local, and the brief forbids running code here.
- The 2026 papers on speculative decoding for MoE (MoE-Spec, DraftExpert, *The Limits of Speculation*) were read only
  through their abstracts.
- This report recommends; it records no decision. The choice of direction is the user's.

## 1. Executive assessment

**[Inference] The verdict.**

- **What each project is.** Weightsift is a rigorous research project with a small, well-tested storage core and no
  flagship inference path. Colibrì is a mature, fast-moving engine: 13 engines including both flagships, CPU, CUDA,
  Vulkan and Metal backends, and Windows releases. Its own controlled experiments have already closed many of the
  directions Weightsift might try next.
- **The bound on the target machine.** On 32 GB of RAM, an 8 GB GPU and one NVMe drive at 3.2 GB/s, Inkling's decode is
  bounded by the bytes read from the drive per token. That is about 0.3–0.4 tokens/s for the published int4 experts,
  whatever the engine: no engine reads 9–10 GB faster than the drive.
- **The only large lever.** Only moving fewer bytes changes that floor. Lossless coding is worth at most about 1.36×;
  beyond that, the representation itself must change.
- **The representation is already derived.** On this machine it has changed twice before any engine starts:
  - Inkling's experts run from a derived per-row int4 container whose quality nobody has measured;
  - its dense weights must be converted to int4 to fit in RAM at all.
- **Weightsift's defensible advantage** is therefore neither exactness nor certification. It is **measured quality per
  byte**: choosing, validating and placing representations under a statistically sound Quality Budget, plugged into
  Colibrì's engines through the storage seam Phase 7A built.

| Area | Weightsift today | Prospect |
| --- | --- | --- |
| Flagship inference | None: no Colibrì engine has ever run in Weightsift [Verified] | Only through Colibrì's engines (decision 0014) |
| Exact streaming of a 16B MoE | Moonlight-16B-A3B at 408.9 ms per decode token, bit-exact to an independent reference [Verified, Phase 6B] | Does not transfer to flagships: a PyTorch path, on a model Colibrì does not run [Inference] |
| Storage core | Rust store with leases and a byte budget across layers; 3.02 GB/s on a 3.2 GB/s drive; C ABI [Verified, Phase 7A] | Useful if Colibrì's Inkling loader, not the drive, is on the critical path [Hypothesis] |
| Certified materialization | Negative inside experts (5A, 5A2, 5C); positive but unprofitable in the LM head (1C, 2) [Verified] | No role in flagship decode speed on this machine [Inference, § 2.9 and § 5] |
| Quality evaluation | Specified, not run; dataset revisions unpinned [Verified] | The main differentiator, if made feasible for flagships [Inference, § 4.4] |
| Model support | No native engine | Colibrì's |

**Prospects.**

- **Exact mode.** No evidence supports Weightsift outperforming Colibrì on the same representation, with the same
  output, by more than storage orchestration and lossless coding allow: together about 1.7× (§ 5.6). Colibrì could
  adopt both quickly.
- **Elsewhere.** The evidence does support a credible path to a better latency–quality frontier on small machines, and
  to running the flagships on less RAM. Both need quality measurements that exist nowhere today, in either project.

## 2. Verified implementation state

### 2.1 What runs

- **Moonlight-16B-A3B, out of VRAM and host RAM, bit-exact.** The model has DeepSeek-V3's architecture and is checked
  against an independent transformers reference [Verified]:
  - Phase 4B: 1,273 ms per decode token, streaming;
  - Phase 6A: 807 ms, on the native core;
  - Phase 6B: 408.9 ms (2.45 tokens/s). All 1,116 steps of 10 configurations equal the reference in every digest, in two
    runs with different hash seeds.
- **OLMoE out of VRAM (Phase 4A)** [Verified].
- **A certified LM head on SmolLM2-135M (Phase 1C)** [Verified]. It reads 0.40 of the head's bytes and certifies 90.4%
  of tokens, but takes 24.5 ms per token against 0.21 ms for the reference GEMV.
- **The Rust expert store on Moonlight's experts (Phase 7A)** [Verified]. It reaches 3.02 GB/s with batched leases while
  the drive is busy at 3.1 GB/s, and all 512 rows equal the reference.
- **The Colibrì bridge** (the lease-contract shim and the segment runtime) is tested on Colibrì's fixtures only
  [Verified].

### 2.2 What has been validated experimentally

| Claim | Evidence | Scope |
| --- | --- | --- |
| Exact out-of-core MoE decode, including host and device caches, GPU-decoded rANS rows (0.6747 of BF16), CUDA Graphs | Phases 4B, 6A, 6B: digests, I/O audits against OS counters, two seeds [Verified] | One model (Moonlight), one machine |
| An LRU cache serves nothing below one token's working set (2.70 GB of BF16 rows on Moonlight) | 6B replay of Phase 4B and 5A routing [Verified] | Moonlight's routing; 48 prompts |
| A hotness policy below that cliff serves 0.04–0.38 of decode bytes from 0.5 to 3 GB | `experiments/phase6b/gpucache/replay.txt` [Verified] | Same |
| Experts share no exact structure | 5C1: 0 of 1,536 deltas cheaper than the expert alone [Verified]. Colibrì's GLM-5.2: 0.38–0.69% shared energy, near the random 1/256 [Verified, its ledger] | Two model families agree |
| Certification inside routed experts does not pay | 5A: 0 tokens certified with any routed byte unread. 5A2: CROWN 15× looser than box sets. 5C: exact bit-plane sets need 0.82–1.00 of the bytes [Verified] | Moonlight's last MoE layer |
| Storage reads at the drive's rate | 3.19 GB/s (Phase 4B), 3.38–3.52 GB/s (6A), 3.02–3.1 GB/s (7A store) on a drive limited to about 3.37 GB/s by its PCIe 3.0 x4 link [Verified] | This machine |

### 2.3 Research prototypes, not on any inference path

- The AWPMI oracles (5A, 5A2, 5C), the CROWN environment and the expert-delta codecs.
- The Quality Budget (`awpmi.quality`): specification, results schema, comparison statistics. No dataset is pinned and
  nothing has been evaluated [Verified].
- The Colibrì baseline specification (`configs/phase7a-baseline.yaml`): not run.
- The memory planner: arithmetic only.
- ApexQuant: BLOCKED (no license).

### 2.4 Dependence on third-party code

| Dependency | Used for | Note |
| --- | --- | --- |
| PyTorch 2.14, transformers 5.18 | The reference and the Moonlight runtime | `DecodeGraphs` refuses any other source of DeepSeek-V3's modules (sha256 of each method) [Verified, 6B § 9] |
| nvCOMP 5.3 (proprietary, CUDA only) | GPU decoding of encoded expert rows | It does not detect corruption; integrity rests on Weightsift's hashes [Verified, 6B] |
| auto_LiRPA 0.7.2 | The 5A2 study, in a separate environment | Research only |
| Colibrì | Nine pinned files (`expert_store.h`, the segment runtime, its fixtures) | No engine used [Verified, `native/upstream.json`] |
| zstd, lz4, SciPy | Phase 5C research | Research only |

### 2.5 Assumptions that remain unproven

- **That the storage path is on Colibrì's critical path** for a flagship on a small machine [Unknown]. Phase 7A states
  that it showed no speed advantage over Colibrì's loader [Verified].
- **The flagships' routing locality** [Unknown]. Colibrì's Inkling engine initializes routing telemetry but emits no
  route records [Verified, Colibrì [docs/experiments/cnre-offline-simulator.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/experiments/cnre-offline-simulator.md)]. Nobody has an Inkling routing trace.
- **The quality of any flagship representation usable on 32 GB** [Unknown]. Colibrì's Inkling container states "no
  quality evals yet beyond greedy smoke tests" [Verified, its model card].
- **That the planner's sizing yields a useful hit rate.** Its own output at detected memory is a 3.22 GB cache, 0.30 of
  one token's working set [Verified, `experiments/phase7a/plan-inkling`]. By Moonlight's replay, an LRU cache of that
  size serves nothing [Inference].
- **The brief's statement that Phase 7A "integrated existing COLIBRI engines".** It holds only in part: 7A integrates
  Colibrì's lease contract and segment runtime against fixtures, and no engine [Verified].

### 2.6 Failures: fundamental or implementation-specific

| Result | Classification |
| --- | --- |
| 5A's rounding floor: 8.1% of tokens certify even with every routed byte read; the tightest pair's error terms average 3.2 logits | **Specific to the reference profile** [Inference]. It comes from modelling BF16 kernels with undocumented accumulation orders as faithful rounding. Round-to-nearest-even what-ifs raise coverage to 17.8–24.7%, real arithmetic to 100% [Verified, 5A]. An integer-exact or fully specified reference would shrink it (§ 2.9) |
| Bounds on unread weights (5A, 5A2, 5C) | **Fundamental for norm-type metadata** [Verified]. The sets that 5A's metadata defines contain weights that flip the token until 0.82–1.62 of the routed bytes are read, metadata included (5A2's witnesses). 5C's exact bit-plane sets contain them until 0.80–0.99 of the representation, and certifying needed 0.82–1.00. This limits the information, not the verifier: no sound verifier can certify what the set does not exclude |
| Shared expert bases (5C1) | **Fundamental for post-training experts** [Verified]: Weightsift on Moonlight and Colibrì on GLM-5.2 agree. Only retraining changes it |
| 5C's bit-plane format on the GPU | **Implementation and hardware-specific** [Verified, 6B]: zstd decoding on this GPU costs more than half the copy it saves. rANS over raw rows replaced it |
| Phase 2: suffix certification not beneficial (−0.0002 of the region) | The same two obstacles one layer earlier [Verified] |
| Phase 1C: the head's bytes saved but 117× slower than the GEMV | **Implementation-specific** (Python, many small operations) [Inference]. Even at zero cost it would not matter for flagships: the head is about 1% of a token's time on this machine (§ 5) |

### 2.7 The bottlenecks that dominate

- **Moonlight, measured.** A warm decode token of 409 ms splits into:
  - copies to the GPU, 198 ms, overlapping most of the rest;
  - waits for the host cache's copies and the drive, 124 ms;
  - assembly, 55 ms;
  - an eager remainder of about 172 ms;
  - meanwhile the GPU is idle about 35% of the step [Verified, 6B § 11].
- **Flagships on the target machine, modelled (§ 5).**
  - Disk bytes per token: about 80% of a serialized token.
  - Next, RAM bandwidth for the dense set read every token.
  - The CPU's expert arithmetic is a minor term at these hit rates [Inference].

### 2.8 What differentiates Weightsift from Colibrì today

- **Verification depth** [Verified].
  - Weightsift: bit-exact digests of every recorded intermediate against an independent reference; I/O audited against
    OS counters; two-seed reproducibility; gates registered before results.
  - Colibrì: token-exactness against Hugging Face on tiny fixtures for every engine; MiMo also on the real checkpoint (32
    of 70 layers); in-process ABBA timing; a failure ledger.
  - Both are rigorous, in different ways. Colibrì's performance methodology is as careful as Weightsift's; Weightsift's
    numerical verification is deeper.
- **A statistically designed quality comparison** [Verified, not run]: paired bootstrap, Newcombe intervals, verdicts
  that refuse invented scores. Colibrì's `coli bench` runs 40 questions per task [Verified, its [docs/benchmarks.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/benchmarks.md)].
  At 40 items the 95% interval on an accuracy near 0.6 is about ±15 points [Inference: Wilson interval].
- **One byte budget across all layers, with leases** [Verified], against Colibrì's slots per layer (`cap`).
- **Compressed expert streaming decoded on the GPU, measured end to end** (Moonlight: 0.776× the decode time)
  [Verified]. Colibrì has an ANS-compressed VRAM tier for GLM under CUDA: +13.9% capacity, 6.19 → 7.12 tokens/s
  [Verified, its paper test matrix]. This is parity, not uniqueness.
- **What certification taught, and Colibrì has no counterpart for** [Inference]: certification does not pay inside
  experts, and why.

### 2.9 What Phase 5's negative results imply

The brief asks whether Phase 5 shows a fundamental limitation of certifying next-token decisions from partially
materialized expert weights, or only rules out particular formulations. Separated:

1. **A lower bound relative to the information.** Take a set of weights consistent with what has been read: each row
   within a radius of its read approximation, in the norms the metadata records.
   - The minimum of the logit difference Δ·y over that set is attained at a vertex of the activation box (5A2's
     argument, `crown_oracle/rowsets.py`).
   - Phase 5A2 evaluated that minimum with witnesses: real weights inside the set, run through the real forward. It
     stays below zero until 0.82–1.62 of the routed bytes are read, metadata included [Verified, 5A2]. For 5C's exact
     bit-plane sets it stays below zero until 0.80–0.99 of the representation [Verified, 5C].
   - So no certificate, however sophisticated, can certify those tokens from that metadata. **This is information, not
     verifier looseness.**
2. **A scaling argument for any compact metadata** [Inference].
   - An unread block contributes Δᵀ·W_u·a, where neither the activations a nor the difference direction Δ is known in
     advance; both change with every token and every contender.
   - A norm-type set lets W_u align adversarially with any direction, so its worst case is the product of norms.
   - The typical value for incoherent weights is about √n smaller: about 45 for n = 2,048 (Phase 5A's estimate).
   - Closing that gap needs directional information about W_u for every direction a token might need. That is
     information comparable in size to W_u itself.
   - 5C's sketches through shared bases moved a certificate by at most one checkpoint, at a 1% sketch [Verified].
3. **Limits of the existing certificates.**
   - The mature verifier did worse, not better: auto_LiRPA's CROWN with L2 sets was 34–35× looser than Phase 5A's own
     bound at the q6 state [Verified, 5A2].
   - Even a perfect verifier on the L2 sets could close at most 12–16% of the distance between 5A's realistic bound and
     its ideal one, and at most 36–38% with every resident byte [Verified, 5A2].
4. **Rounding artifacts.** The certified-tier floor is a property of the reference profile, not of the model (§ 2.6).
5. **Representation.** Bit planes are 3× closer to the truth than 5A's levels at the same bytes and are still
   insufficient [Verified, 5C].
6. **Engineering.** Nothing here was decisive.
7. **Unexplored, in order of plausibility** [Inference]:
   - **Certifying selections among many candidates with a wide spread.** The LM head (200k rows) and the router's top-k
     (256 experts) are the favorable geometry: most candidates are far from the threshold, so coarse information
     eliminates them. Phase 1C's LM head shows it works. But both are small and resident in a flagship: Inkling's head
     is about 1.2 GB of int8, read once per token from RAM. They save no disk bytes.
   - **An integer-exact reference.** Kernels that accumulate integer products and round once compute their dot products
     exactly. Colibrì has them: its integer dot kernels (`idot.h`), and its `COLI_EXACT_VERIFY` attention core, which
     "accumulate[s] integer products and round[s] once" [Verified, its [docs/tuning.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/tuning.md)]. Such a reference removes the
     rounding floor. It does not close the information gap.
   - **Exactness by verification instead of certification.** Speculative decoding drafts cheaply and verifies with the
     exact model, so the output is exact by construction, with no bound. It is the practical replacement for AWPMI's
     "stop early if proven" (H5, H7).

**Answer** [Inference]:

- The negative results do not prove that every conceivable metadata fails.
- They do prove that every metadata Weightsift measured fails, and the scaling argument says compact metadata will not
  do.
- Certifying expert contributions is therefore not a credible speed path.
- **What would justify reopening:** a metadata of at most 10% of an expert's bytes whose attained set minimum stays
  above zero on at least half of the tokens, in real arithmetic, on 5C2's existing samples. That is a one-day oracle
  (E10, § 9). Only then is a reference with exact arithmetic worth building for it.

## 3. Colibrì: competitive analysis

### 3.1 What Colibrì actually implements

Read in its code and documentation at `bf244291` and `2fc033dc`. "Measured" means measured by Colibrì's maintainers or
contributors, on their machines.

| Area | Implemented | Status |
| --- | --- | --- |
| Engines | GLM-5.2/5.3, GLM-5.3-Flash, Inkling, Kimi K3 (2.8T), DeepSeek V4 and V4.1 Flash, MiMo-V2.6 Flash and Pro, Qwen3.8-Flash-Next, Qwen3.6, OLMoE; plus Qwen-Image and three decision models [Verified, `c/*.c`, README] | Each token-exact against Hugging Face on a tiny fixture in CI [Verified] |
| RAM tier | Per-layer LRU slots (`cap`) plus a pinned hot set learned from `.coli_usage`, updated every turn; live LFRU re-pinning (`--repin`) [Verified, `c/inkling.c` 1550–1672, [docs/tuning.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/tuning.md)] | Measured on many machines |
| VRAM tiers | CUDA (experts and dense; Inkling's needs about 37 GB of VRAM for its BF16 residents), Vulkan expert tier and dense chain on any Vulkan 1.2 GPU, Metal, HIP [Verified, docs] | Inkling's Vulkan tier is "correct, not yet measured for speed" [Verified, [docs/inkling.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/inkling.md)]. Discrete-GPU Vulkan numbers come only from users [Verified, README] |
| I/O | One `pread` per expert (GLM: three matrices in one); an async load pool overlapping reads with arithmetic (`PIPE`, on by default in GLM); io_uring on Linux; `O_DIRECT` (+65% on one Strix Halo); copies on a second drive (+37.5% with two NVMe drives); Windows readahead through the thread pool and file mappings [Verified, [docs/tuning.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/tuning.md), [docs/multidisk.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/multidisk.md), [docs/windows.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/windows.md)] | Measured |
| I/O in Inkling's engine specifically | Pairs (token, expert) in token-major order, in rounds of at most `cap`: acquire, fill the round's misses in parallel (OpenMP), compute, accumulate. No cross-layer prefetch; on the CPU path no overlap of a layer's reads with its arithmetic. The code notes: "decode fills run at queue depth ~1, so every pinned expert removes a ~35ms stall" [Verified, `c/inkling.c` 1640–1642, 2346–2474] | Measured (0.25 tokens/s steady state, below) |
| Lossless compression | `int4-rans256-g0`: 0.76 of per-row int4 bytes, 0.89 of g64 [Verified, [docs/int4-rans256-g0.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/int4-rans256-g0.md)]. GLM-5.2's int4 codes carry 2.91 bits of entropy per weight (72.9%) [Verified, [docs/experiments/glm52-split-entropy-2026-07-31.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/experiments/glm52-split-entropy-2026-07-31.md)]. An ANS-compressed VRAM tier (DietGPU, CUDA, GLM) | rANS: format and offline tools only, no engine decode merged. ANS tier: measured (+13.9% capacity, 6.19 → 7.12 tokens/s). Fused ANS decode with the matvec: rejected (−18.8%) [Verified, ledger] |
| Prediction and routing levers | `PILOT`, next-layer readahead: 71.6% recall of GLM's top-8 against 41.3% for "same as last token". `CACHE_ROUTE`, cache-aware routing (lossy, opt-in). `TOPP`, adaptive top-p (lossy): 30–40% fewer reads, "a clean 1.6×" on a 24 GB machine. `DEGRADE_ZERO` (lossy; OLMoE: +2.9% perplexity at 22% of slots zeroed) [Verified, docs] | Prefetch rejected for resident decode (−2.5%); learned coupling rejected [Verified, ledger]. The lossy levers' quality is mostly unmeasured |
| Speculation | GLM's int8 MTP head (2.2–2.8 tokens per forward); Qwen3.8's MTP (+20% at 2 drafts); prompt lookup (+6–7% on code edits); a cost-aware gate (`spec_draft.h`); exact verify rows (each row computed as a decode step) [Verified, [docs/speculative.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/speculative.md)] | MTP as a default rejected on GLM: rejected drafts widen the expert union (−15.6% at 1 draft, −55.6% at 3, on a resident host) [Verified, ledger]. **Inkling: MTP not loaded; MiMo: MTP and DFlash not implemented** [Verified, docs] |
| KV and serving | Prefix reuse (second-request TTFT 17.7 → 0.57 s), persistent KV across restarts, up to 16 KV slots, OpenAI- and Anthropic-compatible API [Verified] | Measured |
| Measurement | A five-rule benchmark protocol, a datapoint tool, in-process ABBA, a decode failure ledger, a paper-claim test matrix [Verified] | Exemplary |
| Quality | `coli bench`: MMLU, HellaSwag and ARC at 40 questions each. The int4 container: 62.5% mean acc_norm. OLMoE fp16 → per-row int4: −8.2 points. GLM's per-row int4 mirrors "about 9 points worse" than grouped scales [Verified, docs]. **Inkling's container: "no quality evals yet beyond greedy smoke tests"** [Verified, its model card] | Weak for flagships |
| Platforms | Release v2.0.0, 2026-10-06 [Verified, GitHub API]. `windows-x86_64.zip` holds every engine, `inkling.exe` included, built with Vulkan [Verified, `.github/workflows/release.yml`]. A CUDA DLL ships for the GLM, Qwen3.6 and Kimi K3 engines only | **Weightsift's "gcc blocker" (7A) applies to patching Colibrì's engines, not to running its baseline** [Inference] |

**Colibrì's flagship numbers** [Verified as its claims]:

- **Inkling 975B** on a Ryzen 9 7900 with 187 GB of DDR5, an RTX A6000 and NVMe:
  - 0.25 tokens/s in steady state, at an 82.2% hit rate, from a diverse 11-prompt history;
  - 2.51 tokens/s on the trained prompt with deep pins;
  - "~90% CPU expert matmul" at high hit rates.
  - On a 25 GB machine: "tens of seconds per token on a single NVMe". That is stated, without a table.
- **MiMo-V2.6 Pro** on 8 cores with 64 GB and NVMe RAID at 10 GB/s: 0.66–0.79 tokens/s at 32–39% residency. It needs
  54 GB of RAM to start, 64 GB to run as measured.

### 3.2 Dimension by dimension

The verdicts are those the brief asks for: confirmed advantage or disadvantage (for Weightsift), plausible advantage
needing validation, or insufficient evidence.

| Dimension | Weightsift | Colibrì | Verdict for Weightsift |
| --- | --- | --- | --- |
| Inference speed (decode, TTFT, prefill) | No flagship. Moonlight 2.45 tokens/s, prefill 3.6 s on average over Phase 4B's prompts (a model Colibrì does not run) | Both flagships run (above) | **Confirmed disadvantage** |
| Hardware accessibility | No flagship runs | Inkling on 25 GB with int4-gs64 dense weights; MiMo Pro needs ≥ 54 GB | Confirmed disadvantage. Plausible advantage: MiMo Pro on 32 GB through a lossy dense representation [Hypothesis] |
| Memory efficiency | One byte budget across layers, leases, a block pool inside the budget [Verified] | Fixed `cap` per layer; its simulator explores per-layer budgets [Verified, cnre doc] | Plausible advantage (non-uniform budgets across layers) [Hypothesis] |
| Storage efficiency | Physical bytes equal the rows' 4 KiB blocks: amplification 1.0005, audited [Verified] | One read per expert; no amplification figures for Inkling | Insufficient evidence |
| Compression | rANS over BF16 rows, GPU-decoded, end to end (Moonlight) [Verified] | rANS for int4 (offline); ANS VRAM tier (GLM, CUDA) [Verified] | Parity; insufficient evidence for flagships |
| Data movement | Compressed H2D, device decode, device tier, all measured [Verified] | Vulkan tier sends experts as cached; zero-copy GPU reads of host experts measured and rejected (2.8 GB/s effective) [Verified, ledger] | Insufficient evidence for flagships |
| Caching | LRU, hotness and fixed slots on Moonlight; replays [Verified] | Learned pins, LFRU, frequency fill, cross-validated placement, held-out tests [Verified] | **Confirmed disadvantage** (policy maturity) |
| Scheduling | Exact prefetch, deadlock-free leases [Verified] | Async pools, readahead, gates, CPU–GPU overlap (+7.1% before NUMA, +2.0% after) [Verified] | Disadvantage overall. Plausible advantage at Inkling's seam, whose engine has none of this [Hypothesis] |
| Kernels | PyTorch and cuBLAS only | Owns AVX2, AVX-512 and VNNI integer kernels and CUDA, Vulkan, Metal, HIP and XDNA paths [Verified] | **Confirmed disadvantage** |
| Precision | Exact against a BF16 reference, Moonlight [Verified] | "Placement only decides speed", exact at fixed kernels on its containers; containers derived and partly unevaluated [Verified] | Plausible advantage in evaluating quality [Hypothesis] |
| Scalability | 16B | Up to 2.8T (Kimi K3: 9.4 s per token, experts at 6.3 GB/s) [Verified] | **Confirmed disadvantage** |
| Model support | One adapter (DeepSeek-V3 through transformers) | 13 engines, one C file per family over shared headers | **Confirmed disadvantage** |
| Portability | Windows with CUDA tested | Linux, macOS and Windows; CPU, CUDA, Vulkan, Metal, HIP [Verified] | **Confirmed disadvantage** |
| Research differentiation | Exactness, certification (negative), statistical quality design | Placement learning, speculation gates, failure ledger, community datapoints | Different; neither dominates |

### 3.3 Where Colibrì is ahead, and why

- **Colibrì is ahead almost everywhere that produces tokens** [Inference]:
  - it owns the engines and the kernels;
  - it measures on many machines (3,735 commits on `main`; a community of contributors with their own hardware);
  - it tests and rejects ideas quickly: its decode failure ledger closes 19 directions with controlled evidence.
- **Its design rule is the right exact-mode contract:** "placement only ever decides speed: the router's decisions and
  the weights' precision are the same whether an expert answered from VRAM, from RAM or from disk".
- **Weightsift should not try to beat it at building engines.**

**Where Colibrì is weaker** [Verified facts; the consequences are Inference]:

1. **The quality of its small-machine representations is unmeasured.**
   - Inkling runs from a per-row int4 container with no evaluation.
   - The same recipe measured about 9 points worse than grouped scales on GLM.
   - Its own benchmark runs 40 items per task.
   - On a 32 GB machine every Inkling run also uses an int4 dense set, with 11% relative L2 error in attention, shared
     experts and the dense MLP.
2. **Its controlled rejections were measured mostly on one full-resident 6× RTX 5090 host,** with 251 GiB of RAM and
   zero disk traffic in decode.
   - Rejecting prefetch, impact-aware placement and default speculation holds where the disk is out of the loop.
   - It does not transfer automatically to 2% residency, where misses dominate [Inference].
   - The ledger says so itself: "Reopen only on a genuine disk-miss workload".
3. **Inkling's engine has the simplest I/O path of its engines:** no async pool, no readahead, no io_uring,
   token-major rounds, no route records.

### 3.4 What a fair comparison requires

Equal or controlled conditions:

| Condition | Requirement |
| --- | --- |
| Hardware and OS | The same machine and OS for both systems |
| Representation | The same container and dense set (Inkling: the int4 container plus int4-gs64 dense, or the official NVFP4) |
| Kernels | The same arithmetic: Colibrì's |
| Workload | The same prompts and decoding: greedy, fixed tokens replayed |
| Memory | The same budgets in bytes |
| Cache state | The same cold or warm state: page cache dropped; `.coli_usage` fixed, or `PIN=off` |
| Build | The same build: release v2.0.0 or a pinned commit |
| Correctness | Identical output tokens, which separates storage from computation |
| Accounting | Drive bytes from the OS's counters on both sides |
| Timing | ABBA order in one session where possible |

- Separately for quality comparisons: the same benchmark items, with intervals.
- **Invalid:** any comparison of Weightsift's Moonlight numbers with Colibrì's flagship numbers (different models,
  frameworks, precisions and machines).

## 4. Fundamental limitations

### 4.1 Hard constraints of the target machine

| Constraint | Value | Consequence |
| --- | --- | --- |
| Drive | Samsung 990 PRO at about 3.2–3.4 GB/s with direct I/O, limited by its PCIe 3.0 x4 link [Verified, Phases 3–7A] | Inkling's int4 experts, 10.89 GB per token if none is cached [Verified, 7A], take ≥ 3.4 s per token cold |
| RAM | 31.9 GiB [Verified] | Inkling's dense set as released (BF16) is 49.4 GB and does not fit; its int4-gs64 conversion is 15.3 GB and does [Verified as Colibrì's figures]. MiMo Pro's dense weights are 30.2 GiB as released and 21.7 GiB in int8: neither leaves room [Verified, Colibrì [docs/mimo.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/mimo.md)] |
| No exact flagship path | Follows from the row above | **Neither flagship can run on this machine exactly against its release** [Inference]. "Exact" there can only mean exact against a derived representation chosen by someone. That representation's quality is the question that matters |
| Long prompts | Prefill reads every touched expert at least once | A cold prompt of a few hundred tokens touches nearly all 16,384 experts: 464.6 GB, about 145 s at 3.2 GB/s for the int4 container (§ 5.4) [Inference] |
| VRAM | 8 GB | Holds at most about 1.6% of Inkling's expert bank (about 265 int4 experts in 7.5 GB), or half of its int4 dense set [Inference] |
| PCIe to the GPU | 3.0 x8, 6.1–6.2 GB/s pinned [Verified, 6B] | Faster than the drive: not binding for streaming |
| CPU | i7-8700K, 6 cores, AVX2, no AVX-512 or VNNI [Verified, 5C setup] | Colibrì reports AVX2-only machines kernel-bound before disk-bound on DeepSeek V4 Flash (#1119) [Verified]. This machine's balance must be measured [Unknown] |
| RAM bandwidth | Not measured. Dual-channel DDR4 at 2,666 MT/s is 42.7 GB/s in theory [Inference] | The dense set is read every token: about 0.4 s per token at 35 GB/s (§ 5) |

### 4.2 Failed hypotheses, in either project, and what would reopen them

| Direction | Where it failed | Reopen only if |
| --- | --- | --- |
| Certified partial materialization inside experts | Weightsift 5A, 5A2, 5C | E10's metadata test passes (§ 2.9) |
| Exact shared bases, deltas, neuron alignment | Weightsift 5C1; Colibrì GLM-5.2 (common basis, permutation alignment, SERE substitution) | A retrained or distilled derivative model |
| Bit-plane formats decoded on the GPU | Weightsift 6B | A GPU decode costing less than the copy it saves |
| Fused compressed decode with the matvec | Colibrì C3 (−18.8% integrated) | A decode-native, vector-aligned encoding whose fused kernel first wins in isolation |
| Cross-layer prefetch, learned coupling | Colibrì M3, on a mostly resident host | **Reopen for the 2%-residency regime**: optimize recall of experts that will miss, not of routing (H8) |
| Impact-aware placement without hard residency | Colibrì M2 (−46.2%) | Capacity is constrained and miss costs are in the objective (H4) |
| Speculation as a default | Colibrì S1, ledger | Union growth is controlled and bytes per token are low (H5, § 5.5) |
| Native copy issuing | Weightsift 6B3-B | The host, not the copy engine, paces the transfers |
| ApexQuant | Weightsift 7A (no license; the 4-bit class does not pay) | A license and a 2-bit quality result on MoE |

### 4.3 Unresolved problems

- The quality of every representation that fits the target machine, original Inkling included, by any statistically
  meaningful measure [Unknown].
- The flagships' routing locality and hit-rate curves at 1–3% residency [Unknown]. Colibrì's Inkling engine emits no
  traces.
- How Colibrì's Inkling engine actually spends a token on a 32 GB machine [Unknown]. Its only profile is from a 187 GB
  host.
- This machine's RAM bandwidth, the second drive's rate and link (`E:`), and the PCIe topology (why the GPU links at x8)
  [Unknown].
- The CPU rANS decoding rate on AVX2: Colibrì's fast decode arms are NEON and AVX-512 [Verified, `c/rans.h` per its
  doc], so it matters whether CPU decoding keeps up with the drive [Unknown].

### 4.4 Quality evaluation on the target machine is itself a hard problem

[Inference]

- **Generation is too slow.** At 0.3–1 tokens/s a generation-based benchmark is infeasible here. GSM8K's 1,319 items at
  about 300 generated tokens each is 400k tokens: 4–15 days per configuration, and reasoning benchmarks need far more
  tokens.
- **Teacher-forced metrics are feasible only with GPU arithmetic.** Per-token KL divergence to a reference, top-1
  agreement, perplexity on fixed corpora:
  - computed in large chunks, they read each expert once per chunk;
  - a 4,096-token chunk on Inkling reads the whole bank, 464.6 GB at 3.2 GB/s, about 145 s;
  - it computes about 4,096 × 82 GFLOP = 336 TFLOP. That is over 30 minutes on this CPU at about 150 GFLOP/s, but about
    20 s on the RTX 4060 Ti at an assumed 20 TFLOP/s;
  - with GPU arithmetic, about 28 scored tokens/s; 100k tokens in about an hour per configuration.
- **The original cannot even be stored.** Inkling's BF16 checkpoint is 1.9 TB; `E:` has 1.37 TB free [Verified, 7A]. The
  practical reference is therefore the publisher's NVFP4 release, `thinkingmachines/Inkling-NVFP4`, about 592 GB
  [Verified as NVIDIA Dynamo's recipe states], plus the publisher's own benchmark scores.
- **Generation benchmarks need a proxy or a rented machine.** They must run either on a smaller model of the same family
  of architectures (Moonlight, locally), with the KL-to-benchmark relation validated there, or on a rented machine.

### 4.5 Exact mode and quality-budgeted mode

The brief's two regimes, as they apply here [Inference].

**Exact mode** preserves a declared reference's semantics: token-identical to Colibrì's engine on a fixed container,
whatever the placement.

- Its speedups come only from moving the same bits faster or fewer times:
  - lossless coding (≤ 1.36×);
  - overlap (≤ 1.25×);
  - a better cache;
  - more disk bandwidth;
  - speculation verified exactly (H5, H7).
- It is worth keeping as the correctness referee and the fallback.

**Quality-budgeted mode** permits measured deviations.

- Its levers are the only ones that move the floor by 2× or more on this machine:
  - lower-bit and mixed-precision experts (H1, H6);
  - fewer routed experts per token (top-p, H10);
  - pruned experts (H11);
  - quantized dense weights.
- Measurement covers six categories: reasoning (MATH-500, GPQA), code (HumanEval, MBPP), long context (LongBench v2),
  instruction following (IFEval), tool use (BFCL), and multilingual prompts. Multilingual is **missing from today's
  Quality Budget**; MGSM or Global-MMLU-Lite would fill it.
- Each category needs paired comparisons on common items with intervals, which `awpmi.quality` already implements, plus
  the teacher-forced KL of § 4.4 as the screening metric for flagships.
- A quantized run is never described as bit-exact to an unquantized reference (guide § 19).

## 5. Performance bottleneck model

### 5.1 The model

One decode token of a disk-streamed MoE, per layer: route, then fetch the misses and compute every routed expert, plus
the dense work.

```text
T_disk  = E · b · q · e · p · (1 − h) / B_disk        bytes read from the drive
T_ram   = (D + E · b · q · p) / B_ram                  dense set + experts read from RAM by the CPU kernels
T_pcie  = (experts sent to the GPU) / B_pcie           only if the GPU computes them
T_fixed = launches, synchronizations, routing (per layer × layers)

serialized (no overlap):        T ≈ T_disk + T_ram + T_fixed
perfectly overlapped (bound):   T ≥ max(T_disk, T_ram, T_pcie, T_gpu) + T_fixed_critical
```

| Symbol | Meaning |
| --- | --- |
| E | Routed experts per token |
| b | Bytes of one expert in the reference container |
| q | Relative size of a lower-bit representation (1 for int4) |
| e | Lossless coding ratio (1 if raw) |
| p | Fraction of routed experts kept (1 without top-p) |
| h | Host-cache hit rate |
| D | Dense bytes read per token |

What is additive and what overlaps:

- **Additive.** Within one layer, a miss's read and that expert's arithmetic are additive unless something prefetches.
- **Overlappable, without prediction.** Reads of one layer's misses with arithmetic on its resident experts.
- **Overlappable only with prediction.** Reads of the next layer's experts during this layer's dense work.
- **Pipelinable.** GPU transfers and decoding, behind disk reads.

### 5.2 Parameters for Inkling on the target machine

| Parameter | Value | Source |
| --- | --- | --- |
| E | 384 (64 MoE layers × top-6) | [Verified, config and headers] |
| b | 28.36 MB (int4 container), so E·b = 10.89 GB | [Verified, 7A] |
| Expert bank | 16,384 experts, 464.6 GB | [Verified, arithmetic] |
| D | about 14 GB: the 15.3 GB int4-gs64 dense set less the embedding table, read by rows | [Inference from Colibrì's figure] |
| B_disk | 3.2 GB/s | [Verified] |
| B_ram | 35 GB/s | [Unknown: assumed; measure it (E1)] |
| B_pcie | 6.1 GB/s | [Verified] |
| Host cache | about 9–10 GB: 26–28 GB usable, less the dense set and state. That is about 330 experts, 2.0% of the bank, 0.86 of one token's working set | [Inference] |
| h | 0.05–0.25, central value 0.15 | [Inference]: Moonlight's replays give 0 for LRU below one token's working set and 0.04–0.14 for hotness at 1.7–3.5% of the bank. Colibrì's 82–84% hit rates on Inkling came with about 76 experts cached per layer, about 30% of the bank (from the pin comment in `c/inkling.c`) |

### 5.3 Scenarios: a decode token, warm, h = 0.15 unless stated

T_ram assumes the CPU computes every routed expert from RAM. "Bound" is the perfectly overlapped lower bound. A
"serialized" value is given where it differs.

| Scenario | Lever | q · e · p | h | T_disk (s) | T_ram (s) | Bound (tokens/s) |
| --- | --- | --- | --- | --- | --- | --- |
| S0 exact int4, as Colibrì runs it | — | 1 | 0.15 | 2.89 | 0.71 | 0.35 (serialized 3.60 s: 0.28) |
| S0 range | h = 0.05 to 0.25 | 1 | — | 3.23–2.55 | 0.71 | 0.31–0.39 |
| S1 exact + rANS coding, compressed cache | e = 0.76 (GLM's per-row ratio; Inkling's unmeasured), capacity × 1.3 | 0.76 | 0.18 | 2.12 | 0.71 + decode | 0.47 |
| S2 3-bit experts (quality-budgeted) | q = 0.75, e = 0.85 | 0.64 | 0.20 | 1.74 | 0.63 | 0.58 |
| S3 usage-weighted 2.5 bits on average (H1) | q = 0.625, e = 0.9 | 0.56 | 0.22 | 1.49 | 0.60 | 0.67 |
| S4 = S3 + top-p 0.7 (H10) | p = 0.7 (Colibrì: 30–40% fewer reads) | 0.39 | 0.22 | 1.05 | 0.54 | 0.96 |
| S5 = S4 + a second NVMe on CPU lanes | B_disk = 6.4 | 0.39 | 0.22 | 0.52 | 0.54 | 1.86 (now RAM-bound) |
| S6 = S5 + 7 GB of the dense set on the GPU | D_ram = 7 GB | 0.39 | 0.22+ | 0.52 | 0.34 | 1.9 (disk-bound again) |

Every number is a model output, not a projection of measured speed. The e, q and p values for Inkling are hypotheses
(H1, H2, H10), and h is unmeasured.

### 5.4 Prefill and TTFT

Each layer reads the distinct experts its prompt touches. With uniform, independent routing (the worst case) a prompt of
S tokens touches u(S) = 256 · (1 − (1 − 6/256)^S) experts per layer:

| Prompt | u(S) per layer | Bytes, int4, cold | Read time at 3.2 GB/s |
| --- | --- | --- | --- |
| 13 tokens | 68 | 123 GB | 38 s |
| 100 tokens | 232 | 421 GB | 132 s |
| ≥ 300 tokens | about 256 | 464.6 GB | 145 s |

- Skewed routing lowers these numbers.
- For comparison, Colibrì measured 35.4 s for a 13-token prompt on a 187 GB host with an 82% hit rate [Verified].
- **A cold system prompt of a few thousand tokens costs minutes of TTFT on this machine whatever the engine.** Only three
  things remedy it [Inference]:
  - persistent KV and prefix reuse, which Colibrì has;
  - fewer bytes;
  - more disk bandwidth.
- **Inkling's CPU path can re-read experts within one prefill** [Inference, from the code's structure; the magnitude is
  unmeasured]. It processes pairs in token-major rounds of `cap`. With a small `cap` (about 5 per layer on 32 GB), an
  expert that recurs across distant tokens is evicted between its uses and read again: up to S · 6 reads per layer
  instead of u(S).
- **An expert-major schedule bounds the reads at u(S).** Read each distinct expert once and apply it to every token
  routed to it. To stay token-exact it must keep each row's routing-order sum. Colibrì's MiMo engine adopted that
  row-order rule for its prefix reuse [Verified, [docs/mimo.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/mimo.md)]. Scratch: S · 6 · 6,144 floats, 147 MB at S = 1,000.

### 5.5 Where speculation can and cannot pay

Speculative decoding with k drafts and E(k) accepted tokens per forward, per accepted token:

```text
T_spec ≈ [ D/B_ram + U(k+1)·b·q·e·(1−h)/B_disk + U(k+1)·b·q/B_ram ] / E(k)
```

U(k+1) is the union of the k+1 rows' routed experts. The rejected rows' experts are read all the same.

- **In S0, speculation pays little** [Inference]. The dense term is 0.40 s of a serialized 3.60 s token (11%).
  - If speculation reads the same expert bytes per accepted token as plain decoding, the dense term caps the gain near
    1.12×.
  - It gains more only if consecutive tokens share experts that the cache would not otherwise keep.
  - With E(2) = 2.2 and a union of 80% of 3 × 384, a token costs 3.68 s: no gain.
  - At a union of 60%, 2.80 s: 1.29×. That needs locality among consecutive tokens which an LRU of 0.86 working sets
    would already partly capture.
- **In S4, the dense term is 25% of a serialized token**, so the same cap is about 1.33×. It rises further in S5 and S6.
- **Consequence:** speculation's value rises as bytes per token fall. It belongs after the representation work, not
  before. This also explains the conditions Colibrì's ledger sets for reopening speculation.

### 5.6 Amdahl: how much an engine alone can gain

[Inference]

- **The ceiling for exact int4.** In S0, serialized, the drive is 80% of a token and RAM 20%.
  - Perfect overlap and prefetch: from 3.60 to 2.89 s, at most 1.25×.
  - Lossless coding alone: at most 1.36×.
  - Both together: about 1.7×, from 3.60 to 2.12 s.
  - Every exact engine-side improvement, Weightsift's or Colibrì's, lives inside that 1.7× for the int4 container on
    this machine. More caching, through more RAM, is the exception.
- **Beyond it, three things.**
  - Lossy bytes: S2–S4, another 2–3×.
  - Hardware: a second drive on independent lanes, or more RAM.
  - Speculation, once bytes are low.
- **After S4 the dense set's RAM traffic dominates.** About 0.4 s per token, it becomes the next bottleneck. That moves
  the GPU's best use from caching experts to holding dense weights.
  - A dense byte is read on every token.
  - A cached expert byte is read in only a fraction of tokens: about 6/256 per expert per layer on average, more for the
    hottest experts.
  - So VRAM given to dense weights saves more bytes per byte than VRAM given to experts, until the dense set is resident
    (H4).
- **Warm and cold.**
  - Cold means h = 0 and the full prefill cost of § 5.4.
  - Warm means the steady-state h.
  - Sequential versus random: Inkling's expert is a few contiguous extents of about 9–19 MB [Inference from the
    container's tensor layout], so reads are sequential-class.
  - 4 KiB-granular partial reads, as 5A's spatial strategies needed, cost 14.4 µs each with 8 threads on this drive
    [Verified, Phase 3] and should be avoided.

## 6. Research landscape

Results are each paper's own, on its own hardware and models, as their abstracts or project pages report them
[Verified as claims]. "Colibrì" says whether Colibrì implements the mechanism, from its code and docs.

### 6.1 Out-of-core and offloaded inference

| Work | Date | Mechanism | Reported result | Relevance | Colibrì |
| --- | --- | --- | --- | --- | --- |
| [FlexGen](https://arxiv.org/abs/2303.06865) | Mar 2023 (ICML) | LP-searched placement over GPU, CPU and disk; 4-bit weights and KV | OPT-175B on one 16 GB GPU at 1 token/s, effective batch 144 | Throughput mode; batch is what amortizes weights | No (single-stream focus) |
| [DeepSpeed ZeRO-Inference](https://www.deepspeed.ai/2022/09/09/zero-inference.html) | Sep 2022 | Weights in DRAM or NVMe, streamed per layer | Massive models on one GPU, throughput-oriented | Baseline idea | — |
| [LLM in a flash](https://arxiv.org/abs/2312.11514) | Dec 2023 | Flash cost model; reuse of activated neurons ("windowing"); row–column bundling | Models up to 2× DRAM; 4–5× (CPU) and 20–25× (GPU) over naive loading | Bundling is Weightsift 5A's neuron-major layout; windowing is cache reuse | Experts read whole |
| [PowerInfer](https://arxiv.org/abs/2312.12456) | Dec 2023 | Hot neurons on GPU, cold on CPU; activation predictors | Up to 11.69× over llama.cpp (RTX 4090) | ReLU-era sparsity; for MoE the unit is the expert | Expert tiers |
| [SpecExec](https://arxiv.org/abs/2406.02532) | Jun 2024 (NeurIPS) | Large draft trees verified in one offloaded forward | 50B+ on consumer GPUs: 4–6 tokens/s at 4 bits, 2–3 at 16 bits | Dense models: a forward costs the same for 1 or 1,000 tokens. MoE breaks this through the expert union (§ 5.5) | Speculation gated, MTP and lookup |
| [llama.cpp `--n-cpu-moe`, `-ot`](https://dev.to/someoddcodeguy/understanding-moe-offloading-5co6) | 2025 | Experts of chosen layers kept in RAM and computed on the CPU; tensor placement by regular expression | Community reports only (not used here) | The common practice | Equivalent tiers |

### 6.2 MoE offloading, caching and prediction

| Work | Date | Mechanism | Reported result | Relevance | Colibrì |
| --- | --- | --- | --- | --- | --- |
| [Towards MoE Deployment](https://arxiv.org/abs/2303.06182) | Mar 2023 | Expert buffering; temporal locality; load imbalance | GPU static memory 1.47× lower | Locality exists in older MoEs | LRU and pins |
| [Pre-gated MoE](https://arxiv.org/abs/2308.12066) | Aug 2023 (ISCA 2024) | Layer N's gate picks layer N+1's experts (retrained) | Single-GPU deployment at quality | Prediction by changing the model | No |
| [Eliseev & Mazur](https://arxiv.org/abs/2312.17238) | Dec 2023 | LRU expert cache, speculative expert loading, mixed quantization | Mixtral-8x7B on desktop and free Colab | Same pattern, small scale | Yes (tiers, pins) |
| [MoE-Infinity](https://arxiv.org/abs/2401.14361) | Jan 2024 | Sequence-level activation tracing for prefetch and caching | 4–20× lower latency | Request-level locality | `.coli_usage`, repin |
| [ExFlow (inter-layer affinity)](https://arxiv.org/abs/2401.08383) | Jan 2024 | Strong inter-layer expert affinity in pretrained MoEs | Up to 67% less cross-GPU routing latency | Supports cross-layer prediction | `PILOT`, coupling (rejected on a resident host) |
| [Fiddler](https://arxiv.org/abs/2402.07033) | Feb 2024 (ICLR 2025) | Compute experts on the CPU instead of moving them | Mixtral-8x7B > 3 tokens/s on 24 GB; 8.2–10.1× | Right when PCIe is slower than RAM-side arithmetic; here the drive dominates | CPU computes misses |
| [AdapMoE](https://arxiv.org/abs/2408.10284) | Aug 2024 | Sensitivity-based adaptive number of experts, prefetch | −25% activated experts, 1.35× "without accuracy degradation" | Lossy routing reduction, measured | `TOPP` (quality unmeasured) |
| [ProMoE](https://arxiv.org/abs/2410.22134) | Oct 2024 | Predict later experts from intermediate results | Integrated in transformers and llama.cpp | Miss prediction | `PILOT` |
| [HOBBIT](https://arxiv.org/abs/2411.01433) | Nov 2024 | Low-precision versions for less critical cache misses; layer prefetch; multidimensional cache | Up to 9.93× decode over prior offloading (edge devices, llama.cpp) | Closest prior art to H1 and H6 | No |
| [MoE-Lightning](https://arxiv.org/abs/2411.11217) | Nov 2024 | CPU–GPU–I/O pipelining; Hierarchical Roofline Model | Up to 10.3× throughput, Mixtral on a T4 | A performance-model template (§ 5) | Partly (async groups) |
| [Toward Efficient Inference for MoE](https://arxiv.org/abs/2412.00099) | Dec 2024 (NeurIPS) | Includes max-rank cache-aware routing | — | Colibrì's `CACHE_ROUTE` cites it | Yes (opt-in, lossy) |
| [KTransformers](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/) | SOSP 2025 | AMX CPU kernels, async CPU–GPU scheduling, Expert Deferral | Prefill 4.62–19.74×, decode 1.25–4.09×; Deferral ≤ 0.5% accuracy drop | Hybrid CPU/GPU at large RAM | Similar split, AVX-512/VNNI not AMX |

### 6.3 Compression and quantization

| Work | Date | Mechanism | Reported result | Relevance |
| --- | --- | --- | --- | --- |
| [QMoE](https://arxiv.org/abs/2310.16795) | Oct 2023 (MLSys 2024) | Sub-1-bit custom format with GPU decoding | SwitchTransformer-c2048 (1.6T) at 0.8 bits/parameter, minor accuracy loss | An older, top-1, heavily overparameterized MoE; no evidence it transfers to fine-grained modern MoEs [Inference] |
| [MoQE](https://arxiv.org/abs/2310.02410) | Oct 2023 | 2-bit expert weights only | Experts far more robust to quantization than dense FFNs | Supports quantizing experts below dense layers |
| [Any-Precision LLM](https://arxiv.org/abs/2402.10517) | Feb 2024 (ICML) | Bit-plane storage: n-bit model serves 3…n bits by MSBs | Multiple precisions in one n-bit footprint | Nested storage for H6 |
| [QTIP](https://arxiv.org/abs/2406.11235) | Jun 2024 (NeurIPS) | Trellis-coded quantization after Hadamard incoherence | 2-bit Llama-2-70B within 0.7 perplexity of FP16 | State of the art at 2 bits (dense); its decode is compute-heavier than int4 |
| [MC-MoE](https://arxiv.org/abs/2410.06270) | Oct 2024 (ICLR 2025) | LP bit allocation by expert importance; dynamic pruning | 2.54 bits, −3.8% average on eight commonsense benchmarks; −15% activated parameters at < 0.6% | Prior art for H1; commonsense tasks only |
| [ZipNN](https://arxiv.org/abs/2411.05239) | Nov 2024 | Exponent-separated Huffman coding | Often 33% saved, sometimes > 50% | Lossless for floating point; int4 codes behave differently |
| [Matryoshka Quantization](https://arxiv.org/abs/2502.06786) | Feb 2025 (ICML) | Train once, nested int8/int4/int2 | int2 up to 4–7% better than standard int2 | Nesting needs training for good int2 |
| [DFloat11](https://arxiv.org/abs/2504.11651) | Apr 2025 | Entropy-coded BF16 exponents, GPU decoding | About 70% of the size, bit-identical; Llama-3.1-405B on 8×80 GB | Same idea as Weightsift 6B's rANS rows (0.6747) |
| [REAP](https://arxiv.org/abs/2510.13999) | Oct 2025 (ICLR 2026) | Prune experts by router-weighted activation norms | Near-lossless code generation at 50% pruning on Qwen3-Coder-480B and Kimi-K2 | H11; Colibrì ships a REAP DeepSeek V4 Flash |
| [Inkling NVFP4](https://docs.nvidia.com/dynamo/recipes/inkling) | 2026 | The publisher's quantized release | About 592 GB | The strongest available released-quantized baseline (Quality Budget order) |

### 6.4 Speculation

| Work | Date | Mechanism | Reported result | Relevance |
| --- | --- | --- | --- | --- |
| [QSpec](https://arxiv.org/abs/2410.11305) | Oct 2024 | Draft with low-precision weight-and-activation quantization, verify with high-precision weight-only quantization, sharing weights and KV | Up to 1.80× with no quality loss | Self-speculation by precision |
| [EAGLE-3](https://arxiv.org/abs/2503.01840) | Mar 2025 (NeurIPS) | Feature-fusion draft head, training-time test | Up to 6.5× | Needs training per target |
| [MoE-Spec](https://arxiv.org/abs/2602.16052) | Feb 2026 | Fixed expert budget per layer during verification | 10–30% over EAGLE-3 "at comparable quality" | Drops experts: lossy verification |
| [DraftExpert](https://www.alphaxiv.org/abs/2607.24434v1) | Jul 2026 | Resident draft expert per layer; confidence–expansion truncation; target prefetch | 1.45× average on DeepSeek-V2-Lite and **Moonlight-16B-A3B**; plain speculative decoding slowed down offloaded MoE | Directly relevant; on a model Weightsift has locally |
| [The Limits of Speculation](https://arxiv.org/abs/2609.22156) | Aug 2026 | Offline oracle of speculative decisions for MoE | A linear boundary between marginal cost and expected progress | Supports a cost-aware gate (Colibrì's `spec_draft.h`) |

### 6.5 Numerics, verification and activation sparsity

| Work | Date | Relevance |
| --- | --- | --- |
| [auto_LiRPA / CROWN](https://arxiv.org/abs/2002.12920) | Feb 2020 (NeurIPS) | Weightsift 5A2 used it; bound propagation cannot exceed its input sets' information (§ 2.9) |
| [Higham & Mary, probabilistic rounding error analysis](https://eprints.maths.manchester.ac.uk/2731/) | 2019 (SIAM J. Sci. Comput.) | Typical rounding errors grow far slower than worst-case γ_n bounds. That explains 5A's floor, but probabilistic bounds are not certificates |
| [Horace He (Thinking Machines), "Defeating Nondeterminism in LLM Inference"](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) | 2025-09-10 | Batch-invariant kernels make inference deterministic. Needed for exact speculation across row counts (Weightsift 6B § 12; Colibrì's per-row verify) |
| [TEAL](https://arxiv.org/abs/2408.14690) | Aug 2024 (ICLR 2025) | 40–50% magnitude-based activation sparsity, 1.53–1.8× on dense models. Inside a routed expert the gate and up projections must be computed before the activations are known, so skipping saves at most the down projection's third of the bytes times the sparsity: about 15–17% at 50% [Inference]. Not pursued |

### 6.6 I/O interfaces

| Work | Relevance |
| --- | --- |
| [Windows I/O rings (`ioringapi.h`)](https://learn.microsoft.com/en-us/windows/win32/api/ioringapi/) | Batched asynchronous reads on Windows 11. Weightsift's store uses a thread pool of positioned reads, which already reaches the drive's rate here [Verified, 7A]; I/O rings would save threads, not bandwidth |
| [DirectStorage GPU decompression (GDeflate)](https://devblogs.microsoft.com/directx/directstorage-1-1-coming-soon/) | GPU-side decompression on Windows, vendor-neutral. An alternative to nvCOMP for a non-CUDA path (H2, H3) |

## 7. Novel hypotheses

Twelve hypotheses, with the brief's twelve fields: the name is the heading, then the eleven numbered fields. "Novelty"
is stated honestly in field 3.

### H1. Residency-aware, mass-weighted precision for streamed experts

1. **Core idea.** Allocate bits per expert, and per layer, by routing mass × measured output sensitivity per byte of disk
   traffic. Resident experts stay at the reference precision; experts that stream from disk go to 2–3 bits where their
   expected contribution is small.
2. **Mechanism.** Disk bytes come from misses. An expert's expected quality impact scales with its routed weight times
   its output error.
   - With skewed routing, many rarely used experts carry little mass per byte read, so lowering their bits saves bytes
     at small expected error.
   - Layers also differ in sensitivity (MC-MoE, MxMoE).
3. **Why existing systems don't, and novelty.**
   - HOBBIT switches precision dynamically on misses, from separate low-precision copies.
   - MC-MoE allocates bits by importance with LP, for memory, without residency or I/O in the objective.
   - Nobody ties the allocation to a measured residency and miss distribution on a disk-bound machine.
   - Novelty: modest, a combination [Inference].
4. **Existing research.** MoQE, MC-MoE, HOBBIT, QTIP, REAP's importance score.
5. **Difference from Weightsift's experiments.** Phase 5 used precision levels to certify. This uses them to approximate,
   under a statistical Quality Budget. 7A's ApexQuant screening covered one quantizer's uniform 4-bit class.
6. **Assumptions.**
   - Routing mass is skewed enough among non-resident experts.
   - Sensitivity can be measured from calibration activations.
   - The Quality Budget can resolve regressions of 1–2 points (§ 4.4).
7. **Failure modes.**
   - At 2% residency, about 85% of accesses miss, so most of the mass sits in streamed experts. If that mass is spread
     evenly, weighting reduces to "quantize lower", which is known art.
   - 2-bit post-training quantization of fine-grained experts may fall outside any useful budget.
8. **Expected benefit.** Bytes per token 0.56–0.64 of int4 (§ 5.3 S2, S3): 1.6–1.9× decode in the disk-bound regime. The
   quality cost is [Unknown].
9. **Complexity.** Medium. Existing quantizers, an allocator and evaluation runs; engine support for per-expert formats
   (Colibrì already dispatches several container formats per tensor) [Inference].
10. **Smallest credible experiment.** E2 (skew from Moonlight's traces) and E7 (allocation at equal average bits on
    Moonlight, against its BF16 original), locally.
11. **Criterion.**
    - **Success:** at an average ≤ 2.75 bits per routed weight, the weighted allocation stays within the budget (each
      category's regression bound ≤ 2 points at 95%) and beats uniform allocation at equal bytes outside the intervals on
      ≥ 2 categories.
    - **Failure:** weighted and uniform are indistinguishable at equal bytes. The lever is then plain lower precision,
      which Colibrì can adopt directly.

### H2. Exact entropy-coded int4 experts on disk, in the host cache and across PCIe

1. **Core idea.** Store, cache and transfer Inkling's int4 codes rANS-coded, decoded at the consumer. Output is
   unchanged: the decode restores the exact bytes.
2. **Mechanism.** GLM-5.2's int4 codes carry 2.91 bits of entropy per 4 [Verified]. Coding saves about 24% of drive
   bytes and gives the host cache about 1.3× more experts.
3. **Why not yet, and novelty.**
   - Colibrì has the format and offline tools; engine decoding is a pending PR [Verified].
   - Its VRAM ANS tier exists for GLM under CUDA only.
   - Weightsift 6B did the same for BF16 rows (0.6747).
   - Novelty: low; the value is in being first and exact on the flagship path [Inference].
4. **Existing research.** DFloat11, ZipNN, nvCOMP, DietGPU, `int4-rans256-g0`.
5. **Difference from Weightsift's experiments.** int4 codes instead of BF16 patterns, and the consumer may be the CPU.
6. **Assumptions.**
   - Inkling's per-row codes are as skewed as GLM's per-row codes: 0.76, against 0.89 for grouped scales.
   - Decoding keeps up with the drive without stealing the CPU the expert arithmetic needs.
7. **Failure modes.** CPU rANS on AVX2 below about 4 GB/s of output. Colibrì's fast arms are NEON and AVX-512 [Verified,
   its doc]. A GPU consumer needs H3.
8. **Expected benefit.** At most 1.36× decode in S0 (§ 5.6); about 1.3× cache capacity.
9. **Complexity.** Medium: a decoder at the store or engine seam, a pack format, integrity checks. rANS detects
   corruption only by framing; 6B relied on hashes.
10. **Smallest credible experiment.** E4 (code entropy of a range-request sample, needs approval) and E5 (CPU decode rate
    on this CPU, no download).
11. **Criterion.**
    - **Success:** e ≤ 0.80 on the sample, and decoding ≥ 6 GB/s of output on ≤ 2 cores (or a GPU consumer).
    - **Failure:** e > 0.85, or the CPU decode rate is below 4 GB/s with no GPU consumer.

### H3. Misses to the GPU, hits on the CPU

1. **Core idea.** Experts that stream from the drive go straight to the GPU: pinned staging, still compressed, then
   decoded and computed there. The CPU computes RAM-resident experts and the dense set.
2. **Mechanism.**
   - PCIe at 6.1 GB/s exceeds the drive's 3.2 GB/s, so streamed experts never need the CPU's RAM bandwidth or
     arithmetic.
   - The GPU's 288 GB/s and idle compute absorb them.
   - The CPU's RAM bandwidth is spent only on dense weights and hits.
3. **Why not yet, and novelty.**
   - Colibrì's tiers send resident experts to the GPU and compute misses on the CPU (the Vulkan tier: "the CPU reads and
     computes the others meanwhile") [Verified].
   - Its rejected zero-copy experiment was the GPU reading host RAM at 2.8 GB/s, a different path.
   - Novelty: a placement inversion for the disk-bound regime [Hypothesis].
4. **Existing research.** Fiddler argues the opposite, rightly when PCIe is the bottleneck. MoE-Lightning, KTransformers.
5. **Difference from Weightsift's experiments.** Phase 6B did exactly this for Moonlight in PyTorch: encoded rows from the
   drive, decoded and computed on the GPU, 0.776× the time. New is doing it under Colibrì's engine, for a flagship, with
   int4.
6. **Assumptions.**
   - Int4 expert kernels on the GPU (Colibrì's Vulkan tier has them for the int4 container [Verified]).
   - A per-layer synchronization costing well under a millisecond.
   - The routing result available to the GPU path in time.
7. **Failure modes.**
   - In S0 the drive dominates, so the gain is at most T_ram's expert share, about 0.31 s per token.
   - Vulkan arithmetic differs from the CPU's (f32 activations against int8 rounding), so tokens may differ: perplexity
     equality, not token equality, is the check [Verified, Colibrì [docs/benchmarks.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/benchmarks.md)].
8. **Expected benefit.** About 10% in S0. 25–40% once bytes are reduced (S4–S6), where T_ram binds [Inference].
9. **Complexity.** Medium to high: a change to Colibrì's Inkling MoE path, kept as a pinned patch.
10. **Smallest credible experiment.** In Colibrì's Qwen3.6-35B-A3B engine, which already has a VRAM tier and a disk-bound
    setting at small caps (needs a 23 GB download): the GPU computing misses against the CPU computing them, at equal
    bytes.
11. **Criterion.**
    - **Success:** ≥ 15% decode at equal drive bytes and equal perplexity within its interval.
    - **Failure:** < 5%.

### H4. Placement by access frequency per byte, dense weights included

1. **Core idea.** One planner places dense and expert bytes across VRAM and RAM by expected bytes saved per byte placed.
   Dense weights are read on every token, so they are the hottest bytes there are.
2. **Mechanism.** A VRAM byte holding dense weights removes one RAM read per token. A VRAM byte holding a cached expert
   removes one read only with probability equal to that expert's use rate: about 6/256 per expert per layer, more for
   hot ones.
3. **Why not yet, and novelty.**
   - Colibrì places the whole dense set on the device when it fits (Inkling's CUDA path needs about 37 GB).
   - Partial dense residency on an 8 GB card is not documented for Inkling's engines [Unknown: whether the Vulkan dense
     chain can hold part of the stack].
   - Its placement solver handles experts only.
   - Novelty: modest.
4. **Existing research.** FlexGen's LP placement; MoE-Lightning's HRM; Colibrì's capacity-aware solver.
5. **Difference from Weightsift's experiments.** 7A's planner reserves budgets but optimizes nothing.
6. **Assumptions.** RAM bandwidth is the binding term after bytes are cut (S4–S6). Splitting a layer's work between
   devices costs little: a 6,144-float state per layer crossing PCIe.
7. **Failure modes.** Per-layer device round trips: Colibrì's ledger rejected GPU microkernels in a split layer for
   exactly this reason, "launches, synchronization, transfers ... dominated". Any gain is invisible while the drive
   dominates (S0).
8. **Expected benefit.** About 0.2 s per token where T_ram binds (S5 → S6), plus RAM freed for the expert cache.
9. **Complexity.** Medium (planner) to high (engine support for a partial dense set).
10. **Smallest credible experiment.** E1 (RAM bandwidth) plus a planner simulation on Moonlight's traces. Then an A/B on
    Inkling after E8.
11. **Criterion.**
    - **Success:** ≥ 10% decode at equal tokens in a configuration where the drive is not saturated.
    - **Failure:** < 3%, or the split costs more than it saves.

### H5. Speculation for the disk-bound regime: Inkling's own MTP layers and a union-aware gate

1. **Core idea.**
   - Draft with Inkling's MTP layers: its config declares `num_nextn_predict_layers: 8` [Verified, `config.json`], and
     Colibrì loads none [Verified].
   - Verify exactly in one forward.
   - Gate the depth on the predicted miss bytes of the expert union, not on acceptance.
2. **Mechanism.** It amortizes the dense set's RAM read and the per-forward fixed costs over accepted tokens (§ 5.5).
3. **Why not yet, and novelty.**
   - Colibrì's gate counts forward time, and its ledger shows acceptance is not enough.
   - DraftExpert (2026) addresses expert expansion on Moonlight with a resident draft expert.
   - Novelty: low to modest.
4. **Existing research.** SpecExec, EAGLE-3, QSpec, MoE-Spec, DraftExpert, *The Limits of Speculation*, Colibrì's
   `spec_draft.h`.
5. **Difference from Weightsift's experiments.** None tried.
6. **Assumptions.**
   - The MTP layers are usable and small; their own experts may also need streaming [Unknown].
   - Consecutive tokens share experts beyond what the cache already captures.
7. **Failure modes.**
   - Union growth: Colibrì measured GLM's loads per token rising from 590.6 to 739.1 at one draft, at 62% acceptance.
   - The MTP layers' expert reads.
   - § 5.5: at most about 1.12× in S0.
8. **Expected benefit.** About 1.0–1.1× in S0; up to 1.3× in S4 and more after.
9. **Complexity.** High: MTP support in Colibrì's Inkling engine, a verify path, rollback of local-attention rings.
10. **Smallest credible experiment.** E2's U(k) on Moonlight's existing traces, through § 5.5's model. Zero cost.
11. **Criterion.**
    - **Continue** only if U(3) / (3·E) ≤ 0.75 under the simulated cache.
    - **Stop** for the disk-bound regime otherwise.

### H6. Nested precision on misses (anytime experts)

1. **Core idea.** Store each streamed expert MSB-first in bit planes. A miss under deadline reads only the top planes
   (half the bytes for 2 of 4 bits) and computes; the cache refines it in the background for later hits. No duplicate
   storage.
2. **Mechanism.** It trades bytes for precision per miss, under the Quality Budget, with one artifact.
3. **Why not yet, and novelty.**
   - HOBBIT keeps separate low-precision copies.
   - Any-Precision LLM and MatQuant nest precisions, for dense models.
   - Novelty: modest, a combination applied to streamed experts [Inference].
4. **Existing research.** HOBBIT, Any-Precision LLM, MatQuant; Weightsift 5C (exact bit planes) and 5A strategy D
   (levels).
5. **Difference from Weightsift's experiments.** 5A's levels cost 1.5–1.63× storage and served certification. Nested
   planes cost about 1.0× and serve approximation.
6. **Assumptions.** A nested quantizer whose 2-bit prefix is usable. Plain truncation of a round-to-nearest int4 code is
   a poor 2-bit quantizer; MatQuant needed training for good int2.
7. **Failure modes.**
   - Truncated quality.
   - Plane-major layouts break the fast int4 inner loop. Colibrì's ledger: "an address-dependent codec ... destroys the
     cheap raw-INT4 inner loop".
8. **Expected benefit.** Up to 0.5× bytes on the misses served early. The net effect is [Unknown] until quality is
   measured.
9. **Complexity.** High.
10. **Smallest credible experiment.** On Moonlight: teacher-forced KL of truncated nested experts against the BF16
    original (E6's harness).
11. **Criterion.**
    - **Success:** the 2-bit prefix stays within the budget for ≥ 50% of streamed accesses.
    - **Failure:** the prefix's KL exceeds that of an independently quantized 2-bit expert by more than 30%.

### H7. An integer-exact reference profile for exact mode

1. **Core idea.** Define "exact" as integer W4A8 arithmetic with fixed reduction orders, which every device can
   reproduce. Then exact speculation, CPU/GPU portability and any future certification share one deterministic
   semantics.
2. **Mechanism.** Integer dot products are associative and exact; rounding happens once per output. That removes 5A's
   rounding floor and the dependence on batch shapes (6B § 12).
3. **Why not yet, and novelty.** Colibrì's `IDOT` and `COLI_EXACT_VERIFY` exist on the CPU, at about 0.6× speed for the
   exact attention core on a tiny oracle [Verified]. Its GPU paths use f32 activations. Novelty: modest.
4. **Existing research.** Batch-invariant kernels (Thinking Machines, 2025); Higham–Mary (why typical errors stay small);
   Weightsift's 6C preparation.
5. **Difference from Weightsift's experiments.** Weightsift's reference was PyTorch BF16 with kernels chosen by shape.
6. **Assumptions.** Activation quantization to int8 is part of the declared reference, which is lossy against BF16. GPU
   integer kernels are fast enough.
7. **Failure modes.** The speed cost; and quality, if int8 activations hurt the model (Colibrì's Qwen kernels already
   round activations to int8 on the CPU).
8. **Expected benefit.** Correctness portability, not speed. It is the precondition for H5 across devices and for E10.
9. **Complexity.** Medium to high.
10. **Smallest credible experiment.** On Colibrì's tiny Inkling fixture: the CPU's integer path against a GPU integer
    kernel, bitwise.
11. **Criterion.**
    - **Success:** bit-identical logits at ≤ 1.2× the CPU kernel's time.
    - **Failure:** no GPU integer path within 2×.

### H8. Prefetch targeted at misses, for 2% residency

1. **Core idea.** Predict the next layer's routing from the current state (`PILOT`'s method: 71.6% recall on GLM)
   [Verified]. Issue reads only for predicted experts that will miss, below demand priority.
2. **Mechanism.** In a serialized token the drive idles during the 0.71 s of arithmetic. Filling that idle time with
   correct prefetches recovers up to about 20% (§ 5.6).
3. **Why not yet, and novelty.**
   - Colibrì rejected prefetch where the predictable experts were already resident: "useful learning must target
     cold-miss probability directly" [Verified, its test matrix].
   - At 2% residency, most predicted experts are misses.
   - Novelty: low; the regime is new.
4. **Existing research.** Pre-gated MoE, ProMoE, ExFlow, HOBBIT, AdapMoE.
5. **Difference from Weightsift's experiments.** Weightsift's prefetch was exact (chunks known in advance).
6. **Assumptions.** Inkling's next-layer predictability resembles GLM's [Unknown], and wrong prefetches do not displace
   useful residents.
7. **Failure modes.**
   - The drive is busy for most of a token anyway, so wrong reads cost real time.
   - The gain is bounded by the idle fraction.
8. **Expected benefit.** At most about 1.25× in S0; less with imperfect precision.
9. **Complexity.** Medium. The store's priorities and advisory prefetch exist [Verified, 7A]; the predictor goes in the
   engine.
10. **Smallest credible experiment.** Moonlight locally (about 0.4 s per token): a hook applies layer L+1's router to
    layer L's state, and the recall of non-resident experts is measured under a simulated 2% cache. No download.
11. **Criterion.**
    - **Success:** precision ≥ 0.6 and recall ≥ 0.5 of misses at one layer of lookahead.
    - **Failure:** precision < 0.4.

### H9. Expert-major prefill for Inkling, exact

1. **Core idea.** For a prompt, read each distinct expert of a layer once and apply it to all its tokens. Keep each
   row's routing-order sum so the logits stay bit-identical.
2. **Mechanism.** It bounds prefill reads at u(S) per layer, against up to S·6 with token-major rounds at a small `cap`
   (§ 5.4) [Inference from `c/inkling.c`].
3. **Why not yet, and novelty.** Colibrì's GLM engine already "reads each expert once" per batch of positions
   [Verified, README]; its Inkling CPU path does not [Verified, code structure]. Novelty: none, a port. Its value is
   for small machines.
4. **Existing research.** Standard grouped MoE execution (vLLM, SGLang).
5. **Difference from Weightsift's experiments.** Weightsift's chunked calls (Phase 4B) bounded prefill memory. They did
   not reorder reads.
6. **Assumptions.** Scratch memory for per-pair outputs (about 147 MB at S = 1,000).
7. **Failure modes.** The re-reads may already be rare at the caps a 32 GB machine allows (E9 checks).
8. **Expected benefit.** TTFT up to several times lower at small caps on long prompts [Inference]; nothing for decode.
9. **Complexity.** Low to medium, as a patch to Colibrì's Inkling engine. Colibrì could take it upstream.
10. **Smallest credible experiment.** E9: count fills against distinct experts on the tiny oracle at `cap` 1–2.
11. **Criterion.**
    - **Proceed** if fills exceed 1.5× distinct experts at S = 64.
    - **Drop** if ≤ 1.2×.

### H10. Lossy routing reductions, measured instead of assumed

1. **Core idea.** Measure Colibrì's existing lossy levers on the same quality scale: `TOPP`, `CACHE_ROUTE` and
   `DEGRADE_ZERO`, together with AdapMoE-style sensitivity gating. Then publish the quality–bytes curves.
2. **Mechanism.** Fewer routed experts per token means proportionally fewer bytes: `TOPP` 0.7 means 30–40% fewer reads
   [Verified, Colibrì].
3. **Why not yet, and novelty.** Colibrì ships the levers with mostly unmeasured quality. Novelty: in the measurement,
   not the mechanism.
4. **Existing research.** AdapMoE (−25% experts "without accuracy degradation"); MC-MoE's online pruning; dynamic
   top-k routing.
5. **Difference from Weightsift's experiments.** None tried.
6. **Assumptions.** Moonlight's sensitivity is informative for Inkling's [Unknown]; checked later with H12.
7. **Failure modes.** Small regressions concentrated on hard tasks: per-row int4 already showed this pattern on GLM.
8. **Expected benefit.** About 0.7× bytes at a [Unknown] quality cost.
9. **Complexity.** Low.
10. **Smallest credible experiment.** On Moonlight: KL and benchmark subsets at p ∈ {1, 0.9, 0.8, 0.7, 0.6} (E6).
11. **Criterion.**
    - **Success:** some p ≤ 0.75 of routed bytes within the budget.
    - **Failure:** none below 0.9 within it.

### H11. A pruned derivative of Inkling, quality-budgeted

1. **Core idea.** REAP-prune 25–50% of the experts by router-weighted activation norms.
2. **Mechanism.** The bank halves, so residency doubles at the same RAM and the cold prefill bound halves. Bytes per
   token change only through the hit rate.
3. **Why not yet, and novelty.** No pruned Inkling is known [Unknown]; Colibrì ships a REAP DeepSeek V4 Flash
   [Verified]. Novelty: none in the method.
4. **Existing research.** REAP; NAEE ("not all experts are equal").
5. **Difference from Weightsift's experiments.** None.
6. **Assumptions.** Calibration needs the original's activations: teacher-forced runs on calibration text (H12's
   machinery, or rented hardware).
7. **Failure modes.** Quality on tasks outside the calibration domain. REAP's near-losslessness is reported for code.
8. **Expected benefit.** h roughly doubles at the same cache size (Moonlight's replay: 0.14 → 0.27 when the cached
   fraction of the bank doubles from 3.5% to 6.9%) [Inference]; TTFT × 0.5.
9. **Complexity.** Medium, given H12.
10. **Smallest credible experiment.** Moonlight at 25% and 50% pruning (REAP's code), against BF16, on E6's harness.
11. **Criterion.**
    - **Success:** 25% pruning within the budget on every category.
    - **Failure:** 25% pruning fails any category.

### H12. A streamed teacher-forced evaluator: Weightsift as the referee

1. **Core idea.** A GPU-arithmetic, chunked, teacher-forced scorer streams every expert once per chunk and measures KL
   and top-1 agreement against a reference representation. It makes flagship quality measurable on this machine.
2. **Mechanism.** § 4.4: about 145 s of reads and about 20 s of GPU arithmetic per 4,096-token chunk. That is about one
   hour per 100k scored tokens, against days for generation.
3. **Why not yet, and novelty.**
   - Colibrì evaluates by generation and by log-likelihood at 40 items.
   - Nobody evaluates disk-streamed flagship representations at scale on consumer hardware.
   - Novelty: practical, and it uses what Weightsift already built (Phase 4B's streamed transformers model, 6B's GPU
     decode and device tier).
4. **Existing research.** Perplexity and KL as quantization proxies are standard; their correlation with benchmark
   regressions must be shown, not assumed.
5. **Difference from Weightsift's experiments.** It applies 4B/6B's exactness machinery to measurement instead of
   decoding speed.
6. **Assumptions.**
   - transformers can instantiate Inkling: its model card lists transformers among the engines [Verified]; the
     implementation's details are [Unknown].
   - Int4 dequantization on the GPU is trivial.
   - A reference is available: the NVFP4 release, about 592 GB; with the int4 container, about 1.1 TB of the 1.37 TB
     free on `E:`.
7. **Failure modes.**
   - The architecture is unavailable or too slow in transformers.
   - KL may not predict benchmark regressions, which is calibrated on Moonlight first (E6).
8. **Expected benefit.** Every quality-budgeted claim about flagships, and a quality-measured comparison with Colibrì's
   containers.
9. **Complexity.** Medium.
10. **Smallest credible experiment.** E6 on Moonlight: KL against benchmark regressions over 8 representations.
11. **Criterion.**
    - **Success:** Spearman ≥ 0.8 between KL and mean benchmark regression across representations.
    - **Failure:** < 0.5. KL is then not a usable screen, and flagship quality must be measured on rented hardware.

## 8. Top five technical opportunities

Ranked by expected information gain per unit of engineering effort, not by upside.

1. **Make flagship quality measurable, then measure what people actually run (H12, with H10's curves).**
   - Evidence:
     - GLM's per-row int4 measured about 9 points below grouped scales [Verified];
     - Inkling's per-row container has no evaluation [Verified];
     - Colibrì's benchmark uses 40 items [Verified];
     - generation is infeasible here (§ 4.4).
   - Value: every quality-budgeted claim, and a quality comparison with Colibrì's containers, depend on it. It is also
     the capability Colibrì lacks most.
   - Risks: transformers support for Inkling; KL as a proxy (E6 gates it).
2. **Mass-weighted precision for streamed experts (H1), with measured routing reductions (H10).**
   - Evidence: the only levers worth ≥ 1.6× on the dominant term (§ 5.3 S2–S4); prior art in HOBBIT, MC-MoE and MoQE.
   - Risks: 2-bit quality on fine-grained experts; flat mass over the non-resident experts, which turns it into plain
     lower precision.
3. **Exact entropy coding of int4 experts (H2).**
   - Evidence: GLM's codes at 2.91 bits of 4; per-row rANS at 0.76 [Verified]; Weightsift's own 6B experience.
   - Value: exact, up to 1.36×, it compounds with everything else, and E4 and E5 screen it cheaply.
   - Risks: CPU decoding speed on AVX2; Colibrì may merge its engine decode first, which is fine for users but removes
     differentiation.
4. **Inkling's I/O path at the storage seam: expert-major prefill (H9), batched misses, miss-targeted prefetch (H8),
   through the 7A store.**
   - Evidence: the engine's structure (§ 3.1) and its own comment on queue-depth-1 fills.
   - Value: exact; up to 1.25× decode and possibly much lower TTFT at small caps.
   - Risks: E8 may show the drive saturated already. Colibrì can adopt every part of it.
5. **Placement by frequency per byte, dense weights included (H4); then speculation once bytes are low (H5).**
   - Evidence: § 5.5–5.6.
   - Value: it matters only after 2–4 have cut the bytes.
   - Risks: per-layer device round trips (Colibrì's ledger).

## 9. Experiments: minimal, falsifiable, with gates

**No download and no change to the inference path.** These are runnable now, after the user's go-ahead for code.

| ID | Tests | Cost | Metric | Success gate | Failure gate |
| --- | --- | --- | --- | --- | --- |
| E1 | The machine's unknowns | Minutes | RAM read bandwidth (STREAM-like); `E:`'s direct-I/O rate and its link; the PCIe topology (why the GPU links at x8; whether a CPU-attached x4 slot is free) | Parameters for § 5 | — |
| E2 | H1, H4, H5 inputs, from Moonlight's existing traces (Phase 4B, 5A) | Hours | Expert-mass skew per layer; hit-rate curves at 1–5% residency for LRU, LFU and pins; consecutive-token union U(k), k = 1–4 | H1: the hottest 20% of non-resident experts carry ≥ 50% of their mass. H5: U(3)/(3·E) ≤ 0.75 | H1: the mass is near uniform. H5: U(3)/(3·E) > 0.85 |
| E5 | H2 decode cost | Hours | Output GB/s of Colibrì's `rans.h` scalar and branch-free arms on this CPU, on symbols drawn from GLM's histogram | ≥ 6 GB/s on ≤ 2 cores | < 4 GB/s |
| E8a | Colibrì's Windows release works here | Hours | `olmoe.exe` from v2.0.0 on the local OLMoE (its int8 conversion), at a forced small `cap`, with page-cache control and the OS's I/O counters. A methodology shakedown, not a claim | Runs, with counters matching its stats | — |
| E9 | H9 | Hours | Slot fills against distinct experts per layer, Colibrì's tiny Inkling oracle, `cap` 1–2, S = 64 | Fills > 1.5× distinct | ≤ 1.2× |
| E10 | Reopening certification | One day | On 5C2's 12 samples, real arithmetic: the best metadata of ≤ 10% of an expert's bytes keeping the attained set minimum > 0 | ≥ 50% of tokens | < 10% |

**Small dataset downloads, which need the Quality Budget's datasets pinned (the user's open decision).**

| ID | Tests | Cost | Metric | Success gate | Failure gate |
| --- | --- | --- | --- | --- | --- |
| E6 | H12 proxy, H10 | Days | On Moonlight, 8 representations (BF16; int4 per-row and g64; int3 and int2 g64; top-p 0.8 and 0.7; one mixed allocation): teacher-forced KL against BF16, and benchmark subsets with intervals (GSM8K, MMLU-Pro, HumanEval, a multilingual set) | Spearman(KL, mean regression) ≥ 0.8 | < 0.5 |
| E7 | H1 | Days | At equal average bits (2.5, 3.0): uniform, mass-weighted and mass × sensitivity allocations | Weighted beats uniform outside the intervals on ≥ 2 categories, or matches uniform 4-bit at ≤ 0.75 of the bytes | Indistinguishable on every category |

**Downloads that need the user's approval.**

| ID | Tests | Cost | Metric | Success gate | Failure gate |
| --- | --- | --- | --- | --- | --- |
| E4 | H2 ratio, H1 statistics | About 1.8 GB by range requests: 64 sampled Inkling experts from the int4 container, plus BF16 rows of a few of them | Code entropy per row and layer; scale distributions; per-row int4 against NVFP4 reconstruction error on the BF16 rows (a screen only: weight error is not quality) | e ≤ 0.80 | e > 0.85 |
| E8 | Every flagship claim: Colibrì's Inkling baseline on this machine | 514 GB (int4 container) plus the dense conversion, about 14 minutes | `configs/phase7a-baseline.yaml` run on release v2.0.0's `inkling.exe`, CPU and Vulkan tier; PIN off and history pins; cold, warm and rotating prompts; decode tokens/s, TTFT, drive busy %, hit rate, drive bytes from OS counters | Storage on the critical path (drive busy < 80% in decode, or prefill fills > 1.5× distinct experts): build the store patch | Drive busy ≥ 90% with fills ≈ distinct experts: stop the store work; only bytes matter |
| E12 | Quality of the containers people run | Plus 592 GB (NVFP4) | H12's evaluator: per-row int4 + int4-gs64 dense against NVFP4, KL over 100k tokens | — (a measurement) | — |

## 10. Recommended architecture

**Reuse, redesign, eliminate**:

| Component | Recommendation | Reason |
| --- | --- | --- |
| Model execution (routing, kernels, KV, speculation) | **Reuse Colibrì's engines**, pinned, with reviewed patches at the storage seam only (decision 0014 stands) | § 3.3; maintaining 13 model families is out of reach |
| Weight storage backend (`weightsift-io`, `weightsift-store`, C ABI) | **Keep, narrow and gate.** Patch Inkling's `slot_fill` only if E8 shows storage on the critical path. Then add expert-major prefill (H9), batched misses and miss-targeted prefetch (H8), and compressed host caching (H2) | The only Weightsift code on a flagship path; its value is conditional on E8 |
| Runtime planner (`weightsift-runtime`) | **Redesign** into a frequency-per-byte placement solver over dense and expert bytes (H4), fed by E1 and E2 | Today it reserves budgets but optimizes nothing |
| Quality Budget (`awpmi.quality`) | **Promote to the centre.** Pin datasets; add multilingual; add H12's teacher-forced evaluator as the screening tier, calibrated by E6 | The differentiator (§ 8, item 1) |
| Representation tools | **New, small.** A "representation compiler" taking a model, a machine and a budget to bits per expert, coding, pruning and a top-p policy, as artifacts with manifests and a quality report, consumed by engines through the catalog | Artifacts are model data, not engine code: independence without maintaining engines |
| PyTorch Moonlight runtime (Phases 4B–6B) | **Keep as the local laboratory** for E2, E6, E7 and H8; stop optimizing it (no Phase 6C) | It cannot run a flagship, but it is an exact, instrumented model of the problem |
| Certification (AWPMI oracles) | **Eliminate from the roadmap**; keep the code and reports as negative results; reopen only through E10 | § 2.9 |
| Colibrì segment-runtime bridge | **Freeze** until an engine runs through it | Fixtures only today |
| ApexQuant track | **Eliminate** (BLOCKED, and its 4-bit class does not pay, 7A) | — |

**The target architecture**:

```text
Quality Budget + evaluator (Python)     what is allowed: categories, intervals, KL screen, benchmarks
        ↓
Representation compiler (Python/Rust)   bits per expert and layer, lossless coding, pruning, top-p; artifacts + manifest
        ↓
Planner (Rust)                          dense and expert bytes over VRAM / RAM / disk, by frequency per byte
        ↓
Store (Rust, C ABI v1)                  leases, byte budget, compressed host cache, priorities, prefetch, accounting
        ↓
Engine (Colibrì, pinned)                routing, kernels (CPU / Vulkan / CUDA), KV, speculation, serving
```

**Two operating modes, with explicit contracts.**

- **Exact:** token-identical to the pinned engine on a fixed container, whatever the placement (Colibrì's rule).
- **Quality-budgeted:** every derived artifact carries its quality report against the strongest available baseline.
  For Inkling that means NVFP4, then the per-row container; the original BF16 cannot be stored here.

## 11. Development roadmap

Each proposal lists the brief's fields. Priorities run from P1 (first) to P3. Every item that downloads a checkpoint, or
buys hardware, needs the user's approval.

### Immediate (days, no flagship download)

**I1. Measure the machine and mine the traces (E1, E2).**

- **Priority:** P1.
- **Hypothesis:** the § 5 model's unknowns (B_ram, the second drive, routing skew, U(k)) decide which of H1, H4 and H5
  are worth anything.
- **Mechanism:** direct measurement and trace replay.
- **Prerequisites:** the existing Moonlight traces; nothing else.
- **Evidence:** § 5.2's [Unknown] rows.
- **Experiment:** E1, E2.
- **Metrics:** GB/s, skew, hit-rate curves, U(k).
- **Success gate:** every parameter measured, with the E2 gates evaluated.
- **Failure gate:** none (a measurement).
- **Effort:** low.
- **Risk:** Moonlight's routing may not represent Inkling's.

**I2. Screen the exact levers (E5, E9).**

- **Priority:** P1.
- **Hypothesis:** H2's decode keeps up with the drive; H9's re-reads exist.
- **Mechanism:** a microbenchmark and a fixture count.
- **Prerequisites:** Colibrì's sources (pinned fetch) and its tiny-oracle tools.
- **Evidence:** § 3.1, § 5.4.
- **Experiment:** E5, E9.
- **Metrics:** GB/s; fills per distinct expert.
- **Success gates:** E5 and E9's.
- **Failure gates:** E5 and E9's.
- **Effort:** low.
- **Risk:** a microbenchmark win does not transfer end to end. It gates further work only.

**I3. Shake down Colibrì's Windows release here (E8a).**

- **Priority:** P2.
- **Hypothesis:** the baseline needs no gcc.
- **Mechanism:** release v2.0.0 binaries.
- **Prerequisites:** none.
- **Evidence:** `release.yml`, the GitHub API.
- **Experiment:** E8a.
- **Metrics:** runs; counters consistent.
- **Success gate:** it runs with consistent counters.
- **Failure gate:** it does not run. Then the toolchain question of 7A returns.
- **Effort:** low.
- **Risk:** Smart App Control blocks the binaries (Colibrì's [docs/windows.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/windows.md) documents this).

**I4. Pin the Quality Budget's datasets and add a multilingual category.**

- **Priority:** P1.
- **Hypothesis:** quality measurement is the bottleneck of every lossy claim.
- **Mechanism:** pinned revisions in `configs/quality-budget.yaml`.
- **Prerequisites:** the user's confirmation of the datasets (an open decision since 7A).
- **Evidence:** § 4.4.
- **Experiment:** `weightsift quality check` passes.
- **Metrics:** —.
- **Success gate:** all revisions pinned.
- **Failure gate:** —.
- **Effort:** low.
- **Risk:** licence terms of some datasets.

### Near-term (weeks)

**N1. The proxy study on Moonlight (E6, E7): KL against benchmarks, mass-weighted allocation, top-p curves.**

- **Priority:** P1.
- **Hypothesis:** H12's screen is valid; H1 and H10 pay.
- **Mechanism:** existing quantizers (round-to-nearest grouped, HQQ- or GPTQ-class through existing libraries, no new
  algorithm), teacher-forced KL, benchmark subsets.
- **Prerequisites:** I4; Moonlight (local).
- **Evidence:** HOBBIT, MC-MoE, MoQE (§ 6.3).
- **Experiment:** E6, E7.
- **Metrics:** KL; per-category regressions with intervals; bytes per token through the cache simulator.
- **Success and failure gates:** E6's and E7's.
- **Effort:** medium.
- **Risk:** transfer from 64 to 256 experts per layer.

**N2. Colibrì's Inkling baseline on this machine (E8).**

- **Priority:** P1, once approved.
- **Hypothesis:** it locates the bottleneck (drive, orchestration, CPU, RAM).
- **Mechanism:** the 7A baseline specification, on the release binary.
- **Prerequisites:** approval of the 514 GB download.
- **Evidence:** § 5.
- **Experiment:** E8.
- **Metrics:** § 9.
- **Success and failure gates:** E8's.
- **Effort:** medium (mostly waiting).
- **Risk:** the dense conversion's quality, which is unmeasured; it is part of the baseline.

**N3. The store under Inkling's engine: the 7A patch, plus H9 and H8.**

- **Priority:** P2.
- **Hypothesis:** exact speedups up to 1.25× in decode and several times in TTFT at small caps.
- **Mechanism:** § 7 H8, H9.
- **Prerequisites:** E8 showing storage on the critical path; a gcc toolchain (MSYS2 UCRT64, not MSYS: Colibrì's #1979)
  or a Linux host.
- **Evidence:** § 3.1.
- **Experiment:** an ABBA of engine plus store against engine alone, token-identical.
- **Metrics:** decode tokens/s, TTFT, drive busy %, fills.
- **Success gate:** ≥ 10% decode or ≥ 30% TTFT, with identical tokens.
- **Failure gate:** < 3% and no TTFT gain. Then stop storage work.
- **Effort:** medium.
- **Risk:** upstream drift. Colibrì changed little in `inkling.c` since the pin (3 lines on `dev`).

**N4. Exact compressed experts (H2) through the store.**

- **Priority:** P2.
- **Hypothesis:** about 1.3× in S0, exact.
- **Mechanism:** rANS packs; a decoder where the experts are consumed.
- **Prerequisites:** E4 (approval), E5.
- **Evidence:** § 7 H2.
- **Experiment:** an ABBA on the E8 configuration.
- **Metrics:** drive bytes, decode tokens/s, CPU time in decoding.
- **Success gate:** ≥ 1.2× at identical tokens.
- **Failure gate:** < 1.05×.
- **Effort:** medium.
- **Risk:** decode CPU cost; Colibrì merging its own first.

### Research (months)

**R1. Mass-weighted precision on Inkling (H1), with H12 as the referee (E12).**

- **Priority:** P1.
- **Hypothesis:** 1.6–1.9× decode at a measured quality cost within the budget.
- **Mechanism:** § 7 H1.
- **Prerequisites:** N1 passed; downloads (int4 container, NVFP4 reference) approved; the BF16 rows of the experts being
  re-quantized, by range requests, or quantization from NVFP4.
- **Evidence:** N1.
- **Experiment:** derived Inkling artifacts evaluated with H12 against NVFP4.
- **Metrics:** KL, top-1 agreement, bytes per token, decode tokens/s.
- **Success gate:** KL within the band that E6 maps to the budget, at ≤ 0.65 of int4 bytes.
- **Failure gate:** no allocation below 0.8 of the bytes within the band.
- **Effort:** high.
- **Risk:** re-quantizing from the per-row container compounds error. From BF16 needs 1.9 TB, which cannot be stored, so
  expert-by-expert range reads would be required.

**R2. Union-aware speculation with Inkling's MTP layers (H5).**

- **Priority:** P3.
- **Hypothesis:** about 1.3× once bytes are low.
- **Mechanism:** § 5.5.
- **Prerequisites:** E2's U(k) gate; R1 or N4 done; MTP support in the engine.
- **Evidence:** Colibrì's ledger; DraftExpert.
- **Experiment:** an ABBA at fixed tokens.
- **Metrics:** tokens per forward, union, miss bytes, tokens/s.
- **Success gate:** ≥ 15% at identical output.
- **Failure gate:** < 0.
- **Effort:** high.
- **Risk:** union growth.

**R3. Nested precision on misses (H6) and pruning (H11).**

- **Priority:** P3.
- **Hypothesis:** further byte cuts under the budget.
- **Mechanism:** § 7.
- **Prerequisites:** R1.
- **Evidence:** HOBBIT, MatQuant, REAP.
- **Experiment:** Moonlight first.
- **Metrics:** as R1.
- **Success and failure gates:** as H6 and H11.
- **Effort:** high.
- **Risk:** quality.

**R4. An integer-exact reference across devices (H7).**

- **Priority:** P3.
- **Hypothesis:** portable exactness; a precondition for exact speculation on mixed devices.
- **Mechanism:** § 7 H7.
- **Prerequisites:** Colibrì's integer kernels.
- **Evidence:** batch invariance (2025).
- **Experiment:** the tiny fixture, bitwise.
- **Metrics:** bit equality, time.
- **Success and failure gates:** H7's.
- **Effort:** medium to high.
- **Risk:** speed.

### Long-term

**L1. The representation compiler as Weightsift's product.**

- **Priority:** P2.
- **Hypothesis:** quality-certified (statistically, not formally) representations and placement plans per machine,
  usable by Colibrì and other engines.
- **Mechanism:** § 10.
- **Prerequisites:** R1.
- **Evidence:** —.
- **Experiment:** two machines × two models × three budgets.
- **Metrics:** the latency–quality frontier against Colibrì's defaults.
- **Success gate:** a frontier strictly better at every budget on both machines.
- **Failure gate:** dominated at any budget.
- **Effort:** high.
- **Risk:** engines' format support.

**L2. Hardware as a lever, measured before it is bought.**

- **Priority:** P2.
- **Hypothesis:** the cheapest speedups on this machine are hardware.
  - A second NVMe on CPU-attached lanes: B_disk × about 2, S5 [Inference].
  - RAM from 32 to 64 GB, if the board allows it: an Inkling cache about 3× larger, and MiMo Pro's exact path (54 GB).
- **Mechanism:** § 5.
- **Prerequisites:** E1 (the topology), the user's budget.
- **Evidence:** Colibrì's +37.5% with a second drive on an independent controller [Verified].
- **Experiment:** the store's striping over two files on two drives.
- **Metrics:** GB/s, tokens/s.
- **Success gate:** ≥ 1.6× drive rate.
- **Failure gate:** < 1.2× (a shared link).
- **Effort:** low.
- **Risk:** the board's lane sharing [Unknown].

**L3. Retrained, locality-aware derivatives.**

- **Priority:** out of scope for this hardware.
- **Hypothesis:** Colibrì's ledger names it as the remaining honest direction for large gains.
- **Prerequisites:** training compute Weightsift does not have.
- **Status:** recorded, not planned.

## 12. Adversarial critique

**The brief's questions, answered against this report** [Inference]:

- **Which proposals are renamed existing techniques?**
  - H2 is DFloat11/ZipNN/rANS for int4.
  - H8 is PILOT, ProMoE and Pre-gated prediction.
  - H9 is grouped MoE execution, already in Colibrì's GLM engine.
  - H10 and H11 are AdapMoE and REAP.
  - H5 is MTP speculation with a different cost model.
  - H1 is close to HOBBIT plus MC-MoE.
  - Only H12, and the residency coupling in H1, are not plainly in the literature, and H12 is engineering, not a new
    idea.
- **Which rely on weak locality assumptions?** H1 (skew among non-resident experts), H5 (consecutive-token overlap), H8
  (next-layer predictability on Inkling). E2 measures them on Moonlight only. Inkling has no traces.
- **Which cannot overcome storage bandwidth?** H4, H7 and H9 in decode; H8 beyond about 1.25×; and speculation in S0.
  Only H1, H2, H6, H10, H11 and hardware reduce the bytes or raise the bandwidth.
- **Which underestimate decompression or compute cost?** H2 on an AVX2 CPU (E5 gates it); H6 (plane-major layouts break
  the int4 inner loop); H3 (Vulkan's arithmetic and per-layer synchronization).
- **Which conflict with exact semantics?** H1, H6, H10, H11 are lossy by design. H3 changes arithmetic on Vulkan, so
  identical tokens cannot be expected. H7 redefines the reference.
- **Which already failed in Weightsift?** Certification (§ 2.9) and shared bases. H6's plane layouts were costly on the
  GPU in 6B.
- **Which would be straightforward for Colibrì to adopt?** H2, H8, H9, H10, top-p and pruning: all of them in days to
  weeks. **Weightsift's durable differentiation is only in H12 and the evaluated artifacts H1 produces**, together with
  the evidence behind them. That holds unless Colibrì builds the same evaluation, which its owners could also do.
- **Which require unrealistic engineering?** H5 on Inkling (MTP plus rollback in a 4,480-line engine, upstream); R1 at
  full scale without a BF16 copy.
- **Which a simple experiment would invalidate?** H5 (E2's U(k)), H9 (E9), H2 (E4 and E5), H12 (E6), the store's value
  (E8).

**The strongest objections to the top recommendations.**

1. **Against H12 and quality-first.**
   - KL to NVFP4 is a distance to another quantization, not to the original, and the original cannot be stored here.
   - A representation could be closer to NVFP4 than to BF16, or the reverse.
   - Mitigation: anchor with the publisher's own scores (AIME 2026 97.1%, SWE-bench Verified 77.6%, GPQA Diamond 87.2%,
     Global-MMLU-Lite 88.7% on the model card [Verified]) on the few benchmarks reproducible on rented hardware.
   - Without that, quality claims stay relative.
2. **Against H1.**
   - At 2% residency the cache holds so little that "resident at full precision" barely matters. The saving is then just
     lower precision everywhere, and 2-bit post-training quantization of fine-grained experts is the risky part. MC-MoE
     reports −3.8% average at 2.54 bits on commonsense tasks, which are the easy ones.
   - The bet is that mass weighting finds a much better point. If E2 shows flat mass, the bet is lost before any
     quantization.
3. **Against the store work (N3).**
   - The drive may already be saturated by Colibrì's own fills.
   - Even if not, Colibrì can add an async pool to Inkling's engine as it did for GLM's. Building the patch before E8
     repeats 7A's pattern of building before measuring.
4. **Against the whole strategy.**
   - The target machine may be the wrong target. Colibrì's numbers suggest that 64 GB of RAM and a second NVMe change
     the regime more than any software here.
   - A quality-budgeted 975B at about 1 token/s with minutes of TTFT may still be unusable for the stated workloads
     (code, tool use, long context).
   - Inkling-Small (276B, 12B active [Verified, its model page]) or Qwen3.6-35B-A3B may serve users better on 32 GB.
   - The brief names the flagships, so they stay the target. The question deserves a measured answer (§ 13).

**Attractive but misleading directions.**

- **"A better verifier will rescue certification."** The sets themselves contain flipping weights (§ 2.9).
- **"Lower-bit weight error ≈ quality."** The brief forbids the equation: per-row int4's 11% relative L2 error says
  nothing about the 9-point loss measured on GLM.
- **"Microbenchmark wins transfer."** Colibrì's ledger records several reversals under ABBA. Weightsift's own 6B3-B went
  from −1.3% to unmeasurable.
- **"Routing predictability is high, so prefetch pays."** Recall of routing is not recall of misses.
- **"Speculation always pays on offloaded models."** True for dense models (SpecExec); false for MoE in the disk-bound
  regime (§ 5.5).
- **"Rust makes it faster."** On this machine the drive sets the rate; Rust's value is safety at the seam.
- **"Sub-1-bit MoE compression exists (QMoE)."** Measured on SwitchTransformer, not on fine-grained modern MoEs.
- **"Phase 6B's 408.9 ms is progress toward flagships."** It is progress on a model and framework that do not carry
  over.

## 13. Final judgment

- **What the evidence does not support.** A path to outperforming Colibrì as an inference engine. Colibrì owns the
  engines, the kernels and the measurements on real flagships, and any exact engine-side gain on this machine is capped
  near 1.7× for everyone (§ 5.6) and adoptable by Colibrì.
- **What it does support,** with the risks of § 12, is a more modest and more defensible advantage on the brief's own
  categories:
  - **Goal 4, a better latency–quality tradeoff on the same hardware: the first priority.** Measured quality per byte:
    a referee that makes flagship quality measurable on consumer hardware (H12), and representations allocated by
    routing mass and validated under the Quality Budget (H1, H10, H11).
  - **Goal 3, a larger usable model on the same hardware: second priority.** A planner that runs the flagships on less
    RAM: MiMo-V2.6 Pro on 32 GB needs a quality-budgeted dense representation, and Inkling's needs fewer streamed bytes
    (H4, H1).
  - **Goal 1, same quality and faster, in exact mode: third.** Only exact storage gains (H2, H9, H8), each gated by a
    measurement and each adoptable by Colibrì. They are worth doing for users, not as a moat.
- **Not prioritized.** Goal 2 (same quality, less RAM or VRAM) falls out of goal 3. Goal 5 (adaptability across
  hardware) is where Colibrì's setup and planner are already ahead.
- **The test of the strategy.** At equal hardware and equal quality as measured by the Quality Budget, Weightsift's
  artifacts plus Colibrì's engine decode faster than Colibrì's defaults. If after N1 and R1 no representation beats
  Colibrì's per-row container on the KL–bytes plane, the honest conclusion is that Weightsift's contribution is
  evaluation methodology. It should then be contributed upstream rather than grown into an independent system.

### A. The single most promising change not yet explored sufficiently

**Residency-aware precision: deciding each expert's bits jointly with its tier and routing mass, so that the bytes
streamed from disk shrink where the expected quality impact is smallest, and measuring the result under the Quality
Budget.**

- Weightsift has spent five phases on bit-exactness. It has never measured the quality of a single lossy expert
  representation, while on its target machine every flagship run is lossy anyway.
- Disk bytes are 80% of a token (§ 5.3), and representation is the only lever with ≥ 2× headroom on them.

### B. The biggest conceptual mistake or unnecessary constraint

**Treating exactness against a declared reference as the product, and certification as the thesis.**

- On the target hardware **neither flagship can run exactly against its own release.**
  - Inkling's BF16 dense set does not fit in RAM, and its experts run from a derived per-row int4 container with no
    evaluation.
  - MiMo Pro's dense weights do not fit even in int8.
- So the binding question is quality per byte, which the guide relegates to a secondary "quantized mode" (§ 19, § 24).
- **Two constraints follow from the same mistake.**
  - Building before measuring: the 7A store before the baseline; 6B's optimization of a PyTorch path that cannot reach
    flagships.
  - Treating the gcc toolchain as a blocker for the baseline, when Colibrì's Windows release ships `inkling.exe`
    prebuilt.

### C. One month, one direction

**The direction:** quality per byte for routed experts (H1 with H12's screen, H10's curves on the way), on Moonlight
locally, with no flagship download.

**Implement first:**

1. The teacher-forced KL scorer on Moonlight against its BF16 original, chunked, GPU arithmetic. Weightsift's Phase
   4B/6B streaming model already computes these logits exactly.
2. The pinned benchmark subsets with intervals (I4).
3. A per-expert score: routing mass from E2's traces × output error on calibration activations, at 4, 3 and 2 bits with
   an existing quantizer.

Then E6 and E7.

**Continue if both hold:**

- KL predicts benchmark regressions: Spearman ≥ 0.8 across the 8 representations.
- At ≤ 2.75 average bits per routed weight, about 0.69 of int4's bytes, the mass-weighted allocation stays within the
  budget (every category's regression bound ≤ 2 points at 95%) and beats uniform allocation at equal bytes outside the
  intervals on at least two categories.

**Abandon the direction if either holds:**

- Weighted and uniform allocations are indistinguishable at equal bytes. That is known art, and Colibrì can ship it.
- No allocation below 3 bits stays within the budget. Then the lossy headroom on this hardware is about 1.3× (S2), not
  2×.

In that case the next month should go to E8 and the exact storage levers instead.

## Sources

Weightsift (this repository, `54c31eb`):

- Phase reports in `history/`:
  - 1C (`2026-10-02-awpmi-phase1c-report.md`), 2 (`2026-10-02-awpmi-phase2-report.md`);
  - 4B (`2026-10-03-awpmi-phase4b-report.md`);
  - 5A (`2026-10-04-awpmi-phase5a-report.md`), 5A2 (`2026-10-07-awpmi-phase5a2-report.md`), 5C
    (`2026-10-08-awpmi-phase5c-report.md`);
  - 6A (`2026-10-08-weightsift-phase6a-report.md`), 6B (`2026-10-09-weightsift-phase6b-report.md`);
  - 7A (`2026-10-12-weightsift-phase7a-report.md`).
- Decisions 0009–0014; `state/Weightsift_Implementation_Guide.md`.
- Code: `native/core/src/file.rs`, `native/core/src/engine.rs`.
- Data: `experiments/phase6b/gpucache/replay.txt`, `experiments/phase7a/plan-inkling/`,
  `research/apexquant/geometry.json`.

Colibrì (Apache-2.0), [github.com/JustVugg/colibri](https://github.com/JustVugg/colibri), `bf244291` (2026-10-06) and
`dev` `2fc033dc` (2026-10-15):

- Documentation:
  - `README.md`;
  - [docs/inkling.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/inkling.md), [docs/mimo.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/mimo.md), [docs/windows.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/windows.md), [docs/tuning.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/tuning.md), [docs/speculative.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/speculative.md);
  - [docs/CACHE_ROUTE.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/CACHE_ROUTE.md), [docs/int4-rans256-g0.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/int4-rans256-g0.md), [docs/benchmarks.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/benchmarks.md);
  - [docs/experiments/glm52-decode-failure-ledger-2026-07-31.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/experiments/glm52-decode-failure-ledger-2026-07-31.md),
    [docs/experiments/glm52-split-entropy-2026-07-31.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/experiments/glm52-split-entropy-2026-07-31.md),
    [docs/experiments/inference-paper-test-matrix-2026-07-28.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/experiments/inference-paper-test-matrix-2026-07-28.md), [docs/experiments/cnre-offline-simulator.md](https://github.com/JustVugg/colibri/blob/2fc033dc62753f2fa75a04326f55830b700805c1/docs/experiments/cnre-offline-simulator.md).
- Code and build: `c/inkling.c` (lines 1550–1672, 2206–2474), `.github/workflows/release.yml`.
- [Release v2.0.0](https://github.com/JustVugg/colibri/releases/tag/v2.0.0) (2026-10-06).

Models:

- [thinkingmachines/Inkling](https://huggingface.co/thinkingmachines/Inkling) and its
  [config.json](https://huggingface.co/thinkingmachines/Inkling/blob/main/config.json).
- [nbeerbower/Inkling-colibri-int4](https://huggingface.co/nbeerbower/Inkling-colibri-int4).
- [NVIDIA Dynamo: Inkling NVFP4 recipe](https://docs.nvidia.com/dynamo/recipes/inkling).
- [Inkling-Small (Together AI model page)](https://www.together.ai/models/inkling-small).

Papers and project pages: linked in § 6, with their dates. The 2026 speculative-decoding papers (MoE-Spec, DraftExpert,
*The Limits of Speculation*) were read through their abstracts only.
