# 0014 — Phase 7A: an independent Rust-native flagship runtime around Colibrì's engines; ApexQuant behind a licensing gate

Status: accepted (Phase 7A, 2026-10-12)

## Context

Phases 1A–6B built AWPMI's certificates and oracles, then an exact external-memory runtime: Moonlight-16B-A3B runs out of
VRAM and host RAM bit for bit equal to an independent BF16 reference (decision 0008), on a native Rust I/O core with a
host-RAM expert cache (0012), with encoded experts, a device tier and decode graphs at 408.9 ms per decode token (0013).
Everything so far ran a 16B model in BF16 with PyTorch as the execution engine.

The user's Phase 7A brief (2026-10-12; committed in 54c31eb, since replaced by the Phase 7B brief) turns the project toward flagship MoE models of about one
trillion parameters on constrained consumer hardware:

- **Weightsift stays an independent project with its own architecture and a Rust-native systems core.** Colibrì and DS4
  are not forked and not vendored; upstream components are reused selectively through version-pinned dependencies, FFI
  boundaries or properly licensed extraction; validated engines and kernels are preferred over new ones; optimized C,
  CUDA or Vulkan is not rewritten because it is C. The project stays model-agnostic through common interfaces, with
  model-specific optimized backends allowed. Validated results, extrapolations and hypotheses are kept apart.
- **Quality, not bit-exactness, for new quantized modes**: the goal is near-original model quality; existing bit-exact
  regression tests are kept, but not imposed on new quantized inference modes.
- **Targets**: first Inkling 975B (Thinking Machines; Colibrì's expert streaming as the execution reference, its
  low-memory configuration with int4 dense weights and int4 routed experts), then MiMo-V2.6 Pro 1.02T (Xiaomi; routed
  experts released in MXFP4), whose documented memory needs do not fit 32 GB as they are. The reference hardware (RTX 4060
  Ti 8 GB, 32 GB RAM, NVMe, PCIe 3.0 x8) is detected at run time, never assumed. No full checkpoint or large dataset is
  downloaded without the user's approval; performance and quality claims need real flagship workloads.
- **Stage A**: a component-level reuse and licensing audit of Colibrì, DS4 and Weightsift, and the smallest viable reuse
  architecture; Colibrì's `expert_store.h` and registry are examined without assuming the seam exists for Inkling or MiMo.
- **Stage B**: separate the model execution backend (forward computation, attention, routing, quantized kernels), the
  weight storage backend (pages, I/O, admission, eviction, prefetch, reservations, budgets, statistics) and the runtime
  planner (memory limits, expert requests, placement, later precision policies); a minimal, versioned, model-neutral native
  interface for expert storage and its execution integration; Rust for new storage and scheduling infrastructure; the
  existing cache, direct-I/O and worker machinery kept; narrow FFI with explicit lifetime, ownership and thread-safety
  contracts; no use-after-free, double release, invalid lease or eviction of a buffer in use; strict RAM and VRAM
  accounting; the existing implementation as the fallback; deterministic tests; a small real vertical slice used from
  Rust (fixtures where the flagship is unavailable, never presented as 1T inference).
- **Stage C, in parallel and independently stoppable**: ApexQuant (`afristrup/ApexQuant`, arXiv 2610.07904). A licensing
  gate first; then a technical study (re-isotropization, nested passes, scalar/E8/trellis stages, fused kernels, prefix
  reconstruction, layout and decoding cost, pass allocation) and whether its formats can serve routed expert matrices;
  real-weight measurements only on locally available or approved flagship tensors; screening against the native MXFP4 or
  INT4 format (a 20% byte reduction at acceptable compute is a screening target, not a promise); no full conversion.
- **A Quality Budget** (coding, mathematical reasoning, general reasoning, instruction following, long context, tool use):
  original checkpoint quality kept apart from a quantized release's, the strongest available baseline, held-out data
  distinct from calibration data, per-category regressions with sample counts and uncertainty, configurable thresholds,
  no invented scores. Specification, schema and comparison utilities now; certification later.
- **A benchmarking contract**: a reproducible Colibrì baseline on identical hardware, checkpoint, precision, budget and
  workload before any optimization claim; decode tokens/s, prefill latency, time to first token, cold and warm, rotating
  prompts, SSD and host-to-device bytes per token, hit ratios, peak RAM and VRAM, utilization, I/O wait, output
  differences; microbenchmarks never reported as end-to-end results.
- **Non-goals**: rewriting Colibrì or DS4 in Rust, a new flagship architecture, a new quantization algorithm, per-token
  quantization planning, expert skipping or altered routing, greedy-argmax certification, large conversions, training,
  unapproved downloads, a broad refactor, 1T claims without a 1T run. **Stop conditions**: illegal reuse, an unexpectedly
  large upstream rewrite, hardware that cannot run a measurement, unauthorized downloads or compute, no credible headroom,
  an unresolved correctness or ownership issue; a stop is reported as a stop.

## Decision

1. **Two meanings of "good enough", kept apart.** An exact path (the BF16 Moonlight runtime, every certificate, every
   fallback) is still judged bit for bit against its declared `ReferenceProfile` (decisions 0001, 0007), and its tests
   stay. A new quantized inference mode is judged by the Quality Budget against the strongest available baseline, per
   category, with sample counts and uncertainty; weight-space error is never quality evidence, and a mode without
   measured quality is labelled as such.
2. **Stage A: the reuse audit (pinned: Colibrì `bf2442915d6e`, DS4 `fc80bd695da7`, 2026-10-12).**
   - *Colibrì* (Apache-2.0, `NOTICE`: Copyright 2026 Vincenzo Fornaro; parts adapted from DS4 under MIT) has the flagship
     engines: `c/inkling.c` (CPU; CUDA for BF16 residents; a Vulkan expert tier), `c/mimo.c` (CPU; Vulkan), their int4,
     int8, MXFP4 and FP8 CPU kernels, KV and state management, per-layer expert slot caches with pins, token-exact oracles
     against Hugging Face on tiny checkpoints and real-checkpoint checks for MiMo. Its `c/expert_store.h` (lookup, release,
     prefetch, stats, destroy, with a written lease contract) is used by the DeepSeek-V4 engine only, and its backend
     registry (`c/expert_store_registry.h`) opens stores with DeepSeek-V4 types; Inkling and MiMo keep their expert caches
     inside their engines (`slot_acquire`, `slot_fill`). Its engine-neutral layer-segment runtime (`c/segment_runtime.h`,
     ABI v1, explicit registration, portable C) has an Inkling adapter (CPU) among seven. The engines build with gcc and
     POSIX (MSYS2 on Windows); the headers, the segment runtime and its conformance fixtures are portable C.
   - *DS4* (MIT; Apache-2.0 parts under `third_party/`) runs DeepSeek V4, GLM 5.x and Qwen3.8 Flash Next, not Inkling or
     MiMo; its SSD streaming and expert cache live inside its engine. Patterns only (hits protected before eviction, staged
     direct reads, cancellable readers), already reused in decision 0012.
   - *Weightsift* keeps its storage path: `weightsift-io` (plans equal to Python's, direct reads on a worker pool, a host
     cache under a strict budget with leases and load-once, accounting) is the storage backend; nothing in Colibrì or DS4
     replaces it with a demonstrated advantage.
3. **One integration path: Weightsift's storage under Colibrì's execution, through a versioned C ABI.**
   - The weight storage backend is Rust: a model-neutral expert store over `weightsift-io` (experts described by a
     catalog that a model adapter writes from checkpoint headers; leases that pin host-cache rows; prefetch; statistics),
     exposed through Weightsift's own versioned C ABI.
   - Colibrì's `ColiExpertStoreOps` lease contract is served by a thin C shim over that ABI, compiled against Colibrì's
     pinned header; Colibrì's segment runtime ABI v1 is the execution seam Rust drives. Engines stay Colibrì's.
   - For Inkling the minimal adapter is a patch of `inkling.c` replacing `slot_fill`'s reads and the slot LRU with store
     leases; it is an upstream modification of a model engine, needs a gcc toolchain and the checkpoint to validate, and
     is the next acceptance gate, not part of this phase.
   - Rejected for this phase: forking or vendoring Colibrì or DS4; a Rust rewrite of an engine; Colibrì's DeepSeek-V4
     registry as the integration point (another model, engine-specific open signature); DS4 for the flagships (no
     support); Python in the loop of a C engine's expert loads.
4. **Upstream sources are pinned, fetched and never committed.** A pin file records each upstream's URL, exact commit,
   license, the files used and their sha256; a fetch script checks them out into a gitignored directory and verifies them;
   builds that need them skip cleanly without them; updating a pin is a reviewed change of that file.
5. **ApexQuant behind a licensing gate.** No license was found in the repository, its package metadata, its sources or a
   package index; its code is therefore not copied, vendored, run in Weightsift or made a dependency until the authors
   grant a license. Work is limited to independent analysis of the paper and of the published design, and to interface
   design; real-weight measurements need flagship tensors that are neither local nor approved.
6. **The Quality Budget and the benchmarking contract are specifications with machine-readable schemas** (configuration,
   results, baseline comparison), not results: certification and the Colibrì baseline need the checkpoint and its
   engine, which this phase does not have.

7. **What was built** (the brief's Stage B, model-neutral throughout):
   - **Leases in `weightsift-io`.** `Engine::acquire` holds rows in the host cache, unmoved, for a caller that reads them
     in place. Leases and prefetches are waited for; a transfer's load is not (`Loader`). No room, or a transfer's row,
     fails at once and holds nothing.
   - **`weightsift-store`** (`native/store`). A catalog of experts written by a model adapter from checkpoint headers; an
     expert store with all-or-nothing leases, advisory prefetch, statistics and staging allocated at open. Its C ABI v1
     (`include/weightsift_store.h`): handles in tables, versioned structs, caught panics, close refused while leases
     live.
   - **`weightsift-runtime`.** The memory planner, and the `ExecutionBackend` interface (Colibrì's segment ABI's shape).
   - **`weightsift-colibri`.** A C shim serving Colibrì's lease contract over the store, and Colibrì's segment runtime as
     an `ExecutionBackend`. Built only from the verified pinned checkout.
   - **Python.** `awpmi.hardware` (detection, never assumption), `awpmi.models.catalog`, `awpmi.quality`,
     `awpmi.baseline`, and the benchmark and research drivers.

## Results

Full report: `history/2026-10-12-weightsift-phase7a-report.md`. Raw data: `experiments/phase7a/` (`store-moonlight`,
`store-moonlight-run2`, `plan-inkling`); ApexQuant: `research/apexquant/` (`feasibility.json`, `headers/`).

- **Tests.** 74 Rust tests pass; `fmt` and `clippy -D warnings` are clean. They cover:
  - leases in the core: 7 integration tests and 2 cache tests;
  - the store: 18, including a C client compiled by MSVC and the header's layout checked against Rust;
  - the planner: 3;
  - the bridge: 6, including Colibrì's own conformance checks for its 10 fixture families, all of them driven from Rust,
    and its lease contract through its inline helpers over Weightsift's store.

  The concurrent tests were stable over repeats. Mutating the lease guard made its tests fail. The full Python suite
  passes (905 passed, 87 skipped as before), with the new Quality Budget, baseline, catalog and layering tests.
- **The vertical slice on real bytes** (Moonlight-16B's experts, not a flagship).
  - 3.02 GB/s in batched leases, 2.20–2.51 GB/s one 17.3 MB expert at a time; cold lease p99 8.8 ms.
  - Warm leases take 1–3 µs and copy no bytes.
  - Every one of 512 rows equals the reference Python store's.
  - Physical bytes are exactly the rows' 4 KiB blocks. Two runs, on frozen trees.
- **The flagships, measured from headers only.**
  - MiMo-V2.6 Pro's experts are MXFP4: 4.25 bits per weight, 20.05 MB per expert, 11.07 GB per decode token cold.
  - Colibrì's int4 container of Inkling: 4.0069 bits per weight, 28.36 MB per expert, 10.89 GB per token.
  - This machine, detected: 31.9 GiB RAM, RTX 4060 Ti 8 GB, PCIe link at most Gen 3 x8.
- **ApexQuant: BLOCKED.** No license; no approved flagship tensors. Its four-bit class saves 4.4% of MXFP4's bytes and
  costs 1.4% against Inkling's int4 container (an I/O-bound bound of 1.046 at best). Only its two-bit prefix meets the
  20% screening target, at an unmeasured quality.
- **Specifications, not results.** The Quality Budget (its datasets' revisions are unpinned, so comparisons are
  PENDING) and the Colibrì baseline (blocked: no gcc-built engine; the checkpoint's download needs approval).
- **Not shown:** any performance advantage of the Rust store over Colibrì's own loader; no flagship was run.

## Rejected

- *Forking Colibrì or DS4, or committing their sources*: the brief forbids it; pinned fetches serve every build need.
- *A Rust rewrite of Colibrì's Inkling or MiMo engine*: a non-goal; their kernels, KV handling and oracles are reused.
- *Colibrì's DeepSeek-V4 store registry as the flagship seam*: it opens stores with DeepSeek-V4 engine types and is not
  used by Inkling or MiMo.
- *DS4 components for the flagships*: no Inkling or MiMo support; its streaming is internal to its engine.
- *Running or copying ApexQuant's code*: no license.
- *Raw pointers as C handles for stores and leases*: a stale or doubled release would be undefined behaviour. Handles in
  tables make it an error code.
- *Leases that wait for room, or for a transfer job's load*: either can close a cycle with the caller (the 6B deadlock's
  lesson). They fail at once (`Unavailable`), and the caller releases and retries.
- *Copying leased bytes into a lease-owned buffer*: an extra copy of every expert. The store's cache holds each part in
  one piece (its block is the largest part), and leases read it in place.
- *Fetching upstream sources from `build.rs`*: a network side effect in a build. A separate, verifiable fetch writes a
  marker, and builds re-hash.
- *Building Colibrì's model engines with MSVC*: they need gcc and POSIX headers. Only Colibrì's portable C is built here.
