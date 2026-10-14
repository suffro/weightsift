# Weightsift Phase 7A report — an independent flagship runtime's foundation; ApexQuant feasibility

Date: 2026-10-12 · Follows: `history/2026-10-09-weightsift-phase6b-report.md` (Phase 6B) ·
Decision: `decisions/0014-phase7a-independent-flagship-runtime.md` · Brief: `state/phase_7A.md` ·
Status: **the foundation is implemented and tested; no flagship was run.** Every flagship measurement waits for a
checkpoint whose download needs the user's approval, and for a gcc toolchain to build Colibrì's engines. **ApexQuant:
BLOCKED** (no license; no approved flagship tensors).

## Outcome in brief

| Question | Answer |
| --- | --- |
| Reuse architecture (Stage A) | Colibrì (Apache-2.0) has both flagships' engines. Its `expert_store.h` lease contract is wired only into its DeepSeek-V4 engine; Inkling and MiMo cache experts inside their engines. Its engine-neutral segment runtime (ABI v1, portable C) has an Inkling adapter. DS4 (MIT) supports neither flagship. **Chosen path:** Weightsift's Rust storage under Colibrì's execution, through Weightsift's own versioned C ABI and a thin shim to Colibrì's lease contract, with Colibrì's segment runtime as the execution seam. For Inkling, the minimal adapter is a patch of `slot_fill` and the slot LRU: the next gate, not done |
| Implemented (Stage B) | **Core:** leases in `weightsift-io` (`Engine::acquire`, rows held unmoved in the host cache). **`weightsift-store`:** an expert catalog, an expert store and a C ABI v1 with a header. **`weightsift-runtime`:** the memory planner and the execution backend interface. **`weightsift-colibri`:** a C shim serving Colibrì's lease contract, plus Colibrì's segment runtime as an `ExecutionBackend`. **Build:** pinned, fetched, hash-verified upstream sources. **Python:** hardware detection, expert catalogs from index packs, the Quality Budget, the baseline contract |
| Tests | **Rust:** 74, all passing. Lease lifecycle, ownership, budgets, concurrency, errors; the C ABI from C and from Rust; Colibrì's own conformance checks; its lease contract through its inline helpers over Weightsift's store. **Python:** the full suite; new Quality Budget, baseline, catalog and layering tests. **Guard check:** the lease guard was mutated once and both of its tests failed |
| Vertical slice on real bytes (not a flagship) | Moonlight-16B's experts through the Rust store, verified row by row against the Python store. **Throughput:** 3.02 GB/s batched; 2.20–2.51 GB/s leasing one 17.3 MB expert at a time. **Latency:** p99 8.8 ms cold; 1–3 µs per warm lease, with no bytes copied. **Accounting:** physical bytes exactly the rows' 4 KiB blocks. Two runs on frozen trees |
| Measured, flagships (metadata only) | **MiMo Pro:** experts are MXFP4, exactly 4.25 bits per weight, 20.05 MB per expert, 11.07 GB per decode token cold. **Inkling:** Colibrì's int4 container is 4.0069 bits per weight, 28.36 MB per expert, 10.89 GB per token cold. **Both:** every expert row splits into whole 512-groups. **This machine:** 31.9 GiB RAM, RTX 4060 Ti 8 GB, PCIe link at most Gen 3 x8 |
| ApexQuant | **BLOCKED:** no license anywhere; no approved flagship tensors. **Screening (bytes):** its 4-bit class saves 4.4% against MXFP4 and costs 1.4% against Inkling's int4 container, so its optimistic I/O-bound gain is negligible. Only its 2-bit prefix meets the 20% target, at a quality the paper reports as poor on dense 7B models and nobody has measured on MoE |
| Quality Budget | **Built:** a specification (six categories, held-out benchmarks, configurable thresholds), a results JSON Schema and parser that refuse invented scores, and comparison utilities (paired bootstrap; Newcombe intervals checked against the literature; per-category verdicts). **Not built:** certification. No dataset revision is pinned yet, so every comparison is PENDING |
| Benchmarking contract | A pinned specification of the Colibrì baseline (engine commit, checkpoint and artifact revisions, configurations, cold and warm rotating prompts, metrics, rules), a dry-run planner, and a parser of Inkling's printed metrics. **Not run** |

## 1. The brief, and what this phase could reach

The brief turns Weightsift toward MoE flagships of about a trillion parameters on constrained hardware, as an
independent project with a Rust-native systems core that reuses proven engines (Colibrì first, DS4 where better)
without forking them. It asks for:

1. a reuse and licensing audit;
2. one integration path, implemented as a small real vertical slice with deterministic tests;
3. a bounded ApexQuant feasibility track behind a licensing gate;
4. a Quality Budget specification and schema;
5. a reproducible baseline benchmark specification;
6. decisions and context.

It forbids rewrites, new quantization algorithms, unapproved downloads and 1T claims without a 1T run. Stops must be
reported as stops.

Two facts bound the phase from the start:

- **No flagship checkpoint is on this machine.** Inkling's int4 container is 514 GB; MiMo-V2.6 Pro is 563.6 GB without
  its unused files. Downloading either needs the user's approval, which this phase did not ask for: the brief's own
  rule.
- **Colibrì's model engines need gcc and POSIX headers** (MSYS2's MinGW-w64 on Windows). This machine has MSVC only, so
  only Colibrì's portable C was built here: the expert-store header, the segment runtime and its conformance fixtures.

Every flagship workstream that needs either fact (the baseline, quality, the engine adapter, ApexQuant on real weights)
stopped at its specification. Everything else was built and tested.

## 2. Stage A: the reuse and licensing audit

Pinned on 2026-10-12: Colibrì `bf2442915d6e3dd4cdfd2eb9c2a3d2aa44a25850` (2026-10-06), DS4
`fc80bd695da76ee14bcb86820bf900a5cdfda806` (2026-10-07), ApexQuant `42b6efd532c5c1cf4df730ffd969d1bfbe8016c5`
(2026-10-05).

**Licenses:**

- **Colibrì:** Apache-2.0; its NOTICE file: Copyright 2026 Vincenzo Fornaro. Its third-party notices list portions
  adapted from DS4 (MIT), kernels adapted from Swiftlet (Apache-2.0), and DeepGEMM and CUTLASS fetched for one CUDA
  flavour, not vendored.
- **DS4:** MIT (the ds4.c authors, the ggml authors, DeepSeek), with an Apache-2.0 notice for a third-party directory.
- **The models:** Inkling Apache-2.0, MiMo-V2.6 MIT (Hugging Face metadata).
- **Weightsift itself:** no license file. That is the user's choice to make; nothing here depends on it.

| Component | Weightsift now | Colibrì (pinned) | DS4 (pinned) | Reuse, adapter, effort |
| --- | --- | --- | --- | --- |
| Model execution | PyTorch and transformers (BF16 models up to 16B); no flagship | `c/inkling.c` (CPU; CUDA for BF16 residents; Vulkan expert tier), `c/mimo.c` (CPU; Vulkan); token-exact against Hugging Face on tiny checkpoints; MiMo also against the real checkpoint | DeepSeek V4/V4.1, GLM 5.x, Qwen3.8 Flash Next; not Inkling or MiMo | **Colibrì, unmodified.** Library use through its segment runtime ABI (adapter objects built with `-DINKLING_NO_MAIN`); needs gcc. A Rust rewrite is a non-goal |
| MoE routing, expert compute | transformers' experts module, compact and chunked calls | inside the engines (Inkling: sigmoid router, experts in rounds of `cap`) | — | Colibrì, inside its engines |
| Quantized CPU kernels | none | int4, int8, MXFP4 and FP8 kernels (`idot.h`, `qgemv.h`, `native_quant*.h`; AVX2/VNNI) | GGUF-style kernels in `ds4.c` | Colibrì's; nothing to rewrite |
| CUDA, Vulkan, Metal | PyTorch, cuBLAS, nvCOMP | `backend_cuda_ink.cu`, `vk_tier.c`, Metal | `ds4_cuda.cu`, Metal, ROCm | Colibrì's, in a later phase |
| Expert storage, loading | `weightsift-io` (plans equal to Python's), index packs from headers | per engine: Inkling's `slot_fill` reads per expert; MiMo one `O_DIRECT` read per expert; **`expert_store.h` (lookup, release, prefetch, stats, destroy; a lease contract) used by DeepSeek-V4 only**, its registry typed on DeepSeek-V4 | inside its engine (GGUF) | **Weightsift's store under the engine:** its C ABI plus a shim to Colibrì's contract (built here). Inkling needs a patch of `slot_acquire`/`slot_fill` (an upstream modification of a 4,480-line engine; about a hundred lines, by inspection): the next gate |
| SSD streaming, direct I/O | worker pool of positioned direct reads, at the drive's rate (decision 0006) | `O_DIRECT`, per-layer read threads, io_uring | staged direct reads, cancellable reader | Weightsift's. Already measured at the drive's rate; not compared with Colibrì's (no engine here) |
| RAM cache, eviction | host cache: strict byte budget across all experts, LRU, leases, load-once, block pool | per-layer slots (`cap` experts per layer), LRU with usage-history pins | expert cache with LRU stamps, hits protected | **Weightsift's** (a budget in bytes across layers, leases that pin). Colibrì's pins are a policy idea for later |
| GPU expert cache | `SlotCache` (PyTorch) | Vulkan tier; DeepSeek-V4 CUDA mirror | CUDA device cache | none in this phase |
| Prefetch, scheduling | exact chunk prefetch | router look-ahead (DeepSeek/GLM engine), parallel fills, pins | look-ahead loads, hotlists | the store's advisory prefetch, which Colibrì's `prefetch` op maps onto |
| KV cache | transformers' cache | per engine (rings for sliding layers, prefix reuse, snapshots) | `ds4_kvstore.c` | Colibrì's |
| Formats, quantization | safetensors index; refinement and encoded packs | Inkling's int4 container and int4-gs64 dense set; MXFP4 and FP8 as released; qpack; rANS | its own GGUF quants | **read as published:** a catalog from headers; no new format |
| Instrumentation, benchmarking | environment records, digests, I/O audits against OS counters, profiles | routing telemetry; a five-rule benchmarking protocol and `tools/datapoint.py` | `ds4_bench`, speed bench | **Weightsift's contract adopting Colibrì's protocol** (`configs/phase7a-baseline.yaml`) |

**Where Rust could help, and what is shown:**

- **Shown:** safety at the boundary (handles instead of pointers, leases the cache honours); one byte budget shared with a
  planner across all layers instead of slots per layer; the 6A core's measured I/O.
- **Not shown:** any speed advantage over Colibrì's own loader. That needs Colibrì's engine running, which is the next
  gate; the brief requires it to be demonstrated, not assumed.

**Integration path chosen, and why** (decision 0014, point 3):

- It reuses Colibrì's engines whole (execution) and Weightsift's tested storage (I/O, budgets, leases, accounting).
- It meets at a contract Colibrì already defines (`ColiExpertStoreOps`), so a patched engine calls something Colibrì
  already knows how to call.
- It keeps model code out of Weightsift's Rust: the catalog carries the layout as data.

**Rejected:**

- *Forking or vendoring* (forbidden).
- *A Rust engine* (a non-goal).
- *Colibrì's DeepSeek-V4 registry* (another model; engine-specific types).
- *DS4 for the flagships* (no support).
- *Python in a C engine's expert path*: GIL and latency, for nothing.

## 3. The target architecture

Three parts, kept separate (`truth/architecture.md`):

```text
runtime planner          weightsift-runtime   one detected budget: execution's needs, then the store's staging and cache
weight storage backend   weightsift-store     catalog -> leases from weightsift-io's host cache -> C ABI v1 (+ Colibrì shim)
model execution backend  Colibrì's engines    segment runtime ABI v1 (ExecutionBackend); a patched engine calls the store
```

Weightsift's Rust names no model (the layering test now scans the store, the runtime, the bridge, their C header, shim
and C tests, and also refuses `inkling` and `mimo`). A model adapter writes the catalog from the checkpoint's headers
(`awpmi.models.catalog`). The existing PyTorch runtime, its Python storage backend and every bit-exact test are
unchanged and remain the fallback.

## 4. What was implemented

### 4.1 Leases in the core (`native/core`)

`Engine::acquire(requests, slots) -> Leased` blocks until every requested row is resident in the host cache. It returns
one `Arc<Entry>` per row and how each was served (hit, waited, loaded).

While the caller holds an entry:

- its row is never evicted or cleared;
- its bytes do not move or change, and `Entry::contiguous` reads them in place (the store sets the cache's block to its
  largest part, so a part is one piece).

How a call proceeds:

- Misses are loaded like a prefetch's (pieces that release their own slots), at foreground priority, through staging
  the caller owns.
- Rows a prefetch or another lease is loading are waited for; their loads need no caller, so waiting cannot close a
  cycle.
- A row a transfer job is loading makes the call fail `Unavailable` instead of waiting: that load moves only as its
  caller consumes pieces, so waiting for it could close a cycle (the 6B deadlock's lesson).
- So does a row that cannot be admitted without evicting a leased one; the call fails at once and holds nothing.
- It returns only once none of its tasks touches the caller's slots.

`Loader { Transfer, Prefetch, Lease }` replaces the cache's prefetch flag; transfers and prefetches behave exactly as
before. The 22 integration tests include 7 new ones:

- leased bytes equal the files' and never move across evictions;
- leases are never evicted, and a full cache refuses;
- each row is read once, and counted like any request;
- concurrent leases load each row once (stable over 30 repeats);
- a lease racing a prefetch waits for it;
- a transfer's row is not waited for;
- errors and close hold nothing.

Mutating the guard (letting leased entries be evicted) made both lease-guard tests fail, in the core and in the store.

### 4.2 The expert store (`native/store`)

- **Catalog** (`weightsift-expert-catalog` v1, JSON). Files (sizes checked on open), segments (plain or composed, as
  Weightsift's layout), and layers: expert e's part p is row `first_row + e` of a segment. Views name byte ranges of
  parts with a shape and a storage format the store never interprets. Validation is strict: unknown fields are refused,
  and views must lie within their parts.
- **`ExpertStore`.**
  - `acquire` and `acquire_many` are all or nothing, one plan per segment for what is missing.
  - `Lease` is RAII and exposes parts and views in place.
  - `prefetch` is advisory; refused with `Unavailable` while `max_prefetches` are in flight.
  - Statistics: expert hits, waits and misses; leases; prefetches; the core's I/O and cache counters; staging.
  - Staging is allocated at open, as sets taken whole before any lookup, so a load in flight always has its staging.
  - A lease may outlive the store's closing; its rows stay valid.
- **C ABI v1** (`include/weightsift_store.h`; `ffi.rs`; built as `rlib`, `cdylib` and `staticlib`).
  - 64-bit store and lease handles in registries, never reused. A stale, foreign, doubled or unknown handle is a status,
    not undefined behaviour.
  - A read–write gate makes close linearizable with acquires. Close is refused while leases live.
  - Versioned structs (`struct_size`: the caller's size is honoured both ways).
  - Panics are caught at every entry point. Errors come back in caller buffers.
  - Ordinary functions: open, close, layers, layer info, view index and info, acquire, acquire many, release, lease key,
    part and view, prefetch, stats.
  - Compiling the header with MSVC caught a real defect: `ws_store_stats` was both a typedef and a function; the
    function became `ws_store_get_stats`.
- **Tests (18).**
  - Every byte equals the files' and the reference read path's (the core's uncached `read_rows`).
  - A full cache refuses and holds nothing.
  - `acquire_many` is all or nothing.
  - Eight threads stay within the budget.
  - Prefetched experts are waited for or hit.
  - Bad catalogs and options are refused; a read error holds no lease.
  - The C header's layout equals the Rust structs'.
  - A C client compiled by MSVC runs the whole lifecycle and reads the files' bytes.
  - Misuse returns error codes.
  - Eight threads acquire and release through the ABI.

### 4.3 The runtime planner and execution interface (`native/runtime`)

- **`plan(hardware, footprint, policy) -> MemoryPlan`.**
  - Inputs: the machine as detected, and the backend's resident weights, state and workspace on host and device.
  - Out of the host budget (available memory less a reserve): those needs first, then the store's staging (its sets
    plus prefetches in flight), then the expert cache.
  - Result: named reservations that never exceed the budget, the store's options, the cache's experts, and its ratio to
    one token's working set and to all experts, with a note.
  - Refuses infeasible budgets, devices that are missing or too small, and a cache that cannot hold one expert.
- **`ExecutionBackend` and `ExecutionSession`** (open a layer range, capabilities, sessions, run, snapshot, restore):
  the shape of Colibrì's segment ABI. Implemented by the bridge.

### 4.4 The bridge to Colibrì (`native/colibri`), built from pinned sources

- **Pins.** `native/upstream.json` pins Colibrì's commit and the sha256 of the nine files used (the repository's LF
  content: a Windows checkout with `autocrlf` would not match, which is why the fetch disables it).
  `native/fetch_upstream.py` makes a sparse, shallow checkout into the gitignored `native/third_party/`, verifies it,
  and writes a marker. A tampered file was detected; the re-fetch after it exposed and fixed read-only git objects on
  Windows.
- **Build.** `build.rs` re-hashes the files before compiling. Without them it warns, and the tests are ignored with the
  reason (exercised once).
- **The shim** (`csrc/ws_colibri_store.c`, Weightsift's code, compiled with strict warnings against Colibrì's header):
  - `ws_colibri_store_open` checks every layer's `gate`, `up` and `down` views against Colibrì's formats.
  - `lookup` leases and fills `ColiExpertView` in place; `release`, `prefetch`, `stats` and `destroy` follow Colibrì's
    contract.
  - Destroy with live leases frees nothing and reports it.
- **Segment runtime from Rust.** `#[repr(C)]` mirrors of `segment_runtime.h` (their layout checked against the C
  compiler's), and `SegmentEngine` and `SegmentSession` implementing the execution interface. Sessions borrow the
  engine. Snapshots go through callbacks.
- **Tests (6), on Colibrì's own code.**
  - The mirrors' layout.
  - Colibrì's conformance checks for every one of its ten fixture families (Inkling's and MiMo's among them).
  - Every fixture driven from Rust: exact continuation after snapshot and restore; isolated sessions; corrupt, short and
    foreign snapshots refused without changing the state; non-contiguous positions, wrong sizes and missing tokens
    refused.
  - Colibrì's lease contract, checked through its inline helpers on Weightsift's store over real files:
    - a garbage view is filled;
    - a failed lookup clears the view;
    - release clears it; double and zero releases are no-ops;
    - a copied view's stale lease is refused, not double-freed;
    - prefetch is accepted;
    - stats count;
    - destroy is refused with live leases and succeeds without.
  - Catalogs Colibrì cannot read are refused at open.

The fixtures prove the seams, nothing about a model.

### 4.5 Python

- `awpmi.hardware`: the machine as detected (memory from the OS; GPU and PCIe link from `nvidia-smi`; storage).
- `awpmi.models.catalog`: a catalog from an expert index pack (tested: it rebuilds the pack's rows exactly).
- `awpmi.quality`: the Quality Budget (§ 7).
- `awpmi.baseline` and `benchmarks/colibri_baseline.py`: the baseline contract (§ 8).
- `benchmarks/store_check.py` with `native/store/examples/store_check.rs`: the store on real bytes.
- `research/apexquant/`: § 6.
- `weightsift quality check|compare`.
- `native_tree_sha256` now excludes `native/third_party/`, which is pinned by `upstream.json` and hashed through it.

## 5. Measurements

Only measurements. Hardware: Windows 11, 6 logical CPUs (Intel family 6 model 158), 31.9 GiB RAM, RTX 4060 Ti 8 GB
(driver 610.62; PCIe link at most Gen 3 x8, detected by `nvidia-smi`, Gen 2 at the moment of detection), NVMe system
drive. Software: Rust 1.95.0, MSVC 14.44 (VS 2022 Build Tools), Python 3.13.

**The expert store on Moonlight-16B-A3B's routed experts** (`experiments/phase7a/store-moonlight`, `-run2`):

- Layers 1–4: 256 experts, 512 rows, 4.43 GB.
- The experts are BF16 rows of the published checkpoint, described by a catalog written from the Phase 4B index.
- Settings: direct I/O, 8 readers, a 6 GiB budget, 2 staging sets of 2 × 8 MiB.
- The same source and native trees in both runs. A microbenchmark of the store alone, not inference.

| Pass | Run 1 | Run 2 |
| --- | --- | --- |
| Cold, one expert at a time | 2.20 GB/s; lease p50 8.19 ms, p99 8.85, max 9.47 | 2.51 GB/s; p50 6.59 ms, p99 8.79, max 9.84 |
| Warm (every expert a hit, no byte copied) | p50 3 µs, p99 6 µs | p50 1.3 µs, p99 1.7 µs |
| Cold, one `acquire_many` per layer | 3.02 GB/s; 371–383 ms per 64 experts | 3.02 GB/s; 358–400 ms |
| Verification | 512 rows equal to the reference | 512 rows equal |

**Counters:**

- Physical bytes 4,431,282,176, exactly the rows' 4 KiB blocks (read amplification 1.0005).
- The drive was busy 1.42 s of each cold pass (3.1 GB/s while busy); the one-at-a-time pass leaves it idle between
  leases, which batching removes.
- Peak held 4.43 GB within the 6 GiB budget; zero bypasses and evictions.
- The reference digests are the Python store's (`FileBackedPageStore`, direct I/O), the path validated bit for bit since
  Phase 4A.

**The flagships' formats, from their safetensors headers** (no tensor data; `research/apexquant/headers`):

| Checkpoint (revision) | Expert format | Bits per weight | Bytes per expert |
| --- | --- | --- | --- |
| MiMo-V2.6 Pro (`6a7f202e`) | MXFP4 as released: U8 nibbles, one U8 scale per 32 | 4.2500 | 20,054,016 |
| Inkling (`828496ee`) | BF16 (`w13`, `w2` fused per layer) | 16 | 113,246,208 |
| Inkling, Colibrì's container (`3981e658`) | U8 nibbles, one F32 scale per row | 4.0069 | 28,360,704 |

From config and index:

- **Inkling:** 64 MoE layers (2–65), top 6, so 384 experts and 10.89 GB per decode token if none is cached.
- **MiMo Pro:** 69 MoE layers, top 8, so 552 experts and 11.07 GB.
- Colibrì's documentation gives 18.9 MiB per MiMo expert; the headers give 19.13 MiB.

**Unit and integration tests:**

- **Rust:** 74, passing. `cargo fmt` and `clippy -D warnings` are clean.
- **Python:** the full suite (`tests`, `research/apexquant/tests`) on the final trees: 905 passed, 87 skipped, in
  7 min 21 s. The skips are the existing CPU variants of the CUDA-only encoded-row tests, as before this phase. The
  suite run after the core's lease changes, before the new Python code, gave 883 passed.

## 6. ApexQuant (Stage C)

The full study is `research/apexquant/README.md`.

**Verdict: BLOCKED.**

- **Licensing gate.** No license in the repository at any level, in its package metadata, in its sources, or in GitHub's
  detection. Not on PyPI. The paper's arXiv license covers distributing the paper only. It also vendors a fork of
  gemlite's kernel. So: no copying, running or dependency. Analysis only.
- **Technical study, from the paper.**
  - Fresh Hadamard rotations per pass return each residual to a known spherical marginal.
  - Three stages (scalar, E8P, trellis L = 14).
  - Exact nested prefixes: a row's passes are stored back to back.
  - Storage P(2 + 16/512) bits per weight.
  - Triton GPU kernels that rotate the activation per pass; no CPU kernels.
  - Dense models up to 7–8B only. Two bits far from usable on several; the calibration pass does not transfer to
    language models; no MoE quality result.
- **Routed experts.** Structurally compatible: stacked expert tensors are handled; every flagship expert row splits into
  whole 512-groups (measured); a pass is a byte range a store can read.
- **Screening against the native formats (measured bytes).**
  - P = 2: MiMo 0.956 of MXFP4 (I/O-bound bound 1.046); Inkling 1.014 of Colibrì's int4 container (slower).
  - P = 1: 0.48–0.51, which meets the 20% target on bytes only.
  - MXFP4 compounds: MiMo's own quantization error cannot be measured (no higher-precision release), and ApexQuant
    would add its own.
- **Measurements not possible:** error, conversion cost, kernels, quality. There is no license to run the code and no
  approved tensors.

Even unblocked, the four-bit class does not pay, so the track stops there. A larger experiment would first need a
license, then approval to sample real Inkling BF16 experts, then a per-pass error probe against Colibrì's int4
container; only if a P = 1 / P = 2 mix came close would a Quality Budget run follow.

## 7. The Quality Budget

`awpmi.quality`, `configs/quality-budget.yaml`, `configs/quality-results.schema.json`.

- **Specification.** Six categories (coding, mathematical reasoning, general reasoning, instruction following, long
  context, and tool use, which is optional). Each category has an absolute regression threshold and held-out benchmarks
  named at their published homes (HumanEval, MBPP sanitized, GSM8K, MATH-500, MMLU-Pro, GPQA diamond, IFEval,
  LongBench v2, BFCL), each with a minimum sample count. Baselines in order of strength: original, released-quantized,
  derived-quantized. No dataset was downloaded and **no revision is pinned**: `weightsift quality check` lists the nine
  and exits 1.
- **Results** (`weightsift-quality-results` v1).
  - Run metadata, subject, calibration data, and per-benchmark results: measured (n, score, optional items) or
    unavailable/skipped with a reason.
  - A result that is not measured cannot carry a number; the parser refuses it.
  - Items must average to the score.
- **Comparison.**
  - Regression = baseline − candidate.
  - A paired percentile bootstrap over common items (seeded), otherwise Newcombe's hybrid score interval for binary
    metrics. Wilson's interval and Newcombe's 1998 example (a) are reproduced.
  - Per benchmark: PASS, FAIL, INCONCLUSIVE, PENDING, INVALID or SKIPPED. INVALID covers calibration overlap and
    different data or items.
  - Categories take their worst benchmark; the whole takes the worst category; nothing is averaged.
  - Notes when the baseline is not the original checkpoint.
- **The thresholds are configuration, not statistics.** The tests show why: at 1,000 unpaired items per side and an
  accuracy of 0.7, the interval is about ±4 points and cannot certify a 2-point budget; the verdict says INCONCLUSIVE
  rather than PASS.

## 8. The benchmarking contract

`configs/phase7a-baseline.yaml`, `awpmi.baseline`, `benchmarks/colibri_baseline.py`.

- **Pins.** Colibrì's commit (the same pin as the build's), Inkling's original and int4-container revisions and sizes,
  the dense conversion tool, the geometry.
- **Plan.**
  - Configurations: `cpu-cap2`, `cpu-auto`; `PIN=off`, `USAGE_SAVE=0`, one OpenMP thread per core.
  - Rotating prompts: 10 diverse prompts, sha256 recorded, chat template, 64 decode tokens.
  - States and repeats: cold and warm, two repeats.
- **Required metrics.** Decode tokens/s, prefill, TTFT, SSD and host-to-device bytes per token, hit ratio, peak RAM and
  VRAM, utilization, expert-fill (I/O) time, outputs.
- **Rules.** Colibrì's five-rule benchmarking protocol applies in full. One variable at a time. Measured vs
  extrapolated. No microbenchmark reported as end to end. A Rust replacement is checked for correctness and ownership
  before throughput and tail latency.
- **Dry run.** Prints the eight processes, the detected machine and the blockers: no gcc-built engine; no snapshot (the
  514 GB download needs approval). The parser reads the formats Inkling's engine prints at the pinned commit. Executing
  the plan, with the OS's I/O counters, is the next milestone's.

**The planner on Inkling's inputs** (`experiments/phase7a/plan-inkling`). This is arithmetic, not measurement. Its
inputs: the machine as detected; the int4-gs64 dense set at 15.3 GB as documented by Colibrì; an assumed 1 GiB of state
and 0.5 GiB of workspace; expert bytes measured from the headers.

- **With the memory available at detection** (22.4 GB): a 3.22 GB cache, 113 experts, 0.30 of one token's working set.
  An LRU cache serves almost nothing there; about 3.6 s of I/O per token would follow at the store's measured rate
  (extrapolated).
- **On a dedicated machine** (a hypothesis): 12.97 GB, 457 experts, 1.19 tokens' working set, 2.8% of all experts. Its
  hit rate must be measured.

## 9. Remaining blockers

- **A flagship checkpoint:** Inkling's int4 container (514 GB at `3981e658`; `E:` has 1.37 TB free, detected), or
  MiMo-V2.6 Pro (563.6 GB). It needs the user's approval.
- **A gcc toolchain** (MSYS2 MinGW-w64 on this machine, or a Linux host) to build Colibrì's engines and their segment
  adapters.
- **The Inkling engine patch** (`slot_fill` and the slot LRU through a `ColiExpertStore`). Not started: it needs the
  toolchain, and it is an upstream modification to keep as a pinned patch file, applied at build time.
- **ApexQuant:** a license, then approval for a bounded sample of real experts.
- **The Quality Budget:** dataset revisions confirmed and pinned, and an engine and checkpoint to evaluate.
- **Not demonstrated:** any performance advantage of Weightsift's store over Colibrì's own loader. It needs the
  baseline.

## 10. Recommended next milestone

**Inkling's engine reading its experts through Weightsift's store, token-exact on Colibrì's tiny Inkling oracle.** This
needs gcc but no download. In order:

1. Install MSYS2's MinGW-w64 gcc (or use a Linux host) and build Colibrì's `inkling` at the pinned commit.
2. Write the patch: `slot_fill` and the slot LRU through `ws_colibri_store_open`. Keep it a reviewed `.patch` file next
   to the pin, applied by the fetch.
3. Have `tools/make_tiny_inkling.py` generate the tiny checkpoint and its catalog.
4. Require identical tokens to Hugging Face's with and without the store, under eviction, together with the store's
   byte accounting.

Then, with the user's approval of the download, the Colibrì baseline of `configs/phase7a-baseline.yaml` on the real
975B, and only then any comparison of the store against Colibrì's loader.
