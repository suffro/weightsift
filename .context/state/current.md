# Current State

**Project name: Weightsift.** Use this name consistently in documentation, code comments, and filenames.

**CLI:** `weightsift` is the command; `wsift` is its equivalent shorthand. Both expose
the `pack` command (`lm-head`, `experts`, `expert-index`, `encoded-experts`) and, since Phase 7A, `quality` (`check`,
`compare`). The Python package is `awpmi`.

## Latest: Phase 7A (the independent flagship runtime's foundation) is complete as a foundation (2026-10-12)

**Answer: the foundation is built and tested; no flagship has run, and ApexQuant is BLOCKED.**

- Report: `history/2026-10-12-weightsift-phase7a-report.md`. Decision: 0014. Brief: `state/phase_7A.md` (the user's of
  2026-10-12).
- **Done:** every implementation gate of the brief.
- **Waiting:** every flagship gate, which the brief makes subsequent. It needs a checkpoint (its download needs the
  user's approval) and a gcc toolchain for Colibrì's engines.

- **Direction (decision 0014).**
  - An independent project with a Rust systems core. Colibrì and DS4 are reused through pinned sources and narrow C
    ABIs, never forked or vendored.
  - Near-original quality (a Quality Budget) for new quantized modes. Every exact path and bit-exact test is kept.
- **Stage A: the reuse audit, and one path.**
  - Colibrì (Apache-2.0) runs both flagships. DS4 (MIT) runs neither.
  - Colibrì's `expert_store.h` lease contract serves DeepSeek-V4 only.
  - The chosen path: Weightsift's Rust store under Colibrì's engines, through Weightsift's C ABI v1 and a shim to that
    contract, with Colibrì's segment runtime as the execution seam. For Inkling, a patch of its `slot_fill` remains the
    next gate.
- **Stage B: built.**
  - Leases in `weightsift-io`.
  - `weightsift-store`: catalog, leases, prefetch, statistics, C ABI with header and a C client test.
  - `weightsift-runtime`: the memory planner and the execution interface.
  - `weightsift-colibri`: the lease-contract shim and the segment runtime from Rust, on Colibrì's pinned, fetched,
    hash-verified sources.
  - 74 Rust tests, including Colibrì's own conformance checks and its lease contract over Weightsift's store.
  - On Moonlight's real experts (not a flagship): 3.02 GB/s batched leases, warm leases in microseconds, every row equal
    to the reference.
- **Stage C: ApexQuant BLOCKED.** No license; no approved flagship tensors. Screening from the flagships' headers: its
  four-bit class saves 4.4% against MiMo's MXFP4 and costs 1.4% against Inkling's int4 container. Only its two-bit
  prefix meets the 20% target, at an unknown quality.
- **Specifications.** The Quality Budget (`awpmi.quality`; dataset revisions not yet pinned) and the Colibrì baseline
  (`configs/phase7a-baseline.yaml`; not run).

## Previous focus: Phase 6B (GPU transfers and decode execution) is complete (2026-10-09)

**Answer: 408.9 ms per decode token, 0.507 of Phase 6A's 807, bit for bit equal to the reference.** The report is
`history/2026-10-09-weightsift-phase6b-report.md`; the decision is 0013. The brief was the user's of 2026-10-09
(committed in 4a70869, since replaced by the Phase 7A brief).

- **Baseline first**: Phase 6A's best configuration reproduced on its own tree (810.8 and 806.9 ms; 5,195 launches per
  token, 524 ms of copies, the GPU idle 39–41%).
- **6B1, the host cache's submit cost** (native core): rows held in blocks of one size reused across row sizes (the pool
  inside the budget), readers woken one at a time instead of all per task; 100 → about 1 ms per decode token; BF16
  host-cache decode 0.964 of the 6A tree's. A latent 6A deadlock (two jobs waiting on each other's loads) fixed: a request
  reads a row another request is loading itself, unadmitted.
- **6B2, experts encoded for the GPU**: nvCOMP decodes Phase 5C's zstd frames exactly but too slowly with the plane merge;
  nvCOMP's rANS over raw BF16 rows (1 MiB chunks) stores 0.6747 and decodes a call in 0.7 ms. The encoded pack
  (`weightsift pack encoded-experts`, 19.42 GB) is read and cached by the native core as bytes and decoded on the GPU into
  the experts' buffers: 610.8 against 787.2 ms at the same 12 GB host cache (0.776); every row audited before inference.
- **6B3-A, a device tier**: `SlotCache`, fixed slots per stored row size (1.9 GB, LRU, prefill admission frozen) under the
  6 GB cap; it pays only from one token's working set (an LRU cliff at 1.82 GB): 493.7 against 610.8 ms (0.808).
- **6B3-B, native copy issuing**: measured (−1.3%, then nothing with the fills off), not built.
- **6B4, decode graphs**: `DecodeGraphs` replays each DeepSeek-V3 layer's static decode pieces from two CUDA Graphs (54 in
  all, 65–68 MB), the KV cache, attention and routed experts eager between them; and PyTorch's NaN fill of uninitialized
  memory (deterministic mode's debugging aid: 270 fill kernels per token) is off per configuration: 408.9 against 493.7
  ms (0.828); 2,344 kernel + 54 graph launches per token.
- **Correctness**: 1,116 steps per run (10 configurations) equal to the reference in every digest, two runs with
  identical digests, I/O parity, both cache tiers equal to their replays, the encoded pack's 3,328 rows equal.
- **What bounds a token now**: the bytes still copied (1.24 GB, 198 ms), the host cache's copies and the drive behind
  them (124 ms of waits), and the eager remainder of the transformer (about 170 ms of the main thread).
- **§ 6C (prepared, not started)**: shape-dependent kernels (kept by graph capture), workspace-dependent cuBLAS choices,
  reduction orders, where PyTorch changes execution without changing values, and the candidates for an explicitly
  reproducible native reference are in the report's § 12.

## Earlier

**Phase 6A (the native runtime foundation) is complete (2026-10-08)** (report
`history/2026-10-08-weightsift-phase6a-report.md`, decision 0012): a small, model-agnostic Rust I/O core (`native/`)
behind the Python storage contract, with a host-RAM expert cache; 807 ms per decode token against 1,157 ms for the best
Python configuration, bit for bit equal to the independent reference.

Phases 1A, 1B, 1C, 2, 3, 4A, 4B, 5A, 5A2, 5C, 6A and 6B are complete, and 7A as a foundation; their reports are in
`history/`.

- Phase 4B (decision 0008) runs Moonlight out of VRAM and host RAM, bit for bit equal to an independent reference.
- Phase 6A (decision 0012) runs it on the native core; Phase 6B (decision 0013) moves encoded experts to the GPU and
  graphs the decode step.
- Phase 7A (decision 0014) turns the project toward flagships, with a Rust storage backend under Colibrì's engines.

## Recent relevant changes

- **`native/core`** (`cache.rs`, `engine.rs`, `error.rs`): leases.
  - `Engine::acquire`, which returns `Leased`: entries plus how each was served.
  - `Loader` (who loads a row decides who may wait for it).
  - `Entry::contiguous`, `Prefetch::is_done`, `Error::Unavailable`.
  - 7 engine and 2 cache tests.
- **`native/store`** (new, `weightsift-store`):
  - the catalog, the store and its C ABI v1 (`include/weightsift_store.h`);
  - a C client test (`tests/c/abi_client.c`);
  - an example on real bytes (`examples/store_check.rs`).
- **`native/runtime`** (new): the planner, the execution interface, and an example that plans from JSON inputs.
- **`native/colibri`** (new): the shim (`csrc/`) and the segment runtime binding; its tests on Colibrì's own code.
- **Upstream sources.** `native/upstream.json` (the pins) and `native/fetch_upstream.py`; `native/third_party/` is
  gitignored.
- **`native/README.md`**: the crates, the upstream policy, the C ABI, the new `unsafe` boundaries.
- **`src/awpmi`**:
  - `hardware.py`, `baseline.py`, `quality/` and `models/catalog.py`, all new;
  - `cli.py` (`quality check|compare`);
  - `tracing.py` (`native_tree_sha256` excludes `native/third_party/`).
- **`benchmarks`**: `colibri_baseline.py` and `store_check.py`, new.
- **`configs`**: `phase7a-baseline.yaml`, `phase7a-baseline-prompts.txt`, `quality-budget.yaml` and
  `quality-results.schema.json`, all new.
- **`research/apexquant/`** (new): the licensing record, the study and the screening, from headers only.
- **`experiments/phase7a/`**: the store on Moonlight (two runs) and the planner on Inkling.
- **Tests**: `test_quality.py`, `test_colibri_baseline.py` and `test_expert_catalog.py`, all new. `test_layering.py`
  now scans the new crates and their C, and refuses `inkling` and `mimo` in them.
- **Decision 0014** is new, and the implementation guide has § 24.

## Next

The next milestone needs the user's go-ahead. Recommended (report § 10): **Inkling's engine reading its experts through
Weightsift's store, token-exact on Colibrì's tiny Inkling oracle.** This needs a gcc toolchain but no download:

1. Build Colibrì's `inkling` at the pinned commit.
2. Write a reviewed patch of `slot_fill` and the slot LRU through `ws_colibri_store_open`, applied at fetch time.
3. Require identical tokens with and without the store, under eviction.

Then, with the download approved, the Colibrì baseline of `configs/phase7a-baseline.yaml` on the real 975B. Only after
that, any comparison of the store with Colibrì's own loader. The Phase 6C items (the BF16 reference's semantics, more of
the decode step in graphs, more experts per device byte) stay recorded in decision 0013 and the Phase 6B report.

Open decisions for the user:

- Approval (or not) to download a flagship artifact: Inkling's int4 container is 514 GB, MiMo-V2.6 Pro 563.6 GB without
  its unused files. Alternatively, a bounded sample of expert tensors by range requests.
- A gcc toolchain on this machine (MSYS2 MinGW-w64), or a Linux host, to build Colibrì's engines.
- ApexQuant: a license from its authors before any use of its code.
- The Quality Budget's datasets: confirm and pin their revisions.
- Weightsift's own license (the repository has none).
- Still open from before: the rounding model for certificates (RN-even, from Phase 2); the reference for FP8 experts;
  Phase 2 on a larger model.

## Blockers

- **No flagship checkpoint is local**, and downloading one needs the user's approval. Every flagship measurement waits
  for it: the Colibrì baseline, quality, ApexQuant on real experts.
- **Colibrì's model engines need gcc and POSIX headers** (MSYS2 on Windows). This machine has MSVC only, so only
  Colibrì's portable C builds here: its headers, segment runtime and conformance fixtures.
- **ApexQuant's code has no license.**
