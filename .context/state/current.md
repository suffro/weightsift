# Current State

**Project name: Weightsift.** Use this name consistently in documentation, code comments, and filenames.

**CLI:** `weightsift` is the command; `wsift` is its equivalent shorthand. Both expose
the `pack` command (`lm-head`, `experts`, `expert-index`, `encoded-experts`) and, since Phase 7A, `quality` (`check`,
`compare`). The Python package is `awpmi`.

## Latest: Phase 7B (an independent technical review against Colibrì) is complete (2026-10-15)

**Answer: Weightsift cannot outperform Colibrì as an engine. On this machine any exact gain, by anyone, is capped near
1.7× for Inkling's int4 experts. The credible advantage is measured quality per byte: making flagship quality
measurable, then choosing and placing representations under the Quality Budget, inside Colibrì's engines.**

- **Report:** `history/2026-10-15-weightsift-phase7b-review.md`.
- **Brief:** `state/phase_7B.md` (the user's of 2026-10-14). It forbade code and repository changes, so only `.context`
  changed; nothing was run or measured.
- **No decision record:** the review recommends, and the direction is the user's to choose.
- **Facts that change the picture** [verified in Colibrì at `bf244291` and `dev` `2fc033dc`, and on model cards]:
  - Colibrì's release v2.0.0 (2026-10-06) ships `inkling.exe` for Windows, built with Vulkan. The Colibrì baseline
    needs no gcc; only patching its engines does.
  - Colibrì's Inkling container (per-row int4) has "no quality evals yet beyond greedy smoke tests". The same recipe
    measured about 9 points below grouped scales on GLM. Colibrì's own benchmark uses 40 items per task.
  - The publisher's own Inkling NVFP4 release exists (about 592 GB). Inkling's config declares MTP layers, which
    Colibrì does not load.
  - Colibrì's Inkling engine is the simplest of its engines on I/O: token-major rounds of `cap` pairs, no prefetch, no
    overlap on the CPU path, no route records.
  - Colibrì's decode failure ledger rejects 19 directions on a full-resident 6× RTX 5090 host: shared expert bases,
    prefetch for resident decode, fused ANS decoding, MTP as a default, among others. The rejections do not transfer
    automatically to 2% residency.
- **The model** (§ 5 of the report, not measured):
  - Inkling on 32 GB with one 3.2 GB/s drive is disk-bound at about 0.3–0.4 tokens/s.
  - A cold long prompt costs about 145 s of reads.
  - Neither flagship runs here exactly against its release: both need a lossy dense conversion.
- **AWPMI:** certification inside experts is closed by information, not by the verifiers (§ 2.9). It reopens only
  through the one-day oracle E10.
- **Recommended month** (§ 13 C): quality per byte for routed experts on Moonlight, locally:
  - a teacher-forced KL scorer;
  - pinned benchmarks;
  - mass-weighted bit allocation against uniform at equal bytes, with pre-set continue and abandon gates.

## Previous focus: Phase 7A (the independent flagship runtime's foundation) is complete as a foundation (2026-10-12)

**Answer: the foundation is built and tested; no flagship has run, and ApexQuant is BLOCKED.**

- Report: `history/2026-10-12-weightsift-phase7a-report.md`. Decision: 0014. Brief: the user's of 2026-10-12 (committed
  in 54c31eb, since replaced by the Phase 7B brief).
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

## Earlier

**Phase 6B (GPU transfers and decode execution) is complete (2026-10-09)** (report
`history/2026-10-09-weightsift-phase6b-report.md`, decision 0013; brief committed in 4a70869): 408.9 ms per decode token,
0.507 of Phase 6A's 807, bit for bit equal to the reference, through encoded experts decoded on the GPU (nvCOMP rANS,
0.6747 of BF16), a fixed-slot device tier and CUDA Graphs of the static decode pieces. What bounded a token then: the
bytes still copied (198 ms), the host cache's waits (124 ms) and the eager remainder (about 170 ms). Phase 6C was
prepared (its § 12), not started.

**Phase 6A (the native runtime foundation) is complete (2026-10-08)** (report
`history/2026-10-08-weightsift-phase6a-report.md`, decision 0012): a small, model-agnostic Rust I/O core (`native/`)
behind the Python storage contract, with a host-RAM expert cache; 807 ms per decode token against 1,157 ms for the best
Python configuration, bit for bit equal to the independent reference.

Phases 1A, 1B, 1C, 2, 3, 4A, 4B, 5A, 5A2, 5C, 6A, 6B and 7B are complete, and 7A as a foundation; their reports are
in `history/`.

- Phase 4B (decision 0008) runs Moonlight out of VRAM and host RAM, bit for bit equal to an independent reference.
- Phase 6A (decision 0012) runs it on the native core; Phase 6B (decision 0013) moves encoded experts to the GPU and
  graphs the decode step.
- Phase 7A (decision 0014) turns the project toward flagships, with a Rust storage backend under Colibrì's engines.

## Recent relevant changes

**Phase 7B changed only `.context`:**

- the review report;
- `state/current.md` and `index.md`;
- the user's brief, committed;
- references to the replaced 7A brief, which now cite its commit.

**Phase 7A's code changes:**

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

The next milestone needs the user's go-ahead. The direction is the user's choice, among the 7B review's
recommendations.

**Immediate, with no download** (report § 11):

- I1: measure this machine's RAM bandwidth, the second drive and the PCIe topology, and mine Moonlight's routing traces
  for expert-mass skew, hit-rate curves at 1–5% residency and consecutive-token unions.
- I2: screen the exact levers: the CPU rANS decode rate, and prefill re-reads on Colibrì's tiny Inkling oracle.
- I3: shake down Colibrì's Windows release here.
- I4: pin the Quality Budget's datasets and add a multilingual category.

**The recommended month** (§ 13 C): quality per byte on Moonlight, with its continue and abandon gates.

**Gated on approval:** Colibrì's Inkling baseline (E8, 514 GB) decides whether the 7A store patch is worth building.
The patch is built only if storage, not the drive, is on the critical path.

The Phase 6C items stay recorded in decision 0013 and the Phase 6B report. The 7B review recommends not pursuing them as
product work.

Open decisions for the user:

- **The direction among the 7B review's recommendations** (its § 10 and § 13).
- **Approval (or not) to download.**
  - A flagship artifact: Inkling's int4 container is 514 GB, its NVFP4 release about 592 GB, MiMo-V2.6 Pro 563.6 GB
    without its unused files.
  - Or a bounded sample of expert tensors by range requests: about 1.8 GB for E4.
- **A gcc toolchain** (MSYS2's UCRT64 gcc, not the MSYS one), or a Linux host. It is needed only to patch Colibrì's
  engines; its Windows release runs the baseline.
- **Hardware, after E1:** a second NVMe on CPU-attached lanes, or 64 GB of RAM. The report's model ranks both above any
  software change for exact mode.
- ApexQuant: a license from its authors before any use of its code.
- The Quality Budget's datasets: confirm and pin their revisions.
- Weightsift's own license (the repository has none).
- Still open from before: the rounding model for certificates (RN-even, from Phase 2); the reference for FP8 experts;
  Phase 2 on a larger model.

## Blockers

- **No flagship checkpoint is local**, and downloading one needs the user's approval. Every flagship measurement waits
  for it: the Colibrì baseline, quality, ApexQuant on real experts.
- **Colibrì's model engines need gcc and POSIX headers to be built** (MSYS2's UCRT64 on Windows). This machine has MSVC
  only, so only Colibrì's portable C builds here: its headers, segment runtime and conformance fixtures.
  - This blocks patching the engines (the store integration), not running them.
  - Colibrì's release v2.0.0 ships prebuilt Windows engines, `inkling.exe` included, with Vulkan.
  - Not yet tried on this machine.
- **ApexQuant's code has no license.**
