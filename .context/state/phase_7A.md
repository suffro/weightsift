# Weightsift — Phase 7A: Independent Flagship Runtime Foundation & ApexQuant Feasibility

## Role and objective

You are the lead systems engineer working on **Weightsift**, an independent open-source runtime for executing frontier-scale Mixture-of-Experts (MoE) language models on severely memory-constrained consumer hardware.

Your task is to **begin the next architectural phase by implementing a practical, maintainable foundation**, reusing proven technology from existing inference engines instead of rebuilding solved components.

This is an implementation task, not merely a research report. However, this phase must remain bounded: implement the useful foundation, establish reproducible integration and validation procedures, and stop before expensive or speculative work without evidence.

### Repositories

- Weightsift: https://github.com/suffro/weightsift
- Working branch: `weightsift-and-AWPMI-implementation`
- Colibrì: https://github.com/JustVugg/colibri
- DwarfStar / DS4: https://github.com/antirez/ds4
- ApexQuant: https://github.com/afristrup/ApexQuant
- ApexQuant paper: https://arxiv.org/abs/2610.07904

Read the current Weightsift source, `.context/` documentation, architectural decisions, and Phase 6A/6B reports before making changes.

**Do not discard, rewrite or regress validated earlier work.**

---

## 1. Non-negotiable project direction

Weightsift must remain an **independent project with its own architecture and Rust-native systems core**.

### Hard constraints

1. **DO NOT fork Colibrì or DS4.**
2. Do not copy or vendor either complete repository into Weightsift.
3. Reuse upstream components selectively through documented, version-pinned dependencies, FFI boundaries or appropriately licensed source-level extraction where necessary.
4. Prefer existing, validated inference engines and kernels over implementing new transformer architectures from scratch.
5. **Rust is the preferred language for new systems infrastructure**, especially memory ownership, cache management, I/O, concurrency and scheduling.
6. Do not rewrite optimized C/C++/CUDA/Vulkan code simply because it is written in C.
7. Keep the project broadly model-agnostic through common interfaces, while allowing model-specific optimized backends.
8. Maintain a strict distinction between validated results, extrapolations and hypotheses.
9. Optimize for **near-original model quality**, not numerical bit-exactness. Preserve existing bit-exact regression tests, but do not impose them on new quantized inference modes.
10. Do not start another open-ended research program or implement an unproven quantization algorithm from scratch.

The objective is not to create "Colibrì in Rust."

The objective is to build a distinct runtime that can leverage proven execution engines while independently optimizing weight representation, storage, placement, memory usage and execution.

---

## 2. Primary target

Focus on **real flagship models around one trillion total parameters**.

### First target

**Inkling 975B**, using Colibrì's existing expert-streaming implementation as the initial execution reference.

Relevant documentation:

https://github.com/JustVugg/colibri/blob/main/docs/inkling.md

The project documents a lower-memory configuration with quantized dense weights, approximately 25 GB RAM requirements, and INT4 routed experts.

Important qualifications:

- This configuration is extremely SSD-bound.
- Its quantized quality is not automatically equivalent to the original checkpoint.
- Full model execution requires substantial disk space.
- No full checkpoint or large quantization dataset may be downloaded without explicit user approval.

### Secondary target

**MiMo-V2.6 Pro 1.02T**, especially because its experts are distributed in native MXFP4.

https://github.com/JustVugg/colibri/blob/main/docs/mimo.md

Use MiMo as the secondary architecture and quantization reference. Do not pretend its documented memory requirements fit the target's 32 GB RAM without further changes.

### Hardware reference

- NVIDIA RTX 4060 Ti, 8 GB VRAM
- 32 GB host RAM
- NVMe SSD
- PCIe 3.0 x8 effective GPU connection
- CUDA when supported

Detect actual hardware capabilities at runtime. Do not hardcode these values as universal requirements.

Smaller fixtures may be used for CI and correctness tests, but **performance and quality claims must be validated on genuine flagship workloads**.

---

## 3. Stage A — Architecture and reuse audit

Before implementing integration code, inspect Colibrì, DS4 and the current Weightsift runtime.

Produce an explicit component-level reuse matrix covering:

- Model architecture execution
- MoE routing and expert execution
- Quantized CPU kernels
- CUDA/Vulkan/Metal kernels
- Expert storage and loading
- SSD streaming and direct I/O
- RAM cache and eviction
- GPU expert cache
- Prefetch and scheduling
- KV cache management
- Model formats and quantization
- Instrumentation and benchmarking

For each component identify:

- Current implementation and source location
- Existing upstream functionality
- Licensing and attribution requirements
- Whether it can be reused without modification
- Whether a C ABI or adapter is needed
- Whether Rust could offer an engineering or performance advantage
- Estimated integration effort and maintenance risk

### Required decision

Choose the **smallest viable reuse architecture**.

Prefer Colibrì for flagship-specific execution, but evaluate DS4 components where they are demonstrably better or more reusable.

Do not merge the two engines wholesale.

Investigate Colibrì's existing:

- `c/expert_store.h`
- `c/expert_store_registry.h`
- Model-specific engine implementations
- Build and runtime interfaces

The expert-store interface already provides lookup, release, prefetch and statistics operations. Its pluggable backend registration is documented particularly for DeepSeek V4; **do not assume the same integration seam already exists for Inkling or MiMo**.

Determine the minimal adapter required for the first flagship.

Do not silently turn integration into a complete engine rewrite.

---

## 4. Stage B — Implement the independent Rust runtime foundation

Extend the existing Rust implementation under `native/`.

Preserve the proven `weightsift-io` storage engine and existing Python bindings.

Create a clear separation between:

**Model Execution Backend**

Owns model-specific forward computation, attention, routing and quantized compute kernels.

**Weight Storage Backend**

Owns page/block access, filesystem I/O, cache admission and eviction, prefetch, reservations, memory budgeting and statistics.

**Runtime Planner**

Coordinates memory limits, expert requests, placement decisions and future precision policies.

### Implementation requirements

- Introduce a minimal, versioned, model-neutral native interface for expert storage and execution integration.
- Use Rust for new storage and scheduling infrastructure where appropriate.
- Keep the existing Rust cache, direct-I/O and worker machinery unless profiling demonstrates that it needs replacement.
- Avoid unnecessary asynchronous-runtime dependencies or major refactoring.
- Use narrow FFI boundaries with explicit lifetime, ownership and thread-safety contracts.
- Prevent use-after-free, double release, invalid leases and cache eviction while expert buffers are in use.
- Preserve strict RAM/VRAM budget accounting.
- Allow fallback to the existing implementation when a new backend is unavailable.
- Keep external model-specific code isolated and independently updatable.
- Add deterministic tests for ABI lifecycle, buffer ownership, concurrent requests and error handling.

### Upstream integration policy

Do not create a GitHub fork.

Do not import an entire upstream repository into the project history.

When upstream sources are needed:

1. Pin an exact revision.
2. Record license and provenance.
3. Use an external build/dependency mechanism where feasible.
4. Keep third-party build artifacts outside committed source files.
5. Add only the minimum adapter/shim needed.
6. Document the procedure for updating the upstream revision.

A locally fetched upstream source tree for building is acceptable; it must not become a wholesale vendored copy committed to Weightsift.

If an upstream component cannot reasonably be linked or isolated without extensive modification, **document that limitation rather than pretending integration is complete**.

### Expected working artifact

Implement a small, real vertical slice proving the selected native interface can be used from Rust.

At minimum:

- Compiling Rust/C integration where the local toolchain permits
- Exercising expert acquisition, release and cache lifecycle through the interface
- Reporting actual I/O and cache statistics
- Preserving an existing backend as reference

If the real flagship model is not locally available, use explicit interface-level fixtures for the integration test. Do not claim that these fixtures demonstrate working 1T inference.

Full flagship integration remains a subsequent acceptance gate when the necessary model artifacts are available.

---

## 5. Stage C — ApexQuant feasibility track

Start this investigation **in parallel with the architecture work**, but keep it small, measurable and independently stoppable.

Reference:

https://github.com/afristrup/ApexQuant

### C.1 Licensing gate

First determine whether ApexQuant's implementation has an explicit license authorizing reuse.

At the time this phase was proposed, no clear repository-level license had been identified.

Therefore:

- Inspect the current repository, packages and individual source headers.
- Record findings and unresolved restrictions.
- Do not copy or vendor unlicensed source code.
- Do not make an unlicensed package a required Weightsift dependency.
- If reuse permission is unclear, limit work to lawful independent analysis and integration-interface design until authorization is obtained.

Do not bypass this gate for convenience.

### C.2 Technical investigation

Study ApexQuant's:

- Residual re-isotropization
- Nested/progressive representations
- Scalar, E8 and trellis quantization stages
- Fused quantized GEMM/GEMV kernels
- Per-prefix reconstruction properties
- Memory layout and decoding costs
- Optional sensitivity-based pass allocation

Determine whether its formats can realistically serve **routed MoE expert matrices** rather than merely ordinary dense-model Linear layers.

### C.3 Real-weight feasibility

Use real expert tensors from an existing flagship checkpoint if they are locally available or explicitly approved for acquisition.

Do not substitute a tiny language model as evidence of flagship quality.

Measure, where permissible:

- Actual stored bytes per weight, including metadata
- Base-only representation size
- Incremental residual size
- Conversion time and peak RAM/VRAM
- Matrix-output error on representative activations, if available
- Decode/GEMV cost with each prefix
- Whether the kernels can operate directly on the compressed representation
- Overhead from rotations or additional passes
- Throughput versus the original MXFP4/INT4 implementation

For native MXFP4 checkpoints, explicitly distinguish:

1. Error introduced by the original MXFP4 quantization.
2. Additional error introduced by ApexQuant.
3. Any comparison against original higher-precision weights, if actually available.

Do not equate small weight-space error with preserved reasoning or coding capability.

Do not claim a method passes the Quality Budget based on isolated matrix tests.

### C.4 Cost-based early stopping

Use the native flagship format as the primary baseline, **not BF16 alone**.

For this initial feasibility stage, use the following screening criteria:

- Meaningful effective storage or I/O reduction relative to native MXFP4/INT4.
- No obvious kernel overhead that destroys the expected end-to-end benefit.
- A plausible, documented path to converting full expert banks.
- No fundamental incompatibility with the target model's tensor geometry or execution backend.

A 20% reduction in effective bytes at acceptable incremental compute cost is a **screening target**, not a guaranteed performance result.

If the optimistic end-to-end bound is negligible, stop the track and explain why.

Do not proceed to a full checkpoint conversion automatically.

---

## 6. Quality Budget — required architecture

Treat quality preservation as a first-class runtime constraint.

The goal is to preserve the original flagship's practical capabilities as closely as possible.

Create an extensible quality-evaluation specification covering:

- Coding
- Mathematical reasoning
- General reasoning
- Instruction following
- Long-context comprehension
- Tool-use behavior, where supported

Requirements:

- Distinguish original checkpoint quality from the quality of an already quantized release.
- Compare the candidate against the strongest relevant available baseline.
- Use held-out evaluation data distinct from calibration data.
- Record per-category regressions rather than hiding them in a single average.
- Report evaluation sample counts and uncertainty.
- Never invent scores for unavailable hardware or checkpoint artifacts.
- Keep quality thresholds configurable; do not silently assume that a fixed percentage is statistically sufficient.

In this phase, implement the evaluation configuration, result schema and baseline-comparison utilities where reasonable.

Full quality certification can remain pending until an actual flagship checkpoint and sufficient evaluation resources are available.

---

## 7. Benchmarking contract

Before claiming any runtime optimization, establish a reproducible Colibrì baseline.

Compare on identical hardware, checkpoint, precision, cache budget and workloads.

Required metrics:

- Decode tokens/second
- Prefill latency
- Time to first token
- Cold and warm performance
- Diverse rotating-prompt performance
- SSD bytes read per generated token
- Host-to-device bytes per token
- Cache hit/miss ratios
- Peak RAM and VRAM
- GPU and CPU utilization
- I/O wait and scheduling overhead
- Model output/quality differences

Distinguish measured results from extrapolations.

For isolated Rust replacements, first validate correctness and resource ownership, then compare throughput and tail latency to the original C component.

**Rust is preferred for maintainable systems engineering, but performance advantages must be demonstrated, not assumed.**

Do not report a microbenchmark speedup as an end-to-end inference speedup.

Preserve results with hardware metadata, exact source revisions, commands and configuration.

---

## 8. Explicit non-goals

Do not implement any of the following in this initial phase:

- A full rewrite of Colibrì or DS4 in Rust.
- An entire new flagship architecture from scratch.
- A new custom quantization algorithm.
- Full dynamic per-token quantization planning.
- Speculative expert skipping or altered MoE routing.
- Exact greedy-argmax certification.
- Large-scale checkpoint conversion.
- Training or fine-tuning flagship models.
- Unapproved multi-hundred-gigabyte downloads.
- A broad refactor of the existing Weightsift runtime.
- New benchmarks presented as validated 1T results when no 1T model was executed.

These may be considered later only if supported by measured evidence.

---

## 9. Deliverables

Make real code changes in the existing Weightsift repository where feasible.

Deliver:

1. A documented target architecture for the independent Weightsift Rust-native runtime.
2. A component-level Colibrì/DS4 reuse and licensing audit.
3. A minimal native interoperability implementation with tests.
4. Integration instructions and reproducible build commands.
5. A reproducible baseline benchmark specification.
6. An ApexQuant technical and licensing feasibility report.
7. Any safely executable low-cost ApexQuant measurements.
8. A Quality Budget evaluation specification and machine-readable results schema.
9. Updated project state, roadmap and architectural decision records under `.context/`.

Suggested document locations, subject to the existing repository conventions:

- `.context/state/phase_7A.md`
- `.context/history/2026-10-12-weightsift-phase7a-report.md`
- A new numbered architecture decision record
- `research/apexquant/` for independent evaluation scripts and findings, if appropriate

Preserve historical reports and decisions; do not rewrite past results to fit the new direction.

Do not mark Phase 7A complete unless its actual implementation gates are satisfied.

---

## 10. Execution procedure and hard-stop gates

Follow this order:

**Step 1 — Repository inspection**

Read the current code and architecture. Confirm the branch and existing tests.

**Step 2 — External component audit**

Identify viable integration points, licensing constraints and the least-invasive architecture.

**Step 3 — Choose one integration path**

Document why it is preferable to the alternatives. Do not start several competing implementations.

**Step 4 — Implement the minimal working integration**

Build real interfaces and tests. Avoid rewriting unrelated code.

**Step 5 — ApexQuant feasibility**

Perform the license check and bounded technical investigation in parallel with the integration work.

**Step 6 — Validation**

Run available tests, verify resource accounting and record results.

**Step 7 — Decision**

Report which capabilities are implemented, which remain blocked and whether ApexQuant merits a larger experiment.

### Mandatory stop conditions

Stop the affected workstream when:

- A required dependency cannot legally be reused.
- The chosen C integration requires an unexpectedly large upstream rewrite.
- Available hardware cannot run a requested measurement.
- A large model download or substantial paid compute would be required without authorization.
- There is no demonstrated or credible performance headroom.
- A foundational correctness, memory-safety or ownership issue remains unresolved.

A stop condition must not be disguised as success.

Continue independent workstreams that are not blocked.

---

## 11. Final report format

At the end, provide a concise but technically rigorous report containing:

### Implemented
Actual code, interfaces and tests added.

### Reused technology
Upstream project, pinned revision, component, license and integration mechanism.

### Measurements
Only actual measurements, with baseline and methodology.

### ApexQuant verdict
One of:

- **GO** — technically and legally viable for a larger flagship experiment.
- **CONDITIONAL GO** — promising, but specific unresolved prerequisites remain.
- **NO-GO** — insufficient benefit or fundamental incompatibility.
- **BLOCKED** — essential code, license, hardware or model data unavailable.

### Remaining blockers
Explicitly identify missing artifacts, performance evidence or required user authorization.

### Recommended next milestone
One bounded, concrete next step justified by the findings.

---

## Final instruction

**Prioritize engineering value over novelty for its own sake.**

Weightsift must not become a renamed copy of Colibrì, but neither should it reinvent functioning inference engines.

Build an independent Rust-oriented runtime around proven native technology, preserve its existing validated work, and investigate ApexQuant as a potential future differentiator.

The long-term goal is to execute approximately 1T-parameter flagship MoE models on constrained consumer hardware with the best achievable quality, memory efficiency and inference speed.

**Begin implementing Phase 7A now. Do not stop after writing a plan. Do not prematurely begin later experimental phases. Report evidence, not aspirations.**