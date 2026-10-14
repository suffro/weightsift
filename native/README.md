# Weightsift native runtime

Phase 6A (decision `.context/decisions/0012-phase6a-native-runtime-foundation.md`): the performance-critical part of
Weightsift's storage path, in Rust, behind the Python storage contract. Python's `awpmi.storage.native.NativePageStore`
uses it; the Python backend (`FileBackedPageStore`) remains the reference implementation and the fallback.

Phase 7A (decision `.context/decisions/0014-phase7a-independent-flagship-runtime.md`): the foundation of the independent
flagship runtime. The weight storage backend for execution engines written in C or C++ (Colibrì's first), the runtime
planner, and the bridge to Colibrì, built from its pinned, fetched sources.

| Crate | What it is |
| --- | --- |
| `core/` (`weightsift-io`) | Read plans (the same as Python's `plan_reads` and `PageStreamer._pieces`), positioned reads on a pool of threads (direct I/O), a host-RAM row cache under a strict byte budget, transfer jobs into caller-owned staging slots, leases (Phase 7A: `Engine::acquire`, rows held in the cache for a caller that reads them in place), byte accounting. No Python, no CUDA, no model knowledge: rows are bytes, whatever they encode (BF16 rows, Phase 6B's encoded rows, a quantized container's). |
| `python/` (`weightsift-native`) | The PyO3 module `weightsift_native` (abi3, Python 3.11+): `Engine` and `Job`. |
| `store/` (`weightsift-store`) | The expert store: routed experts described by a catalog (`weightsift-expert-catalog` v1, written by a model adapter from checkpoint headers), leased from the core's host cache, with prefetch, statistics and a versioned C ABI (`include/weightsift_store.h`, ABI 1; `rlib`, `cdylib`, `staticlib`). |
| `runtime/` (`weightsift-runtime`) | The runtime planner (one detected memory budget split strictly between the execution backend and the store) and the execution backend interface (`ExecutionBackend`, `ExecutionSession`). |
| `colibri/` (`weightsift-colibri`) | The bridge to Colibrì: its expert-store lease contract (`c/expert_store.h`) served by the store through a C shim (`csrc/ws_colibri_store.c`), and its engine-neutral segment runtime (`c/segment_runtime.h`, ABI 1) as an `ExecutionBackend`. Built only from the pinned, fetched sources (below). |

## Upstream sources (decision 0014)

Colibrì is not forked, vendored or committed. `upstream.json` pins it: URL, commit, license, attribution, and the sha256
of every file the build uses (the repository's content, LF line endings). `fetch_upstream.py` (standard library and
git) makes a shallow, sparse checkout of exactly those files into `third_party/<name>/` (gitignored), verifies them, and
writes a marker; `--check` verifies without network. `colibri/build.rs` re-hashes the files against the pin before
compiling them; without them the crate builds without its C parts, warns, and its tests are ignored with the reason.

```bash
python native/fetch_upstream.py          # fetch (or confirm) every pinned upstream
python native/fetch_upstream.py --check  # verify what is there
```

To move a pin: edit `upstream.json` (commit, and every file's new sha256 from `git cat-file blob <commit>:<path> |
sha256sum`), say why in the commit, run the fetch, then the bridge's tests. Licenses: Colibrì is Apache-2.0 (its
`NOTICE`, Copyright 2026 Vincenzo Fornaro, and `THIRD_PARTY_NOTICES.md` come with the fetch); a binary that links its
segment runtime carries that notice.

## The expert store's C ABI

`store/include/weightsift_store.h` is the contract: stores and leases are 64-bit handles in tables (never pointers), so
an unknown, closed, released or doubly released handle is an error code; a lease's bytes stay valid, unmoved and
unevicted until its release; close is refused while leases live and waits for acquires in flight; every function may
be called from any thread; panics never cross the boundary (`WS_ERR_INTERNAL`); structs are versioned by `struct_size`.
When an expert cannot be made resident without evicting a leased row, `ws_store_acquire` returns `WS_ERR_UNAVAILABLE`
at once (the caller releases and retries; the store never waits for room). Its tests include a C client compiled by the
platform's C compiler (`store/tests/c/abi_client.c`) and the header's layout checked against the Rust structs.

## The host cache's memory (Phase 6B, decision 0013)

A cached row is held in blocks of one size (`HostCache::block_bytes`; by default the greatest common divisor of the
segments' rows of at least 1 MiB, at least 1 MiB: 5.5 MiB for Moonlight's BF16 rows, 3.7 MiB for its encoded rows). An
evicted row's blocks go to a pool from which any later row takes its blocks, whatever its size, so evictions and
admissions neither free nor allocate on the critical path; the pool counts against the budget (rows, pooled blocks and
reservations of loads in flight never exceed it; surplus blocks are freed). The LRU order, leases, load-once and the hits
and misses do not depend on the block size (tested). A transfer's tasks are handed to the readers one wake-up at a time
(a reader that takes a task wakes the next while tasks remain), and a request never waits for a transfer's load of a
row: it reads the row itself without admitting it (counted as a miss and a bypass), so two jobs cannot wait on each
other's staging slots. Loads by prefetches and, since Phase 7A, by leases are waited for: the pool carries them out
without any caller. The expert store sets the block to its largest part, so that every leased part is one piece.

## Building

The root project builds the Python module: `python -m uv sync` (the default dependency groups include `native`). uv
calls maturin as a PEP 517 backend on `native/pyproject.toml` (which builds `python/` and the core only) and installs the
module into `.venv`; it rebuilds when any `*.rs`, `Cargo.toml` or `Cargo.lock` changes. The other crates are built by
cargo.

| Requirement | Version |
| --- | --- |
| Rust toolchain | 1.95.0, pinned by `rust-toolchain.toml` (rustup installs it; `rustfmt` and `clippy` components) |
| C compiler and linker | MSVC build tools on Windows (`x86_64-pc-windows-msvc`, found by the `cc` crate); the system `cc` on Linux |
| maturin | ≥ 1.15, < 2 (fetched by uv for the build) |
| Python | ≥ 3.11 (abi3 wheel); and to fetch upstream sources, any Python 3 with git on PATH |

Crates: `pyo3` 0.29 (bindings), `lru` 0.18 (the LRU order), `libc` 0.2 (Linux only: `O_DIRECT`), `serde` and
`serde_json` 1 (catalogs), `cc` 1.2 (building C), `sha2` 0.10 (verifying fetched sources; tests). Positioned reads and
direct-I/O flags are the standard library's (`FileExt`, `OpenOptionsExt`).

Development:

```bash
python native/fetch_upstream.py                              # once: Colibrì's pinned files for weightsift-colibri
cd native
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings      # PYO3_PYTHON=<repo>/.venv/Scripts/python.exe if needed
cargo test --workspace                                       # unit, integration (temporary files) and C ABI tests
cd .. && python -m uv run python -m pytest tests/test_native_storage.py tests/test_moe.py tests/test_moe_compact.py tests/test_moe_chunked.py
```

The store on real checkpoint bytes (Moonlight's experts; a microbenchmark of the store alone):

```bash
python -m uv run python benchmarks/store_check.py --output experiments/phase7a/store-moonlight --layers 1,2,3,4
cd native && cargo run --release -p weightsift-store --example store_check -- ../experiments/phase7a/store-moonlight
```

## Platforms

| Platform | State |
| --- | --- |
| Windows 11, x86-64 (MSVC) | built and tested here; direct I/O with `FILE_FLAG_NO_BUFFERING`, one handle per reader thread; the C parts compiled by MSVC (C11) |
| Linux, x86-64 | builds; direct I/O with `O_DIRECT`; not exercised on this machine (as the Python store's POSIX path). The C test archives are passed to the test link as arguments, which GNU ld's library order may require adjusting |
| Others | buffered reads only (`direct=False`); direct I/O is refused |

Colibrì's model engines (`c/inkling.c`, `c/mimo.c`) need gcc and POSIX headers (MSYS2's MinGW-w64 on Windows): none of
them is built here.

## Without the extension

`python -m uv sync --no-group native` installs everything but the extension. `awpmi.storage.native.NATIVE_AVAILABLE` is
then False, `Pack.store(backend="native")` raises, the native tests are skipped (reported as such), and every Python
path works as before Phase 6A: the native backend changes where bytes move, never what is computed.

## The `unsafe` boundary

`core/src/buffer.rs` holds the core's only `unsafe` type, `RawBuffer`: writing into staging memory the caller owns
(PyTorch's pinned buffers, or the store's staging), so that reads land where the device copies from, and (Phase 6B) into
a loading row's blocks. Its invariants (the memory outlives every task; tasks write disjoint ranges; a slot is handed to
the caller only after its writes finished and refilled only after the caller released it; a row becomes an entry only
after its last admission) are documented there and upheld by the engine and the binding (`Job` keeps its buffers
referenced until no task can touch them; a row's fill owns its blocks until it completes or gives them back; `acquire`
returns only once none of its tasks touches the caller's slots). Seven `unsafe` blocks use it, six in
`core/src/engine.rs` (the tasks that read into, copy into or out of, gather in and admit from a slot, and `read_rows`'
copy out) and one in the binding (wrapping a Python buffer), each stating the invariant it relies on.

Phase 7A adds two boundaries whose requirement is a C ABI, each `unsafe` block stating what it relies on:

- `store/src/ffi.rs`: dereferencing the caller's pointers (paths, keys, output structs, error buffers) as the header
  specifies;
- `colibri/src/segment.rs` and `store.rs`: calling Colibrì's segment runtime and the shim, and the snapshot callbacks.

Leased bytes need no `unsafe`: they are the cache's entries, reached through `Entry::contiguous`.
