//! weightsift-colibri: the bridge between Weightsift's runtime and Colibrì's engines (Phase 7A, decision 0014).
//!
//! Colibrì (JustVugg/colibri, Apache-2.0) runs the flagship models; Weightsift reuses it, never forks or vendors it.
//! This crate is built from Colibrì's sources as pinned in `native/upstream.json` and fetched by
//! `native/fetch_upstream.py` (verified by sha256 when it builds), and it holds the two seams:
//!
//!   * storage (`store`): Colibrì's expert-store lease contract (`c/expert_store.h`) served by Weightsift's expert
//!     store, through a C shim (`csrc/ws_colibri_store.c`) over the store's C ABI: an engine that reads its experts
//!     through a `ColiExpertStore` gets them from Weightsift's host cache, leased in place;
//!   * execution (`segment`): Colibrì's engine-neutral layer-segment runtime (`c/segment_runtime.h`, ABI v1) as
//!     Weightsift's `ExecutionBackend`, driven from Rust.
//!
//! Colibrì's model engines are not built here (they need gcc and POSIX headers; this crate builds what MSVC can), and
//! no Colibrì engine reads through the store yet: for the first flagship that needs a patch of the engine, the next
//! acceptance gate. Without the fetched sources the crate builds empty and says so (`UPSTREAM_AVAILABLE`).

/// Whether this build has Colibrì's pinned sources (and so the modules below).
pub const UPSTREAM_AVAILABLE: bool = cfg!(colibri_sources);

#[cfg(colibri_sources)]
pub mod segment;
#[cfg(colibri_sources)]
pub mod store;
