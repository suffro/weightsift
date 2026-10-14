//! weightsift-store: the weight storage backend of Weightsift's flagship runtime (Phase 7A, decision 0014).
//!
//! An `ExpertStore` serves the routed experts a `Catalog` describes: it reads them with `weightsift-io` (plans equal to
//! Python's, direct reads on a worker pool), keeps them in its host-RAM cache under a strict byte budget, and leases
//! them to an execution engine, which reads their bytes in place until it releases them. Rust callers use the store
//! directly; C and C++ engines use its versioned C ABI (`include/weightsift_store.h`, `ffi`).
//!
//! The store is model-agnostic: what a layer's experts are made of (parts, views, formats) is the catalog's data,
//! written by a model adapter. The execution engine (Colibrì's, through `weightsift-colibri`) owns the computation.

pub mod catalog;
pub mod ffi;
pub mod store;

pub use catalog::Catalog;
pub use store::{
    ExpertKey, ExpertStore, Layer, Lease, Part, Result, Scales, StoreError, StoreOptions, StoreStats, View, ViewBytes,
    PREFETCH_STAGING_BYTES,
};
