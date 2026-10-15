//! Research prototype for the lossless MoE compression study (`research/lossless_moe`). Not a runtime: nothing in
//! Weightsift's product code depends on it.
//!
//! - `rans`: static rANS with a table per context, streams by rows, a lockstep decoder with checked stream ends.
//! - `container`: the `LMC1` block record and its validating parser; every byte is counted.
//! - `colibri` (when its pinned sources are fetched): Colibri's int4-rans256-g0 codec, called through its own bridge.

pub mod container;
pub mod rans;

#[cfg(colibri_rans)]
pub mod colibri;

/// Whether Colibri's codec was compiled in (its pinned sources were present and verified at build time).
pub const COLIBRI: bool = cfg!(colibri_rans);
