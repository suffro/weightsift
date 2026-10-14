//! What Weightsift asks of a model execution backend (decision 0014): open a contiguous range of layers, report what it
//! can do, and run boundary activations through it in isolated sessions, with snapshots of a session's state.
//!
//! The backend owns the model's math, weights, kernels and sequence state; Weightsift owns storage (`weightsift-store`)
//! and planning (`plan`). The shape of this interface is Colibrì's engine-neutral layer-segment runtime (ABI v1), which
//! `weightsift-colibri` implements it with; a backend that runs every layer is the special case of one full range.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
        }
    }
}

/// What an opened range can do, as its backend reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capabilities {
    pub engine_id: String,
    /// The layout of boundary activations and snapshots.
    pub state_schema: String,
    /// Builds whose results and snapshots agree (precision, reductions, backend rules).
    pub numeric_class: String,
    pub dtype: DType,
    /// Values per row of boundary activations.
    pub state_width: u32,
    pub max_batch_rows: u32,
    pub max_context_tokens: u32,
    /// Layers of the whole model; the range is `layer_begin..layer_end`.
    pub num_layers: u32,
    pub layer_begin: u32,
    pub layer_end: u32,
    /// A run needs one token id per row.
    pub token_ids: bool,
    pub snapshots: bool,
    /// No weights outside the range were loaded.
    pub range_native: bool,
    pub multi_session: bool,
    /// Where it computes ("cpu", "cuda", "hip", "metal", "vulkan").
    pub backends: Vec<&'static str>,
}

/// One run: `rows` rows of boundary activations at sequence position `position`.
pub struct RunRequest<'a> {
    pub position: u64,
    pub rows: u32,
    pub token_ids: Option<&'a [i32]>,
    pub input: &'a [u8],
    pub output: &'a mut [u8],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionError(pub String);

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ExecutionError {}

/// One conversation's state in an opened range. Not shared between threads while a call runs.
pub trait ExecutionSession {
    fn run(&mut self, request: RunRequest<'_>) -> Result<(), ExecutionError>;
    /// The session's state, in the backend's private format (named by `Capabilities::state_schema`).
    fn snapshot(&mut self) -> Result<Vec<u8>, ExecutionError>;
    /// Replace the session's state with a snapshot's; a rejected snapshot leaves the state as it was.
    fn restore(&mut self, snapshot: &[u8]) -> Result<(), ExecutionError>;
}

/// An opened range of layers. Sessions borrow it: it cannot close while one is alive.
pub trait ExecutionBackend {
    type Session<'a>: ExecutionSession
    where
        Self: 'a;

    fn capabilities(&self) -> &Capabilities;

    fn session(&self, context_tokens: u32, memory_limit_bytes: u64) -> Result<Self::Session<'_>, ExecutionError>;
}
