//! weightsift-runtime: the runtime planner and the execution backend interface of Weightsift's flagship runtime
//! (Phase 7A, decision 0014).
//!
//!   * `plan`: one memory budget, detected on the machine at run time, split strictly between what the execution
//!     backend needs (resident weights, sequence state, workspaces) and the weight storage backend (the expert store's
//!     cache and staging); the store's options come out of it.
//!   * `execution`: what Weightsift asks of a model execution backend: run a contiguous range of layers on boundary
//!     activations in isolated sessions, with snapshots. Colibrì's segment runtime is the first implementation
//!     (`weightsift-colibri`).
//!
//! Placement beyond host RAM (a device tier of experts) and precision policies are not planned here yet: the store
//! serves routed experts in the format the checkpoint stores them in.

pub mod execution;
pub mod plan;

pub use execution::{Capabilities, DType, ExecutionBackend, ExecutionError, ExecutionSession, RunRequest};
pub use plan::{plan, Device, Footprint, Hardware, MemoryPlan, PlanError, Policy, Reservation};
