//! The runtime planner's memory budget: what the machine has (detected at run time by the caller, never assumed), what
//! the execution backend needs, and what is left for the expert store, split into named reservations that never sum to
//! more than the budget.
//!
//! The host budget is the available memory less a reserve for everything else on the machine. The execution backend's
//! resident weights, sequence state and workspaces come first, then the store's staging (its staging sets and its
//! prefetches in flight); the expert cache gets the rest, capped if the policy says so. The device budget is checked the
//! same way for what the backend places there. The plan also states what the cache can be expected to do relative to
//! one decode token's routed experts: arithmetic on the footprint, not a measurement.

use std::fmt;

use weightsift_store::{StoreOptions, PREFETCH_STAGING_BYTES};

/// An accelerator, as detected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Device {
    pub name: String,
    pub total_bytes: u64,
    pub free_bytes: u64,
}

/// The machine, as detected at run time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hardware {
    pub host_total_bytes: u64,
    pub host_available_bytes: u64,
    pub device: Option<Device>,
}

/// What the model needs: from its execution backend (resident weights, state at the planned context, workspaces, on the
/// host and on the device) and from its catalog (routed experts as stored).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Footprint {
    pub dense_host_bytes: u64,
    pub dense_device_bytes: u64,
    pub state_host_bytes: u64,
    pub state_device_bytes: u64,
    pub workspace_host_bytes: u64,
    pub workspace_device_bytes: u64,
    /// One routed expert, as stored (all its parts).
    pub expert_bytes: u64,
    /// The routed experts one decode token reads (experts per token times MoE layers).
    pub experts_per_token: u64,
    /// Every routed expert of the model.
    pub total_experts: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Policy {
    /// Host memory left to the system and everything else.
    pub host_reserve_bytes: u64,
    pub device_reserve_bytes: u64,
    /// The store's options; the plan sets `host_cache_bytes`.
    pub store: StoreOptions,
    /// An upper bound on the expert cache, if any.
    pub max_expert_cache_bytes: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reservation {
    pub name: &'static str,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryPlan {
    pub host_budget_bytes: u64,
    pub host: Vec<Reservation>,
    pub device_budget_bytes: u64,
    pub device: Vec<Reservation>,
    /// The store's options, its cache budget set.
    pub store: StoreOptions,
    pub expert_cache_bytes: u64,
    /// Experts the cache holds at once.
    pub cache_experts: u64,
    /// One decode token's routed experts, as stored.
    pub working_set_bytes: u64,
    /// The cache relative to one token's working set, and to all experts.
    pub cache_tokens: f64,
    pub resident_fraction: f64,
    pub notes: Vec<String>,
}

impl MemoryPlan {
    pub fn host_reserved_bytes(&self) -> u64 {
        self.host.iter().map(|r| r.bytes).sum()
    }

    pub fn device_reserved_bytes(&self) -> u64 {
        self.device.iter().map(|r| r.bytes).sum()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanError {
    /// The execution backend's host needs and the store's staging exceed the host budget.
    Host {
        needed: u64,
        budget: u64,
    },
    /// The backend places bytes on a device the machine does not have.
    NoDevice {
        needed: u64,
    },
    Device {
        needed: u64,
        budget: u64,
    },
    /// What is left for the cache does not hold one expert (an expert must be resident to be leased).
    NoRoomForOneExpert {
        cache: u64,
        expert: u64,
    },
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlanError::Host { needed, budget } => {
                write!(
                    f,
                    "the host needs {needed} bytes before any expert cache; the budget is {budget}"
                )
            }
            PlanError::NoDevice { needed } => write!(f, "{needed} bytes are placed on a device, and there is none"),
            PlanError::Device { needed, budget } => {
                write!(f, "the device needs {needed} bytes; its budget is {budget}")
            }
            PlanError::NoRoomForOneExpert { cache, expert } => {
                write!(
                    f,
                    "{cache} bytes are left for the expert cache, less than one expert ({expert})"
                )
            }
        }
    }
}

impl std::error::Error for PlanError {}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

pub fn plan(hardware: &Hardware, footprint: &Footprint, policy: &Policy) -> Result<MemoryPlan, PlanError> {
    let host_budget = hardware.host_available_bytes.saturating_sub(policy.host_reserve_bytes);
    let staging = policy.store.staging_bytes() + policy.store.max_prefetches as u64 * PREFETCH_STAGING_BYTES;
    let mut host = vec![
        Reservation {
            name: "dense weights",
            bytes: footprint.dense_host_bytes,
        },
        Reservation {
            name: "sequence state",
            bytes: footprint.state_host_bytes,
        },
        Reservation {
            name: "workspace",
            bytes: footprint.workspace_host_bytes,
        },
        Reservation {
            name: "store staging",
            bytes: staging,
        },
    ];
    let fixed: u64 = host.iter().map(|r| r.bytes).sum();
    if fixed > host_budget {
        return Err(PlanError::Host {
            needed: fixed,
            budget: host_budget,
        });
    }
    let mut cache = host_budget - fixed;
    if let Some(cap) = policy.max_expert_cache_bytes {
        cache = cache.min(cap);
    }
    if footprint.expert_bytes == 0 || cache < footprint.expert_bytes {
        return Err(PlanError::NoRoomForOneExpert {
            cache,
            expert: footprint.expert_bytes,
        });
    }
    host.push(Reservation {
        name: "expert cache",
        bytes: cache,
    });
    let device = vec![
        Reservation {
            name: "dense weights",
            bytes: footprint.dense_device_bytes,
        },
        Reservation {
            name: "sequence state",
            bytes: footprint.state_device_bytes,
        },
        Reservation {
            name: "workspace",
            bytes: footprint.workspace_device_bytes,
        },
    ];
    let device_needed: u64 = device.iter().map(|r| r.bytes).sum();
    let device_budget = match &hardware.device {
        Some(d) => d.free_bytes.saturating_sub(policy.device_reserve_bytes),
        None if device_needed > 0 => return Err(PlanError::NoDevice { needed: device_needed }),
        None => 0,
    };
    if device_needed > device_budget {
        return Err(PlanError::Device {
            needed: device_needed,
            budget: device_budget,
        });
    }
    let working_set = footprint.expert_bytes * footprint.experts_per_token;
    let all = footprint.expert_bytes * footprint.total_experts;
    let cache_tokens = if working_set > 0 {
        cache as f64 / working_set as f64
    } else {
        f64::INFINITY
    };
    let resident_fraction = if all > 0 {
        (cache as f64 / all as f64).min(1.0)
    } else {
        1.0
    };
    let mut notes = Vec::new();
    if cache_tokens < 1.0 {
        notes.push(format!(
            "the expert cache ({:.2} GiB) holds {:.2} of one decode token's routed experts ({:.2} GiB): below one \
             token's working set a least-recently-used cache serves almost nothing (decision 0012's measurement), \
             so a decode token reads about {:.2} GiB from storage",
            gib(cache),
            cache_tokens,
            gib(working_set),
            gib(working_set)
        ));
    } else {
        notes.push(format!(
            "the expert cache ({:.2} GiB) holds {:.2} decode tokens' routed experts and {:.2}% of all experts; its hit \
             rate depends on the routing and must be measured",
            gib(cache),
            cache_tokens,
            100.0 * resident_fraction
        ));
    }
    Ok(MemoryPlan {
        host_budget_bytes: host_budget,
        host,
        device_budget_bytes: device_budget,
        device,
        store: StoreOptions {
            host_cache_bytes: cache,
            ..policy.store
        },
        expert_cache_bytes: cache,
        cache_experts: cache / footprint.expert_bytes,
        working_set_bytes: working_set,
        cache_tokens,
        resident_fraction,
        notes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;

    fn machine(host: u64, device: Option<u64>) -> Hardware {
        Hardware {
            host_total_bytes: host,
            host_available_bytes: host,
            device: device.map(|free| Device {
                name: "test".into(),
                total_bytes: free,
                free_bytes: free,
            }),
        }
    }

    fn policy() -> Policy {
        Policy {
            host_reserve_bytes: 4 * GIB,
            device_reserve_bytes: GIB,
            store: StoreOptions::default(),
            max_expert_cache_bytes: None,
        }
    }

    fn footprint() -> Footprint {
        Footprint {
            dense_host_bytes: 15 * GIB,
            state_host_bytes: GIB,
            workspace_host_bytes: 512 * MIB,
            expert_bytes: 28 * MIB,
            experts_per_token: 6 * 64,
            total_experts: 256 * 64,
            ..Footprint::default()
        }
    }

    #[test]
    fn the_cache_gets_what_the_backend_and_staging_leave() {
        let plan = plan(&machine(32 * GIB, None), &footprint(), &policy()).unwrap();
        let staging = StoreOptions::default().staging_bytes() + PREFETCH_STAGING_BYTES;
        let expected = 32 * GIB - 4 * GIB - 15 * GIB - GIB - 512 * MIB - staging;
        assert_eq!(plan.expert_cache_bytes, expected);
        assert_eq!(plan.store.host_cache_bytes, expected);
        assert_eq!(plan.host_reserved_bytes(), plan.host_budget_bytes);
        assert_eq!(plan.cache_experts, expected / (28 * MIB));
        assert_eq!(plan.working_set_bytes, 384 * 28 * MIB);
        // About 11.4 GiB of cache for 10.5 GiB of experts per token: just over one token's working set.
        assert!(
            plan.cache_tokens > 1.0 && plan.cache_tokens < 1.2,
            "{}",
            plan.cache_tokens
        );
        assert!(plan.notes[0].contains("must be measured"));
    }

    #[test]
    fn a_cache_below_one_token_is_reported() {
        let capped = Policy {
            max_expert_cache_bytes: Some(4 * GIB),
            ..policy()
        };
        let plan = plan(&machine(32 * GIB, None), &footprint(), &capped).unwrap();
        assert_eq!(plan.expert_cache_bytes, 4 * GIB);
        assert!(plan.cache_tokens < 1.0);
        assert!(plan.notes[0].contains("serves almost nothing"), "{}", plan.notes[0]);
    }

    #[test]
    fn infeasible_budgets_are_refused() {
        let heavy = Footprint {
            dense_host_bytes: 30 * GIB,
            ..footprint()
        };
        assert!(matches!(
            plan(&machine(32 * GIB, None), &heavy, &policy()),
            Err(PlanError::Host { .. })
        ));
        let tight = Footprint {
            dense_host_bytes: 32 * GIB - 4 * GIB - GIB - 512 * MIB - 100 * MIB,
            ..footprint()
        };
        assert!(matches!(
            plan(&machine(32 * GIB, None), &tight, &policy()),
            Err(PlanError::Host { .. }) | Err(PlanError::NoRoomForOneExpert { .. })
        ));
        let on_device = Footprint {
            dense_device_bytes: 7 * GIB + MIB,
            ..footprint()
        };
        assert!(matches!(
            plan(&machine(32 * GIB, None), &on_device, &policy()),
            Err(PlanError::NoDevice { .. })
        ));
        assert!(matches!(
            plan(&machine(32 * GIB, Some(8 * GIB)), &on_device, &policy()),
            Err(PlanError::Device { .. })
        ));
        let fits = Footprint {
            dense_device_bytes: 6 * GIB,
            ..footprint()
        };
        let plan = plan(&machine(32 * GIB, Some(8 * GIB)), &fits, &policy()).unwrap();
        assert_eq!(plan.device_budget_bytes, 7 * GIB);
        assert_eq!(plan.device_reserved_bytes(), 6 * GIB);
    }
}
