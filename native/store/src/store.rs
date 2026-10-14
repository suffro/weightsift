//! The expert store: a checkpoint's routed experts, described by a catalog, leased from a host-RAM cache.
//!
//! `ExpertStore::acquire` returns a `Lease`: every part of an expert resident in the host cache of `weightsift-io`, held
//! in one piece each and unmoved until the lease is dropped, so that an execution engine reads the bytes in place. The
//! store is the weight storage backend of the flagship runtime (decision 0014): it owns file I/O, admission and eviction
//! (the cache's least-recently-used order under a strict byte budget), prefetch, leases and statistics; the engine owns
//! the computation. It knows no model: layers, experts, parts and views come from the catalog.
//!
//! Guarantees, all from the core's cache and engine (tested there and here):
//!
//!   * a leased expert's rows are never evicted or cleared, and their bytes never move or change while the lease lives;
//!   * the cache never holds more than its budget, leased rows and rows being loaded included; when an expert cannot be
//!     made resident without evicting a leased row, `acquire` fails with `Unavailable` and holds nothing (the caller
//!     releases leases and retries: the store never blocks waiting for room, so leases cannot deadlock it);
//!   * a row is read once however many callers want it at the same time (they wait for the one load);
//!   * every byte served is the byte on storage (direct reads planned exactly as Python's store plans them).
//!
//! Every method takes `&self` and may be called from any thread.

use std::collections::HashMap;
use std::ffi::CString;
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use weightsift_io::engine::{Engine, EngineConfig, Prefetch, Request, Served, SlotBuffers};
use weightsift_io::{AlignedBuffer, CacheStats, Entry, FileTable, IoStats, RawBuffer, Segment, DIRECT_ALIGNMENT};

use crate::catalog::Catalog;

/// What one prefetch allocates for its own staging (`weightsift_io::engine`'s prefetch slots).
pub const PREFETCH_STAGING_BYTES: u64 = 2 * (32 << 20);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoreOptions {
    /// The host cache's budget: every byte it holds, leased rows included.
    pub host_cache_bytes: u64,
    /// Direct I/O (no OS page cache).
    pub direct: bool,
    /// Reader threads.
    pub workers: usize,
    /// Staging for loads: `staging_sets` acquires load at once, each through `staging_slots` slots of
    /// `staging_slot_bytes` (a multiple of 4 KiB), allocated when the store opens.
    pub staging_sets: usize,
    pub staging_slots: usize,
    pub staging_slot_bytes: u64,
    /// Prefetches in flight at most (each allocates `PREFETCH_STAGING_BYTES` while it runs); more are refused.
    pub max_prefetches: usize,
    /// The cache's block; 0: the largest part, so that every part is held in one piece. A smaller block is refused.
    pub cache_block_bytes: u64,
}

impl Default for StoreOptions {
    fn default() -> Self {
        StoreOptions {
            host_cache_bytes: 0,
            direct: true,
            workers: 8,
            staging_sets: 2,
            staging_slots: 2,
            staging_slot_bytes: 8 << 20,
            max_prefetches: 1,
            cache_block_bytes: 0,
        }
    }
}

impl StoreOptions {
    fn validate(&self) -> Result<()> {
        if self.host_cache_bytes == 0 {
            return Err(StoreError::Invalid("a store needs a host cache budget".into()));
        }
        if self.workers == 0 || self.staging_sets == 0 || self.staging_slots == 0 {
            return Err(StoreError::Invalid(
                "workers, staging sets and slots must be positive".into(),
            ));
        }
        if self.staging_slot_bytes == 0 || self.staging_slot_bytes % DIRECT_ALIGNMENT != 0 {
            return Err(StoreError::Invalid(format!(
                "staging slots must be a positive multiple of {DIRECT_ALIGNMENT} bytes"
            )));
        }
        Ok(())
    }

    /// The staging memory the store allocates when it opens.
    pub fn staging_bytes(&self) -> u64 {
        (self.staging_sets * self.staging_slots) as u64 * self.staging_slot_bytes
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StoreError {
    /// The catalog is malformed or inconsistent, or does not match the files on disk.
    Catalog(String),
    /// A bad argument: an unknown layer or expert, bad options.
    Invalid(String),
    /// The expert cannot be leased now (no room without evicting a leased row, larger than the cache, or a load it
    /// waited for was given up). Nothing is held; release leases and retry.
    Unavailable(String),
    /// A file could not be opened or read.
    Io(String),
    Closed,
    Internal(String),
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreError::Catalog(m) => write!(f, "catalog: {m}"),
            StoreError::Invalid(m) => write!(f, "{m}"),
            StoreError::Unavailable(m) => write!(f, "unavailable: {m}"),
            StoreError::Io(m) => write!(f, "I/O: {m}"),
            StoreError::Closed => write!(f, "the store is closed"),
            StoreError::Internal(m) => write!(f, "internal error: {m}"),
        }
    }
}

impl std::error::Error for StoreError {}

impl From<weightsift_io::Error> for StoreError {
    fn from(error: weightsift_io::Error) -> Self {
        use weightsift_io::Error as E;
        match error {
            E::Invalid(m) | E::Index(m) => StoreError::Invalid(m),
            E::Unavailable(m) => StoreError::Unavailable(m),
            e @ (E::Io { .. } | E::ShortRead { .. }) => StoreError::Io(e.to_string()),
            E::Cancelled => StoreError::Unavailable("the load was cancelled".into()),
            E::Closed => StoreError::Closed,
            E::Internal(m) => StoreError::Internal(m),
        }
    }
}

pub type Result<T> = std::result::Result<T, StoreError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExpertKey {
    pub layer: u32,
    pub expert: u32,
}

/// One part of a layer's experts: expert e is row `first_row + e` of the engine's segment `segment`.
#[derive(Debug)]
pub struct Part {
    pub name: String,
    pub segment: u32,
    pub first_row: u64,
    pub bytes: u64,
}

#[derive(Debug)]
pub struct Scales {
    pub part: usize,
    pub offset: u64,
    pub bytes: u64,
    pub format: String,
    pub(crate) c_format: CString,
}

/// A named range of an expert's parts, with the shape and the storage format the catalog gives it.
#[derive(Debug)]
pub struct View {
    pub name: String,
    pub part: usize,
    pub offset: u64,
    pub bytes: u64,
    pub format: String,
    pub rows: i64,
    pub columns: i64,
    pub block_rows: u32,
    pub block_columns: u32,
    pub scales: Option<Scales>,
    pub(crate) c_name: CString,
    pub(crate) c_format: CString,
}

#[derive(Debug)]
pub struct Layer {
    pub layer: u32,
    pub experts: u32,
    pub parts: Vec<Part>,
    pub views: Vec<View>,
}

impl Layer {
    pub fn view_index(&self, name: &str) -> Option<usize> {
        self.views.iter().position(|v| v.name == name)
    }

    /// Bytes of one expert (all its parts).
    pub fn expert_bytes(&self) -> u64 {
        self.parts.iter().map(|p| p.bytes).sum()
    }
}

fn c_string(value: &str) -> CString {
    CString::new(value).expect("the catalog refuses strings with NUL")
}

/// Leases alive and their bytes, shared by the store and its leases (a lease may outlive the store).
#[derive(Default)]
struct LeaseCounts {
    active: AtomicU64,
    bytes: AtomicU64,
    peak_bytes: AtomicU64,
    releases: AtomicU64,
}

/// An expert leased from the store: its parts' bytes, valid and unchanged until the lease is dropped.
pub struct Lease {
    key: ExpertKey,
    layer: Arc<Layer>,
    parts: Vec<Arc<Entry>>,
    bytes: u64,
    counts: Arc<LeaseCounts>,
}

/// A view of a leased expert: its bytes and its scales' bytes, in place.
pub struct ViewBytes<'a> {
    pub view: &'a View,
    pub data: &'a [u8],
    pub scales: Option<&'a [u8]>,
}

impl Lease {
    pub fn key(&self) -> ExpertKey {
        self.key
    }

    pub fn layer(&self) -> &Layer {
        &self.layer
    }

    pub fn parts(&self) -> usize {
        self.parts.len()
    }

    /// Part `index`'s bytes, in place in the host cache.
    pub fn part(&self, index: usize) -> &[u8] {
        self.parts[index]
            .contiguous()
            .expect("the store's cache holds every part in one piece")
    }

    pub fn view(&self, index: usize) -> Option<ViewBytes<'_>> {
        let view = self.layer.views.get(index)?;
        let range = |part: usize, offset: u64, bytes: u64| &self.part(part)[offset as usize..(offset + bytes) as usize];
        Some(ViewBytes {
            view,
            data: range(view.part, view.offset, view.bytes),
            scales: view.scales.as_ref().map(|s| range(s.part, s.offset, s.bytes)),
        })
    }

    /// Bytes of the expert's parts.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.counts.active.fetch_sub(1, Ordering::AcqRel);
        self.counts.bytes.fetch_sub(self.bytes, Ordering::AcqRel);
        self.counts.releases.fetch_add(1, Ordering::AcqRel);
    }
}

/// Staging sets for loads, allocated once: an acquire takes a whole set before it looks anything up, so a load in
/// flight always has its staging and never waits for another's.
struct Staging {
    sets: Mutex<Vec<Vec<AlignedBuffer>>>,
    free: Condvar,
}

impl Staging {
    fn new(options: &StoreOptions) -> Self {
        let sets = (0..options.staging_sets)
            .map(|_| {
                (0..options.staging_slots)
                    .map(|_| AlignedBuffer::new(options.staging_slot_bytes as usize, DIRECT_ALIGNMENT as usize))
                    .collect()
            })
            .collect();
        Staging {
            sets: Mutex::new(sets),
            free: Condvar::new(),
        }
    }

    fn take(&self) -> Vec<AlignedBuffer> {
        let mut sets = self.sets.lock().unwrap();
        loop {
            if let Some(set) = sets.pop() {
                return set;
            }
            sets = self.free.wait(sets).unwrap();
        }
    }

    fn give(&self, set: Vec<AlignedBuffer>) {
        self.sets.lock().unwrap().push(set);
        self.free.notify_one();
    }
}

/// A staging set taken from the pool, given back when dropped.
struct StagingSet<'a> {
    pool: &'a Staging,
    buffers: Vec<AlignedBuffer>,
}

impl<'a> StagingSet<'a> {
    fn take(pool: &'a Staging) -> Self {
        StagingSet {
            pool,
            buffers: pool.take(),
        }
    }
}

impl Drop for StagingSet<'_> {
    fn drop(&mut self) {
        self.pool.give(std::mem::take(&mut self.buffers));
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Counters {
    acquire_calls: u64,
    acquire_failures: u64,
    experts: u64,
    expert_hits: u64,
    expert_waits: u64,
    expert_misses: u64,
    prefetch_calls: u64,
    prefetch_refused: u64,
    prefetch_experts: u64,
    prefetch_rows: u64,
}

/// The store's counters: its own (experts, leases, prefetches) and its engine's and cache's (`weightsift-io`).
#[derive(Clone, Debug, Default)]
pub struct StoreStats {
    pub acquire_calls: u64,
    pub acquire_failures: u64,
    /// Experts leased; each is a hit (every part cached), a wait (no part read by the call, some loaded by another),
    /// or a miss (some part read by the call).
    pub experts: u64,
    pub expert_hits: u64,
    pub expert_waits: u64,
    pub expert_misses: u64,
    pub releases: u64,
    pub active_leases: u64,
    /// Bytes of the parts of the leases alive (a row two leases share counted twice), and the peak.
    pub leased_bytes: u64,
    pub peak_leased_bytes: u64,
    pub prefetch_calls: u64,
    pub prefetch_refused: u64,
    pub prefetch_experts: u64,
    /// Rows prefetches admitted to load (rows cached or loading already are not counted).
    pub prefetch_rows: u64,
    pub prefetches_in_flight: u64,
    pub staging_bytes: u64,
    pub io: IoStats,
    pub cache: CacheStats,
    pub cache_capacity_bytes: u64,
    pub cache_block_bytes: u64,
    pub cache_resident_bytes: u64,
    pub cache_held_bytes: u64,
    pub cache_peak_held_bytes: u64,
}

pub struct ExpertStore {
    name: String,
    engine: Engine,
    layers: HashMap<u32, Arc<Layer>>,
    staging: Staging,
    prefetches: Mutex<Vec<Prefetch>>,
    counters: Mutex<Counters>,
    leases: Arc<LeaseCounts>,
    options: StoreOptions,
    closed: AtomicBool,
}

impl ExpertStore {
    /// Open the store a catalog file describes (relative paths in it are relative to its directory).
    pub fn open(catalog: &Path, options: StoreOptions) -> Result<Self> {
        let (catalog, base) = Catalog::from_path(catalog).map_err(StoreError::Catalog)?;
        Self::from_catalog(&catalog, &base, options)
    }

    pub fn from_catalog(catalog: &Catalog, base: &Path, options: StoreOptions) -> Result<Self> {
        catalog.validate().map_err(StoreError::Catalog)?;
        options.validate()?;
        let paths = (0..catalog.files.len())
            .map(|k| (catalog.files[k].key.clone(), catalog.file_path(k, base)))
            .collect();
        let table = FileTable::open(paths, options.direct).map_err(|e| StoreError::Io(e.to_string()))?;
        for (file, spec) in catalog.files.iter().zip(&table.files) {
            if let Some(size) = file.size {
                if size != spec.size {
                    return Err(StoreError::Catalog(format!(
                        "{} holds {} bytes, the catalog says {size}",
                        spec.path.display(),
                        spec.size
                    )));
                }
            }
        }
        let segments = catalog
            .segments
            .iter()
            .map(|s| match (&s.offset, &s.part_bytes, &s.spans) {
                (Some(offset), _, _) => Segment::plain(&s.name, s.files[0], *offset, s.rows, s.row_bytes),
                (None, Some(part_bytes), Some(spans)) => {
                    Segment::composed(&s.name, s.files.clone(), s.rows, part_bytes.clone(), spans.clone())
                }
                _ => unreachable!("validated"),
            })
            .collect::<weightsift_io::Result<Vec<_>>>()?;
        // Every part a lease hands out is held in one piece: the cache's block is at least the largest part.
        let largest = catalog
            .layers
            .iter()
            .flat_map(|l| l.parts.iter().map(|p| catalog.segments[p.segment as usize].row_bytes))
            .max()
            .unwrap_or(1);
        let block = match options.cache_block_bytes {
            0 => largest,
            block if block >= largest => block,
            block => {
                return Err(StoreError::Invalid(format!(
                    "a cache block of {block} bytes would split parts of {largest} bytes"
                )))
            }
        };
        let config = EngineConfig {
            direct: options.direct,
            workers: options.workers,
            host_cache_bytes: options.host_cache_bytes,
            cache_block_bytes: block,
            ..EngineConfig::default()
        };
        let engine = Engine::new(table, segments, config).map_err(|e| match e {
            weightsift_io::Error::Invalid(m) => StoreError::Catalog(m),
            e => e.into(),
        })?;
        let layers = catalog
            .layers
            .iter()
            .map(|l| {
                let parts = l
                    .parts
                    .iter()
                    .map(|p| Part {
                        name: p.name.clone(),
                        segment: p.segment,
                        first_row: p.first_row,
                        bytes: catalog.segments[p.segment as usize].row_bytes,
                    })
                    .collect();
                let views = l
                    .views
                    .iter()
                    .map(|v| View {
                        name: v.name.clone(),
                        part: v.part as usize,
                        offset: v.offset,
                        bytes: v.bytes,
                        format: v.format.clone(),
                        rows: v.rows,
                        columns: v.columns,
                        block_rows: v.block_rows,
                        block_columns: v.block_columns,
                        scales: v.scales.as_ref().map(|s| Scales {
                            part: s.part as usize,
                            offset: s.offset,
                            bytes: s.bytes,
                            format: s.format.clone(),
                            c_format: c_string(&s.format),
                        }),
                        c_name: c_string(&v.name),
                        c_format: c_string(&v.format),
                    })
                    .collect();
                let layer = Layer {
                    layer: l.layer,
                    experts: l.experts,
                    parts,
                    views,
                };
                (l.layer, Arc::new(layer))
            })
            .collect();
        Ok(ExpertStore {
            name: catalog.name.clone(),
            engine,
            layers,
            staging: Staging::new(&options),
            prefetches: Mutex::new(Vec::new()),
            counters: Mutex::new(Counters::default()),
            leases: Arc::new(LeaseCounts::default()),
            options,
            closed: AtomicBool::new(false),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn options(&self) -> StoreOptions {
        self.options
    }

    pub fn layer(&self, layer: u32) -> Option<&Arc<Layer>> {
        self.layers.get(&layer)
    }

    /// The layers, in ascending order.
    pub fn layer_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self.layers.keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    fn requests(&self, keys: &[ExpertKey]) -> Result<(Vec<Request>, Vec<Arc<Layer>>)> {
        let mut requests = Vec::new();
        let mut layers = Vec::with_capacity(keys.len());
        for key in keys {
            let layer = self
                .layers
                .get(&key.layer)
                .ok_or_else(|| StoreError::Invalid(format!("no layer {}", key.layer)))?;
            if key.expert >= layer.experts {
                return Err(StoreError::Invalid(format!(
                    "layer {} has {} experts, not {}",
                    key.layer,
                    layer.experts,
                    key.expert + 1
                )));
            }
            for part in &layer.parts {
                requests.push(Request {
                    segment: part.segment,
                    rows: Some(vec![(part.first_row + key.expert as u64) as i64]),
                    positions: None,
                });
            }
            layers.push(Arc::clone(layer));
        }
        Ok((requests, layers))
    }

    /// Lease one expert (blocks while its parts load).
    pub fn acquire(&self, key: ExpertKey) -> Result<Lease> {
        Ok(self.acquire_many(&[key])?.pop().expect("one lease per key"))
    }

    /// Lease several experts in one call (one plan per segment for everything missing): all of them, or none.
    pub fn acquire_many(&self, keys: &[ExpertKey]) -> Result<Vec<Lease>> {
        if self.closed.load(Ordering::Acquire) {
            return Err(StoreError::Closed);
        }
        let (requests, layers) = self.requests(keys)?;
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let outcome = {
            let mut set = StagingSet::take(&self.staging);
            let slots: Vec<SlotBuffers> = set
                .buffers
                .iter_mut()
                .map(|b| SlotBuffers {
                    staging: RawBuffer::from_slice(b.as_mut_slice()),
                    compact: None,
                })
                .collect();
            // `acquire` returns only once none of its tasks touches the slots; the set then goes back to the pool.
            self.engine.acquire(&requests, slots)
        };
        let leased = match outcome {
            Ok(leased) => leased,
            Err(error) => {
                let mut counters = self.counters.lock().unwrap();
                counters.acquire_calls += 1;
                counters.acquire_failures += 1;
                return Err(error.into());
            }
        };
        let mut entries = leased.entries.into_iter();
        let mut served = leased.served.into_iter();
        let mut leases = Vec::with_capacity(keys.len());
        let (mut hits, mut waits, mut misses) = (0, 0, 0);
        for (&key, layer) in keys.iter().zip(layers) {
            let n = layer.parts.len();
            let parts: Vec<Arc<Entry>> = entries.by_ref().take(n).collect();
            let how: Vec<Served> = served.by_ref().take(n).collect();
            if how.contains(&Served::Loaded) {
                misses += 1;
            } else if how.contains(&Served::Waited) {
                waits += 1;
            } else {
                hits += 1;
            }
            let bytes = layer.expert_bytes();
            self.leases.active.fetch_add(1, Ordering::AcqRel);
            let now = self.leases.bytes.fetch_add(bytes, Ordering::AcqRel) + bytes;
            self.leases.peak_bytes.fetch_max(now, Ordering::AcqRel);
            leases.push(Lease {
                key,
                layer,
                parts,
                bytes,
                counts: Arc::clone(&self.leases),
            });
        }
        let mut counters = self.counters.lock().unwrap();
        counters.acquire_calls += 1;
        counters.experts += keys.len() as u64;
        counters.expert_hits += hits;
        counters.expert_waits += waits;
        counters.expert_misses += misses;
        Ok(leases)
    }

    /// Start loading experts into the host cache in the background (advisory; nothing is leased). Returns the rows
    /// admitted for loading (0 if they are all cached or loading already); `Unavailable` when `max_prefetches`
    /// prefetches are still in flight (refused, and counted as such).
    pub fn prefetch(&self, keys: &[ExpertKey]) -> Result<u64> {
        if self.closed.load(Ordering::Acquire) {
            return Err(StoreError::Closed);
        }
        let (requests, _) = self.requests(keys)?;
        let mut running = self.prefetches.lock().unwrap();
        running.retain(|p| !p.is_done());
        let mut counters = self.counters.lock().unwrap();
        counters.prefetch_calls += 1;
        if running.len() >= self.options.max_prefetches {
            counters.prefetch_refused += 1;
            return Err(StoreError::Unavailable(format!(
                "{} prefetches are in flight",
                running.len()
            )));
        }
        drop(counters);
        let prefetch = self.engine.prefetch(&requests)?;
        let rows = prefetch.rows();
        if rows > 0 {
            running.push(prefetch);
        }
        let mut counters = self.counters.lock().unwrap();
        counters.prefetch_experts += keys.len() as u64;
        counters.prefetch_rows += rows;
        Ok(rows)
    }

    /// Wait for every prefetch in flight (tests and benchmarks).
    pub fn wait_prefetches(&self) -> Result<()> {
        let running: Vec<Prefetch> = std::mem::take(&mut *self.prefetches.lock().unwrap());
        for prefetch in &running {
            prefetch.wait()?;
        }
        Ok(())
    }

    pub fn active_leases(&self) -> u64 {
        self.leases.active.load(Ordering::Acquire)
    }

    pub fn stats(&self) -> StoreStats {
        let c = *self.counters.lock().unwrap();
        let in_flight = {
            let mut running = self.prefetches.lock().unwrap();
            running.retain(|p| !p.is_done());
            running.len() as u64
        };
        let cache = self.engine.cache().expect("a store has a host cache");
        StoreStats {
            acquire_calls: c.acquire_calls,
            acquire_failures: c.acquire_failures,
            experts: c.experts,
            expert_hits: c.expert_hits,
            expert_waits: c.expert_waits,
            expert_misses: c.expert_misses,
            releases: self.leases.releases.load(Ordering::Acquire),
            active_leases: self.leases.active.load(Ordering::Acquire),
            leased_bytes: self.leases.bytes.load(Ordering::Acquire),
            peak_leased_bytes: self.leases.peak_bytes.load(Ordering::Acquire),
            prefetch_calls: c.prefetch_calls,
            prefetch_refused: c.prefetch_refused,
            prefetch_experts: c.prefetch_experts,
            prefetch_rows: c.prefetch_rows,
            prefetches_in_flight: in_flight,
            staging_bytes: self.options.staging_bytes() + in_flight * PREFETCH_STAGING_BYTES,
            io: self.engine.stats(),
            cache: cache.stats(),
            cache_capacity_bytes: cache.capacity(),
            cache_block_bytes: cache.block_bytes(),
            cache_resident_bytes: cache.resident_bytes(),
            cache_held_bytes: cache.held_bytes(),
            cache_peak_held_bytes: cache.peak_held_bytes(),
        }
    }

    /// Start a new window of counters (the cache's contents and the leases are kept).
    pub fn reset_stats(&self) {
        *self.counters.lock().unwrap() = Counters::default();
        self.leases.releases.store(0, Ordering::Release);
        self.leases
            .peak_bytes
            .store(self.leases.bytes.load(Ordering::Acquire), Ordering::Release);
        self.engine.reset_stats(false);
    }

    /// Stop: prefetches are cancelled, the readers stop, and the cache gives back every row nobody leases. Leases still
    /// alive keep their rows (and stay valid) until dropped. Later calls fail with `Closed`.
    pub fn close(&self) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        drop(std::mem::take(&mut *self.prefetches.lock().unwrap()));
        self.engine.close();
    }
}

impl Drop for ExpertStore {
    fn drop(&mut self) {
        self.close();
    }
}
