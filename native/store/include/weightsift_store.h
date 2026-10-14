/*
 * weightsift_store.h: the C ABI of Weightsift's expert store (Phase 7A, decision 0014), version 1.
 *
 * The weight storage backend of Weightsift's flagship runtime, for execution engines written in C or C++. A store
 * serves the routed experts a catalog describes (`weightsift-expert-catalog` v1, written by a model adapter from a
 * checkpoint's headers): it reads them from storage (direct I/O), keeps them in a host-RAM cache under a strict byte
 * budget (least recently used first), and leases them to the engine, which reads their bytes in place.
 *
 * Handles. A store and a lease are 64-bit handles, never pointers, and never reused: an unknown, closed or released
 * handle is reported (WS_ERR_INVALID_HANDLE, WS_ERR_INVALID_LEASE), never dereferenced. A lease belongs to the store
 * that issued it.
 *
 * Lifetimes and ownership.
 *   - ws_store_acquire: on WS_OK the caller owns one lease and must release it exactly once (ws_store_release). On any
 *     other status nothing is held and *lease is 0.
 *   - The bytes a lease exposes (ws_lease_part, ws_lease_view) stay valid, unmoved and unchanged from the acquire until
 *     the release; the cache never evicts a leased row. They are read-only.
 *   - Strings the store returns (view names and formats) stay valid until the store is closed.
 *   - ws_store_close fails with WS_ERR_LEASES_OUTSTANDING while any lease is alive (nothing changes then); after a
 *     successful close the handle is invalid.
 *
 * Budgets. The cache never holds more than ws_store_options.host_cache_bytes, leased rows and rows being loaded
 * included. When an expert cannot be made resident without evicting a leased row, ws_store_acquire returns
 * WS_ERR_UNAVAILABLE at once and holds nothing: release leases and retry (the store never waits for room). Staging for
 * loads (staging_sets x staging_slots x staging_slot_bytes) is allocated when the store opens; a prefetch in flight
 * holds 64 MiB more.
 *
 * Threads. Every function may be called from any thread, concurrently, on the same store; ws_store_acquire blocks
 * while the expert loads (a row several callers want is read once, the others wait for it). ws_store_close waits for
 * acquires in flight. A lease may be released from another thread than the one that acquired it.
 *
 * Errors. Functions return a ws_status. Those taking (error, error_size) write a NUL-terminated message there on
 * failure when error is not NULL (truncated to fit). A panic inside the store never crosses this boundary: it becomes
 * WS_ERR_INTERNAL.
 *
 * Versioning. Every struct the caller passes or receives starts with struct_size: set it to sizeof the struct of the
 * header you compile against. The store reads at most that many bytes of an input, and for an output writes what it
 * knows and zeroes the rest of the caller's struct; it sets struct_size to the size it filled. ws_abi_version()
 * returns WS_STORE_ABI_VERSION of the library; a caller refuses a library whose major version differs.
 */
#ifndef WEIGHTSIFT_STORE_H
#define WEIGHTSIFT_STORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WS_STORE_ABI_VERSION 1u

typedef int32_t ws_status;

#define WS_OK 0
#define WS_ERR_INVALID_ARGUMENT (-1)   /* a NULL pointer, a bad struct_size, an unknown layer, expert, part or view */
#define WS_ERR_INVALID_HANDLE (-2)     /* not an open store */
#define WS_ERR_INVALID_LEASE (-3)      /* not a live lease of this store (released already, or never issued) */
#define WS_ERR_UNAVAILABLE (-4)        /* cannot lease now: release leases and retry */
#define WS_ERR_IO (-5)                 /* a file could not be opened or read */
#define WS_ERR_CATALOG (-6)            /* the catalog is malformed, inconsistent, or does not match the files */
#define WS_ERR_LEASES_OUTSTANDING (-7) /* close refused: leases are alive */
#define WS_ERR_CLOSED (-8)             /* the store is closing or closed */
#define WS_ERR_INTERNAL (-9)           /* a bug of the store (a caught panic) */

typedef struct {
    uint32_t struct_size;
    uint32_t direct;              /* 1: direct I/O (no OS page cache) */
    uint64_t host_cache_bytes;    /* the cache's budget; required */
    uint32_t workers;             /* reader threads */
    uint32_t staging_sets;        /* acquires that load at once */
    uint32_t staging_slots;       /* slots per staging set */
    uint32_t max_prefetches;      /* prefetches in flight at most; more are refused */
    uint64_t staging_slot_bytes;  /* a multiple of 4096 */
    uint64_t cache_block_bytes;   /* 0: the largest part (every part held in one piece); smaller is refused */
} ws_store_options;

typedef struct {
    uint32_t layer;
    uint32_t expert;
} ws_expert_key;

typedef struct {
    const uint8_t *data;
    uint64_t bytes;
} ws_span;

/* A named range of a leased expert's parts, with the shape and storage format the catalog gives it. The store never
 * interprets formats: they are the catalog's strings. */
typedef struct {
    uint32_t struct_size;
    uint32_t part;                /* the part holding data */
    const char *name;
    const char *format;
    const uint8_t *data;
    uint64_t bytes;
    const char *scale_format;     /* NULL when the view has no scales */
    const uint8_t *scales;
    uint64_t scale_bytes;
    int64_t rows;
    int64_t columns;
    uint32_t block_rows;
    uint32_t block_columns;
} ws_view;

typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;
    /* experts and leases */
    uint64_t acquire_calls;
    uint64_t acquire_failures;
    uint64_t experts;             /* experts leased: hits + waits + misses */
    uint64_t expert_hits;         /* every part was cached */
    uint64_t expert_waits;        /* no part read by the call; some loaded by a prefetch or another acquire */
    uint64_t expert_misses;       /* some part read by the call */
    uint64_t releases;
    uint64_t active_leases;
    uint64_t leased_bytes;
    uint64_t peak_leased_bytes;
    /* prefetches */
    uint64_t prefetch_calls;
    uint64_t prefetch_refused;
    uint64_t prefetch_experts;
    uint64_t prefetch_rows;       /* rows admitted to load */
    uint64_t prefetch_used;       /* prefetched rows a request then used */
    uint64_t prefetch_wasted;     /* prefetched rows evicted unused */
    /* storage (what the reads returned) */
    uint64_t requests;
    uint64_t rows;
    uint64_t logical_bytes;
    uint64_t physical_bytes;
    uint64_t read_calls;
    uint64_t extents;
    uint64_t blocks_4k;
    uint64_t read_ns;             /* sum of the read calls' durations */
    uint64_t busy_ns;             /* time with a read in flight */
    /* the host cache (rows) */
    uint64_t cache_lookups;
    uint64_t cache_hits;
    uint64_t cache_waits;
    uint64_t cache_misses;
    uint64_t cache_inserts;
    uint64_t cache_evictions;
    uint64_t cache_bypassed;
    uint64_t cache_capacity_bytes;
    uint64_t cache_block_bytes;
    uint64_t cache_resident_bytes;
    uint64_t cache_held_bytes;
    uint64_t cache_peak_held_bytes;
    /* memory outside the cache */
    uint64_t staging_bytes;       /* staging sets, plus prefetches in flight */
} ws_store_stats;

uint32_t ws_abi_version(void);
/* A short description of a status. */
const char *ws_status_string(ws_status status);

/* Fill *options with the defaults (direct I/O, 8 workers, 2 staging sets of 2 slots of 8 MiB, one prefetch in flight,
 * the block from the parts); host_cache_bytes is left 0 and must be set. */
ws_status ws_store_options_default(ws_store_options *options);

ws_status ws_store_open(const char *catalog_path, const ws_store_options *options, uint64_t *store, char *error,
                        size_t error_size);
ws_status ws_store_close(uint64_t store, char *error, size_t error_size);

/* The catalog's layers, ascending: up to `capacity` ids into layers (may be NULL when capacity is 0); *count receives
 * how many there are. */
ws_status ws_store_layers(uint64_t store, uint32_t *layers, size_t capacity, size_t *count);
/* The layer's expert count, parts and views. */
ws_status ws_store_layer_info(uint64_t store, uint32_t layer, uint32_t *experts, uint32_t *parts, uint32_t *views);
/* The index of the layer's view named `name`. */
ws_status ws_store_view_index(uint64_t store, uint32_t layer, const char *name, uint32_t *index);
/* What the catalog says of a layer's view, without a lease: data and scales are NULL, bytes and scale_bytes their
 * sizes. */
ws_status ws_store_view_info(uint64_t store, uint32_t layer, uint32_t view, ws_view *out);

ws_status ws_store_acquire(uint64_t store, ws_expert_key key, uint64_t *lease, char *error, size_t error_size);
/* Lease `count` experts at once (one load plan for all their missing parts): all of them (leases[0..count]) or none. */
ws_status ws_store_acquire_many(uint64_t store, const ws_expert_key *keys, size_t count, uint64_t *leases, char *error,
                                size_t error_size);
ws_status ws_store_release(uint64_t store, uint64_t lease);

ws_status ws_lease_key(uint64_t store, uint64_t lease, ws_expert_key *key);
ws_status ws_lease_part(uint64_t store, uint64_t lease, uint32_t part, ws_span *span);
ws_status ws_lease_view(uint64_t store, uint64_t lease, uint32_t view, ws_view *out);

/* Advisory: start loading experts into the cache in the background; *rows (optional) receives the rows admitted to
 * load (0 when they are all cached or loading already). WS_ERR_UNAVAILABLE: refused, max_prefetches are in flight. */
ws_status ws_store_prefetch(uint64_t store, const ws_expert_key *keys, size_t count, uint64_t *rows);

ws_status ws_store_get_stats(uint64_t store, ws_store_stats *stats);

#ifdef __cplusplus
}
#endif

#endif /* WEIGHTSIFT_STORE_H */
