/*
 * Test support for weightsift-colibri (linked into its tests only).
 *
 * ws_colibri_contract checks Colibrì's expert-store lease contract, as its c/expert_store.h states it, on a
 * ColiExpertStore served by Weightsift's store over real files: the points Colibrì's own tests/test_expert_store_ops.c
 * checks on a mock (a lookup fills the view; a failed lookup clears the view; release clears it; a second release and
 * the release of a zero view are no-ops; prefetch is advisory; stats count; destroy needs zero leases), through
 * Colibrì's own inline helpers (coli_expert_lookup, coli_expert_release).
 *
 * ws_colibri_segment_layout reports how this compiler lays out Colibrì's segment runtime structs, for the Rust mirrors.
 */
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "expert_store.h"
#include "segment_runtime.h"
#include "weightsift_store.h"
#include "ws_colibri_store.h"

#define CHECK(condition)                                                                       \
    do {                                                                                       \
        if (!(condition)) {                                                                    \
            snprintf(message, message_size, "ws_colibri_tests.c:%d: %s", __LINE__, #condition); \
            return __LINE__;                                                                   \
        }                                                                                      \
    } while (0)

void ws_colibri_segment_layout(uint64_t *out, size_t count) {
    const uint64_t values[] = {
        sizeof(ColiSegmentCapabilities),
        offsetof(ColiSegmentCapabilities, engine_id),
        offsetof(ColiSegmentCapabilities, state_dtype),
        offsetof(ColiSegmentCapabilities, reserved_u64),
        sizeof(ColiSegmentEngineOptions),
        offsetof(ColiSegmentEngineOptions, model_dir),
        offsetof(ColiSegmentEngineOptions, memory_limit_bytes),
        offsetof(ColiSegmentEngineOptions, resource_plan_size),
        sizeof(ColiSegmentSessionOptions),
        sizeof(ColiSegmentRunRequest),
        offsetof(ColiSegmentRunRequest, token_ids),
        offsetof(ColiSegmentRunRequest, output_bytes),
        offsetof(ColiSegmentRunRequest, should_cancel),
        offsetof(ColiSegmentRunRequest, reserved_u64),
    };
    for (size_t k = 0; k < count && k < sizeof(values) / sizeof(values[0]); k++) out[k] = values[k];
}

static uint64_t fnv1a(const void *data, size_t bytes) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t hash = 0xcbf29ce484222325ull;
    for (size_t k = 0; k < bytes; k++) {
        hash ^= p[k];
        hash *= 0x100000001b3ull;
    }
    return hash;
}

static int cleared(const ColiExpertView *view) {
    static const ColiExpertView zero;
    return memcmp(view, &zero, sizeof(*view)) == 0;
}

/*
 * Run the contract on expert (layer, expert) of the catalog. checksums receives the FNV-1a hashes of the gate, up and
 * down data and of their scales (0 when a view has none), for the Rust side to compare with the files; formats receives
 * the three ColiTensorFormat values. Returns 0, or the line of the first failed check (described in message).
 */
int ws_colibri_contract(const char *catalog, uint64_t cache_bytes, int layer, int expert, int experts,
                        uint64_t *checksums, int *formats, char *message, size_t message_size) {
    ws_store_options options;
    ColiExpertStore *store = NULL;
    ColiExpertView view, again, bad;
    ColiExpertKey key = {layer, expert};
    ColiExpertKey missing = {layer, experts};
    ColiExpertStoreStats stats;
    char error[256];

    memset(&options, 0, sizeof(options));
    options.struct_size = sizeof(options);
    CHECK(ws_store_options_default(&options) == WS_OK);
    options.host_cache_bytes = cache_bytes;
    options.staging_slot_bytes = 64 << 10;
    CHECK(ws_colibri_store_open(catalog, &options, &store, error, sizeof(error)) == 0 && store != NULL);
    CHECK(store->ops && store->ops->lookup && store->ops->release && store->ops->prefetch && store->ops->stats &&
          store->ops->destroy && store->gpu == NULL);

    /* A lookup fills the view, whatever the view held before. */
    memset(&view, 0x5a, sizeof(view));
    CHECK(coli_expert_lookup(store, key, &view) == 0);
    CHECK(view.key.layer == layer && view.key.expert == expert && view.lease != NULL);
    {
        const ColiTensorView *tensors[3] = {&view.gate, &view.up, &view.down};
        for (int k = 0; k < 3; k++) {
            CHECK(tensors[k]->data != NULL && tensors[k]->data_bytes > 0 && tensors[k]->gpu == NULL);
            CHECK((tensors[k]->scales == NULL) == (tensors[k]->scale_format == COLI_SCALE_NONE));
            checksums[k] = fnv1a(tensors[k]->data, tensors[k]->data_bytes);
            checksums[3 + k] = tensors[k]->scales ? fnv1a(tensors[k]->scales, tensors[k]->scale_bytes) : 0;
            formats[k] = (int)tensors[k]->format;
        }
    }

    /* A second lease of the same expert: the same bytes, in place. */
    memset(&again, 0, sizeof(again));
    CHECK(coli_expert_lookup(store, key, &again) == 0 && again.gate.data == view.gate.data);
    CHECK(again.lease != view.lease);

    /* A failed lookup clears the view. */
    memset(&bad, 0x5a, sizeof(bad));
    CHECK(coli_expert_lookup(store, missing, &bad) != 0 && cleared(&bad));
    {
        ColiExpertKey negative = {-1, 0};
        memset(&bad, 0x5a, sizeof(bad));
        CHECK(coli_expert_lookup(store, negative, &bad) != 0 && cleared(&bad));
    }

    /* Stats: two leases (a miss read from storage, a hit); the bad keys were refused before any lookup. */
    store->ops->stats(store, &stats);
    CHECK(stats.requests == 2 && stats.hits == 1 && stats.misses == 1 && stats.bytes_read > 0);
    CHECK(stats.resident_bytes > 0 && stats.resident_bytes <= stats.capacity_bytes);

    /* Destroy with leases alive is refused: nothing is freed, the leased bytes stay readable, releases still work. */
    store->ops->destroy(store);
    CHECK(fnv1a(view.gate.data, view.gate.data_bytes) == checksums[0]);

    /* Release clears the view; a second release and the release of a zero view are no-ops. */
    coli_expert_release(store, &view);
    CHECK(cleared(&view));
    coli_expert_release(store, &view);
    memset(&bad, 0, sizeof(bad));
    coli_expert_release(store, &bad);
    CHECK(cleared(&bad));
    /* A copied view's stale lease is refused by Weightsift, not double-freed. */
    {
        ColiExpertView copy = again;
        coli_expert_release(store, &again);
        CHECK(cleared(&again));
        coli_expert_release(store, &copy);
        CHECK(cleared(&copy));
    }

    /* Prefetch is advisory: the keys are accepted. */
    {
        ColiExpertKey next = {layer, (expert + 1) % experts};
        CHECK(store->ops->prefetch(store, &next, 1) == 1);
        CHECK(store->ops->prefetch(store, NULL, 0) == 0);
    }

    /* Zero leases: destroy succeeds (the store's memory is freed: nothing may touch it after). */
    {
        uint64_t handle = ws_colibri_store_handle(store);
        ws_store_stats s;
        memset(&s, 0, sizeof(s));
        s.struct_size = sizeof(s);
        CHECK(ws_store_get_stats(handle, &s) == WS_OK && s.active_leases == 0 && s.releases == 2);
        store->ops->destroy(store);
        CHECK(ws_store_get_stats(handle, &s) == WS_ERR_INVALID_HANDLE);
    }
    return 0;
}
