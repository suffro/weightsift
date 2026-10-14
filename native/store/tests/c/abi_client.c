/*
 * A C client of the expert store's ABI (include/weightsift_store.h), as a C execution engine would use it. The Rust
 * integration tests (tests/abi.rs) call these functions on fixture catalogs and compare what they report with the
 * files' bytes and with the Rust structs' layout.
 */
#include "weightsift_store.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition)                                                                   \
    do {                                                                                   \
        if (!(condition)) {                                                                \
            snprintf(message, message_size, "abi_client.c:%d: %s", __LINE__, #condition); \
            return __LINE__;                                                               \
        }                                                                                  \
    } while (0)

/* The sizes and a few offsets of the header's structs, as this compiler lays them out. */
void ws_abi_client_layout(uint64_t *out, size_t count) {
    const uint64_t values[] = {
        sizeof(ws_store_options),
        sizeof(ws_expert_key),
        sizeof(ws_span),
        sizeof(ws_view),
        sizeof(ws_store_stats),
        offsetof(ws_store_options, staging_slot_bytes),
        offsetof(ws_store_options, cache_block_bytes),
        offsetof(ws_view, data),
        offsetof(ws_view, scale_bytes),
        offsetof(ws_view, block_columns),
        offsetof(ws_store_stats, prefetch_used),
        offsetof(ws_store_stats, cache_lookups),
        offsetof(ws_store_stats, staging_bytes),
    };
    for (size_t k = 0; k < count && k < sizeof(values) / sizeof(values[0]); k++) out[k] = values[k];
}

static uint64_t fnv1a(const uint8_t *data, uint64_t bytes) {
    uint64_t hash = 0xcbf29ce484222325ull;
    for (uint64_t k = 0; k < bytes; k++) {
        hash ^= data[k];
        hash *= 0x100000001b3ull;
    }
    return hash;
}

/*
 * The whole lifecycle on one expert: open, layer and view lookups, acquire, parts and views in place, a second lease of
 * the same expert (a hit, the same bytes), stats, close refused while leases live, double release, errors, prefetch,
 * close, and every call on the closed handle. checksums[k] receives the FNV-1a hash of part k; returns 0, or the line of
 * the first failed check (described in message).
 */
int ws_abi_client_lifecycle(const char *catalog, uint64_t cache_bytes, uint32_t layer, uint32_t expert,
                            uint64_t *checksums, size_t checksum_count, char *message, size_t message_size) {
    char error[256];
    CHECK(ws_abi_version() == WS_STORE_ABI_VERSION);

    ws_store_options options;
    memset(&options, 0, sizeof(options));
    options.struct_size = sizeof(options);
    CHECK(ws_store_options_default(&options) == WS_OK);
    CHECK(options.struct_size == sizeof(options) && options.host_cache_bytes == 0 && options.workers > 0);
    uint64_t store = 0;
    /* No budget: refused, with a message. */
    CHECK(ws_store_open(catalog, &options, &store, error, sizeof(error)) == WS_ERR_INVALID_ARGUMENT);
    CHECK(store == 0 && strstr(error, "budget") != NULL);
    options.host_cache_bytes = cache_bytes;
    CHECK(ws_store_open(catalog, &options, &store, error, sizeof(error)) == WS_OK && store != 0);

    uint32_t ids[4];
    size_t layers = 0;
    CHECK(ws_store_layers(store, ids, 4, &layers) == WS_OK && layers >= 1 && layers <= 4);
    int listed = 0;
    for (size_t k = 0; k < layers; k++) listed |= ids[k] == layer;
    CHECK(listed);
    CHECK(ws_store_layers(store, NULL, 0, &layers) == WS_OK && layers >= 1);
    uint32_t experts = 0, parts = 0, views = 0, gate = 0;
    CHECK(ws_store_layer_info(store, layer, &experts, &parts, &views) == WS_OK);
    CHECK(expert < experts && parts == checksum_count && views >= 1);
    CHECK(ws_store_view_index(store, layer, "gate", &gate) == WS_OK);
    CHECK(ws_store_view_index(store, layer, "no-such-view", &gate) == WS_ERR_INVALID_ARGUMENT);
    ws_view info;
    memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    CHECK(ws_store_view_info(store, layer, gate, &info) == WS_OK);
    CHECK(info.data == NULL && info.scales == NULL && info.bytes > 0 && strcmp(info.name, "gate") == 0);
    CHECK(ws_store_layer_info(store, 9999, &experts, &parts, &views) == WS_ERR_INVALID_ARGUMENT);

    ws_expert_key key = {layer, expert};
    uint64_t lease = 0;
    CHECK(ws_store_acquire(store, key, &lease, error, sizeof(error)) == WS_OK && lease != 0);
    ws_expert_key back;
    CHECK(ws_lease_key(store, lease, &back) == WS_OK && back.layer == layer && back.expert == expert);
    ws_span first[8];
    CHECK(parts <= 8);
    for (uint32_t p = 0; p < parts; p++) {
        CHECK(ws_lease_part(store, lease, p, &first[p]) == WS_OK && first[p].data != NULL && first[p].bytes > 0);
        checksums[p] = fnv1a(first[p].data, first[p].bytes);
    }
    ws_span span;
    CHECK(ws_lease_part(store, lease, parts, &span) == WS_ERR_INVALID_ARGUMENT);
    for (uint32_t v = 0; v < views; v++) {
        ws_view view;
        memset(&view, 0, sizeof(view));
        view.struct_size = sizeof(view);
        CHECK(ws_lease_view(store, lease, v, &view) == WS_OK);
        CHECK(view.struct_size == sizeof(view) && view.name != NULL && view.format != NULL);
        CHECK(view.part < parts && view.data >= first[view.part].data &&
              view.data + view.bytes <= first[view.part].data + first[view.part].bytes);
        CHECK((view.scales == NULL) == (view.scale_format == NULL) && (view.scales == NULL) == (view.scale_bytes == 0));
    }

    /* The same expert again: a hit, served by the same bytes in place. */
    uint64_t again = 0;
    CHECK(ws_store_acquire(store, key, &again, error, sizeof(error)) == WS_OK && again != lease);
    CHECK(ws_lease_part(store, again, 0, &span) == WS_OK && span.data == first[0].data);
    ws_store_stats stats;
    memset(&stats, 0, sizeof(stats));
    stats.struct_size = sizeof(stats);
    CHECK(ws_store_get_stats(store, &stats) == WS_OK && stats.abi_version == WS_STORE_ABI_VERSION);
    CHECK(stats.experts == 2 && stats.expert_misses == 1 && stats.expert_hits == 1 && stats.active_leases == 2);
    CHECK(stats.physical_bytes > 0 && stats.cache_resident_bytes <= stats.cache_capacity_bytes);

    /* Close is refused while leases live, and changes nothing. */
    CHECK(ws_store_close(store, error, sizeof(error)) == WS_ERR_LEASES_OUTSTANDING);
    CHECK(ws_lease_part(store, lease, 0, &span) == WS_OK && span.data == first[0].data);

    /* Release once each; a second release, an unknown lease and lease 0 are errors, not crashes. */
    CHECK(ws_store_release(store, lease) == WS_OK);
    CHECK(ws_store_release(store, lease) == WS_ERR_INVALID_LEASE);
    CHECK(ws_lease_part(store, lease, 0, &span) == WS_ERR_INVALID_LEASE);
    CHECK(ws_store_release(store, again) == WS_OK);
    CHECK(ws_store_release(store, 0) == WS_ERR_INVALID_LEASE);

    /* Bad requests fail and hold nothing. */
    ws_expert_key bad = {layer, experts};
    CHECK(ws_store_acquire(store, bad, &lease, error, sizeof(error)) == WS_ERR_INVALID_ARGUMENT && lease == 0);
    CHECK(ws_store_acquire(store, key, NULL, error, sizeof(error)) == WS_ERR_INVALID_ARGUMENT);
    uint64_t rows = 99;
    ws_expert_key other = {layer, (expert + 1) % experts};
    CHECK(ws_store_prefetch(store, &other, 1, &rows) == WS_OK);
    CHECK(ws_store_get_stats(store, &stats) == WS_OK && stats.active_leases == 0 && stats.releases == 2);

    /* Close; then the handle is invalid everywhere. */
    CHECK(ws_store_close(store, error, sizeof(error)) == WS_OK);
    CHECK(ws_store_acquire(store, key, &lease, error, sizeof(error)) == WS_ERR_INVALID_HANDLE);
    CHECK(ws_store_release(store, again) == WS_ERR_INVALID_HANDLE);
    CHECK(ws_store_get_stats(store, &stats) == WS_ERR_INVALID_HANDLE);
    CHECK(ws_store_close(store, error, sizeof(error)) == WS_ERR_INVALID_HANDLE);
    return 0;
}
