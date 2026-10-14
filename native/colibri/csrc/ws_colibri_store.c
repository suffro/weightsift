/*
 * Colibrì's ColiExpertStoreOps over Weightsift's store (see ws_colibri_store.h). A view's lease field carries the
 * Weightsift lease handle; every lookup is one Weightsift acquire, every release one Weightsift release.
 */
#include "ws_colibri_store.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* A lease handle travels in ColiExpertView.lease (a pointer). */
_Static_assert(sizeof(void *) >= sizeof(uint64_t), "lease handles need 64-bit pointers");

typedef struct {
    uint64_t store;
} WsColibriState;

static const struct {
    const char *name;
    ColiTensorFormat format;
} k_formats[] = {
    {"f32", COLI_TENSOR_F32},
    {"q8_row", COLI_TENSOR_Q8_ROW},
    {"q4_row", COLI_TENSOR_Q4_ROW},
    {"q2_row", COLI_TENSOR_Q2_ROW},
    {"fp8_e4m3_block", COLI_TENSOR_FP8_E4M3_BLOCK},
    {"fp4_native_block", COLI_TENSOR_FP4_NATIVE_BLOCK},
    {"int8_block", COLI_TENSOR_INT8_BLOCK},
    {"int4_block", COLI_TENSOR_INT4_BLOCK},
};

static const char *const k_roles[3] = {"gate", "up", "down"};

static int tensor_format(const char *name, ColiTensorFormat *out) {
    for (size_t k = 0; name && k < sizeof(k_formats) / sizeof(k_formats[0]); k++) {
        if (strcmp(name, k_formats[k].name) == 0) {
            *out = k_formats[k].format;
            return 0;
        }
    }
    return -1;
}

static int scale_format(const char *name, ColiScaleFormat *out) {
    if (!name) {
        *out = COLI_SCALE_NONE;
        return 0;
    }
    if (strcmp(name, "f32") == 0) {
        *out = COLI_SCALE_F32;
        return 0;
    }
    if (strcmp(name, "ue8m0") == 0) {
        *out = COLI_SCALE_UE8M0;
        return 0;
    }
    return -1;
}

/* The view of `role` in a lease, as a ColiTensorView over the leased bytes. */
static int tensor_view(uint64_t store, uint64_t lease, uint32_t layer, const char *role, ColiTensorView *out) {
    uint32_t index = 0;
    ws_view view;
    ColiTensorFormat format;
    ColiScaleFormat scales;
    memset(out, 0, sizeof(*out));
    memset(&view, 0, sizeof(view));
    view.struct_size = sizeof(view);
    if (ws_store_view_index(store, layer, role, &index) != WS_OK) return -1;
    if (ws_lease_view(store, lease, index, &view) != WS_OK) return -1;
    if (tensor_format(view.format, &format) != 0 || scale_format(view.scale_format, &scales) != 0) return -1;
    out->format = format;
    out->scale_format = scales;
    out->data = view.data;
    out->scales = view.scales;
    out->data_bytes = (size_t)view.bytes;
    out->scale_bytes = (size_t)view.scale_bytes;
    out->rows = view.rows;
    out->columns = view.columns;
    out->block_rows = view.block_rows;
    out->block_columns = view.block_columns;
    out->gpu = NULL;
    return 0;
}

static int ws_lookup(ColiExpertStore *store, ColiExpertKey key, ColiExpertView *view) {
    const WsColibriState *state = (const WsColibriState *)store->state;
    ws_expert_key k;
    uint64_t lease = 0;
    if (!view || key.layer < 0 || key.expert < 0) return -1;
    k.layer = (uint32_t)key.layer;
    k.expert = (uint32_t)key.expert;
    if (ws_store_acquire(state->store, k, &lease, NULL, 0) != WS_OK) return -1;
    memset(view, 0, sizeof(*view));
    view->key = key;
    if (tensor_view(state->store, lease, k.layer, k_roles[0], &view->gate) != 0 ||
        tensor_view(state->store, lease, k.layer, k_roles[1], &view->up) != 0 ||
        tensor_view(state->store, lease, k.layer, k_roles[2], &view->down) != 0) {
        ws_store_release(state->store, lease);
        memset(view, 0, sizeof(*view));
        return -1;
    }
    view->lease = (void *)(uintptr_t)lease;
    return 0;
}

static void ws_release(ColiExpertStore *store, ColiExpertView *view) {
    const WsColibriState *state = (const WsColibriState *)store->state;
    if (!view || !view->lease) return; /* a cleared or zero view: nothing to do */
    /* A stale or copied view's lease is refused by Weightsift's lease table (WS_ERR_INVALID_LEASE): nothing breaks. */
    (void)ws_store_release(state->store, (uint64_t)(uintptr_t)view->lease);
    memset(view, 0, sizeof(*view));
}

static int ws_prefetch(ColiExpertStore *store, const ColiExpertKey *keys, size_t count) {
    const WsColibriState *state = (const WsColibriState *)store->state;
    ws_expert_key small[64];
    ws_expert_key *converted = small;
    int accepted = 0;
    if (!keys || count == 0 || count > (size_t)0x7fffffff) return 0;
    if (count > sizeof(small) / sizeof(small[0])) {
        converted = (ws_expert_key *)malloc(count * sizeof(*converted));
        if (!converted) return 0;
    }
    for (size_t k = 0; k < count; k++) {
        if (keys[k].layer < 0 || keys[k].expert < 0) goto done;
        converted[k].layer = (uint32_t)keys[k].layer;
        converted[k].expert = (uint32_t)keys[k].expert;
    }
    /* Advisory: the keys are accepted (count) or refused (0), never an error for the engine. */
    if (ws_store_prefetch(state->store, converted, count, NULL) == WS_OK) accepted = (int)count;
done:
    if (converted != small) free(converted);
    return accepted;
}

static void ws_stats(const ColiExpertStore *store, ColiExpertStoreStats *stats) {
    const WsColibriState *state = (const WsColibriState *)store->state;
    ws_store_stats s;
    memset(stats, 0, sizeof(*stats));
    memset(&s, 0, sizeof(s));
    s.struct_size = sizeof(s);
    if (ws_store_get_stats(state->store, &s) != WS_OK) return;
    /* Colibrì counts lookups; Weightsift counts experts: a hit had every part cached, a wait was loaded by a prefetch
     * (or another lookup) and not read by this one, a miss was read. */
    stats->requests = s.acquire_calls;
    stats->hits = s.expert_hits;
    stats->misses = s.expert_misses + s.acquire_failures;
    stats->prefetched = s.prefetch_experts;
    stats->prefetch_hits = s.expert_waits;
    stats->bytes_read = s.physical_bytes;
    stats->resident_bytes = s.cache_resident_bytes;
    stats->capacity_bytes = s.cache_capacity_bytes;
}

static void ws_destroy(ColiExpertStore *store) {
    WsColibriState *state;
    char error[192];
    if (!store) return;
    state = (WsColibriState *)store->state;
    if (ws_store_close(state->store, error, sizeof(error)) != WS_OK) {
        /* Colibrì's contract wants zero leases here. Freeing anything now would leave the leases' releases (which go
         * through this store) dangling, so the store stays as it is and the misuse is reported. */
        fprintf(stderr, "weightsift-colibri: destroy refused: %s\n", error);
        return;
    }
    free(state);
    free(store);
}

static const ColiExpertStoreOps k_ops = {ws_lookup, ws_release, ws_prefetch, ws_stats, ws_destroy};

static int fail(char *error, size_t error_size, const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
    return -1;
}

/* Every layer has gate, up and down views whose formats Colibrì knows. */
static int check_catalog(uint64_t store, char *error, size_t error_size) {
    size_t count = 0;
    uint32_t *layers;
    int status = 0;
    if (ws_store_layers(store, NULL, 0, &count) != WS_OK || count == 0) {
        return fail(error, error_size, "the catalog has no layers");
    }
    layers = (uint32_t *)malloc(count * sizeof(*layers));
    if (!layers) return fail(error, error_size, "out of memory");
    if (ws_store_layers(store, layers, count, &count) != WS_OK) status = fail(error, error_size, "cannot list layers");
    for (size_t k = 0; status == 0 && k < count; k++) {
        for (int r = 0; status == 0 && r < 3; r++) {
            uint32_t index = 0;
            ws_view view;
            ColiTensorFormat format;
            ColiScaleFormat scales;
            memset(&view, 0, sizeof(view));
            view.struct_size = sizeof(view);
            if (ws_store_view_index(store, layers[k], k_roles[r], &index) != WS_OK ||
                ws_store_view_info(store, layers[k], index, &view) != WS_OK) {
                if (error && error_size)
                    snprintf(error, error_size, "layer %u has no view named %s", layers[k], k_roles[r]);
                status = -1;
            } else if (tensor_format(view.format, &format) != 0 || scale_format(view.scale_format, &scales) != 0) {
                if (error && error_size)
                    snprintf(error, error_size, "layer %u view %s: format %s (scales %s) is not one Colibri reads",
                             layers[k], k_roles[r], view.format, view.scale_format ? view.scale_format : "none");
                status = -1;
            }
        }
    }
    free(layers);
    return status;
}

int ws_colibri_store_open(const char *catalog_path, const ws_store_options *options, ColiExpertStore **out,
                          char *error, size_t error_size) {
    uint64_t handle = 0;
    ColiExpertStore *store;
    WsColibriState *state;
    if (!out) return fail(error, error_size, "out is NULL");
    *out = NULL;
    if (ws_abi_version() != WS_STORE_ABI_VERSION) return fail(error, error_size, "Weightsift store ABI mismatch");
    if (ws_store_open(catalog_path, options, &handle, error, error_size) != WS_OK) return -1;
    if (check_catalog(handle, error, error_size) != 0) {
        ws_store_close(handle, NULL, 0);
        return -1;
    }
    store = (ColiExpertStore *)calloc(1, sizeof(*store));
    state = (WsColibriState *)calloc(1, sizeof(*state));
    if (!store || !state) {
        free(store);
        free(state);
        ws_store_close(handle, NULL, 0);
        return fail(error, error_size, "out of memory");
    }
    state->store = handle;
    store->ops = &k_ops;
    store->state = state;
    store->gpu = NULL;
    *out = store;
    return 0;
}

uint64_t ws_colibri_store_handle(const ColiExpertStore *store) {
    return store && store->state ? ((const WsColibriState *)store->state)->store : 0;
}

void ws_colibri_store_destroy(ColiExpertStore *store) {
    if (store && store->ops && store->ops->destroy) store->ops->destroy(store);
}
