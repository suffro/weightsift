/*
 * ws_colibri_store.h: Colibrì's expert-store lease contract (its c/expert_store.h) served by Weightsift's expert store
 * (weightsift_store.h), decision 0014. Weightsift's own code; Colibrì's header is included from its pinned, fetched
 * checkout and never copied here.
 *
 * A Colibrì engine that reads its routed experts through a ColiExpertStore (lookup, release, prefetch, stats, destroy)
 * can be given one served by Weightsift: lookup leases the expert from Weightsift's host cache (read from storage with
 * direct I/O if needed) and points the view's gate, up and down tensors at the leased bytes in place; release ends the
 * lease. Colibrì's contract holds as its header states it: one release per successful lookup, a failed lookup clears
 * the view, release of a cleared view is a no-op, prefetch is advisory, destroy needs zero leases. Beyond it, every
 * function is safe to call from several threads at once, and a stale or doubled release is refused by Weightsift's
 * lease table instead of corrupting anything.
 *
 * The catalog decides the layout: every layer must have three views named "gate", "up" and "down" whose formats name
 * ColiTensorFormat values ("f32", "q8_row", "q4_row", "q2_row", "fp8_e4m3_block", "fp4_native_block", "int8_block",
 * "int4_block") and whose scales' formats name ColiScaleFormat values ("f32", "ue8m0"; no scales: none). A catalog
 * that does not is refused when the store opens.
 */
#ifndef WS_COLIBRI_STORE_H
#define WS_COLIBRI_STORE_H

#include <stddef.h>
#include <stdint.h>

#include "expert_store.h"
#include "weightsift_store.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Open the catalog as a Weightsift store and wrap it: zero on success (*out set), non-zero with a message otherwise.
 * The returned store owns the Weightsift store; its destroy closes it. If leases are still alive, destroy reports it
 * on stderr and frees nothing (their bytes and their releases stay valid). */
int ws_colibri_store_open(const char *catalog_path, const ws_store_options *options, ColiExpertStore **out,
                          char *error, size_t error_size);

/* The Weightsift store behind a store opened by ws_colibri_store_open (its statistics, its catalog). */
uint64_t ws_colibri_store_handle(const ColiExpertStore *store);

/* Destroy through the store's own ops (for callers that do not include expert_store.h). */
void ws_colibri_store_destroy(ColiExpertStore *store);

#ifdef __cplusplus
}
#endif

#endif /* WS_COLIBRI_STORE_H */
