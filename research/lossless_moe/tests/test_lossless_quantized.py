"""Quantized formats: packing conventions, scale decoding, causal block contexts, and Colibri's per-row recipe."""

from __future__ import annotations

import numpy as np
import pytest

from lossless_moe import contexts as cx
from lossless_moe import quantized, symbols


def _qm(fmt: str, rows: int, cols: int, weight_bytes: np.ndarray, scale_bytes: np.ndarray) -> quantized.QMatrix:
    group = cols if fmt == "int4_row" else 32
    return quantized.QMatrix("t", fmt, 0, 0, "gate", rows, cols, weight_bytes, scale_bytes, group, "", "")


def test_compressed_tensors_int32_packing_is_low_nibble_first_bytes():
    """compressed-tensors packs 8 nibbles per int32, nibble i at bits 4i: on a little-endian machine its bytes are
    exactly the low-nibble-first byte packing that `unpack_nibbles` reads."""
    rng = np.random.default_rng(0)
    levels = rng.integers(-8, 8, size=(4, 64))
    unsigned = (levels + 8).astype(np.uint32)
    words = np.zeros((4, 8), dtype=np.uint32)
    for i in range(8):
        words |= unsigned[:, i::8] << (4 * i)
    as_bytes = words.astype("<u4").view(np.uint8).reshape(4, 32)
    nib = symbols.unpack_nibbles(as_bytes)
    assert np.array_equal(symbols.signed_int4(nib, "offset8"), levels)


def test_scales_decode_each_format():
    rows, cols = 2, 64
    w = np.zeros((rows, cols // 2), dtype=np.uint8)
    f32 = np.array([0.5, 2.0], dtype="<f4").view(np.uint8)
    assert np.allclose(_qm("int4_row", rows, cols, w, f32).scales()[:, 0], [0.5, 2.0])
    bf16 = np.array([[0x3F80, 0x4000], [0x3F00, 0x4080]], dtype="<u2").view(np.uint8).reshape(-1)  # 1, 2, 0.5, 4
    assert np.allclose(_qm("int4_g32_bf16", rows, cols, w, bf16).scales(), [[1, 2], [0.5, 4]])
    e8m0 = np.array([[127, 128], [126, 130]], dtype=np.uint8).reshape(-1)
    assert np.allclose(_qm("mxfp4", rows, cols, w, e8m0).scales(), [[1, 2], [0.5, 8]])


def test_block_context_is_causal_and_sees_the_maximum():
    rng = np.random.default_rng(1)
    top = rng.random((3, 64)) < 0.1
    ids = quantized.block_context(top, 32).ids
    for r in range(3):
        for c in range(64):
            changed = top.copy()
            changed[r, c] = not changed[r, c]
            diff = np.argwhere(quantized.block_context(changed, 32).ids != ids)
            assert all((rr, cc) > (r, c) for rr, cc in map(tuple, diff)), (r, c)
    # once a maximal code has been seen in a group, every later position of that group says so
    one = np.zeros((1, 32), dtype=bool)
    one[0, 5] = True
    seen = quantized.block_context(one, 32).ids[0] % 2
    assert seen[:6].sum() == 0 and seen[6:].all()


def test_group_scale_classes_are_constant_within_a_group():
    rows, cols = 3, 96
    w = np.zeros((rows, cols // 2), dtype=np.uint8)
    e8m0 = np.array([[120, 127, 134], [127, 127, 127], [100, 127, 140]], dtype=np.uint8).reshape(-1)
    cls = quantized.group_scale_classes(_qm("mxfp4", rows, cols, w, e8m0))
    assert cls.shape == (rows, cols)
    for g in range(3):
        assert (cls[:, g * 32 : (g + 1) * 32] == cls[:, [g * 32]]).all()
    assert cls[0, 0] < cls[0, 32] < cls[0, 64]


def test_colibri_per_row_recipe_round_trips_its_levels():
    from requantize import quant_int4_rows

    rng = np.random.default_rng(2)
    w = rng.standard_normal((5, 64)).astype(np.float32)
    packed, scales = quant_int4_rows(w)
    levels = symbols.signed_int4(symbols.unpack_nibbles(packed), "offset8").astype(np.float32)
    assert np.abs(levels).max() == 7  # the row maximum maps to +-7, -8 is never produced
    assert np.allclose(levels * scales[:, None], np.clip(np.rint(w / scales[:, None]), -8, 7) * scales[:, None])
    assert (np.abs(levels).max(axis=1) == 7).all()


@pytest.mark.parametrize("fmt", ["int4_row", "mxfp4"])
def test_nibble_specs_are_well_formed(fmt):
    rng = np.random.default_rng(3)
    rows, cols = 8, 64
    nib = rng.integers(0, 16, size=(rows, cols)).astype(np.uint8)
    w = symbols.pack_nibbles(nib)
    scales = (np.full(rows, 0.01, dtype="<f4").view(np.uint8) if fmt == "int4_row"
              else np.full(rows * 2, 127, dtype=np.uint8))
    m = _qm(fmt, rows, cols, w, scales)
    for s in quantized.nibble_specs(m, nib, {}):
        assert s.ctx.ids.shape == (rows, cols) and s.ctx.ids.min() >= 0 and s.ctx.ids.max() < s.ctx.n, s.context
        assert s.symbols.max() < s.alphabet
    assert isinstance(cx.none((1, 1)), cx.Context)
