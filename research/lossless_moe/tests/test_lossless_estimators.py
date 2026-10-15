"""Symbols are bijections; estimators equal their definitions; causal contexts never look ahead."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lossless_moe import contexts as ctx
from lossless_moe import entropy, symbols


def test_bf16_fields_round_trip_every_pattern():
    p = np.arange(1 << 16, dtype=np.uint16)
    s, e, m = symbols.bf16_fields(p)
    assert np.array_equal(symbols.bf16_merge(s, e, m), p)
    hi, lo = symbols.bf16_high_low(p)
    assert np.array_equal(((hi.astype(np.uint16) << 8) | lo).astype(np.uint16), p)


def test_nibbles_round_trip_and_levels():
    b = np.arange(256, dtype=np.uint8).reshape(4, 64)
    n = symbols.unpack_nibbles(b)
    assert n.shape == (4, 128) and np.array_equal(symbols.pack_nibbles(n), b)
    assert n[0, 0] == 0 and n[0, 2] == 1 and n[0, 3] == 0  # byte 1 = 0x01: low nibble first
    levels = np.arange(-8, 8)
    assert np.array_equal(symbols.signed_int4((levels + 8).astype(np.uint8), "offset8"), levels)
    assert np.array_equal(symbols.signed_int4((levels & 15).astype(np.uint8), "twos"), levels)
    with pytest.raises(ValueError):
        symbols.pack_nibbles(np.zeros((1, 3), np.uint8))


def test_e2m1_round_trip():
    n = np.arange(16, dtype=np.uint8)
    s, m = symbols.e2m1_fields(n)
    assert np.array_equal(symbols.e2m1_merge(s, m), n)


def _sequential_kt(seq, contexts, k, alpha):
    counts = {}
    bits = 0.0
    for c, x in zip(contexts, seq):
        row = counts.setdefault(c, np.zeros(k))
        bits += -math.log2((row[x] + alpha) / (row.sum() + k * alpha))
        row[x] += 1
    return bits


def test_estimators_equal_their_definitions():
    rng = np.random.default_rng(0)
    k = 5
    seq = rng.choice(k, size=600, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    con = rng.integers(0, 3, size=600)
    counts = entropy.joint_counts(con, seq, 3, k)
    # plug-in: sum of n_c H(p_c)
    expected = 0.0
    for c in range(3):
        sub = seq[con == c]
        for x in range(k):
            n = (sub == x).sum()
            if n:
                expected += -n * math.log2(n / sub.size)
    assert entropy.plugin_bits(counts) == pytest.approx(expected, rel=1e-12)
    # adaptive KT: the sequential coder's code length, in any order
    for alpha in (0.5, 1.0):
        assert entropy.kt_bits(counts, alpha) == pytest.approx(_sequential_kt(seq, con, k, alpha), rel=1e-10)
        perm = rng.permutation(600)
        assert entropy.kt_bits(counts, alpha) == pytest.approx(_sequential_kt(seq[perm], con[perm], k, alpha), rel=1e-10)
    # cross-entropy of a trained table
    train = entropy.joint_counts(con[:300], seq[:300], 3, k)
    test = entropy.joint_counts(con[300:], seq[300:], 3, k)
    p = (train + 0.5) / (train.sum(1, keepdims=True) + 0.5 * k)
    assert entropy.cross_bits(train, test) == pytest.approx(-sum(math.log2(p[c, x]) for c, x in zip(con[300:], seq[300:])))
    # prior-adaptive: sequential with pseudo-counts strength * prior
    prior = entropy.smoothed_table(train)
    strength = 7.0
    bits, state = 0.0, {}
    floor = 1e-6
    q = (1 - floor) * prior / prior.sum(1, keepdims=True) + floor / k
    for c, x in zip(con[300:], seq[300:]):
        row = state.setdefault(c, strength * q[c].copy())
        bits += -math.log2(row[x] / row.sum())
        row[x] += 1
    assert entropy.prior_kt_bits(prior, strength, test) == pytest.approx(bits, rel=1e-10)
    # support restriction refuses an undeclared symbol
    with pytest.raises(ValueError):
        entropy.kt_bits(counts, support=np.array([True, True, True, True, False]))


def test_plugin_is_below_achievable_and_conditioning_never_hurts_in_sample():
    rng = np.random.default_rng(1)
    seq = rng.choice(4, size=5000)
    one = entropy.joint_counts(None, seq, 1, 4)
    con = entropy.joint_counts(rng.integers(0, 50, 5000), seq, 50, 4)
    assert entropy.plugin_bits(con) <= entropy.plugin_bits(one) + 1e-9
    # but with random contexts the achievable (adaptive) cost is higher: the in-sample gain is pure estimation bias
    assert entropy.kt_bits(con) > entropy.kt_bits(one)


def _changes_at_or_after(builder, a, i, j):
    """True when changing the symbol at (i, j) changes the context of (i, j) itself or of any earlier position in
    raster order (a causal context depends only on strictly earlier symbols)."""
    base = builder(a).ids
    b = a.copy()
    b[i, j] = (b[i, j] + 1) % 7
    changed = np.argwhere(builder(b).ids != base)
    return any((r, c) <= (i, j) for r, c in map(tuple, changed))


@pytest.mark.parametrize("name", ["left", "left2", "above", "left_above", "row_activity", "column_activity"])
def test_causal_contexts_never_look_ahead(name):
    rng = np.random.default_rng(2)
    a = rng.integers(0, 7, size=(6, 9)).astype(np.int32)
    builders = {
        "left": lambda x: ctx.left(x, 7),
        "left2": lambda x: ctx.left2(x, 7),
        "above": lambda x: ctx.above(x, 7),
        "left_above": lambda x: ctx.left_above(x, 7),
        "row_activity": lambda x: ctx.row_activity(x, 3, 0, 7, 4),
        "column_activity": lambda x: ctx.column_activity(x, 0, 7, 4),
    }
    for i in range(6):
        for j in range(9):
            assert not _changes_at_or_after(builders[name], a, i, j), (name, i, j)


def test_the_causality_check_catches_a_look_ahead():
    """Negative control: a side-information context (computed from the whole tensor) is not causal, and the check
    above must say so."""
    rng = np.random.default_rng(3)
    a = rng.integers(0, 7, size=(6, 9)).astype(np.int32)
    assert any(_changes_at_or_after(lambda x: ctx.column_class(x, 4), a, i, j) for i in range(6) for j in range(9))


def test_side_information_contexts_charge_their_bits():
    a = np.arange(40, dtype=np.float64).reshape(4, 10)
    c = ctx.column_class(a, 4)
    assert c.n == 4 and c.side_bits == 10 * 2 + 32 * 3
    r = ctx.row_class(a, 2)
    assert r.side_bits == 4 * 1 + 32
    assert ctx.rank1_class(a, 8).side_bits == (4 + 10) * 8 + 32 * 7
