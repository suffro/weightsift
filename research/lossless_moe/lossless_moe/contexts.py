"""Context models over a weight matrix's symbols, each with what a decoder needs to rebuild it.

A context is an integer id per symbol. It is either *causal* (computed from symbols a raster-order decoder has already
produced: the left neighbours, the row above, running statistics) or derived from *side information* (a class per row
or column computed by the encoder from the whole tensor), whose bits are returned with it and must be charged. A
context may also come from another tensor the decoder already holds (the same expert's gate when coding its up).

Every builder returns `Context(ids, n, side_bits)`; ids are int32 of the matrix's shape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Context:
    ids: np.ndarray
    n: int
    side_bits: float = 0.0


def dense(symbols: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, int]:
    """Symbols mapped onto their rank in `support` (sorted unique values a header declares), so that contexts built
    from them stay small; returns (ranks, len(support))."""
    lookup = np.full(int(max(support.max(), symbols.max())) + 1, -1, dtype=np.int32)
    lookup[support] = np.arange(support.size, dtype=np.int32)
    ranks = lookup[symbols]
    if (ranks < 0).any():
        raise ValueError("a symbol outside the support")
    return ranks, int(support.size)


def none(shape: tuple[int, ...]) -> Context:
    return Context(np.zeros(shape, dtype=np.int32), 1)


def left(ranks: np.ndarray, k: int) -> Context:
    """The previous symbol in the row (k at the row's start)."""
    ids = np.empty(ranks.shape, dtype=np.int32)
    ids[:, 0] = k
    ids[:, 1:] = ranks[:, :-1]
    return Context(ids, k + 1)


def left2(ranks: np.ndarray, k: int) -> Context:
    a = left(ranks, k).ids
    b = np.empty(ranks.shape, dtype=np.int32)
    b[:, :2] = k
    b[:, 2:] = ranks[:, :-2]
    return Context(a * (k + 1) + b, (k + 1) ** 2)


def above(ranks: np.ndarray, k: int) -> Context:
    """The symbol in the same column of the previous row (k in the first row)."""
    ids = np.empty(ranks.shape, dtype=np.int32)
    ids[0, :] = k
    ids[1:, :] = ranks[:-1, :]
    return Context(ids, k + 1)


def left_above(ranks: np.ndarray, k: int) -> Context:
    return Context(left(ranks, k).ids * (k + 1) + above(ranks, k).ids, (k + 1) ** 2)


def quantize(values: np.ndarray, lo: float, hi: float, bins: int) -> np.ndarray:
    """Fixed, data-independent bins of [lo, hi] (clipped), 0..bins-1."""
    q = np.floor((np.asarray(values, dtype=np.float64) - lo) / (hi - lo) * bins)
    return np.clip(q, 0, bins - 1).astype(np.int32)


def row_activity(activity: np.ndarray, window: int, lo: float, hi: float, bins: int) -> Context:
    """Mean activity of the previous `window` symbols of the row (fewer at its start; its own class at j = 0)."""
    a = np.asarray(activity, dtype=np.float64)
    rows, cols = a.shape
    csum = np.zeros((rows, cols + 1))
    np.cumsum(a, axis=1, out=csum[:, 1:])
    j = np.arange(cols)
    start = np.maximum(j - window, 0)
    count = j - start
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (csum[:, j] - csum[:, start]) / np.where(count > 0, count, 1)
    ids = quantize(mean, lo, hi, bins)
    ids[:, 0] = bins
    return Context(ids, bins + 1)


def column_activity(activity: np.ndarray, lo: float, hi: float, bins: int) -> Context:
    """Mean activity of the column over every previous row (its own class in the first row)."""
    a = np.asarray(activity, dtype=np.float64)
    rows = a.shape[0]
    csum = np.cumsum(a, axis=0)
    mean = np.empty_like(a)
    mean[1:] = csum[:-1] / np.arange(1, rows)[:, None]
    ids = quantize(mean, lo, hi, bins)
    ids[0, :] = bins
    return Context(ids, bins + 1)


def _classes(stat: np.ndarray, q: int) -> tuple[np.ndarray, float]:
    """Per-item classes by the tensor's own quantiles of `stat` (q classes); the edges' cost (32 bits each)."""
    edges = np.quantile(stat, np.linspace(0, 1, q + 1)[1:-1])
    return np.searchsorted(edges, stat, side="right").astype(np.int32), 32.0 * (q - 1)


def column_class(activity: np.ndarray, q: int) -> Context:
    """Side information: the class of each column's mean activity over the whole tensor."""
    cls, edge_bits = _classes(np.asarray(activity, dtype=np.float64).mean(axis=0), q)
    ids = np.broadcast_to(cls[None, :], activity.shape).astype(np.int32)
    return Context(ids, q, activity.shape[1] * math.ceil(math.log2(q)) + edge_bits)


def row_class(activity: np.ndarray, q: int) -> Context:
    cls, edge_bits = _classes(np.asarray(activity, dtype=np.float64).mean(axis=1), q)
    ids = np.broadcast_to(cls[:, None], activity.shape).astype(np.int32)
    return Context(ids, q, activity.shape[0] * math.ceil(math.log2(q)) + edge_bits)


def rank1_class(activity: np.ndarray, q: int, value_bits: int = 8) -> Context:
    """Side information: a row term and a column term (each sent at `value_bits`), their sum's class. Models an
    additive (log-domain) row-times-column scale structure."""
    a = np.asarray(activity, dtype=np.float64)
    r = a.mean(axis=1)
    c = a.mean(axis=0)
    total = r[:, None] + c[None, :] - a.mean()
    edges = np.quantile(total, np.linspace(0, 1, q + 1)[1:-1])
    ids = np.searchsorted(edges, total, side="right").astype(np.int32)
    side = (a.shape[0] + a.shape[1]) * value_bits + 32.0 * (q - 1)
    return Context(ids, q, side)


def given(other_ranks: np.ndarray, k: int) -> Context:
    """The co-located symbol of another tensor the decoder already holds (same shape)."""
    if other_ranks.shape[0] == 0:
        raise ValueError("empty")
    return Context(np.asarray(other_ranks, dtype=np.int32), k)


def given_rows(row_classes: np.ndarray, n: int, shape: tuple[int, int], axis: int) -> Context:
    """A class per row (axis 0) or column (axis 1) known to the decoder from another tensor (no side bits)."""
    cls = np.asarray(row_classes, dtype=np.int32)
    ids = cls[:, None] if axis == 0 else cls[None, :]
    return Context(np.broadcast_to(ids, shape).astype(np.int32), n)


def product(a: Context, b: Context) -> Context:
    return Context(a.ids.astype(np.int64).astype(np.int32) * b.n + b.ids, a.n * b.n, a.side_bits + b.side_bits)
