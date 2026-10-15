"""Code-length estimators, in bits, for symbols under context models.

Every estimator returns a total in bits, so that per-weight figures are a division and costs add. Four kinds, from
optimistic to achievable:

  plugin_bits         sum over contexts of n_c * H(empirical p_c). The in-sample ("oracle") conditional entropy: the
                      tables are free and fitted to the very data they code. Biased LOW; its bias grows with the number
                      of contexts. Used only as an optimistic bound on what a context could be worth.
  two_part_bits       plugin_bits plus the cost of transmitting every used table (a static coder with per-block tables).
                      Achievable up to the frequency quantization of a real coder.
  kt_bits             the code length of an adaptive arithmetic coder with Krichevsky-Trofimov (Dirichlet 1/2) counts
                      per context, from the final counts (the sequential code length of a Dirichlet-multinomial model
                      does not depend on the order of the symbols). Achievable, every learning cost included, no side
                      information.
  cross_bits          a static table per context trained on calibration data and applied to held-out data: no per-block
                      side information, mismatch paid. Achievable.
  prior_kt_bits       an adaptive coder whose Dirichlet prior is a calibration table: the held-out generalization of both.

`joint_counts` turns (context id, symbol) arrays into a [contexts, alphabet] count matrix.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln

LOG2E = 1.0 / math.log(2.0)


def joint_counts(context: np.ndarray | None, symbol: np.ndarray, n_contexts: int, alphabet: int) -> np.ndarray:
    """Counts [n_contexts, alphabet] (int64) of each (context, symbol) pair; `context` None means one context."""
    s = np.asarray(symbol).reshape(-1).astype(np.int64)
    if s.size and (s.min() < 0 or s.max() >= alphabet):
        raise ValueError(f"symbols outside [0, {alphabet})")
    if context is None:
        return np.bincount(s, minlength=alphabet).reshape(1, alphabet)
    c = np.asarray(context).reshape(-1).astype(np.int64)
    if c.shape != s.shape:
        raise ValueError("context and symbol arrays differ in size")
    if c.size and (c.min() < 0 or c.max() >= n_contexts):
        raise ValueError(f"contexts outside [0, {n_contexts})")
    return np.bincount(c * alphabet + s, minlength=n_contexts * alphabet).reshape(n_contexts, alphabet)


def plugin_bits(counts: np.ndarray) -> float:
    c = np.asarray(counts, dtype=np.float64)
    n = np.broadcast_to(c.sum(axis=-1, keepdims=True), c.shape)
    mask = c > 0
    terms = np.where(mask, c * (np.log2(np.where(mask, n, 1.0)) - np.log2(np.where(mask, c, 1.0))), 0.0)
    return float(terms.sum())


def tables_used(counts: np.ndarray) -> int:
    return int((np.asarray(counts).sum(axis=-1) > 0).sum())


def two_part_bits(counts: np.ndarray, bits_per_table: float) -> float:
    """plugin_bits plus `bits_per_table` for every context that occurs (its table is transmitted)."""
    return plugin_bits(counts) + bits_per_table * tables_used(counts)


def kt_bits(counts: np.ndarray, alpha: float = 0.5, support: np.ndarray | None = None) -> float:
    """Adaptive Dirichlet(alpha) code length summed over contexts. `support` (bool [alphabet]) restricts the alphabet
    the coder spreads its prior over (a range declared in a header); symbols outside it must not occur."""
    c = np.asarray(counts, dtype=np.float64)
    if c.ndim == 1:
        c = c[None, :]
    if support is not None:
        support = np.asarray(support, dtype=bool)
        if c[:, ~support].sum() > 0:
            raise ValueError("a symbol outside the declared support occurs")
        c = c[:, support]
    k = c.shape[-1]
    n = c.sum(axis=-1)
    used = n > 0
    nats = gammaln(n + k * alpha) - gammaln(k * alpha) - (gammaln(c + alpha) - gammaln(alpha)).sum(axis=-1)
    return float(nats[used].sum() * LOG2E)


def smoothed_table(counts: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    c = np.asarray(counts, dtype=np.float64)
    return (c + alpha) / (c.sum(axis=-1, keepdims=True) + alpha * c.shape[-1])


def cross_bits(train_counts: np.ndarray, test_counts: np.ndarray, alpha: float = 0.5) -> float:
    """Held-out code length of `test_counts` under per-context tables estimated from `train_counts` (additive
    smoothing alpha, so that no occurring symbol has probability zero)."""
    p = smoothed_table(train_counts, alpha)
    t = np.asarray(test_counts, dtype=np.float64)
    return float(-(t * np.log2(p)).sum())


def prior_kt_bits(prior: np.ndarray, strength: float, test_counts: np.ndarray, floor: float = 1e-6) -> float:
    """Adaptive code length of `test_counts` per context under a Dirichlet prior of total mass `strength` shaped like
    `prior` (rows are distributions, mixed with a uniform floor so that every symbol stays codable)."""
    p = np.asarray(prior, dtype=np.float64)
    k = p.shape[-1]
    p = (1.0 - floor) * p / p.sum(axis=-1, keepdims=True) + floor / k
    a = strength * p
    c = np.asarray(test_counts, dtype=np.float64)
    n = c.sum(axis=-1)
    used = n > 0
    nats = gammaln(n + strength) - gammaln(strength) - (gammaln(c + a) - gammaln(a)).sum(axis=-1)
    return float(nats[used].sum() * LOG2E)


def entropy_of(counts: np.ndarray) -> float:
    """Shannon entropy (bits per symbol) of a count vector."""
    c = np.asarray(counts, dtype=np.float64).reshape(-1)
    n = c.sum()
    return plugin_bits(c[None, :]) / n if n else 0.0


def uniform_index_bits(n_items: int, n_classes: int) -> float:
    """Side information: a class index per item, coded at a fixed length."""
    return float(n_items * math.ceil(math.log2(n_classes))) if n_classes > 1 else 0.0
