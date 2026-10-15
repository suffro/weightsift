"""Stage B's measurements: code lengths of a matrix's fields under many context models, accumulated over a corpus.

For each (field, context) pair and each matrix, four numbers are kept (bits, see `entropy`): the in-sample conditional
entropy (`plugin`, optimistic), the two-part static code (`two_part`: plugin + its tables), the adaptive code (`kt`), all
plus the context's side information; and the counts are pooled per (model, kind, split) so that tables trained on
calibration matrices can be scored on evaluation matrices (`cross`, `prior_kt`).

Symbols are dense ranks over a declared support (the model's set of values of that field), so that context tables stay
small; declaring a support costs a bitmap per model, which is negligible and not charged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import contexts as cx
from . import entropy

TABLE_ENTRY_BITS = 16


@dataclass
class Spec:
    """One (field, context) measurement of one matrix."""

    field: str
    context: str
    symbols: np.ndarray  # dense ranks, int32
    alphabet: int
    ctx: cx.Context


@dataclass
class Totals:
    weights: int = 0
    plugin: float = 0.0
    two_part: float = 0.0
    kt: float = 0.0
    side: float = 0.0

    def add(self, other: "Totals") -> None:
        self.weights += other.weights
        self.plugin += other.plugin
        self.two_part += other.two_part
        self.kt += other.kt
        self.side += other.side

    def per_weight(self) -> dict:
        w = max(self.weights, 1)
        return {"weights": self.weights, "plugin": (self.plugin + self.side) / w,
                "two_part": (self.two_part + self.side) / w, "kt": (self.kt + self.side) / w, "side": self.side / w}


def measure(spec: Spec) -> tuple[Totals, np.ndarray]:
    counts = entropy.joint_counts(spec.ctx.ids, spec.symbols, spec.ctx.n, spec.alphabet)
    support = counts.sum(axis=0) > 0
    t = Totals(
        weights=int(spec.symbols.size),
        plugin=entropy.plugin_bits(counts),
        two_part=entropy.two_part_bits(counts, TABLE_ENTRY_BITS * int(support.sum())),
        kt=entropy.kt_bits(counts[:, support], 0.5),
        side=float(spec.ctx.side_bits),
    )
    return t, counts


@dataclass
class Accumulator:
    """Per (field, context): totals over every matrix, and pooled counts per (group, split) for held-out scoring."""

    totals: dict = field(default_factory=dict)
    pooled: dict = field(default_factory=dict)
    per_matrix: list = field(default_factory=list)

    def add(self, key: str, memberships: list[tuple[str, str]], spec: Spec) -> None:
        """Measure one matrix once and add it to every (group, split) it belongs to."""
        t, counts = measure(spec)
        name = (spec.field, spec.context)
        for group, split in memberships:
            self.totals.setdefault((group, split) + name, Totals()).add(t)
            slot = self.pooled.get((group, split) + name)
            self.pooled[(group, split) + name] = counts.astype(np.int64) if slot is None else slot + counts
        self.per_matrix.append({"key": key, "field": spec.field, "context": spec.context, **t.per_weight()})

    def held_out(self, group: str, name: tuple[str, str], strengths=(16.0, 256.0, 4096.0)) -> dict | None:
        """Tables trained on the group's calibration counts, scored on its evaluation counts."""
        train = self.pooled.get((group, "calibration") + name)
        test = self.pooled.get((group, "evaluation") + name)
        if train is None or test is None:
            return None
        weights = int(test.sum())
        side = self.totals[(group, "evaluation") + name].side
        out = {"weights": weights, "cross": (entropy.cross_bits(train, test) + side) / weights}
        prior = entropy.smoothed_table(train)
        out["prior_kt"] = {str(int(s)): (entropy.prior_kt_bits(prior, s, test) + side) / weights for s in strengths}
        return out


# ---------------------------------------------------------------------------------------------------------------------
# Context sets


def bf16_specs(e: np.ndarray, s: np.ndarray, m: np.ndarray, support: np.ndarray, extra: dict) -> list[Spec]:
    """The exponent under every context model, the mantissa and the sign under a few. `support`: the model's exponent
    values (sorted). `extra` may hold `given_exponent` (another matrix's exponents the decoder already holds, same
    shape: the same expert's gate for its up, the previous sampled expert's same matrix) and `neuron_classes` (a class
    per column of a down matrix from the same expert's gate and up rows)."""
    ranks, k = cx.dense(e, support)
    activity = e.astype(np.float64)
    nonzero = support[support > 0]
    lo, hi = float(nonzero.min()), float(support.max()) + 1.0
    specs = [
        Spec("exponent", "none", ranks, k, cx.none(e.shape)),
        Spec("exponent", "left", ranks, k, cx.left(ranks, k)),
        Spec("exponent", "left2", ranks, k, cx.left2(ranks, k)),
        Spec("exponent", "above", ranks, k, cx.above(ranks, k)),
        Spec("exponent", "left_above", ranks, k, cx.left_above(ranks, k)),
        Spec("exponent", "row_mean8", ranks, k, cx.row_activity(activity, 8, lo, hi, 16)),
        Spec("exponent", "row_mean64", ranks, k, cx.row_activity(activity, 64, lo, hi, 16)),
        Spec("exponent", "col_mean_causal", ranks, k, cx.column_activity(activity, lo, hi, 16)),
        Spec("exponent", "col_class16", ranks, k, cx.column_class(activity, 16)),
        Spec("exponent", "row_class16", ranks, k, cx.row_class(activity, 16)),
        Spec("exponent", "rank1_class32", ranks, k, cx.rank1_class(activity, 32)),
        Spec("exponent", "col_class16+row_mean64", ranks, k,
             cx.product(cx.column_class(activity, 16), cx.row_activity(activity, 64, lo, hi, 16))),
    ]
    if "given_exponent" in extra:
        for label, other in extra["given_exponent"].items():
            oranks, ok = cx.dense(other, support)
            specs.append(Spec("exponent", f"given_{label}", ranks, k, cx.given(oranks, ok)))
    if "neuron_classes" in extra:
        specs.append(Spec("exponent", "neuron_class16", ranks, k,
                          cx.given_rows(extra["neuron_classes"], 16, e.shape, axis=1)))
    mant = m.astype(np.int32)
    specs += [
        Spec("mantissa", "none", mant, 128, cx.none(e.shape)),
        Spec("mantissa", "exponent", mant, 128, cx.Context(ranks.astype(np.int32), k)),
        Spec("mantissa_top3", "none", (mant >> 4), 8, cx.none(e.shape)),
        Spec("mantissa_top3", "exponent", (mant >> 4), 8, cx.Context(ranks.astype(np.int32), k)),
        Spec("mantissa_top3", "exponent+col_class16", (mant >> 4), 8,
             cx.product(cx.Context(ranks.astype(np.int32), k), cx.column_class(activity, 16))),
    ]
    sign = s.astype(np.int32)
    specs += [
        Spec("sign", "none", sign, 2, cx.none(e.shape)),
        Spec("sign", "left", sign, 2, cx.left(sign, 2)),
        Spec("sign", "above", sign, 2, cx.above(sign, 2)),
    ]
    if "given_sign" in extra:
        for label, other in extra["given_sign"].items():
            specs.append(Spec("sign", f"given_{label}", sign, 2, cx.given(other.astype(np.int32), 2)))
    return specs


def int4_specs(levels: np.ndarray, extra: dict) -> list[Spec]:
    """A level-coded int4 matrix (levels in [-8, 7]) under the context models; symbols are the nibble (level + 8).
    `extra` may hold `row_scale_class` (a class per row from the stored per-row scale: free side information, it is
    already in the container), `given_levels` and `neuron_classes`."""
    sym = (levels.astype(np.int32) + 8)
    k = 16
    mag = np.abs(levels.astype(np.float64))
    specs = [
        Spec("level", "none", sym, k, cx.none(sym.shape)),
        Spec("level", "left", sym, k, cx.left(sym, k)),
        Spec("level", "left2", sym, k, cx.left2(sym, k)),
        Spec("level", "above", sym, k, cx.above(sym, k)),
        Spec("level", "left_above", sym, k, cx.left_above(sym, k)),
        Spec("level", "row_mean8", sym, k, cx.row_activity(mag, 8, 0.0, 7.0, 16)),
        Spec("level", "row_mean64", sym, k, cx.row_activity(mag, 64, 0.0, 4.0, 16)),
        Spec("level", "row_mean512", sym, k, cx.row_activity(mag, 512, 0.0, 4.0, 16)),
        Spec("level", "col_mean_causal", sym, k, cx.column_activity(mag, 0.0, 4.0, 16)),
        Spec("level", "col_class16", sym, k, cx.column_class(mag, 16)),
        Spec("level", "row_class16", sym, k, cx.row_class(mag, 16)),
        Spec("level", "rank1_class32", sym, k, cx.rank1_class(mag, 32)),
        Spec("level", "col_class16+row_mean512", sym, k,
             cx.product(cx.column_class(mag, 16), cx.row_activity(mag, 512, 0.0, 4.0, 16))),
    ]
    if "row_scale_class" in extra:
        specs.append(Spec("level", "row_scale16", sym, k, cx.given_rows(extra["row_scale_class"], 16, sym.shape, axis=0)))
        specs.append(Spec("level", "row_scale16+col_class16", sym, k,
                          cx.product(cx.given_rows(extra["row_scale_class"], 16, sym.shape, axis=0),
                                     cx.column_class(mag, 16))))
    if "given_levels" in extra:
        for label, other in extra["given_levels"].items():
            specs.append(Spec("level", f"given_{label}", sym, k, cx.given(other.astype(np.int32) + 8, 16)))
    if "neuron_classes" in extra:
        specs.append(Spec("level", "neuron_class16", sym, k, cx.given_rows(extra["neuron_classes"], 16, sym.shape, axis=1)))
    return specs


def neuron_classes(gate_activity: np.ndarray, up_activity: np.ndarray, q: int = 16) -> np.ndarray:
    """A class per neuron (row of gate and up) from their mean activity: what a decoder holding gate and up knows
    about each column of down (quantiles of the neurons' own statistic, so no side information)."""
    stat = gate_activity.mean(axis=1) + up_activity.mean(axis=1)
    edges = np.quantile(stat, np.linspace(0, 1, q + 1)[1:-1])
    return np.searchsorted(edges, stat, side="right").astype(np.int32)
