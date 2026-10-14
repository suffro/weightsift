"""Comparing a candidate with a baseline under a Quality Budget (comparison report: `weightsift-quality-comparison` v1).

For every benchmark the regression is baseline score minus candidate score (positive: the candidate is worse), with a
confidence interval: a paired bootstrap over the items both evaluated (when both list their items), otherwise, for
binary metrics, Newcombe's hybrid score interval for a difference of independent proportions. The benchmark's verdict
compares the interval with its category's threshold:

    PASS          the interval's upper bound is within the threshold (the regression is bounded by the budget)
    FAIL          the interval's lower bound exceeds the threshold (the regression is beyond the budget)
    INCONCLUSIVE  the interval straddles the threshold, or there are fewer samples than the budget requires
    PENDING       a side has no measurement (unavailable, missing, or the benchmark's revision is not pinned)
    INVALID       the two sides measured different data or metrics, or the evaluation data overlaps the candidate's
                  calibration data
    SKIPPED       an optional category neither side applies to (both skipped it, with a reason)

A category's verdict is its worst benchmark's (never an average), the overall verdict the worst category's (SKIPPED
categories aside). Order of severity: FAIL, INVALID, PENDING, INCONCLUSIVE, PASS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np

from awpmi.quality.results import BenchmarkResult, Results
from awpmi.quality.spec import Benchmark, QualityBudget

COMPARISON_FORMAT = "weightsift-quality-comparison"
COMPARISON_VERSION = 1


class Verdict(str, Enum):
    PASS = "PASS"
    INCONCLUSIVE = "INCONCLUSIVE"
    PENDING = "PENDING"
    INVALID = "INVALID"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


_SEVERITY = [Verdict.PASS, Verdict.INCONCLUSIVE, Verdict.PENDING, Verdict.INVALID, Verdict.FAIL]


def worst(verdicts: Sequence[Verdict]) -> Verdict:
    counted = [v for v in verdicts if v is not Verdict.SKIPPED]
    if not counted:
        return Verdict.SKIPPED
    return max(counted, key=_SEVERITY.index)


@dataclass(frozen=True)
class Interval:
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError("an interval's low end exceeds its high end")


def _z(confidence: float) -> float:
    return NormalDist().inv_cdf(1 - (1 - confidence) / 2)


def wilson_interval(successes: float, n: int, confidence: float) -> Interval:
    """Wilson's score interval for a proportion (successes may be fractional: a mean of 0/1 scores times n)."""
    if n < 1 or not 0 <= successes <= n:
        raise ValueError("need 0 <= successes <= n and n >= 1")
    z = _z(confidence)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return Interval(max(0.0, centre - half), min(1.0, centre + half))


def newcombe_difference(p1: float, n1: int, p2: float, n2: int, confidence: float) -> Interval:
    """Newcombe's hybrid score interval (1998, method 10) for p1 - p2, two independent proportions."""
    w1 = wilson_interval(p1 * n1, n1, confidence)
    w2 = wilson_interval(p2 * n2, n2, confidence)
    delta = p1 - p2
    low = delta - math.sqrt((p1 - w1.low) ** 2 + (w2.high - p2) ** 2)
    high = delta + math.sqrt((w1.high - p1) ** 2 + (p2 - w2.low) ** 2)
    return Interval(max(-1.0, low), min(1.0, high))


def paired_bootstrap(
    baseline: Sequence[float], candidate: Sequence[float], confidence: float, samples: int, seed: int
) -> Interval:
    """Percentile bootstrap interval of mean(baseline - candidate) over paired items (deterministic for a seed)."""
    differences = np.asarray(baseline, dtype=np.float64) - np.asarray(candidate, dtype=np.float64)
    n = differences.size
    if n < 1 or np.asarray(candidate).size != n:
        raise ValueError("paired scores of equal, positive length are required")
    rng = np.random.Generator(np.random.PCG64(seed))
    means = np.empty(samples, dtype=np.float64)
    chunk = max(1, min(samples, 4_000_000 // n))
    for start in range(0, samples, chunk):
        stop = min(samples, start + chunk)
        indices = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = differences[indices].mean(axis=1)
    alpha = 1 - confidence
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2], method="linear")
    return Interval(float(low), float(high))


@dataclass(frozen=True)
class BenchmarkComparison:
    name: str
    category: str
    threshold: float
    verdict: Verdict
    reason: str | None = None
    method: str | None = None
    n_baseline: int | None = None
    n_candidate: int | None = None
    n_paired: int | None = None
    baseline_score: float | None = None
    candidate_score: float | None = None
    regression: float | None = None
    interval: Interval | None = None


@dataclass(frozen=True)
class CategoryComparison:
    name: str
    threshold: float
    required: bool
    verdict: Verdict
    benchmarks: tuple[BenchmarkComparison, ...]


@dataclass(frozen=True)
class Comparison:
    baseline_kind: str
    baseline_run: str
    candidate_run: str
    confidence: float
    verdict: Verdict
    categories: tuple[CategoryComparison, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def category(self, name: str) -> CategoryComparison:
        return next(c for c in self.categories if c.name == name)

    def to_dict(self) -> dict[str, Any]:
        def benchmark(b: BenchmarkComparison) -> dict[str, Any]:
            return {
                "name": b.name,
                "verdict": b.verdict.value,
                "reason": b.reason,
                "method": b.method,
                "n_baseline": b.n_baseline,
                "n_candidate": b.n_candidate,
                "n_paired": b.n_paired,
                "baseline_score": b.baseline_score,
                "candidate_score": b.candidate_score,
                "regression": b.regression,
                "interval": None if b.interval is None else [b.interval.low, b.interval.high],
                "threshold": b.threshold,
            }

        return {
            "format": COMPARISON_FORMAT,
            "version": COMPARISON_VERSION,
            "baseline": {"kind": self.baseline_kind, "run": self.baseline_run},
            "candidate": {"run": self.candidate_run},
            "confidence": self.confidence,
            "verdict": self.verdict.value,
            "notes": list(self.notes),
            "categories": [
                {
                    "name": c.name,
                    "threshold": c.threshold,
                    "required": c.required,
                    "verdict": c.verdict.value,
                    "benchmarks": [benchmark(b) for b in c.benchmarks],
                }
                for c in self.categories
            ],
        }


def strongest_baseline(budget: QualityBudget, available: Sequence[Results]) -> Results | None:
    """The strongest available baseline in the budget's order (the original checkpoint first)."""
    for kind in budget.baselines:
        for results in available:
            if results.subject.kind == kind:
                return results
    return None


def _compare_benchmark(
    budget: QualityBudget,
    spec: Benchmark,
    threshold: float,
    baseline: BenchmarkResult | None,
    candidate: BenchmarkResult | None,
    calibration: set[tuple[str, str | None, str]],
) -> BenchmarkComparison:
    def verdict(v: Verdict, reason: str, **values: Any) -> BenchmarkComparison:
        return BenchmarkComparison(spec.name, spec.category, threshold, v, reason, **values)

    if spec.data in calibration:
        return verdict(Verdict.INVALID, "the evaluation data is part of the candidate's calibration data")
    if not spec.pinned:
        return verdict(Verdict.PENDING, "the budget does not pin this benchmark's dataset revision")
    for side, result in (("baseline", baseline), ("candidate", candidate)):
        if result is None:
            return verdict(Verdict.PENDING, f"the {side} has no result for it")
    for side, result in (("baseline", baseline), ("candidate", candidate)):
        if (result.dataset, result.config, result.split, result.revision, result.metric) != (
            spec.dataset, spec.config, spec.split, spec.revision, spec.metric
        ):
            return verdict(Verdict.INVALID, f"the {side} measured other data or another metric than the budget's")
    if not baseline.measured or not candidate.measured:
        if baseline.status == candidate.status == "skipped":
            return verdict(Verdict.SKIPPED, f"skipped by both: {candidate.reason}")
        side, result = ("baseline", baseline) if not baseline.measured else ("candidate", candidate)
        return verdict(Verdict.PENDING, f"the {side}'s result is {result.status}: {result.reason}")
    values = dict(
        n_baseline=baseline.n,
        n_candidate=candidate.n,
        baseline_score=baseline.score,
        candidate_score=candidate.score,
        regression=baseline.score - candidate.score,
    )
    if min(baseline.n, candidate.n) < spec.min_samples:
        return verdict(Verdict.INCONCLUSIVE, f"fewer than the {spec.min_samples} samples the budget requires", **values)
    if baseline.items is not None and candidate.items is not None:
        base, cand = dict(baseline.items), dict(candidate.items)
        if set(base) != set(cand):
            return verdict(Verdict.INVALID, "the two sides scored different items", **values)
        ids = sorted(base)
        interval = paired_bootstrap(
            [base[i] for i in ids], [cand[i] for i in ids], budget.confidence, budget.bootstrap_samples, budget.seed
        )
        values.update(method="paired-bootstrap", n_paired=len(ids), interval=interval)
    elif spec.binary:
        interval = newcombe_difference(baseline.score, baseline.n, candidate.score, candidate.n, budget.confidence)
        values.update(method="newcombe", interval=interval)
    else:
        return verdict(Verdict.INCONCLUSIVE, "a non-binary metric needs item scores on both sides", **values)
    if interval.high <= threshold:
        return verdict(Verdict.PASS, None, **values)
    if interval.low > threshold:
        return verdict(Verdict.FAIL, None, **values)
    return verdict(Verdict.INCONCLUSIVE, "the interval straddles the threshold: more samples are needed", **values)


def compare(budget: QualityBudget, baseline: Results, candidate: Results) -> Comparison:
    """Judge `candidate` against `baseline` on every benchmark of the budget, category by category."""
    if candidate.subject.kind != "candidate":
        raise ValueError("the candidate's subject kind must be 'candidate'")
    if baseline.subject.kind not in budget.baselines:
        raise ValueError(f"a {baseline.subject.kind} baseline is not one the budget accepts")
    notes = []
    if baseline.subject.kind != "original":
        notes.append(
            f"the baseline is a {baseline.subject.kind} checkpoint: the original checkpoint's quality is not measured "
            "here, and the regressions are relative to the baseline, not to the original"
        )
    if (baseline.subject.repository, baseline.subject.model) != (candidate.subject.repository, candidate.subject.model):
        notes.append("the baseline and the candidate come from different checkpoints")
    calibration = {ref.data for ref in candidate.calibration}
    categories = []
    for name, category in budget.categories.items():
        compared = tuple(
            _compare_benchmark(
                budget,
                spec,
                category.max_regression,
                baseline.benchmark(spec.name),
                candidate.benchmark(spec.name),
                calibration,
            )
            for spec in budget.benchmarks_of(name)
        )
        verdict = worst([b.verdict for b in compared])
        if verdict is Verdict.SKIPPED and category.required:
            verdict = Verdict.PENDING
        categories.append(CategoryComparison(name, category.max_regression, category.required, verdict, compared))
    return Comparison(
        baseline_kind=baseline.subject.kind,
        baseline_run=baseline.run_id,
        candidate_run=candidate.run_id,
        confidence=budget.confidence,
        verdict=worst([c.verdict for c in categories]),
        categories=tuple(categories),
        notes=tuple(notes),
    )
