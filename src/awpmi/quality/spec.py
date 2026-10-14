"""The Quality Budget's specification (format `weightsift-quality-budget`, version 1): what a quantized inference mode is
judged on, against which baselines, and with which thresholds.

Six categories are always present (coding, mathematical reasoning, general reasoning, instruction following, long
context, tool use); each has an absolute regression threshold in its metrics' units and its own benchmarks, every one a
held-out dataset at a pinned revision. A category may be optional (tool use, for a model without tool calling). The
thresholds are configuration, not statistics: `awpmi.quality.compare` calls a regression only when its confidence
interval clears the threshold, and reports anything in between as inconclusive.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

FORMAT = "weightsift-quality-budget"
VERSION = 1

CATEGORIES = (
    "coding",
    "mathematical_reasoning",
    "general_reasoning",
    "instruction_following",
    "long_context",
    "tool_use",
)

# Strongest first: the publisher's original-precision checkpoint; a quantization the publisher released; a quantization
# derived by someone else (a conversion such as an engine's container).
BASELINE_KINDS = ("original", "released-quantized", "derived-quantized")


class BudgetError(ValueError):
    """A specification that is malformed or inconsistent."""


@dataclass(frozen=True)
class Benchmark:
    name: str
    category: str
    dataset: str
    revision: str | None
    config: str | None
    split: str
    metric: str
    min_samples: int
    # Every item scores 0 or 1 (accuracy, pass@1, exact match): aggregate results can then be compared without items.
    binary: bool
    # A capability the model must have for the benchmark to apply (e.g. "tool-calling").
    requires: str | None

    @property
    def pinned(self) -> bool:
        return self.revision is not None

    @property
    def data(self) -> tuple[str, str | None, str]:
        """What evaluation reads: (dataset, config, split)."""
        return (self.dataset, self.config, self.split)


@dataclass(frozen=True)
class Category:
    name: str
    max_regression: float
    required: bool


@dataclass(frozen=True)
class QualityBudget:
    confidence: float
    bootstrap_samples: int
    seed: int
    baselines: tuple[str, ...]
    categories: Mapping[str, Category]
    benchmarks: tuple[Benchmark, ...]

    def benchmarks_of(self, category: str) -> list[Benchmark]:
        return [b for b in self.benchmarks if b.category == category]

    def benchmark(self, name: str) -> Benchmark:
        for benchmark in self.benchmarks:
            if benchmark.name == name:
                return benchmark
        raise KeyError(name)

    def problems(self) -> list[str]:
        """What keeps the budget from being run as it stands (it is still a valid specification)."""
        return [
            f"{b.name}: no pinned dataset revision ({b.dataset}); pin it before any run"
            for b in self.benchmarks
            if not b.pinned
        ]


def _field(entry: Mapping[str, Any], key: str, kind: type | tuple[type, ...], where: str, default: Any = ...) -> Any:
    if key not in entry:
        if default is ...:
            raise BudgetError(f"{where}: {key} is required")
        return default
    value = entry[key]
    if not isinstance(value, kind) or isinstance(value, bool) and kind in (int, float, (int, float)):
        raise BudgetError(f"{where}: {key} must be {kind}")
    return value


def _check_keys(entry: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(entry) - allowed
    if unknown:
        raise BudgetError(f"{where}: unknown fields {sorted(unknown)}")


def parse_budget(document: Mapping[str, Any]) -> QualityBudget:
    _check_keys(document, {"format", "version", "confidence", "bootstrap_samples", "seed", "baselines", "categories",
                           "benchmarks"}, "budget")
    if document.get("format") != FORMAT or document.get("version") != VERSION:
        raise BudgetError(f"not a {FORMAT} v{VERSION} document")
    confidence = float(_field(document, "confidence", (int, float), "budget"))
    if not 0.5 < confidence < 1:
        raise BudgetError("confidence must lie in (0.5, 1)")
    samples = _field(document, "bootstrap_samples", int, "budget")
    if samples < 1000:
        raise BudgetError("bootstrap_samples must be at least 1000")
    seed = _field(document, "seed", int, "budget")
    baselines = tuple(_field(document, "baselines", list, "budget"))
    if not baselines or any(b not in BASELINE_KINDS for b in baselines) or len(set(baselines)) != len(baselines):
        raise BudgetError(f"baselines must be distinct kinds among {BASELINE_KINDS}, strongest first")
    raw_categories = _field(document, "categories", dict, "budget")
    if set(raw_categories) != set(CATEGORIES):
        raise BudgetError(f"categories must be exactly {CATEGORIES}")
    categories = {}
    for name, entry in raw_categories.items():
        where = f"category {name}"
        _check_keys(entry, {"max_regression", "required"}, where)
        threshold = float(_field(entry, "max_regression", (int, float), where))
        if not 0 <= threshold <= 1:
            raise BudgetError(f"{where}: max_regression is an absolute drop in [0, 1]")
        categories[name] = Category(name, threshold, _field(entry, "required", bool, where, True))
    benchmarks = []
    names = set()
    for entry in _field(document, "benchmarks", list, "budget"):
        where = f"benchmark {entry.get('name', '?')}"
        _check_keys(entry, {"name", "category", "dataset", "revision", "config", "split", "metric", "min_samples",
                            "binary", "requires"}, where)
        benchmark = Benchmark(
            name=_field(entry, "name", str, where),
            category=_field(entry, "category", str, where),
            dataset=_field(entry, "dataset", str, where),
            revision=_field(entry, "revision", (str, type(None)), where, None),
            config=_field(entry, "config", (str, type(None)), where, None),
            split=_field(entry, "split", str, where),
            metric=_field(entry, "metric", str, where),
            min_samples=_field(entry, "min_samples", int, where),
            binary=_field(entry, "binary", bool, where, False),
            requires=_field(entry, "requires", (str, type(None)), where, None),
        )
        if benchmark.category not in categories:
            raise BudgetError(f"{where}: unknown category {benchmark.category}")
        if benchmark.name in names:
            raise BudgetError(f"{where}: appears twice")
        if benchmark.min_samples < 1:
            raise BudgetError(f"{where}: min_samples must be positive")
        names.add(benchmark.name)
        benchmarks.append(benchmark)
    for name in CATEGORIES:
        if not any(b.category == name for b in benchmarks):
            raise BudgetError(f"category {name} has no benchmark")
    return QualityBudget(confidence, samples, seed, baselines, categories, tuple(benchmarks))


def load_budget(path: str | Path) -> QualityBudget:
    return parse_budget(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
