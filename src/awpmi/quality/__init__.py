"""The Quality Budget (Phase 7A, decision 0014): how a quantized inference mode is judged against the strongest available
baseline of the same model, per category, with sample counts and uncertainty, never from weight-space error.

`spec` (the budget: categories, held-out benchmarks at pinned revisions, configurable thresholds), `results` (what one
subject measured, with its run; unavailable results carry a reason, never a number), `compare` (confidence intervals
and verdicts per benchmark and category). Certifying a flagship needs its checkpoint and an engine to run it: until
then results are `unavailable` and comparisons `PENDING`.
"""

from awpmi.quality.compare import (
    BenchmarkComparison,
    CategoryComparison,
    Comparison,
    Interval,
    Verdict,
    compare,
    newcombe_difference,
    paired_bootstrap,
    strongest_baseline,
    wilson_interval,
)
from awpmi.quality.results import RESULTS_SCHEMA, Results, ResultsError, load_results, parse_results
from awpmi.quality.spec import BASELINE_KINDS, CATEGORIES, BudgetError, QualityBudget, load_budget, parse_budget

__all__ = [
    "BASELINE_KINDS",
    "BenchmarkComparison",
    "BudgetError",
    "CATEGORIES",
    "CategoryComparison",
    "Comparison",
    "Interval",
    "QualityBudget",
    "RESULTS_SCHEMA",
    "Results",
    "ResultsError",
    "Verdict",
    "compare",
    "load_budget",
    "load_results",
    "newcombe_difference",
    "paired_bootstrap",
    "parse_budget",
    "parse_results",
    "strongest_baseline",
    "wilson_interval",
]
