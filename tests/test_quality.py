"""The Quality Budget (Phase 7A, decision 0014): specification, results and comparison.

Intervals are checked against values from the literature, the bootstrap for determinism and exact cases; comparisons on
synthetic results cover every verdict, per category, without averaging; results that claim what they did not measure
are refused."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from awpmi.quality import (
    RESULTS_SCHEMA,
    BudgetError,
    ResultsError,
    Verdict,
    compare,
    load_budget,
    newcombe_difference,
    paired_bootstrap,
    parse_budget,
    parse_results,
    strongest_baseline,
    wilson_interval,
)

ROOT = Path(__file__).resolve().parents[1]
BUDGET = ROOT / "configs" / "quality-budget.yaml"


def pinned_budget(**changes):
    """The committed budget with every revision pinned (to a placeholder), so comparisons can complete."""
    import yaml

    document = yaml.safe_load(BUDGET.read_text(encoding="utf-8"))
    for benchmark in document["benchmarks"]:
        benchmark["revision"] = "pinned-for-test"
    document.update(changes)
    return parse_budget(document)


def results(kind, scores, *, items=None, calibration=(), status=None, run="run"):
    """A results document: `scores` maps benchmark names to a score, with n from the budget (or items)."""
    budget = pinned_budget()
    benchmarks = []
    for spec in budget.benchmarks:
        entry = {
            "name": spec.name,
            "category": spec.category,
            "dataset": spec.dataset,
            "config": spec.config,
            "revision": spec.revision,
            "split": spec.split,
            "metric": spec.metric,
        }
        state = (status or {}).get(spec.name, "measured")
        if state != "measured":
            entry.update(status=state, reason="not runnable here", n=None, score=None)
        elif items and spec.name in items:
            values = items[spec.name]
            entry.update(
                status="measured",
                n=len(values),
                score=sum(values) / len(values),
                items=[{"id": f"q{k}", "score": v} for k, v in enumerate(values)],
            )
        else:
            # 20,000 items: an unpaired interval of about +-0.009 at 0.7 (with 1,000 it would be +-0.04, too wide to
            # certify a two-point budget: the point of reporting intervals).
            entry.update(status="measured", n=max(spec.min_samples, 20_000), score=scores.get(spec.name, 0.7))
        benchmarks.append(entry)
    return parse_results(
        {
            "format": "weightsift-quality-results",
            "version": 1,
            "run": {"id": run, "date": "2026-10-12", "hardware": {}, "software": {}, "commands": []},
            "subject": {
                "kind": kind,
                "model": "a flagship",
                "checkpoint": {"repository": "org/model", "revision": "abc"},
                "representation": "test",
                "engine": {"name": "test", "revision": "0"},
            },
            "calibration": list(calibration),
            "benchmarks": benchmarks,
        }
    )


def test_the_committed_budget_is_valid_and_says_what_is_missing():
    budget = load_budget(BUDGET)
    assert set(budget.categories) == {
        "coding",
        "mathematical_reasoning",
        "general_reasoning",
        "instruction_following",
        "long_context",
        "tool_use",
    }
    assert budget.baselines[0] == "original"
    assert not budget.categories["tool_use"].required
    # No dataset was downloaded in Phase 7A: every revision is unpinned, and the budget says so.
    assert len(budget.problems()) == len(budget.benchmarks)
    assert all("pin" in problem for problem in budget.problems())


def test_malformed_budgets_are_refused():
    import yaml

    document = yaml.safe_load(BUDGET.read_text(encoding="utf-8"))
    for change in (
        lambda d: d.update(version=2),
        lambda d: d["categories"].pop("tool_use"),
        lambda d: d.update(confidence=1.5),
        lambda d: d.update(baselines=["original", "original"]),
        lambda d: d["categories"]["coding"].update(max_regression=2),
        lambda d: d["benchmarks"][0].update(category="poetry"),
        lambda d: d["benchmarks"][0].update(surprise=1),
        lambda d: d["benchmarks"].append(dict(d["benchmarks"][0])),
        lambda d: d.update(benchmarks=[b for b in d["benchmarks"] if b["category"] != "long_context"]),
    ):
        broken = copy.deepcopy(document)
        change(broken)
        with pytest.raises(BudgetError):
            parse_budget(broken)


def test_wilson_and_newcombe_intervals_match_the_literature():
    # Wilson, 50 of 100 at 95%: 0.4038 to 0.5962.
    interval = wilson_interval(50, 100, 0.95)
    assert interval.low == pytest.approx(0.40383, abs=1e-4) and interval.high == pytest.approx(0.59617, abs=1e-4)
    # Wilson at 0 of 10: 0 to 0.2775.
    assert wilson_interval(0, 10, 0.95).high == pytest.approx(0.27753, abs=1e-4)
    # Newcombe (1998), example (a): 56/70 - 48/80, method 10 (hybrid score): 0.0524 to 0.3339.
    difference = newcombe_difference(56 / 70, 70, 48 / 80, 80, 0.95)
    assert difference.low == pytest.approx(0.0524, abs=1e-4) and difference.high == pytest.approx(0.3339, abs=1e-4)
    # Swapping the sides mirrors the interval.
    mirrored = newcombe_difference(48 / 80, 80, 56 / 70, 70, 0.95)
    assert mirrored.low == pytest.approx(-difference.high) and mirrored.high == pytest.approx(-difference.low)


def test_the_paired_bootstrap_is_deterministic_and_exact_where_it_must_be():
    baseline = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0] * 25
    shifted = [max(0.0, v - 0.25) for v in baseline]
    same = paired_bootstrap(baseline, baseline, 0.95, 2000, 7)
    assert (same.low, same.high) == (0.0, 0.0)
    a = paired_bootstrap(baseline, shifted, 0.95, 2000, 7)
    b = paired_bootstrap(baseline, shifted, 0.95, 2000, 7)
    assert a == b, "a seed fixes the interval"
    assert a.low <= 0.25 * 5 / 8 <= a.high
    noisy = [v if k % 3 else 1.0 - v for k, v in enumerate(baseline)]
    interval = paired_bootstrap(baseline, noisy, 0.95, 2000, 7)
    assert interval.low < sum(b - n for b, n in zip(baseline, noisy)) / len(baseline) < interval.high


def test_every_category_gets_its_own_verdict_and_the_worst_decides():
    budget = pinned_budget()
    baseline = results("original", {}, run="base")
    candidate = results(
        "candidate",
        {
            "humaneval": 0.70,  # no regression
            "mbpp-sanitized": 0.70,
            "gsm8k": 0.55,  # 15 points worse: beyond the budget
            "mmlu-pro": 0.68,  # two points worse: the interval straddles the 0.02 threshold
        },
        status={"longbench-v2": "unavailable", "bfcl": "skipped"},
        run="cand",
    )
    baseline_skips = results("original", {}, status={"bfcl": "skipped"}, run="base")
    comparison = compare(budget, baseline_skips, candidate)
    assert comparison.category("coding").verdict is Verdict.PASS
    assert comparison.category("mathematical_reasoning").verdict is Verdict.FAIL
    assert comparison.category("general_reasoning").verdict is Verdict.INCONCLUSIVE
    assert comparison.category("long_context").verdict is Verdict.PENDING
    assert comparison.category("tool_use").verdict is Verdict.SKIPPED
    assert comparison.verdict is Verdict.FAIL
    gsm8k = comparison.category("mathematical_reasoning").benchmarks[0]
    assert gsm8k.method == "newcombe" and gsm8k.regression == pytest.approx(0.15)
    assert gsm8k.interval.low > 0.02
    report = comparison.to_dict()
    assert report["format"] == "weightsift-quality-comparison" and report["verdict"] == "FAIL"
    json.dumps(report)  # serializable
    # The same without the failing benchmark: what is pending keeps the whole from passing.
    better = results("candidate", {"mmlu-pro": 0.70}, status={"longbench-v2": "unavailable", "bfcl": "skipped"})
    assert compare(budget, baseline_skips, better).verdict is Verdict.PENDING
    # A required category skipped is pending, not skipped.
    skipped = results("candidate", {}, status={"ifeval": "skipped", "bfcl": "skipped"})
    base = results("original", {}, status={"ifeval": "skipped", "bfcl": "skipped"})
    assert compare(budget, base, skipped).category("instruction_following").verdict is Verdict.PENDING


def test_item_scores_are_compared_in_pairs():
    budget = pinned_budget()
    base_items = [1.0 if k % 4 else 0.0 for k in range(600)]
    equal = compare(
        budget,
        results("original", {}, items={"gsm8k": base_items}),
        results("candidate", {}, items={"gsm8k": base_items}),
    )
    gsm8k = equal.category("mathematical_reasoning").benchmarks[0]
    assert gsm8k.method == "paired-bootstrap" and gsm8k.n_paired == 600
    assert (gsm8k.interval.low, gsm8k.interval.high) == (0.0, 0.0) and gsm8k.verdict is Verdict.PASS
    worse = [0.0 if k % 5 == 0 else v for k, v in enumerate(base_items)]
    gsm8k = compare(
        budget,
        results("original", {}, items={"gsm8k": base_items}),
        results("candidate", {}, items={"gsm8k": worse}),
    ).category("mathematical_reasoning").benchmarks[0]
    assert gsm8k.regression > 0.1 and gsm8k.verdict is Verdict.FAIL


def test_invalid_and_insufficient_comparisons_are_named():
    budget = pinned_budget()
    baseline = results("original", {})
    # Evaluation data in the candidate's calibration set.
    contaminated = results(
        "candidate", {}, calibration=[{"dataset": "openai/gsm8k", "config": "main", "split": "test"}]
    )
    comparison = compare(budget, baseline, contaminated)
    assert comparison.category("mathematical_reasoning").verdict is Verdict.INVALID
    assert "calibration" in comparison.category("mathematical_reasoning").benchmarks[0].reason
    # Fewer samples than the budget requires.
    few = results("candidate", {}, items={"humaneval": [1.0] * 100})
    base = results("original", {}, items={"humaneval": [1.0] * 100})
    assert compare(budget, base, few).category("coding").benchmarks[0].verdict is Verdict.INCONCLUSIVE
    # Different items on the two sides.
    a = results("original", {}, items={"gsm8k": [1.0] * 600})
    b_doc = copy.deepcopy(dict(results("candidate", {}, items={"gsm8k": [1.0] * 600}).document))
    for entry in b_doc["benchmarks"]:
        if entry["name"] == "gsm8k":
            entry["items"][0]["id"] = "other"
    assert compare(budget, a, parse_results(b_doc)).category("mathematical_reasoning").verdict is Verdict.INVALID
    # The unpinned committed budget keeps everything pending.
    assert compare(load_budget(BUDGET), baseline, results("candidate", {})).verdict is Verdict.PENDING


def test_the_strongest_available_baseline_is_used_and_its_kind_reported():
    budget = pinned_budget()
    derived = results("derived-quantized", {}, run="derived")
    original = results("original", {}, run="original")
    assert strongest_baseline(budget, [derived, original]).run_id == "original"
    assert strongest_baseline(budget, [derived]).run_id == "derived"
    comparison = compare(budget, derived, results("candidate", {}))
    assert comparison.baseline_kind == "derived-quantized"
    assert any("original checkpoint's quality is not measured" in note for note in comparison.notes)
    with pytest.raises(ValueError):
        compare(budget, results("candidate", {}), results("candidate", {}))


def test_results_never_claim_what_was_not_measured():
    good = dict(results("candidate", {}).document)
    for change, message in (
        (lambda e: e.update(status="unavailable", reason="no checkpoint", n=None, score=0.5), "no score"),
        (lambda e: e.update(status="unavailable", reason=None, n=None, score=None), "reason"),
        (lambda e: e.update(n=None), "n >= 1"),
        (lambda e: e.update(score=1.5), r"in \[0, 1\]"),
        (lambda e: e.update(n=2, score=0.5, items=[{"id": "a", "score": 1.0}, {"id": "b", "score": 1.0}]), "mean"),
        (lambda e: e.update(n=2, score=1.0, items=[{"id": "a", "score": 1.0}, {"id": "a", "score": 1.0}]), "unique"),
        (lambda e: e.update(extra=1), "unknown"),
    ):
        broken = copy.deepcopy(good)
        change(broken["benchmarks"][0])
        with pytest.raises(ResultsError, match=message):
            parse_results(broken)


def test_the_committed_schema_is_the_codes():
    committed = json.loads((ROOT / "configs" / "quality-results.schema.json").read_text(encoding="utf-8"))
    assert committed == RESULTS_SCHEMA


def test_the_cli_checks_a_budget_and_compares_results(tmp_path, capsys):
    from awpmi import cli

    assert cli.main(["quality", "check", "--budget", str(BUDGET)]) == 1  # unpinned revisions: not runnable yet
    assert "pin" in capsys.readouterr().out
    pinned = tmp_path / "budget.yaml"
    import yaml

    document = yaml.safe_load(BUDGET.read_text(encoding="utf-8"))
    for benchmark in document["benchmarks"]:
        benchmark["revision"] = "pinned-for-test"
    pinned.write_text(yaml.safe_dump(document), encoding="utf-8")
    assert cli.main(["quality", "check", "--budget", str(pinned)]) == 0
    base, cand = tmp_path / "base.json", tmp_path / "cand.json"
    base.write_text(json.dumps(dict(results("original", {}).document)), encoding="utf-8")
    cand.write_text(json.dumps(dict(results("candidate", {"gsm8k": 0.5}).document)), encoding="utf-8")
    report = tmp_path / "report.json"
    status = cli.main(
        ["quality", "compare", "--budget", str(pinned), "--baseline", str(base), "--candidate", str(cand),
         "--output", str(report)]
    )
    assert status == 1  # FAIL
    assert json.loads(report.read_text(encoding="utf-8"))["verdict"] == "FAIL"
