"""Quality results (format `weightsift-quality-results`, version 1): one subject (a baseline or a candidate) evaluated on a
Quality Budget's benchmarks, with the run that produced it.

A result is measured (a score in [0, 1] over n items, optionally every item's score, which makes paired comparisons
possible), unavailable or skipped; the last two carry a reason and no score. Nothing is ever filled in: a benchmark the
hardware or the checkpoint could not run is `unavailable`, never estimated. `RESULTS_SCHEMA` is the JSON Schema of the
format (also committed as `configs/quality-results.schema.json` for tools outside Weightsift); `parse_results` checks a
document strictly, beyond what the schema can say (an item list consistent with its score and count).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from awpmi.quality.spec import BASELINE_KINDS

FORMAT = "weightsift-quality-results"
VERSION = 1
SUBJECT_KINDS = (*BASELINE_KINDS, "candidate")
STATUSES = ("measured", "unavailable", "skipped")

_DATASET = {
    "type": "object",
    "additionalProperties": False,
    "required": ["dataset", "split"],
    "properties": {
        "dataset": {"type": "string", "minLength": 1},
        "config": {"type": ["string", "null"]},
        "revision": {"type": ["string", "null"]},
        "split": {"type": "string", "minLength": 1},
    },
}

RESULTS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/suffro/weightsift/configs/quality-results.schema.json",
    "title": "Weightsift quality results",
    "description": "One subject (a baseline or a candidate) evaluated on a Quality Budget's benchmarks (decision 0014). "
    "A benchmark that could not be run is 'unavailable' with a reason and no score; nothing is estimated.",
    "type": "object",
    "additionalProperties": False,
    "required": ["format", "version", "run", "subject", "calibration", "benchmarks"],
    "properties": {
        "format": {"const": FORMAT},
        "version": {"const": VERSION},
        "run": {
            "type": "object",
            "additionalProperties": False,
            "required": ["id", "date", "hardware", "software", "commands"],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "date": {"type": "string", "minLength": 1},
                "hardware": {"type": "object"},
                "software": {"type": "object"},
                "commands": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string"},
            },
        },
        "subject": {
            "type": "object",
            "additionalProperties": False,
            "required": ["kind", "model", "checkpoint", "representation", "engine"],
            "properties": {
                "kind": {"enum": list(SUBJECT_KINDS)},
                "model": {"type": "string", "minLength": 1},
                "checkpoint": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["repository", "revision"],
                    "properties": {
                        "repository": {"type": "string", "minLength": 1},
                        "revision": {"type": "string", "minLength": 1},
                    },
                },
                "representation": {"type": "string", "minLength": 1},
                "engine": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "revision"],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "revision": {"type": "string", "minLength": 1},
                        "configuration": {"type": "object"},
                    },
                },
            },
        },
        "calibration": {"type": "array", "items": _DATASET},
        "benchmarks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "category", "dataset", "revision", "split", "metric", "status"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "category": {"type": "string", "minLength": 1},
                    "dataset": {"type": "string", "minLength": 1},
                    "config": {"type": ["string", "null"]},
                    "revision": {"type": ["string", "null"]},
                    "split": {"type": "string", "minLength": 1},
                    "metric": {"type": "string", "minLength": 1},
                    "status": {"enum": list(STATUSES)},
                    "reason": {"type": ["string", "null"]},
                    "n": {"type": ["integer", "null"], "minimum": 1},
                    "score": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                    "items": {
                        "type": ["array", "null"],
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["id", "score"],
                            "properties": {
                                "id": {"type": "string", "minLength": 1},
                                "score": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                        },
                    },
                    "generation": {"type": "object"},
                },
                "allOf": [
                    {
                        "if": {"properties": {"status": {"const": "measured"}}},
                        "then": {
                            "required": ["n", "score"],
                            "properties": {"n": {"type": "integer"}, "score": {"type": "number"}},
                        },
                        "else": {
                            "required": ["reason"],
                            "properties": {
                                "reason": {"type": "string", "minLength": 1},
                                "score": {"type": "null"},
                                "n": {"type": "null"},
                                "items": {"type": "null"},
                            },
                        },
                    }
                ],
            },
        },
    },
}


class ResultsError(ValueError):
    """A results document that is malformed, inconsistent, or claims what it did not measure."""


@dataclass(frozen=True)
class DatasetRef:
    dataset: str
    config: str | None
    revision: str | None
    split: str

    @property
    def data(self) -> tuple[str, str | None, str]:
        return (self.dataset, self.config, self.split)


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    category: str
    dataset: str
    config: str | None
    revision: str | None
    split: str
    metric: str
    status: str
    reason: str | None
    n: int | None
    score: float | None
    items: tuple[tuple[str, float], ...] | None

    @property
    def measured(self) -> bool:
        return self.status == "measured"


@dataclass(frozen=True)
class Subject:
    kind: str
    model: str
    repository: str
    revision: str
    representation: str
    engine: str
    engine_revision: str


@dataclass(frozen=True)
class Results:
    run_id: str
    subject: Subject
    calibration: tuple[DatasetRef, ...]
    benchmarks: tuple[BenchmarkResult, ...]
    document: Mapping[str, Any]

    def benchmark(self, name: str) -> BenchmarkResult | None:
        for result in self.benchmarks:
            if result.name == name:
                return result
        return None


def _keys(entry: Mapping[str, Any], required: set[str], allowed: set[str], where: str) -> None:
    if not isinstance(entry, Mapping):
        raise ResultsError(f"{where}: must be an object")
    missing = required - set(entry)
    if missing:
        raise ResultsError(f"{where}: missing {sorted(missing)}")
    unknown = set(entry) - allowed
    if unknown:
        raise ResultsError(f"{where}: unknown fields {sorted(unknown)}")


def _text(value: Any, where: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise ResultsError(f"{where}: a non-empty string is required")
    return value


def _unit(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ResultsError(f"{where}: a number in [0, 1] is required")
    return float(value)


def _dataset(entry: Mapping[str, Any], where: str) -> DatasetRef:
    _keys(entry, {"dataset", "split"}, {"dataset", "config", "revision", "split"}, where)
    return DatasetRef(
        _text(entry["dataset"], where),
        _text(entry.get("config"), where, optional=True),
        _text(entry.get("revision"), where, optional=True),
        _text(entry["split"], where),
    )


def _benchmark(entry: Mapping[str, Any]) -> BenchmarkResult:
    where = f"benchmark {entry.get('name', '?') if isinstance(entry, Mapping) else '?'}"
    _keys(
        entry,
        {"name", "category", "dataset", "revision", "split", "metric", "status"},
        {"name", "category", "dataset", "config", "revision", "split", "metric", "status", "reason", "n", "score",
         "items", "generation"},
        where,
    )
    status = entry["status"]
    if status not in STATUSES:
        raise ResultsError(f"{where}: status must be one of {STATUSES}")
    n, score, items = entry.get("n"), entry.get("score"), entry.get("items")
    if status == "measured":
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ResultsError(f"{where}: a measured result needs n >= 1")
        score = _unit(score, where)
        if items is not None:
            if not isinstance(items, list) or len(items) != n:
                raise ResultsError(f"{where}: items must list exactly n scores")
            parsed = []
            for item in items:
                _keys(item, {"id", "score"}, {"id", "score"}, f"{where} item")
                parsed.append((_text(item["id"], f"{where} item"), _unit(item["score"], f"{where} item")))
            if len({i for i, _ in parsed}) != len(parsed):
                raise ResultsError(f"{where}: item ids must be unique")
            if not math.isclose(sum(s for _, s in parsed) / n, score, rel_tol=0, abs_tol=1e-9):
                raise ResultsError(f"{where}: the score is not the mean of the items")
            items = tuple(parsed)
    else:
        # Never invent: what was not measured carries a reason and no number.
        _text(entry.get("reason"), f"{where}: a {status} result needs a reason")
        if score is not None or n is not None or items is not None:
            raise ResultsError(f"{where}: a {status} result has no score, n or items")
    return BenchmarkResult(
        name=_text(entry["name"], where),
        category=_text(entry["category"], where),
        dataset=_text(entry["dataset"], where),
        config=_text(entry.get("config"), where, optional=True),
        revision=_text(entry["revision"], where, optional=True),
        split=_text(entry["split"], where),
        metric=_text(entry["metric"], where),
        status=status,
        reason=entry.get("reason"),
        n=n if status == "measured" else None,
        score=score if status == "measured" else None,
        items=items if status == "measured" else None,
    )


def parse_results(document: Mapping[str, Any]) -> Results:
    _keys(document, {"format", "version", "run", "subject", "calibration", "benchmarks"},
          {"format", "version", "run", "subject", "calibration", "benchmarks"}, "results")
    if document["format"] != FORMAT or document["version"] != VERSION:
        raise ResultsError(f"not a {FORMAT} v{VERSION} document")
    run = document["run"]
    _keys(run, {"id", "date", "hardware", "software", "commands"},
          {"id", "date", "hardware", "software", "commands", "notes"}, "run")
    subject = document["subject"]
    _keys(subject, {"kind", "model", "checkpoint", "representation", "engine"},
          {"kind", "model", "checkpoint", "representation", "engine"}, "subject")
    if subject["kind"] not in SUBJECT_KINDS:
        raise ResultsError(f"subject: kind must be one of {SUBJECT_KINDS}")
    checkpoint, engine = subject["checkpoint"], subject["engine"]
    _keys(checkpoint, {"repository", "revision"}, {"repository", "revision"}, "subject checkpoint")
    _keys(engine, {"name", "revision"}, {"name", "revision", "configuration"}, "subject engine")
    benchmarks = tuple(_benchmark(entry) for entry in document["benchmarks"])
    if len({b.name for b in benchmarks}) != len(benchmarks):
        raise ResultsError("a benchmark appears twice")
    return Results(
        run_id=_text(run["id"], "run"),
        subject=Subject(
            kind=subject["kind"],
            model=_text(subject["model"], "subject"),
            repository=_text(checkpoint["repository"], "subject checkpoint"),
            revision=_text(checkpoint["revision"], "subject checkpoint"),
            representation=_text(subject["representation"], "subject"),
            engine=_text(engine["name"], "subject engine"),
            engine_revision=_text(engine["revision"], "subject engine"),
        ),
        calibration=tuple(_dataset(entry, "calibration") for entry in document["calibration"]),
        benchmarks=benchmarks,
        document=document,
    )


def load_results(path: str | Path) -> Results:
    return parse_results(json.loads(Path(path).read_text(encoding="utf-8")))
