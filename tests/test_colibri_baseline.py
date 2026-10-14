"""The Colibri baseline's specification and records (Phase 7A, decision 0014): the committed specification is complete and
pinned, the plan covers every configuration cold and warm, a machine without the engine or the checkpoint is told what
blocks it, and the engine's printed metrics (the formats of Inkling's engine at the pinned commit) are parsed."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from awpmi.baseline import SpecError, blockers, load_spec, parse_output, plan, prompts
from awpmi.hardware import detect

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs" / "phase7a-baseline.yaml"

# Lines as c/inkling.c prints them at the pinned commit (its printf formats, filled in).
INKLING_LOG = """\
[13 prompt tokens] The capital of France is
 Paris, the city of light.
[prefill 35.4s | 48 tokens in 190.2s = 0.25 tok/s | RSS 23.1 GB]
TUNE decode: 47 tokens in 190.214s
[phases] fill 120.5s | expert-mm 40.2s | shared 9.1s | attn 12.0s | other 43.8s
[cache] hit 82.2% (12345 hit / 2672 load) | usage history saved
PEAK RSS: 23.40 GB | expert cache hit 82.2% | 0.25 tok/s
"""


def test_the_committed_specification_is_complete_and_pinned():
    spec = load_spec(SPEC)
    assert spec["engine"]["commit"] == "bf2442915d6e3dd4cdfd2eb9c2a3d2aa44a25850"
    assert spec["model"]["artifact"]["kind"] == "derived-quantized"
    assert len(prompts(spec, ROOT)) >= 8
    # The engine pin is the one the build fetches.
    import json

    pins = json.loads((ROOT / "native" / "upstream.json").read_text(encoding="utf-8"))
    assert spec["engine"]["commit"] == next(p for p in pins["upstreams"] if p["name"] == "colibri")["commit"]


def test_incomplete_specifications_are_refused(tmp_path):
    document = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    for change in (
        lambda d: d["engine"].update(commit="main"),
        lambda d: d["model"]["artifact"].update(revision=None),
        lambda d: d["metrics"].remove("ssd_bytes_per_token"),
        lambda d: d["workload"].update(repeats=1),
        lambda d: d["configurations"].append(dict(d["configurations"][0])),
        lambda d: d["workload"]["states"].pop("warm"),
    ):
        broken = copy.deepcopy(document)
        change(broken)
        path = tmp_path / "spec.yaml"
        path.write_text(yaml.safe_dump(broken), encoding="utf-8")
        with pytest.raises(SpecError):
            load_spec(path)


def test_the_plan_runs_every_configuration_cold_and_warm_twice(tmp_path):
    spec = load_spec(SPEC)
    steps = plan(spec, colibri=tmp_path / "colibri", snapshot=tmp_path / "snap", prompt_file=tmp_path / "p.txt",
                 physical_cores=6)
    assert len(steps) == len(spec["configurations"]) * 2 * 2
    assert {(s.configuration, s.state, s.repeat) for s in steps} == {
        (c["name"], state, r) for c in spec["configurations"] for state in ("cold", "warm") for r in (0, 1)
    }
    first = steps[0]
    assert first.command[0].endswith("inkling") and "--chat" in first.command and "-f" in first.command
    assert first.environment["PIN"] == "off" and first.environment["OMP_NUM_THREADS"] == "6"
    assert first.environment["SNAP"] == str(tmp_path / "snap")


def test_what_blocks_the_run_is_said(tmp_path):
    spec = load_spec(SPEC)
    found = blockers(spec, colibri=None, snapshot=None)
    assert len(found) == 2
    assert any("approval" in b for b in found) and any("gcc" in b for b in found)
    built = tmp_path / "colibri" / "c"
    built.mkdir(parents=True)
    (built / "inkling").write_text("")
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    assert blockers(spec, colibri=tmp_path / "colibri", snapshot=snapshot) == []


def test_the_engines_printed_metrics_are_parsed():
    metrics = parse_output(INKLING_LOG)
    (prompt,) = metrics["prompts"]
    assert prompt["prefill_seconds"] == 35.4 and prompt["generated_tokens"] == 48
    assert prompt["decode_tokens"] == 47 and prompt["decode_seconds"] == 190.214
    assert prompt["decode_tokens_per_second"] == pytest.approx(47 / 190.214)
    assert prompt["phases_seconds"]["fill"] == 120.5 and prompt["phases_seconds"]["other"] == 43.8
    assert metrics["expert_cache"] == {"hits": 12345, "loads": 2672, "hit_ratio": pytest.approx(12345 / 15017)}
    assert metrics["peak_rss_gb"] == 23.40
    assert parse_output("nothing printed") == {"prompts": []}


def test_the_machine_is_detected_not_assumed():
    machine = detect(ROOT)
    memory = machine["host_memory"]
    assert memory["total_bytes"] and 0 < memory["available_bytes"] <= memory["total_bytes"]
    assert machine["logical_cpus"] >= 1
    assert machine["storage"]["total_bytes"] >= machine["storage"]["free_bytes"] > 0
    for gpu in machine["gpus"]:
        assert gpu["name"] and gpu["memory_total_bytes"] > 0
