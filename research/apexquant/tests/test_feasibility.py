"""The ApexQuant screening's arithmetic (Phase 7A): the native formats' bits per weight from the checkpoints' headers,
the paper's accounting, and the screening's conclusions."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("feasibility", HERE / "feasibility.py")
feasibility = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feasibility)


@pytest.fixture(scope="module")
def screened():
    geometry = json.loads((HERE / "geometry.json").read_text(encoding="utf-8"))
    headers = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((HERE / "headers").glob("*.json"))}
    return feasibility.screen(geometry, headers, store_rate=3e9)


def test_the_papers_accounting():
    # Appendix H: 2.03, 4.06 and 6.09 bits per weight at one, two and three passes (b = 2, g = 512).
    assert [round(feasibility.apexquant_bits_per_weight(p), 2) for p in (1, 2, 3)] == [2.03, 4.06, 6.09]


def test_native_formats_from_the_headers(screened):
    mimo = screened["mimo-pro"]["formats"]["mimo-v2.6-pro"]
    # MXFP4: 4-bit values and one 8-bit scale per 32 values.
    assert mimo["bits_per_weight"] == pytest.approx(4.25, abs=1e-12)
    assert mimo["expert_bytes"] == 3 * (6_291_456 + 393_216)
    int4 = screened["inkling"]["formats"]["inkling-colibri-int4"]
    # Colibri's container: 4-bit values and one f32 scale per row (rows of 6144 and 3072 values).
    expected = (2 * 3072 * (6144 * 4 + 32) + 6144 * (3072 * 4 + 32)) / (3 * 3072 * 6144)
    assert int4["bits_per_weight"] == pytest.approx(expected)
    assert screened["inkling"]["formats"]["inkling-bf16"]["bits_per_weight"] == 16


def test_the_screening(screened):
    for model in screened.values():
        assert model["groups_fit"], "every expert matrix row splits into whole groups of 512"
        formats = model["formats"]
        assert formats["apexquant-p1"]["meets_screening_target"]
        assert not formats["apexquant-p2"]["meets_screening_target"]
        assert not formats["apexquant-p3"]["meets_screening_target"]
    # The four-bit class: about 4.4% fewer bytes than MXFP4, 1.4% more than Colibri's int4 container.
    assert screened["mimo-pro"]["formats"]["apexquant-p2"]["byte_reduction"] == pytest.approx(0.0441, abs=1e-4)
    assert screened["inkling"]["formats"]["apexquant-p2"]["byte_reduction"] == pytest.approx(-0.0139, abs=1e-4)
    assert screened["inkling"]["experts_per_token"] == 6 * 64 and screened["mimo-pro"]["experts_per_token"] == 8 * 69
