"""Expert catalogs (Phase 7A, decision 0014): a catalog written from an expert index pack describes exactly the bytes the
pack's segments hold (rebuilt from the catalog's spans and files, compared with the Python store's reads), and its views
split each expert's rows where the caller said."""

from __future__ import annotations

import json

import pytest
import torch

from awpmi.models.catalog import ViewSpec, expert_catalog, write_expert_catalog
from awpmi.storage.layout import ComposedSegment
from awpmi.storage.pack import PackWriter, SourceFile, open_pack, sha256_file_direct
from tests.test_composed_storage import FILES, build_layout

PATTERN = r"^model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$"
EXPERTS = 6
# Rows of two BF16 matrices: gate_up [2 * 4, 16], down [16, 4].
GATE_UP = (2 * 4, 16)
DOWN = (16, 4)


@pytest.fixture
def index(tmp_path):
    generator = torch.Generator().manual_seed(7)
    segments = []
    layouts = []
    for layer in (1, 2):
        # Rows of 8 x 16 and 16 x 4 BF16 values: the gate/up row as two spans (its gate and up halves), down as one.
        for parameter, shape, parts in (("gate_up_proj", GATE_UP, (128, 128)), ("down_proj", DOWN, (128,))):
            spans, sizes = build_layout(generator, EXPERTS, parts)
            layouts.append((spans, sizes))
            segments.append(
                ComposedSegment(
                    f"model.layers.{layer}.mlp.experts.{parameter}", EXPERTS, sum(parts), "BF16", shape, parts,
                    tuple(tuple(spans[(r, p)] for p in range(len(parts))) for r in range(EXPERTS)),
                )
            )
    # One set of files large enough for every layout (each layout placed its spans from its own cursors).
    sizes = {f: max(size[f] for _, size in layouts) for f in FILES}
    paths = {}
    for f in FILES:
        paths[f] = tmp_path / f"{f}.bin"
        paths[f].write_bytes(torch.randint(0, 256, (sizes[f],), dtype=torch.uint8, generator=generator).numpy().tobytes())
    writer = PackWriter(tmp_path / "index", "moe-experts")
    for f in FILES:
        writer.add_source(f, SourceFile("example/moe", "0" * 40, f"{f}.bin"), paths[f], sha256=sha256_file_direct(paths[f]))
    for segment in segments:
        writer.add_composed_segment(segment)
    writer.write({"note": "test index"})
    return open_pack(tmp_path / "index", verify="size", resolve=lambda s: paths[s.filename.removesuffix(".bin")])


VIEWS = (
    ViewSpec("gate", "gate_up_proj", 0, 4, "bf16"),
    ViewSpec("up", "gate_up_proj", 4, 4, "bf16"),
    ViewSpec("down", "down_proj", 0, 16, "bf16"),
)


def test_a_catalog_describes_exactly_the_packs_bytes(index, tmp_path):
    catalog = expert_catalog(
        index, name="test", pattern=PATTERN, parameters=("gate_up_proj", "down_proj"), views=VIEWS
    )
    path = write_expert_catalog(catalog, tmp_path / "catalog.json")
    assert json.loads(path.read_text(encoding="utf-8")) == catalog
    assert [layer["layer"] for layer in catalog["layers"]] == [1, 2]
    store = index.store(direct=True)
    data = {k: (tmp_path / f"{k}.bin").read_bytes() for k in FILES}
    try:
        for entry in catalog["segments"]:
            parts = entry["part_bytes"]
            reads = store.read_rows(entry["name"])
            for row in range(entry["rows"]):
                rebuilt = b"".join(
                    data[catalog["files"][entry["files"][file]]["key"]][offset : offset + parts[p]]
                    for p, (file, offset) in enumerate(entry["spans"][row * len(parts) : (row + 1) * len(parts)])
                )
                assert rebuilt == reads[row].numpy().tobytes(), f"{entry['name']} row {row}"
    finally:
        store.close()
    layer = catalog["layers"][0]
    assert layer["experts"] == EXPERTS and [p["name"] for p in layer["parts"]] == ["gate_up_proj", "down_proj"]
    gate, up, down = layer["views"]
    row_bytes = 16 * 2  # one row of the gate_up matrix, BF16
    assert (gate["part"], gate["offset"], gate["bytes"]) == (0, 0, 4 * row_bytes)
    assert (up["part"], up["offset"], up["bytes"]) == (0, 4 * row_bytes, 4 * row_bytes)
    assert (down["part"], down["offset"], down["bytes"]) == (1, 0, 16 * 4 * 2)
    assert (gate["rows"], gate["columns"]) == (4, 16)


def test_a_catalog_refuses_what_it_cannot_describe(index):
    with pytest.raises(ValueError, match="no segment"):
        expert_catalog(index, name="t", pattern=PATTERN, parameters=("gate_up_proj", "w3"), views=())
    with pytest.raises(ValueError, match="runs past"):
        expert_catalog(index, name="t", pattern=PATTERN, parameters=("gate_up_proj", "down_proj"),
                       views=(ViewSpec("gate", "gate_up_proj", 4, 5, "bf16"),))
    restricted = expert_catalog(index, name="t", pattern=PATTERN, parameters=("gate_up_proj", "down_proj"), views=(),
                                layers=[2])
    assert [layer["layer"] for layer in restricted["layers"]] == [2] and len(restricted["segments"]) == 2
