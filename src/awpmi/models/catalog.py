"""Expert catalogs (Phase 7A, decision 0014): the description Weightsift's expert store (`native/store`) reads, written
by a model adapter from a checkpoint index (an expert index pack of Phases 4A and 4B: segments over the published
checkpoint's files, built from headers only).

A catalog (`weightsift-expert-catalog` v1) lists files, segments (plain or composed, as Weightsift's storage layout has
them) and layers: a layer's parts are segments whose row e is expert e, and its views name ranges of those rows with a
shape and a storage format. The model knowledge (which segments are a layer's experts, how a fused tensor splits into
gate, up and down) is the caller's; this module only translates.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from awpmi.storage.layout import ComposedSegment, Segment
from awpmi.storage.pack import Pack

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1}


@dataclass(frozen=True)
class ViewSpec:
    """A view: rows `first_row` to `first_row + rows` of a parameter's row (whose first dimension they index)."""

    name: str
    parameter: str
    first_row: int
    rows: int
    format: str


def expert_catalog(
    pack: Pack,
    *,
    name: str,
    pattern: str,
    parameters: Sequence[str],
    views: Sequence[ViewSpec],
    layers: Iterable[int] | None = None,
) -> dict[str, Any]:
    """The catalog of the experts of `pack`. `pattern` matches a segment name with two groups, the layer and the
    parameter; a layer's parts are `parameters` in that order (every one must exist); `layers` restricts the catalog."""
    regex = re.compile(pattern)
    found: dict[int, dict[str, str]] = {}
    for segment in pack.segments:
        match = regex.match(segment)
        if match and match.group(2) in parameters:
            found.setdefault(int(match.group(1)), {})[match.group(2)] = segment
    chosen = sorted(found) if layers is None else sorted(set(layers))
    files: dict[str, int] = {}
    segments: list[dict[str, Any]] = []
    entries = []
    for layer in chosen:
        by_parameter = found.get(layer, {})
        missing = [p for p in parameters if p not in by_parameter]
        if missing:
            raise ValueError(f"layer {layer} has no segment for {missing}")
        part_entries = []
        experts = None
        for parameter in parameters:
            info = pack.segments[by_parameter[parameter]]
            experts = info.rows if experts is None else experts
            if info.rows != experts:
                raise ValueError(f"layer {layer}: {parameter} has {info.rows} rows, not {experts}")
            for key in info.files:
                files.setdefault(key, len(files))
            segments.append(_segment(info, files))
            part_entries.append({"name": parameter, "segment": len(segments) - 1})
        view_entries = []
        for view in views:
            info = pack.segments[by_parameter[view.parameter]]
            per_row = math.prod(info.row_shape[1:]) * DTYPE_BYTES[info.dtype]
            if view.first_row + view.rows > info.row_shape[0]:
                raise ValueError(f"layer {layer}: view {view.name} runs past {view.parameter}'s rows")
            view_entries.append(
                {
                    "name": view.name,
                    "part": parameters.index(view.parameter),
                    "offset": view.first_row * per_row,
                    "bytes": view.rows * per_row,
                    "format": view.format,
                    "rows": view.rows,
                    "columns": math.prod(info.row_shape[1:]),
                }
            )
        entries.append({"layer": layer, "experts": experts, "parts": part_entries, "views": view_entries})
    return {
        "format": "weightsift-expert-catalog",
        "version": 1,
        "name": name,
        "files": [
            {
                "key": key,
                "path": str(pack.files[key]),
                "size": pack.manifest["files"][key]["bytes"],
                "sha256": pack.manifest["files"][key].get("sha256"),
            }
            for key in sorted(files, key=files.get)
        ],
        "segments": segments,
        "layers": entries,
    }


def _segment(info: Segment | ComposedSegment, files: Mapping[str, int]) -> dict[str, Any]:
    if isinstance(info, ComposedSegment):
        local = list(dict.fromkeys(key for row in info.spans for key, _ in row))
        return {
            "name": info.name,
            "files": [files[key] for key in local],
            "rows": info.rows,
            "row_bytes": info.row_bytes,
            "part_bytes": list(info.part_bytes),
            "spans": [[local.index(key), offset] for row in info.spans for key, offset in row],
        }
    return {
        "name": info.name,
        "files": [files[info.file]],
        "rows": info.rows,
        "row_bytes": info.row_bytes,
        "offset": info.offset,
    }


def write_expert_catalog(catalog: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, indent=1) + "\n", encoding="utf-8")
    return path
