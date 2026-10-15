"""Quantized expert matrices of the fetched sample: their stored bytes, decoded into symbols and scales, and the
context models specific to quantized formats.

Formats (the sample manifest's `layout` and the configuration's `format`):

  int4_row        Colibri's per-row containers (GLM-5.2, Inkling): packed nibbles, low nibble first, encoding detected
                  per matrix (Colibri's converters write level + 8); one F32 scale per row (`.qs`)
  int4_g32_bf16   compressed-tensors pack-quantized (Kimi-K2-Thinking): int32 words of 8 nibbles (level + 8, nibble i at
                  bits 4i, i.e. low nibble first in little-endian bytes); one BF16 scale per 32 weights of a row
  mxfp4           OCP MXFP4 (MiMo-V2.6 Pro, gpt-oss): packed E2M1 nibbles, low first; one E8M0 byte per 32 weights

A matrix keeps its exact stored bytes (`weight_bytes`, `scale_bytes`), so that every reconstruction is compared with
what the checkpoint holds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import contexts as cx
from . import symbols
from .measure import Spec

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class QMatrix:
    source: str
    format: str
    layer: int
    expert: int
    kind: str
    rows: int
    cols: int
    weight_bytes: np.ndarray  # uint8 [rows, cols / 2], the stored packed bytes
    scale_bytes: np.ndarray  # uint8, the stored scale bytes of this matrix
    group: int  # weights per scale (cols for per-row scales)
    weight_sha256: str
    scale_sha256: str

    @property
    def key(self) -> str:
        return f"{self.source}/L{self.layer}/E{self.expert}/{self.kind}"

    @property
    def weights(self) -> int:
        return self.rows * self.cols

    def nibbles(self) -> np.ndarray:
        return symbols.unpack_nibbles(self.weight_bytes)

    def scales(self) -> np.ndarray:
        """Scales as values (float64) per (row, group)."""
        if self.format == "int4_row":
            return self.scale_bytes.view("<f4").astype(np.float64).reshape(self.rows, 1)
        if self.format == "int4_g32_bf16":
            p = self.scale_bytes.view("<u2").astype(np.uint32) << 16
            return p.view(np.float32).astype(np.float64).reshape(self.rows, -1)
        if self.format == "mxfp4":
            return np.exp2(self.scale_bytes.astype(np.float64) - 127.0).reshape(self.rows, -1)
        raise ValueError(self.format)


def load(manifest_path: Path, config: dict) -> list[QMatrix]:
    """Every matrix of the sample, from the manifest's slices (sha256 checked on read)."""
    import hashlib

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    by_expert: dict[tuple[str, int, int], list[dict]] = {}
    for s in manifest["slices"]:
        by_expert.setdefault((s["source"], s["layer"], s["expert"]), []).append(s)
    out = []
    for (source, layer, expert), slices in sorted(by_expert.items()):
        spec = config["samples"][source]
        fmt = spec["format"]
        data = {}
        for s in slices:
            raw = (REPO_ROOT / s["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != s["sha256"]:
                raise SystemExit(f"{s['path']}: sha256 differs from the manifest")
            data[(s["kind"], s["role"])] = np.frombuffer(raw, dtype=np.uint8)
        for kind, rows, cols, w, sc, row0 in _matrices(spec, data):
            group = cols if fmt == "int4_row" else 32
            wb = np.ascontiguousarray(w.reshape(rows, cols // 2))
            out.append(QMatrix(source, fmt, layer, expert, kind, rows, cols, wb, np.ascontiguousarray(sc), group,
                               hashlib.sha256(wb.tobytes()).hexdigest(), hashlib.sha256(sc.tobytes()).hexdigest()))
    return out


def _matrices(spec: dict, data: dict):
    """(kind, rows, cols, weight bytes, scale bytes, first row) of each matrix of one expert."""
    shapes = spec["shapes"]
    if spec["layout"] in ("per_expert_colibri_int4", "per_expert_mxfp4", "per_expert_ct_int4"):
        for kind in ("gate", "up", "down"):
            rows, cols = shapes[kind]
            yield kind, rows, cols, data[(kind, "weight")], data[(kind, "scale")], 0
        return
    if spec["layout"] == "fused_colibri_int4":  # gate rows then up rows in gate_up; per-row F32 scales
        rows, cols = shapes["gate_up"]
        w = data[("gate_up", "weight")].reshape(rows, cols // 2)
        s = data[("gate_up", "scale")].reshape(rows, 4)
        half = rows // 2
        yield "gate", half, cols, w[:half], s[:half].reshape(-1), 0
        yield "up", half, cols, w[half:], s[half:].reshape(-1), half
        rows, cols = shapes["down"]
        yield "down", rows, cols, data[("down", "weight")], data[("down", "scale")], 0
        return
    if spec["layout"] == "fused_gptoss_mxfp4":
        for kind in ("gate_up", "down"):
            rows, cols = shapes[kind]
            yield kind, rows, cols, data[(kind, "weight")], data[(kind, "scale")], 0
        return
    raise ValueError(spec["layout"])


# ---------------------------------------------------------------------------------------------------------------------
# Symbols and activities


def magnitudes(m: QMatrix, nib: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """(activity per weight: the code's magnitude; a 'top' flag: the code is at its group's maximal level; encoding)."""
    if m.format == "mxfp4":
        _, mag = symbols.e2m1_fields(nib)
        return symbols.E2M1_MAGNITUDES[mag], mag >= 6, "e2m1"
    enc = "offset8" if m.format == "int4_g32_bf16" else symbols.detect_int4_encoding(nib)
    lev = symbols.signed_int4(nib, enc).astype(np.int16)
    return np.abs(lev).astype(np.float64), np.abs(lev) >= 7, enc


def group_scale_classes(m: QMatrix, q: int = 16) -> np.ndarray:
    """A class per weight from its group's scale relative to its row's median scale (log2, fixed bins of [-4, 4]):
    derivable by any decoder from the scales it already holds, so it costs no side information. For per-row formats,
    the class of the row's scale relative to the matrix's median."""
    s = m.scales()
    with np.errstate(divide="ignore"):
        ls = np.log2(np.maximum(s, 1e-30))
    if m.format == "int4_row":
        rel = ls[:, 0] - np.median(ls[:, 0])
        cls = cx.quantize(rel, -2.0, 2.0, q)
        return np.broadcast_to(cls[:, None], (m.rows, m.cols)).astype(np.int32)
    rel = ls - np.median(ls, axis=1, keepdims=True)
    cls = cx.quantize(rel, -4.0, 4.0, q)
    return np.repeat(cls, m.group, axis=1)[:, : m.cols].astype(np.int32)


def block_context(top: np.ndarray, group: int) -> cx.Context:
    """Causal, within each group of `group` weights of a row: whether a maximal-level code has been seen so far in the
    group, and how many weights of the group remain (capped at 8). A group quantized by its own absolute maximum
    contains such a code; when none has appeared, the last positions must carry it."""
    rows, cols = top.shape
    t = top.reshape(rows, cols // group, group).astype(np.int32)
    seen = np.zeros_like(t)
    seen[:, :, 1:] = np.cumsum(t, axis=2)[:, :, :-1] > 0
    remaining = np.minimum(group - np.arange(group), 8)[None, None, :] - 1
    ids = (remaining * 2 + seen).reshape(rows, cols).astype(np.int32)
    return cx.Context(ids, 16)


def nibble_specs(m: QMatrix, nib: np.ndarray, extra: dict) -> list[Spec]:
    """The nibble stream under the context models (alphabet 16), activity = the code's magnitude."""
    sym = nib.astype(np.int32)
    mag, top, _ = magnitudes(m, nib)
    k = 16
    hi = float(mag.max()) if mag.size else 1.0
    specs = [
        Spec("code", "none", sym, k, cx.none(sym.shape)),
        Spec("code", "left", sym, k, cx.left(sym, k)),
        Spec("code", "left2", sym, k, cx.left2(sym, k)),
        Spec("code", "above", sym, k, cx.above(sym, k)),
        Spec("code", "left_above", sym, k, cx.left_above(sym, k)),
        Spec("code", "row_mean8", sym, k, cx.row_activity(mag, 8, 0.0, hi, 16)),
        Spec("code", "row_mean64", sym, k, cx.row_activity(mag, 64, 0.0, hi / 2, 16)),
        Spec("code", "row_mean512", sym, k, cx.row_activity(mag, 512, 0.0, hi / 2, 16)),
        Spec("code", "col_mean_causal", sym, k, cx.column_activity(mag, 0.0, hi / 2, 16)),
        Spec("code", "col_class16", sym, k, cx.column_class(mag, 16)),
        Spec("code", "row_class16", sym, k, cx.row_class(mag, 16)),
        Spec("code", "rank1_class32", sym, k, cx.rank1_class(mag, 32)),
        Spec("code", "col_class16+row_mean512", sym, k,
             cx.product(cx.column_class(mag, 16), cx.row_activity(mag, 512, 0.0, hi / 2, 16))),
    ]
    gsc = group_scale_classes(m)
    specs.append(Spec("code", "scale_class16", sym, k, cx.Context(gsc, 16)))
    specs.append(Spec("code", "scale_class16+col_class16", sym, k,
                      cx.product(cx.Context(gsc, 16), cx.column_class(mag, 16))))
    if m.group < m.cols:
        specs.append(Spec("code", "block_max", sym, k, block_context(top, m.group)))
        specs.append(Spec("code", "block_max+scale_class16", sym, k,
                          cx.product(block_context(top, m.group), cx.Context(gsc, 16))))
    for label, other in extra.get("given", {}).items():
        specs.append(Spec("code", f"given_{label}", sym, k, cx.given(other.astype(np.int32), 16)))
    if "neuron_classes" in extra:
        specs.append(Spec("code", "neuron_class16", sym, k, cx.given_rows(extra["neuron_classes"], 16, sym.shape, axis=1)))
    return specs


def scale_specs(m: QMatrix) -> list[Spec]:
    """The scale stream of a group-quantized matrix: E8M0 bytes, or the exponent byte of BF16 scales (their sign and
    mantissa are measured too)."""
    if m.format == "int4_row":
        return []
    # Scale symbols keep the full byte alphabet (the same for every matrix, so that counts pool across matrices); the
    # adaptive estimator spreads its prior only over the symbols a matrix holds (a header's bitmap).
    k = 256
    if m.format == "mxfp4":
        s = m.scale_bytes.reshape(m.rows, -1).astype(np.int32)
        act = s.astype(np.float64)
        return [
            Spec("scale_e8m0", "none", s, k, cx.none(s.shape)),
            Spec("scale_e8m0", "left", s, k, cx.left(s, k)),
            Spec("scale_e8m0", "above", s, k, cx.above(s, k)),
            Spec("scale_e8m0", "col_class16", s, k, cx.column_class(act, 16)),
            Spec("scale_e8m0", "row_class16", s, k, cx.row_class(act, 16)),
        ]
    p = m.scale_bytes.view("<u2").reshape(m.rows, -1)
    sign, e, mant = symbols.bf16_fields(p)
    ranks = e.astype(np.int32)
    act = e.astype(np.float64)
    mt = mant.astype(np.int32)
    return [
        Spec("scale_bf16_exponent", "none", ranks, k, cx.none(e.shape)),
        Spec("scale_bf16_exponent", "left", ranks, k, cx.left(ranks, k)),
        Spec("scale_bf16_exponent", "above", ranks, k, cx.above(ranks, k)),
        Spec("scale_bf16_exponent", "col_class16", ranks, k, cx.column_class(act, 16)),
        Spec("scale_bf16_exponent", "row_class16", ranks, k, cx.row_class(act, 16)),
        Spec("scale_bf16_mantissa", "none", mt, 128, cx.none(e.shape)),
        Spec("scale_bf16_mantissa", "exponent", mt, 128, cx.Context(ranks.astype(np.int32), k)),
        Spec("scale_bf16_sign", "none", sign.astype(np.int32), 2, cx.none(e.shape)),
    ]


def neuron_classes_from(gate: QMatrix, up: QMatrix, q: int = 16) -> np.ndarray:
    """A class per neuron from the mean code magnitude of its gate and up rows (decoded before down)."""
    gm = magnitudes(gate, gate.nibbles())[0].mean(axis=1)
    um = magnitudes(up, up.nibbles())[0].mean(axis=1)
    stat = gm + um
    edges = np.quantile(stat, np.linspace(0, 1, q + 1)[1:-1])
    return np.searchsorted(edges, stat, side="right").astype(np.int32)
