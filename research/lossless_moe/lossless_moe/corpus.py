"""The study's corpus: routed-expert matrices of local checkpoints, as stored, with a deterministic sample.

A checkpoint is named in the configuration (repository, pinned revision, expert layout). Its routed experts are found
from the safetensors headers alone; a sample is a fixed list of layers and, per layer, experts drawn by a seeded
permutation (`numpy.random.default_rng([seed, crc32(model), layer])`, independent of Python's hash seed).

Layouts:
  per_expert      `model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight`, one 2-D tensor each (Moonlight, OLMoE)
  fused_granite   `model.layers.{L}.block_sparse_moe.input_linear.weight` [E, 2I, H] (gate rows, then up rows, as
                  transformers' GraniteMoe chunks them) and `output_linear.weight` [E, H, I] (down)

Two read paths that share no code, as in Phase 5C:
  read_patterns   positioned reads of the file at the header's offset: the bytes as stored, uint16 [rows, cols]
  read_reference  safetensors' own `safe_open(...).get_slice`: the independent copy every reconstruction is compared to
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from awpmi.storage.layout import read_safetensors_header

MATRICES = ("gate", "up", "down")


@dataclass(frozen=True)
class Matrix:
    model: str
    layer: int
    expert: int
    kind: str  # gate, up, down
    tensor: str  # the checkpoint tensor holding it
    file: str
    offset: int  # absolute byte offset of the matrix in the file
    rows: int
    cols: int
    dtype: str
    index: tuple[int, ...] = ()  # leading indices into `tensor` (fused layouts: expert, then a row range start/stop)

    @property
    def nbytes(self) -> int:
        return self.rows * self.cols * 2

    @property
    def key(self) -> str:
        return f"{self.model}/L{self.layer}/E{self.expert}/{self.kind}"

    def to_json(self) -> dict:
        return {"key": self.key, "tensor": self.tensor, "file": self.file, "offset": self.offset,
                "shape": [self.rows, self.cols], "dtype": self.dtype, "index": list(self.index)}


def sample_experts(seed: int, model: str, layer: int, n_experts: int, k: int) -> list[int]:
    rng = np.random.default_rng([seed, zlib.crc32(model.encode("utf-8")), layer])
    return sorted(int(e) for e in rng.choice(n_experts, size=k, replace=False))


class Checkpoint:
    """A pinned checkpoint in the local Hugging Face cache (never downloaded here)."""

    def __init__(self, name: str, spec: dict) -> None:
        from huggingface_hub import hf_hub_download

        self.name, self.spec = name, spec
        self.repository, self.revision, self.layout = spec["repository"], spec["revision"], spec["layout"]
        # The pinned snapshot's directory, from a file it must hold (snapshot_download refuses a snapshot that lacks
        # files the study never reads, such as a README's figures).
        config = hf_hub_download(self.repository, "config.json", revision=self.revision, local_files_only=True)
        self.root = Path(config).parent
        self.files = sorted(p.name for p in self.root.glob("*.safetensors"))
        self._headers: dict[str, tuple[int, dict]] = {}
        self._where: dict[str, str] = {}
        for file in self.files:
            start, header = read_safetensors_header(self.root / file)
            self._headers[file] = (start, header)
            for tensor in header:
                if tensor != "__metadata__":
                    self._where[tensor] = file

    def path(self, file: str) -> Path:
        return self.root / file

    def _entry(self, tensor: str) -> tuple[str, int, dict]:
        file = self._where[tensor]
        start, header = self._headers[file]
        return file, start, header[tensor]

    def n_experts(self, layer: int) -> int:
        if self.layout == "per_expert":
            count = 0
            while f"model.layers.{layer}.mlp.experts.{count}.gate_proj.weight" in self._where:
                count += 1
            return count
        if self.layout == "fused_granite":
            return int(self._entry(f"model.layers.{layer}.block_sparse_moe.input_linear.weight")[2]["shape"][0])
        raise ValueError(f"unknown layout {self.layout!r}")

    def matrix(self, layer: int, expert: int, kind: str) -> Matrix:
        if self.layout == "per_expert":
            tensor = f"model.layers.{layer}.mlp.experts.{expert}.{kind}_proj.weight"
            file, start, entry = self._entry(tensor)
            rows, cols = entry["shape"]
            begin, end = entry["data_offsets"]
            if end - begin != rows * cols * 2:
                raise ValueError(f"{tensor}: inconsistent size")
            return Matrix(self.name, layer, expert, kind, tensor, file, start + begin, rows, cols, entry["dtype"])
        if self.layout == "fused_granite":
            fused = "input_linear" if kind in ("gate", "up") else "output_linear"
            tensor = f"model.layers.{layer}.block_sparse_moe.{fused}.weight"
            file, start, entry = self._entry(tensor)
            n, rows, cols = entry["shape"]
            begin = entry["data_offsets"][0] + expert * rows * cols * 2
            if kind == "down":
                return Matrix(self.name, layer, expert, kind, tensor, file, start + begin, rows, cols, entry["dtype"],
                              (expert, 0, rows))
            half = rows // 2
            first = 0 if kind == "gate" else half
            return Matrix(self.name, layer, expert, kind, tensor, file, start + begin + first * cols * 2, half, cols,
                          entry["dtype"], (expert, first, first + half))
        raise ValueError(f"unknown layout {self.layout!r}")

    def sample(self, seed: int) -> list[tuple[int, int]]:
        pairs = []
        for layer in self.spec["layers"]:
            for expert in sample_experts(seed, self.name, layer, self.n_experts(layer), self.spec["experts_per_layer"]):
                pairs.append((layer, expert))
        return pairs

    # The two read paths

    def read_patterns(self, m: Matrix) -> np.ndarray:
        if m.dtype != "BF16":
            raise ValueError(f"{m.key}: {m.dtype}, expected BF16")
        out = np.empty(m.rows * m.cols, dtype=np.uint16)
        view = memoryview(out.view(np.uint8))
        with open(self.path(m.file), "rb", buffering=0) as handle:
            handle.seek(m.offset)
            done = 0
            while done < m.nbytes:
                count = handle.readinto(view[done:])
                if not count:
                    raise EOFError(f"{m.key}: short read")
                done += count
        return out.reshape(m.rows, m.cols)

    def read_reference(self, m: Matrix) -> np.ndarray:
        from safetensors import safe_open

        with safe_open(str(self.path(m.file)), framework="pt", device="cpu") as handle:
            if m.index:
                expert, first, stop = m.index
                t = handle.get_slice(m.tensor)[expert : expert + 1, first:stop][0]
            else:
                t = handle.get_tensor(m.tensor)
        if t.dtype != torch.bfloat16:
            raise ValueError(f"{m.key}: {t.dtype}")
        return t.contiguous().view(torch.int16).numpy().view(np.uint16).reshape(m.rows, m.cols)


def verify_files(checkpoint: Checkpoint, files: list[str]) -> dict[str, dict]:
    """Each file's sha256 by a full direct read, against the publisher's (the Hub's LFS id; None when unavailable)."""
    from awpmi.models.checkpoint import checkpoint_sources
    from awpmi.storage.pack import sha256_file_direct

    sources = checkpoint_sources(checkpoint.repository, checkpoint.revision, declared_sha256=True)
    result = {}
    for file in files:
        declared = sources[file].sha256 if file in sources else None
        actual = sha256_file_direct(checkpoint.path(file))
        result[file] = {"declared": declared, "actual": actual, "equal": declared is not None and declared == actual}
    return result


def load_config(path: Path) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def describe(checkpoint: Checkpoint) -> dict:
    config = json.loads((checkpoint.root / "config.json").read_text(encoding="utf-8"))
    keep = ("model_type", "hidden_size", "moe_intermediate_size", "intermediate_size", "num_hidden_layers", "n_routed_experts",
            "num_experts", "num_local_experts", "num_experts_per_tok", "first_k_dense_replace", "torch_dtype")
    return {k: config[k] for k in keep if k in config}
