"""Run metadata: environment, source trees, codec versions, digests that leave timings out."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import zstandard

from awpmi.tracing import canonical_digest, environment_metadata, source_tree_sha256

REPO_ROOT = Path(__file__).resolve().parents[3]
STUDY = REPO_ROOT / "research" / "lossless_moe"
CODEC = STUDY / "codec"
LMC = CODEC / "target" / "release" / ("lmc.exe" if os.name == "nt" else "lmc")
EXCLUDED_FIELDS = ("timings", "timing", "wall_s", "cpu_s", "best_s", "median_s", "encode_s", "throughput",
                   "peak_working_set_bytes", "system", "seconds", "read_s", "decode_s", "total_s", "original_read_s",
                   "compress_s", "decompress_s")


def study_tree_sha256() -> str:
    """Weightsift's source tree (src, benchmarks, configs), this study's Python, Rust and configuration (tests and
    build output excluded), and the Phase 5C transforms it reuses."""
    digest = hashlib.sha256(source_tree_sha256(REPO_ROOT).encode("ascii"))
    files = [p for p in STUDY.rglob("*") if p.is_file() and p.suffix in (".py", ".rs", ".toml", ".yaml", ".json")]
    files = [p for p in files if not {"tests", "target", "third_party", "__pycache__"} & set(p.relative_to(STUDY).parts)]
    files.append(REPO_ROOT / "research" / "expert_deltas" / "expert_deltas" / "bits.py")
    for path in sorted(files):
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def _rustc() -> str | None:
    try:
        return subprocess.run(["rustc", "--version"], capture_output=True, text=True, cwd=CODEC).stdout.strip() or None
    except OSError:
        return None


def environment(models: dict | None = None) -> dict:
    env = environment_metadata(REPO_ROOT, models or {}, {})
    env["codecs"] = {
        "zstandard": zstandard.__version__,
        "libzstd": ".".join(map(str, zstandard.ZSTD_VERSION)),
        "python_lzma_bz2": f"CPython {platform.python_version()} standard library",
        "rustc": _rustc(),
        "colibri": json.loads((STUDY / "upstream.json").read_text(encoding="utf-8"))["upstreams"][0]["commit"],
    }
    try:
        from awpmi.streaming import nvcomp

        env["codecs"]["nvcomp"] = nvcomp.version() if nvcomp.library() else None
    except Exception as error:  # noqa: BLE001 - recorded, not fatal
        env["codecs"]["nvcomp"] = f"unavailable: {error}"
    env["study_tree_sha256"] = study_tree_sha256()
    env["python_hash_seed"] = os.environ.get("PYTHONHASHSEED")
    env["cpu_count"] = os.cpu_count()
    env["processor"] = platform.processor()
    env["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    env["argv"] = sys.argv
    return env


def strip(record, excluded=EXCLUDED_FIELDS):
    if isinstance(record, dict):
        return {k: strip(v, excluded) for k, v in record.items() if k not in excluded}
    if isinstance(record, list):
        return [strip(v, excluded) for v in record]
    return record


def digest(record) -> str:
    """Digest of a record (a dict) or of a list of records, timings and other non-deterministic fields excluded."""
    stripped = strip(record)
    return canonical_digest(stripped if isinstance(stripped, list) else [stripped])


def write_json(path: Path, record) -> None:
    Path(path).write_text(json.dumps(record, indent=1), encoding="utf-8")


def sha256_bytes(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()
