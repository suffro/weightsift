"""The Colibri baseline's specification and records (Phase 7A, decision 0014; `configs/phase7a-baseline.yaml`).

`load_spec` checks the specification (pinned engine commit, pinned checkpoint and artifact revisions, configurations,
workload, metrics, rules); `plan` turns it into the exact commands, environments and states of a run; `blockers` says
what keeps the run from starting on this machine (no engine binary, no snapshot, a prompt file whose hash changed);
`parse_output` reads the metrics Inkling's engine prints at the pinned commit (its `[prefill ...]`, `TUNE decode`,
`[phases]`, `[cache]` and `PEAK RSS` lines). Executing the plan, with the OS's I/O counters around each process, is the
next milestone's: it needs the checkpoint.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

FORMAT = "weightsift-baseline-benchmark"
VERSION = 1
_COMMIT = re.compile(r"^[0-9a-f]{40}$")

REQUIRED_METRICS = (
    "decode_tokens_per_second",
    "prefill_seconds",
    "time_to_first_token_seconds",
    "ssd_bytes_per_token",
    "host_to_device_bytes_per_token",
    "expert_cache_hit_ratio",
    "peak_ram_bytes",
    "peak_vram_bytes",
    "cpu_utilization",
    "gpu_utilization",
    "expert_fill_seconds",
    "output_tokens",
)


class SpecError(ValueError):
    """A baseline specification that is malformed or incomplete."""


@dataclass(frozen=True)
class Step:
    configuration: str
    state: str
    repeat: int
    command: tuple[str, ...]
    environment: Mapping[str, str]


def load_spec(path: str | Path) -> dict[str, Any]:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if spec.get("format") != FORMAT or spec.get("version") != VERSION:
        raise SpecError(f"not a {FORMAT} v{VERSION} document")
    for section in ("engine", "model", "configurations", "workload", "metrics", "rules"):
        if not spec.get(section):
            raise SpecError(f"{section} is required")
    if not _COMMIT.match(str(spec["engine"].get("commit", ""))):
        raise SpecError("the engine must be pinned to a full commit")
    for key in ("original", "artifact"):
        entry = spec["model"].get(key) or {}
        if not _COMMIT.match(str(entry.get("revision", ""))) or not entry.get("repository"):
            raise SpecError(f"model.{key} must name a repository pinned to a full revision")
    missing = set(REQUIRED_METRICS) - set(spec["metrics"])
    if missing:
        raise SpecError(f"metrics missing: {sorted(missing)}")
    names = [c.get("name") for c in spec["configurations"]]
    if len(set(names)) != len(names) or not all(names):
        raise SpecError("configurations need distinct names")
    if spec["workload"].get("repeats", 0) < 2:
        raise SpecError("a baseline runs at least twice")
    if set(spec["workload"].get("states", {})) != {"cold", "warm"}:
        raise SpecError("the workload states are cold and warm")
    return spec


def prompts(spec: Mapping[str, Any], root: Path) -> list[str]:
    path = root / spec["workload"]["prompts"]
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def prompts_sha256(spec: Mapping[str, Any], root: Path) -> str:
    return hashlib.sha256((root / spec["workload"]["prompts"]).read_bytes()).hexdigest()


def plan(spec: Mapping[str, Any], *, colibri: Path, snapshot: Path, prompt_file: Path, physical_cores: int) -> list[Step]:
    """Every process of the run, in order: per configuration, per repeat, a cold then a warm process."""
    binary = colibri / spec["engine"]["binary"]
    steps = []
    for configuration in spec["configurations"]:
        environment = {
            key: str(value).replace("{physical_cores}", str(physical_cores))
            for key, value in configuration.get("env", {}).items()
        }
        environment["SNAP"] = str(snapshot)
        for repeat in range(spec["workload"]["repeats"]):
            for state in ("cold", "warm"):
                command = [str(binary), *configuration.get("args", []), "-f", str(prompt_file),
                           "-n", str(spec["workload"]["decode_tokens"])]
                if spec["workload"].get("chat_template"):
                    command.insert(1, "--chat")
                steps.append(Step(configuration["name"], state, repeat, tuple(command), dict(environment)))
    return steps


def blockers(spec: Mapping[str, Any], *, colibri: Path | None, snapshot: Path | None) -> list[str]:
    """What keeps the run from starting here (empty: it can start)."""
    found = []
    if colibri is None or not (colibri / spec["engine"]["binary"]).exists():
        found.append(
            f"no Colibri engine binary: build {spec['engine']['target']} at {spec['engine']['commit']} with "
            f"`{spec['engine']['build']}` ({spec['engine']['toolchain']})"
        )
    if snapshot is None or not snapshot.exists():
        artifact = spec["model"]["artifact"]
        found.append(
            f"no snapshot of {artifact['repository']}@{artifact['revision']} ({artifact['bytes'] / 1e9:.0f} GB): "
            "downloading it needs the user's approval"
        )
    return found


_PREFILL = re.compile(
    r"\[prefill (?P<prefill>[\d.]+)s \| (?P<tokens>\d+) tokens in (?P<seconds>[\d.]+)s = (?P<rate>[\d.]+) tok/s \| "
    r"RSS (?P<rss>[\d.]+) GB\]"
)
_TUNE = re.compile(r"^TUNE decode: (?P<tokens>\d+) tokens in (?P<seconds>[\d.]+)s$", re.MULTILINE)
_PHASES = re.compile(
    r"\[phases\] fill (?P<fill>[\d.]+)s \| expert-mm (?P<expert_mm>[\d.]+)s \| shared (?P<shared>[\d.]+)s \| "
    r"attn (?P<attn>[\d.]+)s \| other (?P<other>-?[\d.]+)s"
)
_CACHE = re.compile(r"\[cache\] hit (?P<percent>[\d.]+)% \((?P<hits>\d+) hit / (?P<loads>\d+) load\)")
_PEAK = re.compile(r"PEAK RSS: (?P<rss>[\d.]+) GB \| expert cache hit (?P<percent>[\d.]+)% \| (?P<rate>[\d.]+) tok/s")


def parse_output(text: str) -> dict[str, Any]:
    """The metrics Inkling's engine prints (pinned commit), per prompt where it prints them per prompt."""
    prompts_ = [
        {
            "prefill_seconds": float(m["prefill"]),
            "generated_tokens": int(m["tokens"]),
            "generation_seconds": float(m["seconds"]),
            "rss_gb": float(m["rss"]),
        }
        for m in _PREFILL.finditer(text)
    ]
    for record, m in zip(prompts_, _TUNE.finditer(text)):
        tokens, seconds = int(m["tokens"]), float(m["seconds"])
        record["decode_tokens"] = tokens
        record["decode_seconds"] = seconds
        record["decode_tokens_per_second"] = tokens / seconds if seconds > 0 else None
    for record, m in zip(prompts_, _PHASES.finditer(text)):
        record["phases_seconds"] = {key: float(value) for key, value in m.groupdict().items()}
    out: dict[str, Any] = {"prompts": prompts_}
    cache = _CACHE.search(text)
    if cache:
        hits, loads = int(cache["hits"]), int(cache["loads"])
        out["expert_cache"] = {"hits": hits, "loads": loads, "hit_ratio": hits / (hits + loads) if hits + loads else None}
    peak = _PEAK.search(text)
    if peak:
        out["peak_rss_gb"] = float(peak["rss"])
    return out
