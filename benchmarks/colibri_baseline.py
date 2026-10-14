"""The Colibri baseline of a flagship (Phase 7A, decision 0014; `configs/phase7a-baseline.yaml`, `awpmi.baseline`).

    python -m uv run python benchmarks/colibri_baseline.py --dry-run [--colibri DIR] [--snapshot DIR] [--output PLAN.json]
    python -m uv run python benchmarks/colibri_baseline.py --parse ENGINE.log

--dry-run   the plan (every process: configuration, cold or warm, repeat, command, environment), the pinned revisions,
            the prompts' sha256, the machine as detected, and what blocks the run here. Exits 2 when blocked.
--parse     the metrics in a log of Inkling's engine at the pinned commit, as JSON.

Executing the plan is the next milestone's (it needs the checkpoint, whose download needs the user's approval, and an
engine built with gcc): nothing here pretends to have run it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from awpmi.baseline import blockers, load_spec, parse_output, plan, prompts, prompts_sha256
from awpmi.hardware import detect
from awpmi.tracing import native_tree_sha256, source_tree_sha256

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "phase7a-baseline.yaml"))
    parser.add_argument("--colibri", default=None, help="a full Colibri checkout at the pinned commit, built")
    parser.add_argument("--snapshot", default=None, help="the artifact's snapshot directory")
    parser.add_argument("--output", default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--parse", default=None)
    args = parser.parse_args()
    if args.parse:
        print(json.dumps(parse_output(Path(args.parse).read_text(encoding="utf-8", errors="replace")), indent=2))
        return 0
    spec = load_spec(args.config)
    colibri = Path(args.colibri) if args.colibri else None
    snapshot = Path(args.snapshot) if args.snapshot else None
    machine = detect(snapshot or ROOT)
    steps = plan(
        spec,
        colibri=colibri or Path("<colibri>"),
        snapshot=snapshot or Path("<snapshot>"),
        prompt_file=ROOT / spec["workload"]["prompts"],
        physical_cores=machine["logical_cpus"] or 1,
    )
    record = {
        "engine": spec["engine"],
        "model": spec["model"],
        "prompts": {"count": len(prompts(spec, ROOT)), "sha256": prompts_sha256(spec, ROOT)},
        "weightsift": {"source_tree_sha256": source_tree_sha256(ROOT), "native_tree_sha256": native_tree_sha256(ROOT)},
        "machine": machine,
        "steps": [
            {"configuration": s.configuration, "state": s.state, "repeat": s.repeat, "command": list(s.command),
             "environment": dict(s.environment)}
            for s in steps
        ],
        "blockers": blockers(spec, colibri=colibri, snapshot=snapshot),
        "executed": False,
    }
    text = json.dumps(record, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 2 if record["blockers"] else 0


if __name__ == "__main__":
    sys.exit(main())
