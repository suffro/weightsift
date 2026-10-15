"""Fetch the Colibri files this study uses, exactly as pinned in upstream.json, into third_party/ (gitignored).

It is native/fetch_upstream.py run against this directory's pins: the same sparse, shallow checkout of the pinned
commit, the same sha256 verification and marker. Nothing fetched is committed.

    python research/lossless_moe/fetch_upstream.py            # fetch (or confirm)
    python research/lossless_moe/fetch_upstream.py --check    # verify, no network
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NATIVE_FETCH = HERE.parents[1] / "native" / "fetch_upstream.py"


def _native():
    spec = importlib.util.spec_from_file_location("weightsift_fetch_upstream", NATIVE_FETCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PINS = HERE / "upstream.json"
    module.THIRD_PARTY = HERE / "third_party"
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    native = _native()
    pins = json.loads(native.PINS.read_text(encoding="utf-8"))
    failed = False
    for pin in pins["upstreams"]:
        if args.check:
            problems = native.verify(pin, native.THIRD_PARTY / pin["name"])
            print(f"{pin['name']}: " + ("verified" if not problems else "; ".join(problems)))
            failed |= bool(problems)
        else:
            native.fetch(pin)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
