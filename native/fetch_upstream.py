"""Fetch the upstream sources Weightsift builds against, exactly as pinned in native/upstream.json (decision 0014).

For each upstream (or the ones named): a shallow, sparse checkout of the pinned commit holding only the pinned files,
into native/third_party/<name>/ (gitignored), with line endings as the repository stores them; then every file's sha256
is compared with the pin, and a marker (.weightsift-upstream.json) records what was verified. Builds read the marker and
re-hash the files before using them. Nothing fetched is ever committed to Weightsift.

    python native/fetch_upstream.py            # fetch (or confirm) every upstream
    python native/fetch_upstream.py --check    # verify what is there, no network

Standard library only; needs git on PATH.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

NATIVE = Path(__file__).resolve().parent
PINS = NATIVE / "upstream.json"
THIRD_PARTY = NATIVE / "third_party"
MARKER = ".weightsift-upstream.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, cwd: Path) -> str:
    # Line endings as stored in the repository, whatever the machine's git configuration.
    command = ["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf", "-c", "advice.detachedHead=false", *args]
    done = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {cwd}: {done.stderr.strip()}")
    return done.stdout.strip()


def remove_tree(path: Path) -> None:
    """Remove a checkout, including the read-only object files git leaves on Windows."""

    def writable_then_retry(function, target, _error):
        os.chmod(target, stat.S_IWRITE)
        function(target)

    shutil.rmtree(path, onexc=writable_then_retry)


def verify(pin: dict, dest: Path) -> list[str]:
    """What differs from the pin (missing or altered files, another commit); empty when everything matches."""
    problems = []
    marker = dest / MARKER
    if not marker.is_file():
        return [f"{dest} has no verification marker"]
    recorded = json.loads(marker.read_text(encoding="utf-8"))
    if recorded.get("commit") != pin["commit"] or recorded.get("url") != pin["url"]:
        problems.append(f"{dest} holds {recorded.get('commit')}, the pin is {pin['commit']}")
    for name, expected in pin["files"].items():
        path = dest / name
        if not path.is_file():
            problems.append(f"{name} is missing")
        elif sha256(path) != expected:
            problems.append(f"{name} differs from the pin")
    return problems


def fetch(pin: dict) -> None:
    dest = THIRD_PARTY / pin["name"]
    if dest.exists() and not verify(pin, dest):
        print(f"{pin['name']}: up to date ({pin['commit']})")
        return
    if dest.exists():
        # Only a directory this script made (its marker, or a git checkout of the pinned URL) is replaced.
        ours = (dest / MARKER).is_file()
        if not ours and (dest / ".git").is_dir():
            try:
                ours = git("remote", "get-url", "origin", cwd=dest) == pin["url"]
            except RuntimeError:
                ours = False
        if not ours:
            raise SystemExit(f"{dest} exists and was not made by this script: move it away first")
        remove_tree(dest)
    dest.mkdir(parents=True)
    git("init", "-q", cwd=dest)
    git("remote", "add", "origin", pin["url"], cwd=dest)
    git("config", "core.autocrlf", "false", cwd=dest)
    git("sparse-checkout", "set", "--no-cone", *[f"/{name}" for name in pin["files"]], cwd=dest)
    git("fetch", "-q", "--depth", "1", "origin", pin["commit"], cwd=dest)
    git("checkout", "-q", "FETCH_HEAD", cwd=dest)
    head = git("rev-parse", "HEAD", cwd=dest)
    if head != pin["commit"]:
        raise SystemExit(f"{pin['name']}: fetched {head}, the pin is {pin['commit']}")
    marker = {
        "name": pin["name"],
        "url": pin["url"],
        "commit": pin["commit"],
        "license": pin["license"],
        "files": pin["files"],
    }
    (dest / MARKER).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    problems = verify(pin, dest)
    if problems:
        (dest / MARKER).unlink()
        raise SystemExit(f"{pin['name']}: the checkout does not match the pin: " + "; ".join(problems))
    print(f"{pin['name']}: fetched and verified {pin['commit']} ({len(pin['files'])} files, {pin['license']})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="upstreams to fetch (default: all)")
    parser.add_argument("--check", action="store_true", help="verify what is there; fetch nothing")
    args = parser.parse_args()
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    if pins.get("format") != "weightsift-upstream-pins" or pins.get("version") != 1:
        raise SystemExit(f"{PINS} is not a weightsift-upstream-pins v1 file")
    selected = [p for p in pins["upstreams"] if not args.names or p["name"] in args.names]
    unknown = set(args.names) - {p["name"] for p in pins["upstreams"]}
    if unknown:
        raise SystemExit(f"no pin for {sorted(unknown)}")
    failed = False
    for pin in selected:
        if args.check:
            problems = verify(pin, THIRD_PARTY / pin["name"])
            print(f"{pin['name']}: " + ("verified" if not problems else "; ".join(problems)))
            failed |= bool(problems)
        else:
            fetch(pin)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
