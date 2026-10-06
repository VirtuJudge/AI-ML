"""Enforce the repository's formatting and no-regression quality baseline."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "scripts" / "quality-baseline.json"
MYPY_ERROR = re.compile(
    r"^(?P<path>.+?\.py):\d+(?::\d+)?: error: (?P<message>.*?)(?:\s+\[(?P<code>[\w-]+)\])?$"
)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)


def _relative_path(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (ROOT / candidate).resolve()
    try:
        return candidate.relative_to(ROOT).as_posix()
    except ValueError:
        return candidate.name


def _ruff_diagnostics() -> Counter[str]:
    result = _run(["ruff", "check", "app", "scripts", "tests", "--output-format", "json"])
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr or "Ruff failed to run.")
    diagnostics = json.loads(result.stdout or "[]")
    return Counter(
        f"{_relative_path(item['filename'])}|{item['code']}|{item['message']}"
        for item in diagnostics
    )


def _mypy_diagnostics() -> Counter[str]:
    result = _run(
        ["mypy", "app", "--no-error-summary", "--hide-error-context", "--no-color-output"]
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr or "MyPy failed to run.")

    errors: Counter[str] = Counter()
    for line in result.stdout.splitlines():
        if ": error:" not in line:
            continue
        match = MYPY_ERROR.match(line)
        if match:
            errors[
                f"{_relative_path(match.group('path'))}|{match.group('code') or 'uncoded'}|"
                f"{match.group('message')}"
            ] += 1
        else:
            errors[f"unparsed|{line}"] += 1
    return errors


def _format_check() -> None:
    result = _run(["ruff", "format", "--check", "app", "scripts", "tests"])
    if result.returncode != 0:
        raise RuntimeError(result.stdout + result.stderr)


def _read_baseline() -> dict[str, dict[str, int]]:
    with BASELINE_PATH.open(encoding="utf-8") as baseline_file:
        data: Any = json.load(baseline_file)
    if not isinstance(data, dict):
        raise RuntimeError("Quality baseline must be a JSON object.")
    return data


def _write_baseline(ruff: Counter[str], mypy: Counter[str]) -> None:
    baseline = {
        "ruff": dict(sorted(ruff.items())),
        "mypy": dict(sorted(mypy.items())),
    }
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")


def _new_diagnostics(current: Counter[str], baseline: dict[str, int]) -> list[str]:
    return [
        f"{key} (baseline {baseline.get(key, 0)}, current {count})"
        for key, count in sorted(current.items())
        if count > baseline.get(key, 0)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="replace the checked-in lint/type baseline after an intentional cleanup",
    )
    args = parser.parse_args()

    try:
        _format_check()
        ruff = _ruff_diagnostics()
        mypy = _mypy_diagnostics()
        if args.write_baseline:
            _write_baseline(ruff, mypy)
            print(f"Wrote quality baseline: {len(ruff)} Ruff and {len(mypy)} MyPy diagnostics.")
            return 0

        baseline = _read_baseline()
        new_ruff = _new_diagnostics(ruff, baseline.get("ruff", {}))
        new_mypy = _new_diagnostics(mypy, baseline.get("mypy", {}))
        if new_ruff or new_mypy:
            if new_ruff:
                print("New Ruff diagnostics:", file=sys.stderr)
                print("\n".join(new_ruff), file=sys.stderr)
            if new_mypy:
                print("New MyPy diagnostics:", file=sys.stderr)
                print("\n".join(new_mypy), file=sys.stderr)
            return 1

        print(
            f"Quality check passed: {len(ruff)} Ruff and {len(mypy)} MyPy diagnostics, "
            "with no additions to the baseline."
        )
        return 0
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Quality check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
