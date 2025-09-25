"""Run linting and tests to enforce quality gates locally or in CI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Iterable, Sequence

COMMANDS = [
    ["uv", "run", "ruff", "check"],
    ["uv", "run", "pytest", "--cov=src", "--cov-report=term-missing", "--cov-fail-under=90"],
]


def run_commands(commands: Iterable[Sequence[str]]) -> int:
    for command in commands:
        print(f"→ Running: {' '.join(command)}")
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(f"✗ Command failed with exit code {result.returncode}")
            return result.returncode
        print("✓ Success")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enforce lint/test quality gates")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest run")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    commands = COMMANDS.copy()
    if args.skip_tests:
        commands = commands[:-1]
    return run_commands(commands)


if __name__ == "__main__":  # pragma: no cover - manual script
    sys.exit(main())
