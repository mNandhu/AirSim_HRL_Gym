"""Helpers for creating and managing experiment artifact directories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

__all__ = ["ArtifactPaths", "ArtifactManager"]


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    run_dir: Path
    logs_dir: Path


class ArtifactManager:
    """Manage per-run artifact directories and structured logging."""

    def __init__(self, root: Path | str = "artifacts") -> None:
        self._root = Path(root)
        self._current_run: ArtifactPaths | None = None

    @property
    def root(self) -> Path:
        return self._root

    @property
    def current_run(self) -> ArtifactPaths | None:
        return self._current_run

    def start_run(self, experiment_id: str) -> ArtifactPaths:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        run_dir = self._root / f"{timestamp}_{experiment_id}"
        logs_dir = run_dir / "logs"
        run_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        self._current_run = ArtifactPaths(root=self._root, run_dir=run_dir, logs_dir=logs_dir)
        return self._current_run

    def write_json(self, name: str, payload: Mapping[str, Any]) -> Path:
        if self._current_run is None:
            raise RuntimeError("start_run must be called before writing artifacts")
        path = self._current_run.run_dir / name
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return path

    def append_jsonl(self, name: str, payload: Mapping[str, Any]) -> Path:
        if self._current_run is None:
            raise RuntimeError("start_run must be called before writing artifacts")
        path = self._current_run.logs_dir / name
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        return path
