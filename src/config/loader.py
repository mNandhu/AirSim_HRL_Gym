"""Utilities for loading experiment definitions from disk."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml

from .experiment import ExperimentDefinition

__all__ = ["load_experiment", "compute_config_hash"]


def load_experiment(path: str | Path) -> ExperimentDefinition:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment config missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}
    experiment = ExperimentDefinition.model_validate(data)
    return experiment.model_copy(update={"config_hash": compute_config_hash(experiment)})


def compute_config_hash(experiment: ExperimentDefinition) -> str:
    payload = (
        experiment.model_dump() if hasattr(experiment, "model_dump") else experiment.__dict__.copy()
    )
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
