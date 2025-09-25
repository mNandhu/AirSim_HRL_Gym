"""Utility helpers for computing deterministic hashes of AirSim settings files."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
import hashlib
import json
from typing import Any

JsonLike = Mapping[str, Any] | Sequence[Any] | str | int | float | bool | None


def _normalize(value: JsonLike) -> JsonLike:
    """Return a structure with stable ordering for hashing."""
    if isinstance(value, Mapping):
        return {key: _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize(item) for item in value]
    return value


def _load_settings(settings: Path | str | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(settings, Mapping):
        return settings
    path = Path(settings)
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, MutableMapping):
        raise TypeError("Settings JSON must be an object at the top level.")
    return data


def compute_settings_hash(settings: Path | str | Mapping[str, Any]) -> str:
    """Compute a SHA256 hash for the provided settings mapping or JSON file."""
    data = _load_settings(settings)
    normalized = _normalize(data)
    serialized = json.dumps(normalized, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


__all__ = ["compute_settings_hash"]
