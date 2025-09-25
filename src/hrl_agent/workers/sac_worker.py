"""Wrapper around stable-baselines3 SAC for low-level continuous control."""

from __future__ import annotations

from typing import Any, Mapping

try:  # pragma: no cover - optional dependency
    from stable_baselines3 import SAC
except ImportError:  # pragma: no cover
    SAC = None  # type: ignore

__all__ = ["SACWorker"]


class SACWorker:
    """Produces continuous control actions for throttle, brake, and steering."""

    def __init__(self, *, model: Any | None = None) -> None:
        self._model = model

    def attach_model(self, model: Any) -> None:
        self._model = model

    def load_from_path(self, path: str) -> None:
        if SAC is None:
            raise RuntimeError("stable-baselines3 is required to load SAC models")
        self._model = SAC.load(path)

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("Cannot save before attaching a model")
        self._model.save(path)

    def act(
        self, observation: Mapping[str, Any], *, deterministic: bool = False
    ) -> dict[str, float]:
        if self._model is None:
            # Return neutral control until a model is attached.
            return {"throttle": 0.0, "brake": 0.0, "steering": 0.0}
        action, _ = self._model.predict(observation, deterministic=deterministic)
        throttle, brake, steering = action
        return {"throttle": float(throttle), "brake": float(brake), "steering": float(steering)}
