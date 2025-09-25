"""Wrapper around a stable-baselines3 DQN policy for high-level command selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

try:  # pragma: no cover - optional when stable-baselines3 is unavailable
    from stable_baselines3 import DQN
except ImportError:  # pragma: no cover
    DQN = None  # type: ignore

__all__ = ["CommandPolicy", "DQNManager"]


@dataclass(frozen=True)
class CommandPolicy:
    commands: Sequence[str]

    def command_from_index(self, index: int) -> str:
        return self.commands[index]


class DQNManager:
    """High-level manager that chooses discrete commands via a DQN policy."""

    def __init__(
        self,
        policy: CommandPolicy,
        *,
        model: Any | None = None,
    ) -> None:
        self._policy = policy
        self._model = model

    @property
    def policy(self) -> CommandPolicy:
        return self._policy

    @property
    def model(self) -> Any | None:
        return self._model

    def select_command(self, observation: Any, *, deterministic: bool = True) -> str:
        if self._model is None:
            # Default to first command if no model is attached yet.
            return self._policy.command_from_index(0)
        action, _ = self._model.predict(observation, deterministic=deterministic)
        return self._policy.command_from_index(int(action))

    def attach_model(self, model: Any) -> None:
        self._model = model

    def load_from_path(self, path: str) -> None:
        if DQN is None:
            raise RuntimeError("stable-baselines3 is required to load DQN models")
        self._model = DQN.load(path)

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("Cannot save before attaching a model")
        self._model.save(path)

    def set_policy(self, policy: CommandPolicy) -> None:
        object.__setattr__(self, "_policy", policy)
