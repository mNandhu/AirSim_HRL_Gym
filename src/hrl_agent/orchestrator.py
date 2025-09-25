"""Utilities to run episodes by coordinating manager, workers, and environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from airsim_env.env import AirSimEnv

from .coordination import CommandCoordinator

__all__ = ["EpisodeResult", "HRLOrchestrator"]


@dataclass
class EpisodeResult:
    cumulative_reward: float
    steps: int
    info: dict[str, Any]


class HRLOrchestrator:
    def __init__(self, env: AirSimEnv, coordinator: CommandCoordinator) -> None:
        self._env = env
        self._coordinator = coordinator

    def run_episode(
        self, *, max_steps: int | None = None, deterministic: bool = True
    ) -> EpisodeResult:
        observation = self._env.reset()
        cumulative_reward = 0.0
        steps = 0
        info: dict[str, Any] = {}

        while True:
            command, action = self._coordinator.act(observation, deterministic=deterministic)
            self._env.set_command(command, completed=False)
            observation, reward, terminated, truncated, info = self._env.step(action)
            cumulative_reward += reward
            steps += 1

            if terminated or truncated or (max_steps is not None and steps >= max_steps):
                break

        return EpisodeResult(cumulative_reward=cumulative_reward, steps=steps, info=info)
