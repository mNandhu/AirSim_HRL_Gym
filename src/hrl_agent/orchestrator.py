"""Utilities to run episodes by coordinating manager, workers, and environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

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
        self,
        *,
        max_steps: int | None = None,
        deterministic: bool = True,
        metrics_tracker: Optional[Any] = None,
    ) -> EpisodeResult:
        observation = self._env.reset()
        self._coordinator.reset()
        cumulative_reward = 0.0
        steps = 0
        info: dict[str, Any] = {}

        while True:
            completion_flag = self._coordinator.command_completed(observation)
            command, action = self._coordinator.act(observation, deterministic=deterministic)
            self._env.set_command(command, completed=completion_flag)
            next_observation, reward, terminated, truncated, info = self._env.step(action)

            # Log metrics if tracker is provided
            if metrics_tracker is not None:
                # Extract telemetry and reward components from observation
                telemetry = {}
                reward_components = {}

                if hasattr(observation, "telemetry"):
                    telemetry = observation.telemetry
                elif isinstance(observation, dict) and "telemetry" in observation:
                    telemetry = observation["telemetry"]

                if hasattr(observation, "reward_components"):
                    reward_components = observation.reward_components
                elif isinstance(observation, dict) and "reward_components" in observation:
                    reward_components = observation["reward_components"]

                metrics_tracker.log_step(
                    step=steps,
                    reward=reward,
                    action=action,
                    telemetry=telemetry,
                    reward_components=reward_components,
                    command=command,
                )

            if not deterministic:
                self._coordinator.observe_transition(
                    observation,
                    reward,
                    next_observation,
                    done=terminated or truncated,
                )
            observation = next_observation
            cumulative_reward += reward
            steps += 1

            if terminated or truncated or (max_steps is not None and steps >= max_steps):
                break

        return EpisodeResult(cumulative_reward=cumulative_reward, steps=steps, info=info)
