"""Utilities to run episodes by coordinating manager, workers, and environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

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

    def run_parallel(
        self,
        envs: Mapping[str, AirSimEnv],
        *,
        max_steps: int | None = None,
        deterministic: bool = True,
        metrics_tracker: Optional[Any] = None,
    ) -> dict[str, EpisodeResult]:
        if not envs:
            return {}

        self._coordinator.reset()
        observations: dict[str, Any] = {}
        results: dict[str, EpisodeResult] = {}
        step_counts: dict[str, int] = {}
        active_envs = dict(envs)

        for index, (env_id, env) in enumerate(active_envs.items()):
            observation = env.reset()
            self._coordinator.reset_env(env_id)
            observations[env_id] = observation
            step_counts[env_id] = 0
            results[env_id] = EpisodeResult(cumulative_reward=0.0, steps=0, info={})
            if metrics_tracker is not None and hasattr(metrics_tracker, "start_episode"):
                metrics_tracker.start_episode(index)

        while active_envs:
            commands: dict[str, tuple[str, Any]] = {}
            for env_id, observation in observations.items():
                if env_id not in active_envs:
                    continue
                completion = self._coordinator.command_completed_for_env(env_id, observation)
                command, action = self._coordinator.act_for_env(
                    env_id, observation, deterministic=deterministic
                )
                active_envs[env_id].set_command(command, completed=completion)
                commands[env_id] = (command, action)

            finished: list[str] = []
            for env_id, (command, action) in commands.items():
                env = active_envs.get(env_id)
                if env is None:
                    continue

                prev_observation = observations[env_id]
                next_observation, reward, terminated, truncated, info = env.step(action)

                step_counts[env_id] += 1
                episode_result = results[env_id]
                episode_result.cumulative_reward += reward
                episode_result.steps = step_counts[env_id]
                episode_result.info = info

                if metrics_tracker is not None and hasattr(metrics_tracker, "log_step"):
                    telemetry = {}
                    reward_components = {}
                    if hasattr(prev_observation, "telemetry"):
                        telemetry = prev_observation.telemetry  # type: ignore[attr-defined]
                    elif isinstance(prev_observation, dict) and "telemetry" in prev_observation:
                        telemetry = prev_observation["telemetry"]

                    if hasattr(prev_observation, "reward_components"):
                        reward_components = prev_observation.reward_components  # type: ignore[attr-defined]
                    elif (
                        isinstance(prev_observation, dict)
                        and "reward_components" in prev_observation
                    ):
                        reward_components = prev_observation["reward_components"]

                    metrics_tracker.log_step(
                        step=step_counts[env_id] - 1,
                        reward=reward,
                        action=action,
                        telemetry=telemetry,
                        reward_components=reward_components,
                        command=command,
                    )

                if not deterministic:
                    self._coordinator.observe_transition_for_env(
                        env_id,
                        prev_observation,
                        reward,
                        next_observation,
                        done=terminated or truncated,
                    )

                observations[env_id] = next_observation

                env_done = terminated or truncated
                if max_steps is not None and step_counts[env_id] >= max_steps:
                    env_done = True

                if env_done:
                    finished.append(env_id)

            for env_id in finished:
                active_envs.pop(env_id, None)
                self._coordinator.reset_env(env_id)

        return results
