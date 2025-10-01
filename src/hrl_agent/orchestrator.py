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

    def _extract_waypoints_from_env(self, env: AirSimEnv) -> list[tuple[float, float]]:
        """Extract waypoints from environment experiment config for trajectory plotting."""
        waypoints = []
        try:
            experiment = env.experiment
            if hasattr(experiment, "waypoints") and experiment.waypoints:
                # Convert waypoint poses to (x, y) tuples
                waypoints = [(float(wp.x), float(wp.y)) for wp in experiment.waypoints]
            elif hasattr(experiment, "goal_pose") and experiment.goal_pose:
                # Legacy: create path from start to goal
                start = experiment.start_pose
                goal = experiment.goal_pose
                waypoints = [(float(start.x), float(start.y)), (float(goal.x), float(goal.y))]
        except Exception:
            # If waypoint extraction fails, return empty list
            pass
        return waypoints

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
                # CRITICAL FIX: Use next_observation for reward_components and telemetry
                # The reward returned from step() corresponds to the transition TO next_observation,
                # not from the previous observation. Logging old observation components with new
                # reward creates mismatches (e.g., collision penalty in reward but not in components).
                telemetry = {}
                reward_components = {}

                # Use NEXT observation (current state after step) for telemetry
                if hasattr(next_observation, "telemetry"):
                    telemetry = next_observation.telemetry
                elif isinstance(next_observation, dict) and "telemetry" in next_observation:
                    telemetry = next_observation["telemetry"]

                # Use NEXT observation (current state after step) for reward components
                if hasattr(next_observation, "reward_components"):
                    reward_components = next_observation.reward_components
                elif isinstance(next_observation, dict) and "reward_components" in next_observation:
                    reward_components = next_observation["reward_components"]

                # Keep previous telemetry for backwards compatibility in metrics tracker
                prev_telemetry = None
                if hasattr(observation, "telemetry"):
                    prev_telemetry = observation.telemetry
                elif isinstance(observation, dict) and "telemetry" in observation:
                    prev_telemetry = observation["telemetry"]

                metrics_tracker.log_step(
                    step=steps,
                    reward=reward,
                    action=action,
                    telemetry=telemetry,  # Now uses NEXT observation (matches reward)
                    reward_components=reward_components,  # Now uses NEXT observation (matches reward)
                    command=command,
                    next_telemetry=prev_telemetry,  # Optional: previous state for delta calculations
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
                waypoints = self._extract_waypoints_from_env(env)
                metrics_tracker.start_episode(episode=index, waypoints=waypoints)

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

                    next_telemetry = None
                    if hasattr(next_observation, "telemetry"):
                        next_telemetry = next_observation.telemetry
                    elif isinstance(next_observation, dict) and "telemetry" in next_observation:
                        next_telemetry = next_observation["telemetry"]

                    metrics_tracker.log_step(
                        step=step_counts[env_id] - 1,
                        reward=reward,
                        action=action,
                        telemetry=telemetry,
                        reward_components=reward_components,
                        command=command,
                        next_telemetry=next_telemetry,
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
