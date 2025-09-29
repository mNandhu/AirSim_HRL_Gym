"""Coordinator that bridges the DQN manager with SAC workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from airsim_env.observation import ObservationPacket

from .manager.dqn_manager import CommandPolicy, DQNManager
from .workers.sac_worker import SACWorker

__all__ = ["CommandCoordinator", "CoordinatorState"]


DEFAULT_ENVIRONMENT_ID = "__singleton_env__"


@dataclass
class CoordinatorState:
    command: str | None = None
    last_action: Mapping[str, float] | None = None
    manager_features: np.ndarray | None = None
    worker_features: np.ndarray | None = None
    command_start_distance: float | None = None
    command_start_heading: float | None = None
    # Completion management
    completion_awarded: bool = False
    stop_below_threshold_steps: int = 0


class CommandCoordinator:
    """Route high-level commands to the appropriate low-level worker."""

    def __init__(
        self,
        manager: DQNManager,
        workers: Mapping[str, SACWorker],
    ) -> None:
        if not workers:
            raise ValueError("At least one worker must be provided")
        self._manager = manager
        self._workers = dict(workers)
        self._state = CoordinatorState()
        self._per_env_state: dict[str, CoordinatorState] = {}
        self._schedule = list(self._workers.keys())
        self._schedule_index = 0

    @property
    def state(self) -> CoordinatorState:
        return self._per_env_state.get(DEFAULT_ENVIRONMENT_ID, self._state)

    def act(
        self, observation: Any, *, deterministic: bool = True
    ) -> tuple[str, Mapping[str, float]]:
        command, action = self.act_for_env(
            DEFAULT_ENVIRONMENT_ID, observation, deterministic=deterministic
        )
        return command, action

    def act_batch(
        self, observations: Mapping[str, Any], *, deterministic: bool = True
    ) -> dict[str, tuple[str, Mapping[str, float]]]:
        results: dict[str, tuple[str, Mapping[str, float]]] = {}
        for env_id, observation in observations.items():
            results[env_id] = self.act_for_env(env_id, observation, deterministic=deterministic)
        return results

    def act_for_env(
        self, env_id: str, observation: Any, *, deterministic: bool = True
    ) -> tuple[str, Mapping[str, float]]:
        features = self._vectorize_observation(observation)

        command_input = self._adapt_features(features, self._manager_input_dim())
        command = self._manager.select_command(command_input, deterministic=deterministic)
        worker = self._workers.get(command)
        if worker is None:
            worker = next(iter(self._workers.values()))

        worker_input = self._adapt_features(features, self._worker_input_dim(worker))
        action = worker.act(worker_input, deterministic=deterministic)
        telemetry = self._extract_telemetry(observation)
        start_distance = telemetry.get("distance_to_goal")
        heading = telemetry.get("heading_deg")

        prev = self._per_env_state.get(env_id)
        # If command changes, re-arm completion and establish new baselines
        if prev is None or prev.command != command:
            state = CoordinatorState(
                command=command,
                last_action=action,
                manager_features=command_input,
                worker_features=worker_input,
                command_start_distance=float(start_distance)
                if start_distance is not None
                else None,
                command_start_heading=float(heading) if heading is not None else None,
                completion_awarded=False,
                stop_below_threshold_steps=0,
            )
        else:
            # Preserve baselines and completion state across steps for same command
            state = CoordinatorState(
                command=command,
                last_action=action,
                manager_features=command_input,
                worker_features=worker_input,
                command_start_distance=prev.command_start_distance,
                command_start_heading=prev.command_start_heading,
                completion_awarded=prev.completion_awarded,
                stop_below_threshold_steps=prev.stop_below_threshold_steps,
            )
        self._per_env_state[env_id] = state
        if env_id == DEFAULT_ENVIRONMENT_ID:
            self._state = state
        return command, action

    def update_command_policy(self, policy: CommandPolicy) -> None:
        self._manager.set_policy(policy)

    def register_worker(self, command: str, worker: SACWorker) -> None:
        self._workers[command] = worker
        self._schedule = list(self._workers.keys())
        self._schedule_index = 0

    def act_sequential(
        self, observation: Any, *, deterministic: bool = True
    ) -> tuple[str, Mapping[str, float]]:
        features = self._vectorize_observation(observation)
        if not self._schedule:
            self._schedule = list(self._workers.keys())
        command = self._schedule[self._schedule_index % len(self._schedule)]
        self._schedule_index += 1
        worker = self._workers[command]
        worker_input = self._adapt_features(features, self._worker_input_dim(worker))
        action = worker.act(worker_input, deterministic=deterministic)
        self._state = CoordinatorState(command=command, last_action=action)
        return command, action

    def command_completed(self, observation: Any) -> bool:
        return self.command_completed_for_env(DEFAULT_ENVIRONMENT_ID, observation)

    def command_completed_for_env(self, env_id: str, observation: Any) -> bool:
        state = self._per_env_state.get(env_id)
        if state is None or state.command is None or state.manager_features is None:
            return False
        telemetry = self._extract_telemetry(observation)
        if not telemetry:
            return False

        command = state.command
        speed = float(telemetry.get("speed_mps", 0.0))
        distance = float(telemetry.get("distance_to_goal", 0.0))
        heading = float(telemetry.get("heading_deg", 0.0))
        start_distance = state.command_start_distance
        start_heading = state.command_start_heading

        # Hysteresis for STOP: require K consecutive steps below threshold
        completed = False
        if command == "STOP":
            threshold = 0.2
            hysteresis_steps = 5
            if speed < threshold:
                state.stop_below_threshold_steps += 1
            else:
                state.stop_below_threshold_steps = 0
            completed = state.stop_below_threshold_steps >= hysteresis_steps
        elif command == "FOLLOW_LANE":
            # Completion relative to the starting distance of this command instance
            if start_distance is not None and distance is not None:
                completed = (start_distance - distance) >= 5.0
        elif command == "TURN_LEFT_AT_INTERSECTION":
            if start_heading is not None and heading is not None:
                completed = self._signed_heading_delta(start_heading, heading) >= 70.0
        elif command == "TURN_RIGHT_AT_INTERSECTION":
            if start_heading is not None and heading is not None:
                completed = self._signed_heading_delta(start_heading, heading) <= -70.0

        # Enforce single completion per command issuance
        if completed and state.completion_awarded:
            return False
        if completed:
            state.completion_awarded = True
        return completed

    def observe_transition(
        self,
        previous_observation: Any,
        reward: float,
        next_observation: Any,
        *,
        done: bool,
    ) -> None:
        self.observe_transition_for_env(
            DEFAULT_ENVIRONMENT_ID,
            previous_observation,
            reward,
            next_observation,
            done=done,
        )

    def observe_transition_for_env(
        self,
        env_id: str,
        previous_observation: Any,
        reward: float,
        next_observation: Any,
        *,
        done: bool,
    ) -> None:
        state = self._per_env_state.get(env_id)
        if state is None or state.command is None or state.manager_features is None:
            return

        next_features = self._vectorize_observation(next_observation)
        action_index = self._manager.policy.index_of(state.command)
        self._manager.process_experience(
            state.manager_features,
            action_index,
            reward,
            next_features,
            done,
        )

        last_action = state.last_action or {}
        worker = self._workers.get(state.command)
        if worker is not None and state.worker_features is not None:
            worker_action = np.array(
                [
                    float(last_action.get("throttle", 0.0)),
                    float(last_action.get("brake", 0.0)),
                    float(last_action.get("steering", 0.0)),
                ],
                dtype=np.float32,
            )
            worker_features = state.worker_features
            next_worker_features = self._adapt_features(
                self._vectorize_observation(next_observation),
                self._worker_input_dim(worker),
            )
            worker.process_experience(
                worker_features,
                worker_action,
                reward,
                next_worker_features,
                done,
            )

    def reset(self) -> None:
        self._state = CoordinatorState()
        self._per_env_state.clear()

    def reset_env(self, env_id: str) -> None:
        self._per_env_state.pop(env_id, None)

    def _vectorize_observation(self, observation: Any) -> np.ndarray:
        if isinstance(observation, ObservationPacket):
            telemetry = observation.telemetry
        elif isinstance(observation, Mapping):
            telemetry = observation.get("telemetry", observation)
        elif isinstance(observation, (np.ndarray, Sequence)):
            return np.asarray(observation, dtype=np.float32).reshape(-1)
        else:
            raise TypeError(
                f"Unsupported observation type {type(observation).__name__} for coordination"
            )

        distance = float(telemetry.get("distance_to_goal", 0.0))
        speed = float(telemetry.get("speed_mps", 0.0))
        lane_ratio = float(telemetry.get("lane_mask_coverage_ratio", 0.0))
        collision = 1.0 if telemetry.get("collision", False) else 0.0
        progress_flag = 1.0 if telemetry.get("progress_possible", True) else 0.0

        raw = np.array([distance, speed, lane_ratio, collision, progress_flag], dtype=np.float32)
        # Rough normalization to keep magnitudes comparable for placeholder models.
        raw[0] = np.clip(raw[0] / 100.0, -1.0, 1.0)  # distance scaled to ~[-1,1]
        raw[1] = np.clip(raw[1] / 30.0, -1.0, 1.0)  # speed scaled for car speeds
        raw[2:] = np.clip(raw[2:], -1.0, 1.0)
        return raw

    def _manager_input_dim(self) -> int:
        model = self._manager.model
        return self._infer_dim(model, default=5)

    def _worker_input_dim(self, worker: SACWorker) -> int:
        return self._infer_dim(worker.model, default=5)

    @staticmethod
    def _infer_dim(model: Any, *, default: int) -> int:
        space = getattr(model, "observation_space", None)
        if space is not None and getattr(space, "shape", None):
            shape = space.shape
            if isinstance(shape, tuple) and shape:
                return int(shape[0])
        return default

    @staticmethod
    def _adapt_features(features: np.ndarray, expected_dim: int) -> np.ndarray:
        if expected_dim <= 0:
            return features
        flat = features.reshape(-1)
        if flat.size == expected_dim:
            return flat
        if flat.size > expected_dim:
            return flat[:expected_dim]
        padded = np.zeros(expected_dim, dtype=flat.dtype)
        padded[: flat.size] = flat
        return padded

    @staticmethod
    def _extract_telemetry(observation: Any) -> Mapping[str, Any]:
        if isinstance(observation, ObservationPacket):
            return observation.telemetry
        if isinstance(observation, Mapping):
            return observation.get("telemetry", observation)
        return {}

    @staticmethod
    def _heading_delta(start: float, current: float) -> float:
        diff = (current - start + 180.0) % 360.0 - 180.0
        return abs(diff)

    @staticmethod
    def _signed_heading_delta(start: float, current: float) -> float:
        return (current - start + 180.0) % 360.0 - 180.0
