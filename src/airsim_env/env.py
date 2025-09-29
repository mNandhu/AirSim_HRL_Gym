"""Core environment wrapper that coordinates simulator, perception, and rewards."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID

import numpy as np

from config.experiment import ExperimentDefinition, SeedBundle
from config.seeds import apply_seed_bundle

from .observation import ObservationPacket, assemble_observation
from .reward import RewardCalculator, RewardConfig, VehicleState

# Threshold for reaching the goal in meters
_GOAL_THRESHOLD = 0.5


@dataclass
class _CommandContext:
    name: str | None = None
    pending_completion: bool = False
    completion_awarded: bool = False


class AirSimEnv:
    """Environment façade complying with the documented interface."""

    def __init__(
        self,
        experiment: ExperimentDefinition,
        *,
        simulator: Any,
        perception: Any,
        reward_calculator: RewardCalculator | None = None,
    ) -> None:
        self._experiment = experiment
        self._simulator = simulator
        self._perception = perception
        self._reward_calculator = reward_calculator or RewardCalculator(RewardConfig())

        self._seed_bundle = experiment.seeds
        self._config_hash = experiment.config_hash or _hash_experiment(experiment)

        self._step_index = 0
        self._last_observation: ObservationPacket | None = None
        self._command_context = _CommandContext()

    @property
    def experiment(self) -> ExperimentDefinition:
        return self._experiment

    def reset(self, seed_bundle: SeedBundle | None = None) -> ObservationPacket:
        bundle = seed_bundle or self._seed_bundle
        airsim_client = getattr(self._simulator, "client", None)
        apply_seed_bundle(bundle, airsim_client=airsim_client)
        self._step_index = 0
        self._command_context = _CommandContext()

        sim_state = self._simulator.reset(self._experiment)
        observation = self._build_observation(
            sim_state,
            reward_components={
                "command_shaping": 0.0,
                "collision_penalty": 0.0,
                "completion_bonus": 0.0,
                "idle_penalty": 0.0,
            },
            done_flags={"terminated": False, "truncated": False},
        )
        self._last_observation = observation
        return observation

    @property
    def last_observation(self) -> ObservationPacket | None:
        return self._last_observation

    def step(
        self, action: Mapping[str, float]
    ) -> tuple[ObservationPacket, float, bool, bool, dict[str, Any]]:
        if self._last_observation is None:
            raise RuntimeError("Environment must be reset before stepping")
        if self._step_index >= self._experiment.horizon:
            raise RuntimeError("Environment horizon exceeded")

        sim_state = self._simulator.step(action)
        telemetry = sim_state.get("telemetry", {})
        prev_telemetry = self._last_observation.telemetry
        progress_possible = bool(telemetry.get("progress_possible", True))
        reward_components, reward = self.compute_reward(
            prev_telemetry, telemetry, progress_possible=progress_possible
        )

        terminated = self._check_terminated(telemetry)
        truncated = self._check_truncated()
        done_flags = {
            "terminated": terminated,
            "truncated": truncated,
            "collision": bool(telemetry.get("collision", False)),
            "goal_reached": terminated and not telemetry.get("collision", False),
        }

        observation = self._build_observation(sim_state, reward_components, done_flags)
        self._step_index += 1
        self._last_observation = observation

        info = {
            "config_hash": self._config_hash,
            "seed_bundle": self._seed_bundle.model_dump()
            if hasattr(self._seed_bundle, "model_dump")
            else self._seed_bundle.__dict__,
            "episode_step": self._step_index,
            "command_name": self._command_context.name,
        }
        return observation, reward, terminated, truncated, info

    def compute_reward(
        self,
        previous: Mapping[str, Any],
        current: Mapping[str, Any],
        *,
        progress_possible: bool,
    ) -> tuple[dict[str, float], float]:
        current_state = self._vehicle_state_from_telemetry(current, previous=previous)
        active_command = self._command_context.name or ""
        command_completed = self._command_context.pending_completion
        components, total = self._reward_calculator.compute(
            current_state,
            active_command=active_command,
            command_completed=command_completed,
            progress_possible=progress_possible,
        )
        # Command completion is single-use until re-armed
        if command_completed:
            self._command_context.completion_awarded = True
        self._command_context.pending_completion = False
        return components, total

    def set_command(self, name: str | None, completed: bool = False) -> None:
        ctx = self._command_context
        if name != ctx.name:
            ctx = _CommandContext(name=name)
        else:
            if not completed:
                ctx.pending_completion = False
                if ctx.completion_awarded:
                    ctx.completion_awarded = False

        if completed and not ctx.completion_awarded:
            ctx.pending_completion = True

        self._command_context = ctx

    def close(self) -> None:
        terminate = getattr(self._simulator, "close", None)
        if callable(terminate):
            terminate()
        self._last_observation = None

    def _build_observation(
        self,
        sim_state: Mapping[str, Any],
        reward_components: Mapping[str, float],
        done_flags: Mapping[str, bool],
    ) -> ObservationPacket:
        perception_inputs = self._perception.build_observation_inputs(sim_state.get("image"))
        return assemble_observation(
            image=sim_state.get("image"),
            segmentation_mask=perception_inputs.get("segmentation_mask"),
            detections=perception_inputs.get("detections", []),
            telemetry=sim_state.get("telemetry", {}),
            reward_components=reward_components,
            command=self._command_context.name,
            done_flags=done_flags,
        )

    def _check_terminated(self, telemetry: Mapping[str, Any]) -> bool:
        if telemetry.get("collision", False):
            return True
        distance = float(telemetry.get("distance_to_goal", _GOAL_THRESHOLD + 1))
        return distance <= _GOAL_THRESHOLD

    def _check_truncated(self) -> bool:
        return self._step_index + 1 >= self._experiment.horizon

    def _vehicle_state_from_telemetry(
        self,
        telemetry: Mapping[str, Any],
        *,
        previous: Mapping[str, Any] | None = None,
    ) -> VehicleState:
        speed = float(telemetry.get("speed_mps", 0.0))
        collision = bool(telemetry.get("collision", False))
        distance_to_goal = float(telemetry.get("distance_to_goal", 0.0))

        if "lane_deviation_m" in telemetry:
            lane_deviation = float(telemetry.get("lane_deviation_m", 0.0))
        else:
            lane_ratio = float(telemetry.get("lane_mask_coverage_ratio", 1.0))
            lane_deviation = max(0.0, 1.0 - lane_ratio)

        heading_deg = telemetry.get("heading_deg")
        heading_rad = float(math.radians(heading_deg)) if heading_deg is not None else 0.0
        forward_vector = np.array(
            [math.cos(heading_rad), math.sin(heading_rad), 0.0], dtype=np.float32
        )

        vector_to_goal = telemetry.get("vector_to_goal")
        if vector_to_goal is None and previous is not None:
            vector_to_goal = previous.get("vector_to_goal")
        if vector_to_goal is not None:
            goal_vector = np.asarray(vector_to_goal, dtype=np.float32)
        else:
            goal_vector = forward_vector.copy()

        norm = float(np.linalg.norm(goal_vector))
        if norm > 1e-6:
            goal_vector = goal_vector / norm
        else:
            goal_vector = forward_vector.copy()

        return VehicleState(
            speed_mps=speed,
            collision=collision,
            distance_to_goal=distance_to_goal,
            distance_from_lane_center=lane_deviation,
            forward_vector=forward_vector,
            vector_to_next_waypoint=goal_vector,
        )


def _hash_experiment(experiment: ExperimentDefinition) -> str:
    payload = (
        experiment.model_dump() if hasattr(experiment, "model_dump") else experiment.__dict__.copy()
    )
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
