"""Core environment wrapper that coordinates simulator, perception, and rewards."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID
import hashlib
import json

from config.experiment import ExperimentDefinition, SeedBundle
from config.seeds import apply_seed_bundle
from .observation import ObservationPacket, assemble_observation
from .reward import RewardCalculator, RewardConfig

# Threshold for reaching the goal in meters
_GOAL_THRESHOLD = 0.5


@dataclass
class _CommandContext:
    name: str | None = None
    completed: bool = False


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
        apply_seed_bundle(bundle)
        self._step_index = 0

        sim_state = self._simulator.reset(self._experiment)
        observation = self._build_observation(
            sim_state,
            reward_components={
                "progress": 0.0,
                "lane_adherence": float(
                    sim_state["telemetry"].get("lane_mask_coverage_ratio", 0.0)
                ),
                "collision": 0.0,
                "command_completion": 0.0,
                "idle_penalty": 0.0,
            },
            done_flags={"terminated": False, "truncated": False},
        )
        self._last_observation = observation
        return observation

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
        components, total = self._reward_calculator.compute(
            previous,
            current,
            command_completed=self._command_context.completed,
            progress_possible=progress_possible,
        )
        # Command completion is single-use until reset
        self._command_context.completed = False
        return components, total

    def set_command(self, name: str | None, completed: bool = False) -> None:
        self._command_context = _CommandContext(name=name, completed=completed)

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
