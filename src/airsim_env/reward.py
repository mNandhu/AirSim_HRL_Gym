"""Reward computation utilities for the AirSim hierarchical environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = ["RewardConfig", "RewardCalculator"]


@dataclass(frozen=True)
class RewardConfig:
    """Configuration parameters for reward shaping components."""

    completion_bonus: float = 2.0
    idle_penalty_coef: float = 0.5
    idle_threshold: float = 0.2
    collision_penalty: float = 1.0


class RewardCalculator:
    """Compute shaped reward components according to the documented contract."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self._config = config or RewardConfig()

    @property
    def config(self) -> RewardConfig:
        return self._config

    def compute(
        self,
        previous: Mapping[str, Any],
        current: Mapping[str, Any],
        *,
        command_completed: bool,
        progress_possible: bool,
    ) -> tuple[dict[str, float], float]:
        progress = self._progress(previous, current)
        lane_adherence = float(current.get("lane_mask_coverage_ratio", 0.0))
        collision = -self._config.collision_penalty if current.get("collision", False) else 0.0
        command_completion = self._config.completion_bonus if command_completed else 0.0
        idle_penalty = self._idle_penalty(current, progress_possible)

        components = {
            "progress": progress,
            "lane_adherence": lane_adherence,
            "collision": collision,
            "command_completion": command_completion,
            "idle_penalty": idle_penalty,
        }
        total = float(sum(components.values()))
        return components, total

    def _progress(self, previous: Mapping[str, Any], current: Mapping[str, Any]) -> float:
        prev_distance = float(previous.get("distance_to_goal", 0.0))
        current_distance = float(current.get("distance_to_goal", prev_distance))
        return prev_distance - current_distance

    def _idle_penalty(self, current: Mapping[str, Any], progress_possible: bool) -> float:
        if not progress_possible:
            return 0.0
        speed = float(current.get("speed_mps", 0.0))
        if speed < self._config.idle_threshold:
            return -self._config.idle_penalty_coef
        return 0.0
