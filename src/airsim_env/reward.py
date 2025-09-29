"""
Reward computation utilities for the AirSim hierarchical environment.

This version incorporates advanced reward shaping techniques to provide denser
learning signals and promote safer, more effective driving behaviors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# from typing import Any, Mapping
import numpy as np

__all__ = ["RewardConfig", "VehicleState", "RewardCalculator"]


@dataclass(frozen=True)
class VehicleState:
    """A structured snapshot of the vehicle's state for reward calculation."""

    # Core Telemetry
    speed_mps: float = 0.0
    collision: bool = False

    # Positional & Navigational Data
    distance_to_goal: float = 0.0
    distance_from_lane_center: float = 0.0

    # Vector Data (should be normalized unit vectors)
    forward_vector: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    vector_to_next_waypoint: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))


@dataclass(frozen=True)
class RewardConfig:
    """Configuration parameters for reward shaping components."""

    # --- Major Penalties (Dominant) ---
    collision_penalty: float = 200.0  # Large penalty to strongly discourage collisions.

    # --- Goal & Task Bonuses (Sparse) ---
    completion_bonus: float = 100.0  # Large bonus for successfully completing a high-level command.

    # --- Dense Shaping Coefficients (Per-step guidance) ---
    # Amplified to provide stronger immediate incentives for moving toward the goal
    progress_velocity_coef: float = 1.2  # Rewards velocity in the correct direction.
    lane_deviation_penalty_coef: float = 2.0  # Penalizes deviation from the lane center (squared).
    # Amplified to encourage decisive turning toward the waypoint
    heading_alignment_coef: float = 1.0  # Rewards aligning with the path during turns.
    idle_penalty_coef: float = 0.5  # Small penalty for being stationary when progress is possible.

    # --- Thresholds ---
    idle_threshold_mps: float = 0.1  # Speed below which the idle penalty applies.

    # --- Efficiency penalty (Per-step) ---
    # Small, negative value applied each timestep to incentivize finishing quickly
    time_penalty: float = -0.02


class RewardCalculator:
    """Computes a shaped reward based on vehicle state changes."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self._config = config or RewardConfig()

    @property
    def config(self) -> RewardConfig:
        return self._config

    def compute(
        self,
        current_state: VehicleState,
        *,
        active_command: str,
        command_completed: bool,
        progress_possible: bool,
    ) -> tuple[dict[str, float], float]:
        """
        Calculates the total reward and its individual components.

        Args:
            current_state: The current state of the vehicle.
            active_command: The name of the active high-level command (e.g., "FOLLOW_LANE").
            command_completed: Flag indicating if the high-level command just finished.
            progress_possible: Flag indicating if the agent is in a state where it should be moving.

        Returns:
            A tuple containing a dictionary of reward components and the total scalar reward.
        """
        # --- Calculate all potential reward components ---
        progress = self._progress_velocity(current_state)
        lane_deviation_penalty = self._lane_deviation_penalty(current_state)
        heading_alignment = self._heading_alignment_reward(current_state)

        # --- Apply command-specific shaping ---
        # This is the core of HRL reward shaping: the reward function adapts to the current task.
        command_shaping_reward = 0.0
        if active_command == "FOLLOW_LANE":
            command_shaping_reward = lane_deviation_penalty + progress
        elif "TURN" in active_command:
            command_shaping_reward = heading_alignment + progress

        # --- Calculate penalties and bonuses ---
        collision = -self._config.collision_penalty if current_state.collision else 0.0
        completion_bonus = self._config.completion_bonus if command_completed else 0.0
        idle_penalty = self._idle_penalty(current_state, progress_possible)

        components = {
            "command_shaping": command_shaping_reward,
            "collision_penalty": collision,
            "completion_bonus": completion_bonus,
            "idle_penalty": idle_penalty,
            # Applied every time step to push for efficiency
            "time_penalty": self._config.time_penalty,
        }
        total = float(sum(components.values()))
        return components, total

    def _progress_velocity(self, state: VehicleState) -> float:
        """Rewards speed in the direction of the next waypoint."""
        # Dot product of forward vector and goal vector gives cosine of the angle.
        # 1.0 = perfectly aligned, -1.0 = facing opposite direction.
        # We only want to reward forward progress, so clip at 0.
        alignment = float(np.dot(state.forward_vector, state.vector_to_next_waypoint))
        forward_alignment = max(0.0, alignment)

        # Scale the alignment by the current speed.
        reward = state.speed_mps * forward_alignment * self._config.progress_velocity_coef
        return reward

    def _lane_deviation_penalty(self, state: VehicleState) -> float:
        """Penalizes distance from the lane centerline, squared."""
        # Squaring the distance heavily punishes large deviations but is lenient on small ones.
        penalty = -self._config.lane_deviation_penalty_coef * (state.distance_from_lane_center**2)
        return penalty

    def _heading_alignment_reward(self, state: VehicleState) -> float:
        """Dense reward for aligning the vehicle's heading towards the next waypoint."""
        # This is crucial for guiding turns.
        alignment = float(np.dot(state.forward_vector, state.vector_to_next_waypoint))
        reward = alignment * self._config.heading_alignment_coef
        return reward

    def _idle_penalty(self, state: VehicleState, progress_possible: bool) -> float:
        """Applies a small penalty for not moving when it's possible to do so."""
        if not progress_possible:
            return 0.0
        if state.speed_mps < self._config.idle_threshold_mps:
            return -self._config.idle_penalty_coef
        return 0.0
