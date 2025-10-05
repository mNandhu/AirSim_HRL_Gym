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

    # --- Major Penalties (Reduced to not dominate episode) ---
    collision_penalty: float = (
        10.0  # Penalty for collision (reduced from 50 to allow learning signal)
    )

    # --- Goal & Task Bonuses (Sparse) ---
    completion_bonus: float = 100.0  # Large bonus for successfully completing a high-level command.
    waypoint_progress_bonus: float = 50.0  # Bonus for reaching a waypoint

    # --- Dense Shaping Coefficients (Per-step guidance) ---
    # Increased to provide stronger rewards for good navigation behavior
    progress_velocity_coef: float = (
        2.0  # Rewards velocity in the correct direction (was 1.2, +67% boost)
    )
    lane_deviation_penalty_coef: float = (
        1.0  # Penalizes deviation from lane center (was 2.0, -50% to allow path-following)
    )
    # Amplified to encourage decisive turning toward the waypoint
    heading_alignment_coef: float = 1.0  # Rewards aligning with the path during turns.
    idle_penalty_coef: float = 0.5  # Small penalty for being stationary when progress is possible.
    action_smoothness_coef: float = (
        2.0  # Penalty for rapid steering changes (increased from 0.5 to reduce zigzag)
    )

    # --- Thresholds ---
    idle_threshold_mps: float = 0.1  # Speed below which the idle penalty applies.

    # --- Efficiency penalty (Per-step) ---
    # Reduced to minimize constant drain and allow more exploration
    time_penalty: float = -0.005  # Per-step penalty (was -0.02, reduced by 75%)


class RewardCalculator:
    """Computes a shaped reward based on vehicle state changes."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self._config = config or RewardConfig()
        self._previous_steering: float = 0.0  # Track previous steering for smoothness penalty

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
        waypoint_reached: bool = False,
        action: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], float]:
        """
        Calculates the total reward and its individual components.

        Args:
            current_state: The current state of the vehicle.
            active_command: The name of the active high-level command (e.g., "FOLLOW_LANE").
            command_completed: Flag indicating if the high-level command just finished.
            progress_possible: Flag indicating if the agent is in a state where it should be moving.
            waypoint_reached: Flag indicating if a waypoint was just reached.
            action: The action taken by the agent (dict with 'steering' key).

        Returns:
            A tuple containing a dictionary of reward components and the total scalar reward.
        """
        # --- Calculate all potential reward components ---
        progress = self._progress_velocity(current_state)
        lane_deviation_penalty = self._lane_deviation_penalty(current_state)
        heading_alignment = self._heading_alignment_reward(current_state)

        # --- Calculate action smoothness penalty ---
        smoothness_penalty = 0.0
        if action is not None:
            current_steering = float(action.get("steering", 0.0))
            steering_change = abs(current_steering - self._previous_steering)
            smoothness_penalty = -self._config.action_smoothness_coef * steering_change
            self._previous_steering = current_steering

        # --- Apply command-specific shaping ---
        # Single-agent now includes lane deviation to encourage road-following
        command_shaping_reward = 0.0
        if not active_command or active_command == "":
            # Single-agent mode: combine ALL navigation rewards including lane keeping
            command_shaping_reward = progress + heading_alignment + lane_deviation_penalty
        elif active_command == "FOLLOW_LANE":
            command_shaping_reward = lane_deviation_penalty + progress
        elif "TURN" in active_command:
            command_shaping_reward = heading_alignment + progress

        # --- Calculate penalties and bonuses ---
        collision = -self._config.collision_penalty if current_state.collision else 0.0
        completion_bonus = self._config.completion_bonus if command_completed else 0.0
        waypoint_bonus = self._config.waypoint_progress_bonus if waypoint_reached else 0.0
        idle_penalty = self._idle_penalty(current_state, progress_possible)

        components = {
            "command_shaping": command_shaping_reward,
            "collision_penalty": collision,
            "completion_bonus": completion_bonus,
            "waypoint_progress_bonus": waypoint_bonus,
            "idle_penalty": idle_penalty,
            "action_smoothness": smoothness_penalty,
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
