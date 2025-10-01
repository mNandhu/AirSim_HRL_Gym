"""Path management for waypoint-based navigation.

This module provides waypoint management to align reward signals with path-following
behavior, resolving conflicts between navigation commands and goal-reaching incentives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["PathManager", "Waypoint"]


@dataclass(frozen=True)
class Waypoint:
    """A single waypoint in 3D space."""

    x: float
    y: float
    z: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, z]."""
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def distance_to(
        self, other: Waypoint | tuple[float, float] | tuple[float, float, float]
    ) -> float:
        """Calculate Euclidean distance to another waypoint or position."""
        if isinstance(other, Waypoint):
            other_arr = other.to_array()
        elif len(other) == 2:
            other_arr = np.array([other[0], other[1], 0.0], dtype=np.float32)
        else:
            other_arr = np.array([other[0], other[1], other[2]], dtype=np.float32)

        return float(np.linalg.norm(self.to_array() - other_arr))


class PathManager:
    """Manages waypoint-based path following.

    Tracks progress along a predefined path of waypoints, automatically advancing
    to the next waypoint when the current one is reached within a threshold distance.

    This resolves the fundamental learning issue where agents were rewarded for moving
    towards a distant final goal (straight line) while simultaneously being told to
    follow curved roads, creating contradictory reward signals.
    """

    def __init__(
        self,
        waypoints: Sequence[Waypoint | tuple[float, float] | tuple[float, float, float]],
        *,
        waypoint_threshold: float = 5.0,
    ) -> None:
        """Initialize path manager with a sequence of waypoints.

        Args:
            waypoints: Ordered sequence of waypoints defining the path. Each can be:
                - Waypoint object
                - (x, y) tuple (z defaults to 0.0)
                - (x, y, z) tuple
            waypoint_threshold: Distance (meters) within which a waypoint is considered reached

        Raises:
            ValueError: If waypoints list is empty
        """
        if not waypoints:
            raise ValueError("Waypoints list cannot be empty")

        # Convert all waypoints to Waypoint objects
        self._waypoints: list[Waypoint] = []
        for wp in waypoints:
            if isinstance(wp, Waypoint):
                self._waypoints.append(wp)
            elif len(wp) == 2:
                self._waypoints.append(Waypoint(x=float(wp[0]), y=float(wp[1]), z=0.0))
            elif len(wp) == 3:
                self._waypoints.append(Waypoint(x=float(wp[0]), y=float(wp[1]), z=float(wp[2])))
            else:
                raise ValueError(f"Invalid waypoint format: {wp}")

        self._waypoint_threshold = float(waypoint_threshold)
        self._current_index = 0

    def reset(self) -> None:
        """Reset to the first waypoint."""
        self._current_index = 0

    @property
    def current_waypoint_index(self) -> int:
        """Get the index of the current target waypoint."""
        return self._current_index

    @property
    def current_waypoint(self) -> Waypoint:
        """Get the current target waypoint."""
        return self._waypoints[self._current_index]

    @property
    def total_waypoints(self) -> int:
        """Get total number of waypoints in the path."""
        return len(self._waypoints)

    @property
    def is_final_waypoint(self) -> bool:
        """Check if currently targeting the final waypoint."""
        return self._current_index >= len(self._waypoints) - 1

    @property
    def all_waypoints_reached(self) -> bool:
        """Check if all waypoints have been reached (past final waypoint)."""
        return self._current_index >= len(self._waypoints)

    def get_distance_to_current_waypoint(
        self,
        position: tuple[float, float] | tuple[float, float, float],
    ) -> float:
        """Calculate distance from current position to current target waypoint.

        Args:
            position: Current position as (x, y) or (x, y, z) tuple

        Returns:
            Distance in meters
        """
        if self.all_waypoints_reached:
            return 0.0
        return self.current_waypoint.distance_to(position)

    def get_vector_to_current_waypoint(
        self,
        position: tuple[float, float] | tuple[float, float, float],
        *,
        normalize: bool = True,
    ) -> np.ndarray:
        """Calculate vector from current position to current target waypoint.

        Args:
            position: Current position as (x, y) or (x, y, z) tuple
            normalize: If True, return unit vector (default). If False, return raw vector.

        Returns:
            3D vector as numpy array [x, y, z]
        """
        if self.all_waypoints_reached:
            # Return zero vector if all waypoints reached
            return np.zeros(3, dtype=np.float32)

        # Convert position to array
        if len(position) == 2:
            pos_arr = np.array([position[0], position[1], 0.0], dtype=np.float32)
        else:
            pos_arr = np.array([position[0], position[1], position[2]], dtype=np.float32)

        # Calculate vector to current waypoint
        waypoint_arr = self.current_waypoint.to_array()
        vector = waypoint_arr - pos_arr

        if normalize:
            norm = np.linalg.norm(vector)
            if norm > 1e-6:
                vector = vector / norm
            # If norm is near zero, return zero vector (already at waypoint)

        return vector

    def update(
        self,
        current_position: tuple[float, float] | tuple[float, float, float],
    ) -> bool:
        """Update waypoint progress based on current position.

        Automatically advances to the next waypoint if current waypoint is reached.

        Args:
            current_position: Current position as (x, y) or (x, y, z) tuple

        Returns:
            True if a waypoint was reached and advanced, False otherwise
        """
        if self.all_waypoints_reached:
            return False

        distance = self.get_distance_to_current_waypoint(current_position)

        if distance <= self._waypoint_threshold:
            # Waypoint reached, advance to next
            self._current_index += 1
            return True

        return False

    def get_progress_ratio(self) -> float:
        """Get overall progress along the path as a ratio [0.0, 1.0].

        Returns:
            0.0 at start, 1.0 when all waypoints reached
        """
        if len(self._waypoints) <= 1:
            return 1.0 if self.all_waypoints_reached else 0.0

        # Use waypoint index as progress measure
        return min(1.0, self._current_index / (len(self._waypoints) - 1))

    def get_remaining_waypoints(self) -> int:
        """Get number of waypoints remaining (including current target).

        Returns:
            Number of waypoints not yet reached
        """
        if self.all_waypoints_reached:
            return 0
        return len(self._waypoints) - self._current_index
