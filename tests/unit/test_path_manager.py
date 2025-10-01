"""Unit tests for PathManager waypoint-based navigation."""

from __future__ import annotations

import numpy as np
import pytest

from utils.path_manager import PathManager, Waypoint


class TestWaypoint:
    """Test Waypoint dataclass."""

    def test_waypoint_creation(self):
        wp = Waypoint(x=10.0, y=20.0, z=0.5)
        assert wp.x == 10.0
        assert wp.y == 20.0
        assert wp.z == 0.5

    def test_waypoint_default_z(self):
        wp = Waypoint(x=5.0, y=15.0)
        assert wp.z == 0.0

    def test_waypoint_to_array(self):
        wp = Waypoint(x=1.0, y=2.0, z=3.0)
        arr = wp.to_array()
        np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.0]))

    def test_waypoint_distance_to_waypoint(self):
        wp1 = Waypoint(x=0.0, y=0.0, z=0.0)
        wp2 = Waypoint(x=3.0, y=4.0, z=0.0)
        distance = wp1.distance_to(wp2)
        assert abs(distance - 5.0) < 1e-6

    def test_waypoint_distance_to_tuple_2d(self):
        wp = Waypoint(x=0.0, y=0.0, z=0.0)
        distance = wp.distance_to((3.0, 4.0))
        assert abs(distance - 5.0) < 1e-6

    def test_waypoint_distance_to_tuple_3d(self):
        wp = Waypoint(x=0.0, y=0.0, z=0.0)
        distance = wp.distance_to((0.0, 0.0, 10.0))
        assert abs(distance - 10.0) < 1e-6


class TestPathManager:
    """Test PathManager navigation logic."""

    def test_creation_with_waypoint_objects(self):
        waypoints = [
            Waypoint(0.0, 0.0, 0.0),
            Waypoint(10.0, 0.0, 0.0),
            Waypoint(10.0, 10.0, 0.0),
        ]
        pm = PathManager(waypoints)
        assert pm.total_waypoints == 3
        assert pm.current_waypoint_index == 0

    def test_creation_with_tuples_2d(self):
        waypoints = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        pm = PathManager(waypoints)
        assert pm.total_waypoints == 3
        assert pm.current_waypoint.x == 0.0
        assert pm.current_waypoint.y == 0.0
        assert pm.current_waypoint.z == 0.0

    def test_creation_with_tuples_3d(self):
        waypoints = [(0.0, 0.0, 1.0), (10.0, 0.0, 1.0)]
        pm = PathManager(waypoints)
        assert pm.total_waypoints == 2
        assert pm.current_waypoint.z == 1.0

    def test_creation_empty_waypoints_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            PathManager([])

    def test_creation_invalid_waypoint_format_raises(self):
        with pytest.raises(ValueError, match="Invalid waypoint format"):
            PathManager([(1.0,)])  # Too few elements

    def test_reset(self):
        waypoints = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
        pm = PathManager(waypoints)
        pm.update((3.0, 0.0))  # Should advance to waypoint 1
        assert pm.current_waypoint_index == 1
        pm.update((13.0, 0.0))  # Should advance to waypoint 2
        assert pm.current_waypoint_index == 2
        pm.reset()
        assert pm.current_waypoint_index == 0

    def test_get_distance_to_current_waypoint(self):
        waypoints = [(0.0, 0.0), (10.0, 0.0)]
        pm = PathManager(waypoints)
        distance = pm.get_distance_to_current_waypoint((0.0, 0.0))
        assert abs(distance) < 1e-6
        distance = pm.get_distance_to_current_waypoint((3.0, 4.0))
        assert abs(distance - 5.0) < 1e-6

    def test_get_vector_to_current_waypoint_normalized(self):
        waypoints = [(10.0, 0.0), (20.0, 0.0)]
        pm = PathManager(waypoints)
        vector = pm.get_vector_to_current_waypoint((0.0, 0.0), normalize=True)
        # Should point in +x direction
        expected = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(vector, expected, decimal=5)

    def test_get_vector_to_current_waypoint_unnormalized(self):
        waypoints = [(10.0, 0.0), (20.0, 0.0)]
        pm = PathManager(waypoints)
        vector = pm.get_vector_to_current_waypoint((0.0, 0.0), normalize=False)
        expected = np.array([10.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(vector, expected, decimal=5)

    def test_update_advances_waypoint_when_within_threshold(self):
        waypoints = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
        pm = PathManager(waypoints, waypoint_threshold=5.0)

        assert pm.current_waypoint_index == 0
        # Move close to first waypoint
        advanced = pm.update((2.0, 0.0))
        assert advanced is True
        assert pm.current_waypoint_index == 1

        # Move close to second waypoint
        advanced = pm.update((12.0, 0.0))
        assert advanced is True
        assert pm.current_waypoint_index == 2

    def test_update_does_not_advance_when_far(self):
        waypoints = [(0.0, 0.0), (10.0, 0.0)]
        pm = PathManager(waypoints, waypoint_threshold=2.0)

        advanced = pm.update((5.0, 0.0))  # Still 5m away
        assert advanced is False
        assert pm.current_waypoint_index == 0

    def test_is_final_waypoint(self):
        waypoints = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
        pm = PathManager(waypoints)

        assert pm.is_final_waypoint is False
        pm.update((5.0, 0.0))  # Advance to waypoint 1
        assert pm.is_final_waypoint is False
        pm.update((15.0, 0.0))  # Advance to waypoint 2 (final)
        assert pm.is_final_waypoint is True

    def test_all_waypoints_reached(self):
        waypoints = [(0.0, 0.0), (10.0, 0.0)]
        pm = PathManager(waypoints, waypoint_threshold=5.0)

        assert pm.all_waypoints_reached is False
        pm.update((3.0, 0.0))  # Reach waypoint 0, advance to 1
        assert pm.all_waypoints_reached is False
        pm.update((12.0, 0.0))  # Reach waypoint 1 (final)
        assert pm.all_waypoints_reached is True

    def test_get_progress_ratio(self):
        waypoints = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
        pm = PathManager(waypoints)

        assert abs(pm.get_progress_ratio() - 0.0) < 1e-6
        pm.update((3.0, 0.0))  # Advance to waypoint 1
        assert abs(pm.get_progress_ratio() - 0.5) < 1e-6
        pm.update((13.0, 0.0))  # Advance to waypoint 2 (final)
        assert abs(pm.get_progress_ratio() - 1.0) < 1e-6

    def test_get_remaining_waypoints(self):
        waypoints = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]
        pm = PathManager(waypoints)

        assert pm.get_remaining_waypoints() == 4
        pm.update((3.0, 0.0))  # Advance to waypoint 1
        assert pm.get_remaining_waypoints() == 3
        pm.update((13.0, 0.0))  # Advance to waypoint 2
        assert pm.get_remaining_waypoints() == 2

    def test_all_waypoints_reached_returns_zero_distance(self):
        waypoints = [(0.0, 0.0), (10.0, 0.0)]
        pm = PathManager(waypoints, waypoint_threshold=5.0)
        pm.update((3.0, 0.0))  # Reach first waypoint
        pm.update((12.0, 0.0))  # Reach final waypoint

        assert pm.all_waypoints_reached
        distance = pm.get_distance_to_current_waypoint((999.0, 999.0))
        assert distance == 0.0

    def test_all_waypoints_reached_returns_zero_vector(self):
        waypoints = [(0.0, 0.0), (10.0, 0.0)]
        pm = PathManager(waypoints, waypoint_threshold=5.0)
        pm.update((3.0, 0.0))
        pm.update((12.0, 0.0))

        vector = pm.get_vector_to_current_waypoint((999.0, 999.0))
        expected = np.zeros(3, dtype=np.float32)
        np.testing.assert_array_equal(vector, expected)

    def test_single_waypoint_path(self):
        """Edge case: path with single waypoint."""
        waypoints = [(10.0, 10.0)]
        pm = PathManager(waypoints, waypoint_threshold=5.0)

        assert pm.is_final_waypoint is True
        assert pm.get_remaining_waypoints() == 1

        pm.update((12.0, 12.0))  # Within threshold
        assert pm.all_waypoints_reached is True
        assert pm.get_progress_ratio() == 1.0
