"""Test that distance_to_goal telemetry reflects current waypoint distance."""

from airsim_env.env import AirSimEnv
from airsim_env.reward import RewardCalculator
from config.experiment import ExperimentDefinition, Pose, SeedBundle


class SimpleSimulator:
    """Simulator that provides position but calculates distance to final goal."""

    def __init__(self):
        self._position = (0.0, 0.0)
        self._experiment = None

    def reset(self, experiment):
        # Save experiment for step calls
        if experiment is not None:
            self._experiment = experiment

        # Start at start_pose
        self._position = (self._experiment.start_pose.x, self._experiment.start_pose.y)

        # Calculate distance to FINAL goal (wrong for waypoint navigation)
        if self._experiment.waypoints:
            final_wp = self._experiment.waypoints[-1]
            dist_to_final = (
                (self._position[0] - final_wp.x) ** 2 + (self._position[1] - final_wp.y) ** 2
            ) ** 0.5
        else:
            dist_to_final = 999.0

        return {
            "telemetry": {
                "distance_to_goal": dist_to_final,  # WRONG: distance to final goal
                "speed_mps": 0.0,
                "collision": False,
                "lane_mask_coverage_ratio": 1.0,
                "progress_possible": True,
                "position_xy": self._position,
            },
            "image": None,
        }

    def step(self, action):
        # Move forward
        self._position = (self._position[0] + 1.0, self._position[1])

        # Recalculate distance to final goal
        if self._experiment.waypoints:
            final_wp = self._experiment.waypoints[-1]
            dist_to_final = (
                (self._position[0] - final_wp.x) ** 2 + (self._position[1] - final_wp.y) ** 2
            ) ** 0.5
        else:
            dist_to_final = 999.0

        return {
            "telemetry": {
                "distance_to_goal": dist_to_final,
                "speed_mps": 1.0,
                "collision": False,
                "lane_mask_coverage_ratio": 1.0,
                "progress_possible": True,
                "position_xy": self._position,
            },
            "image": None,
        }


class NoOpPerception:
    def build_observation_inputs(self, raw_rgb):
        return {"segmentation_mask": None, "detections": []}


def test_distance_reflects_current_waypoint():
    """Test that distance_to_goal shows final goal, distance_to_current_waypoint shows progress."""

    # Create experiment with 3 waypoints
    experiment = ExperimentDefinition(
        scene="Test",
        vehicle="TestCar",
        start_pose=Pose(x=0, y=0, z=0, yaw=0),
        waypoints=[
            Pose(x=10, y=0, z=0, yaw=0),  # First waypoint 10m away
            Pose(x=20, y=0, z=0, yaw=0),  # Second waypoint
            Pose(x=30, y=0, z=0, yaw=0),  # Final goal
        ],
        horizon=100,
        seeds=SeedBundle(python=1, numpy=2, torch=3, airsim=4),
    )

    env = AirSimEnv(
        experiment,
        simulator=SimpleSimulator(),
        perception=NoOpPerception(),
        reward_calculator=RewardCalculator(),
    )

    # Reset and check initial distances
    obs = env.reset()
    distance_to_goal = obs.telemetry.get("distance_to_goal", 0.0)
    initial_waypoint_distance = obs.telemetry.get("distance_to_current_waypoint", 0.0)

    print(f"Distance to final goal: {distance_to_goal:.2f}m")
    print(f"Distance to current waypoint: {initial_waypoint_distance:.2f}m")

    # distance_to_goal should be ~30m (distance to FINAL goal)
    assert 29.0 < distance_to_goal < 31.0, (
        f"Expected distance_to_goal ~30m to final waypoint, got {distance_to_goal}m"
    )

    # distance_to_current_waypoint should be ~10m (distance to first waypoint)
    assert 9.0 < initial_waypoint_distance < 11.0, (
        f"Expected distance_to_current_waypoint ~10m to first waypoint, got {initial_waypoint_distance}m"
    )

    # Take a few steps to move closer to first waypoint (but not past it)
    for i in range(3):
        obs, reward, terminated, truncated, info = env.step(
            {"target_speed": 1.0, "target_steering": 0.0}
        )
        distance_to_waypoint = obs.telemetry.get("distance_to_current_waypoint", 0.0)
        print(f"Step {i + 1}: distance to current waypoint = {distance_to_waypoint:.2f}m")

    # distance_to_current_waypoint should have decreased
    final_waypoint_distance = obs.telemetry.get("distance_to_current_waypoint", 0.0)
    assert final_waypoint_distance < initial_waypoint_distance, (
        f"Distance to waypoint should decrease: initially {initial_waypoint_distance:.2f} -> {final_waypoint_distance:.2f}"
    )

    print(
        "Test passed! distance_to_goal shows final goal, distance_to_current_waypoint shows waypoint progress."
    )
    print(f"   Final goal distance: {distance_to_goal:.2f}m (stays constant)")
    print(
        f"   Current waypoint: {initial_waypoint_distance:.2f}m -> {final_waypoint_distance:.2f}m"
    )


if __name__ == "__main__":
    test_distance_reflects_current_waypoint()
