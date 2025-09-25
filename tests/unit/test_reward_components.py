import numpy as np
import pytest

from airsim_env.reward import RewardCalculator, RewardConfig, VehicleState


def make_state(
    *,
    speed: float = 0.0,
    lane_deviation: float = 0.0,
    collision: bool = False,
    forward: np.ndarray | None = None,
    goal_vector: np.ndarray | None = None,
) -> VehicleState:
    forward_vec = forward if forward is not None else np.array([1.0, 0.0, 0.0])
    goal_vec = goal_vector if goal_vector is not None else np.array([1.0, 0.0, 0.0])
    return VehicleState(
        speed_mps=speed,
        collision=collision,
        distance_to_goal=10.0,
        distance_from_lane_center=lane_deviation,
        forward_vector=forward_vec,
        vector_to_next_waypoint=goal_vec,
    )


def test_follow_lane_shaping_balances_progress_and_lane_centering():
    config = RewardConfig(
        collision_penalty=50.0,
        completion_bonus=5.0,
        progress_velocity_coef=0.5,
        lane_deviation_penalty_coef=2.0,
        heading_alignment_coef=0.3,
        idle_penalty_coef=0.1,
        idle_threshold_mps=0.1,
    )
    calc = RewardCalculator(config)

    state = make_state(speed=2.0, lane_deviation=0.1)

    components, total = calc.compute(
        state,
        active_command="FOLLOW_LANE",
        command_completed=False,
        progress_possible=True,
    )

    expected_progress = 2.0 * 1.0 * config.progress_velocity_coef
    expected_lane_penalty = -config.lane_deviation_penalty_coef * (0.1**2)
    expected_command = expected_progress + expected_lane_penalty

    assert components["command_shaping"] == pytest.approx(expected_command)
    assert components["collision_penalty"] == 0.0
    assert components["completion_bonus"] == 0.0
    assert components["idle_penalty"] == 0.0
    assert total == pytest.approx(sum(components.values()))


def test_turn_alignment_rewards_steering_toward_goal():
    config = RewardConfig(heading_alignment_coef=0.75, progress_velocity_coef=0.3)
    calc = RewardCalculator(config)

    forward = np.array([1.0, 0.0, 0.0])
    goal_vector = np.array([0.0, 1.0, 0.0])
    state = make_state(speed=1.5, lane_deviation=0.0, forward=forward, goal_vector=goal_vector)

    components, _ = calc.compute(
        state,
        active_command="TURN_LEFT_AT_INTERSECTION",
        command_completed=False,
        progress_possible=True,
    )

    expected_progress = 1.5 * 0.0 * config.progress_velocity_coef
    expected_heading = np.dot(forward, goal_vector) * config.heading_alignment_coef
    assert components["command_shaping"] == pytest.approx(expected_progress + expected_heading)


def test_collision_completion_and_idle_terms():
    config = RewardConfig(
        collision_penalty=60.0,
        completion_bonus=7.0,
        idle_penalty_coef=0.4,
        idle_threshold_mps=0.5,
    )
    calc = RewardCalculator(config)

    state = make_state(speed=0.2, collision=True)

    components, total = calc.compute(
        state,
        active_command="STOP",
        command_completed=True,
        progress_possible=True,
    )

    assert components["collision_penalty"] == -config.collision_penalty
    assert components["completion_bonus"] == config.completion_bonus
    assert components["idle_penalty"] == -config.idle_penalty_coef
    assert total == pytest.approx(sum(components.values()))
