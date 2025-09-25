import pytest

from airsim_env.reward import RewardCalculator, RewardConfig


def make_telemetry(
    distance: float, speed: float, collision: bool = False, lane_ratio: float = 1.0
) -> dict:
    return {
        "distance_to_goal": distance,
        "speed_mps": speed,
        "collision": collision,
        "lane_mask_coverage_ratio": lane_ratio,
    }


def test_reward_progress_and_lane_adherence():
    calc = RewardCalculator(
        RewardConfig(completion_bonus=5.0, idle_penalty_coef=0.5, idle_threshold=0.1)
    )

    prev = make_telemetry(distance=10.0, speed=2.0, lane_ratio=0.8)
    curr = make_telemetry(distance=8.0, speed=2.5, lane_ratio=0.9)

    components, total = calc.compute(prev, curr, command_completed=False, progress_possible=True)

    assert components["progress"] == pytest.approx(2.0)
    assert components["lane_adherence"] == pytest.approx(0.9)
    assert components["collision"] == 0.0
    assert components["command_completion"] == 0.0
    assert components["idle_penalty"] == 0.0
    assert total == pytest.approx(sum(components.values()))


def test_reward_handles_collision_and_command_completion():
    calc = RewardCalculator(
        RewardConfig(completion_bonus=3.0, idle_penalty_coef=1.0, idle_threshold=0.2)
    )

    prev = make_telemetry(distance=5.0, speed=1.0)
    curr = make_telemetry(distance=5.0, speed=0.0, collision=True)

    components, total = calc.compute(prev, curr, command_completed=True, progress_possible=False)

    assert components["collision"] == -1.0
    assert components["command_completion"] == 3.0
    assert total == pytest.approx(sum(components.values()))


def test_idle_penalty_applies_when_stationary(monkeypatch):
    calc = RewardCalculator(
        RewardConfig(completion_bonus=2.0, idle_penalty_coef=0.75, idle_threshold=0.5)
    )

    prev = make_telemetry(distance=4.0, speed=0.6)
    curr = make_telemetry(distance=4.0, speed=0.0)

    components, _ = calc.compute(prev, curr, command_completed=False, progress_possible=True)

    assert components["idle_penalty"] == pytest.approx(-0.75)
