from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

import utils.metrics_tracker as metrics_module
from utils.metrics_tracker import MetricsTracker


@pytest.fixture
def frozen_time(monkeypatch):
    counter = itertools.count(start=1000, step=1)
    monkeypatch.setattr(metrics_module.time, "time", lambda: next(counter))


def _sample_reward_components(step: int) -> dict[str, float]:
    return {
        "command_shaping": 0.1 * step,
        "collision_penalty": -0.5 if step == 5 else 0.0,
        "completion_bonus": 1.0 if step == 10 else 0.0,
        "idle_penalty": -0.05 if step % 2 else 0.0,
    }


def test_metrics_tracker_end_to_end(tmp_path: Path, frozen_time) -> None:
    tracker = MetricsTracker(tmp_path, update_interval=3, async_plots=False)

    tracker.start_episode(1)
    for step in range(1, 11):
        tracker.log_step(
            step=step,
            reward=float(step),
            action={
                "target_speed": min(5.0, step / 2),
                "target_steering": (-1) ** step * 0.2,
            },
            telemetry={
                "speed_mps": step * 0.5,
                "distance_to_goal": max(0.0, 10 - step),
                "collision": step == 5,
                "position_xy": (float(step), float(step) * 0.1),
                "goal_xy": (10.0, 0.0),
            },
            reward_components=_sample_reward_components(step),
            command="FORWARD" if step % 2 == 0 else "HOLD",
        )
    tracker.finish_episode(completed_successfully=True)

    tracker.start_episode(2)
    tracker.log_step(
        step=1,
        reward=0.5,
        action={"target_speed": 0.6, "target_steering": 0.1},
        telemetry={
            "speed_mps": 1.2,
            "distance_to_goal": 5.0,
            "collision": False,
            "position_xy": (0.5, 0.2),
            "goal_xy": (10.0, 0.0),
        },
        reward_components=_sample_reward_components(1),
        command="FORWARD",
    )
    tracker.finish_episode()

    metrics_dir = tmp_path / "metrics"
    episodes_path = metrics_dir / "episodes.json"
    steps_path = metrics_dir / "episode_2_steps.json"
    trajectory_path = metrics_dir / "trajectory_ep1.json"
    trajectory_plot = metrics_dir / "trajectory.png"
    assert episodes_path.exists()
    assert steps_path.exists()
    assert trajectory_plot.exists()
    assert trajectory_path.exists()

    episodes = json.loads(episodes_path.read_text(encoding="utf-8"))
    assert [episode["episode"] for episode in episodes] == [1, 2]
    assert episodes[0]["completed_successfully"] is True
    assert episodes[1]["completed_successfully"] is False

    steps_data = json.loads(steps_path.read_text(encoding="utf-8"))
    assert [entry["step"] for entry in steps_data] == [1]
    assert steps_data[0]["action"]["target_speed"] == pytest.approx(0.6)
    assert steps_data[0]["position_xy"] == [0.5, 0.2]
    assert steps_data[0]["goal_xy"] == [10.0, 0.0]

    trajectory_data = json.loads(trajectory_path.read_text(encoding="utf-8"))
    assert trajectory_data["target"] == [10.0, 0.0]
    assert trajectory_data["positions"][0] == [1.0, 0.1]

    summary = tracker.get_summary_stats()
    assert summary["total_episodes"] == 2
    assert summary["success_rate"] == pytest.approx(0.5)
    assert summary["collision_rate"] == pytest.approx(0.5)
    assert summary["avg_max_speed"] > 0


def test_metrics_tracker_handles_idle_states(tmp_path: Path, frozen_time, capsys) -> None:
    tracker = MetricsTracker(tmp_path, update_interval=1, async_plots=False)

    tracker.log_step(step=1, reward=1.0, action={}, telemetry={}, reward_components={})
    tracker.finish_episode()
    assert tracker.get_summary_stats() == {}

    tracker.start_episode(1)
    tracker.log_step(
        step=1,
        reward=1.0,
        action={"target_speed": 0.5},
        telemetry={"speed_mps": 1.0, "distance_to_goal": 9.0, "collision": False},
        reward_components=_sample_reward_components(1),
    )

    def boom(*_args, **_kwargs) -> None:
        raise RuntimeError("plot failure")

    tracker._plot_reward_trends = boom  # type: ignore[method-assign]
    tracker._update_graphs()

    captured = capsys.readouterr()
    assert "Failed to update graphs" in captured.out

    tracker.finish_episode()
    assert tracker.get_summary_stats()["total_episodes"] == 1


def test_normalize_xy_variants() -> None:
    assert metrics_module._normalize_xy({"x": 1, "y": 2}) == (1.0, 2.0)
    assert metrics_module._normalize_xy([3, "4"]) == (3.0, 4.0)
    assert metrics_module._normalize_xy(np.array([5.0, 6.0, 7.0])) == (5.0, 6.0)
    assert metrics_module._normalize_xy(None) is None
    assert metrics_module._normalize_xy({"x": "bad"}) is None
    assert metrics_module._normalize_xy({"y": 1}) is None
    assert metrics_module._normalize_xy(np.array([42.0])) is None
    assert metrics_module._normalize_xy((1.0,)) is None


def test_log_step_with_next_telemetry_tracks_positions(tmp_path: Path, frozen_time) -> None:
    tracker = MetricsTracker(tmp_path, update_interval=0, async_plots=False)
    tracker.start_episode(1)

    tracker.log_step(
        step=1,
        reward=0.0,
        action={"target_speed": 1.0, "target_steering": 0.0},
        telemetry={
            "speed_mps": 0.0,
            "distance_to_goal": 1.0,
            "collision": False,
            "position_xy": (0.0, 0.0),
        },
        reward_components={},
        next_telemetry={
            "position_xy": (0.5, 0.0),
            "goal_xy": (1.0, 0.0),
        },
    )

    tracker.log_step(
        step=2,
        reward=0.0,
        action={"target_speed": 1.0, "target_steering": 0.0},
        telemetry={
            "speed_mps": 0.0,
            "distance_to_goal": 0.5,
            "collision": False,
            "position_xy": (0.5, 0.0),
        },
        reward_components={},
        next_telemetry={
            "position_xy": (0.5, 0.0),
        },
    )

    assert tracker.current_episode_positions == [(0.0, 0.0), (0.5, 0.0)]
    assert tracker.current_target_xy == (1.0, 0.0)
    assert tracker.current_episode_steps[-1].goal_xy == (1.0, 0.0)

    tracker._plot_trajectory_map()
    assert (tmp_path / "metrics" / "trajectory.png").exists()

    tracker.current_episode_positions = [(1.0, 1.0)]
    tracker.current_target_xy = (1.0, 1.0)
    tracker._plot_trajectory_map()

    before = list(tracker.current_episode_positions)
    tracker._record_position(None)
    tracker._record_position(before[-1])
    assert tracker.current_episode_positions == before


def test_log_step_ignored_when_no_active_episode(tmp_path: Path) -> None:
    tracker = MetricsTracker(tmp_path, update_interval=0, async_plots=False)
    tracker.log_step(step=1, reward=0.0, action={}, telemetry={}, reward_components={})
    assert tracker.current_episode_steps == []
