from __future__ import annotations

from pathlib import Path

from utils.metrics_tracker import MetricsTracker


def test_metrics_drops_stale_first_point(tmp_path: Path):
    tracker = MetricsTracker(tmp_path, update_interval=0, async_plots=False)
    tracker.start_episode(1)

    # First call logs previous episode end location
    telemetry_prev = {"position_xy": (35.0, 5.0)}
    next_telemetry_first = {"position_xy": (0.0, 0.0)}
    tracker.log_step(
        step=0,
        reward=0.0,
        action={"throttle": 0.0},
        telemetry=telemetry_prev,
        reward_components={},
        command="FOLLOW_LANE",
        next_telemetry=next_telemetry_first,
    )

    # Subsequent step continues from near origin
    telemetry = {"position_xy": (0.2, 0.1)}
    next_telemetry = {"position_xy": (0.4, 0.2)}
    tracker.log_step(
        step=1,
        reward=0.0,
        action={"throttle": 0.1},
        telemetry=telemetry,
        reward_components={},
        command="FOLLOW_LANE",
        next_telemetry=next_telemetry,
    )

    # The first stale point (35,5) should be dropped after adding the second, leaving ~[(0,0),(0.2,0.1),(0.4,0.2)]
    positions = tracker.current_episode_positions
    assert len(positions) >= 2
    x0, y0 = positions[0]
    assert abs(x0) < 1.0 and abs(y0) < 1.0
