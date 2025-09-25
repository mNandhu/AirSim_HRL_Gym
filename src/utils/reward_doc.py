"""Generate reward contract documentation from the reward configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from airsim_env.reward import RewardCalculator, RewardConfig

__all__ = ["generate_reward_contract"]


TEMPLATE = """# Reward Contract

| Component | Weight/Formula | Notes |
| --------- | -------------- | ----- |
{rows}
"""


def generate_reward_contract(
    path: str | Path,
    *,
    config: RewardConfig | None = None,
    additional_notes: Mapping[str, str] | None = None,
) -> Path:
    calculator = RewardCalculator(config)
    rows = [
        _row("progress", "prev_distance - current_distance", "Positive when moving toward goal"),
        _row(
            "lane_adherence", "lane_mask_coverage_ratio", "Normalized coverage within drivable area"
        ),
        _row(
            "collision",
            f"-{calculator.config.collision_penalty}",
            "Penalty applied when AirSim reports a collision",
        ),
        _row(
            "command_completion",
            f"+{calculator.config.completion_bonus}",
            "Bonus when high-level command objective completes",
        ),
        _row(
            "idle_penalty",
            f"-{calculator.config.idle_penalty_coef} (speed < {calculator.config.idle_threshold})",
            "Discourages idling when progress possible",
        ),
    ]

    if additional_notes:
        for component, note in additional_notes.items():
            rows.append(_row(component, "-", note))

    content = TEMPLATE.format(rows="\n".join(rows))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _row(component: str, formula: str, notes: str) -> str:
    return f"| {component} | {formula} | {notes} |"
