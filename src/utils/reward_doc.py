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
        _row(
            "command_shaping",
            "progress_velocity + lane/heading shaping",
            "Adaptive dense signal based on the active command",
        ),
        _row(
            "collision_penalty",
            f"-{calculator.config.collision_penalty}",
            "Applied immediately when a collision is reported",
        ),
        _row(
            "completion_bonus",
            f"+{calculator.config.completion_bonus}",
            "Sparse bonus awarded once a command objective completes",
        ),
        _row(
            "idle_penalty",
            f"-{calculator.config.idle_penalty_coef} (speed < {calculator.config.idle_threshold_mps})",
            "Discourages idling when the agent could make progress",
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
