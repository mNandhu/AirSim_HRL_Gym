"""Observation data structures for the AirSim hierarchical environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


__all__ = ["ObservationPacket", "assemble_observation"]


@dataclass(frozen=True)
class ObservationPacket:
    """Container bundling perception outputs and telemetry for agents."""

    image: Any
    segmentation_mask: Any
    detections: list[dict[str, Any]]
    command: Any
    telemetry: Mapping[str, Any]
    reward_components: Mapping[str, float]
    done_flags: Mapping[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "segmentation_mask": self.segmentation_mask,
            "detections": self.detections,
            "command": self.command,
            "telemetry": dict(self.telemetry),
            "reward_components": dict(self.reward_components),
            "done_flags": dict(self.done_flags),
        }


def assemble_observation(
    *,
    image: Any,
    segmentation_mask: Any,
    detections: list[dict[str, Any]],
    telemetry: Mapping[str, Any],
    reward_components: Mapping[str, float],
    command: Any = None,
    done_flags: Mapping[str, bool] | None = None,
) -> ObservationPacket:
    done = done_flags or {"terminated": False, "truncated": False}
    return ObservationPacket(
        image=image,
        segmentation_mask=segmentation_mask,
        detections=detections,
        command=command,
        telemetry=telemetry,
        reward_components=reward_components,
        done_flags=done,
    )
