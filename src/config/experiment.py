"""Pydantic models describing experiment configuration for reproducible rollouts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class Pose(BaseModel):
    """Simple pose representation for AirSim coordinates."""

    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    z: float
    yaw: float = Field(..., ge=-360.0, le=360.0)


class SeedBundle(BaseModel):
    """Collection of RNG seeds applied before environment resets."""

    model_config = ConfigDict(frozen=True)

    python: int
    numpy: int
    torch: int
    airsim: int
    deterministic: bool = True

    @field_validator("python", "numpy", "torch", "airsim")
    @classmethod
    def _check_seed(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Seeds must be non-negative integers")
        return value


class ExperimentDefinition(BaseModel):
    """Immutable experiment configuration describing a deterministic rollout."""

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    scene: str
    vehicle: str
    start_pose: Pose
    goal_pose: Optional[Pose] = None  # Deprecated: use waypoints instead
    waypoints: Optional[list[Pose]] = None  # Preferred: ordered path waypoints
    horizon: int = Field(..., gt=0)
    seeds: SeedBundle
    weather_profile: Optional[str] = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    config_hash: Optional[str] = None

    @field_validator("scene", "vehicle")
    @classmethod
    def _strip_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Field cannot be empty")
        return value

    @field_validator("config_hash")
    @classmethod
    def _validate_hash(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if len(value) != 64:
            raise ValueError("config_hash must be a 64-character SHA256 hex digest")
        int(value, 16)
        return value

    def model_post_init(self, __context) -> None:
        """Validate that either goal_pose or waypoints is provided."""
        if self.goal_pose is None and self.waypoints is None:
            raise ValueError("Either goal_pose or waypoints must be provided")
        if self.waypoints is not None and len(self.waypoints) < 2:
            raise ValueError("Waypoints list must contain at least 2 waypoints")


__all__ = ["Pose", "SeedBundle", "ExperimentDefinition", "ValidationError"]
