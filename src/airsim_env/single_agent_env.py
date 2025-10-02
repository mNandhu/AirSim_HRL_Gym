"""Gymnasium-compatible single-agent wrapper around :class:`AirSimEnv`."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

try:  # pragma: no cover - exercised in training script environments
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - surfaced to callers
    raise ImportError(
        "gymnasium is required for the single-agent environment. Install with: pip install gymnasium"
    ) from exc

from airsim_env.env import AirSimEnv
from airsim_env.observation import ObservationPacket
from airsim_env.reward import RewardCalculator
from config.experiment import ExperimentDefinition

__all__ = ["SingleAgentAirSimEnv"]


class SingleAgentAirSimEnv(gym.Env):  # type: ignore[misc]
    """Expose AirSim as a flat continuous-control RL environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        experiment: ExperimentDefinition,
        *,
        simulator: Any,
        perception: Any,
        reward_calculator: RewardCalculator | None = None,
        target_speed_range: tuple[float, float] | None = None,
        speed_pid_config: Mapping[str, Any] | None = None,
        steering_pid_config: Mapping[str, Any] | None = None,
        distance_normalizer: float = 50.0,
    ) -> None:
        super().__init__()
        min_speed, max_speed = target_speed_range or (0.0, 5.0)
        if max_speed <= min_speed:
            raise ValueError("target_speed_range must have max > min")

        self._min_speed = float(min_speed)
        self._max_speed = float(max_speed)
        self._distance_scale = float(distance_normalizer)
        self._experiment = experiment

        self._env = AirSimEnv(
            experiment,
            simulator=simulator,
            perception=perception,
            reward_calculator=reward_calculator,
            speed_pid_config=speed_pid_config,
            steering_pid_config=steering_pid_config,
            target_speed_range=(self._min_speed, self._max_speed),
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "telemetry": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(8,),
                    dtype=np.float32,
                )
            }
        )

        self._last_info: dict[str, Any] = {}
        self._last_observation: ObservationPacket | None = None
        self._last_action: dict[str, Any] = {
            "target_speed": self._min_speed,
            "target_steering": 0.0,
            "normalized": [0.0, 0.0],
        }

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        seed_bundle = options.get("seed_bundle") if options else None
        packet = self._env.reset(seed_bundle)
        self._last_observation = packet
        obs = self._encode_observation(packet)
        info = self._build_reset_info(packet)
        self._last_info = info
        return obs, info

    def step(self, action):  # type: ignore[override]
        target_speed, target_steering, normalized = self._scale_action(action)
        command = {
            "target_speed": target_speed,
            "target_steering": target_steering,
        }
        packet, reward, terminated, truncated, env_info = self._env.step(command)
        self._last_observation = packet

        obs = self._encode_observation(packet)
        info = self._build_step_info(
            env_info,
            packet,
            target_speed,
            target_steering,
            normalized,
        )
        self._last_info = info
        return obs, reward, terminated, truncated, info

    def render(self):  # type: ignore[override]
        if self._last_observation is None:
            return None
        return self._last_observation.image

    def close(self) -> None:  # type: ignore[override]
        self._env.close()
        self._last_observation = None
        self._last_info = {}

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @property
    def experiment(self) -> ExperimentDefinition:
        return self._experiment

    @property
    def last_info(self) -> dict[str, Any]:
        return self._last_info

    def get_wrapped_env(self) -> AirSimEnv:
        return self._env

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_reset_info(self, packet: ObservationPacket) -> dict[str, Any]:
        telemetry = dict(packet.telemetry)
        reward_components = dict(packet.reward_components)
        done_flags = dict(packet.done_flags)

        info = {
            "episode_step": 0,
            "telemetry": telemetry,
            "reward_components": reward_components,
            "done_flags": done_flags,
            "goal_reached": False,
            "control_action": getattr(self._env, "_last_control_action", {}).copy(),
            "setpoint": {
                "target_speed": self._min_speed,
                "target_steering": 0.0,
            },
            "applied_action": self._last_action.copy(),
            "config_hash": getattr(self._env, "_config_hash", None),
            "raw_observation": packet.to_dict(),
            "observation_vector": self._encode_observation(packet)["telemetry"],
        }
        return info

    def _build_step_info(
        self,
        env_info: Mapping[str, Any],
        packet: ObservationPacket,
        target_speed: float,
        target_steering: float,
        normalized: np.ndarray,
    ) -> dict[str, Any]:
        info = dict(env_info)
        telemetry = dict(packet.telemetry)
        reward_components = dict(packet.reward_components)
        done_flags = dict(packet.done_flags)

        info.update(
            {
                "telemetry": telemetry,
                "reward_components": reward_components,
                "done_flags": done_flags,
                "goal_reached": done_flags.get("goal_reached", False),
                "raw_observation": packet.to_dict(),
                "observation_vector": self._encode_observation(packet)["telemetry"],
            }
        )

        setpoint = info.get("setpoint", {})
        setpoint.update({"target_speed": target_speed, "target_steering": target_steering})
        info["setpoint"] = setpoint

        applied = {
            "target_speed": target_speed,
            "target_steering": target_steering,
            "normalized": normalized.astype(np.float32).tolist(),
        }
        info["applied_action"] = applied
        self._last_action = applied
        return info

    def _encode_observation(self, packet: ObservationPacket) -> dict[str, np.ndarray]:
        telemetry = packet.telemetry
        distance = float(
            telemetry.get(
                "distance_to_current_waypoint",
                telemetry.get("distance_to_goal", 0.0),
            )
        )
        distance_norm = float(np.tanh(distance / max(self._distance_scale, 1e-6)))

        speed = float(telemetry.get("speed_mps", 0.0))
        speed_norm = float(np.clip(speed / max(self._max_speed, 1e-6), 0.0, 1.0))

        lane_ratio = float(telemetry.get("lane_mask_coverage_ratio", 1.0))
        lane_centering = float(np.clip(2.0 * lane_ratio - 1.0, -1.0, 1.0))

        collision_flag = 1.0 if telemetry.get("collision", False) else 0.0
        progress_flag = 1.0 if telemetry.get("progress_possible", True) else 0.0

        heading_deg = float(telemetry.get("heading_deg", 0.0))
        heading_rad = math.radians(heading_deg)
        sin_heading = float(np.sin(heading_rad))
        cos_heading = float(np.cos(heading_rad))

        current_idx = telemetry.get("current_waypoint_index")
        total_waypoints = telemetry.get("total_waypoints")
        progress_ratio = 0.0
        if isinstance(current_idx, (int, float)) and isinstance(total_waypoints, (int, float)):
            total = max(int(total_waypoints) - 1, 1)
            index = min(max(int(current_idx), 0), int(total_waypoints))
            progress_ratio = float(np.clip(index / total, 0.0, 1.0))

        features = np.array(
            [
                distance_norm,
                speed_norm,
                lane_centering,
                collision_flag,
                progress_flag,
                sin_heading,
                cos_heading,
                progress_ratio,
            ],
            dtype=np.float32,
        )
        return {"telemetry": features}

    def _scale_action(self, action: Any) -> tuple[float, float, np.ndarray]:
        if isinstance(action, Mapping):
            action = [
                float(action.get("target_speed", action.get("speed", 0.0))),
                float(action.get("target_steering", action.get("steering", 0.0))),
            ]

        array = np.asarray(action, dtype=np.float32).reshape(-1)

        if array.size < 2:
            raise ValueError("Action must contain exactly two elements: [speed, steering]")

        normalized = np.clip(array[:2], -1.0, 1.0)
        speed_norm = float(normalized[0])
        steering_norm = float(normalized[1])

        speed_span = self._max_speed - self._min_speed
        target_speed = self._min_speed + 0.5 * (speed_norm + 1.0) * speed_span
        target_speed = float(np.clip(target_speed, self._min_speed, self._max_speed))
        target_steering = float(np.clip(steering_norm, -1.0, 1.0))

        return target_speed, target_steering, normalized
