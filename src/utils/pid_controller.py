"""Simple configurable PID controller for cascaded control loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


def _clamp(value: float, limits: Tuple[Optional[float], Optional[float]]) -> float:
    lower, upper = limits
    if lower is not None and value < lower:
        return lower
    if upper is not None and value > upper:
        return upper
    return value


@dataclass
class PIDController:
    """Discrete PID controller with optional output limiting."""

    kp: float
    ki: float
    kd: float
    output_limits: Tuple[Optional[float], Optional[float]] = (None, None)
    integral_limits: Tuple[Optional[float], Optional[float]] = (None, None)
    default_dt: float = 1.0

    def __post_init__(self) -> None:
        if self.default_dt <= 0:
            raise ValueError("default_dt must be positive")
        self._integral = 0.0
        self._prev_error: Optional[float] = None

    def reset(self) -> None:
        """Clear accumulated integral and derivative history."""

        self._integral = 0.0
        self._prev_error = None

    def update(self, target: float, current: float, *, dt: Optional[float] = None) -> float:
        """Compute the next control signal for the provided setpoint."""

        error = target - current
        step_dt = dt if dt is not None and dt > 0 else self.default_dt

        self._integral += error * step_dt
        self._integral = _clamp(self._integral, self.integral_limits)

        derivative = 0.0
        if self._prev_error is not None:
            derivative = (error - self._prev_error) / step_dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return _clamp(output, self.output_limits)
