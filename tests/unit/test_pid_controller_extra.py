import pytest

from utils.pid_controller import PIDController, _clamp


def test_clamp_limits_apply():
    assert _clamp(-2.0, (0.0, 1.0)) == 0.0
    assert _clamp(2.0, (0.0, 1.0)) == 1.0
    assert _clamp(0.5, (0.0, 1.0)) == 0.5


def test_pid_controller_rejects_non_positive_dt():
    with pytest.raises(ValueError):
        PIDController(kp=1.0, ki=0.0, kd=0.0, default_dt=0.0)
