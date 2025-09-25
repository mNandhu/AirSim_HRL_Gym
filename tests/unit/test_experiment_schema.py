from datetime import datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from config.experiment import ExperimentDefinition, Pose, SeedBundle


def _valid_payload() -> dict:
    return {
        "scene": "Neighborhood",
        "vehicle": "DefaultSedan",
        "start_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        "goal_pose": {"x": 10.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        "horizon": 200,
        "seeds": {
            "python": 1,
            "numpy": 2,
            "torch": 3,
            "airsim": 4,
            "deterministic": True,
        },
    }


def test_experiment_definition_defaults():
    definition = ExperimentDefinition(**_valid_payload())

    assert isinstance(definition.id, UUID)
    assert isinstance(definition.generated_at, datetime)
    assert definition.horizon == 200
    assert isinstance(definition.start_pose, Pose)
    assert isinstance(definition.seeds, SeedBundle)


def test_horizon_must_be_positive():
    payload = _valid_payload()
    payload["horizon"] = 0
    with pytest.raises(ValidationError):
        ExperimentDefinition(**payload)


def test_config_hash_must_be_valid_sha256():
    payload = _valid_payload()
    payload["config_hash"] = "abc"
    with pytest.raises(ValidationError):
        ExperimentDefinition(**payload)

    payload["config_hash"] = "g" * 64
    with pytest.raises(ValidationError):
        ExperimentDefinition(**payload)


def test_seed_bundle_requires_non_negative_values():
    with pytest.raises(ValidationError):
        SeedBundle(python=-1, numpy=0, torch=0, airsim=0)
