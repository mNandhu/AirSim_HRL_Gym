import json
from pathlib import Path

import pytest

from config.experiment import ExperimentDefinition, Pose, SeedBundle
from config.loader import compute_config_hash, load_experiment
from config.settings_hash import compute_settings_hash


@pytest.fixture
def sample_experiment(tmp_path: Path) -> Path:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
scene: Neighborhood
vehicle: DefaultSedan
start_pose:
  x: 0
  y: 0
  z: 0
  yaw: 0
goal_pose:
  x: 10
  y: 0
  z: 0
  yaw: 0
horizon: 3
seeds:
  python: 1
  numpy: 2
  torch: 3
  airsim: 4
  deterministic: true
""",
        encoding="utf-8",
    )
    return config_path


def test_load_experiment_assigns_hash(sample_experiment: Path):
    experiment = load_experiment(sample_experiment)
    assert isinstance(experiment, ExperimentDefinition)
    assert experiment.config_hash is not None


def test_compute_config_hash_stable():
    experiment = ExperimentDefinition(
        scene="Neighborhood",
        vehicle="DefaultSedan",
        start_pose=Pose(x=0, y=0, z=0, yaw=0),
        goal_pose=Pose(x=10, y=0, z=0, yaw=0),
        horizon=3,
        seeds=SeedBundle(python=1, numpy=2, torch=3, airsim=4, deterministic=True),
    )
    hash1 = compute_config_hash(experiment)
    hash2 = compute_config_hash(experiment.model_copy())
    assert hash1 == hash2


def test_compute_settings_hash_from_mapping(tmp_path: Path):
    mapping = {"Vehicles": {"Car": {"AutoCreate": True}}}
    file_path = tmp_path / "settings.json"
    file_path.write_text(json.dumps(mapping), encoding="utf-8")

    from_file = compute_settings_hash(file_path)
    from_mapping = compute_settings_hash(mapping)

    assert from_file == from_mapping
