from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import pytest

from airsim_env.pool import EnvironmentPool
from config.experiment import ExperimentDefinition, Pose, SeedBundle
from config.loader import compute_config_hash


@dataclass
class DummyEnv:
    experiment: ExperimentDefinition

    def reset(self) -> None:  # pragma: no cover - placeholder for interface compatibility
        return None


@pytest.fixture()
def base_experiment() -> ExperimentDefinition:
    seeds = SeedBundle(python=1, numpy=2, torch=3, airsim=4, deterministic=True)
    definition = ExperimentDefinition(
        id=uuid4(),
        scene="Neighborhood",
        vehicle="DefaultSedan",
        start_pose=Pose(x=0.0, y=0.0, z=0.0, yaw=0.0),
        goal_pose=Pose(x=10.0, y=0.0, z=0.0, yaw=0.0),
        horizon=50,
        seeds=seeds,
        weather_profile="clear",
        config_hash=None,
    )
    return definition.model_copy(update={"config_hash": compute_config_hash(definition)})


def test_environment_pool_derives_unique_seeds(base_experiment: ExperimentDefinition) -> None:
    pool = EnvironmentPool.build(
        size=2,
        base_experiment=base_experiment,
        factory=lambda experiment, index: (DummyEnv(experiment=experiment), None),
        seed_stride=1000,
    )

    with pool.acquire() as first, pool.acquire() as second:
        assert first.index != second.index
        assert first.experiment.seeds.python == base_experiment.seeds.python
        assert second.experiment.seeds.python != base_experiment.seeds.python
        assert second.experiment.seeds.python != first.experiment.seeds.python


def test_environment_pool_release_allows_reuse(base_experiment: ExperimentDefinition) -> None:
    pool = EnvironmentPool.build(
        size=1,
        base_experiment=base_experiment,
        factory=lambda experiment, index: (DummyEnv(experiment=experiment), None),
    )

    with pool.acquire() as first:
        index = first.index
        env_id = id(first.env)

    with pool.acquire() as second:
        assert second.index == index
        assert id(second.env) == env_id


def test_environment_pool_timeout(base_experiment: ExperimentDefinition) -> None:
    pool = EnvironmentPool.build(
        size=1,
        base_experiment=base_experiment,
        factory=lambda experiment, index: (DummyEnv(experiment=experiment), None),
    )

    lease = pool.acquire()
    with lease:
        with pytest.raises(TimeoutError):
            pool.acquire(timeout=0.01)
    lease.close()
