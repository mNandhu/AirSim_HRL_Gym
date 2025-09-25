import os
import random
from types import SimpleNamespace

import pytest

from config import seeds as seed_module
from config.experiment import SeedBundle
from config.seeds import SeedApplicationError, apply_seed_bundle


class DummyTorch:
    def __init__(self) -> None:
        self.manual_seed_value = None
        self.manual_seed_all_value = None
        self.deterministic_algorithms = False
        self.cuda = SimpleNamespace(
            is_available=lambda: False,
            manual_seed_all=self._manual_seed_all,
        )
        self.backends = SimpleNamespace(cudnn=SimpleNamespace(deterministic=None, benchmark=None))

    def manual_seed(self, value: int) -> None:
        self.manual_seed_value = value

    def use_deterministic_algorithms(self, flag: bool) -> None:
        self.deterministic_algorithms = flag

    def _manual_seed_all(self, value: int) -> None:
        self.manual_seed_all_value = value


@pytest.fixture(autouse=True)
def cleanup_env(monkeypatch):
    monkeypatch.delenv("AIRSIM_RANDOM_SEED", raising=False)
    monkeypatch.delenv("AIRSIM_DETERMINISTIC", raising=False)
    yield
    monkeypatch.delenv("AIRSIM_RANDOM_SEED", raising=False)
    monkeypatch.delenv("AIRSIM_DETERMINISTIC", raising=False)


def test_apply_seed_bundle_sets_all_rngs(monkeypatch):
    bundle = SeedBundle(python=123, numpy=456, torch=789, airsim=101112, deterministic=True)

    numpy_seed = {}
    dummy_np = SimpleNamespace(
        random=SimpleNamespace(seed=lambda value: numpy_seed.setdefault("seed", value))
    )
    dummy_torch = DummyTorch()

    monkeypatch.setattr(seed_module, "np", dummy_np)
    monkeypatch.setattr(seed_module, "torch", dummy_torch)

    random.seed(999)
    apply_seed_bundle(bundle)

    random.seed(bundle.python)
    expected = [random.random(), random.random()]

    random.seed(999)
    apply_seed_bundle(bundle)
    actual = [random.random(), random.random()]

    assert actual == expected
    assert numpy_seed["seed"] == 456
    assert dummy_torch.manual_seed_value == 789
    assert dummy_torch.deterministic_algorithms is True


def test_apply_seed_bundle_sets_environment_variables():
    bundle = SeedBundle(python=1, numpy=2, torch=3, airsim=4, deterministic=False)
    apply_seed_bundle(bundle)

    assert os.environ["AIRSIM_RANDOM_SEED"] == "4"
    assert os.environ["AIRSIM_DETERMINISTIC"] == "0"


def test_apply_seed_bundle_uses_client_methods(monkeypatch):
    captured = {}

    class Client:
        def setRandomSeed(self, seed):  # noqa: N802 - mimic AirSim style
            captured["seed"] = seed

    bundle = SeedBundle(python=1, numpy=2, torch=3, airsim=5)
    numpy_dummy = SimpleNamespace(random=SimpleNamespace(seed=lambda _: None))
    monkeypatch.setattr(seed_module, "np", numpy_dummy)
    monkeypatch.setattr(seed_module, "torch", None)

    apply_seed_bundle(bundle, airsim_client=Client())

    assert captured["seed"] == 5


def test_apply_seed_bundle_requires_seed_api(monkeypatch):
    bundle = SeedBundle(python=1, numpy=2, torch=3, airsim=5)
    monkeypatch.setattr(seed_module, "np", None)
    monkeypatch.setattr(seed_module, "torch", None)

    class Client:
        pass

    with pytest.raises(SeedApplicationError):
        apply_seed_bundle(bundle, airsim_client=Client())
