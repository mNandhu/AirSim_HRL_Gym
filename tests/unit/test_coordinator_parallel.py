from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, cast

import pytest

from hrl_agent.coordination import CommandCoordinator
from hrl_agent.manager.dqn_manager import CommandPolicy, DQNManager
from hrl_agent.orchestrator import HRLOrchestrator
from hrl_agent.workers.sac_worker import SACWorker


class _DummyDQNModel:
    def __init__(self) -> None:
        self.observation_space = SimpleNamespace(shape=(5,))

    def predict(self, observation, deterministic: bool = True):  # noqa: D401
        return 0, {}


class TestDQNManager(DQNManager):
    __test__ = False

    def __init__(self, commands: list[str], schedule: list[str] | None = None) -> None:
        model = _DummyDQNModel()
        super().__init__(policy=CommandPolicy(commands=commands), model=model)
        self._schedule = list(schedule or commands)
        self.captured_experiences: list[tuple[Any, int, float, Any, bool]] = []

    def select_command(self, observation: Any, *, deterministic: bool = True) -> str:  # type: ignore[override]
        if self._schedule:
            return self._schedule.pop(0)
        return self.policy.command_from_index(0)

    def process_experience(  # type: ignore[override]
        self,
        observation: Any,
        action_index: int,
        reward: float,
        next_observation: Any,
        done: bool,
    ) -> None:
        self.captured_experiences.append(
            (observation, action_index, reward, next_observation, done)
        )


class TestSACWorker(SACWorker):
    __test__ = False

    def __init__(self, action: Dict[str, float]) -> None:
        model = SimpleNamespace(observation_space=SimpleNamespace(shape=(5,)))
        super().__init__(model=model)
        self._action = action
        self.captured_experiences: list[tuple[Any, Any, float, Any, bool]] = []

    def act(self, observation: Any, *, deterministic: bool = True) -> Dict[str, float]:  # type: ignore[override]
        return self._action

    def process_experience(  # type: ignore[override]
        self,
        observation: Any,
        action: Any,
        reward: float,
        next_observation: Any,
        done: bool,
    ) -> None:
        self.captured_experiences.append((observation, action, reward, next_observation, done))


@dataclass
class DummyEnv:
    env_id: str
    steps: int = 0
    distance: float = 12.0

    def reset(self):
        self.steps = 0
        self.distance = 12.0
        return {
            "telemetry": {
                "distance_to_goal": self.distance,
                "speed_mps": 1.0,
                "lane_mask_coverage_ratio": 1.0,
                "progress_possible": True,
                "heading_deg": 0.0,
            },
            "reward_components": {},
        }

    def set_command(self, command: str, *, completed: bool = False) -> None:
        # Track last command for verification in tests
        self._last_command = (command, completed)

    def step(self, action: Dict[str, float]):
        self.steps += 1
        self.distance = max(0.0, self.distance - 6.0)
        observation = {
            "telemetry": {
                "distance_to_goal": self.distance,
                "speed_mps": 1.0,
                "lane_mask_coverage_ratio": 1.0,
                "progress_possible": True,
                "heading_deg": 0.0,
            },
            "reward_components": {},
        }
        terminated = self.distance <= 0.0
        info = {"env_id": self.env_id, "steps": self.steps}
        return observation, 1.0, terminated, False, info

    def close(self) -> None:  # pragma: no cover - provided for completeness
        pass


def make_observation(distance: float, speed: float = 1.0) -> dict[str, Any]:
    return {
        "telemetry": {
            "distance_to_goal": distance,
            "speed_mps": speed,
            "lane_mask_coverage_ratio": 1.0,
            "progress_possible": True,
            "heading_deg": 0.0,
        }
    }


def test_act_batch_tracks_per_environment_state():
    manager = TestDQNManager(["FOLLOW_LANE", "STOP"], schedule=["FOLLOW_LANE", "STOP"])
    worker_a = TestSACWorker({"throttle": 0.5, "brake": 0.0, "steering": 0.0})
    worker_b = TestSACWorker({"throttle": 0.0, "brake": 1.0, "steering": 0.0})
    coordinator = CommandCoordinator(manager, {"FOLLOW_LANE": worker_a, "STOP": worker_b})

    observations = {
        "env_a": make_observation(10.0),
        "env_b": make_observation(3.0, speed=0.05),
    }

    actions = coordinator.act_batch(observations)

    assert actions["env_a"][0] == "FOLLOW_LANE"
    assert actions["env_b"][0] == "STOP"

    observations["env_a"] = make_observation(4.0)
    assert coordinator.command_completed_for_env("env_a", observations["env_a"]) is True
    assert coordinator.command_completed_for_env("env_b", observations["env_b"]) is True


def test_observe_transition_for_env_routes_experience():
    manager = TestDQNManager(["FOLLOW_LANE"], schedule=["FOLLOW_LANE"])
    worker = TestSACWorker({"throttle": 0.5, "brake": 0.0, "steering": 0.0})
    coordinator = CommandCoordinator(manager, {"FOLLOW_LANE": worker})

    observation = make_observation(10.0)
    coordinator.act_for_env("env_a", observation)

    next_observation = make_observation(8.0)
    coordinator.observe_transition_for_env(
        "env_a",
        observation,
        1.0,
        next_observation,
        done=False,
    )

    assert len(manager.captured_experiences) == 1
    assert len(worker.captured_experiences) == 1


def test_run_parallel_executes_all_environments():
    manager = TestDQNManager(
        ["FOLLOW_LANE"], schedule=["FOLLOW_LANE", "FOLLOW_LANE", "FOLLOW_LANE"]
    )
    worker = TestSACWorker({"throttle": 0.7, "brake": 0.0, "steering": 0.0})
    coordinator = CommandCoordinator(manager, {"FOLLOW_LANE": worker})

    env_a = DummyEnv("env_a")
    env_b = DummyEnv("env_b")
    orchestrator = HRLOrchestrator(cast(Any, env_a), coordinator)

    results = orchestrator.run_parallel({"env_a": cast(Any, env_a), "env_b": cast(Any, env_b)})

    assert set(results.keys()) == {"env_a", "env_b"}
    assert results["env_a"].steps == 2
    assert results["env_b"].steps == 2
    assert results["env_a"].cumulative_reward == pytest.approx(2.0)
    assert results["env_b"].cumulative_reward == pytest.approx(2.0)

    assert env_a._last_command[0] == "FOLLOW_LANE"
    assert env_b._last_command[0] == "FOLLOW_LANE"
