from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest

from hrl_agent.coordination import CommandCoordinator
from hrl_agent.manager.dqn_manager import CommandPolicy, DQNManager
from hrl_agent.orchestrator import HRLOrchestrator
from hrl_agent.workers.sac_worker import SACWorker


class ScalingDQNManager(DQNManager):
    __test__ = False

    def __init__(self, commands: list[str]) -> None:
        model = SimpleNamespace(observation_space=SimpleNamespace(shape=(5,)))
        super().__init__(policy=CommandPolicy(commands=commands), model=model)

    def select_command(self, observation, *, deterministic: bool = True):  # type: ignore[override]
        return self.policy.command_from_index(0)


class ScalingSACWorker(SACWorker):
    __test__ = False

    def __init__(self, action: dict[str, float]) -> None:
        model = SimpleNamespace(observation_space=SimpleNamespace(shape=(5,)))
        super().__init__(model=model)
        self._action = action

    def act(self, observation, *, deterministic: bool = True):  # type: ignore[override]
        return self._action


@dataclass
class SlowEnv:
    env_id: str
    step_delay: float
    distance: float = 12.0
    steps: int = 0

    def reset(self):
        self.distance = 12.0
        self.steps = 0
        return {
            "telemetry": {
                "distance_to_goal": self.distance,
                "speed_mps": 1.0,
                "lane_mask_coverage_ratio": 1.0,
                "progress_possible": True,
                "heading_deg": 0.0,
            }
        }

    def set_command(self, command: str, *, completed: bool = False) -> None:
        self._last_command = (command, completed)

    def step(self, action: dict[str, float]):
        time.sleep(self.step_delay)
        self.steps += 1
        self.distance = max(0.0, self.distance - 6.0)
        terminated = self.distance <= 0.0
        observation = {
            "telemetry": {
                "distance_to_goal": self.distance,
                "speed_mps": 1.0,
                "lane_mask_coverage_ratio": 1.0,
                "progress_possible": True,
                "heading_deg": 0.0,
            }
        }
        info = {"env_id": self.env_id, "steps": self.steps}
        return observation, 1.0, terminated, False, info


@pytest.mark.integration
def test_parallel_throughput_matches_or_exceeds_baseline() -> None:
    commands = ["FOLLOW_LANE"]
    worker_action = {"throttle": 0.6, "brake": 0.0, "steering": 0.0}

    baseline_manager = ScalingDQNManager(commands)
    baseline_worker = ScalingSACWorker(worker_action)
    baseline_coordinator = CommandCoordinator(baseline_manager, {"FOLLOW_LANE": baseline_worker})

    seq_env = SlowEnv("sequential", step_delay=0.01)
    baseline_orchestrator = HRLOrchestrator(seq_env, baseline_coordinator)

    sequential_steps = 0
    sequential_time = 0.0
    for _ in range(2):
        start = time.perf_counter()
        result = baseline_orchestrator.run_episode()
        sequential_time += time.perf_counter() - start
        sequential_steps += result.steps

    parallel_manager = ScalingDQNManager(commands)
    parallel_worker = ScalingSACWorker(worker_action)
    parallel_coordinator = CommandCoordinator(parallel_manager, {"FOLLOW_LANE": parallel_worker})

    env_a = SlowEnv("env_a", step_delay=0.01)
    env_b = SlowEnv("env_b", step_delay=0.01)
    parallel_orchestrator = HRLOrchestrator(env_a, parallel_coordinator)

    start = time.perf_counter()
    parallel_results = parallel_orchestrator.run_parallel(
        {"env_a": cast(Any, env_a), "env_b": cast(Any, env_b)}
    )
    parallel_time = time.perf_counter() - start
    parallel_steps = sum(result.steps for result in parallel_results.values())

    baseline_rate = sequential_steps / sequential_time
    parallel_rate = parallel_steps / parallel_time

    assert parallel_steps == sequential_steps
    assert parallel_rate >= baseline_rate * 0.9
