"""Coordinator that bridges the DQN manager with SAC workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .manager.dqn_manager import CommandPolicy, DQNManager
from .workers.sac_worker import SACWorker

__all__ = ["CommandCoordinator", "CoordinatorState"]


@dataclass
class CoordinatorState:
    command: str | None = None
    last_action: Mapping[str, float] | None = None


class CommandCoordinator:
    """Route high-level commands to the appropriate low-level worker."""

    def __init__(
        self,
        manager: DQNManager,
        workers: Mapping[str, SACWorker],
    ) -> None:
        if not workers:
            raise ValueError("At least one worker must be provided")
        self._manager = manager
        self._workers = dict(workers)
        self._state = CoordinatorState()
        self._schedule = list(self._workers.keys())
        self._schedule_index = 0

    @property
    def state(self) -> CoordinatorState:
        return self._state

    def act(
        self, observation: Any, *, deterministic: bool = True
    ) -> tuple[str, Mapping[str, float]]:
        command = self._manager.select_command(observation, deterministic=deterministic)
        worker = self._workers.get(command)
        if worker is None:
            # Default to the first worker if specific command missing
            worker = next(iter(self._workers.values()))
        action = worker.act(observation, deterministic=deterministic)
        self._state = CoordinatorState(command=command, last_action=action)
        return command, action

    def update_command_policy(self, policy: CommandPolicy) -> None:
        self._manager.set_policy(policy)

    def register_worker(self, command: str, worker: SACWorker) -> None:
        self._workers[command] = worker
        self._schedule = list(self._workers.keys())
        self._schedule_index = 0

    def act_sequential(
        self, observation: Any, *, deterministic: bool = True
    ) -> tuple[str, Mapping[str, float]]:
        if not self._schedule:
            self._schedule = list(self._workers.keys())
        command = self._schedule[self._schedule_index % len(self._schedule)]
        self._schedule_index += 1
        worker = self._workers[command]
        action = worker.act(observation, deterministic=deterministic)
        self._state = CoordinatorState(command=command, last_action=action)
        return command, action
