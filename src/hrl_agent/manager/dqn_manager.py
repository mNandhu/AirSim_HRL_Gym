"""Wrapper around a stable-baselines3 DQN policy for high-level command selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

try:  # pragma: no cover - optional when gymnasium is unavailable
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None  # type: ignore
    spaces = None  # type: ignore

try:  # pragma: no cover - optional when stable-baselines3 is unavailable
    from stable_baselines3 import DQN
    from stable_baselines3.common.logger import configure as configure_logger
except ImportError:  # pragma: no cover
    DQN = None  # type: ignore
    configure_logger = None  # type: ignore

__all__ = ["CommandPolicy", "DQNManager"]


@dataclass(frozen=True)
class CommandPolicy:
    commands: Sequence[str]

    def command_from_index(self, index: int) -> str:
        return self.commands[index]

    def index_of(self, command: str) -> int:
        try:
            return self.commands.index(command)
        except ValueError as exc:  # pragma: no cover - defensive path
            raise KeyError(f"Unknown command {command!r}") from exc


class DQNManager:
    """High-level manager that chooses discrete commands via a DQN policy."""

    def __init__(
        self,
        policy: CommandPolicy,
        *,
        model: Any | None = None,
        observation_space: Any | None = None,
        action_space: Any | None = None,
        learning_rate: float = 3e-4,
        buffer_size: int = 50_000,
        learning_starts: int = 128,
        log_formats: Sequence[str] | None = None,
        default_log_dir: Path | str | None = None,
    ) -> None:
        self._policy = policy
        self._model = model
        self._buffer_size = buffer_size
        self._learning_rate = learning_rate
        self._learning_starts = learning_starts
        self._steps = 0
        self.on_learning_start: Optional[Callable[[dict], None]] = None
        self._learner_started_logged = False
        self._log_formats = tuple(log_formats) if log_formats else ("stdout", "csv")
        self._default_log_dir = (
            Path(default_log_dir)
            if default_log_dir is not None
            else Path("artifacts") / "sb3" / "dqn_manager"
        )
        if spaces is not None:
            self._observation_space = observation_space or spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(5,),
                dtype=np.float32,
            )
            self._action_space = action_space or spaces.Discrete(len(policy.commands))
        else:  # pragma: no cover - gymnasium missing
            self._observation_space = observation_space
            self._action_space = action_space

    @property
    def policy(self) -> CommandPolicy:
        return self._policy

    @property
    def model(self) -> Any | None:
        return self._model

    def select_command(self, observation: Any, *, deterministic: bool = True) -> str:
        if self._model is None:
            # Default to first command if no model is attached yet.
            return self._policy.command_from_index(0)
        action, _ = self._model.predict(observation, deterministic=deterministic)
        return self._policy.command_from_index(int(action))

    def attach_model(self, model: Any) -> None:
        self._model = model
        self._steps = 0

    def load_from_path(self, path: str, *, log_dir: Path | str | None = None) -> None:
        if DQN is None:
            raise RuntimeError("stable-baselines3 is required to load DQN models")
        self._model = DQN.load(path)
        self._configure_logger(log_dir)
        self._steps = 0

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("Cannot save before attaching a model")
        self._model.save(path)

    def set_policy(self, policy: CommandPolicy) -> None:
        object.__setattr__(self, "_policy", policy)

    def build_default_model(self, *, log_dir: Path | str | None = None) -> None:
        if DQN is None:
            raise RuntimeError("stable-baselines3 is required to initialize DQN models")
        if gym is None or spaces is None:
            raise RuntimeError("gymnasium is required to initialize DQN models")

        class _StaticEnv(gym.Env):  # type: ignore[misc]
            metadata = {"render_modes": []}

            def __init__(self, observation_space: Any, action_space: Any) -> None:
                super().__init__()
                self.observation_space = observation_space
                self.action_space = action_space

            def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
                super().reset(seed=seed)
                shape = getattr(self.observation_space, "shape", (1,))
                return np.zeros(shape, dtype=np.float32), {}

            def step(self, action):  # type: ignore[override]
                shape = getattr(self.observation_space, "shape", (1,))
                obs = np.zeros(shape, dtype=np.float32)
                reward = 0.0
                terminated = True
                truncated = False
                info: dict[str, Any] = {}
                return obs, reward, terminated, truncated, info

        env = _StaticEnv(self._observation_space, self._action_space)
        self._model = DQN(
            "MlpPolicy",
            env,
            learning_rate=self._learning_rate,
            buffer_size=self._buffer_size,
            learning_starts=self._learning_starts,
            train_freq=1,
            gradient_steps=1,
            verbose=0,
        )
        self._configure_logger(log_dir)
        self._steps = 0

    def _configure_logger(self, log_dir: Path | str | None) -> None:
        if configure_logger is None or self._model is None:
            return

        target_dir = Path(log_dir) if log_dir is not None else self._default_log_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        logger = configure_logger(str(target_dir), list(self._log_formats))
        self._model.set_logger(logger)
        self._default_log_dir = target_dir

    def process_experience(
        self,
        observation: np.ndarray,
        action_index: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        if self._model is None or getattr(self._model, "replay_buffer", None) is None:
            return

        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        next_obs = np.asarray(next_observation, dtype=np.float32).reshape(1, -1)
        action = np.array([[action_index]], dtype=np.int64)
        reward_arr = np.array([reward], dtype=np.float32)
        done_arr = np.array([done], dtype=np.bool_)
        infos: list[dict[str, Any]] = [{}]

        replay_buffer = getattr(self._model, "replay_buffer", None)
        if replay_buffer is None:
            return

        replay_buffer.add(obs, next_obs, action, reward_arr, done_arr, infos)
        self._steps += 1

        # Fire one-time learning-start event at the first training step
        threshold = getattr(self._model, "learning_starts", self._learning_starts)
        if not self._learner_started_logged and self._steps >= threshold:
            self._learner_started_logged = True
            if callable(self.on_learning_start):
                try:
                    self.on_learning_start(
                        {
                            "event": "learner_started",
                            "algo": "dqn",
                            "step": self._steps,
                            "learning_starts": int(threshold),
                            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                        }
                    )
                except Exception:
                    pass

        if self._steps >= threshold:
            batch_size = getattr(self._model, "batch_size", 64)
            self._model.train(batch_size=batch_size, gradient_steps=1)

            # Update SB3's internal timestep counters for proper logging
            if hasattr(self._model, "_total_timesteps"):
                self._model._total_timesteps += 1  # type: ignore[attr-defined]
            if hasattr(self._model, "_n_updates"):
                self._model._n_updates += 1  # type: ignore[attr-defined]

            # Trigger SB3 logging every 100 training steps to populate progress.csv
            if self._steps % 100 == 0:
                if hasattr(self._model, "logger") and self._model.logger is not None:
                    # Log training metrics that SB3 typically tracks
                    self._model.logger.record("time/total_timesteps", self._steps)
                    self._model.logger.record("rollout/ep_len_mean", self._steps)

                    # Log recent reward if we have access to it
                    if reward_arr.size > 0:
                        self._model.logger.record("train/reward", float(reward_arr[0]))

                    # Log command selection metrics for DQN
                    self._model.logger.record("train/command_selection_step", self._steps)

                    # Dump the logs to CSV file
                    self._model.logger.dump(step=self._steps)
