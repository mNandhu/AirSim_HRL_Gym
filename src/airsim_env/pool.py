"""Utilities for pooling AirSim environments for parallel training."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Lock
from typing import Any, Callable, Generic, TypeVar
from uuid import UUID, uuid4

from config.experiment import ExperimentDefinition, SeedBundle
from config.loader import compute_config_hash
from config.seeds import derive_seed_bundle

EnvT = TypeVar("EnvT")

EnvironmentFactory = Callable[[ExperimentDefinition, int], tuple[EnvT, Any | None]]

__all__ = [
    "EnvironmentFactory",
    "EnvironmentLease",
    "EnvironmentPool",
    "PooledEnvironment",
]


@dataclass(frozen=True)
class PooledEnvironment(Generic[EnvT]):
    """Container describing an environment instance managed by the pool."""

    env: EnvT
    experiment: ExperimentDefinition
    index: int
    seed_bundle: SeedBundle
    session: Any | None = None


class EnvironmentLease(Generic[EnvT]):
    """Context manager that returns a pooled environment and releases it on exit."""

    def __init__(self, pool: EnvironmentPool[EnvT], entry: PooledEnvironment[EnvT]) -> None:
        self._pool = pool
        self._entry = entry
        self._released = False

    def __enter__(self) -> PooledEnvironment[EnvT]:
        return self._entry

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001 - typing mirrors context protocol
        self.close()

    def close(self) -> None:
        if not self._released:
            self._pool._release(self._entry)
            self._released = True

    @property
    def released(self) -> bool:
        return self._released

    @property
    def entry(self) -> PooledEnvironment[EnvT]:
        return self._entry


class EnvironmentPool(Generic[EnvT]):
    """Manage a set of AirSim environments for parallel rollouts.

    The pool prepares a number of :class:`AirSimEnv` instances (or compatible wrappers)
    ahead of time and provides blocking acquisition semantics so that each simulator
    maintains isolated seed bundles. Callers should request a lease via :meth:`acquire`
    and use the context manager interface to ensure release.
    """

    def __init__(
        self,
        *,
        base_experiment: ExperimentDefinition,
        factory: EnvironmentFactory[EnvT],
        entries: list[PooledEnvironment[EnvT]],
        seed_stride: int,
    ) -> None:
        self._base_experiment = base_experiment
        self._factory = factory
        self._seed_stride = seed_stride
        self._entries: dict[int, PooledEnvironment[EnvT]] = {
            entry.index: entry for entry in entries
        }
        self._available: Queue[PooledEnvironment[EnvT]] = Queue()
        for entry in entries:
            self._available.put(entry)
        self._lock = Lock()
        self._in_use: set[int] = set()

    @classmethod
    def build(
        cls,
        *,
        size: int,
        base_experiment: ExperimentDefinition,
        factory: EnvironmentFactory[EnvT],
        seed_stride: int = 9973,
    ) -> "EnvironmentPool[EnvT]":
        if size <= 0:
            raise ValueError("size must be positive")

        normalized = base_experiment
        if base_experiment.config_hash is None:
            normalized = base_experiment.model_copy(
                update={"config_hash": compute_config_hash(base_experiment)}
            )

        entries: list[PooledEnvironment[EnvT]] = []
        for index in range(size):
            derived_seeds = derive_seed_bundle(normalized.seeds, offset=index, stride=seed_stride)
            experiment = _clone_experiment(normalized, derived_seeds, index)
            env, session = factory(experiment, index)
            entries.append(
                PooledEnvironment(
                    env=env,
                    experiment=experiment,
                    index=index,
                    seed_bundle=experiment.seeds,
                    session=session,
                )
            )

        return cls(
            base_experiment=normalized,
            factory=factory,
            entries=entries,
            seed_stride=seed_stride,
        )

    def __len__(self) -> int:
        return len(self._entries)

    def acquire(self, *, timeout: float | None = None) -> EnvironmentLease[EnvT]:
        try:
            entry = self._available.get(timeout=timeout)
        except Empty as exc:
            raise TimeoutError("No available environments in pool") from exc

        with self._lock:
            self._in_use.add(entry.index)
        return EnvironmentLease(self, entry)

    def mark_unhealthy(self, index: int) -> None:
        with self._lock:
            if index in self._in_use:
                raise RuntimeError("Cannot mark an environment as unhealthy while leased")
            if index not in self._entries:
                raise KeyError(f"Pool does not contain environment index {index}")

        removed = self._extract_from_queue(index)
        if removed is not None:
            _safe_close_entry(removed)
        new_entry = self._build_entry(index)
        with self._lock:
            self._entries[index] = new_entry
        self._available.put(new_entry)

    def _release(self, entry: PooledEnvironment[EnvT]) -> None:
        with self._lock:
            if entry.index not in self._in_use:
                return
            self._in_use.remove(entry.index)
        self._available.put(entry)

    def _build_entry(self, index: int) -> PooledEnvironment[EnvT]:
        seeds = derive_seed_bundle(
            self._base_experiment.seeds, offset=index, stride=self._seed_stride
        )
        experiment = _clone_experiment(self._base_experiment, seeds, index)
        env, session = self._factory(experiment, index)
        return PooledEnvironment(
            env=env,
            experiment=experiment,
            index=index,
            seed_bundle=experiment.seeds,
            session=session,
        )

    def _extract_from_queue(self, index: int) -> PooledEnvironment[EnvT] | None:
        drained: list[PooledEnvironment[EnvT]] = []
        removed: PooledEnvironment[EnvT] | None = None
        while True:
            try:
                entry = self._available.get_nowait()
            except Empty:
                break
            if entry.index == index and removed is None:
                removed = entry
                continue
            drained.append(entry)
        for entry in drained:
            self._available.put(entry)
        return removed


def _clone_experiment(
    base: ExperimentDefinition,
    seeds: SeedBundle,
    index: int,
) -> ExperimentDefinition:
    update: dict[str, Any] = {"seeds": seeds, "config_hash": None}
    if index != 0:
        update["id"] = uuid4()
    else:
        update["id"] = base.id if isinstance(base.id, UUID) else uuid4()
    clone = base.model_copy(update=update)
    return clone.model_copy(update={"config_hash": compute_config_hash(clone)})


def _safe_close_entry(entry: PooledEnvironment[Any]) -> None:
    closer = getattr(entry.env, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:  # pragma: no cover - defensive close
            pass

    session = entry.session
    process = getattr(session, "process", None) if session is not None else None
    if process is not None and hasattr(process, "terminate"):
        try:
            process.terminate()
        except Exception:  # pragma: no cover - defensive close
            pass
