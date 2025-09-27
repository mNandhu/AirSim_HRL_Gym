"""Utilities for applying deterministic seed bundles across libraries and AirSim."""

from __future__ import annotations

import os
import random
from typing import Any

from .experiment import SeedBundle

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency during bootstrap
    np = None  # type: ignore

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency during bootstrap
    torch = None  # type: ignore

try:
    import airsim
except ImportError:  # pragma: no cover - optional dependency during bootstrap
    airsim = None  # type: ignore


class SeedApplicationError(RuntimeError):
    """Raised when the AirSim client cannot accept a seed value."""


def apply_seed_bundle(bundle: SeedBundle, *, airsim_client: Any | None = None) -> None:
    """Apply a :class:`SeedBundle` to Python, NumPy, Torch, and AirSim.

    Parameters
    ----------
    bundle:
        The seed bundle to apply.
    airsim_client:
        Optional AirSim client instance. If provided, we attempt to set the
        simulator's seed directly. Otherwise an environment variable is set
        that can be consumed by the launcher.
    """

    random.seed(bundle.python)

    has_numpy = np is not None
    has_torch = torch is not None

    if not has_numpy and not has_torch:
        raise SeedApplicationError(
            "Cannot apply seed bundle because both NumPy and PyTorch are unavailable"
        )

    if has_numpy:
        assert np is not None  # narrow type for static analyzers
        np.random.seed(bundle.numpy)

    if has_torch:
        assert torch is not None
        torch.manual_seed(bundle.torch)
        if torch.cuda.is_available():  # pragma: no cover - hardware dependent
            torch.cuda.manual_seed_all(bundle.torch)
        if bundle.deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    _apply_airsim_seed(
        bundle.airsim, airsim_client=airsim_client, deterministic=bundle.deterministic
    )


def _apply_airsim_seed(seed: int, *, airsim_client: Any | None, deterministic: bool) -> None:
    if airsim_client is None:
        os.environ["AIRSIM_RANDOM_SEED"] = str(seed)
        os.environ["AIRSIM_DETERMINISTIC"] = "1" if deterministic else "0"
        return

    for candidate in ("setRandomSeed", "simSetSeed", "simRunCommand"):
        if hasattr(airsim_client, candidate):
            method = getattr(airsim_client, candidate)
            try:
                if candidate == "simRunCommand":
                    method("SetSeed", seed)  # pragma: no cover - depends on AirSim API
                else:
                    method(seed)
                return
            except Exception as exc:  # pragma: no cover - network errors
                raise SeedApplicationError(
                    f"Failed to apply AirSim seed via {candidate}: {exc}"
                ) from exc

    # Fallback: store intent in environment variables for launcher consumption.
    os.environ["AIRSIM_RANDOM_SEED"] = str(seed)
    os.environ["AIRSIM_DETERMINISTIC"] = "1" if deterministic else "0"


def derive_seed_bundle(bundle: SeedBundle, *, offset: int, stride: int = 9973) -> SeedBundle:
    """Derive a deterministic seed bundle for parallel environments.

    Parameters
    ----------
    bundle:
        The base :class:`SeedBundle` to offset.
    offset:
        Zero-based index identifying the derived bundle. ``0`` returns the original bundle.
    stride:
        Increment applied per offset step. Defaults to ``9973`` (prime) to minimise collisions.

    Returns
    -------
    SeedBundle
        A new bundle with each seed offset to ensure isolation between environments.
    """

    if offset < 0:
        raise ValueError("offset must be non-negative")
    if offset == 0:
        return bundle

    delta = offset * stride
    return SeedBundle(
        python=bundle.python + delta,
        numpy=bundle.numpy + delta,
        torch=bundle.torch + delta,
        airsim=bundle.airsim + delta,
        deterministic=bundle.deterministic,
    )


__all__ = ["SeedApplicationError", "apply_seed_bundle", "derive_seed_bundle"]
