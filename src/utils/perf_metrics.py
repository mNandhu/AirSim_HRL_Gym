"""Utilities for sampling latency metrics in perception and environment loops."""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import DefaultDict, Dict, Iterator, List

__all__ = ["LatencyTracker", "time_block"]


@dataclass
class LatencyTracker:
    samples: DefaultDict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    def record(self, name: str, duration_ms: float) -> None:
        self.samples[name].append(duration_ms)

    def summary(self) -> Dict[str, dict[str, float]]:
        report: Dict[str, dict[str, float]] = {}
        for name, values in self.samples.items():
            if not values:
                continue
            if len(values) == 1:
                p50 = values[0]
            else:
                p50 = statistics.quantiles(values, n=2)[0]
            report[name] = {
                "count": float(len(values)),
                "p50": p50,
                "mean": statistics.fmean(values),
                "max": max(values),
            }
        return report


@contextmanager
def time_block(tracker: LatencyTracker, name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000.0
        tracker.record(name, duration_ms)
