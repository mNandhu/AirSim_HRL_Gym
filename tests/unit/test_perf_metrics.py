import time

from utils.perf_metrics import LatencyTracker, time_block


def test_latency_tracker_records_samples():
    tracker = LatencyTracker()
    tracker.record("perception", 10.0)
    tracker.record("perception", 20.0)
    summary = tracker.summary()

    assert summary["perception"]["count"] == 2.0
    assert summary["perception"]["max"] == 20.0


def test_time_block_records_duration(monkeypatch):
    tracker = LatencyTracker()

    def fake_perf_counter():
        fake_perf_counter.calls += 1
        return fake_perf_counter.calls * 0.001

    fake_perf_counter.calls = 0
    monkeypatch.setattr(time, "perf_counter", fake_perf_counter)

    with time_block(tracker, "step"):
        pass

    summary = tracker.summary()
    assert summary["step"]["count"] == 1.0
    assert summary["step"]["p50"] > 0
