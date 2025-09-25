# Contract: Simulator Management Utility (airsim_runner)

## Purpose

Start, monitor, and terminate AirSim processes (GUI or headless) for deterministic experiments and automated tests.

## Public Interface

```
launch(mode: Literal['gui','headless'], settings_path: str) -> SimulatorSession
ensure_running(session: SimulatorSession) -> None
terminate(session: SimulatorSession, force: bool = False) -> None
contextmanager airsim_session(mode='headless') -> Iterator[SimulatorSession]
```

## Behavior

- Pass `-settings <settings_path>` and `-RenderOffScreen` (headless) when mode=headless.
- Retry startup up to 3 times with exponential backoff (e.g., 2s,4s,8s) before failing.
- Log failures to `simulator_failures.log` with timestamp + reason.

## Health Checks

- Confirm process alive and listening (if port probing available) within timeout window (default 30s).
- Validate settings hash against expected config.

## Failure Modes

| Scenario              | Handling                                                  |
| --------------------- | --------------------------------------------------------- |
| Startup timeout       | Retry until max attempts; then raise SimulatorLaunchError |
| Unexpected exit       | Attempt relaunch (<= remaining retries)                   |
| Invalid settings path | Immediate error (no retries)                              |

## Observability

- Structured JSON lines: {event, pid, mode, attempt, status, elapsed_ms}
- Metrics counters: launches_success, launches_failed, restarts

## Fixture Integration

`pytest` fixture `airsim_session` manages context for test scope (session-level; yields SimulatorSession).

## Versioning

- Changes to retry semantics or required CLI flags documented in CHANGELOG section of runner module docstring.
