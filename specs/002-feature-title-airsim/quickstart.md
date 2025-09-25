# Quickstart: AirSim HRL Training Framework

## 1. Environment Setup

```
uv python install 3.12
uv init --package hrl_airsim
uv add stable-baselines3 torch torchvision torchaudio numpy pydantic pytest pytest-cov msgpack-python
```

## 2. Acquire YOLOv12 Model

Loaded automatically on first run via `torch.hub.load('ultralytics/yolov12', 'yolov12n')` (example variant). Ensure internet access for initial pull.

## 3. Launch Headless Simulator

```
python -m hrl_airsim.utils.airsim_runner --mode headless --settings ./settings.json
```

(Automatically applies `-RenderOffScreen` and logs lifecycle events.)

## 4. Define Experiment Configuration

Create `configs/experiments/baseline.yaml`:

```yaml
scene: Neighborhood
vehicle: DefaultSedan
start_pose: { x: 0, y: 0, z: 0, yaw: 0 }
goal_pose: { x: 50, y: 0, z: 0, yaw: 0 }
horizon: 200
seeds: { python: 123, numpy: 123, torch: 123, airsim: 123, deterministic: true }
weather_profile: clear_day
```

## 5. Run Seeded Rollout

```
python scripts/run_experiment.py --config configs/experiments/baseline.yaml
```

Outputs stored under `artifacts/<timestamp>/` (config copy, logs, reward trace, telemetry JSON).

## 6. Reward Contract Documentation

Generated / updated at `docs/reward-contract.md` when reward components change.

## 7. Run Tests

```
pytest --cov=src --cov-report=term-missing
```

Expect coverage >=90% for non-learning packages before merging.

## 8. View Logs

- Simulator failures: `simulator_failures.log`
- Run metadata: `artifacts/<timestamp>/run.json`

## 9. Next Steps

Extend high-level command set, refine reward shaping, introduce multi-worker scheduling.
