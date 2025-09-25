# Contract: Perception Module

## Purpose

Provide processed drivable area mask and object detections for inclusion in ObservationPacket without embedding agent logic.

## Interfaces

```
load_models() -> None
capture_segmentation() -> np.ndarray  # (H,W) int mask
run_detection(image: np.ndarray) -> list[Detection]
build_observation_inputs(raw_rgb: np.ndarray) -> dict
```

## Detection Schema

| Field      | Type        | Notes                    |
| ---------- | ----------- | ------------------------ |
| class_id   | int         | Model-dependent class id |
| bbox       | list[float] | [x,y,w,h] normalized 0-1 |
| confidence | float       | 0-1                      |

## Requirements

- Must be callable independently of AirSimEnv (dependency injection of image sources).
- YOLOv12 weights loaded lazily via `torch.hub.load`.
- Segmentation capture uses AirSim API `ImageType.Segmentation` (ground-truth mask).
- Processing time budget <50ms per frame (baseline target).

## Error Handling

- If detector load fails: raise explicit ModelLoadError.
- If segmentation unavailable: return mask=None and set flag in info.

## Observability

- Log inference latency p50/p95 every N frames.
- Count detection classes per episode.

## Versioning

- Adding new fields requires reward contract review if referenced in reward shaping.
