# Data Model: AirSim HRL Training Framework

## Entity: ExperimentDefinition

| Field           | Type       | Notes                                     |
| --------------- | ---------- | ----------------------------------------- |
| id              | UUID       | Generated per run                         |
| scene           | str        | AirSim map name (e.g., "Neighborhood")    |
| vehicle         | str        | Vehicle model identifier                  |
| start_pose      | Pose       | (x,y,z,yaw) seed pose                     |
| goal_pose       | Pose       | (x,y,z,yaw) terminal target               |
| horizon         | int        | Max steps per episode                     |
| seeds           | SeedBundle | All RNG seeds (python, numpy, torch, sim) |
| weather_profile | str        | Optional weather preset                   |
| generated_at    | datetime   | UTC timestamp                             |
| config_hash     | str        | Hash of normalized config JSON            |

## Entity: SeedBundle

| Field         | Type | Notes                                    |
| ------------- | ---- | ---------------------------------------- |
| python        | int  | Python random seed                       |
| numpy         | int  | NumPy seed                               |
| torch         | int  | Torch seed                               |
| airsim        | int  | AirSim internal seed                     |
| deterministic | bool | Force deterministic flags (torch, cudnn) |

## Entity: HighLevelCommand

| Field     | Type  | Notes                                                                    |
| --------- | ----- | ------------------------------------------------------------------------ |
| name      | enum  | FOLLOW_LANE, TURN_LEFT_AT_INTERSECTION, TURN_RIGHT_AT_INTERSECTION, STOP |
| priority  | int   | Lower number = higher priority                                           |
| timestamp | float | Issued time (s)                                                          |

## Entity: ObservationPacket

| Field             | Type             | Notes                                                |
| ----------------- | ---------------- | ---------------------------------------------------- |
| image             | np.ndarray       | Processed RGB or stacked frames (HWC/CHW documented) |
| segmentation_mask | np.ndarray       | Integer mask (H,W)                                   |
| detections        | list[Detection]  | YOLO bounding boxes                                  |
| command           | HighLevelCommand | Active high-level command                            |
| telemetry         | Telemetry        | Speed, acceleration, localization, collisions        |
| reward_components | dict[str, float] | Intermediate reward terms                            |
| done_flags        | dict[str, bool]  | {terminated, truncated, collision, goal_reached}     |

## Entity: Detection

| Field      | Type        | Notes                   |
| ---------- | ----------- | ----------------------- |
| class_id   | int         | Object category id      |
| bbox       | list[float] | [x, y, w, h] normalized |
| confidence | float       | Detection confidence    |

## Entity: Telemetry

| Field             | Type  | Notes                     |
| ----------------- | ----- | ------------------------- |
| speed_mps         | float | Vehicle speed             |
| acceleration_mps2 | float | Longitudinal acceleration |
| heading_deg       | float | Current yaw angle         |
| position          | Pose  | Current position          |
| collision         | bool  | Collision flag            |
| distance_to_goal  | float | Meters remaining          |

## Entity: TelemetryArtifact

| Field            | Type        | Notes                             |
| ---------------- | ----------- | --------------------------------- |
| episode_id       | UUID        | Episode identifier                |
| returns          | float       | Cumulative reward                 |
| reward_curve     | list[float] | Reward per step                   |
| failure_reason   | str         | Null if success                   |
| config_hash      | str         | Reference to ExperimentDefinition |
| seed_bundle      | SeedBundle  | Reproduces run                    |
| duration_seconds | float       | Episode wall-clock                |
| steps            | int         | Steps executed                    |

## Entity: SimulatorSession

| Field         | Type     | Notes                              |
| ------------- | -------- | ---------------------------------- |
| pid           | int      | AirSim process id                  |
| mode          | str      | gui or headless                    |
| start_time    | datetime | UTC                                |
| retries       | int      | Restart attempts                   |
| status        | str      | running, stopped, or failed        |
| settings_path | str      | Settings file passed to simulator  |

## Relationships

- ExperimentDefinition 1:N TelemetryArtifact (multiple episodes per experiment)
- HighLevelCommand 1:1 (latest) in ObservationPacket
- ObservationPacket aggregates Detection\*, Telemetry, HighLevelCommand
- TelemetryArtifact references SeedBundle (composition)
- SimulatorSession managed externally; not embedded in ObservationPacket

## State Transitions (High-level)

1. SimulatorSession: starting -> running -> (failed -> restarting -> running) | stopped
2. Episode: pending -> active -> (terminated | truncated)
3. Command lifecycle: issued -> active -> superseded -> archived

## Validation Rules

- horizon > 0
- seeds all non-negative ints
- priority unique per command enum
- distance_to_goal >= 0
- bbox values within [0,1]

## Notes

- Image tensor shape standardized: (3,H,W) float32 normalized [0,1]
- Segmentation mask optional; if disabled, field is None and reward components adapt
