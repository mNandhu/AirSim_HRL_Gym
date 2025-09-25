# Contract: AirSimEnv

## Purpose

Gym-compatible wrapper isolating simulator interaction, reward computation, and observation assembly. Agent-agnostic.

## Public Interface

```
reset(seed_bundle: SeedBundle | None = None) -> ObservationPacket
step(action: dict[str, float]) -> tuple[ObservationPacket, float, bool, bool, dict]
close() -> None
compute_reward(prev_obs: ObservationPacket, action: dict, new_obs: ObservationPacket) -> float
get_observation(raw=False) -> ObservationPacket
```

## Action Space

| Control  | Range  | Type  | Notes                                            |
| -------- | ------ | ----- | ------------------------------------------------ |
| throttle | [-1,1] | float | Negative for reverse/regen; mapped to AirSim API |
| brake    | [0,1]  | float | 1 = full braking                                 |
| steering | [-1,1] | float | -1 left, +1 right                                |

## Command Space (HighLevelCommand)

FOLLOW_LANE, TURN_LEFT_AT_INTERSECTION, TURN_RIGHT_AT_INTERSECTION, STOP

## Observation Structure (ObservationPacket subset)

| Field             | Type                   | Notes                             |
| ----------------- | ---------------------- | --------------------------------- |
| image             | float32 tensor (3,H,W) | Normalized RGB                    |
| segmentation_mask | int tensor (H,W)       | Optional if segmentation disabled |
| detections        | list[Detection]        | See detection schema              |
| command           | HighLevelCommand       | Current active command            |
| telemetry         | Telemetry              | Vehicle state snapshot            |
| reward_components | dict[str,float]        | Decomposed reward terms           |
| done_flags        | dict[str,bool]         | Termination/truncation metadata   |

## Reward Components (initial)

| Component          | Description                                      | Sign | Initial Formula                                                             |
| ------------------ | ------------------------------------------------ | ---- | --------------------------------------------------------------------------- |
| progress           | Delta reduction in distance_to_goal              | +    | `prev_distance_to_goal - curr_distance_to_goal`                             |
| lane_adherence     | Staying within drivable area                     | +    | `lane_mask_coverage_ratio`                                                  |
| collision          | Collision penalty                                | -    | `-1.0 * int(collision_occurred)`                                            |
| command_completion | Reward when command-specific objective achieved  | +    | `int(command_objective_met) * completion_bonus`                             |
| idle_penalty       | Penalty for zero velocity when progress possible | -    | `-idle_penalty_coef * int(speed_mps < idle_threshold && progress_possible)` |

## Termination Conditions

- distance_to_goal <= threshold
- collision == True
- step_count >= horizon

## Truncation Conditions

- SimulatorSession failure mid-episode
- Perception pipeline exhaustion / timeout

## Info Dictionary Keys

| Key            | Type     | Meaning                     |
| -------------- | -------- | --------------------------- |
| config_hash    | str      | Experiment config reference |
| seed_bundle    | dict     | RNG seeds used              |
| episode_step   | int      | Current step index          |
| command_name   | str      | Active high-level command   |
| raw_detections | optional | Unfiltered detector output  |

## Invariants

- Reward computed only in `compute_reward`.
- `step` must call `compute_reward` exactly once per environment transition.
- No agent package imports.

## Versioning

Changes to observation field names or reward component semantics require doc update + minor version bump in reward contract.
