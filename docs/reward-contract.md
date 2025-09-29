# Reward Contract (Version 2.0)

**Last Updated**: 2025-09-25

This document outlines the reward shaping components for the AirSim HRL agent. The total reward is the sum of these components. The function adapts its shaping based on the active high-level command.

## 1. Major Penalties & Bonuses (Event-Driven)

These rewards are typically large and sparse, defining catastrophic failures or major successes.

| Component               | Value/Formula | Trigger Condition                | Notes                                                                                                                                |
| :---------------------- | :------------ | :------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------- |
| **`collision_penalty`** | `-200.0`      | `VehicleState.collision == True` | A large, dominant penalty to strongly deter any contact with obstacles. Designed to be the most significant term in any single step. |
| **`completion_bonus`**  | `+100.0`      | `command_completed == True`      | A large, sparse bonus awarded once upon the successful completion of a high-level command (e.g., navigating an intersection).        |

## 2. Dense Shaping Rewards (Per-Step Guidance)

These rewards are calculated on every time step to provide continuous feedback to the agent, guiding it toward correct behavior. The primary shaping reward adapts based on the active command.

### 2.1. Command-Specific Shaping (`command_shaping`)

This is the main guidance signal, composed of other dense rewards depending on the task.

| Active Command                 | Formula                                          | Purpose                                                            |
| :----------------------------- | :----------------------------------------------- | :----------------------------------------------------------------- |
| **`FOLLOW_LANE`**              | `progress_velocity` + `lane_deviation_penalty`   | To drive efficiently along the road center.                        |
| **`TURN_LEFT` / `TURN_RIGHT`** | `progress_velocity` + `heading_alignment_reward` | To smoothly turn towards the next waypoint without overshooting.   |
| **`STOP` / Other**             | `0.0`                                            | No specific guidance needed when stationary or for other commands. |

### 2.2. Core Dense Components

These are the building blocks for the command-specific shaping rewards.

| Component                      | Formula                              | Sign    | Notes                                                                                                                                                                                                            |
| :----------------------------- | :----------------------------------- | :------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`progress_velocity`**        | `speed_mps * max(0, cos(θ)) * 1.2`   | **+**   | Rewards forward velocity scaled by alignment with the next waypoint. `θ` is the angle between the car's forward vector and the waypoint vector. Amplified to provide a stronger gradient for policy improvement. |
| **`lane_deviation_penalty`**   | `-2.0 * (distance_from_center ** 2)` | **-**   | A squared penalty for deviating from the lane centerline. It is lenient on minor adjustments but harshly penalizes large deviations, promoting smooth driving.                                                   |
| **`heading_alignment_reward`** | `cos(θ) * 1.0`                       | **+/-** | Rewards aligning the car's heading with the next waypoint. It is positive when turning towards the waypoint and negative when turning away, providing critical guidance for turns.                               |

## 3. General Penalties (Per-Step)

These penalties apply in all situations to discourage universally undesirable behavior.

| Component          | Value/Formula    | Trigger Condition                                 | Notes                                                                                                             |
| :----------------- | :--------------- | :------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------- |
| **`idle_penalty`** | `-0.5`           | `speed_mps < 0.1` AND `progress_possible == True` | A small penalty to discourage the agent from stopping when it should be moving (e.g., on a clear, straight road). |
| **`time_penalty`** | `-0.02` per step | Always                                            | A small per-timestep penalty to incentivize shorter episodes and efficient completion.                            |

---

**Configuration**: All coefficients and thresholds are configurable in `RewardConfig`. The values in this document reflect the current defaults. Bump `progress_velocity_coef` and `heading_alignment_coef` to make progress more rewarding, and tweak `time_penalty` (negative) to balance episode length vs. risk-taking.
