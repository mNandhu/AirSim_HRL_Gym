# Reward Contract (Version 3.0)

**Last Updated**: 2025-10-01

This document outlines the reward shaping components for the AirSim HRL agent. The total reward is the sum of these components. The function adapts its shaping based on the active high-level command.

## Changes in Version 3.0

-   **Collision penalty reduced**: 200.0 → 50.0 (to allow learning from mistakes)
-   **Progress velocity boosted**: 1.2 → 2.0 (+67% stronger navigation incentive)
-   **Lane deviation relaxed**: 2.0 → 1.0 (-50% to allow path-following flexibility)
-   **Time penalty reduced**: -0.02 → -0.005 (-75% to reduce constant drain)
-   **Command persistence added**: Minimum 10 steps per command (enables completion bonuses)

## 1. Major Penalties & Bonuses (Event-Driven)

These rewards are typically large and sparse, defining catastrophic failures or major successes.

| Component               | Value/Formula | Trigger Condition                | Notes                                                                                                                                    |
| :---------------------- | :------------ | :------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- |
| **`collision_penalty`** | `-50.0`       | `VehicleState.collision == True` | Significant penalty to discourage collisions, but not so large that it erases entire episode progress. Allows learning from near-misses. |
| **`completion_bonus`**  | `+100.0`      | `command_completed == True`      | A large, sparse bonus awarded once upon the successful completion of a high-level command (e.g., navigating an intersection).            |

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

| Component                      | Formula                              | Sign    | Notes                                                                                                                                                                                                                |
| :----------------------------- | :----------------------------------- | :------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`progress_velocity`**        | `speed_mps * max(0, cos(θ)) * 2.0`   | **+**   | Rewards forward velocity scaled by alignment with the next waypoint. `θ` is the angle between the car's forward vector and the waypoint vector. Increased coefficient (2.0) provides stronger gradient for learning. |
| **`lane_deviation_penalty`**   | `-1.0 * (distance_from_center ** 2)` | **-**   | A squared penalty for deviating from the lane centerline. Reduced coefficient (1.0) allows more flexibility for path-following while still encouraging lane centering.                                               |
| **`heading_alignment_reward`** | `cos(θ) * 1.0`                       | **+/-** | Rewards aligning the car's heading with the next waypoint. It is positive when turning towards the waypoint and negative when turning away, providing critical guidance for turns.                                   |

## 3. General Penalties (Per-Step)

These penalties apply in all situations to discourage universally undesirable behavior.

| Component          | Value/Formula     | Trigger Condition                                 | Notes                                                                                                             |
| :----------------- | :---------------- | :------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------- |
| **`idle_penalty`** | `-0.5`            | `speed_mps < 0.1` AND `progress_possible == True` | A small penalty to discourage the agent from stopping when it should be moving (e.g., on a clear, straight road). |
| **`time_penalty`** | `-0.005` per step | Always                                            | Reduced per-timestep penalty to minimize constant drain while still incentivizing efficient completion.           |

---

**Configuration**: All coefficients and thresholds are configurable in `RewardConfig`. The values in this document reflect the current defaults. Bump `progress_velocity_coef` and `heading_alignment_coef` to make progress more rewarding, and tweak `time_penalty` (negative) to balance episode length vs. risk-taking.
