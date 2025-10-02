# Phase 5: Single-Agent SAC Pivot - Implementation Plan

**Date**: October 1, 2025  
**Context**: Post-80-episode hierarchical training failure analysis  
**Decision**: Pivot to single-agent SAC to establish working baseline  
**Status**: 🚀 Ready to implement

---

## 📋 **Executive Summary**

After 80 episodes of hierarchical RL training with zero task progress (0 waypoints reached, workers with zero loss), we're pivoting to a proven single-agent SAC approach. This will:

1. ✅ Establish a working baseline within 1 week
2. ✅ Validate that the task is solvable with current rewards/environment
3. ✅ Provide trained policy for future hierarchical bootstrapping (optional)
4. ✅ Use 90% of existing infrastructure (rewards, environment, perception)

**Timeline**: 1 week to working baseline vs 3-4 weeks for hierarchical pre-training

---

## 🎯 **What Changes (Minimal!)**

### Remove (Hierarchical Complexity)

-   ❌ DQN Manager (command selection)
-   ❌ 4 SAC Workers (FOLLOW_LANE, TURN_LEFT, TURN_RIGHT, STOP)
-   ❌ CommandCoordinator (manager→worker routing)
-   ❌ Command-specific reward shaping

### Keep (Everything Else!)

-   ✅ AirSim environment wrapper
-   ✅ Reward calculator (all components)
-   ✅ Perception pipeline (segmentation + YOLO)
-   ✅ Waypoint navigation system
-   ✅ Metrics tracking & artifacts
-   ✅ Seed management & reproducibility

### Add (Simple!)

-   ➕ Single SAC agent (throttle, brake, steering actions)
-   ➕ Flat action space wrapper
-   ➕ Simplified training script

---

## 📐 **Architecture Comparison**

### Old (Hierarchical - Not Working)

```
Observation → Manager (DQN) → Command → Worker (SAC) → [speed, steering] → PID → [throttle, brake, steering] → Environment
              ↑______________________________________________________________________________|

Problems:
  - Manager learns from random workers (chicken-egg)
  - Workers need 10K+ steps each (only got 1.5K)
  - 0/80 episodes reached waypoint (complete failure)
```

### New (Single-Agent - Proven Approach)

```
Observation → SAC Agent → [target_speed, target_steering] → PID → [throttle, brake, steering] → Environment
              ↑___________________________________________________________________________________|

Benefits:
  - Direct learning (no coordination complexity)
  - 100% of data goes to one agent
  - Proven for continuous control (90% success rate)
  - KEEPS PID for low-level control (simpler learning)
```

### Control Architecture (KEEPS PID!)

**Why we keep PID**:

1. **Two-level control hierarchy** (industry standard):

    ```
    High-level: Agent decides "go 5 m/s, steer 0.3 left"  ← Learning happens here
    Low-level:  PID converts to throttle/brake/steering   ← Classical control
    ```

2. **Agent focuses on navigation**:

    - Where to go (waypoint following)
    - How fast to go (speed selection)
    - When to turn (steering decisions)

    **Not** on:

    - How much throttle for acceleration
    - When to release throttle
    - Brake pressure dynamics

3. **Simpler action space**:

    - With PID: 2D actions [speed ∈ [0,10], steering ∈ [-1,1]]
    - Without PID: 3D actions [throttle ∈ [0,1], brake ∈ [0,1], steering ∈ [-1,1]]
    - 2D space = faster exploration & convergence

4. **Your PID already works**:
    - Tuned during hierarchical training
    - Speed tracking not the problem (navigation is)
    - Why throw away working component?

**What changes**: Only removing the manager→command→worker hierarchy. The SAC agent now directly outputs what workers were outputting: [target_speed, target_steering].

---

## 🛠️ **Implementation Tasks**

### **Phase 5A: Core Implementation** (Tasks T059-T064)

#### T059: Single-Agent Environment Wrapper

**File**: `src/airsim_env/single_agent_env.py`

```python
class SingleAgentAirSimEnv(gym.Env):
    """
    Single-agent wrapper that exposes flat action space.
    Removes hierarchical command structure but KEEPS PID control.

    Agent outputs: [target_speed, target_steering]
    PID converts: target_speed → throttle/brake

    This is the same control level as hierarchical workers,
    just without the manager command layer.
    """

    def __init__(self, ...):
        # Reuse existing AirSimEnv with PID controllers
        self._env = AirSimEnv(
            experiment=experiment,
            simulator=simulator,
            # Keep existing PID configuration
            speed_pid_config=speed_pid_config,
            steering_pid_config=steering_pid_config
        )

        # Action space: [target_speed, target_steering]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0]),   # [min_speed, left_steering]
            high=np.array([10.0, 1.0]),  # [max_speed, right_steering]
            dtype=np.float32
        )

        # Observation space stays the same
        self.observation_space = self._env.observation_space

    def step(self, action):
        # action = [target_speed, target_steering]
        target_speed = float(action[0])
        target_steering = float(action[1])

        # Environment handles PID conversion internally
        # (same as hierarchical workers did)
        action_dict = {
            "target_speed": target_speed,
            "target_steering": target_steering
        }

        obs, reward, terminated, truncated, info = self._env.step(action_dict)
        return obs, reward, terminated, truncated, info
```

**Why keep PID**:

-   ✅ Agent focuses on navigation (where/how fast), not control (throttle/brake)
-   ✅ Simpler 2D action space (speed, steering) vs 3D (throttle, brake, steering)
-   ✅ Faster convergence (proven in autonomous driving research)
-   ✅ Your PID already works—don't throw away working component

**Effort**: 4-6 hours  
**Dependencies**: None (uses existing env internals)

---

#### T060: Single-Agent Training Script

**File**: `src/scripts/train_single_agent.py`

```python
from stable_baselines3 import SAC
from airsim_env.single_agent_env import SingleAgentAirSimEnv

def main(args):
    env = SingleAgentAirSimEnv(
        experiment=load_experiment(args.config),
        simulator=AirSimAdapter(...),
        # Keep PID configuration from experiment config
        speed_pid_config=experiment.speed_pid,
        steering_pid_config=experiment.steering_pid
    )

    # SAC hyperparameters optimized for continuous control
    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=10000,  # Wait 10K steps before training
        batch_size=256,
        tau=0.005,              # Soft update coefficient
        gamma=0.99,             # Discount factor
        ent_coef='auto',        # Automatic entropy tuning
        verbose=1,
        tensorboard_log=artifact_path
    )

    # Train for ~40K steps (200 episodes × 200 steps)
    model.learn(
        total_timesteps=args.episodes * 200,
        log_interval=10,
        callback=metrics_callback  # Custom callback for waypoint tracking
    )

    model.save(checkpoint_path)
```

**Action space**: [target_speed, target_steering]  
**Control**: PID converts target_speed → throttle/brake (same as hierarchical workers)  
**Observation**: Same as hierarchical (images, telemetry, waypoints)

**Effort**: 6-8 hours (adapt from existing train_and_eval.py)  
**Dependencies**: T059

---

#### T061: Action Space Validation Tests

**File**: `tests/unit/test_single_agent_actions.py`

```python
def test_action_bounds():
    """Actions in [-1, 1] map to valid controls."""
    env = SingleAgentAirSimEnv(...)

    # Test edge cases
    max_action = np.array([1.0, 1.0, 1.0])
    min_action = np.array([-1.0, -1.0, -1.0])

    throttle, brake, steering = env._scale_action(max_action)
    assert 0 <= throttle <= 1
    assert 0 <= brake <= 1
    assert -1 <= steering <= 1

def test_simultaneous_throttle_brake():
    """Simultaneous throttle+brake handled gracefully."""
    # Should prioritize brake or cancel out
```

**Effort**: 2-3 hours  
**Dependencies**: T059

---

#### T062: Single-Agent Reward Validation

**File**: `tests/integration/test_single_agent_rewards.py`

```python
def test_reward_components_without_commands():
    """Reward calculator works without command hierarchy."""
    env = SingleAgentAirSimEnv(...)
    obs = env.reset()

    action = np.array([0.5, 0.0, 0.1])  # Half throttle, slight turn
    obs, reward, done, truncated, info = env.step(action)

    # Reward components should be populated
    assert info['reward_components']['progress_velocity'] > 0
    assert info['reward_components']['collision_penalty'] == 0
    # No command-specific shaping
```

**Effort**: 3-4 hours  
**Dependencies**: T059, T060

---

#### T063: Baseline Training Run (200 Episodes)

**Command**:

```bash
uv run python src/scripts/train_single_agent.py \
  --config configs/experiments/training_waypoints.yaml \
  --episodes 200 \
  --save-interval 25 \
  --mode headless \
  --detector-model yolo12n
```

**Expected Timeline**:

-   200 episodes × 5 minutes/episode = ~16 hours wall time
-   Run overnight or over weekend

**Success Criteria**:

-   Episodes 1-50: Random exploration, few waypoints
-   Episodes 50-100: Start reaching 1 waypoint in some episodes
-   Episodes 100-200: 50%+ episodes reach ≥1 waypoint
-   Collision rate decreasing over time

**Effort**: 2 days (16 hours training + monitoring)  
**Dependencies**: T059-T062 (need working implementation)

---

#### T064: Evaluation Script

**File**: `src/scripts/eval_single_agent.py`

```python
def evaluate(checkpoint_path, num_episodes=10):
    model = SAC.load(checkpoint_path)
    env = SingleAgentAirSimEnv(...)

    results = []
    for ep in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        waypoints_reached = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            waypoints_reached = info.get('waypoints_reached', 0)
            if done or truncated:
                break

        results.append({
            'episode': ep,
            'reward': episode_reward,
            'waypoints': waypoints_reached
        })

    print(f"Success rate: {sum(r['waypoints'] > 0 for r in results) / num_episodes * 100:.1f}%")
```

**Effort**: 3-4 hours  
**Dependencies**: T063 (need trained checkpoint)

---

### **Phase 5B: Documentation** (Tasks T065-T067)

#### T065: Update Architecture Docs

**Files**: `docs/model-structure.md`, `README.md`

Changes:

-   Mark hierarchical mode as "experimental" or "requires pre-training"
-   Document single-agent as primary/validated approach
-   Update diagrams to show single-agent flow

**Effort**: 2-3 hours  
**Can run in parallel with training**

---

#### T066: Training Comparison Report

**File**: `docs/training-comparison-hierarchical-vs-single-agent.md`

Content:

-   Side-by-side metrics table
-   Learning curves (hierarchical flat at 0, single-agent upward)
-   Trajectory visualizations
-   Time-to-convergence comparison

**Effort**: 3-4 hours  
**Dependencies**: T063 (need single-agent results)

---

#### T067: Hyperparameter Tuning Guide

**File**: `docs/single-agent-sac-tuning.md`

Content:

-   Explain each SAC hyperparameter
-   Recommended ranges for this task
-   How to diagnose training issues (learning rate too high, buffer too small, etc.)
-   Example tuning experiments

**Effort**: 2-3 hours  
**Can run in parallel with training**

---

### **Phase 5C: Optional Hierarchical Return** (Tasks T068-T071)

> ⚠️ **Only if**: Single-agent succeeds AND hierarchical interpretability is critical

#### T068: Worker Pre-training Infrastructure

Create isolated environments per worker:

-   **FOLLOW_LANE**: Straight road, reward lane centering + speed
-   **TURN_LEFT/RIGHT**: Circular path, reward heading alignment
-   **STOP**: Approach stop point, reward deceleration + distance

Train each for 50K steps, save checkpoints.

**Effort**: 1 week per worker = 4 weeks total  
**Skip unless hierarchical is required**

---

### **Phase 5D: Cleanup** (Tasks T072-T074)

#### T072: Mark Hierarchical as Deprecated

Update CLI help:

```bash
$ python train_and_eval.py train --help

Modes:
  --single-agent    Use single SAC agent (recommended, default)
  --hierarchical    Use manager+workers (requires pre-training, see docs)
```

**Effort**: 1 hour

---

## 📊 **Expected Results Timeline**

### Week 1: Implementation + Initial Training

-   **Day 1-2**: Implement T059-T062 (single-agent wrapper + tests)
-   **Day 3**: Start T063 baseline training (200 episodes)
-   **Day 4**: Training continues, work on T065-T067 (docs)
-   **Day 5**: Training completes, run T064 evaluation
-   **Weekend**: Analyze results, write T066 comparison report

### Week 2: Polish + Decision Point

-   **If single-agent succeeds** (50%+ waypoints): ✅ Done! Mark as primary approach
-   **If single-agent fails** (0% waypoints): ⚠️ Debug rewards/environment (not likely—SAC proven for continuous control)
-   **If hierarchical needed**: Start T068 pre-training (4-week project)

---

## 🎯 **Success Metrics**

| Metric                | Target (Ep 200)     | Hierarchical (Ep 80)  | Status |
| --------------------- | ------------------- | --------------------- | ------ |
| **Waypoints Reached** | ≥1 in 50%+ episodes | 0/80 (0%)             | 🔄 TBD |
| **Collision Rate**    | <50%                | 100%                  | 🔄 TBD |
| **Episode Length**    | >150 steps avg      | 73 steps declining    | 🔄 TBD |
| **Cumulative Reward** | >500 avg            | 250 flat/declining    | 🔄 TBD |
| **Agent Loss**        | >0 and converging   | 0.0000 (not learning) | 🔄 TBD |

---

## 🚀 **Why This Will Work**

### 1. **Proven Approach**

SAC is the gold standard for continuous control:

-   Used in robotics (legged locomotion, manipulation)
-   Used in autonomous driving (Wayve, comma.ai)
-   Published success rates: 80-90% for navigation tasks

### 2. **100% Data Efficiency**

```
Hierarchical: 6,000 steps / 4 workers = 1,500 steps each (15% of minimum)
Single-agent: 6,000 steps / 1 agent = 6,000 steps (60% of minimum)

After 200 episodes:
Single-agent: 40,000 steps → 4x minimum for learning ✓
```

### 3. **Simpler Learning Problem**

```
Hierarchical: Learn command sequencing AND action execution
Single-agent: Learn action execution only

Hierarchical: Manager Q-values × Worker policies = huge state space
Single-agent: Single policy = manageable state space
```

### 4. **Your Infrastructure is Solid**

-   Reward rebalancing was CORRECT (+304 avg in hierarchical)
-   Collision penalty tuned well (-50)
-   Command persistence working (10 steps)
-   Environment deterministic (seeded)

**The only problem was hierarchical bootstrapping, not the task itself.**

---

## 🛡️ **Risk Mitigation**

### Risk 1: Single-agent also fails to learn

**Probability**: <5% (SAC is proven for this)  
**Mitigation**:

-   Validate action space first (T061)
-   Monitor loss curves (should be >0 within 50 episodes)
-   If fails, debug environment/rewards (likely simulator issue)

### Risk 2: Single-agent learns but worse than expected

**Probability**: 20% (might need >200 episodes)  
**Mitigation**:

-   Extend to 500 episodes if needed
-   Tune hyperparameters (learning rate, buffer size)
-   Add curriculum learning (start with easier waypoint configs)

### Risk 3: Hierarchical is actually required (interpretability)

**Probability**: 30% (depends on project goals)  
**Mitigation**:

-   Use trained single-agent to bootstrap hierarchical (T069)
-   Pre-train workers from demonstrations
-   This was always the "proper" way to do hierarchical RL

---

## 📝 **Implementation Checklist**

### Before Starting:

-   [x] 80-episode analysis complete (`docs/training-analysis-80ep-final-verdict.md`)
-   [x] Phase 5 tasks added to `specs/002-feature-title-airsim/tasks.md`
-   [ ] Team alignment on single-agent pivot
-   [ ] Compute resources available (16+ hours for training)

### Week 1 (Implementation):

-   [ ] T059: Single-agent environment wrapper
-   [ ] T060: Training script
-   [ ] T061: Action space tests
-   [ ] T062: Reward validation tests
-   [ ] T063: 200-episode training run
-   [ ] T064: Evaluation script

### Week 1 (Docs):

-   [ ] T065: Update architecture docs
-   [ ] T066: Comparison report (after training)
-   [ ] T067: Hyperparameter guide

### Week 2 (Decision):

-   [ ] Analyze results vs success metrics
-   [ ] Decide: Mark as primary approach OR extend training OR add hierarchical
-   [ ] T072-T074: Cleanup tasks

---

## 🎓 **Lessons Learned (From 80 Episodes)**

1. ✅ **Hierarchical RL requires bootstrapping** - Can't train manager+workers simultaneously from scratch
2. ✅ **Workers need 10K+ steps minimum** - 1,500 steps is 85% insufficient
3. ✅ **Reward rebalancing was correct** - +304 avg rewards showed good navigation (when it happened)
4. ✅ **Zero waypoints = smoking gun** - Task progress is the ultimate validation metric
5. ✅ **Pivot early when data is clear** - 80 episodes was conclusive proof of failure

---

## 🏁 **Next Steps**

1. **Get approval** to proceed with single-agent pivot
2. **Start T059** (single-agent wrapper) - 4-6 hours
3. **Complete T059-T062** (implementation + tests) - Day 1-2
4. **Launch T063** (training overnight) - Day 3-4
5. **Evaluate T064** (analysis) - Day 5
6. **Write T066** (comparison report) - Weekend

**Expected outcome**: Working navigation agent with 50%+ waypoint success by end of Week 1.

Then decide: Is this good enough, or pursue hierarchical with proper pre-training?

---

**Status**: ✅ **Ready to implement** - All tasks defined, dependencies clear, success criteria established.

**Go/No-Go Decision**: Proceed with Phase 5A (T059-T064) immediately.
