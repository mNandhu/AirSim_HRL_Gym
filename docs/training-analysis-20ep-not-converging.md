# Training Analysis: 20 Episodes - NOT CONVERGING ⚠️

**Training Run**: `20251001T140747Z_train_d3313b8c-bf98-49b6-930d-308b6dc70c84`  
**Date**: October 1, 2025  
**Episodes**: 20  
**Status**: ❌ **NOT LEARNING** - Agent is getting worse over time

---

## 🚨 **Critical Issues Found**

### 1. **Trajectory Length DECREASING** (Your Observation is Correct!)

| Phase  | Episodes | Avg Steps | Trend                   |
| ------ | -------- | --------- | ----------------------- |
| Early  | 1-5      | **212.4** | Starting performance    |
| Middle | 8-12     | **154.0** | -27% decline            |
| Late   | 16-20    | **135.8** | -36% decline from start |

**Diagnosis**: Agent is colliding EARLIER in episodes as training progresses → **Negative learning**

---

### 2. **Rewards DECREASING**

| Phase      | Episodes | Avg Reward | Min   | Max    |
| ---------- | -------- | ---------- | ----- | ------ |
| **Early**  | 1-5      | **887.4**  | 220.4 | 1428.5 |
| **Middle** | 6-10     | **493.7**  | 398.3 | 736.4  |
| **Late 1** | 11-15    | **410.4**  | -34.6 | 801.1  |
| **Late 2** | 16-20    | **527.2**  | 320.9 | 818.8  |

**Diagnosis**:

-   Peak performance in episode 1 (1428.5)
-   Severe degradation to episode 11-15 (410.4)
-   Episode 14 **catastrophic failure** (-34.6, only 3 steps!)
-   No recovery trend

---

### 3. **Distance to Goal GETTING WORSE**

| Phase  | Episodes | Avg Min Distance             |
| ------ | -------- | ---------------------------- |
| Early  | 1-5      | **17.6m** ✓ Getting close!   |
| Middle | 6-10     | **30.7m** ⚠️ Getting farther |
| Late 1 | 11-15    | **37.4m** ❌ Much worse      |
| Late 2 | 16-20    | **26.5m** ⚠️ Slight recovery |

**Diagnosis**: Agent was closest to goal in early episodes (5.2m minimum!), now stays ~30-40m away

---

### 4. **SAC Workers NOT LEARNING AT ALL**

```
FOLLOW_LANE:
  Total updates: 4
  Loss: 0.0000 (both first and last)

STOP:
  Total updates: 4
  Loss: 0.0000 (both first and last)

TURN_LEFT_AT_INTERSECTION:
  Total updates: 6
  Loss: 0.0000 (both first and last)

TURN_RIGHT_AT_INTERSECTION:
  Total updates: 6
  Loss: 0.0000 (both first and last)
```

**Diagnosis**:

-   Workers have ZERO loss → not updating their policies
-   Only 4-6 updates in 20 episodes → insufficient training data
-   Workers are essentially random/untrained

---

### 5. **DQN Manager Learning is Unstable**

| Update | Loss     | Epsilon | Reward | Episode Length |
| ------ | -------- | ------- | ------ | -------------- |
| 145    | 2.50     | 0.99    | -0.00  | 200            |
| 745    | 0.89     | 0.97    | 7.58   | 500            |
| 945    | **4.69** | 0.97    | -0.00  | 600            |
| ...    | ...      | ...     | ...    | ...            |
| 5745   | **7.43** | 0.83    | 3.72   | 3000           |

**Diagnosis**:

-   Loss is unstable (0.89 → 7.43, increasing trend)
-   Exploration rate decreasing (0.99 → 0.83) but policy not improving
-   Manager making worse decisions as epsilon decreases

---

### 6. **Episode 14: Catastrophic Failure**

```
Step 0: reward=+3.61, speed=1.31, TURN_RIGHT
Step 1: reward=+4.91, speed=1.96, TURN_RIGHT
Step 2: reward=-43.14, speed=2.94, COLLISION! → Episode ends
```

**Total reward**: -34.62  
**Total steps**: 3  
**Diagnosis**: Immediate collision at episode start, complete policy failure

---

## 📊 **What the Graph Shows**

Looking at your attached episode summary graph:

### Episode Rewards (Top Left)

-   Episode 1: ~1400 (peak)
-   Episodes 2-3: Drop to ~200-500
-   Episodes 4-5: Recovery to ~1100
-   Episodes 6-20: Chaotic, mostly 300-800
-   Episode 14: Near zero (catastrophic)
-   **No upward trend** → Not learning

### Episode Length (Top Right)

-   Episode 1: ~390 steps (longest)
-   Declining trend visible
-   Episodes 14-20: Mostly <150 steps
-   **Clear negative trend** → Getting worse

### Max Speed (Bottom Left)

-   Relatively stable 5-7 m/s
-   Not the problem

### Waypoint Progress (Bottom Right)

-   Episode 1: 1 waypoint reached
-   Episodes 2-20: 0 waypoints reached
-   **No progress on task** → Not learning objective

---

## 🔍 **Root Cause Analysis**

### Why is the agent NOT learning?

#### Problem 1: Workers Have Insufficient Data

**Current situation**:

-   20 episodes × ~150 steps avg = ~3,000 total steps
-   4 workers + 1 manager = 5 agents
-   Each worker sees: 3,000 / 4 ≈ **750 steps** across 20 episodes
-   SAC typically needs 10,000+ steps to start learning

**Solution**: Workers need WAY more interaction data

#### Problem 2: Manager is Untrained

**Current situation**:

-   DQN manager has 29 updates (5745 steps)
-   DQN typically needs 50,000+ steps for stable learning
-   Exploration rate decreased but Q-values not converged
-   Manager making random/poor command selections

**Solution**: Manager needs many more episodes to learn good command sequences

#### Problem 3: Credit Assignment Problem

**Issue**:

-   Manager selects commands every 10 steps
-   Worker executes for 10 steps
-   Collision happens
-   **Who's responsible?** Manager's bad command choice OR worker's bad execution?

With current setup:

-   Manager blames worker (rewards worker got)
-   Worker blames manager (was following wrong command)
-   Neither learns correctly

#### Problem 4: Reward Structure Still Problematic

**Current rewards**:

```
Episode 1 (391 steps): +1428.5 → ~3.7 per step (good)
Episode 2 (78 steps):  +220.4 → ~2.8 per step (okay)
Episode 14 (3 steps):  -34.6 → -11.5 per step (catastrophic)
```

**Issue**:

-   Early collision gives huge negative signal
-   But agent hasn't learned WHAT caused collision yet
-   Creates noise that disrupts learning

---

## 🛠️ **Why Standard RL Isn't Working Here**

### The Hierarchical RL Challenge

**Standard RL** (works for single agent):

```
Agent → Action → Environment → Reward → Learn
         ↑___________________________|
```

**Hierarchical RL** (your setup):

```
Manager → Command → Worker → Action → Environment → Reward
   ↑         ↑          ↑         ↑
   |         |          |         |
   |         |          |_________|
   |         |____________________|
   |______________________________|
```

**Problems**:

1. Manager sees sparse rewards (command-level)
2. Worker sees dense rewards (action-level)
3. Manager needs worker to be good to learn
4. Worker needs manager to give good commands to learn
5. **Chicken-and-egg problem**: Neither can learn without the other being trained first!

---

## 🚀 **Recommended Solutions**

### Solution 1: Pre-train Workers FIRST (Recommended)

**Approach**: Train workers in isolation before hierarchical training

```python
# Step 1: Pre-train each worker for 50K steps
for worker_name in ['FOLLOW_LANE', 'TURN_LEFT', 'TURN_RIGHT', 'STOP']:
    # Use simple environment with only that command
    # Train until worker can execute command reliably
    # Save worker checkpoint

# Step 2: Freeze workers, train manager
# Workers are now competent, manager learns to sequence them

# Step 3: Fine-tune together (optional)
# Unfreeze workers for final polish
```

**Benefits**:

-   Workers learn basic skills first
-   Manager learns command sequencing with competent workers
-   Breaks chicken-and-egg problem

**Implementation**:
Create `src/scripts/pretrain_workers.py` that trains each worker individually on simple tasks

---

### Solution 2: Increase Training Episodes DRAMATICALLY

**Current**: 20 episodes = 3,000 steps total

**Needed for hierarchical RL**:

-   Workers: 10,000+ steps each = 40,000 steps
-   Manager: 50,000+ steps = 50,000 steps
-   **Total**: 100,000+ steps = **300-500 episodes** at current length

**Why**:

-   Hierarchical RL needs 10-100x more data than single-agent RL
-   Your workers have only seen 750 steps each (way too little)

**Recommendation**: Train for at least 100 episodes before evaluating

---

### Solution 3: Simplify to Single-Agent RL (Alternative)

**Approach**: Remove hierarchical structure temporarily

```python
# Instead of Manager → Worker → Action
# Do: Single Agent → Action

# Use SAC directly on low-level actions
# No command hierarchy
# Single policy learns everything
```

**Benefits**:

-   Simpler learning problem
-   Faster convergence (proven for continuous control)
-   Can add hierarchy later once basic skills learned

**Trade-off**:

-   Loses hierarchical abstraction
-   Policy may be less interpretable

---

### Solution 4: Fix Reward Structure (Critical)

Current issues to address:

#### 4a. Collision Penalty Still Too High

**Current**: -50 per collision  
**Problem**: Early collision (step 3) → -50 penalty → Net reward -35  
**Solution**: Reduce to -20 or even -10

```python
collision_penalty: float = 20.0  # Was 50.0
```

#### 4b. Add Intermediate Rewards

**Current**: Only shaping + collision  
**Problem**: No guidance for sub-goals  
**Solution**: Add waypoint bonuses

```python
# In reward.py
if waypoint_reached:
    waypoint_bonus = +50.0  # Immediate reward for reaching waypoint
```

#### 4c. Reduce Time Penalty Further

**Current**: -0.005 per step  
**Problem**: Still constant drain  
**Solution**: Remove entirely or make positive for survival

```python
time_penalty: float = 0.0  # Or even +0.01 (survival bonus)
```

---

### Solution 5: Curriculum Learning

**Approach**: Start with easy scenarios, gradually increase difficulty

**Stage 1**: Straight road, no obstacles (50 episodes)

-   Learn basic acceleration/steering
-   Learn to follow waypoints

**Stage 2**: Simple turns, no obstacles (50 episodes)

-   Learn turning behavior
-   Learn command following

**Stage 3**: Add obstacles (current scenario) (100+ episodes)

-   Learn collision avoidance
-   Integrate all skills

---

## 📋 **Immediate Action Plan**

### Option A: Quick Fix (Test if more episodes help)

```bash
# Train for 100 episodes
uv run python src/scripts/train_and_eval.py train \
  --config configs/experiments/training_waypoints.yaml \
  --episodes 100 \
  --save-interval 25 \
  --settings settings.json \
  --mode headless \
  --detector-model yolo12n \
  --metrics-update-interval 25
```

**Monitor**:

-   Worker loss (should become non-zero after ~20-30 episodes)
-   Episode length (should stabilize or increase)
-   Rewards (should show upward trend after initial chaos)

**If still not learning after 100 episodes** → Move to Option B

---

### Option B: Pre-train Workers (Proper Fix)

1. **Create worker pre-training script**:

    ```python
    # src/scripts/pretrain_workers.py
    # Train each worker on simple task for 50K steps
    # Save checkpoints
    ```

2. **Train workers individually**:

    ```bash
    uv run python src/scripts/pretrain_workers.py --worker FOLLOW_LANE --steps 50000
    uv run python src/scripts/pretrain_workers.py --worker TURN_LEFT --steps 50000
    uv run python src/scripts/pretrain_workers.py --worker TURN_RIGHT --steps 50000
    uv run python src/scripts/pretrain_workers.py --worker STOP --steps 20000
    ```

3. **Load pre-trained workers, train manager**:

    ```bash
    uv run python src/scripts/train_and_eval.py train \
      --config configs/experiments/training_waypoints.yaml \
      --load-workers checkpoints/workers/ \
      --freeze-workers \
      --episodes 200
    ```

4. **Fine-tune together**:
    ```bash
    uv run python src/scripts/train_and_eval.py train \
      --config configs/experiments/training_waypoints.yaml \
      --load-workers checkpoints/workers/ \
      --load-manager checkpoints/manager/ \
      --episodes 100
    ```

---

### Option C: Simplify to Single-Agent (Fastest Results)

1. **Modify training script** to use single SAC agent (no hierarchy)

2. **Train for 50 episodes**:

    ```bash
    uv run python src/scripts/train_single_agent.py \
      --config configs/experiments/training_waypoints.yaml \
      --episodes 50
    ```

3. **Expect convergence** within 30-50 episodes (proven for continuous control)

---

## 🎯 **Success Criteria** (How to know if it's learning)

### After 50 Episodes:

-   ✅ Worker loss > 0 and decreasing
-   ✅ Episode length stable or increasing (not decreasing)
-   ✅ Rewards showing slight upward trend
-   ✅ Collision rate <90% (was 100%)

### After 100 Episodes:

-   ✅ Worker loss converged (<1.0)
-   ✅ Episode length avg >150 steps
-   ✅ Rewards consistently >500
-   ✅ Collision rate <70%
-   ✅ At least 1 waypoint reached in some episodes

### After 200 Episodes:

-   ✅ Episode length avg >200 steps
-   ✅ Rewards consistently >700
-   ✅ Collision rate <50%
-   ✅ 2+ waypoints reached regularly

---

## 📊 **Current Status vs Expected**

| Metric                | Current (20 ep)   | Expected (20 ep)  | Status      |
| --------------------- | ----------------- | ----------------- | ----------- |
| **Avg Reward**        | 527               | 300-500           | ✓ Okay      |
| **Episode Length**    | 135 (declining)   | 150-200 (stable)  | ❌ Bad      |
| **Worker Loss**       | 0.0000            | >0 and decreasing | ❌ Critical |
| **Manager Loss**      | 7.43 (increasing) | <5.0 (decreasing) | ❌ Bad      |
| **Waypoints Reached** | 0-1               | 0-1               | ✓ Okay      |
| **Collision Rate**    | 100%              | 90-100%           | ✓ Expected  |

**Overall**: ❌ **NOT LEARNING** - Workers have zero loss, trajectories getting shorter

---

## 🏁 **Conclusion**

**Your observation is 100% correct**: The agent is NOT learning and is getting worse.

**Root cause**: Hierarchical RL is much harder than single-agent RL, and 20 episodes is far too few for this setup.

**Recommended path forward**:

1. **Short term**: Try 100 episodes to see if more data helps
2. **If that fails**: Pre-train workers individually (proper hierarchical RL approach)
3. **Alternative**: Simplify to single-agent SAC (faster results, proven approach)

**My strongest recommendation**: **Option B (Pre-train workers)**. This is the correct way to do hierarchical RL and will give you the best long-term results.

Hierarchical RL is powerful but requires careful bootstrapping. Your reward rebalancing was successful, but the learning algorithm needs the right training procedure to work.
