# Metrics Enhancement: Lane Coverage & Waypoint Bonus Tracking

**Date**: October 4, 2025  
**Enhancement**: Added lane deviation and waypoint bonus to performance metrics plots

---

## Changes Made

### 1. ✅ Added Waypoint Progress Bonus to StepMetrics

**File**: `src/utils/metrics_tracker.py`

**What Changed**:

```python
class StepMetrics:
    # ... existing fields ...
    waypoint_progress_bonus: float = 0.0  # NEW
```

**Purpose**: Track the +50 bonus awarded when reaching each waypoint

---

### 2. ✅ Added Lane Coverage Tracking

**File**: `src/utils/metrics_tracker.py`

**What Changed**:

```python
class StepMetrics:
    # ... existing fields ...
    lane_mask_coverage_ratio: Optional[float] = None  # NEW
```

**Purpose**: Track how much of the camera view shows road (0.0 = off-road, 1.0 = fully on-road)

---

### 3. ✅ Extract Reward Components in log_step()

**File**: `src/utils/metrics_tracker.py`

**What Changed**:

```python
step_metrics = StepMetrics(
    # ... existing fields ...
    waypoint_progress_bonus=reward_components.get("waypoint_progress_bonus", 0.0),  # NEW
    lane_mask_coverage_ratio=lane_mask_coverage_ratio,  # NEW
)
```

**Purpose**: Extract these values from telemetry and reward components

---

### 4. ✅ Enhanced Performance Metrics Plot

**File**: `src/utils/metrics_tracker.py` - `_plot_performance_metrics()`

#### 4a. Added Waypoint Bonus to Reward Breakdown

**Visual Change**: New cyan band in reward stacking plot

```python
# Reward components now include:
1. Command Shaping (green)
2. Collision Penalty (red)
3. Completion Bonus (gold)
4. Waypoint Bonus (cyan)      # NEW
5. Idle Penalty (gray)
6. Time Penalty (black)
```

**Purpose**: Visualize when waypoints are reached (spikes of +50)

#### 4b. Added Lane Coverage Subplot (4th plot)

**New Subplot**: "Road Following (Lane Mask Coverage)"

**Features**:

-   Teal line showing lane coverage over time (0.0 to 1.0)
-   Orange dashed line at 0.5 (warning: 50% off-road)
-   Red dashed line at 0.2 (critical: termination threshold)

**Conditional Display**:

-   Only shown when lane data is available
-   Only in waypoint mode (not legacy mode)

---

## Visual Changes

### Before (3 subplots):

```
┌─────────────────────────────────────┐
│  1. Vehicle Speed                   │
├─────────────────────────────────────┤
│  2. Waypoint Progress / Distance    │
├─────────────────────────────────────┤
│  3. Reward Component Breakdown      │
│     - Command Shaping               │
│     - Collision Penalty             │
│     - Completion Bonus              │
│     - Idle Penalty                  │
│     - Time Penalty                  │
└─────────────────────────────────────┘
```

### After (4 subplots):

```
┌─────────────────────────────────────┐
│  1. Vehicle Speed                   │
├─────────────────────────────────────┤
│  2. Waypoint Progress               │
├─────────────────────────────────────┤
│  3. Reward Component Breakdown      │
│     - Command Shaping               │
│     - Collision Penalty             │
│     - Completion Bonus              │
│     - Waypoint Bonus        ← NEW   │
│     - Idle Penalty                  │
│     - Time Penalty                  │
├─────────────────────────────────────┤
│  4. Road Following                  │← NEW SUBPLOT
│     - Lane Coverage (teal line)     │
│     - 50% threshold (orange dash)   │
│     - 20% termination (red dash)    │
└─────────────────────────────────────┘
```

---

## Interpretation Guide

### Waypoint Bonus (Reward Plot)

**What to look for**:

-   **Cyan spikes of +50**: Waypoint reached successfully
-   **Frequency**: More spikes = better navigation
-   **Timing**: Earlier spikes = faster progress

**Example**:

```
Reward: ▁▁▁▁█▁▁▁▁█▁▁▁█
        ^   ^   ^
        WP1 WP2 WP3 reached
```

---

### Lane Coverage (4th Subplot)

**What to look for**:

1. **High coverage (0.8-1.0)**:

    - Agent staying on road ✓
    - Teal line near top

2. **Medium coverage (0.5-0.8)**:

    - Agent drifting off road ⚠️
    - Below orange line
    - May be cutting corners

3. **Low coverage (0.2-0.5)**:

    - Agent mostly off-road ❌
    - Near/below red line
    - Episode will terminate soon

4. **Critical (< 0.2)**:
    - Episode terminated
    - Complete off-road failure

**Example Patterns**:

**Good Episode** (staying on road):

```
1.0 ████████████████████
0.5 ─────────────────────  ← Orange threshold
0.0
```

**Cutting Corners** (drifts off occasionally):

```
1.0 ████▁▁████▁▁████
0.5 ─────────────────────
0.0
```

**Off-Road Episode** (immediate failure):

```
1.0 ██▁
0.5 ───▁───────────────
0.2 ─────▄▄▄▄▄▄▄▄  ← Below termination
0.0      X (terminated)
```

---

## Data Flow

```
Episode Step
    ↓
Simulator (airsim_util.py)
    ├─ Calculates lane_mask_coverage_ratio
    └─ Returns telemetry
    ↓
Environment (env.py)
    ├─ Checks off-road termination (< 0.2)
    └─ Calculates rewards
    ↓
Reward Calculator (reward.py)
    ├─ waypoint_progress_bonus: +50 when waypoint reached
    └─ Returns reward_components
    ↓
Metrics Tracker (metrics_tracker.py)
    ├─ Extracts lane_mask_coverage_ratio from telemetry
    ├─ Extracts waypoint_progress_bonus from reward_components
    └─ Stores in StepMetrics
    ↓
Plot Generation (_plot_performance_metrics)
    ├─ Adds waypoint bonus to reward stacking
    └─ Adds lane coverage subplot (if data available)
```

---

## Testing

### Unit Tests

```bash
uv run pytest tests/unit/test_reward_components.py -xvs
```

**Result**: ✅ All tests pass

### Visual Validation

After next training run, check:

1. ✅ `artifacts/*/metrics/performance_metrics.png` has 4 subplots
2. ✅ Cyan bands appear in reward plot when waypoints reached
3. ✅ Lane coverage plot shows values between 0.0-1.0
4. ✅ Thresholds visible at 0.5 (orange) and 0.2 (red)

---

## Debugging

### Issue: No 4th subplot (lane coverage missing)

**Cause**: Lane data not available

**Check**:

```python
# In episode JSON:
{
  "lane_mask_coverage_ratio": null  # ← Should be 0.0-1.0
}
```

**Fix**: Ensure segmentation is working (see v4.1 implementation)

---

### Issue: Waypoint bonus not showing

**Cause**: Waypoint not being reached OR bonus not in reward components

**Check**:

```python
# In episode JSON:
{
  "reward_components": {
    "waypoint_progress_bonus": 0.0  # ← Should be 50.0 when reached
  }
}
```

**Fix**: Check waypoint threshold (5m) and ensure bonus is calculated

---

## Expected Outcomes

### Successful Training Episode

**Reward Plot**:

-   Green command shaping increasing
-   Cyan spikes at waypoints (3-5 times)
-   Red collision at end (-10)

**Lane Coverage Plot**:

-   Mostly above 0.8 (on road)
-   Brief dips to 0.6-0.7 during turns
-   Never below 0.2 (no termination)

### Failed Training Episode

**Reward Plot**:

-   Green command shaping low
-   No cyan spikes (no waypoints reached)
-   Early red collision

**Lane Coverage Plot**:

-   Rapid drop from 1.0 → 0.2
-   Terminates after 20-50 steps
-   Shows off-road shortcut attempt

---

## Files Modified

1. ✅ `src/utils/metrics_tracker.py`
    - Added `waypoint_progress_bonus` field to `StepMetrics`
    - Added `lane_mask_coverage_ratio` field to `StepMetrics`
    - Extract both in `log_step()` method
    - Added waypoint bonus to reward stacking
    - Added 4th subplot for lane coverage

---

## Version History

-   **v4.0** (Oct 2): Single-agent compatibility
-   **v4.1** (Oct 4): Lane deviation fixes, off-road termination
-   **v4.1.1** (Oct 4): Enhanced metrics plotting ← **Current**

---

## Next Steps

1. ✅ Train 10-20 episodes with new metrics
2. ✅ Verify plots show lane coverage and waypoint bonuses
3. ✅ Analyze patterns to tune rewards if needed

**Status**: ✅ **Enhancement Complete - Ready for Training**
