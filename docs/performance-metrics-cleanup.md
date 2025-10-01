# Performance Metrics Plot Cleanup for Waypoint Navigation

## Problem

The Performance Metrics plot (per-episode detailed view) had **redundant/misleading information** for waypoint-based navigation:

### Original Structure (4 subplots):

1. **Vehicle Speed** - Useful ✅
2. **Distance to Goal** - Flat line at ~1000m for circuits ❌
3. **Waypoint Progress** - Useful ✅
4. **Reward Components** - Useful ✅

**Issues:**

-   "Distance to Goal" shows distance to final waypoint (finish line)
-   For circuits, this is mostly constant (~1000m)
-   Provides no useful information during the episode
-   Takes up valuable screen space
-   Confusing alongside "Waypoint Progress"

## Solution

Restructured plot to show only relevant information:

### New Structure - Waypoint Mode (3 subplots):

1. **Vehicle Speed** - Speed over time
2. **Waypoint Progress** - Distance to current waypoint with transition markers
3. **Reward Components** - Breakdown of reward signals

### Legacy Structure - No Waypoints (3 subplots):

1. **Vehicle Speed** - Speed over time
2. **Distance to Goal** - Distance to single goal
3. **Reward Components** - Breakdown of reward signals

**Benefits:**

-   Cleaner, more focused visualization
-   Removes redundant/misleading information
-   Waypoint progress clearly visible with transition markers
-   Backward compatible with legacy configs

## Implementation

### Detection Logic

```python
has_waypoints = any(
    hasattr(s, "distance_to_current_waypoint")
    and s.distance_to_current_waypoint is not None
    for s in steps_snapshot
)
```

### Conditional Layout

```python
if has_waypoints:
    # 3 subplots: Speed, Waypoint Progress, Rewards
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
else:
    # 3 subplots: Speed, Distance to Goal, Rewards
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
```

### Subplot 2 - Adaptive Content

**Waypoint Mode:**

-   Shows distance to current waypoint (blue line)
-   Vertical green dashed lines mark waypoint transitions
-   Labels show "WP{index}" at each transition
-   Decreasing distance shows approach to each waypoint
-   Step pattern shows advancement through path

**Legacy Mode:**

-   Shows distance to single goal (orange line)
-   Smooth decreasing line (or flat if not approaching)
-   Simple point-to-point visualization

## Visual Improvements

### Before (Waypoint Mode - 4 subplots):

```
┌─────────────────────────┐
│ 1. Vehicle Speed        │ ✅ Useful
├─────────────────────────┤
│ 2. Distance to Goal     │ ❌ Flat ~1000m line
├─────────────────────────┤
│ 3. Waypoint Progress    │ ✅ Useful
├─────────────────────────┤
│ 4. Reward Components    │ ✅ Useful
└─────────────────────────┘
```

### After (Waypoint Mode - 3 subplots):

```
┌─────────────────────────┐
│ 1. Vehicle Speed        │ ✅ Useful
├─────────────────────────┤
│ 2. Waypoint Progress    │ ✅ Useful (with markers)
├─────────────────────────┤
│ 3. Reward Components    │ ✅ Useful
└─────────────────────────┘
```

**Result:** Cleaner, more informative, removes redundancy

## Waypoint Progress Details

The Waypoint Progress subplot now shows:

**Blue line**: Distance to current target waypoint

-   Starts high when waypoint is far
-   Decreases as agent approaches
-   Drops sharply when waypoint reached (advances to next)
-   Creates sawtooth pattern for multi-waypoint paths

**Green dashed lines**: Waypoint transitions

-   Vertical line when waypoint index changes
-   Marks the moment a waypoint is reached

**WP labels**: Waypoint index

-   Shows which waypoint was just reached
-   "WP0", "WP1", "WP2", etc.
-   Helps correlate with trajectory plot

## Edge Cases

### Circuit Paths

-   Shows meaningful progress through lap
-   Distance decreases/resets at each waypoint
-   Clear visualization of path following

### Linear Paths (Point-to-Point)

-   Shows progress through checkpoint sequence
-   Final waypoint = destination
-   Similar to legacy behavior but with checkpoints

### Legacy Configs (No Waypoints)

-   Falls back to "Distance to Goal" on subplot 2
-   Single goal point navigation
-   Maintains original behavior

## Testing

✅ All 152 tests pass
✅ Waypoint mode shows 3 subplots correctly
✅ Legacy mode shows 3 subplots with distance to goal
✅ Waypoint transitions marked with lines and labels
✅ No redundant information

## Files Changed

-   `src/utils/metrics_tracker.py`:
    -   `_plot_performance_metrics`: Restructured from 4 to 3 subplots
    -   Conditional layout based on waypoint data presence
    -   Removed redundant "Distance to Goal" for waypoint mode
    -   Enhanced waypoint progress with transition markers

## Impact

**For Waypoint Navigation:**

-   ✅ Cleaner plots (3 instead of 4 subplots)
-   ✅ No confusing/flat "Distance to Goal" line
-   ✅ Focus on relevant waypoint progress
-   ✅ Better use of screen space

**For Legacy Configs:**

-   ✅ No changes (still shows Distance to Goal)
-   ✅ Backward compatible
-   ✅ Same 3-subplot layout

## Next Training Run

Your performance metrics will now show:

```
Performance Metrics (Episode N)
┌───────────────────────────────┐
│ Vehicle Speed                 │  Purple filled area
├───────────────────────────────┤
│ Waypoint Progress             │  Blue line with green markers
│   ╷     ╷     ╷     ╷         │  (transition lines)
│   WP1   WP2   WP3   WP4       │  (labels)
├───────────────────────────────┤
│ Reward Component Breakdown    │  Stacked colored areas
└───────────────────────────────┘
```

Clean, focused, and informative! 🎯
