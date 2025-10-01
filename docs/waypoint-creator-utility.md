# Waypoint Creator Utility

A utility script to create waypoint paths by manually driving in AirSim.

## Purpose

When setting up training scenarios, you need to define a path with multiple waypoints. This script makes it easy to:

1. Drive manually in the simulator to explore the environment
2. Mark positions as waypoints by pressing Enter
3. Define the final goal position
4. Get YAML-formatted output ready to paste into experiment configs

## Usage

### Basic Usage

```bash
python src/scripts/create_waypoints.py
```

### With Custom Settings

```bash
python src/scripts/create_waypoints.py --settings path/to/settings.json
```

## Workflow

1. **Start the script**

    ```bash
    python src/scripts/create_waypoints.py
    ```

2. **Drive manually in AirSim**

    - Use keyboard/gamepad to drive to your first waypoint location
    - The script keeps API control **disabled** so you can drive freely

3. **Capture waypoints**

    - When you reach a position you want as a waypoint, press **ENTER**
    - The script briefly enables API control, reads position, then disables it again
    - Continue driving to the next waypoint

4. **Set final goal**

    - Drive to your desired goal position
    - Type **`end`** and press ENTER
    - This captures the final goal waypoint

5. **Copy YAML output**
    - The script outputs properly formatted YAML
    - Copy and paste directly into your experiment config

## Commands

While running the script:

| Command                 | Action                                            |
| ----------------------- | ------------------------------------------------- |
| **ENTER** (empty input) | Capture current position as waypoint              |
| **`end`**               | Capture current position as final goal and finish |
| **`quit`**              | Exit without saving                               |
| **Ctrl+C**              | Emergency exit                                    |

## Example Session

```
Connecting to AirSim...
✓ Connected to AirSim

======================================================================
  AirSim Waypoint Creator
======================================================================

INSTRUCTIONS:
  1. Drive manually to your desired waypoint location
  2. Press ENTER to capture that position as a waypoint
  3. Continue driving and capturing waypoints
  4. Type 'end' and press ENTER to capture final goal
  5. Type 'quit' to exit without saving

NOTES:
  - You can capture as many waypoints as needed
  - The script will output YAML format at the end
  - API control is automatically disabled after each capture
======================================================================

Waypoint 1 > Press ENTER to capture, 'end' for final goal, 'quit' to exit:
Capturing waypoint...
✓ Waypoint 1 captured at (10.0, 5.0, -2.0)
  API control disabled - you can now drive manually

Waypoint 2 > Press ENTER to capture, 'end' for final goal, 'quit' to exit:
Capturing waypoint...
✓ Waypoint 2 captured at (25.0, 10.0, -2.0)
  API control disabled - you can now drive manually

Waypoint 3 > Press ENTER to capture, 'end' for final goal, 'quit' to exit: end
Capturing final goal position...
✓ Final goal captured at (45.0, 25.0, -2.0)

======================================================================
  YAML OUTPUT
======================================================================

Copy and paste this into your experiment config:

waypoints:
    - x: 10.0
      y: 5.0
      z: -2.0
      yaw: 0.0

    - x: 25.0
      y: 10.0
      z: -2.0
      yaw: 15.5

    - x: 45.0
      y: 25.0
      z: -2.0
      yaw: 30.2

======================================================================

✓ Captured 2 waypoint(s) + 1 goal

TIP: You can add more intermediate waypoints by running this script again
     and merging the outputs.
```

## Output Format

The script outputs waypoints in YAML format compatible with experiment configs:

```yaml
waypoints:
    - x: 10.0
      y: 5.0
      z: -2.0
      yaw: 15.0
    - x: 20.0
      y: 10.0
      z: -2.0
      yaw: 30.0
    # ... more waypoints
```

Each waypoint includes:

-   **x, y**: Position in meters (NED coordinates)
-   **z**: Altitude in meters (negative = above ground in NED)
-   **yaw**: Orientation in degrees

## Integration with Training Configs

### Step 1: Create waypoints

```bash
python src/scripts/create_waypoints.py
```

### Step 2: Copy output to your experiment config

Edit `configs/experiments/training_waypoints.yaml`:

```yaml
scene: Neighbourhood
vehicle: DefaultSedan

start_pose:
    x: 0.0
    y: 0.0
    z: -2.0
    yaw: 0.0

# Paste the waypoints output here:
waypoints:
    - x: 10.0
      y: 5.0
      z: -2.0
      yaw: 0.0
    - x: 25.0
      y: 10.0
      z: -2.0
      yaw: 15.5
    - x: 45.0
      y: 25.0
      z: -2.0
      yaw: 30.2

horizon: 1000
seeds:
    python: 123
    numpy: 124
    torch: 125
    airsim: 126
    deterministic: true
```

### Step 3: Train with your custom path

```bash
uv run python src/scripts/train_and_eval.py train \
  --config configs/experiments/training_waypoints.yaml \
  --episodes 100 \
  --detector-model yolo12n
```

## Tips

### Waypoint Spacing

-   **Recommended**: 10-20 meters between waypoints
-   **Curves**: Add more waypoints on tight curves
-   **Straights**: Fewer waypoints needed on straight roads

### Coordinate System

-   AirSim uses **NED coordinates** (North-East-Down)
-   **X**: North/South (positive = north)
-   **Y**: East/West (positive = east)
-   **Z**: Altitude (negative = above ground, e.g., -2.0 = 2m above ground)

### Best Practices

1. **Follow the road**: Drive along the actual road path
2. **Capture at decision points**: Waypoints at intersections, curve entries/exits
3. **Consistent altitude**: Keep z-values consistent (e.g., all at -2.0)
4. **Start position**: First waypoint should be near your start_pose
5. **Test the path**: Run a short training episode to verify the trajectory

### Troubleshooting

**Problem**: Can't drive manually

-   **Solution**: The script disables API control after each capture. If stuck, restart the script.

**Problem**: AirSim connection fails

-   **Solution**:
    1. Ensure AirSim is running
    2. Check you're using a car vehicle (not drone)
    3. Verify settings.json is correct

**Problem**: Wrong coordinates captured

-   **Solution**:
    1. Type `quit` to exit
    2. Restart and recapture from scratch
    3. Or manually edit the YAML output

**Problem**: Want to add more waypoints later

-   **Solution**:
    1. Run the script again
    2. Capture additional waypoints
    3. Manually merge the YAML outputs

## Advanced Usage

### Merging Multiple Runs

If you want to add waypoints to an existing path:

1. Run the script and capture new waypoints
2. Copy the YAML output
3. Insert the new waypoints at the appropriate position in your existing config

Example - inserting waypoint between #2 and #3:

```yaml
waypoints:
    - x: 10.0 # Waypoint 1
      y: 5.0
    - x: 25.0 # Waypoint 2
      y: 10.0
    - x: 30.0 # NEW waypoint (inserted)
      y: 15.0
    - x: 45.0 # Waypoint 3 (now #4)
      y: 25.0
```

### Visualizing Your Path

After creating waypoints, test the visualization:

1. Create a minimal experiment config with your waypoints
2. Run 1 episode with high horizon:
    ```bash
    uv run python src/scripts/train_and_eval.py train \
      --config configs/experiments/your_path.yaml \
      --episodes 1 \
      --max-steps 500
    ```
3. Check `artifacts/{run_id}/metrics/trajectory.png` to see if path looks correct

## See Also

-   `docs/waypoint-navigation.md` - Waypoint navigation architecture
-   `configs/experiments/training_waypoints.yaml` - Example waypoint config
-   `src/utils/path_manager.py` - PathManager implementation
