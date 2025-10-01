# Waypoint Creator Utility - Implementation Summary

## Overview

Created `src/scripts/create_waypoints.py` - An interactive utility to create waypoint paths by manually driving in AirSim and marking positions.

## Purpose

Solves the problem of defining accurate waypoint paths for training:

-   **Before**: Manually guessing coordinates or using trial-and-error
-   **After**: Drive the desired path, press Enter to mark waypoints, get instant YAML output

## Key Features

### 1. Interactive Waypoint Capture

-   Drive manually in AirSim (full control)
-   Press **ENTER** to capture current position as waypoint
-   Type **`end`** to capture final goal
-   Type **`quit`** to exit without saving

### 2. Automatic API Control Management

-   **Critical**: Disables API control after each capture
-   User can drive freely between waypoints
-   Briefly enables API control only to read position
-   Prevents control conflicts between script and manual driving

### 3. Complete Pose Capture

Each waypoint includes:

-   **Position**: x, y, z coordinates (NED)
-   **Orientation**: yaw angle in degrees
-   **Altitude**: z-value for elevation

### 4. YAML Output

Generates properly formatted YAML ready to paste into experiment configs:

```yaml
waypoints:
    - x: 10.0
      y: 5.0
      z: -2.0
      yaw: 15.0
    - x: 25.0
      y: 10.0
      z: -2.0
      yaw: 30.5
```

## Usage

### Basic

```bash
python src/scripts/create_waypoints.py
```

### Workflow

1. Script connects to AirSim
2. Shows instructions
3. User drives manually to first waypoint → Press ENTER
4. User drives to next waypoint → Press ENTER
5. Repeat for all waypoints
6. User drives to goal → Type "end"
7. Script outputs YAML

### Integration with Training

Copy YAML output → Paste into experiment config → Train immediately

## Technical Implementation

### Connection Handling

```python
def connect_airsim() -> airsim.CarClient:
    """Connect with error handling and helpful messages"""
```

### Pose Reading

```python
def get_current_pose(client: airsim.CarClient) -> dict[str, float]:
    """
    - Enable API control temporarily
    - Read car state (position, orientation)
    - Convert quaternion to Euler angles (yaw)
    - Disable API control immediately
    """
```

### YAML Formatting

```python
def format_yaml_waypoint(pose: dict, indent: int = 4) -> str:
    """Format pose as indented YAML with proper spacing"""
```

### Main Loop

```python
while True:
    user_input = input("Waypoint N > ...")

    if user_input == "":
        # Capture waypoint
    elif user_input == "end":
        # Capture goal and break
    elif user_input == "quit":
        # Exit without saving
```

## Error Handling

### Connection Failures

-   Checks for AirSim connection
-   Provides troubleshooting steps if connection fails
-   Verifies car vehicle (not drone)

### User Interrupts

-   Handles Ctrl+C gracefully
-   Allows quit at any time
-   No waypoints lost if interrupted

### Edge Cases

-   No waypoints captured: Shows warning
-   Only goal, no intermediate waypoints: Recommends adding waypoints
-   Invalid commands: Shows help message

## Safety Features

1. **API Control Management**

    - Always disables after reading position
    - Prevents control conflicts
    - User has full control between captures

2. **Confirmation Messages**

    - Shows captured position coordinates
    - Displays waypoint count
    - Confirms when ready to drive again

3. **Clear Instructions**
    - Prints full instructions on start
    - Shows command options in prompt
    - Provides usage tips

## Output Example

```
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

## Documentation

Created `docs/waypoint-creator-utility.md` with:

-   Detailed usage instructions
-   Example session walkthrough
-   Integration guide with training configs
-   Tips for waypoint spacing and best practices
-   Troubleshooting section
-   Advanced usage (merging multiple runs)

Updated `README.md` with:

-   New "Waypoint Creation Utility" section
-   Quick start instructions
-   Link to detailed documentation

## Testing

Verified:

-   ✅ Script can be imported
-   ✅ YAML formatting function works correctly
-   ✅ No syntax or lint errors
-   ✅ Proper error handling for missing airsim package

## Benefits

1. **Accuracy**: Capture exact positions from simulator
2. **Speed**: Faster than manual coordinate entry
3. **Visibility**: See the path while creating it
4. **Flexibility**: Easy to add more waypoints later
5. **Integration**: Direct YAML output for configs

## Future Enhancements

Possible additions:

-   Save waypoints to file automatically
-   Load and edit existing waypoint files
-   Visualize captured path in real-time
-   Support for multiple paths (different routes)
-   Undo last waypoint capture
-   Distance calculation between waypoints

## Files Created

-   `src/scripts/create_waypoints.py` - Main utility script (221 lines)
-   `docs/waypoint-creator-utility.md` - Comprehensive documentation (365 lines)
-   `README.md` - Updated with utility section

## Usage in Development

This utility is particularly useful when:

-   Setting up new training scenarios
-   Designing complex paths with curves
-   Testing different route strategies
-   Creating validation paths for evaluation
-   Debugging path-following behavior
