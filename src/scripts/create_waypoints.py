"""
Utility script to create waypoints by manually driving in AirSim.

This script helps you define a path by:
1. Manually driving to a position in the simulator
2. Pressing Enter to capture that position as a waypoint
3. Typing "end" and pressing Enter to capture the final goal
4. Outputs YAML-formatted waypoints ready for training configs

Usage:
    python src/scripts/create_waypoints.py [--settings SETTINGS_PATH]

Controls:
    - Drive manually to desired waypoint location
    - Press ENTER to capture current position as waypoint
    - Type "end" and press ENTER to capture final goal and finish
    - Type "quit" to exit without saving

The script automatically:
    - Disables API control after each capture (so you can drive)
    - Re-enables API control only to read position
    - Captures position, orientation (yaw), and altitude (z)
"""

import argparse
import sys

try:
    import airsim
except ImportError:
    print("ERROR: airsim package not found. Please install it:")
    print("  pip install airsim")
    sys.exit(1)


def connect_airsim() -> airsim.CarClient:
    """Connect to AirSim and return client."""
    try:
        client = airsim.CarClient()
        client.confirmConnection()
        return client
    except Exception as e:
        print(f"ERROR: Failed to connect to AirSim: {e}")
        print("\nMake sure:")
        print("  1. AirSim is running")
        print("  2. You're using the car vehicle (not drone)")
        sys.exit(1)


def get_current_pose(client: airsim.CarClient) -> dict[str, float]:
    """
    Get current vehicle pose with API control temporarily enabled.

    Returns dict with keys: x, y, z, yaw (in degrees)
    """
    # Temporarily enable API control to read state
    client.enableApiControl(True)

    try:
        # Get car state
        car_state = client.getCarState()
        position = car_state.kinematics_estimated.position
        orientation = car_state.kinematics_estimated.orientation

        # Convert quaternion to Euler angles (get yaw)
        pitch, roll, yaw = airsim.to_eularian_angles(orientation)

        pose = {
            "x": float(position.x_val),
            "y": float(position.y_val),
            "z": float(position.z_val),
            "yaw": float(yaw),  # in radians
        }

        return pose

    finally:
        # CRITICAL: Disable API control so user can drive manually
        client.enableApiControl(False)


def format_yaml_waypoint(pose: dict[str, float], indent: int = 4) -> str:
    """Format a pose as YAML waypoint entry."""
    import math

    spaces = " " * indent
    yaw_deg = math.degrees(pose["yaw"])
    return (
        f"{spaces}- x: {pose['x']:.1f}\n"
        f"{spaces}  y: {pose['y']:.1f}\n"
        f"{spaces}  z: {pose['z']:.1f}\n"
        f"{spaces}  yaw: {yaw_deg:.1f}"
    )


def print_instructions():
    """Print usage instructions."""
    print("\n" + "=" * 70)
    print("  AirSim Waypoint Creator")
    print("=" * 70)
    print("\nINSTRUCTIONS:")
    print("  1. Drive manually to your desired waypoint location")
    print("  2. Press ENTER to capture that position as a waypoint")
    print("  3. Continue driving and capturing waypoints")
    print("  4. Type 'end' and press ENTER to capture final goal")
    print("  5. Type 'quit' to exit without saving")
    print("\nNOTES:")
    print("  - You can capture as many waypoints as needed")
    print("  - The script will output YAML format at the end")
    print("  - API control is automatically disabled after each capture")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create waypoints by manually driving in AirSim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--settings",
        type=str,
        default="settings.json",
        help="Path to AirSim settings.json (default: settings.json)",
    )
    # Parse args for future use (e.g., custom settings)
    _ = parser.parse_args()

    # Connect to AirSim
    print("Connecting to AirSim...")
    client = connect_airsim()
    print("✓ Connected to AirSim\n")

    # Ensure API control is disabled initially
    client.enableApiControl(False)

    # Print instructions
    print_instructions()

    # Collect waypoints
    waypoints = []
    final_goal = None

    try:
        while True:
            # Wait for user input
            user_input = (
                input(
                    f"Waypoint {len(waypoints) + 1} > "
                    "Press ENTER to capture, 'end' for final goal, 'quit' to exit: "
                )
                .strip()
                .lower()
            )

            if user_input == "quit":
                print("\n❌ Exiting without saving waypoints.")
                sys.exit(0)

            elif user_input == "end":
                # Capture final goal
                print("Capturing final goal position...")
                final_goal = get_current_pose(client)
                print(
                    f"✓ Final goal captured at ({final_goal['x']:.1f}, "
                    f"{final_goal['y']:.1f}, {final_goal['z']:.1f})"
                )
                break

            elif user_input == "":
                # Capture waypoint
                print("Capturing waypoint...")
                pose = get_current_pose(client)
                waypoints.append(pose)
                print(
                    f"✓ Waypoint {len(waypoints)} captured at "
                    f"({pose['x']:.1f}, {pose['y']:.1f}, {pose['z']:.1f})"
                )
                print("  API control disabled - you can now drive manually\n")

            else:
                print(f"⚠️  Unknown command: '{user_input}'. Use ENTER, 'end', or 'quit'.\n")

    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user. Exiting without saving.")
        sys.exit(0)

    # Ensure we have at least one waypoint
    if not waypoints and final_goal is None:
        print("\n⚠️  No waypoints or goal captured. Exiting.")
        sys.exit(0)

    # If we have final goal but no waypoints, use start position as first waypoint
    if final_goal and not waypoints:
        print("\n⚠️  No intermediate waypoints captured.")
        print("Recommendation: Add at least one waypoint between start and goal.")

    # Output YAML format
    print("\n" + "=" * 70)
    print("  YAML OUTPUT")
    print("=" * 70)
    print("\nCopy and paste this into your experiment config:\n")

    print("waypoints:")

    # Print all waypoints
    for i, wp in enumerate(waypoints, 1):
        print(format_yaml_waypoint(wp))
        if i < len(waypoints) or final_goal:  # Add blank line between waypoints
            print()

    # Print final goal as last waypoint
    if final_goal:
        print(format_yaml_waypoint(final_goal))

    print("\n" + "=" * 70)
    print(f"\n✓ Captured {len(waypoints)} waypoint(s) + 1 goal")
    print("\nTIP: You can add more intermediate waypoints by running this script again")
    print("     and merging the outputs.\n")


if __name__ == "__main__":
    main()
