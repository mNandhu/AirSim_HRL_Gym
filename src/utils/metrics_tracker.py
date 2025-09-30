"""Real-time metrics tracking and visualization for HRL training."""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import matplotlib
import numpy as np

# Force a non-interactive backend so plotting from background threads is safe
try:
    matplotlib.use("Agg", force=True)
except Exception:
    # If backend cannot be changed (already set), continue with existing backend
    print("Unable to set matplotlib backend to 'Agg'; continuing with existing backend.")
    pass

import matplotlib.pyplot as plt

# Set plotting style
try:
    plt.style.use("seaborn-v0_8")  # Modern plotting style
except Exception:
    try:
        plt.style.use("seaborn")  # Fallback for older matplotlib
    except Exception:
        pass  # Use default style if seaborn not available

__all__ = ["MetricsTracker", "EpisodeMetrics", "StepMetrics"]


def _normalize_xy(value: Any) -> Optional[Tuple[float, float]]:
    """Convert various (x, y) representations into a float tuple."""

    if value is None:
        return None

    if isinstance(value, Mapping):
        if "x" not in value or "y" not in value:
            return None
        value = (value["x"], value["y"])

    if isinstance(value, np.ndarray):
        if value.size < 2:
            return None
        value = (value.flat[0], value.flat[1])

    if isinstance(value, (tuple, list)):
        if len(value) < 2:
            return None
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None

    return None  # pragma: no cover - defensive fallback for unsupported types


@dataclass
class StepMetrics:
    """Metrics for a single step within an episode."""

    step: int
    timestamp: float
    reward: float
    cumulative_reward: float
    action: Dict[str, float]
    command: Optional[str] = None
    speed_mps: float = 0.0
    distance_to_goal: float = 0.0
    collision: bool = False

    # Reward components breakdown
    command_shaping: float = 0.0
    collision_penalty: float = 0.0
    completion_bonus: float = 0.0
    idle_penalty: float = 0.0
    time_penalty: float = 0.0
    position_xy: Optional[Tuple[float, float]] = None
    goal_xy: Optional[Tuple[float, float]] = None


@dataclass
class EpisodeMetrics:
    """Metrics for a complete episode."""

    episode: int
    start_time: float
    end_time: float
    total_steps: int
    cumulative_reward: float
    max_speed: float = 0.0
    min_distance_to_goal: float = float("inf")
    collision_occurred: bool = False
    completed_successfully: bool = False

    # Action statistics
    avg_target_speed: float = 0.0
    avg_target_steering_magnitude: float = 0.0

    # Command statistics
    command_distribution: Dict[str, int] = field(default_factory=dict)


class MetricsTracker:
    """Tracks and visualizes training metrics in real-time."""

    def __init__(self, artifacts_dir: Path, update_interval: int = 10, async_plots: bool = True):
        """
        Initialize the metrics tracker.

        Args:
            artifacts_dir: Directory to save metrics and graphs
            update_interval: Update graphs every N steps (to avoid excessive I/O)
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.metrics_dir = self.artifacts_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.update_interval = update_interval
        self._async_plots = async_plots
        self._plot_thread: Optional[threading.Thread] = None
        self.step_counter = 0

        # Data storage
        self.episode_metrics: List[EpisodeMetrics] = []
        self.current_episode_steps: List[StepMetrics] = []
        self.current_episode: Optional[int] = None
        self.current_episode_start_time: Optional[float] = None

        # Rolling statistics for performance
        self.recent_rewards = deque(maxlen=100)  # Last 100 steps
        self.recent_speeds = deque(maxlen=100)

        # Position tracking for trajectory plotting
        self.current_episode_positions: List[Tuple[float, float]] = []
        self.current_target_xy: Optional[Tuple[float, float]] = None

        # Plotting setup
        plt.ioff()  # Turn off interactive mode for better performance

    def start_episode(self, episode: int) -> None:
        """Start tracking a new episode."""
        # Finish previous episode if exists
        if self.current_episode is not None:
            self._finish_episode()

        self.current_episode = episode
        self.current_episode_start_time = time.time()
        self.current_episode_steps = []
        self.current_episode_positions = []
        self.current_target_xy = None
        print(f"📊 Started tracking episode {episode}")

    def log_step(
        self,
        step: int,
        reward: float,
        action: Dict[str, float],
        telemetry: Dict[str, Any],
        reward_components: Dict[str, float],
        command: Optional[str] = None,
        next_telemetry: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log metrics for a single step."""
        if self.current_episode is None:
            return

        # Calculate cumulative reward
        cumulative_reward = sum(s.reward for s in self.current_episode_steps) + reward

        # Extract telemetry data
        speed_mps = telemetry.get("speed_mps", 0.0)
        distance_to_goal = telemetry.get("distance_to_goal", 0.0)
        collision = telemetry.get("collision", False)
        position_xy = _normalize_xy(telemetry.get("position_xy"))
        goal_xy = _normalize_xy(telemetry.get("goal_xy"))

        next_position_xy: Optional[Tuple[float, float]] = None
        next_goal_xy: Optional[Tuple[float, float]] = None
        if next_telemetry is not None:
            next_position_xy = _normalize_xy(next_telemetry.get("position_xy"))
            next_goal_xy = _normalize_xy(next_telemetry.get("goal_xy"))

        if goal_xy is None and next_goal_xy is not None:
            goal_xy = next_goal_xy

        self._record_position(position_xy)
        self._record_position(next_position_xy)

        if goal_xy is None and self.current_target_xy is not None:
            goal_xy = self.current_target_xy

        if goal_xy is not None:
            self.current_target_xy = goal_xy

        # Create step metrics
        step_metrics = StepMetrics(
            step=step,
            timestamp=time.time(),
            reward=reward,
            cumulative_reward=cumulative_reward,
            action=action.copy(),
            command=command,
            speed_mps=speed_mps,
            distance_to_goal=distance_to_goal,
            collision=collision,
            command_shaping=reward_components.get("command_shaping", 0.0),
            collision_penalty=reward_components.get("collision_penalty", 0.0),
            completion_bonus=reward_components.get("completion_bonus", 0.0),
            idle_penalty=reward_components.get("idle_penalty", 0.0),
            time_penalty=reward_components.get("time_penalty", 0.0),
            position_xy=position_xy,
            goal_xy=goal_xy,
        )

        self.current_episode_steps.append(step_metrics)

        # Update rolling statistics
        self.recent_rewards.append(reward)
        self.recent_speeds.append(speed_mps)

        # Update graphs periodically
        self.step_counter += 1
        if self.update_interval and self.update_interval > 0:
            if self.step_counter % self.update_interval == 0:
                if self._async_plots:
                    self._schedule_async_plot()
                else:
                    self._update_graphs()

    def _schedule_async_plot(self) -> None:
        """Schedule a non-blocking plot update if one isn't already running."""
        # If a previous plot is still running, skip scheduling a new one
        if self._plot_thread is not None and self._plot_thread.is_alive():
            return

        def _worker() -> None:
            try:
                self._update_graphs()
            except Exception as e:  # noqa: BLE001
                print(f"⚠️  Warning: Failed to update graphs (async): {e}")

        self._plot_thread = threading.Thread(target=_worker, name="metrics-plotter", daemon=True)
        self._plot_thread.start()

    def _record_position(self, position: Optional[Tuple[float, float]]) -> None:
        """Store a new (x, y) waypoint if it differs from the previous sample."""

        if position is None:
            return

        if self.current_episode_positions:
            last_x, last_y = self.current_episode_positions[-1]
            new_x, new_y = position
            if abs(last_x - new_x) < 1e-6 and abs(last_y - new_y) < 1e-6:
                return

        self.current_episode_positions.append(position)

    def finish_episode(self, completed_successfully: bool = False) -> None:
        """Finish the current episode."""
        if self.current_episode is None:
            return

        self._finish_episode(completed_successfully)
        # Force a synchronous update after episode completion so episode artifacts are current
        self._update_graphs()
        self._save_metrics()

        # Reset state so repeated calls do not duplicate episode metrics.
        self.current_episode_steps = []
        self.current_episode = None
        self.current_episode_start_time = None
        self.current_episode_positions = []
        self.current_target_xy = None

    def _finish_episode(self, completed_successfully: bool = False) -> None:
        """Internal method to finish episode tracking."""
        if (
            not self.current_episode_steps
            or self.current_episode is None
            or self.current_episode_start_time is None
        ):
            return

        end_time = time.time()

        # Calculate episode statistics
        total_reward = self.current_episode_steps[-1].cumulative_reward
        total_steps = len(self.current_episode_steps)

        # Action statistics
        target_speeds = [
            s.action.get("target_speed", s.action.get("throttle", 0.0))
            for s in self.current_episode_steps
        ]
        target_steerings = [
            abs(s.action.get("target_steering", s.action.get("steering", 0.0)))
            for s in self.current_episode_steps
        ]

        # Command distribution
        command_dist = defaultdict(int)
        for step in self.current_episode_steps:
            if step.command:
                command_dist[step.command] += 1

        # Performance metrics
        speeds = [s.speed_mps for s in self.current_episode_steps]
        distances = [s.distance_to_goal for s in self.current_episode_steps]
        collision_occurred = any(s.collision for s in self.current_episode_steps)

        episode_metrics = EpisodeMetrics(
            episode=self.current_episode,
            start_time=self.current_episode_start_time,
            end_time=end_time,
            total_steps=total_steps,
            cumulative_reward=total_reward,
            max_speed=max(speeds) if speeds else 0.0,
            min_distance_to_goal=min(distances) if distances else float("inf"),
            collision_occurred=collision_occurred,
            completed_successfully=completed_successfully,
            avg_target_speed=float(np.mean(target_speeds)) if target_speeds else 0.0,
            avg_target_steering_magnitude=float(np.mean(target_steerings))
            if target_steerings
            else 0.0,
            command_distribution=dict(command_dist),
        )

        self.episode_metrics.append(episode_metrics)

        # Print episode summary
        duration = end_time - self.current_episode_start_time
        print(f"📊 Episode {self.current_episode} completed:")
        print(f"   Reward: {total_reward:.2f} | Steps: {total_steps} | Duration: {duration:.1f}s")
        print(
            f"   Max Speed: {episode_metrics.max_speed:.1f} m/s | Min Distance: {episode_metrics.min_distance_to_goal:.1f}m"
        )

    def _update_graphs(self) -> None:
        """Update all visualization graphs."""
        try:
            self._plot_reward_trends()
            self._plot_episode_summary()
            self._plot_action_analysis()
            self._plot_performance_metrics()
            self._plot_trajectory_map()
        except Exception as e:
            print(f"⚠️  Warning: Failed to update graphs: {e}")

    def _plot_reward_trends(self) -> None:
        """Plot real-time reward trends."""
        if not self.current_episode_steps:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle("Reward Trends", fontsize=16, fontweight="bold")

        # Step-by-step reward
        steps = [s.step for s in self.current_episode_steps]
        rewards = [s.reward for s in self.current_episode_steps]
        cumulative_rewards = [s.cumulative_reward for s in self.current_episode_steps]

        ax1.plot(steps, rewards, "b-", alpha=0.7, linewidth=1, label="Step Reward")
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax1.set_ylabel("Reward")
        ax1.set_title(f"Step Rewards (Episode {self.current_episode})")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Add moving average
        if len(rewards) >= 10:
            window = min(10, len(rewards))
            moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax1.plot(steps[window - 1 :], moving_avg, "r-", linewidth=2, label=f"{window}-step MA")

        # Cumulative reward
        ax2.plot(steps, cumulative_rewards, "g-", linewidth=2, label="Cumulative Reward")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Cumulative Reward")
        ax2.set_title("Cumulative Reward Progress")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        # Save latest and per-episode variants for comparison across episodes
        plt.savefig(self.metrics_dir / "reward.png", dpi=100, bbox_inches="tight")
        try:
            if self.current_episode is not None:
                ep_path = self.metrics_dir / f"reward_ep{self.current_episode}.png"
                plt.savefig(ep_path, dpi=100, bbox_inches="tight")
        except Exception:
            # Proceed even if per-episode save fails
            pass
        plt.close()

    def _plot_episode_summary(self) -> None:
        """Plot episode-level summary statistics."""
        if not self.episode_metrics:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Episode Summary", fontsize=16, fontweight="bold")

        episodes = [e.episode for e in self.episode_metrics]
        rewards = [e.cumulative_reward for e in self.episode_metrics]
        steps = [e.total_steps for e in self.episode_metrics]
        max_speeds = [e.max_speed for e in self.episode_metrics]
        min_distances = [e.min_distance_to_goal for e in self.episode_metrics]

        # Episode rewards
        ax1.bar(episodes, rewards, alpha=0.7, color="skyblue", edgecolor="navy")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.set_title("Episode Rewards")
        ax1.grid(True, alpha=0.3)

        # Episode length
        ax2.bar(episodes, steps, alpha=0.7, color="lightcoral", edgecolor="darkred")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Steps")
        ax2.set_title("Episode Length")
        ax2.grid(True, alpha=0.3)

        # Max speed per episode
        ax3.bar(episodes, max_speeds, alpha=0.7, color="lightgreen", edgecolor="darkgreen")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Max Speed (m/s)")
        ax3.set_title("Max Speed Achieved")
        ax3.grid(True, alpha=0.3)

        # Min distance to goal
        ax4.bar(episodes, min_distances, alpha=0.7, color="gold", edgecolor="orange")
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Min Distance (m)")
        ax4.set_title("Closest to Goal")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.metrics_dir / "episode_summary.png", dpi=100, bbox_inches="tight")
        plt.close()

    def _plot_action_analysis(self) -> None:
        """Plot action distribution and trends."""
        if not self.current_episode_steps:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Action Analysis (Episode {self.current_episode})", fontsize=16, fontweight="bold"
        )

        steps = [s.step for s in self.current_episode_steps]
        target_speeds = [
            s.action.get("target_speed", s.action.get("throttle", 0.0))
            for s in self.current_episode_steps
        ]
        speeds = [s.speed_mps for s in self.current_episode_steps]
        target_steerings = [
            s.action.get("target_steering", s.action.get("steering", 0.0))
            for s in self.current_episode_steps
        ]

        # Target speed over time
        ax1.plot(steps, target_speeds, "r-", alpha=0.7, linewidth=1)
        ax1.fill_between(steps, target_speeds, alpha=0.3, color="red")
        ax1.set_ylabel("Target Speed (m/s)")
        ax1.set_title("Speed Setpoint")
        ax1.grid(True, alpha=0.3)

        # Actual speed over time
        ax2.plot(steps, speeds, "b-", alpha=0.7, linewidth=1)
        ax2.fill_between(steps, speeds, alpha=0.3, color="blue")
        ax2.set_ylabel("Speed (m/s)")
        ax2.set_title("Measured Speed")
        ax2.grid(True, alpha=0.3)

        # Steering over time
        ax3.plot(steps, target_steerings, "g-", alpha=0.7, linewidth=1)
        ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Target Steering")
        ax3.set_title("Steering Setpoint")
        ax3.set_ylim(-1, 1)
        ax3.grid(True, alpha=0.3)

        # Action distribution histogram
        ax4.hist(
            [target_speeds, [abs(s) for s in target_steerings]],
            bins=20,
            alpha=0.7,
            label=["Target Speed", "Abs(Target Steering)"],
        )
        ax4.set_xlabel("Action Value")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Action Distribution")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        # Save latest and per-episode variants
        plt.savefig(self.metrics_dir / "action_analysis.png", dpi=100, bbox_inches="tight")
        try:
            if self.current_episode is not None:
                ep_path = self.metrics_dir / f"action_analysis_ep{self.current_episode}.png"
                plt.savefig(ep_path, dpi=100, bbox_inches="tight")
        except Exception:
            pass
        plt.close()

    def _plot_performance_metrics(self) -> None:
        """Plot performance metrics like speed and distance."""
        if not self.current_episode_steps:
            return

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(
            f"Performance Metrics (Episode {self.current_episode})", fontsize=16, fontweight="bold"
        )

        steps = [s.step for s in self.current_episode_steps]
        speeds = [s.speed_mps for s in self.current_episode_steps]
        distances = [s.distance_to_goal for s in self.current_episode_steps]

        # Reward components
        command_shaping = [s.command_shaping for s in self.current_episode_steps]
        collision_penalties = [s.collision_penalty for s in self.current_episode_steps]
        completion_bonuses = [s.completion_bonus for s in self.current_episode_steps]
        idle_penalties = [s.idle_penalty for s in self.current_episode_steps]
        time_penalties = [s.time_penalty for s in self.current_episode_steps]

        # Speed over time
        ax1.plot(steps, speeds, "purple", linewidth=2, label="Speed")
        ax1.fill_between(steps, speeds, alpha=0.3, color="purple")
        ax1.set_ylabel("Speed (m/s)")
        ax1.set_title("Vehicle Speed")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Distance to goal
        ax2.plot(steps, distances, "orange", linewidth=2, label="Distance to Goal")
        ax2.set_ylabel("Distance (m)")
        ax2.set_title("Distance to Goal")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Reward components stacked
        ax3.fill_between(
            steps, 0, command_shaping, alpha=0.7, label="Command Shaping", color="green"
        )

        collision_base = [
            cs + cp for cs, cp in zip(command_shaping, collision_penalties, strict=False)
        ]
        ax3.fill_between(
            steps,
            command_shaping,
            collision_base,
            alpha=0.7,
            label="Collision Penalty",
            color="red",
        )

        completion_base = [
            cs + cp + cb
            for cs, cp, cb in zip(
                command_shaping, collision_penalties, completion_bonuses, strict=False
            )
        ]
        ax3.fill_between(
            steps,
            collision_base,
            completion_base,
            alpha=0.7,
            label="Completion Bonus",
            color="gold",
        )

        idle_base = [
            cs + cp + cb + ip
            for cs, cp, cb, ip in zip(
                command_shaping,
                collision_penalties,
                completion_bonuses,
                idle_penalties,
                strict=False,
            )
        ]
        ax3.fill_between(
            steps, completion_base, idle_base, alpha=0.7, label="Idle Penalty", color="gray"
        )

        time_base = [
            cs + cp + cb + ip + tp
            for cs, cp, cb, ip, tp in zip(
                command_shaping,
                collision_penalties,
                completion_bonuses,
                idle_penalties,
                time_penalties,
                strict=False,
            )
        ]
        ax3.fill_between(
            steps, idle_base, time_base, alpha=0.7, label="Time Penalty", color="black"
        )

        ax3.set_xlabel("Step")
        ax3.set_ylabel("Reward Component")
        ax3.set_title("Reward Component Breakdown")
        ax3.legend(loc="upper right")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        # Save latest and per-episode variants
        plt.savefig(self.metrics_dir / "performance_metrics.png", dpi=100, bbox_inches="tight")
        try:
            if self.current_episode is not None:
                ep_path = self.metrics_dir / f"performance_metrics_ep{self.current_episode}.png"
                plt.savefig(ep_path, dpi=100, bbox_inches="tight")
        except Exception:
            pass
        plt.close()

    def _plot_trajectory_map(self) -> None:
        """Render a top-down trajectory map for the current episode."""

        if not self.current_episode_positions:
            return

        positions = np.asarray(self.current_episode_positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] < 2:
            return

        xs = positions[:, 0]
        ys = positions[:, 1]

        extent_xs = xs
        extent_ys = ys
        if self.current_target_xy is not None:
            tx, ty = self.current_target_xy
            extent_xs = np.append(extent_xs, tx)
            extent_ys = np.append(extent_ys, ty)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(xs, ys, color="navy", linewidth=2, label="Trajectory")
        ax.scatter(xs[0], ys[0], color="green", s=60, label="Start", zorder=3)
        ax.scatter(xs[-1], ys[-1], color="orange", s=60, label="Latest", zorder=3)

        if self.current_target_xy is not None:
            tx, ty = self.current_target_xy
            ax.scatter(tx, ty, color="red", marker="*", s=140, label="Target", zorder=4)

        min_x, max_x = float(np.min(extent_xs)), float(np.max(extent_xs))
        min_y, max_y = float(np.min(extent_ys)), float(np.max(extent_ys))

        if np.isclose(min_x, max_x):
            min_x -= 1.0
            max_x += 1.0
        if np.isclose(min_y, max_y):
            min_y -= 1.0
            max_y += 1.0

        pad_x = max(0.5, (max_x - min_x) * 0.1)
        pad_y = max(0.5, (max_y - min_y) * 0.1)

        ax.set_xlim(min_x - pad_x, max_x + pad_x)
        ax.set_ylim(min_y - pad_y, max_y + pad_y)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        title_episode = self.current_episode if self.current_episode is not None else "latest"
        ax.set_title(f"Vehicle Trajectory (Episode {title_episode})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        plt.tight_layout()
        plt.savefig(self.metrics_dir / "trajectory.png", dpi=100, bbox_inches="tight")
        try:
            if self.current_episode is not None:
                ep_path = self.metrics_dir / f"trajectory_ep{self.current_episode}.png"
                plt.savefig(ep_path, dpi=100, bbox_inches="tight")
        except Exception:
            pass
        plt.close()

    def _save_metrics(self) -> None:
        """Save metrics data to JSON files."""
        try:
            # Save episode metrics
            episodes_data = []
            for ep in self.episode_metrics:
                episodes_data.append(
                    {
                        "episode": ep.episode,
                        "start_time": ep.start_time,
                        "end_time": ep.end_time,
                        "duration": ep.end_time - ep.start_time,
                        "total_steps": ep.total_steps,
                        "cumulative_reward": ep.cumulative_reward,
                        "max_speed": ep.max_speed,
                        "min_distance_to_goal": ep.min_distance_to_goal,
                        "collision_occurred": ep.collision_occurred,
                        "completed_successfully": ep.completed_successfully,
                        "avg_target_speed": ep.avg_target_speed,
                        "avg_target_steering_magnitude": ep.avg_target_steering_magnitude,
                        "command_distribution": ep.command_distribution,
                    }
                )

            with open(self.metrics_dir / "episodes.json", "w") as f:
                json.dump(episodes_data, f, indent=2)

            # Save current episode step data
            if self.current_episode_steps:
                steps_data = []
                for step in self.current_episode_steps:
                    steps_data.append(
                        {
                            "step": step.step,
                            "timestamp": step.timestamp,
                            "reward": step.reward,
                            "cumulative_reward": step.cumulative_reward,
                            "action": step.action,
                            "command": step.command,
                            "speed_mps": step.speed_mps,
                            "distance_to_goal": step.distance_to_goal,
                            "collision": step.collision,
                            "position_xy": list(step.position_xy)
                            if step.position_xy is not None
                            else None,
                            "goal_xy": list(step.goal_xy) if step.goal_xy is not None else None,
                            "reward_components": {
                                "command_shaping": step.command_shaping,
                                "collision_penalty": step.collision_penalty,
                                "completion_bonus": step.completion_bonus,
                                "idle_penalty": step.idle_penalty,
                                "time_penalty": step.time_penalty,
                            },
                        }
                    )

                with open(
                    self.metrics_dir / f"episode_{self.current_episode}_steps.json", "w"
                ) as f:
                    json.dump(steps_data, f, indent=2)

                if self.current_episode_positions and self.current_episode is not None:
                    trajectory_payload = {
                        "positions": [
                            [float(pos[0]), float(pos[1])] for pos in self.current_episode_positions
                        ],
                        "target": [
                            float(self.current_target_xy[0]),
                            float(self.current_target_xy[1]),
                        ]
                        if self.current_target_xy is not None
                        else None,
                    }

                    with open(
                        self.metrics_dir / f"trajectory_ep{self.current_episode}.json", "w"
                    ) as f:
                        json.dump(trajectory_payload, f, indent=2)

        except Exception as e:
            print(f"⚠️  Warning: Failed to save metrics: {e}")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the current training session."""
        if not self.episode_metrics:
            return {}

        rewards = [e.cumulative_reward for e in self.episode_metrics]
        steps = [e.total_steps for e in self.episode_metrics]
        speeds = [e.max_speed for e in self.episode_metrics]

        return {
            "total_episodes": len(self.episode_metrics),
            "avg_reward": np.mean(rewards),
            "best_reward": max(rewards),
            "worst_reward": min(rewards),
            "avg_episode_length": np.mean(steps),
            "avg_max_speed": np.mean(speeds),
            "collision_rate": sum(e.collision_occurred for e in self.episode_metrics)
            / len(self.episode_metrics),
            "success_rate": sum(e.completed_successfully for e in self.episode_metrics)
            / len(self.episode_metrics),
        }
