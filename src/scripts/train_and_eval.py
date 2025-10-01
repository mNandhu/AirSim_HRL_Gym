"""Train HRL agents in AirSim environment with live visualization."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np

from airsim_env.env import AirSimEnv
from config.loader import load_experiment
from hrl_agent.coordination import CommandCoordinator
from hrl_agent.manager.dqn_manager import CommandPolicy, DQNManager
from hrl_agent.orchestrator import HRLOrchestrator
from hrl_agent.workers.sac_worker import SACWorker
from perception.async_pipeline import AsyncPerceptionPipeline
from perception.detector import ModelLoadError, YoloDetector
from perception.pipeline import PerceptionPipeline
from perception.segmentation import SegmentationAdapter
from utils.airsim_runner import airsim_session
from utils.artifacts import ArtifactManager
from utils.metrics_tracker import MetricsTracker

try:
    import airsim
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install airsim")
    exit(1)

try:
    import stable_baselines3  # type: ignore  # noqa: F401
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install stable-baselines3")
    exit(1)


class AirSimSimulatorAdapter:
    """Real AirSim simulator adapter for training."""

    def __init__(
        self,
        client: Any,
        *,
        horizon: int,
        control_dt: float = 0.1,
        enable_rgb: bool = True,
    ) -> None:
        self._client = client
        self._step = 0
        self._horizon = horizon
        self._initial_pose = None
        self._control_dt = control_dt
        self._previous_velocity = None
        self._previous_timestamp = None
        self._goal_pose = None
        self._waypoints = None
        self._current_waypoint_index = 0
        self._accumulator_time = 0.0
        self._enable_rgb = enable_rgb

    def reset(self, experiment) -> dict[str, Any]:
        self._step = 0
        pose = experiment.start_pose
        self._goal_pose = experiment.goal_pose

        # Support waypoint-based navigation
        if hasattr(experiment, "waypoints") and experiment.waypoints:
            self._waypoints = experiment.waypoints
            self._current_waypoint_index = 0
        else:
            self._waypoints = None

        # Store initial pose for resets
        if self._initial_pose is None:
            self._initial_pose = pose

        # Build desired pose
        position = airsim.Vector3r(pose.x, pose.y, pose.z)
        orientation = airsim.to_quaternion(0, 0, pose.yaw)
        vehicle_pose = airsim.Pose(position, orientation)

        # Pause, reset, teleport, unpause
        try:
            self._client.simPause(True)
        except Exception:
            pass
        try:
            self._client.reset()
        except Exception:
            pass
        try:
            self._client.simSetVehiclePose(vehicle_pose, True)
        except Exception:
            pass
        try:
            self._client.simPause(False)
        except Exception:
            pass

        # Wait for reported pose to settle near target
        target_xy = (float(pose.x), float(pose.y))
        settle_tol_m = 0.5
        timeout_s = 2.0
        poll_dt = 0.02
        start_t = time.monotonic()
        last_state = None
        while time.monotonic() - start_t < timeout_s:
            try:
                car_state = self._client.getCarState()
                last_state = car_state
                pos = car_state.kinematics_estimated.position
                dx = float(pos.x_val) - target_xy[0]
                dy = float(pos.y_val) - target_xy[1]
                if (dx * dx + dy * dy) ** 0.5 <= settle_tol_m:
                    break
            except Exception:
                pass
            time.sleep(poll_dt)

        self._accumulator_time = 0.0

        # Get initial state (with fallback)
        try:
            car_state = self._client.getCarState()
        except Exception:
            car_state = last_state if last_state is not None else None
        if car_state is None:
            # Construct minimal stub
            class _V3:
                def __init__(self, x=0.0, y=0.0, z=0.0):
                    self.x_val = x
                    self.y_val = y
                    self.z_val = z

            class _Kin:
                def __init__(self):
                    self.position = _V3(target_xy[0], target_xy[1], float(pose.z))
                    self.linear_velocity = _V3(0.0, 0.0, 0.0)
                    self.orientation = airsim.to_quaternion(0, 0, pose.yaw)

            class _State:
                def __init__(self):
                    self.kinematics_estimated = _Kin()
                    self.timestamp = int(time.time() * 1e9)

            car_state = _State()

        self._previous_velocity = _vector_from_airsim(
            car_state.kinematics_estimated.linear_velocity
        )
        self._previous_timestamp = getattr(car_state, "timestamp", None)
        return self._build_state_dict(car_state, experiment)

    def step(self, action: dict[str, float]) -> dict[str, Any]:
        # Apply vehicle controls
        throttle = max(0, min(1, action.get("throttle", 0)))
        brake = max(0, min(1, action.get("brake", 0)))
        steering = max(-1, min(1, action.get("steering", 0)))

        car_controls = airsim.CarControls()
        car_controls.throttle = throttle
        car_controls.brake = brake
        car_controls.steering = steering

        self._client.setCarControls(car_controls)
        self._advance_simulation()

        self._step += 1
        car_state = self._client.getCarState()

        return self._build_state_dict(car_state)

    def _build_state_dict(self, car_state, experiment=None) -> dict[str, Any]:
        # Get collision info
        collision_info = self._client.simGetCollisionInfo()
        has_collision = collision_info.has_collided

        # Calculate distance to goal/waypoint
        position = car_state.kinematics_estimated.position
        pos_xy = (float(position.x_val), float(position.y_val))
        distance_to_goal = 999.0
        goal_xy: tuple[float, float] | None = None

        # Priority 1: Use waypoint-based navigation if available
        if self._waypoints is not None and self._current_waypoint_index < len(self._waypoints):
            # Update current waypoint if within threshold (5 meters)
            current_waypoint = self._waypoints[self._current_waypoint_index]
            wp_x, wp_y = float(current_waypoint.x), float(current_waypoint.y)
            dist_to_current = math.sqrt((position.x_val - wp_x) ** 2 + (position.y_val - wp_y) ** 2)

            # Advance to next waypoint if close enough (matching PathManager threshold)
            if dist_to_current <= 5.0 and self._current_waypoint_index < len(self._waypoints) - 1:
                self._current_waypoint_index += 1
                current_waypoint = self._waypoints[self._current_waypoint_index]
                wp_x, wp_y = float(current_waypoint.x), float(current_waypoint.y)
                dist_to_current = math.sqrt(
                    (position.x_val - wp_x) ** 2 + (position.y_val - wp_y) ** 2
                )

            distance_to_goal = dist_to_current
            goal_xy = (wp_x, wp_y)

        # Priority 2: Fall back to single goal_pose (legacy mode)
        elif self._goal_pose is not None or (
            experiment and hasattr(experiment, "goal_pose") and experiment.goal_pose
        ):
            goal_pose = getattr(experiment, "goal_pose", None) or self._goal_pose
            if goal_pose is not None:
                goal_xy = (float(goal_pose.x), float(goal_pose.y))
                distance_to_goal = math.sqrt(
                    (position.x_val - goal_pose.x) ** 2 + (position.y_val - goal_pose.y) ** 2
                )

        velocity = _vector_from_airsim(car_state.kinematics_estimated.linear_velocity)
        speed = float(np.linalg.norm(velocity))
        timestamp = getattr(car_state, "timestamp", None)
        acceleration = 0.0
        if (
            self._previous_velocity is not None
            and timestamp is not None
            and self._previous_timestamp
        ):
            dt = max((timestamp - self._previous_timestamp) * 1e-9, 1e-3)
            acceleration = float(np.linalg.norm((velocity - self._previous_velocity) / dt))
            self._accumulator_time += dt
        else:
            self._accumulator_time = max(self._accumulator_time, self._step * self._control_dt)

        self._previous_velocity = velocity
        self._previous_timestamp = timestamp

        heading = _compute_heading_deg(car_state.kinematics_estimated.orientation)
        progress_possible = not has_collision and speed >= 0.2

        return {
            "telemetry": {
                "distance_to_goal": distance_to_goal,
                "speed_mps": speed,
                "acceleration_mps2": acceleration,
                "heading_deg": heading,
                "collision": has_collision,
                "lane_mask_coverage_ratio": 1.0,
                "progress_possible": progress_possible,
                "sim_time_sec": self._accumulator_time,
                "position_xy": pos_xy,
                "goal_xy": goal_xy,
            },
            "image": self._get_camera_image(),
        }

    def _get_camera_image(self):
        """Get RGB camera image from AirSim."""
        if not self._enable_rgb:
            return None
        try:
            responses = self._client.simGetImages(
                [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)]
            )
            if responses:
                return responses[0]
        except Exception:
            pass
        return None

    def _advance_simulation(self) -> None:
        try:
            if hasattr(self._client, "simContinueForTime"):
                self._client.simContinueForTime(self._control_dt)
                return
        except Exception:
            pass
        time.sleep(self._control_dt)

    @property
    def client(self) -> Any:
        return self._client


def _create_coordinator(
    *,
    load_models: bool = False,
    model_dir: str = "models",
    log_root: Path | str | None = None,
    log_formats: Sequence[str] | None = None,
    # manager exploration controls
    exploration_fraction: float = 0.1,
    exploration_initial_eps: float = 1.0,
    exploration_final_eps: float = 0.05,
    expected_total_timesteps: int | None = None,
) -> CommandCoordinator:
    """Create coordinator with optional pre-trained models."""
    commands = [
        "FOLLOW_LANE",
        "TURN_LEFT_AT_INTERSECTION",
        "TURN_RIGHT_AT_INTERSECTION",
        "STOP",
    ]

    base_log_root = Path(log_root) if log_root is not None else Path("artifacts") / "sb3"
    manager_log_dir = base_log_root / "manager"
    workers_log_root = base_log_root / "workers"

    policy = CommandPolicy(commands=commands)
    manager = DQNManager(
        policy,
        log_formats=log_formats,
        default_log_dir=manager_log_dir,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        total_timesteps=expected_total_timesteps,
    )

    # Load or create DQN manager model
    if load_models:
        # For resume, prefer top-level models directory (backward compatibility)
        model_path = Path(model_dir) / "dqn_manager.zip"
        if model_path.exists():
            manager.load_from_path(str(model_path), log_dir=manager_log_dir)
            print(f"Loaded DQN manager from {model_path}")
        else:
            print(f"No saved DQN manager found at {model_path}, creating new model")
            manager.build_default_model(log_dir=manager_log_dir)
    else:
        # Create new DQN model for training
        manager.build_default_model(log_dir=manager_log_dir)

    # Create workers
    workers = {}
    for command in commands:
        worker_log_dir = workers_log_root / command.lower()
        worker = SACWorker(
            log_formats=log_formats,
            default_log_dir=worker_log_dir,
        )

        if load_models:
            worker_path = Path(model_dir) / f"sac_{command.lower()}.zip"
            if worker_path.exists():
                worker.load_from_path(str(worker_path), log_dir=worker_log_dir)
                print(f"Loaded SAC worker {command} from {worker_path}")
            else:
                print(f"No saved worker found at {worker_path}, creating new model")
                worker.build_default_model(log_dir=worker_log_dir)
        else:
            # Create new SAC model for training
            worker.build_default_model(log_dir=worker_log_dir)

        workers[command] = worker

    return CommandCoordinator(manager, workers)


def _sim_context(mode: str, settings_path: str):
    """Return a no-op context if an external AirSim instance is provided via env vars.

    When AIRSIM_PORT is present, we assume an orchestrator launched the simulator and
    we should not spawn another one. Otherwise, create an AirSim session locally.
    """
    if os.environ.get("AIRSIM_PORT"):
        return nullcontext()
    return airsim_session(mode=mode, settings_path=settings_path)


def train_mode(args) -> int:
    """Training mode - train the HRL agents while showing progress."""
    print("🚗 Starting HRL Training Mode")
    print(f"Config: {args.config}")
    print(f"Training episodes: {args.episodes}")
    print(f"Save interval: {args.save_interval}")

    experiment = load_experiment(args.config)
    artifact_manager = ArtifactManager(args.output)
    run_paths = artifact_manager.start_run(f"train_{experiment.id}")

    # Determine model directory for saving and optionally loading
    model_root = Path(args.models)
    model_root.mkdir(exist_ok=True, parents=True)

    # Resolve resume source if specified
    resume_model_dir = None
    if args.resume_from:
        resume_model_dir = model_root / args.resume_from

        # If --resume-best is specified, load from best-models/ subdirectory
        if args.resume_best:
            resume_model_dir = resume_model_dir / "best-models"

        if not resume_model_dir.exists():
            print(f"❌ Error: Resume directory not found: {resume_model_dir}")
            print(f"\nAvailable runs in {model_root}:")
            available_runs = sorted([d.name for d in model_root.iterdir() if d.is_dir()])
            if available_runs:
                for run in available_runs[-10:]:  # Show last 10
                    print(f"  - {run}")
            else:
                print("  (none)")
            return 1

        model_type = "best" if args.resume_best else "latest"
        print(f"🔄 Resuming from {model_type} models: {resume_model_dir}")
    elif args.resume:
        # Backward compatibility: load from top-level models/
        resume_model_dir = model_root
        print(f"🔄 Resuming from top-level: {resume_model_dir}")

    # Create new model directory scoped to this run for saving
    run_tag = run_paths.run_dir.name  # e.g., 20250925T081041Z_train_<uuid>
    model_dir = model_root / run_tag
    model_dir.mkdir(exist_ok=True, parents=True)

    # Resolve settings path (prefer the orchestrator-provided path if present)
    settings_src = os.environ.get("AIRSIM_SETTINGS_PATH", args.settings)

    # Best-effort parse for ClockSpeed to include in summary
    clock_speed_hint = None
    try:
        with open(settings_src, "r", encoding="utf-8") as _sf:
            _settings_data = json.load(_sf)
            clock_speed_hint = _settings_data.get("ClockSpeed")
    except Exception:
        pass

    # Emit a run-level hyperparameter summary early
    yolo_model = args.detector_model
    yolo_backend = "dummy" if yolo_model.lower() in {"dummy", "none", "off"} else yolo_model
    hp_summary = {
        "experiment": {
            "scene": experiment.scene,
            "vehicle": experiment.vehicle,
            "horizon": experiment.horizon,
        },
        "airsim": {
            "host": os.environ.get("AIRSIM_HOST", "127.0.0.1"),
            "port": int(os.environ.get("AIRSIM_PORT", 41451)),
            "settings_path": settings_src,
            "clock_speed_hint": clock_speed_hint,
        },
        "manager_dqn": {
            "learning_rate": 3e-4,
            "buffer_size": 50000,
            "learning_starts": 128,
            "exploration_fraction": 0.8,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.1,
        },
        "workers_sac": {
            "learning_rate": 3e-4,
            "buffer_size": 100000,
            "learning_starts": 256,
            "batch_size": 64,
            "gamma": 0.99,
            "tau": 0.02,
        },
        "perception": {
            "yolo_model": yolo_backend,
        },
        "milestones": {
            "dqn_learning_starts": 128,
            "sac_learning_starts": 256,
        },
    }
    artifact_manager.write_json("hparams.json", hp_summary)

    # Save a copy of the settings file for reproducibility
    try:
        src_path = Path(settings_src)
        if src_path.exists():
            dst_path = run_paths.run_dir / "settings_used.json"
            shutil.copy2(src_path, dst_path)
    except Exception:
        # Non-fatal: continue even if copy fails
        pass

    with _sim_context(mode=args.mode, settings_path=args.settings):
        client = _get_airsim_client()
        client.enableApiControl(True)
        client.armDisarm(True)
        perception = _build_perception(
            client,
            args.camera_name,
            args.detector_model,
            async_mode=(not getattr(args, "perception_sync", False)),
            enable_segmentation=not args.disable_segmentation,
        )
        # If perception is configured to not use detections (dummy), skip RGB grabs for speed
        detector_obj = getattr(perception, "_detector", None)
        enable_rgb = detector_obj is not None and detector_obj.__class__.__name__ != "DummyDetector"
        simulator = AirSimSimulatorAdapter(
            client, horizon=experiment.horizon, enable_rgb=enable_rgb
        )
        env = AirSimEnv(experiment, simulator=simulator, perception=perception)

        sb3_log_root = run_paths.logs_dir / "sb3"
        expected_total_timesteps = (
            args.episodes * args.max_steps if args.max_steps and args.episodes else None
        )
        # Use resume_model_dir if resuming, otherwise coordinator won't load models
        load_from_dir = str(resume_model_dir) if resume_model_dir else args.models
        coordinator = _create_coordinator(
            load_models=(args.resume or args.resume_from is not None),
            model_dir=load_from_dir,
            log_root=sb3_log_root,
            exploration_fraction=0.8,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.1,
            expected_total_timesteps=expected_total_timesteps,
        )

        # Hook learners to log first learning step
        def _log_learning_start(evt: dict) -> None:
            try:
                artifact_manager.append_jsonl("training_log.jsonl", {"milestone": evt})
            except Exception:
                pass

        if hasattr(coordinator._manager, "_model"):
            try:
                coordinator._manager.on_learning_start = _log_learning_start  # type: ignore[attr-defined]
            except Exception:
                pass
        for _cmd, _worker in coordinator._workers.items():
            try:
                _worker.on_learning_start = _log_learning_start  # type: ignore[attr-defined]
            except Exception:
                pass
        orchestrator = HRLOrchestrator(env, coordinator)

        # Initialize metrics tracker
        metrics_tracker = MetricsTracker(
            run_paths.run_dir, update_interval=args.metrics_update_interval
        )
        if args.metrics_update_interval and args.metrics_update_interval > 0:
            print(
                f"📊 Initialized metrics tracker - graphs will update every {args.metrics_update_interval} steps"
            )
        else:
            print("📊 Metrics tracker initialized - periodic plotting disabled")

        # Training loop
        training_stats = []
        best_reward = float("-inf")
        best_episode = 0
        best_model_dir = model_dir / "best-models"
        for episode in range(args.episodes):
            print(f"\n=== Episode {episode + 1}/{args.episodes} ===")

            # Start tracking this episode with waypoints
            waypoints = _extract_waypoints_from_experiment(experiment)
            metrics_tracker.start_episode(episode + 1, waypoints=waypoints)

            start_time = time.time()
            result = orchestrator.run_episode(
                max_steps=args.max_steps,
                deterministic=False,  # Use exploration during training
                metrics_tracker=metrics_tracker,
            )
            episode_time = time.time() - start_time

            # Finish tracking this episode
            completed_successfully = not result.info.get("collision", False) and result.info.get(
                "goal_reached", False
            )
            metrics_tracker.finish_episode(completed_successfully=completed_successfully)

            stats = {
                "episode": episode + 1,
                "reward": result.cumulative_reward,
                "steps": result.steps,
                "time": episode_time,
                "config_hash": result.info.get("config_hash"),
            }
            training_stats.append(stats)

            print(f"Reward: {result.cumulative_reward:.2f}")
            print(f"Steps: {result.steps}")
            print(f"Time: {episode_time:.2f}s")

            # Save models periodically
            if (episode + 1) % args.save_interval == 0:
                print(f"Saving models at episode {episode + 1}...")

                # Save DQN manager
                if coordinator._manager.model is not None:
                    coordinator._manager.save(str(model_dir / "dqn_manager.zip"))

                # Save SAC workers
                for command, worker in coordinator._workers.items():
                    if worker._model is not None:
                        worker.save(str(model_dir / f"sac_{command.lower()}.zip"))

                print("Models saved!")

            # Save best models when improvement occurs
            if result.cumulative_reward >= best_reward:
                # Improvement or first episode
                best_reward = result.cumulative_reward
                best_episode = episode + 1
                try:
                    best_model_dir.mkdir(parents=True, exist_ok=True)
                    if coordinator._manager.model is not None:
                        coordinator._manager.save(str(best_model_dir / "dqn_manager.zip"))
                    for command, worker in coordinator._workers.items():
                        if worker._model is not None:
                            worker.save(str(best_model_dir / f"sac_{command.lower()}.zip"))
                    print(
                        f"🏅 New best reward {best_reward:.2f} at episode {best_episode} — saved to: {best_model_dir}"
                    )
                except Exception as _exc:
                    print(f"Warning: best-models save failed: {_exc}")

            # Log stats
            artifact_manager.append_jsonl("training_log.jsonl", stats)

    # Save final training statistics
    artifact_manager.write_json(
        "training_stats.json",
        {
            "episodes": args.episodes,
            "total_stats": training_stats,
            "final_reward": training_stats[-1]["reward"] if training_stats else 0,
            "avg_reward": sum(s["reward"] for s in training_stats) / len(training_stats)
            if training_stats
            else 0,
        },
    )

    # Always save final models (ensures latest policy is persisted regardless of interval)
    try:
        if coordinator._manager.model is not None:
            coordinator._manager.save(str(model_dir / "dqn_manager.zip"))
        for command, worker in coordinator._workers.items():
            if worker._model is not None:
                worker.save(str(model_dir / f"sac_{command.lower()}.zip"))
        print(f"Final models saved to: {model_dir}")
    except Exception as _exc:
        # Do not fail the run if save throws; logs already captured
        print(f"Warning: final model save failed: {_exc}")

    # Print final metrics summary
    summary_stats = metrics_tracker.get_summary_stats()
    if summary_stats:
        print("\n📊 Training Summary:")
        print(f"   Total Episodes: {summary_stats.get('total_episodes', 0)}")
        print(f"   Average Reward: {summary_stats.get('avg_reward', 0.0):.2f}")
        print(f"   Best Reward: {summary_stats.get('best_reward', 0.0):.2f}")
        print(
            f"   Average Episode Length: {summary_stats.get('avg_episode_length', 0.0):.1f} steps"
        )
        print(f"   Success Rate: {summary_stats.get('success_rate', 0.0) * 100:.1f}%")
        print(f"   Collision Rate: {summary_stats.get('collision_rate', 0.0) * 100:.1f}%")

    print("\n✅ Training completed!")
    print(f"Models saved to: {model_dir}")
    if best_episode > 0:
        print(
            f"Best models (episode {best_episode}, reward {best_reward:.2f}) saved to: {best_model_dir}"
        )
    print(f"Logs saved to: {run_paths.run_dir}")
    print(f"📊 Metrics and graphs saved to: {run_paths.run_dir}/metrics/")
    print("   - reward.png: Real-time reward trends")
    print("   - episode_summary.png: Episode-level performance")
    print("   - action_analysis.png: Action patterns and distributions")
    print("   - performance_metrics.png: Speed, distance, and reward breakdown")

    return 0


def _get_airsim_client() -> airsim.CarClient:
    """Get an AirSim client, connecting to a host and port from environment variables if available."""
    import os

    host = os.environ.get("AIRSIM_HOST", "127.0.0.1")
    port = int(os.environ.get("AIRSIM_PORT", 41451))

    client = airsim.CarClient(ip=host, port=port)
    client.confirmConnection()
    return client


def inference_mode(args) -> int:
    """Inference mode - run trained model for evaluation."""
    print("🤖 Starting HRL Inference Mode")
    print(f"Config: {args.config}")
    print(f"Models: {args.models}")

    experiment = load_experiment(args.config)
    artifact_manager = ArtifactManager(args.output)
    run_paths = artifact_manager.start_run(f"eval_{experiment.id}")
    completed_episodes = 0

    with _sim_context(mode=args.mode, settings_path=args.settings):
        client = _get_airsim_client()
        client.enableApiControl(True)
        client.armDisarm(True)
        perception = _build_perception(
            client,
            args.camera_name,
            args.detector_model,
            async_mode=(not getattr(args, "perception_sync", False)),
            enable_segmentation=not args.disable_segmentation,
        )
        detector_obj = getattr(perception, "_detector", None)
        enable_rgb = detector_obj is not None and detector_obj.__class__.__name__ != "DummyDetector"
        simulator = AirSimSimulatorAdapter(
            client, horizon=experiment.horizon, enable_rgb=enable_rgb
        )
        env = AirSimEnv(experiment, simulator=simulator, perception=perception)

        # Load trained models
        sb3_log_root = run_paths.logs_dir / "sb3"
        coordinator = _create_coordinator(
            load_models=True,
            model_dir=args.models,
            log_root=sb3_log_root,
        )
        orchestrator = HRLOrchestrator(env, coordinator)

        print("🎮 Running trained model...")
        print("Watch AirSim window to see the car driving!")
        print("Press Ctrl+C to stop")

        episode = 1
        try:
            while True:
                print(f"\n--- Evaluation Run {episode} ---")

                result = orchestrator.run_episode(
                    max_steps=args.max_steps,
                    deterministic=True,  # Use deterministic policy for evaluation
                )

                print(f"Reward: {result.cumulative_reward:.2f}")
                print(f"Steps: {result.steps}")

                _log_evaluation_episode(
                    artifact_manager,
                    run_paths.logs_dir,
                    episode,
                    result,
                    env.last_observation,
                )

                if not args.continuous:
                    break

                episode += 1
                time.sleep(2)  # Brief pause between episodes

        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        finally:
            completed_episodes = episode if args.continuous else min(episode, 1)

    artifact_manager.write_json(
        "evaluation_summary.json",
        {
            "episodes": completed_episodes,
            "config_hash": experiment.config_hash,
            "model_dir": args.models,
        },
    )
    print("✅ Inference completed!")
    return 0


def _build_perception(
    client,
    camera_name: str,
    detector_model: str,
    *,
    async_mode: bool = False,
    enable_segmentation: bool = True,
):
    segmentation = SegmentationAdapter(client, camera_name=camera_name)
    use_dummy = False
    try:
        detector = YoloDetector(model_name=detector_model)
    except ModelLoadError as exc:
        print(f"Warning: Could not load YOLO detector: {exc}")

        # Create dummy detector for training
        class DummyDetector:
            def run_detection(self, image):
                return []

        detector = DummyDetector()
        use_dummy = True

    # If detector is dummy/none/off, don't waste time on segmentation by default
    seg_enabled = enable_segmentation and not use_dummy
    if async_mode and not use_dummy:
        return AsyncPerceptionPipeline(
            segmentation=segmentation,
            detector=cast(YoloDetector, detector),
            enable_segmentation=seg_enabled,
        )
    return PerceptionPipeline(
        segmentation=segmentation,
        detector=cast(YoloDetector, detector),
        enable_segmentation=seg_enabled,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HRL AirSim Training and Inference")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Training mode
    train_parser = subparsers.add_parser("train", help="Train HRL agents")
    train_parser.add_argument("--config", required=True, help="Experiment config YAML")
    train_parser.add_argument("--episodes", type=int, default=100, help="Training episodes")
    train_parser.add_argument(
        "--save-interval", type=int, default=10, help="Save models every N episodes"
    )
    train_parser.add_argument("--models", default="models", help="Model directory")
    train_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from saved models (top-level for backward compatibility)",
    )
    train_parser.add_argument(
        "--resume-from",
        type=str,
        help="Resume from a specific run ID (e.g., 20250930T154745Z_train_8591ac4c...)",
    )
    train_parser.add_argument(
        "--resume-best",
        action="store_true",
        help="Resume from best-models/ subdirectory (use with --resume-from)",
    )
    train_parser.add_argument("--settings", default="settings.json", help="AirSim settings")
    train_parser.add_argument("--mode", choices=["gui", "headless"], default="gui")
    train_parser.add_argument("--max-steps", type=int, default=1000)
    train_parser.add_argument("--output", default="artifacts", help="Artifact directory")
    train_parser.add_argument("--detector-model", default="yolov8n")
    train_parser.add_argument("--camera-name", default="0")
    train_parser.add_argument(
        "--disable-segmentation",
        action="store_true",
        help="Skip segmentation capture to reduce RPC overhead",
    )
    train_parser.add_argument(
        "--perception-sync",
        action="store_true",
        help="Force synchronous perception (default is async)",
    )
    train_parser.add_argument(
        "--metrics-update-interval",
        type=int,
        default=5,
        help="Update metrics plots every N steps (0 or negative to disable)",
    )

    # Inference mode
    eval_parser = subparsers.add_parser("eval", help="Evaluate trained models")
    eval_parser.add_argument("--config", required=True, help="Experiment config YAML")
    eval_parser.add_argument("--models", required=True, help="Trained models directory")
    eval_parser.add_argument("--continuous", action="store_true", help="Run continuously")
    eval_parser.add_argument("--settings", default="settings.json", help="AirSim settings")
    eval_parser.add_argument("--mode", choices=["gui", "headless"], default="gui")
    eval_parser.add_argument("--max-steps", type=int, default=1000)
    eval_parser.add_argument("--detector-model", default="yolov8n")
    eval_parser.add_argument("--camera-name", default="0")
    eval_parser.add_argument("--output", default="artifacts", help="Artifact directory")
    eval_parser.add_argument(
        "--metrics-update-interval",
        type=int,
        default=5,
        help="Update metrics plots every N steps (0 or negative to disable)",
    )
    eval_parser.add_argument(
        "--disable-segmentation",
        action="store_true",
        help="Skip segmentation capture to reduce RPC overhead",
    )
    eval_parser.add_argument(
        "--perception-sync",
        action="store_true",
        help="Force synchronous perception (default is async)",
    )

    return parser.parse_args()


def _vector_from_airsim(vector) -> np.ndarray:
    return np.array([vector.x_val, vector.y_val, vector.z_val], dtype=np.float32)


def _compute_heading_deg(orientation) -> float:
    try:
        yaw = airsim.to_eularian_angles(orientation)[2]
    except Exception:
        yaw = 0.0
    return math.degrees(yaw)


def _extract_waypoints_from_experiment(experiment) -> list[tuple[float, float]]:
    """Extract waypoints from experiment config for trajectory plotting."""
    waypoints = []
    try:
        if hasattr(experiment, "waypoints") and experiment.waypoints:
            # Convert waypoint poses to (x, y) tuples
            waypoints = [(float(wp.x), float(wp.y)) for wp in experiment.waypoints]
        elif hasattr(experiment, "goal_pose") and experiment.goal_pose:
            # Legacy: create path from start to goal
            start = experiment.start_pose
            goal = experiment.goal_pose
            waypoints = [(float(start.x), float(start.y)), (float(goal.x), float(goal.y))]
    except Exception:
        # If waypoint extraction fails, return empty list
        pass
    return waypoints


def _log_evaluation_episode(
    artifact_manager: ArtifactManager,
    logs_dir: Path,
    episode: int,
    result,
    observation,
) -> None:
    payload: dict[str, Any] = {
        "episode": episode,
        "reward": result.cumulative_reward,
        "steps": result.steps,
        "info": result.info,
    }
    if observation is not None:
        payload["reward_components"] = dict(observation.reward_components)
        payload["telemetry"] = dict(observation.telemetry)
        payload["done_flags"] = dict(observation.done_flags)
        payload["command"] = observation.command
        payload["detections"] = observation.detections
        if observation.segmentation_mask is not None:
            mask_path = logs_dir / f"episode_{episode:03d}_segmentation.npy"
            np.save(mask_path, observation.segmentation_mask)
            payload["segmentation_mask_path"] = str(mask_path)
    artifact_manager.append_jsonl("evaluation_log.jsonl", payload)


def main() -> int:
    args = parse_args()

    if args.command == "train":
        return train_mode(args)
    elif args.command == "eval":
        return inference_mode(args)
    else:
        print("Please specify 'train' or 'eval' command")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
