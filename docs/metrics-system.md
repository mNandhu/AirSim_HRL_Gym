# Dynamic Training Metrics System 📊

The HRL training system now includes comprehensive real-time metrics tracking and visualization that automatically generates and updates graphs during training.

## Features

### 🎯 Real-Time Tracking

-   **Step-by-step metrics** collected during each episode
-   **Dynamic graph updates** every 5 steps (configurable)
-   **In-place graph updates** - same files get refreshed with new data
-   **JSON data export** for further analysis

### 📈 Generated Visualizations

The system automatically generates 4 key visualization files in `artifacts/{timestamp}_{id}/metrics/`:

1. **`reward.png`** - Real-time reward trends

    - Step-by-step reward over time
    - Cumulative reward progress
    - Moving average trend line

2. **`episode_summary.png`** - Episode-level performance

    - Total reward per episode
    - Episode length (steps)
    - Max speed achieved per episode
    - Closest distance to goal

3. **`action_analysis.png`** - Action patterns and distributions

    - Throttle, brake, steering over time
    - Action value distributions
    - Control smoothness analysis

4. **`performance_metrics.png`** - Detailed performance breakdown
    - Vehicle speed over time
    - Distance to goal progression
    - Reward component breakdown (stacked)

### 📊 Data Export

The system also saves structured data in JSON format:

-   **`episodes.json`** - Episode-level summary statistics
-   **`episode_N_steps.json`** - Step-by-step data for each episode
-   Includes all telemetry, actions, rewards, and commands

## Usage

### Basic Training with Metrics

The metrics system is **automatically enabled** for all training runs:

```bash
uv run python src/scripts/train_and_eval.py train \
    --config configs/experiments/training.yaml \
    --episodes 100 \
    --save-interval 10 \
    --models models \
    --settings settings.json \
    --mode gui \
    --detector-model yolo12n
```

### Metrics Configuration

You can customize the update frequency by modifying the `update_interval` parameter in the training script:

```python
# Update graphs every 5 steps (default)
metrics_tracker = MetricsTracker(run_paths.run_dir, update_interval=5)

# Update graphs every 10 steps (less frequent, better performance)
metrics_tracker = MetricsTracker(run_paths.run_dir, update_interval=10)

# Update graphs every step (more frequent, may impact performance)
metrics_tracker = MetricsTracker(run_paths.run_dir, update_interval=1)
```

## Key Metrics Tracked

### Episode-Level Metrics

-   **Total reward** and **episode length**
-   **Max speed** achieved during episode
-   **Minimum distance to goal** (closest approach)
-   **Collision rate** and **success rate**
-   **Action statistics** (avg throttle/brake/steering)
-   **Command distribution** (which commands were used)

### Step-Level Metrics

-   **Instantaneous reward** and **cumulative reward**
-   **Vehicle speed** and **distance to goal**
-   **Action values** (throttle, brake, steering)
-   **Active command** from hierarchical controller
-   **Reward component breakdown**:
    -   Command shaping reward
    -   Collision penalties
    -   Completion bonuses
    -   Idle penalties

## Training Summary

At the end of training, you'll see a comprehensive summary:

```
📊 Training Summary:
   Total Episodes: 100
   Average Reward: 15.32
   Best Reward: 28.45
   Average Episode Length: 145.2 steps
   Success Rate: 23.5%
   Collision Rate: 12.1%
```

## Performance Impact

-   **Minimal overhead** during training (< 1% performance impact)
-   **Configurable update frequency** to balance detail vs performance
-   **Efficient graph generation** using matplotlib with optimized settings
-   **Required dependency** - matplotlib is automatically installed with the project

## File Locations

All metrics are saved under: `artifacts/{timestamp}_{experiment_id}/metrics/`

Example structure:

```
artifacts/20250925T140019Z_train_446112d4.../
├── metrics/
│   ├── reward.png                 # Real-time reward trends
│   ├── episode_summary.png        # Episode performance
│   ├── action_analysis.png        # Action patterns
│   ├── performance_metrics.png    # Detailed metrics
│   ├── episodes.json              # Episode summaries
│   ├── episode_1_steps.json       # Step data for episode 1
│   └── episode_2_steps.json       # Step data for episode 2
├── logs/                          # SB3 training logs
└── training_stats.json            # Overall training statistics
```

## Troubleshooting

### No Graphs Generated

-   Check for error messages in training output
-   Verify write permissions to artifacts directory
-   Ensure the training runs for at least one complete step

### Performance Issues

-   Increase `update_interval` to reduce graph update frequency
-   Monitor disk space (graphs can be ~100KB each)
-   Consider disabling metrics for very long training runs

### Missing Data

-   Ensure training runs for at least one complete episode
-   Check that the environment is properly configured
-   Verify telemetry data is being collected from AirSim

The system provides rich insights into training progress and agent behavior, making it easier to debug issues, tune hyperparameters, and understand learning dynamics!
