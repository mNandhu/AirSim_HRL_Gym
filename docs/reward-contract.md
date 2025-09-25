# Reward Contract

| Component          | Weight/Formula                   | Notes                                             |
| ------------------ | -------------------------------- | ------------------------------------------------- |
| progress           | prev_distance - current_distance | Positive when moving toward goal                  |
| lane_adherence     | lane_mask_coverage_ratio         | Normalized coverage within drivable area          |
| collision          | -1.0                             | Penalty applied when AirSim reports a collision   |
| command_completion | +2.0                             | Bonus when high-level command objective completes |
| idle_penalty       | -0.5 (speed < 0.2)               | Discourages idling when progress possible         |
