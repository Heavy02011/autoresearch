# Simulator Evaluation Integration

Complete gym-donkeycar integration for automated model evaluation in DonkeyCar simulation environment.

## Overview

The `SimulatorEvaluator` class (`src/autoresearch/evaluate.py`) provides full integration with the gym-donkeycar simulator. It:

1. **Loads trained PyTorch models** from checkpoint files
2. **Creates deterministic simulator environments** with configurable seeds
3. **Drives evaluation laps** collecting comprehensive metrics
4. **Tracks cross-track error (CTE)** and lap completion rates
5. **Returns standardized metrics** for promotion gate decisions

## Architecture

### Core Workflow

```
evaluate_model(model_path, iteration, dry_run=False)
    ↓
load_model (PyTorch checkpoint)
    ↓
create_env (DonkeySimEnv with deterministic seed)
    ↓
for each lap:
    reset → drive_laps → record_metrics → close
    ↓
aggregate → return_metrics
```

### Metrics Contract

Every evaluation returns a standardized metrics dictionary:

```python
{
    "lap_time": float,          # Mean lap time in seconds
    "cte_mean": float,          # Mean cross-track error across all laps
    "cte_max": float,           # Max cross-track error observed
    "success": bool,            # Whether all laps completed successfully
    "timestamp": float,         # Unix timestamp of evaluation
}
```

## Configuration

### via Python Config Object

```python
from autoresearch.config import EvaluationConfig

eval_config = EvaluationConfig(
    simulator_version="4.2.0",
    map_name="donkey_sim_path",
    num_laps=3,
    seed=42,                    # For reproducibility
    timeout_seconds=300,
    port=9091,                  # Simulator port
    metric_name="lap_time"
)
```

### via YAML File (Hydra)

**configs/evaluation.yaml:**
```yaml
simulator_version: "4.2.0"
map_name: "donkey_sim_path"
num_laps: 3
seed: 42
timeout_seconds: 300
metric_name: "lap_time"
port: 9091
```

## Requirements

### Environment Setup

1. **Install gym-donkeycar:**
   ```bash
   pip install gym-donkeycar==22.11.6
   ```

2. **Donkey Simulator:**
   - Download from: https://github.com/autorope/donkey_car
   - Set `DONKEY_SIM_PATH` environment variable, OR
   - Place executable in standard locations:
     - `~/.donkey/donkey_sim` (Linux)
     - `~/.donkey/donkey_sim.exe` (Windows)
     - `/opt/donkeycar/donkey_sim` (Docker)

3. **PyTorch Model:**
   - Must be saved as `.pt` checkpoint via `torch.save(model, path)`
   - Model interface: `output = model(image_tensor)` → steering action
   - Expected input: image tensor (H, W, C) or (B, C, H, W)
   - Expected output: steering angle (float or tensor)

## Usage

### Basic Evaluation

```python
from autoresearch.evaluate import SimulatorEvaluator
from autoresearch.config import EvaluationConfig
from pathlib import Path

# Create evaluator
config = EvaluationConfig(num_laps=3, seed=42)
evaluator = SimulatorEvaluator(config)

# Evaluate model
model_path = Path("runs/run_001/models/checkpoint_iter_5.pt")
metrics = evaluator.evaluate_model(model_path, iteration=5)

print(metrics)
# {
#   "lap_time": 24.3,
#   "cte_mean": 0.42,
#   "cte_max": 1.1,
#   "success": True,
#   "timestamp": 1711190450.123
# }
```

### Dry-Run Mode (Testing)

```python
# Returns dummy metrics without running simulator
metrics = evaluator.evaluate_model(model_path, iteration=5, dry_run=True)

# Useful for:
# - Testing orchestration logic without simulator
# - CI/CD validation (no GPU/simulator required)
# - Rapid prototyping
```

### Operability Check

```python
# Verify simulator is available before loop starts
is_ready = evaluator.check_operability()
if not is_ready:
    print("Simulator not available!")
```

## Implementation Details

### Model Loading

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device)
model.eval()  # Set to evaluation mode
```

- Device selection: GPU if available, else CPU
- Memory-mapped loading with device placement
- Model set to `.eval()` mode (disables dropout, batch norm updates)

### Observation Processing

```python
obs_tensor = torch.from_numpy(obs).float().to(device)

# Automatic normalization if pixel values > 1
if obs_tensor.max() > 1.0:
    obs_tensor = obs_tensor / 255.0

# Add batch dimension if needed
if obs_tensor.dim() == 3:  # (H, W, C)
    obs_tensor = obs_tensor.unsqueeze(0)  # → (1, H, W, C)
```

Handles variable observation formats transparently.

### Action Generation

```python
with torch.no_grad():  # No gradient computation (inference only)
    action = model(obs_tensor)
    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()
        if action.ndim > 1:
            action = action[0]  # Remove batch dimension
```

- No gradients computed (faster inference)
- Returns numpy array for gym compatibility
- Handles both single and batch outputs

### Metric Aggregation

**Per-Lap Tracking:**
```python
lap_times = []        # One entry per lap
cte_errors = []       # All CTE values across all steps
all_success = True    # Any failed lap → False
```

**Final Aggregation:**
```python
{
    "lap_time": np.mean(lap_times),     # Average across laps
    "cte_mean": np.mean(cte_errors),    # Average CTE all steps
    "cte_max": np.max(cte_errors),      # Worst CTE observed
    "success": all_success,              # All laps completed
    "timestamp": time.time()             # When evaluation finished
}
```

## Logging

All operations logged via structlog JSON format:

```json
{
  "event": "Model loaded",
  "iteration": 5,
  "device": "cuda",
  "model_path": "/path/to/checkpoint.pt"
}

{
  "event": "Starting lap",
  "iteration": 5,
  "lap": 1
}

{
  "event": "Lap complete",
  "iteration": 5,
  "lap": 1,
  "lap_time": 24.3,
  "cte_mean": 0.42
}

{
  "event": "Evaluation complete",
  "iteration": 5,
  "aggregated_metrics": {...},
  "elapsed_seconds": 75.2
}
```

### Error Handling

If evaluation fails (simulator crash, model error, etc.):

```json
{
  "event": "Evaluation failed",
  "iteration": 5,
  "error": "EnvironmentError: Simulator not responding",
  "error_type": "EnvironmentError"
}
```

Returns failure metrics:
```python
{
    "lap_time": 300.0,        # timeout_seconds
    "cte_mean": inf,
    "cte_max": inf,
    "success": False,
    "timestamp": time.time()
}
```

These metrics will be rejected by the promotion gate (due to `success=False` and high error values).

## Determinism & Reproducibility

### Seed Control

```python
# Deterministic evaluation with seed=42
evaluator = SimulatorEvaluator(EvaluationConfig(seed=42))
metrics_run1 = evaluator.evaluate_model(model_path, iteration=5)
metrics_run2 = evaluator.evaluate_model(model_path, iteration=5)
# metrics_run1 == metrics_run2 (same seed, same metrics)
```

### Deterministic Simulator Configuration

```python
env = DonkeySimEnv(
    seed=config.seed,  # Sets numpy/gym randomness
    conf={
        "map_name": config.map_name,
        "frame_skip": 1,  # No frame skipping
        "body_style": "donkey",
        "steer_limit_left": 1.0,
        "steer_limit_right": 1.0,
    }
)
```

- Fixed map
- Fixed body style
- Fixed steering limits
- Deterministic physics step

## Performance Considerations

### Runtime

Typical evaluation run with 3 laps:
- Model loading: ~2 seconds
- Per lap: ~15-30 seconds (depends on lap complexity)
- Total: ~50-100 seconds per evaluation

### Memory

- PyTorch model: Size varies (typically 10-100 MB)
- Simulator environment: ~500 MB (processes)
- Typical total: 1-2 GB RAM

### GPU Usage

- Automatic GPU detection and usage
- Falls back to CPU if unavailable
- Can be forced via environment: `CUDA_VISIBLE_DEVICES=""`

## Troubleshooting

### "Simulator executable not found"

**Solution:** Set environment variable:
```bash
export DONKEY_SIM_PATH="/path/to/donkey_sim"
```

Or place simulator in standard location:
```bash
~/.donkey/donkey_sim          # Linux
~/.donkey/donkey_sim.exe      # Windows
```

### "Port already in use"

**Solution:** Change port in config:
```yaml
port: 9092  # Use different port
```

Or kill existing process:
```bash
lsof -ti:9091 | xargs kill -9
```

### Model output shape mismatch

**Solution:** Ensure model output is compatible:
```python
# ✅ Correct: Returns steering action
output = model(obs)  # Shape: (steering_value,) or scalar

# ❌ Wrong: Returns feature maps
output = model(obs)  # Shape: (128, 4, 4)
```

### Low frame rate / Simulator lag

**Solution:** Reduce image resolution or frame skip:
```yaml
map_name: "donkey_sim_path_small"  # Smaller track
timeout_seconds: 600  # More time per lap
```

## Advanced: Custom Metric Collection

Extend `SimulatorEvaluator` to collect additional metrics:

```python
class AdvancedEvaluator(SimulatorEvaluator):
    def _run_simulator_loop(self, model_path, iteration):
        # ... existing code ...
        
        # Collect additional metrics in lap loop
        speed_samples = []
        for step in range(max_steps):
            # ... existing code ...
            
            # Add custom metric collection
            if "speed" in info:
                speed_samples.append(info["speed"])
        
        # Add to metrics dict
        metrics["max_speed"] = max(speed_samples) if speed_samples else 0.0
        metrics["avg_speed"] = np.mean(speed_samples) if speed_samples else 0.0
        
        return metrics
```

## Future Enhancements

- [ ] Real-time metrics streaming to W&B
- [ ] Multi-simulator parallelization (multiple ports)
- [ ] Video recording of evaluation runs
- [ ] Profiling/performance analytics per lap segment
- [ ] Adversarial scenario testing (wind, obstacles)
- [ ] Hardware-in-the-loop (physical DonkeyCAR) evaluation

