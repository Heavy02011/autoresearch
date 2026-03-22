# autoresearch Module Reference

## Modules Quick Reference

### autoresearch.config
**Type validation and schema models**
- `PromotionState` (Enum) — Discrete states: TRAINING, EVALUATING, PROMOTION_GATE, PROMOTED, ROLLED_BACK
- `TrainingConfig` — Learning rate, batch size, time budget
- `EvaluationConfig` — Simulator version, map, num laps
- `PromotionConfig` — Acceptance criteria thresholds
- `EnvironmentConfig` — Paths, logging level, W&B settings
- `ExperimentConfig` — Master configuration combining all above
- `StateTransition` — Record of state change with metadata
- `RunManifest` — Complete run artifact manifest with history

### autoresearch.state
**State machine for promotion decisions**
- `StateTracker` 
  - `current_state()` → PromotionState
  - `state_history()` → List[StateTransition]
  - `transition(new_state, iteration, metrics, verdict)` — Atomic state change
  - `rollback(target_iteration)` — Revert to previous promoted state
  - `update_latest_metrics(metrics)` — Update metrics without state change

- `StateView` — Read-only queries on manifest
  - `get_current_state()` → PromotionState
  - `get_state_at_iteration(iteration)` → Optional[PromotionState]
  - `is_promoted()` → bool
  - `last_promotion_iteration()` → Optional[int]
  - `metrics_at_iteration(iteration)` → Optional[Dict]

### autoresearch.artifacts
**Filesystem artifact storage and management**
- `ArtifactManager`
  - Properties: `state_file`, `config_snapshot_file`, `metadata_file`, `environment_lock_file`
  - Methods:
    - `checkpoint_path(iteration)` → Path
    - `best_model_path()` → Path
    - `best_model_onnx_path()` → Path
    - `metrics_file(iteration)` → Path
    - `save_config_snapshot(config)` — Snapshot experiment config
    - `save_metrics(iteration, metrics)` — Save evaluation results
    - `load_metrics(iteration)` → Dict
    - `save_metadata(git_commit)` — Git + version info
    - `save_environment_lock(env_vars)` — Environment snapshot
    - `list_checkpoints()` → List[(iteration, Path)]
    - `get_latest_checkpoint()` → Optional[Path]

### autoresearch.evaluate
**Simulator evaluation interface**
- `SimulatorEvaluator`
  - `evaluate_model(model_path, iteration, dry_run)` → Dict[metrics]
    - Returns: `{lap_time, cte_mean, cte_max, success, timestamp}`
  - `check_operability(dry_run)` → bool — Health check
  - Private methods:
    - `_run_simulator_loop(model_path, iteration)` — **PLACEHOLDER: implement gym-donkeycar**
    - `_check_simulator_ready()` → bool
    - `_find_simulator_exe()` → Path
    - `_dummy_metrics(iteration)` — For dry-run testing

### autoresearch.promote
**Model promotion gating logic**
- `PromotionGate`
  - `evaluate(metrics, operability_ok, baseline_metric)` → (bool, str)
    - Returns: `(should_promote, reason)`
  - Validates:
    1. Operability check passed (if required)
    2. Metric below threshold
    3. Improvement over baseline ≥ minimum

### autoresearch.export
**ONNX model export for deployment**
- `ONNXExporter`
  - `export_model(model, checkpoint_path, output_path, input_shape, metadata)` → bool
  - `validate_export(onnx_path)` → bool

### autoresearch.orchestrate
**Main autonomous loop orchestration**
- `Orchestrator`
  - Constructor: `__init__(config: ExperimentConfig, run_id: str)`
  - Main method: `run_autonomous_loop()` — Execute full pipeline
    - For each iteration:
      1. `_train_iteration(iteration)` — **PLACEHOLDER: integrate training code**
      2. Transition to EVALUATING
      3. `evaluate_model(model_path, iteration)` — Get metrics
      4. `save_metrics(iteration, metrics)` — Persist results
      5. Transition to PROMOTION_GATE
      6. `evaluate(metrics, ...)` — Check gating criteria
      7. If promoted: save as best model, transition to PROMOTED
      8. Else: discard, transition back to TRAINING
  - `status()` → Dict — Get current run status

### autoresearch.cli
**Command-line interface using Typer**
- Commands:
  - `run` — Start autonomous loop
    - Options: `--max-iterations`, `--config`, `--dry-run`, `--run-id`
  - `status` — Check run state
    - Argument: `<run_id>`
    - Option: `--runs-dir`
  - `rollback` — Revert to previous promoted state
    - Argument: `<run_id>`
    - Option: `--runs-dir`
  - `export` — Export model (ONNX/PyTorch)
    - Arguments: `<run_id>`
    - Options: `--runs-dir`, `--format`
  - `prepare` — Data/simulator setup (stub)

### autoresearch.logging_config
**Structured logging setup**
- `configure_logging(log_dir, run_id, level)` — Initialize structlog
- `get_logger(name)` → Logger — Get logger instance

## Legal State Transitions

```
TRAINING → EVALUATING
EVALUATING → PROMOTION_GATE
PROMOTION_GATE → PROMOTED
PROMOTION_GATE → TRAINING (if not promoted, loop back)
PROMOTED → ROLLED_BACK
PROMOTED → TRAINING (for next iteration)
ROLLED_BACK → TRAINING
```

## Configuration Hierarchy

```
ExperimentConfig
├── TrainingConfig
├── EvaluationConfig
├── PromotionConfig
└── EnvironmentConfig
```

## File Structure at Runtime

```
runs/
└── run_20260322_123456/
    ├── state.json                    # StateTracker manifest
    ├── config.snapshot.json          # ExperimentConfig snapshot
    ├── metadata.json                 # Git/version info
    ├── environment.lock              # Environment variables
    ├── models/
    │   ├── checkpoint_iter_1.pt      # Training checkpoint
    │   ├── checkpoint_iter_2.pt
    │   └── best_model.pt             # Best promoted model
    ├── metrics/
    │   ├── eval_iter_1.json          # Evaluation results
    │   └── eval_iter_2.json
    └── logs/
        └── run_20260322_123456.log   # Structured log
```

## Error Handling

All modules use structured logging:
```python
from autoresearch.logging_config import get_logger
logger = get_logger("module_name")
logger.error("description", key=value, error=str(e))
```

## Testing

Run integration tests:
```bash
python tests/test_integration.py
```

Tests validate:
- Configuration model creation
- State machine legal transitions
- Artifact storage (save/load)
- Promotion gating logic

---

**Generated:** 2026-03-22  
**Version:** 0.1.0
