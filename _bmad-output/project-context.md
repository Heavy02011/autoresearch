---
workflowType: project-context
project_name: autoresearch
user_name: heavy
date: 2026-03-22
communication_language: English
document_output_language: English
sections_completed: [1, 2, 3]
---

# Project Context: autoresearch

_Critical AI agent implementation rules for the autoresearch DonkeyCar autonomous optimization framework._

---

## 1. Technology Stack & Versions

### Core Environment
- **Python:** 3.10+
- **Package Manager:** Poetry 1.7.1
- **Build System:** setuptools + wheel

### ML & Training
- **ML Framework:** PyTorch 2.9.1 (CUDA 12.8 available)
- **Simulator:** gym-donkeycar 22.11.6
- **Model Export:** ONNX 1.17.0, onnxruntime 1.17.0

### CLI & Configuration
- **CLI Framework:** Typer 0.9.0+ (ALL custom commands use Typer)
- **Configuration:** Hydra 1.3.0, OmegaConf 2.3.0
- **Type Validation:** Pydantic 2.0 (MANDATORY for all config classes)
- **Logging:** structlog 23.3.0 (JSON output format required)

### Experiment Tracking (Optional)
- **W&B Integration:** wandb 0.16.0 (optional, conditional on config flag)

### Testing
- **Framework:** pytest (via pyproject.toml)
- **Testing Pattern:** Integration tests validate core logic, Pydantic models, state transitions

---

## 2. Architecture & Design Patterns

### 2.1 State Machine (Promotion Workflow)

**Pattern:** Discrete state transitions with immutable audit trail.

**Implementation:** `PromotionState` enum in `config.py`:
```python
class PromotionState(str, Enum):
    TRAINING = "training"
    EVALUATING = "evaluating"
    PROMOTION_GATE = "promotion_gate"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"
```

**Rule:** State `StateTracker` class enforces legal transitions via `_valid_moves()` dict:
- TRAINING → EVALUATING only
- EVALUATING → PROMOTION_GATE only
- PROMOTION_GATE → PROMOTED OR TRAINING (based on gate verdict)
- PROMOTED → TRAINING OR ROLLED_BACK only

**Agent Rule:** When implementing any state-based logic, always validate transitions against the `_valid_moves()` dictionary to prevent illegal state changes.

### 2.2 Repository Pattern (Artifact Management)

**Pattern:** `ArtifactManager` class abstracts all filesystem operations.

**Storage Contract:** All artifacts live in deterministic hierarchy:
```
runs/
  run_YYYYMMDD_HHMMSS/
    state.json               # RunManifest (state history)
    config.snapshot.json     # Config snapshot at run time
    metadata.json            # Git commit, branch, timestamp
    environment.lock         # Python versions
    models/
      checkpoint_iter_1.pt   # Intermediate checkpoints
      checkpoint_iter_2.pt
      best_model.pt          # Best promoted model
      best_model.onnx        # ONNX export
    metrics/
      metrics_iter_1.json    # {lap_time, cte_mean, cte_max, success}
      metrics_iter_2.json
    logs/
      run.log                # Structured JSON logs
```

**Agent Rule:** All file writes must go through `ArtifactManager` methods:
- `checkpoint_path(iteration)` → Path to save checkpoint
- `best_model_path()` → Path to best model
- `save_metrics(iteration, dict)` → Save metrics JSON
- `load_metrics(iteration)` → Load metrics dict
- Never direct filesystem writes outside this interface

### 2.3 Facade Pattern (Simulator Evaluation)

**Pattern:** `SimulatorEvaluator` wraps gym-donkeycar environment.

**Contract:** `evaluate_model(model_path, iteration, dry_run)` returns:
```python
{
    "lap_time": float,
    "cte_mean": float,         # Cross-track error mean
    "cte_max": float,          # Cross-track error max
    "success": bool,           # Whether laps completed
    "timestamp": str           # ISO timestamp
}
```

**Agent Rule:** Evaluation metrics must always include these exact keys. Add custom metrics only as additional keys, never rename core keys.

### 2.4 Command Pattern (CLI)

**Pattern:** Typer commands encapsulate orchestration.

**Available Commands:**
- `run` — Execute train→eval→gate loop
- `status` — Query run state
- `rollback` — Revert to previous promoted state
- `export` — Export model to ONNX
- `prepare` — Setup/data preparation (stub)

**Agent Rule:** ALL CLI commands must use Typer decorators. Command options require `help=` text. Use `typer.Option()` for all arguments.

---

## 3. Code Organization & Conventions

### 3.1 File & Module Naming

- **Files:** `snake_case.py` (e.g., `state.py`, `artifacts.py`, `cli.py`)
- **Classes:** `PascalCase` (e.g., `StateTracker`, `ArtifactManager`, `PromotionGate`)
- **Enums:** `UPPER_CASE` values (e.g., `TRAINING`, `EVALUATING`, `PROMOTED`)
- **Functions/Methods:** `snake_case` (e.g., `transition()`, `save_metrics()`, `evaluate_model()`)
- **Module Structure:** One primary class per module; helper functions in same module

### 3.2 Type System & Validation

**Mandatory Rules:**
1. **All functions** must have type hints on parameters and return type
2. **All configuration** must use `Pydantic BaseModel` classes, never raw dicts
3. **All validation** happens at Pydantic model instantiation, not in business logic
4. **Enums** used for fixed choice values (e.g., `PromotionState`, not string literals)

**Example:**
```python
class EvaluationConfig(BaseModel):
    """Simulator evaluation parameters."""
    simulator_version: str = Field(default="4.2.0", description="...")
    num_laps: int = Field(default=3, description="...")
    timeout_seconds: int = Field(default=300, description="...")
```

### 3.3 Documentation & Docstrings

**Required on:**
- All classes (one-line summary + detailed description)
- All public methods (parameter descriptions, return description, example if complex)
- All Pydantic fields (use `Field(description="...")`)

**Pattern:**
```python
def evaluate_model(self, model_path: Path, iteration: int, dry_run: bool = False) -> Dict:
    """Evaluate model in simulator.
    
    Args:
        model_path: Path to checkpoint file
        iteration: Iteration number for logging
        dry_run: If True, return dummy metrics without running simulator
        
    Returns:
        Dict with keys: lap_time, cte_mean, cte_max, success, timestamp
    """
```

### 3.4 Imports & Organization

- Standard library imports first
- Third-party imports second (grouped by category)
- Project imports last
- One blank line between groups

**Example:**
```python
from pathlib import Path
from typing import Dict, Optional
from enum import Enum

import typer
from pydantic import BaseModel, Field

from .config import ExperimentConfig
from .artifacts import ArtifactManager
```

---

## 4. Configuration & Reproducibility

### 4.1 Hydra & OmegaConf

**Master Config:** `configs/experiment.yaml` contains:
```yaml
training:
  learning_rate: 0.001
  batch_size: 32
  ...
evaluation:
  simulator_version: "4.2.0"
  ...
promotion:
  metric_threshold: 25.0
  ...
environment:
  runs_dir: ./runs
  ...
```

**Modular Sub-Configs:** Each subsystem has its own YAML file (imported into master).

**Agent Rule:** When adding config options:
1. Define in YAML sub-config
2. Add corresponding Pydantic field in config.py
3. Load via `OmegaConf.to_object(cfg, ExperimentConfig)`

### 4.2 Artifact Reproducibility

**Rule:** Every run must save:
1. **config.snapshot.json** — Hydra config at run time (for exact reproduction)
2. **environment.lock** — Python version, key dependency versions
3. **metadata.json** — Git commit, branch, timestamp

**Agent Rule:** `ArtifactManager` handles all three automatically. Business logic must call `artifact_manager.save_config_snapshot()` at run start.

---

## 5. State Machine & Promotion Logic

### 5.1 State Transitions

**Legal Paths (enforced by `StateTracker._valid_moves()`):**
```
TRAINING → EVALUATING
EVALUATING → PROMOTION_GATE
PROMOTION_GATE → PROMOTED (if gate passes) OR TRAINING (if gate fails)
PROMOTED → ROLLED_BACK
ROLLED_BACK → TRAINING
```

**Agent Rule:** Every `transition()` call must:
1. Validate new state is in `_valid_moves()[current_state]`
2. Pass metrics dict and iteration number
3. Append to immutable `RunManifest` history
4. Auto-save to `state.json`

### 5.2 Promotion Gate Logic

**Criteria (all checked in `PromotionGate.evaluate()`):**
1. **Operability:** `operability_ok` must be True (simulator health check)
2. **Metric Threshold:** Metric ≤ configured threshold (e.g., lap_time ≤ 25 seconds)
3. **Improvement:** Percent improvement ≥ minimum (e.g., 5% better than baseline)

**Agent Rule:** Gate verdict must be deterministic given same metrics + baseline. Use explicit threshold comparisons, never probabilistic logic.

---

## 6. Logging & Observability

### 6.1 Structured Logging (structlog)

**Rule:** All logging via `structlog`, JSON output format.

**Configuration:** `configure_logging()` in `logging_config.py` sets up:
```python
configure_logging(log_dir, run_id, level="INFO")
logger = get_logger(__name__)
logger.info("event", key1=value1, key2=value2)  # JSON structured
```

**Agent Rule:** 
- Never use `print()` for operational logs
- Always use `logger.info()`, `.warning()`, `.error()`
- Include context as keyword arguments (not string formatting)

### 6.2 What to Log

Log at key checkpoints:
- Run start/end
- State transitions (current → new)
- Training iteration start/end
- Evaluation results (metrics)
- Gate verdict (pass/fail + reason)
- Promotion actions (save best model, rollback)

---

## 7. Testing Strategy

### 7.1 Integration Tests

Located in `tests/test_integration.py`. Test coverage:

1. **Config Creation** — Pydantic models instantiate with correct defaults
2. **State Machine** — State transitions follow legal paths
3. **Artifacts** — Checkpoints save/load correctly, paths are deterministic
4. **Promotion Gate** — Gating logic returns correct pass/fail verdicts

**Agent Rule:** When adding new features, add corresponding integration test validating the core logic.

### 7.2 Running Tests

```bash
python -m pytest tests/test_integration.py -v
```

---

## 8. Integration Points Status

### 8.1 Training Loop
**File:** `src/autoresearch/training.py` + `src/autoresearch/orchestrate.py:_train_iteration()`
**Status:** ✅ IMPLEMENTED — Full integration with train.py GPT training loop
**Implementation Details:**
- `TrainingIterator` class manages training setup and execution
- Lazy initialization: training environment loaded on first use (from train.py globals)
- Respects time budget: runs training loop with configurable timeout per iteration
- Checkpointing: saves model state after each iteration
- Error handling: graceful failure with error logging
- GC management: disables Python GC during training for performance
- Loss monitoring: tracks smoothed training loss and fails fast on explosion

**Configuration:** Controlled via `configs/experiment.yaml` and `TrainingConfig`:
- `learning_rate`: Initial learning rate  
- `batch_size`: Per-device batch size
- `num_epochs`: Number of training epochs
- `time_budget_minutes`: Max time per iteration (default 5.0 minutes)
- `model_type`: Model architecture identifier
- `seed`: Random seed for reproducibility

**Requirements:**
- train.py must be run at setup time to initialize tokenizer, model, optimizer, dataloader
- Existing train.py module-level setup code creates globals used by TrainingIterator
- GPT model, optimizer (MuonAdamW), and tokenizer must be available
- GPU/CUDA for acceleration (CPU fallback supported but slow)

**How It Works:**
```
First iteration:
  1. TrainingIterator lazy-loads train.py globals (model, optimizer, dataloader)
  2. Runs training loop for time_budget_minutes
  3. Saves model state to checkpoint

Subsequent iterations:
  1. Continues training from previous state
  2. Logs progress and loss metrics
  3. Saves checkpoint after time budget expires
```

---

### 8.2 Simulator Evaluation Loop
**File:** `src/autoresearch/evaluate.py:_run_simulator_loop()`
**Status:** ✅ IMPLEMENTED — Full gym-donkeycar integration
**Implementation Details:**
- Loads PyTorch checkpoint from `model_path`
- Creates `DonkeySimEnv` with deterministic seed from config
- Drives N laps (configured in `evaluation.num_laps`)
- Tracks metrics: lap_time, cte_mean, cte_max per lap
- Returns aggregated metrics dict: `{lap_time, cte_mean, cte_max, success, timestamp}`
- Handles model inference with automatic device selection (CPU/CUDA)
- Observes environment state and records cross-track error (CTE) from info dict
- Graceful error handling: returns failure metrics on exception
- All operations logged via structlog JSON output

**Configuration:** Controlled via `configs/evaluation.yaml` and `EvaluationConfig`:
- `simulator_version`: Version string (for compatibility tracking)
- `map_name`: Track map to load in simulator
- `num_laps`: Number of evaluation laps
- `seed`: Deterministic seed for reproducibility
- `timeout_seconds`: Max time per evaluation
- `port`: Simulator port (default 9091)
- `metric_name`: Primary metric for sorting

**Requirements:**
- gym-donkeycar 22.11.6+ installed and available on PATH
- Simulator executable reachable via `DONKEY_SIM_PATH` env var or standard locations
- PyTorch checkpoint at checkpoint path with compatible model interface
- Model must accept image observation (checked/normalized internally)
- Model must output steering action (and optional throttle)

---

## 9. Common Mistakes (Prevent These!)

1. ❌ **Direct filesystem writes** — Always use `ArtifactManager`
2. ❌ **String-based states** — Use `PromotionState` enum, never magic strings
3. ❌ **Untyped functions** — Add type hints to everything
4. ❌ **Mutable default arguments in Pydantic** — Use `Field(default_factory=...)`
5. ❌ **Missing docstrings** — Document all classes and public methods
6. ❌ **print() for logging** — Use structlog logger
7. ❌ **Raw dicts for config** — Create Pydantic models
8. ❌ **Changing artifact schemas** — Coordinate changes; JSON schemas are contracts

---

## 10. Quick Reference: File Locations

| Purpose | Location | Notes |
|---------|----------|-------|
| CLI Commands | `src/autoresearch/cli.py` | Typer decorated functions |
| Config Models | `src/autoresearch/config.py` | Pydantic BaseModel subclasses |
| State Machine | `src/autoresearch/state.py` | StateTracker + StateView |
| Artifact I/O | `src/autoresearch/artifacts.py` | ArtifactManager |
| Evaluation | `src/autoresearch/evaluate.py` | SimulatorEvaluator (PLACEHOLDER: `_run_simulator_loop`) |
| Promotion Logic | `src/autoresearch/promote.py` | PromotionGate.evaluate() |
| Model Export | `src/autoresearch/export.py` | ONNXExporter |
| Orchestration | `src/autoresearch/orchestrate.py` | Orchestrator (PLACEHOLDER: `_train_iteration`) |
| Logging Config | `src/autoresearch/logging_config.py` | structlog setup |
| Hydra Configs | `configs/*.yaml` | experiment.yaml (master), modular sub-configs |
| Tests | `tests/test_integration.py` | Integration test suite |

---

## Document Status

✅ **Sections Completed:** Discovery (1), Tech Stack (2), Patterns (3)
⏳ **Next Phase:** Refinement & Project-Specific Rules

This document is **ready for collaborative refinement** to add project-specific rules, additional integration guidance, and specialized patterns as needed.

