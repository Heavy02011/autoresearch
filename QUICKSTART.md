# Quick Start - autoresearch

## What's Ready

The modular **DonkeyCar steering optimization framework** is fully implemented with:

- ✓ 9 core Python modules (~900 LOC)
- ✓ Pydantic configuration models
- ✓ State machine for model promotion
- ✓ Artifact management system
- ✓ CLI interface (Typer)
- ✓ Integration test suite
- ✓ Hydra configuration files

## Try It Now

### 1. Install Dependencies
```bash
cd /workspaces/autoresearch
pip install pydantic typer omegaconf structlog wandb
```

### 2. Run Dry-Run Test
```bash
python -m autoresearch.cli run --dry-run --max-iterations 2
```

Expected output:
```
Starting autonomous loop...
✓ Run complete: run_20260322_123456

Final Status:
{
  "run_id": "run_20260322_123456",
  "state": "promoted",
  "state_history": [
    {"state": "training", "iteration": null, ...},
    {"state": "evaluating", "iteration": 1, ...},
    {"state": "promotion_gate", "iteration": 1, ...},
    {"state": "promoted", "iteration": 1, ...}
  ],
  "latest_metrics": {
    "lap_time": 24.5
  }
}
```

Artifacts created in `runs/run_20260322_123456/`:
```
runs/run_20260322_123456/
├── state.json                 ← Full state machine history
├── config.snapshot.json       ← Experiment config snapshot
├── metadata.json              ← Git commit, timestamps
├── models/
│   ├── checkpoint_iter_1.pt
│   └── best_model.pt
├── metrics/
│   └── eval_iter_1.json
└── logs/
    └── run_20260322_123456.log
```

### 3. Check Run Status
```bash
python -m autoresearch.cli status run_20260322_123456
```

### 4. Run Integration Tests
```bash
python tests/test_integration.py
```

## File Map

### Core Modules
- **config.py** — Pydantic models for all configurations
- **state.py** — State machine with immutable history
- **artifacts.py** — Filesystem artifact management
- **evaluate.py** — Simulator evaluation interface (placeholder)
- **promote.py** — Promotion gating logic
- **export.py** — ONNX model export
- **orchestrate.py** — Main autonomous loop
- **cli.py** — Typer CLI commands
- **logging_config.py** — Structured logging setup

### Configuration
- **configs/experiment.yaml** — Master experiment config
- **configs/evaluation.yaml** — Simulator settings
- **configs/promotion.yaml** — Gating criteria
- **configs/environment.yaml** — Paths and environment

## What's Integrated

✓ **Config Management** — Pydantic + Hydra YAML  
✓ **State Tracking** — State machine with audit trail  
✓ **Artifact Storage** — Hierarchical filesystem  
✓ **CLI Interface** — Full Typer command structure  
✓ **Testing** — Integration tests for core modules  

## What Needs Integration

The following are **placeholders** waiting for your training/simulator code:

1. **Training Loop** (`orchestrate.py:_train_iteration()`)
   - Replace dummy checkpoint creation with actual training
   - Integrate with your `train.py` logic

2. **Simulator Evaluation** (`evaluate.py:_run_simulator_loop()`)
   - Implement gym-donkeycar environment setup
   - Drive model and record metrics

3. **Model Export** (`export.py`)
   - Wire up with exported ONNX models

4. **W&B Integration** (optional)
   - Add W&B logging if `use_wandb=true`

## Architecture at a Glance

```
┌─────────────────────────────────────┐
│  CLI: autoresearch run              │
└──────────────┬──────────────────────┘
               │
        ┌──────▼──────┐
        │ Orchestrator │
        └──────┬───────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐ ┌───▼────┐ ┌───▼────┐
│Train  │ │Evaluate│ │Promote │
└───┬───┘ └───┬────┘ │Gate    │
    │         │      └───┬────┘
    └─────────┼──────────┘
              │
        ┌─────▼──────┐
        │ StateTracker│ ← Immutable history
        │ Artifacts   │ ← Filesystem storage
        └─────────────┘
```

## Next Steps

### Option A: Integrate Training (DIY)
1. Modify `orchestrate.py:_train_iteration()` to call your training code
2. Modify `evaluate.py:_run_simulator_loop()` to use gym-donkeycar
3. Test end-to-end

### Option B: Create Implementation Epics
Create detailed stories for:
- S1: Training loop integration
- S2: Simulator integration
- S3: End-to-end testing
- S4: W&B dashboard integration

### Option C: Continue with "YOLO Mode"
Start coding integrations directly. Framework is ready for your training/sim code.

---

**Framework Status:** 🟢 Ready for integration  
**Code Quality:** Production-ready patterns + tests  
**Documentation:** Complete API + usage guide  

Next move is yours! 🚗
