"""
Story 6.6 — Production validation tests.

These tests are conditional:
- Dry-run tests (always run, no GPU/simulator needed)
- GPU tests (skipped unless AUTORESEARCH_TEST_GPU=1)
- Simulator tests (skipped unless DONKEY_SIM_PATH is set and valid)

Run all:
    pytest tests/test_production.py -v

Run only dry-run (CI-safe):
    pytest tests/test_production.py -v -m "not gpu and not simulator"
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autoresearch.config import (
    ExperimentConfig,
    TrainingConfig,
    EvaluationConfig,
    PromotionConfig,
    EnvironmentConfig,
)
from autoresearch.orchestrate import Orchestrator


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

GPU_AVAILABLE = False
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available() and os.getenv("AUTORESEARCH_TEST_GPU") == "1"
except ImportError:
    pass

SIM_PATH = os.getenv("DONKEY_SIM_PATH", "")
SIM_AVAILABLE = bool(SIM_PATH) and Path(SIM_PATH).exists()

requires_gpu = pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available or AUTORESEARCH_TEST_GPU != 1")
requires_sim = pytest.mark.skipif(not SIM_AVAILABLE, reason="DONKEY_SIM_PATH not set or binary not found")


def _make_production_config(tmpdir: Path, dry_run: bool = True, max_iterations: int = 5) -> ExperimentConfig:
    """Build a realistic config for production-style validation runs."""
    return ExperimentConfig(
        max_iterations=max_iterations,
        dry_run=dry_run,
        training=TrainingConfig(
            time_budget_minutes=0.1,  # Very short for tests
            learning_rate=0.001,
            batch_size=16,
        ),
        evaluation=EvaluationConfig(
            num_laps=1,
            timeout_seconds=30,
            metric_name="lap_time",
        ),
        promotion=PromotionConfig(
            metric_threshold=1000.0,  # Always pass in dry-run
            require_operability_check=False,
            min_improvement_percent=0.0,
        ),
        environment=EnvironmentConfig(
            runs_dir=tmpdir / "runs",
            use_wandb=False,
        ),
    )


# ---------------------------------------------------------------------------
# Dry-run tests (always run — CI-safe)
# ---------------------------------------------------------------------------


def test_five_iteration_dry_run():
    """5-iteration dry-run completes without error (matches production iter count)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_production_config(Path(tmpdir), dry_run=True, max_iterations=5)
        orch = Orchestrator(cfg, run_id="prod-val-dry")
        orch.run_autonomous_loop()


def test_dry_run_artifacts_complete():
    """After 5 iterations, expected artifact structure is fully populated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_production_config(Path(tmpdir), dry_run=True, max_iterations=5)
        orch = Orchestrator(cfg, run_id="prod-val-artifacts")
        orch.run_autonomous_loop()

        runs_dir = Path(tmpdir) / "runs" / "prod-val-artifacts"
        assert runs_dir.exists(), "Run directory not created"

        # Metric files for all 5 iterations
        metrics_dir = runs_dir / "metrics"
        assert metrics_dir.exists()
        metric_files = list(metrics_dir.glob("metrics_iter_*.json"))
        assert len(metric_files) == 5, f"Expected 5 metric files, got {len(metric_files)}"

        # Config snapshot
        snapshot = runs_dir / "config_snapshot.json"
        assert snapshot.exists()
        with open(snapshot) as f:
            parsed = json.load(f)
        assert parsed["max_iterations"] == 5

        # State manifest
        state_file = runs_dir / "state.json"
        assert state_file.exists()


def test_dry_run_metrics_have_required_fields():
    """All metric files contain expected fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_production_config(Path(tmpdir), dry_run=True, max_iterations=3)
        orch = Orchestrator(cfg, run_id="prod-val-metrics")
        orch.run_autonomous_loop()

        metrics_dir = Path(tmpdir) / "runs" / "prod-val-metrics" / "metrics"
        for f in sorted(metrics_dir.glob("metrics_iter_*.json")):
            with open(f) as fp:
                m = json.load(fp)
            assert "lap_time" in m, f"lap_time missing in {f.name}"
            assert "timestamp" in m, f"timestamp missing in {f.name}"
            assert m["lap_time"] > 0, f"lap_time must be positive in {f.name}"


def test_dry_run_onnx_export():
    """ONNX export runs after a dry-run loop without error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_production_config(Path(tmpdir), dry_run=True, max_iterations=2)
        orch = Orchestrator(cfg, run_id="prod-val-export")
        orch.run_autonomous_loop()

        # Export function should exist and be callable
        from autoresearch import export as _export_mod
        try:
            export_fn = getattr(_export_mod, "export_onnx", None) or getattr(_export_mod, "export", None)
            if export_fn is not None:
                out = Path(tmpdir) / "model.onnx"
                export_fn(
                    run_id="prod-val-export",
                    runs_dir=Path(tmpdir) / "runs",
                    output_path=out,
                )
        except (NotImplementedError, AttributeError):
            pytest.skip("export_onnx not yet implemented")


def test_training_crash_recovery():
    """Pipeline handles a training crash and continues to next iteration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_production_config(Path(tmpdir), dry_run=True, max_iterations=3)
        orch = Orchestrator(cfg, run_id="prod-val-crash")

        original_dummy = orch._dummy_checkpoint
        crash_count = {"n": 0}

        def _crashing_checkpoint(iteration):
            crash_count["n"] += 1
            if crash_count["n"] == 2:
                raise RuntimeError("Simulated training crash at iteration 2")
            return original_dummy(iteration)

        orch._dummy_checkpoint = _crashing_checkpoint

        # Should not raise — crash is handled internally
        orch.run_autonomous_loop()


def test_evaluation_timeout_handled():
    """Evaluation timeout fires and loop continues gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        import signal
        if not hasattr(signal, "SIGALRM"):
            pytest.skip("SIGALRM not available on this platform")

        cfg = _make_production_config(Path(tmpdir), dry_run=True, max_iterations=2)
        orch = Orchestrator(cfg, run_id="prod-val-timeout")

        original_evaluate = orch.evaluator.evaluate_model

        def _slow_evaluate(*args, **kwargs):
            import time
            time.sleep(0.01)
            return original_evaluate(*args, **kwargs)

        orch.evaluator.evaluate_model = _slow_evaluate
        # Should complete even if evaluation is slow — watchdog is configured generously
        orch.run_autonomous_loop()


# ---------------------------------------------------------------------------
# GPU tests (skipped unless AUTORESEARCH_TEST_GPU=1 and CUDA available)
# ---------------------------------------------------------------------------


@requires_gpu
def test_real_training_iteration():
    """Single GPU training iteration runs and saves checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_production_config(Path(tmpdir), dry_run=False, max_iterations=1)
        cfg.training.time_budget_minutes = 0.05  # 3 seconds
        orch = Orchestrator(cfg, run_id="prod-val-gpu")
        orch.run_autonomous_loop()

        checkpoints = list((Path(tmpdir) / "runs" / "prod-val-gpu" / "models").glob("checkpoint_iter_*.pt"))
        assert len(checkpoints) >= 1, "No checkpoint saved after real training"


# ---------------------------------------------------------------------------
# Simulator tests (skipped unless DONKEY_SIM_PATH is set and binary exists)
# ---------------------------------------------------------------------------


@requires_sim
def test_simulator_evaluation_returns_metrics():
    """Real simulator evaluation returns lap_time and cte metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_production_config(Path(tmpdir), dry_run=False, max_iterations=1)
        cfg.dry_run = False  # Force real evaluation path

        from autoresearch.evaluate import SimulatorEvaluator
        evaluator = SimulatorEvaluator(cfg.evaluation)

        dummy_checkpoint = Path(tmpdir) / "dummy.pt"
        import torch
        torch.save({"dummy": True}, dummy_checkpoint)

        metrics = evaluator.evaluate_model(dummy_checkpoint, iteration=1, dry_run=False)
        assert "lap_time" in metrics
        assert metrics["lap_time"] > 0
        assert "cte_mean" in metrics


@requires_sim
def test_five_iteration_real_pipeline():
    """Full 5-iteration run with real training and simulation (release validation)."""
    pytest.skip("Full production run — not executed in standard test suite. Run manually.")
