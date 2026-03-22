"""Config validation, promotion gate, and artifact management tests."""

import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ─── Config Validation Tests ─────────────────────────────────────────────────

def test_default_config():
    from autoresearch.config import ExperimentConfig
    cfg = ExperimentConfig()
    assert cfg.max_iterations == 10
    assert cfg.dry_run is False
    assert cfg.training.time_budget_minutes == 5.0
    assert cfg.evaluation.num_laps == 3
    assert cfg.promotion.metric_threshold == 25.0
    print("✓ Default config values correct")


def test_config_field_override():
    from autoresearch.config import ExperimentConfig
    cfg = ExperimentConfig(max_iterations=99, dry_run=True)
    assert cfg.max_iterations == 99
    assert cfg.dry_run is True
    print("✓ Config field override works")


def test_nested_config_override():
    from autoresearch.config import ExperimentConfig
    cfg = ExperimentConfig(
        training={"learning_rate": 0.01, "time_budget_minutes": 10.0},
        promotion={"metric_threshold": 30.0, "min_improvement_percent": 5.0},
    )
    assert cfg.training.learning_rate == 0.01
    assert cfg.training.time_budget_minutes == 10.0
    assert cfg.promotion.metric_threshold == 30.0
    assert cfg.promotion.min_improvement_percent == 5.0
    print("✓ Nested config override works")


def test_config_snapshot_serializable():
    from autoresearch.config import ExperimentConfig
    cfg = ExperimentConfig(run_id="snap_test", max_iterations=3)
    snapshot = cfg.dict_for_snapshot()
    # Must be JSON-serializable
    json_str = json.dumps(snapshot, default=str)
    reloaded = json.loads(json_str)
    assert reloaded["max_iterations"] == 3
    assert reloaded["run_id"] == "snap_test"
    print("✓ Config snapshot is JSON-serializable")


def test_invalid_config_raises():
    from autoresearch.config import TrainingConfig
    from pydantic import ValidationError
    try:
        TrainingConfig(learning_rate="not_a_number")
        raise AssertionError("Expected ValidationError")
    except ValidationError:
        pass
    print("✓ Invalid config raises ValidationError")


def test_environment_config_paths():
    from autoresearch.config import EnvironmentConfig
    cfg = EnvironmentConfig()
    assert isinstance(cfg.runs_dir, Path)
    assert isinstance(cfg.configs_dir, Path)
    print("✓ EnvironmentConfig paths are Path objects")


# ─── Promotion Gate Tests ─────────────────────────────────────────────────────

def test_gate_passes_when_all_criteria_met():
    from autoresearch.config import PromotionConfig
    from autoresearch.promote import PromotionGate

    gate = PromotionGate(PromotionConfig(metric_threshold=30.0, require_operability_check=True))
    should_promote, reason = gate.evaluate(
        metrics={"lap_time": 20.0},
        operability_check_passed=True,
    )
    assert should_promote is True
    print("✓ Gate passes when all criteria met")


def test_gate_fails_on_operability():
    from autoresearch.config import PromotionConfig
    from autoresearch.promote import PromotionGate

    gate = PromotionGate(PromotionConfig(metric_threshold=30.0, require_operability_check=True))
    should_promote, reason = gate.evaluate(
        metrics={"lap_time": 20.0},
        operability_check_passed=False,
    )
    assert should_promote is False
    assert "operability" in reason.lower()
    print("✓ Gate fails on operability check failure")


def test_gate_fails_on_metric_threshold():
    from autoresearch.config import PromotionConfig
    from autoresearch.promote import PromotionGate

    gate = PromotionGate(PromotionConfig(metric_threshold=30.0, require_operability_check=False))
    should_promote, reason = gate.evaluate(
        metrics={"lap_time": 35.0},
        operability_check_passed=True,
    )
    assert should_promote is False
    assert "threshold" in reason.lower() or "exceeds" in reason.lower()
    print("✓ Gate fails when lap_time exceeds threshold")


def test_gate_fails_on_insufficient_improvement():
    from autoresearch.config import PromotionConfig
    from autoresearch.promote import PromotionGate

    gate = PromotionGate(PromotionConfig(
        metric_threshold=30.0,
        require_operability_check=False,
        min_improvement_percent=10.0,
    ))
    # Only 2% improvement — below 10% minimum
    should_promote, reason = gate.evaluate(
        metrics={"lap_time": 24.5},
        operability_check_passed=True,
        baseline_metric=25.0,
    )
    assert should_promote is False
    assert "improvement" in reason.lower()
    print("✓ Gate fails on insufficient improvement")


def test_gate_passes_no_baseline():
    """First iteration: no baseline, should pass if metric ok."""
    from autoresearch.config import PromotionConfig
    from autoresearch.promote import PromotionGate

    gate = PromotionGate(PromotionConfig(
        metric_threshold=30.0,
        require_operability_check=False,
        min_improvement_percent=5.0,
    ))
    should_promote, reason = gate.evaluate(
        metrics={"lap_time": 20.0},
        operability_check_passed=True,
        baseline_metric=None,  # First iteration
    )
    assert should_promote is True
    print("✓ Gate passes first iteration with no baseline")


def test_gate_fails_missing_lap_time():
    from autoresearch.config import PromotionConfig
    from autoresearch.promote import PromotionGate

    gate = PromotionGate(PromotionConfig(require_operability_check=False))
    should_promote, reason = gate.evaluate(
        metrics={"cte_mean": 0.3},  # Missing lap_time
        operability_check_passed=True,
    )
    assert should_promote is False
    assert "lap_time" in reason.lower()
    print("✓ Gate fails when lap_time metric missing")


def test_gate_skips_operability_when_disabled():
    from autoresearch.config import PromotionConfig
    from autoresearch.promote import PromotionGate

    gate = PromotionGate(PromotionConfig(
        metric_threshold=30.0,
        require_operability_check=False,  # disabled
    ))
    # operability False but check is disabled — should still pass
    should_promote, _ = gate.evaluate(
        metrics={"lap_time": 20.0},
        operability_check_passed=False,
    )
    assert should_promote is True
    print("✓ Gate skips operability when require_operability_check=False")


# ─── Artifact Manager Tests ───────────────────────────────────────────────────

def test_artifact_directory_created():
    from autoresearch.artifacts import ArtifactManager

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = ArtifactManager("run_test", Path(tmpdir))
        assert mgr.models_dir.is_dir()
        assert mgr.metrics_dir.is_dir()
        assert mgr.logs_dir.is_dir()
    print("✓ Artifact directories created")


def test_checkpoint_path_naming():
    from autoresearch.artifacts import ArtifactManager

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = ArtifactManager("run_test", Path(tmpdir))
        assert mgr.checkpoint_path(1).name == "checkpoint_iter_1.pt"
        assert mgr.checkpoint_path(42).name == "checkpoint_iter_42.pt"
        assert mgr.best_model_path().name == "best_model.pt"
        assert mgr.best_model_onnx_path().name == "best_model.onnx"
    print("✓ Checkpoint paths correctly named")


def test_metrics_save_and_load():
    from autoresearch.artifacts import ArtifactManager

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = ArtifactManager("run_test", Path(tmpdir))
        metrics = {"lap_time": 22.5, "cte_mean": 0.3, "cte_max": 1.2, "success": True}
        mgr.save_metrics(1, metrics)
        loaded = mgr.load_metrics(1)
        assert loaded is not None
        assert loaded["lap_time"] == 22.5
        assert loaded["cte_mean"] == 0.3
    print("✓ Metrics save and load correctly")


def test_metrics_missing_returns_none():
    from autoresearch.artifacts import ArtifactManager

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = ArtifactManager("run_test", Path(tmpdir))
        loaded = mgr.load_metrics(999)
        assert loaded is None
    print("✓ Missing metrics returns None")


def test_config_snapshot_saved():
    from autoresearch.artifacts import ArtifactManager
    from autoresearch.config import ExperimentConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = ArtifactManager("run_test", Path(tmpdir))
        cfg = ExperimentConfig(run_id="run_test", max_iterations=7)
        mgr.save_config_snapshot(cfg)

        assert mgr.config_snapshot_file.exists()
        data = json.loads(mgr.config_snapshot_file.read_text())
        assert data["max_iterations"] == 7
    print("✓ Config snapshot saved correctly")


def test_metadata_saved():
    from autoresearch.artifacts import ArtifactManager

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = ArtifactManager("run_test", Path(tmpdir))
        mgr.save_metadata()

        assert mgr.metadata_file.exists()
        data = json.loads(mgr.metadata_file.read_text())
        assert data["run_id"] == "run_test"
        assert "created_at" in data
    print("✓ Metadata saved correctly")


if __name__ == "__main__":
    test_default_config()
    test_config_field_override()
    test_nested_config_override()
    test_config_snapshot_serializable()
    test_invalid_config_raises()
    test_environment_config_paths()

    test_gate_passes_when_all_criteria_met()
    test_gate_fails_on_operability()
    test_gate_fails_on_metric_threshold()
    test_gate_fails_on_insufficient_improvement()
    test_gate_passes_no_baseline()
    test_gate_fails_missing_lap_time()
    test_gate_skips_operability_when_disabled()

    test_artifact_directory_created()
    test_checkpoint_path_naming()
    test_metrics_save_and_load()
    test_metrics_missing_returns_none()
    test_config_snapshot_saved()
    test_metadata_saved()
    print("\n✓ All config/gate/artifact tests passed")
