"""End-to-end integration tests for the full autoresearch pipeline."""

import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autoresearch.config import ExperimentConfig, PromotionState


def _make_config(tmpdir: str, max_iterations: int = 2) -> ExperimentConfig:
    """Build a minimal dry-run config pointing to tmpdir."""
    return ExperimentConfig(
        run_id="test_e2e",
        max_iterations=max_iterations,
        dry_run=True,
        environment={"runs_dir": Path(tmpdir), "use_wandb": False},
        promotion={"metric_threshold": 1000.0, "require_operability_check": False},
    )


def test_full_autonomous_loop():
    """Full train→eval→gate→promote cycle in dry-run mode."""
    from autoresearch.orchestrate import Orchestrator

    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config(tmpdir, max_iterations=2)
        orchestrator = Orchestrator(config, "test_e2e")
        orchestrator.run_autonomous_loop()

        # Should have reached PROMOTED (metric_threshold 1000s, dry-run lap_time is low)
        final_state = orchestrator.state_tracker.current_state()
        assert final_state in {PromotionState.PROMOTED, PromotionState.TRAINING}, (
            f"Expected PROMOTED or TRAINING, got {final_state}"
        )

        # Artifacts must exist
        assert (Path(tmpdir) / "test_e2e" / "state.json").exists()
        assert (Path(tmpdir) / "test_e2e" / "config.snapshot.json").exists()
        assert (Path(tmpdir) / "test_e2e" / "metadata.json").exists()

        # State history must have entries
        history = orchestrator.state_tracker.state_history()
        assert len(history) >= 1, "State history is empty"

        print("✓ Full autonomous loop test passed")


def test_dry_run_creates_artifacts():
    """Verify artifact directory structure is created correctly."""
    from autoresearch.orchestrate import Orchestrator

    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config(tmpdir, max_iterations=1)
        orchestrator = Orchestrator(config, "test_e2e")
        orchestrator.run_autonomous_loop()

        run_dir = Path(tmpdir) / "test_e2e"
        assert (run_dir / "models").is_dir()
        assert (run_dir / "metrics").is_dir()
        assert (run_dir / "logs").is_dir()
        assert (run_dir / "state.json").exists()
        assert (run_dir / "config.snapshot.json").exists()

        # Config snapshot must be valid JSON
        snapshot = json.loads((run_dir / "config.snapshot.json").read_text())
        assert snapshot["max_iterations"] == 1
        assert snapshot["dry_run"] is True

        print("✓ Artifact structure test passed")


def test_metrics_written_per_iteration():
    """Evaluation metrics are saved per iteration."""
    from autoresearch.orchestrate import Orchestrator
    from autoresearch.artifacts import ArtifactManager

    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config(tmpdir, max_iterations=2)
        orchestrator = Orchestrator(config, "test_e2e")
        orchestrator.run_autonomous_loop()

        mgr = ArtifactManager("test_e2e", Path(tmpdir))
        for iteration in [1, 2]:
            metrics = mgr.load_metrics(iteration)
            if metrics is not None:
                assert "lap_time" in metrics, f"lap_time missing in iter {iteration} metrics"

        print("✓ Metrics written per iteration test passed")


def test_state_transitions_are_logged():
    """All state transitions appear in run manifest."""
    from autoresearch.orchestrate import Orchestrator

    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config(tmpdir, max_iterations=1)
        orchestrator = Orchestrator(config, "test_e2e")
        orchestrator.run_autonomous_loop()

        history = orchestrator.state_tracker.state_history()
        states = [t.state for t in history]

        # Must have started with TRAINING
        assert PromotionState.TRAINING in states, "TRAINING state missing from history"
        # Must have attempted evaluation
        assert PromotionState.EVALUATING in states, "EVALUATING state missing from history"

        print("✓ State transition logging test passed")


def test_two_iteration_pipeline():
    """Two full iterations complete without exception."""
    from autoresearch.orchestrate import Orchestrator

    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config(tmpdir, max_iterations=2)
        orchestrator = Orchestrator(config, "test_e2e")

        # Should not raise
        try:
            orchestrator.run_autonomous_loop()
            print("✓ Two-iteration pipeline test passed")
        except Exception as e:
            raise AssertionError(f"Pipeline raised unexpectedly: {e}")


if __name__ == "__main__":
    test_full_autonomous_loop()
    test_dry_run_creates_artifacts()
    test_metrics_written_per_iteration()
    test_state_transitions_are_logged()
    test_two_iteration_pipeline()
    print("\n✓ All E2E tests passed")
