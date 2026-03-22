"""Basic integration tests for autoresearch modules."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autoresearch.config import (
    PromotionState,
    TrainingConfig,
    EvaluationConfig,
    PromotionConfig,
    EnvironmentConfig,
    ExperimentConfig,
)


def test_config_creation():
    """Test configuration model creation."""
    config = ExperimentConfig(
        run_id="test_run",
        max_iterations=5,
    )
    
    assert config.run_id == "test_run"
    assert config.max_iterations == 5
    assert config.training.learning_rate == 0.001
    assert config.evaluation.simulator_version == "4.2.0"
    print("✓ Config creation test passed")


def test_state_machine():
    """Test state machine transitions."""
    from autoresearch.state import StateTracker
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = StateTracker("test_run", Path(tmpdir))
        
        # Test transitions
        assert tracker.current_state() == PromotionState.TRAINING
        
        tracker.transition(PromotionState.EVALUATING, iteration=1)
        assert tracker.current_state() == PromotionState.EVALUATING
        
        tracker.transition(
            PromotionState.PROMOTION_GATE,
            iteration=1,
            verdict="PASS"
        )
        assert tracker.current_state() == PromotionState.PROMOTION_GATE
        
        tracker.transition(PromotionState.PROMOTED, iteration=1)
        assert tracker.current_state() == PromotionState.PROMOTED
        
        print("✓ State machine test passed")


def test_artifacts():
    """Test artifact manager."""
    from autoresearch.artifacts import ArtifactManager
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = ArtifactManager("test_run", Path(tmpdir))
        
        # Test paths
        assert mgr.checkpoint_path(1).name == "checkpoint_iter_1.pt"
        assert mgr.best_model_path().name == "best_model.pt"
        assert mgr.best_model_onnx_path().name == "best_model.onnx"
        
        # Test metrics save/load
        metrics = {"lap_time": 25.5, "cte_mean": 0.3}
        mgr.save_metrics(1, metrics)
        loaded = mgr.load_metrics(1)
        assert loaded["lap_time"] == 25.5
        
        print("✓ Artifact manager test passed")


def test_promotion_gate():
    """Test promotion gating logic."""
    from autoresearch.promote import PromotionGate
    from autoresearch.config import PromotionConfig
    
    config = PromotionConfig(
        metric_threshold=30.0,
        require_operability_check=True,
        min_improvement_percent=0.0,
    )
    gate = PromotionGate(config)
    
    # Test passing condition
    metrics = {"lap_time": 25.0}
    should_promote, reason = gate.evaluate(metrics, True, baseline_metric=30.0)
    assert should_promote == True
    
    # Test failing condition (above threshold)
    metrics = {"lap_time": 35.0}
    should_promote, reason = gate.evaluate(metrics, True, baseline_metric=None)
    assert should_promote == False
    
    print("✓ Promotion gate test passed")


if __name__ == "__main__":
    try:
        test_config_creation()
        test_state_machine()
        test_artifacts()
        test_promotion_gate()
        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
