"""State machine edge case tests - legal/illegal transitions, rollback, immutability."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autoresearch.config import PromotionState


def test_initial_state_is_training():
    """New tracker starts in TRAINING state."""
    from autoresearch.state import StateTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = StateTracker("test", Path(tmpdir))
        assert tracker.current_state() == PromotionState.TRAINING
    print("✓ Initial state is TRAINING")


def test_full_happy_path_transitions():
    """TRAINING → EVALUATING → PROMOTION_GATE → PROMOTED."""
    from autoresearch.state import StateTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = StateTracker("test", Path(tmpdir))
        tracker.transition(PromotionState.EVALUATING, iteration=1)
        assert tracker.current_state() == PromotionState.EVALUATING

        tracker.transition(PromotionState.PROMOTION_GATE, iteration=1, verdict="PASS")
        assert tracker.current_state() == PromotionState.PROMOTION_GATE

        tracker.transition(PromotionState.PROMOTED, iteration=1)
        assert tracker.current_state() == PromotionState.PROMOTED
    print("✓ Full happy path transitions")


def test_reject_cycles_back_to_training():
    """After gate, can cycle back to TRAINING on rejection."""
    from autoresearch.state import StateTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = StateTracker("test", Path(tmpdir))
        tracker.transition(PromotionState.EVALUATING, iteration=1)
        tracker.transition(PromotionState.PROMOTION_GATE, iteration=1, verdict="FAIL")
        tracker.transition(PromotionState.TRAINING, iteration=1, verdict="Discarded")
        assert tracker.current_state() == PromotionState.TRAINING
    print("✓ Reject cycles back to TRAINING")


def test_promoted_can_cycle_to_training():
    """PROMOTED → TRAINING is allowed for next iteration."""
    from autoresearch.state import StateTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = StateTracker("test", Path(tmpdir))
        tracker.transition(PromotionState.EVALUATING, iteration=1)
        tracker.transition(PromotionState.PROMOTION_GATE, iteration=1)
        tracker.transition(PromotionState.PROMOTED, iteration=1)
        tracker.transition(PromotionState.TRAINING, iteration=2)
        assert tracker.current_state() == PromotionState.TRAINING
    print("✓ PROMOTED → TRAINING allowed")


def test_illegal_transition_raises():
    """Skipping states raises ValueError."""
    from autoresearch.state import StateTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = StateTracker("test", Path(tmpdir))
        # TRAINING → PROMOTED directly is illegal
        try:
            tracker.transition(PromotionState.PROMOTED, iteration=1)
            raise AssertionError("Expected ValueError not raised")
        except ValueError as e:
            assert "Invalid transition" in str(e)
    print("✓ Illegal transition raises ValueError")


def test_illegal_transition_training_to_rollback():
    """Cannot roll back from TRAINING (no promoted state)."""
    from autoresearch.state import StateTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = StateTracker("test", Path(tmpdir))
        # No promoted state — rollback should fail
        try:
            tracker.rollback()
            raise AssertionError("Expected ValueError not raised")
        except ValueError as e:
            assert "no previous promoted state" in str(e)
    print("✓ Rollback with no promoted state raises ValueError")


def test_rollback_from_promoted():
    """Can rollback from PROMOTED to ROLLED_BACK."""
    from autoresearch.state import StateTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = StateTracker("test", Path(tmpdir))
        tracker.transition(PromotionState.EVALUATING, iteration=1)
        tracker.transition(PromotionState.PROMOTION_GATE, iteration=1)
        tracker.transition(PromotionState.PROMOTED, iteration=1)
        tracker.rollback()
        assert tracker.current_state() == PromotionState.ROLLED_BACK
    print("✓ Rollback from PROMOTED succeeds")


def test_rolled_back_can_continue_training():
    """ROLLED_BACK → TRAINING is allowed to resume loop."""
    from autoresearch.state import StateTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = StateTracker("test", Path(tmpdir))
        tracker.transition(PromotionState.EVALUATING, iteration=1)
        tracker.transition(PromotionState.PROMOTION_GATE, iteration=1)
        tracker.transition(PromotionState.PROMOTED, iteration=1)
        tracker.rollback()
        tracker.transition(PromotionState.TRAINING, iteration=2)
        assert tracker.current_state() == PromotionState.TRAINING
    print("✓ ROLLED_BACK → TRAINING allowed")


def test_state_history_is_immutable_copy():
    """get_history() returns a copy, not the live list."""
    from autoresearch.state import StateTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = StateTracker("test", Path(tmpdir))
        history = tracker.state_history()
        original_len = len(history)

        # Mutate returned copy
        history.append(None)  # type: ignore

        # Live history must be unchanged
        assert len(tracker.state_history()) == original_len
    print("✓ State history is an immutable copy")


def test_state_persisted_to_disk():
    """State survives tracker restart (loaded from disk)."""
    from autoresearch.state import StateTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = StateTracker("test", Path(tmpdir))
        tracker.transition(PromotionState.EVALUATING, iteration=1)
        tracker.transition(PromotionState.PROMOTION_GATE, iteration=1)
        tracker.transition(PromotionState.PROMOTED, iteration=1)

        # Create new tracker from same directory — should reload state
        tracker2 = StateTracker("test", Path(tmpdir))
        assert tracker2.current_state() == PromotionState.PROMOTED
        assert len(tracker2.state_history()) == len(tracker.state_history())
    print("✓ State persisted and reloaded from disk")


def test_multiple_iterations_history():
    """State history accumulates correctly over multiple iterations."""
    from autoresearch.state import StateTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = StateTracker("test", Path(tmpdir))

        for i in range(1, 4):
            tracker.transition(PromotionState.EVALUATING, iteration=i)
            tracker.transition(PromotionState.PROMOTION_GATE, iteration=i)
            tracker.transition(PromotionState.PROMOTED, iteration=i)
            if i < 3:
                tracker.transition(PromotionState.TRAINING, iteration=i + 1)

        history = tracker.state_history()
        # 3 iterations × 3 transitions each + initial TRAINING = 10 transitions
        assert len(history) >= 9, f"Expected >=9 transitions, got {len(history)}"
    print("✓ Multiple iterations history accumulated correctly")


if __name__ == "__main__":
    test_initial_state_is_training()
    test_full_happy_path_transitions()
    test_reject_cycles_back_to_training()
    test_promoted_can_cycle_to_training()
    test_illegal_transition_raises()
    test_illegal_transition_training_to_rollback()
    test_rollback_from_promoted()
    test_rolled_back_can_continue_training()
    test_state_history_is_immutable_copy()
    test_state_persisted_to_disk()
    test_multiple_iterations_history()
    print("\n✓ All state machine edge case tests passed")
