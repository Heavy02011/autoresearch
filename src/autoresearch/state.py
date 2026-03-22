"""State machine for model promotion and experiment lifecycle."""

from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json

from .config import PromotionState, StateTransition, RunManifest


class StateTracker:
    """
    Manages discrete state transitions and immutable history.
    
    Designed for auditability: all transitions are logged with timestamp and metadata.
    State changes are atomic and cannot be reversed except via explicit rollback.
    """

    def __init__(self, run_id: str, state_dir: Path):
        """
        Initialize state tracker for a run.
        
        Args:
            run_id: Unique run identifier
            state_dir: Directory to persist state.json
        """
        self.run_id = run_id
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.state_dir / "state.json"
        
        # Load or create manifest
        if self.manifest_path.exists():
            self.manifest = RunManifest.load_json(self.manifest_path)
        else:
            self.manifest = RunManifest(run_id=run_id)
            self._save()

    def current_state(self) -> PromotionState:
        """Get current state."""
        return self.manifest.state

    def state_history(self) -> list[StateTransition]:
        """Get immutable state history."""
        return self.manifest.state_history.copy()

    def transition(
        self,
        new_state: PromotionState,
        iteration: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        verdict: Optional[str] = None,
    ) -> None:
        """
        Transition to a new state with metadata.
        
        Args:
            new_state: Target state
            iteration: Training iteration number (if applicable)
            metrics: Evaluation metrics (if applicable)
            verdict: Promotion verdict "PASS"/"FAIL" (if applicable)
        """
        # Validate legal transitions
        current = self.manifest.state
        valid_transitions = self._valid_moves(current)
        
        if new_state not in valid_transitions:
            raise ValueError(
                f"Invalid transition: {current} -> {new_state}. "
                f"Valid moves: {valid_transitions}"
            )

        # Create transition record
        transition = StateTransition(
            state=new_state,
            iteration=iteration,
            metrics=metrics,
            verdict=verdict,
        )
        
        self.manifest.state_history.append(transition)
        self.manifest.state = new_state
        self._save()

    def rollback(self, target_iteration: Optional[int] = None) -> None:
        """
        Rollback to a previous promoted state.
        
        Args:
            target_iteration: If provided, rollback to promoted state at this iteration
        """
        # Find most recent PROMOTED state
        promoted_transitions = [
            t for t in self.manifest.state_history
            if t.state == PromotionState.PROMOTED
        ]
        
        if not promoted_transitions:
            raise ValueError("Cannot rollback: no previous promoted state found")

        if target_iteration:
            matching = [t for t in promoted_transitions if t.iteration == target_iteration]
            if not matching:
                raise ValueError(f"No promoted state found at iteration {target_iteration}")
            target_transition = matching[-1]
        else:
            target_transition = promoted_transitions[-1]

        # Log rollback as new state transition
        self.transition(
            PromotionState.ROLLED_BACK,
            verdict=f"Rolled back to iteration {target_transition.iteration}"
        )

    def update_latest_metrics(self, metrics: Dict[str, float]) -> None:
        """Update metrics in manifest (doesn't change state)."""
        self.manifest.latest_metrics.update(metrics)
        self._save()

    def _valid_moves(self, from_state: PromotionState) -> set[PromotionState]:
        """Define legal state transitions."""
        transitions = {
            PromotionState.TRAINING: {PromotionState.EVALUATING},
            PromotionState.EVALUATING: {PromotionState.PROMOTION_GATE},
            PromotionState.PROMOTION_GATE: {
                PromotionState.PROMOTED,
                PromotionState.TRAINING,  # Loop back if not promoted
            },
            PromotionState.PROMOTED: {PromotionState.ROLLED_BACK, PromotionState.TRAINING},
            PromotionState.ROLLED_BACK: {PromotionState.TRAINING},
        }
        return transitions.get(from_state, set())

    def _save(self) -> None:
        """Persist manifest to disk."""
        self.manifest.save_json(self.manifest_path)


class StateView:
    """Read-only view of run state for queries."""

    def __init__(self, manifest: RunManifest):
        self.manifest = manifest

    def get_current_state(self) -> PromotionState:
        return self.manifest.state

    def get_state_at_iteration(self, iteration: int) -> Optional[PromotionState]:
        """Find what state we were in at a specific iteration."""
        for transition in reversed(self.manifest.state_history):
            if transition.iteration == iteration:
                return transition.state
        return None

    def is_promoted(self) -> bool:
        """Check if current model is promoted."""
        return self.manifest.state == PromotionState.PROMOTED

    def last_promotion_iteration(self) -> Optional[int]:
        """Get iteration number when model was promoted."""
        for transition in reversed(self.manifest.state_history):
            if transition.state == PromotionState.PROMOTED:
                return transition.iteration
        return None

    def metrics_at_iteration(self, iteration: int) -> Optional[Dict[str, Any]]:
        """Get evaluation metrics from a specific iteration."""
        for transition in self.manifest.state_history:
            if transition.iteration == iteration and transition.metrics:
                return transition.metrics
        return None
