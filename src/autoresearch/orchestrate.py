"""Main orchestration loop - coordinates training, evaluation, and promotion."""

from pathlib import Path
from typing import Optional, Dict, Any
import time
import torch

from .config import ExperimentConfig, PromotionState
from .state import StateTracker
from .artifacts import ArtifactManager
from .evaluate import SimulatorEvaluator
from .promote import PromotionGate
from .logging_config import get_logger
from .training import get_training_iterator

logger = get_logger(__name__)


class Orchestrator:
    """
    Coordinates the full autonomous loop:
    1. Train model for time budget
    2. Evaluate in simulator
    3. Check promotion gate
    4. Loop or export
    """

    def __init__(self, config: ExperimentConfig, run_id: str):
        self.config = config
        self.run_id = run_id
        
        # Initialize subsystems
        self.state_tracker = StateTracker(run_id, config.environment.runs_dir / run_id)
        self.artifact_manager = ArtifactManager(run_id, config.environment.runs_dir)
        self.evaluator = SimulatorEvaluator(config.evaluation)
        self.promotion_gate = PromotionGate(config.promotion)
        
        self.logger = get_logger("orchestrator")
        
        # Save config snapshot
        self.artifact_manager.save_config_snapshot(config)
        self.artifact_manager.save_metadata()

    def run_autonomous_loop(self) -> None:
        """
        Execute the full training loop.
        
        For each iteration:
        - Train model
        - Evaluate in simulator
        - Evaluate promotion gate
        - Keep/discard decision
        - Optional: export ONNX
        """
        self.logger.info("Starting autonomous loop", run_id=self.run_id, max_iterations=self.config.max_iterations)
        self.state_tracker.transition(PromotionState.TRAINING)

        best_metric = None

        for iteration in range(1, self.config.max_iterations + 1):
            self.logger.info("Starting iteration", iteration=iteration)

            # ===== TRAINING STAGE =====
            if self.config.dry_run:
                model_path = self._dummy_checkpoint(iteration)
            else:
                model_path = self._train_iteration(iteration)

            # ===== EVALUATION STAGE =====
            self.state_tracker.transition(PromotionState.EVALUATING, iteration=iteration)
            
            try:
                metrics = self.evaluator.evaluate_model(
                    model_path,
                    iteration,
                    dry_run=self.config.dry_run,
                )
            except Exception as e:
                self.logger.error("Evaluation failed", iteration=iteration, error=str(e))
                continue

            # Save evaluation metrics
            self.artifact_manager.save_metrics(iteration, metrics)
            self.state_tracker.update_latest_metrics(metrics)

            # ===== PROMOTION GATE STAGE =====
            self.state_tracker.transition(PromotionState.PROMOTION_GATE, iteration=iteration, metrics=metrics)

            operability_ok = self.evaluator.check_operability(dry_run=self.config.dry_run)
            should_promote, verdict = self.promotion_gate.evaluate(
                metrics,
                operability_ok,
                best_metric,
            )

            # ===== DECISION =====
            if should_promote:
                self.state_tracker.transition(
                    PromotionState.PROMOTED,
                    iteration=iteration,
                    verdict=verdict,
                )
                # Update best model
                best_model_path = self.artifact_manager.best_model_path()
                if model_path.exists():
                    import shutil
                    shutil.copy(model_path, best_model_path)
                    self.logger.info("Model promoted", iteration=iteration, path=str(best_model_path))
                
                best_metric = metrics.get("lap_time")
            else:
                # Discard and loop
                self.state_tracker.transition(
                    PromotionState.TRAINING,
                    iteration=iteration,
                    verdict=f"Discarded: {verdict}",
                )
                self.logger.info("Model discarded", iteration=iteration, reason=verdict)

            # Brief pause between iterations
            time.sleep(1)

        self.logger.info("Autonomous loop complete", run_id=self.run_id)

    def _train_iteration(self, iteration: int) -> Path:
        """
        Train model for one iteration with time budget.
        
        Integrates with train.py training loop, running with bounded time budget.
        """
        checkpoint_path = self.artifact_manager.checkpoint_path(iteration)
        
        try:
            # Get training iterator (lazy-initialized on first use)
            training_iterator = get_training_iterator(
                time_budget_minutes=self.config.training.time_budget_minutes
            )
            
            # Run training loop with time budget constraint
            returned_path = training_iterator.run_iteration(iteration, checkpoint_path)
            
            self.logger.info(
                "Training complete",
                iteration=iteration,
                checkpoint=str(returned_path),
            )
            return returned_path
            
        except Exception as e:
            self.logger.error(
                "Training failed",
                iteration=iteration,
                error=str(e),
                error_type=type(e).__name__,
            )
            # Return dummy checkpoint on failure
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"error": True, "message": str(e)}, checkpoint_path)
            return checkpoint_path

    def _dummy_checkpoint(self, iteration: int) -> Path:
        """Create dummy checkpoint for dry-run."""
        checkpoint_path = self.artifact_manager.checkpoint_path(iteration)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"iteration": iteration}, checkpoint_path)
        return checkpoint_path

    def status(self) -> Dict[str, Any]:
        """Get current run status."""
        current_state = self.state_tracker.current_state()
        return {
            "run_id": self.run_id,
            "state": current_state.value,
            "state_history": [
                {
                    "state": t.state.value,
                    "iteration": t.iteration,
                    "timestamp": t.timestamp.isoformat(),
                }
                for t in self.state_tracker.state_history()
            ],
            "latest_metrics": self.state_tracker.manifest.latest_metrics,
        }
