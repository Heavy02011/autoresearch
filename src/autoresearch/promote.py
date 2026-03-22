"""Model promotion gating - decides which models to keep."""

from typing import Dict, Any, Optional
from pathlib import Path

from .config import PromotionConfig
from .logging_config import get_logger

logger = get_logger(__name__)


class PromotionGate:
    """
    Evaluates models against acceptance criteria.
    
    Criteria:
    - Lap time below threshold
    - Model operability in simulator
    - Minimum improvement over baseline
    """

    def __init__(self, config: PromotionConfig):
        self.config = config
        self.logger = get_logger("promotion")

    def evaluate(
        self,
        metrics: Dict[str, Any],
        operability_check_passed: bool,
        baseline_metric: Optional[float] = None,
    ) -> tuple[bool, str]:
        """
        Determine if model should be promoted.
        
        Args:
            metrics: Evaluation metrics (lap_time, cte, etc.)
            operability_check_passed: Did simulator health check pass?
            baseline_metric: Previous best metric for comparison
        
        Returns:
            (should_promote, reason) tuple
        """
        # Check 1: Operability requirement
        if self.config.require_operability_check and not operability_check_passed:
            reason = "Failed operability check"
            self.logger.info("Promotion decision", verdict="FAIL", reason=reason)
            return False, reason

        # Check 2: Metric threshold
        lap_time = metrics.get("lap_time")
        if lap_time is None:
            reason = "No lap_time metric found"
            self.logger.warning("Promotion decision", verdict="FAIL", reason=reason)
            return False, reason

        if lap_time > self.config.metric_threshold:
            reason = f"Lap time {lap_time:.2f}s exceeds threshold {self.config.metric_threshold}s"
            self.logger.info("Promotion decision", verdict="FAIL", reason=reason)
            return False, reason

        # Check 3: Improvement over baseline
        if baseline_metric is not None:
            improvement_percent = ((baseline_metric - lap_time) / baseline_metric) * 100
            if improvement_percent < self.config.min_improvement_percent:
                reason = (
                    f"Improvement {improvement_percent:.1f}% < "
                    f"minimum {self.config.min_improvement_percent}%"
                )
                self.logger.info("Promotion decision", verdict="FAIL", reason=reason)
                return False, reason

        # All checks passed
        reason = f"All criteria met (lap_time: {lap_time:.2f}s)"
        self.logger.info("Promotion decision", verdict="PASS", reason=reason)
        return True, reason
