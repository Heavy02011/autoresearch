"""Configuration models using Pydantic for type validation and schema."""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field


class PromotionState(str, Enum):
    """Discrete states in the promotion state machine."""
    TRAINING = "training"
    EVALUATING = "evaluating"
    PROMOTION_GATE = "promotion_gate"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"


class TrainingConfig(BaseModel):
    """Training hyperparameters."""
    learning_rate: float = Field(default=0.001, description="Learning rate for optimizer")
    batch_size: int = Field(default=32, description="Batch size")
    num_epochs: int = Field(default=5, description="Number of training epochs")
    time_budget_minutes: float = Field(default=5.0, description="Max training time in minutes")
    model_type: str = Field(default="cnn", description="Model architecture type")
    seed: int = Field(default=42, description="Random seed for reproducibility")

    class Config:
        use_enum_values = True


class EvaluationConfig(BaseModel):
    """Simulator evaluation parameters."""
    simulator_version: str = Field(default="4.2.0", description="Simulator version to use")
    map_name: str = Field(default="donkey_sim_path", description="Map/track name")
    num_laps: int = Field(default=3, description="Number of evaluation laps")
    seed: int = Field(default=42, description="Simulator seed for determinism")
    timeout_seconds: int = Field(default=300, description="Max time per evaluation")
    metric_name: str = Field(default="lap_time", description="Primary evaluation metric")
    port: int = Field(default=9091, description="Simulator port number")


class PromotionConfig(BaseModel):
    """Gating criteria for model promotion."""
    metric_threshold: float = Field(default=25.0, description="Max lap time (seconds)")
    require_operability_check: bool = Field(default=True, description="Must pass simulator health check")
    min_improvement_percent: float = Field(default=0.0, description="Min improvement from baseline (%)")
    allow_rollback: bool = Field(default=True, description="Allow rollback to previous best")


class EnvironmentConfig(BaseModel):
    """Environment and paths."""
    project_root: Path = Field(default_factory=lambda: Path.cwd())
    runs_dir: Path = Field(default_factory=lambda: Path.cwd() / "runs")
    configs_dir: Path = Field(default_factory=lambda: Path.cwd() / "configs")
    log_level: str = Field(default="INFO", description="Logging level")
    use_wandb: bool = Field(default=False, description="Enable W&B integration")
    wandb_project: Optional[str] = Field(default=None, description="W&B project name")


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""
    run_id: Optional[str] = Field(default=None, description="Unique run identifier")
    max_iterations: int = Field(default=10, description="Max training iterations")
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    dry_run: bool = Field(default=False, description="Dry-run mode (no actual training)")

    def dict_for_snapshot(self) -> Dict[str, Any]:
        """Export as serializable dict for snapshot."""
        return self.model_dump()


class StateTransition(BaseModel):
    """Record a state transition with metadata."""
    state: PromotionState
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    iteration: Optional[int] = None
    metrics: Optional[Dict[str, Any]] = None
    verdict: Optional[str] = None  # "PASS", "FAIL", etc.


class RunManifest(BaseModel):
    """Complete run artifact manifest."""
    run_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    state: PromotionState = PromotionState.TRAINING
    state_history: list[StateTransition] = Field(default_factory=list)
    config_snapshot: Dict[str, Any] = Field(default_factory=dict)
    git_commit: Optional[str] = None
    environment_lock: Optional[Dict[str, str]] = None
    best_model_path: Optional[Path] = None
    latest_metrics: Dict[str, float] = Field(default_factory=dict)

    def save_json(self, path: Path) -> None:
        """Save manifest as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2, default=str))

    @classmethod
    def load_json(cls, path: Path) -> "RunManifest":
        """Load manifest from JSON."""
        import json
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
