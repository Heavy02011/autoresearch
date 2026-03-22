"""Artifact management - filesystem organization and metadata."""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import subprocess

from .config import RunManifest, ExperimentConfig


class ArtifactManager:
    """
    Manages artifact storage in hierarchical filesystem.
    
    Structure:
    runs/
      run_20260322_123456/
        ├── state.json
        ├── config.snapshot.json
        ├── metadata.json
        ├── environment.lock
        ├── models/
        │   ├── checkpoint_iter_1.pt
        │   └── best_model.pt
        ├── best_model.onnx
        ├── metrics/
        │   └── evaluation_results.json
        └── logs/
            ├── train.log
            ├── eval.log
            └── promotion_decision.log
    """

    def __init__(self, run_id: str, runs_dir: Path):
        self.run_id = run_id
        self.run_dir = Path(runs_dir) / run_id
        
        # Create subdirectories
        self.models_dir = self.run_dir / "models"
        self.metrics_dir = self.run_dir / "metrics"
        self.logs_dir = self.run_dir / "logs"
        
        for d in [self.run_dir, self.models_dir, self.metrics_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def state_file(self) -> Path:
        return self.run_dir / "state.json"

    @property
    def config_snapshot_file(self) -> Path:
        return self.run_dir / "config.snapshot.json"

    @property
    def metadata_file(self) -> Path:
        return self.run_dir / "metadata.json"

    @property
    def environment_lock_file(self) -> Path:
        return self.run_dir / "environment.lock"

    def checkpoint_path(self, iteration: int) -> Path:
        """Path for checkpoint at iteration."""
        return self.models_dir / f"checkpoint_iter_{iteration}.pt"

    def best_model_path(self) -> Path:
        """Path for best model checkpoint."""
        return self.models_dir / "best_model.pt"

    def best_model_onnx_path(self) -> Path:
        """Path for exported ONNX model."""
        return self.run_dir / "best_model.onnx"

    def metrics_file(self, iteration: int) -> Path:
        """Path for evaluation metrics at iteration."""
        return self.metrics_dir / f"eval_iter_{iteration}.json"

    def save_config_snapshot(self, config: ExperimentConfig) -> None:
        """Save configuration snapshot."""
        with open(self.config_snapshot_file, "w") as f:
            json.dump(config.dict_for_snapshot(), f, indent=2, default=str)

    def save_metrics(self, iteration: int, metrics: Dict[str, Any]) -> None:
        """Save evaluation metrics."""
        with open(self.metrics_file(iteration), "w") as f:
            data = {
                "iteration": iteration,
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": metrics,
            }
            json.dump(data, f, indent=2, default=str)

    def load_metrics(self, iteration: int) -> Optional[Dict[str, Any]]:
        """Load metrics from iteration."""
        path = self.metrics_file(iteration)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return data.get("metrics", {})
        return None

    def save_metadata(self, git_commit: Optional[str] = None) -> None:
        """Save run metadata (git, versions, etc.)."""
        metadata = {
            "run_id": self.run_id,
            "created_at": datetime.utcnow().isoformat(),
            "git_commit": git_commit or self._get_git_commit(),
            "git_branch": self._get_git_branch(),
        }
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def save_environment_lock(self, env_vars: Dict[str, str]) -> None:
        """Save environment variable snapshot."""
        with open(self.environment_lock_file, "w") as f:
            json.dump(env_vars, f, indent=2)

    def list_checkpoints(self) -> list[tuple[int, Path]]:
        """List all checkpoints, return (iteration, path) tuples."""
        checkpoints = []
        for f in sorted(self.models_dir.glob("checkpoint_iter_*.pt")):
            try:
                iteration = int(f.stem.split("_")[-1])
                checkpoints.append((iteration, f))
            except (ValueError, IndexError):
                pass
        return checkpoints

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get most recent checkpoint."""
        checkpoints = self.list_checkpoints()
        if checkpoints:
            return checkpoints[-1][1]
        return None

    @staticmethod
    def _get_git_commit() -> Optional[str]:
        """Get current git commit hash."""
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return None

    @staticmethod
    def _get_git_branch() -> Optional[str]:
        """Get current git branch."""
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return None
