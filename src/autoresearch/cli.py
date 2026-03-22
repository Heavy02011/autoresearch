"""Command-line interface for autoresearch."""

from pathlib import Path
from datetime import datetime
from typing import Optional
import json

import typer
from omegaconf import OmegaConf

from .config import ExperimentConfig, EnvironmentConfig
from .orchestrate import Orchestrator
from .state import StateTracker
from .logging_config import configure_logging

app = typer.Typer(
    name="autoresearch",
    help="Autonomous DonkeyCar steering optimization framework",
)


@app.command()
def run(
    max_iterations: int = typer.Option(
        10,
        "--max-iterations",
        help="Maximum iterations of train-evaluate-promote loop",
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to Hydra experiment config (YAML)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run without actual training/evaluation",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Unique run identifier (auto-generated if not provided)",
    ),
):
    """
    Execute autonomous training loop.
    
    Full workflow: Train → Evaluate → Promote Gate → Keep/Discard → Export
    """
    typer.echo("Starting autonomous loop...")

    # Generate run ID if not provided
    if not run_id:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}"

    # Load config
    if config:
        cfg_dict = OmegaConf.to_container(OmegaConf.load(config))
    else:
        cfg_dict = {}

    # Build experiment config
    exp_config = ExperimentConfig(
        run_id=run_id,
        max_iterations=max_iterations,
        dry_run=dry_run,
        **cfg_dict,
    )

    # Configure logging
    configure_logging(exp_config.environment.logs_dir, run_id, exp_config.environment.log_level)

    # Run orchestrator
    orchestrator = Orchestrator(exp_config, run_id)
    
    try:
        orchestrator.run_autonomous_loop()
        typer.echo(f"\n✓ Run complete: {run_id}")
        
        # Print final status
        status = orchestrator.status()
        typer.echo("\nFinal Status:")
        typer.echo(json.dumps(status, indent=2, default=str))
        
    except Exception as e:
        typer.echo(f"\n✗ Run failed: {e}", err=True)
        raise


@app.command()
def status(
    run_id: str = typer.Argument(
        ...,
        help="Run ID to check",
    ),
    runs_dir: Optional[str] = typer.Option(
        None,
        "--runs-dir",
        help="Runs directory (default: ./runs)",
    ),
):
    """Check status of a run."""
    runs_path = Path(runs_dir or "runs")
    state_file = runs_path / run_id / "state.json"

    if not state_file.exists():
        typer.echo(f"Run not found: {run_id}", err=True)
        raise typer.Exit(1)

    from .state import RunManifest
    manifest = RunManifest.load_json(state_file)
    
    typer.echo(f"Run: {manifest.run_id}")
    typer.echo(f"State: {manifest.state.value}")
    typer.echo(f"Created: {manifest.created_at}")
    typer.echo(f"\nState History:")
    
    for transition in manifest.state_history:
        typer.echo(
            f"  {transition.timestamp.isoformat()} → {transition.state.value} "
            f"(iter={transition.iteration})"
        )
    
    if manifest.latest_metrics:
        typer.echo(f"\nLatest Metrics:")
        for key, value in manifest.latest_metrics.items():
            typer.echo(f"  {key}: {value}")


@app.command()
def rollback(
    run_id: str = typer.Argument(
        ...,
        help="Run ID to rollback",
    ),
    runs_dir: Optional[str] = typer.Option(
        None,
        "--runs-dir",
        help="Runs directory (default: ./runs)",
    ),
):
    """Rollback run to previous promoted state."""
    runs_path = Path(runs_dir or "runs")
    state_tracker = StateTracker(run_id, runs_path / run_id)
    
    try:
        state_tracker.rollback()
        typer.echo(f"✓ Rolled back: {run_id}")
    except Exception as e:
        typer.echo(f"✗ Rollback failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def export(
    run_id: str = typer.Argument(
        ...,
        help="Run ID to export",
    ),
    runs_dir: Optional[str] = typer.Option(
        None,
        "--runs-dir",
        help="Runs directory (default: ./runs)",
    ),
    format: str = typer.Option(
        "onnx",
        "--format",
        help="Export format (onnx, pytorch)",
    ),
):
    """Export model from a run."""
    typer.echo(f"Exporting {run_id} as {format}...")
    # Integration point with export.py
    typer.echo("Export not yet implemented")


@app.command()
def prepare():
    """One-time preparation (data, simulator, etc.)."""
    typer.echo("Prepare not yet implemented")


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
