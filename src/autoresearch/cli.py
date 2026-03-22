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
    output_format: str = typer.Option(
        "text",
        "--format",
        help="Output format: text or json",
    ),
    last: Optional[int] = typer.Option(
        None,
        "--last",
        help="Show only the last N state transitions",
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

    history = manifest.state_history
    if last is not None:
        history = history[-last:]

    if output_format == "json":
        out = {
            "run_id": manifest.run_id,
            "state": manifest.state.value,
            "created_at": manifest.created_at.isoformat(),
            "latest_metrics": manifest.latest_metrics,
            "state_history": [
                {
                    "timestamp": t.timestamp.isoformat(),
                    "state": t.state.value,
                    "iteration": t.iteration,
                    "verdict": t.verdict,
                }
                for t in history
            ],
        }
        typer.echo(json.dumps(out, indent=2, default=str))
        return

    typer.echo(f"Run: {manifest.run_id}")
    typer.echo(f"State: {manifest.state.value}")
    typer.echo(f"Created: {manifest.created_at}")
    typer.echo("\nState History:")

    for transition in history:
        typer.echo(
            f"  {transition.timestamp.isoformat()} → {transition.state.value} "
            f"(iter={transition.iteration})"
        )

    if manifest.latest_metrics:
        typer.echo("\nLatest Metrics:")
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


@app.command()
def compare(
    run_ids: list[str] = typer.Argument(
        ...,
        help="Run IDs to compare (space-separated)",
    ),
    runs_dir: Optional[str] = typer.Option(
        None,
        "--runs-dir",
        help="Runs directory (default: ./runs)",
    ),
    csv: bool = typer.Option(
        False,
        "--csv",
        help="Output as CSV instead of table",
    ),
):
    """Compare metrics across multiple runs side-by-side."""
    import csv as _csv
    import io
    from .state import RunManifest

    runs_path = Path(runs_dir or "runs")
    rows = []

    for rid in run_ids:
        state_file = runs_path / rid / "state.json"
        if not state_file.exists():
            typer.echo(f"[WARN] Run not found: {rid} — skipping", err=True)
            continue
        manifest = RunManifest.load_json(state_file)

        # Find best lap_time across all promoted transitions
        best_lap = None
        promoted_iters = [
            t for t in manifest.state_history if t.state.value == "promoted"
        ]
        if manifest.latest_metrics.get("lap_time") is not None:
            best_lap = manifest.latest_metrics["lap_time"]

        rows.append({
            "run_id": rid,
            "state": manifest.state.value,
            "promotions": len(promoted_iters),
            "best_lap_time": best_lap,
            **{k: v for k, v in manifest.latest_metrics.items()},
        })

    if not rows:
        typer.echo("No runs found.", err=True)
        raise typer.Exit(1)

    if csv:
        buf = io.StringIO()
        fieldnames = list(rows[0].keys())
        writer = _csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        typer.echo(buf.getvalue())
        return

    # Text table
    col_widths = {k: max(len(str(k)), max(len(str(r.get(k, ""))) for r in rows)) for k in rows[0]}
    header = "  ".join(str(k).ljust(col_widths[k]) for k in rows[0])
    separator = "  ".join("-" * col_widths[k] for k in rows[0])
    typer.echo(header)
    typer.echo(separator)
    for row in rows:
        typer.echo("  ".join(str(row.get(k, "")).ljust(col_widths[k]) for k in rows[0]))


@app.command()
def preflight(
    sim_path: Optional[str] = typer.Option(
        None,
        "--sim-path",
        help="Path to simulator binary (overrides DONKEY_SIM_PATH env var)",
    ),
):
    """Validate environment readiness before running the pipeline."""
    from pathlib import Path as _Path
    from .preflight import (
        PreflightResult,
        check_python_version,
        check_required_packages,
        check_optional_packages,
        check_cuda,
        check_simulator_path,
        check_config_files,
        check_git_repo,
        check_disk_space,
    )

    typer.echo("=" * 60)
    typer.echo("autoresearch — preflight validation")
    typer.echo("=" * 60)

    result = PreflightResult()
    sim = _Path(sim_path) if sim_path else None

    check_python_version(result)
    check_required_packages(result)
    check_optional_packages(result)
    check_cuda(result)
    check_simulator_path(result, sim)
    check_config_files(result)
    check_git_repo(result)
    check_disk_space(result)

    result.print_report()

    if result.passed:
        typer.echo("\n✓ Preflight PASSED — ready to run autoresearch")
    else:
        typer.echo(f"\n✗ Preflight FAILED — {len(result.errors)} error(s) must be resolved", err=True)
        raise typer.Exit(1)


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
