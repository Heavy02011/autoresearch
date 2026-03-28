"""Preflight validation — checks environment readiness before running autoresearch."""

from __future__ import annotations

import sys
import importlib
import subprocess
from pathlib import Path
from typing import Optional


_REQUIRED_PACKAGES = [
    "torch",
    "pydantic",
    "typer",
    "structlog",
    "omegaconf",
]

_OPTIONAL_PACKAGES = {
    "gym": "gym-donkeycar (required for simulator evaluation)",
    "gym_donkeycar": "gym-donkeycar (required for simulator evaluation)",
    "onnx": "onnx (required for ONNX export)",
    "wandb": "wandb (optional experiment tracking)",
}


class PreflightResult:
    """Aggregates preflight check results."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def ok(self, msg: str) -> None:
        self.info.append(msg)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def print_report(self) -> None:
        for msg in self.info:
            print(f"  ✓ {msg}")
        for msg in self.warnings:
            print(f"  ⚠ {msg}")
        for msg in self.errors:
            print(f"  ✗ {msg}", file=sys.stderr)


def check_python_version(result: PreflightResult) -> None:
    """Python 3.10+ required."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        result.error(f"Python 3.10+ required, found {major}.{minor}")
    else:
        result.ok(f"Python {major}.{minor} ✓")


def check_required_packages(result: PreflightResult) -> None:
    """All mandatory packages must be importable."""
    for pkg in _REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
            result.ok(f"{pkg} importable")
        except ImportError:
            result.error(f"Required package not found: {pkg}. Run: uv sync")


def check_optional_packages(result: PreflightResult) -> None:
    """Optional packages checked with informative warnings."""
    for pkg, description in _OPTIONAL_PACKAGES.items():
        try:
            importlib.import_module(pkg)
            result.ok(f"{pkg} available")
        except ImportError:
            result.warn(f"Optional package missing: {description}")


def check_cuda(result: PreflightResult) -> None:
    """GPU/CUDA availability check."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            result.ok(f"CUDA available ({device_name})")
        else:
            result.warn("CUDA not available — training will run on CPU (very slow for production runs)")
    except ImportError:
        result.warn("torch not installed — cannot check CUDA")


def check_simulator_path(result: PreflightResult, sim_path: Optional[Path] = None) -> None:
    """Simulator executable check (optional if path provided)."""
    if sim_path is None:
        import os
        sim_path_str = os.environ.get("DONKEY_SIM_PATH")
        if sim_path_str:
            sim_path = Path(sim_path_str)

    if sim_path is None:
        result.warn("DONKEY_SIM_PATH not set — simulator evaluation will be skipped (dry-run only)")
        return

    if not sim_path.exists():
        result.error(f"Simulator not found at: {sim_path}")
    else:
        result.ok(f"Simulator found at {sim_path}")


def check_config_files(result: PreflightResult, project_root: Optional[Path] = None) -> None:
    """Verify configs directory and YAML files exist."""
    if project_root is None:
        project_root = Path.cwd()

    configs_dir = project_root / "configs"
    if not configs_dir.is_dir():
        result.warn(f"configs/ directory not found at {configs_dir} — default config will be used")
        return

    expected_configs = ["experiment.yaml", "evaluation.yaml", "promotion.yaml", "environment.yaml"]
    for cfg_file in expected_configs:
        cfg_path = configs_dir / cfg_file
        if cfg_path.exists():
            result.ok(f"Config found: configs/{cfg_file}")
        else:
            result.warn(f"Config not found: configs/{cfg_file} — default values will apply")


def check_git_repo(result: PreflightResult) -> None:
    """Check if running inside a git repo (for reproducibility metadata)."""
    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            check=True,
        )
        result.ok("Git repository detected")
    except (subprocess.CalledProcessError, FileNotFoundError):
        result.warn("Not inside a git repository — run metadata will not include git commit hash")


def check_disk_space(result: PreflightResult, min_gb: float = 5.0) -> None:
    """Warn if less than min_gb of free space."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(Path.cwd())
        free_gb = free / (1024 ** 3)
        if free_gb < min_gb:
            result.warn(f"Low disk space: {free_gb:.1f} GB free (recommend >{min_gb} GB for checkpoints)")
        else:
            result.ok(f"Disk space: {free_gb:.1f} GB free")
    except Exception:
        result.warn("Could not check disk space")


def run_preflight(
    project_root: Optional[Path] = None,
    simulator_path: Optional[Path] = None,
    verbose: bool = True,
) -> PreflightResult:
    """
    Run all preflight checks and return results.

    Args:
        project_root: Root of the autoresearch repository
        simulator_path: Path to simulator binary (overrides DONKEY_SIM_PATH env var)
        verbose: Print results while running

    Returns:
        PreflightResult with .passed bool and .errors/.warnings lists
    """
    result = PreflightResult()

    checks = [
        ("Python version", lambda: check_python_version(result)),
        ("Required packages", lambda: check_required_packages(result)),
        ("Optional packages", lambda: check_optional_packages(result)),
        ("CUDA / GPU", lambda: check_cuda(result)),
        ("Simulator path", lambda: check_simulator_path(result, simulator_path)),
        ("Config files", lambda: check_config_files(result, project_root)),
        ("Git repo", lambda: check_git_repo(result)),
        ("Disk space", lambda: check_disk_space(result)),
    ]

    for check_name, check_fn in checks:
        if verbose:
            print(f"\n[{check_name}]")
        check_fn()
        if verbose:
            # Print only results added in this check
            result.print_report()
            result.info.clear()
            result.warnings.clear()
            result.errors.clear()

    return result


def main() -> None:
    """CLI entry point for preflight validation."""
    import argparse

    parser = argparse.ArgumentParser(description="autoresearch preflight validation")
    parser.add_argument("--sim-path", type=Path, default=None, help="Path to simulator binary")
    parser.add_argument("--project-root", type=Path, default=None, help="Project root directory")
    args = parser.parse_args()

    print("=" * 60)
    print("autoresearch — preflight validation")
    print("=" * 60)

    result = PreflightResult()

    check_python_version(result)
    check_required_packages(result)
    check_optional_packages(result)
    check_cuda(result)
    check_simulator_path(result, args.sim_path)
    check_config_files(result, args.project_root)
    check_git_repo(result)
    check_disk_space(result)

    print("\n" + "=" * 60)
    result.print_report()

    if result.passed:
        print("\n✓ Preflight PASSED — ready to run autoresearch")
        sys.exit(0)
    else:
        print(f"\n✗ Preflight FAILED — {len(result.errors)} error(s) must be resolved")
        sys.exit(1)


if __name__ == "__main__":
    main()
