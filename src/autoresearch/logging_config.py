"""Structured logging using structlog."""

import structlog
from pathlib import Path
from typing import Optional


def configure_logging(log_dir: Path, run_id: str, level: str = "INFO") -> None:
    """Configure structured logging for a run."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # File logging
    log_file = log_dir / f"{run_id}.log"
    with open(log_file, "w") as f:
        f.write("")


def get_logger(name: str):
    """Get a logger instance."""
    return structlog.get_logger(name)
