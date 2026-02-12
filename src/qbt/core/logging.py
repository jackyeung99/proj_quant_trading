from __future__ import annotations

from pathlib import Path
import logging
import sys
from typing import Optional


_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)

_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    *,
    stream: Optional[object] = None,
    fmt: str = _LOG_FORMAT,
    datefmt: str = _DATE_FORMAT,
    force: bool = False,
    log_file: Optional[str | Path] = None,   
) -> None:
    """
    Configure global logging for the application.

    Adds optional file logging.
    """
    if stream is None:
        stream = sys.stdout

    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(stream)
    console_handler.setFormatter(logging.Formatter(fmt, datefmt))
    handlers.append(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(fmt, datefmt))
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=force,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a module-scoped logger.

    Always use this instead of logging.getLogger(__name__)
    directly, so behavior is consistent across the project.
    """
    return logging.getLogger(name)
