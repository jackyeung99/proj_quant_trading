from __future__ import annotations

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
) -> None:
    """
    Configure global logging for the application.

    Call this ONCE at program entry (CLI, script, Dash app).

    Parameters
    ----------
    level : int
        Logging level (logging.INFO, logging.DEBUG, etc.)
    stream : object, optional
        Output stream (defaults to sys.stdout)
    fmt : str
        Log message format
    datefmt : str
        Date/time format
    force : bool
        If True, reconfigure logging even if handlers exist
        (useful in notebooks or Dash reloads)
    """
    if stream is None:
        stream = sys.stdout

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=[logging.StreamHandler(stream)],
        force=force,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a module-scoped logger.

    Always use this instead of logging.getLogger(__name__)
    directly, so behavior is consistent across the project.
    """
    return logging.getLogger(name)
