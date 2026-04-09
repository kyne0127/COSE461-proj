"""
module.utils.logging
====================
Structured logging setup shared across desktop and server.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    component: str = "lerobot",
) -> logging.Logger:
    """
    Configure root logger with a consistent format.

    Args:
        level:     Log level string ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file:  Optional path to write logs to file as well
        component: Logger name prefix

    Returns:
        Configured logger instance
    """
    fmt = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )

    logger = logging.getLogger(component)
    logger.info("Logging initialized — level=%s component=%s", level, component)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the lerobot namespace."""
    return logging.getLogger(f"lerobot.{name}")
