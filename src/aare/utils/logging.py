"""Logging configuration for Aare Model Trainer."""

import logging
import sys
from typing import Any


def setup_logging(
    level: str = "INFO",
    format_string: str | None = None,
    json_format: bool = False,
) -> None:
    """Set up logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format_string: Custom format string.
        json_format: Use JSON format for structured logging.
    """
    if format_string is None:
        if json_format:
            format_string = '{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
        else:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Reduce noise from third-party libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (usually __name__).

    Returns:
        Configured logger.
    """
    return logging.getLogger(name)


class LoggingContext:
    """Context manager for temporary logging configuration."""

    def __init__(
        self,
        logger: logging.Logger,
        level: int | None = None,
        extra: dict[str, Any] | None = None,
    ):
        """Initialize logging context.

        Args:
            logger: Logger to modify.
            level: Temporary log level.
            extra: Extra context to include.
        """
        self.logger = logger
        self.level = level
        self.extra = extra or {}
        self._original_level: int | None = None

    def __enter__(self) -> logging.Logger:
        """Enter context."""
        if self.level is not None:
            self._original_level = self.logger.level
            self.logger.setLevel(self.level)
        return self.logger

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context."""
        if self._original_level is not None:
            self.logger.setLevel(self._original_level)
