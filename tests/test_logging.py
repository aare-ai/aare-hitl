"""Tests for logging configuration."""

import logging
import pytest

from aare.utils.logging import setup_logging, get_logger, LoggingContext


class TestSetupLogging:
    """Tests for logging setup."""

    def test_setup_default_logging(self):
        """Setup logging with defaults."""
        setup_logging()
        logger = logging.getLogger("test_default")
        assert logger.level == logging.NOTSET  # Inherits from root

    def test_setup_calls_without_error(self):
        """Setup logging completes without error."""
        # Just verify it doesn't raise
        setup_logging(level="DEBUG")
        setup_logging(level="INFO")
        setup_logging(level="WARNING")

    def test_setup_json_format(self):
        """Setup logging with JSON format."""
        # Just verify it doesn't raise
        setup_logging(level="INFO", json_format=True)

    def test_setup_custom_format(self):
        """Setup logging with custom format string."""
        setup_logging(level="INFO", format_string="%(message)s")

    def test_third_party_loggers_reduced(self):
        """Third-party loggers are set to WARNING."""
        setup_logging(level="DEBUG")

        transformers_logger = logging.getLogger("transformers")
        datasets_logger = logging.getLogger("datasets")
        httpx_logger = logging.getLogger("httpx")

        assert transformers_logger.level == logging.WARNING
        assert datasets_logger.level == logging.WARNING
        assert httpx_logger.level == logging.WARNING


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_logger(self):
        """get_logger returns a Logger instance."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_get_logger_same_instance(self):
        """get_logger returns same instance for same name."""
        logger1 = get_logger("test.same")
        logger2 = get_logger("test.same")
        assert logger1 is logger2

    def test_get_logger_different_names(self):
        """get_logger returns different instances for different names."""
        logger1 = get_logger("test.one")
        logger2 = get_logger("test.two")
        assert logger1 is not logger2


class TestLoggingContext:
    """Tests for LoggingContext context manager."""

    def test_context_changes_level(self):
        """Context manager temporarily changes log level."""
        logger = get_logger("test.context.level")
        logger.setLevel(logging.INFO)

        with LoggingContext(logger, level=logging.DEBUG) as ctx_logger:
            assert ctx_logger.level == logging.DEBUG

        assert logger.level == logging.INFO

    def test_context_restores_level(self):
        """Level is restored after context exits."""
        logger = get_logger("test.context.restore")
        original_level = logging.WARNING
        logger.setLevel(original_level)

        with LoggingContext(logger, level=logging.ERROR):
            pass

        assert logger.level == original_level

    def test_context_without_level(self):
        """Context without level change doesn't modify logger."""
        logger = get_logger("test.context.nolevel")
        logger.setLevel(logging.INFO)

        with LoggingContext(logger) as ctx_logger:
            assert ctx_logger.level == logging.INFO

        assert logger.level == logging.INFO

    def test_context_with_exception(self):
        """Level is restored even when exception occurs."""
        logger = get_logger("test.context.exception")
        logger.setLevel(logging.INFO)

        try:
            with LoggingContext(logger, level=logging.DEBUG):
                raise ValueError("Test error")
        except ValueError:
            pass

        assert logger.level == logging.INFO

    def test_context_returns_logger(self):
        """Context manager returns the logger."""
        logger = get_logger("test.context.return")

        with LoggingContext(logger) as ctx_logger:
            assert ctx_logger is logger
