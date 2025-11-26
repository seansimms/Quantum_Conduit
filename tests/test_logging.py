"""Tests for logging utilities."""

import logging
import sys
from io import StringIO

import pytest

from qconduit.logging import (
    configure_logging,
    get_logger,
    set_log_level,
)


def test_get_logger_returns_logger():
    """Test that get_logger returns a logger instance."""
    logger = get_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name.startswith("qconduit.")


def test_get_logger_caching():
    """Test that get_logger caches loggers."""
    logger1 = get_logger("test_module")
    logger2 = get_logger("test_module")
    assert logger1 is logger2


def test_get_logger_different_modules():
    """Test that different modules get different loggers."""
    logger1 = get_logger("module1")
    logger2 = get_logger("module2")
    assert logger1 is not logger2
    assert logger1.name != logger2.name


def test_logger_output():
    """Test that logger outputs messages correctly."""
    import sys
    from io import StringIO
    
    # Capture stderr
    old_stderr = sys.stderr
    sys.stderr = captured = StringIO()
    
    try:
        # Configure logging to INFO level
        configure_logging(level=logging.INFO, stream=captured)
        logger = get_logger("test_module")
        logger.info("Test message")
        
        output = captured.getvalue()
        assert "Test message" in output
        assert "test_module" in output or "qconduit" in output
    finally:
        sys.stderr = old_stderr
        # Reset to default
        configure_logging(level=logging.WARNING)


def test_set_log_level():
    """Test that set_log_level updates logger levels."""
    logger = get_logger("test_module")
    
    # Set to INFO
    set_log_level(logging.INFO)
    # Logger level should be updated (may need to check handler level too)
    assert logger.level <= logging.INFO
    
    # Set back to WARNING
    set_log_level(logging.WARNING)
    assert logger.level <= logging.WARNING


def test_set_log_level_string():
    """Test that set_log_level accepts string levels."""
    logger = get_logger("test_module")
    
    set_log_level("DEBUG")
    assert logger.level == logging.DEBUG
    
    set_log_level("ERROR")
    assert logger.level == logging.ERROR


def test_configure_logging():
    """Test configure_logging function."""
    stream = StringIO()
    configure_logging(level=logging.DEBUG, stream=stream)
    
    logger = get_logger("test_module")
    logger.debug("Debug message")
    
    output = stream.getvalue()
    assert "Debug message" in output


def test_logger_does_not_propagate():
    """Test that loggers don't propagate to root logger."""
    logger = get_logger("test_module")
    assert logger.propagate is False


def test_multiple_loggers_independent():
    """Test that multiple loggers work independently."""
    logger1 = get_logger("module1")
    logger2 = get_logger("module2")
    
    # Set level for one shouldn't affect the other (after initial setup)
    logger1.setLevel(logging.DEBUG)
    logger2.setLevel(logging.ERROR)
    
    assert logger1.level == logging.DEBUG
    assert logger2.level == logging.ERROR

