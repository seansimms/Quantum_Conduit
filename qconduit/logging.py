"""Logging utilities for Quantum Conduit.

Provides structured logging with proper configuration and context management.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

# Default logging level
_DEFAULT_LEVEL = logging.WARNING

# Module-level logger cache
_loggers: dict[str, logging.Logger] = {}


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get or create a logger for the given module name.
    
    Loggers are cached to avoid duplicate handlers. The logger name should
    typically be `__name__` from the calling module.
    
    Args:
        name: Logger name (typically `__name__`). If None, returns root logger.
    
    Returns:
        Configured logger instance.
    
    Example:
        >>> from qconduit.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing quantum state")
    """
    if name is None:
        name = "qconduit"
    
    # Use full module path for logger name
    logger_name = f"qconduit.{name}" if not name.startswith("qconduit.") else name
    
    # Return cached logger if exists
    if logger_name in _loggers:
        return _loggers[logger_name]
    
    # Create new logger
    logger = logging.getLogger(logger_name)
    
    # Only configure if not already configured (avoid duplicate handlers)
    if not logger.handlers:
        logger.setLevel(_DEFAULT_LEVEL)
        
        # Create console handler with formatter
        handler = logging.StreamHandler(sys.stderr)  # Use stderr for logs
        handler.setLevel(_DEFAULT_LEVEL)
        
        # Format: [LEVEL] module: message
        formatter = logging.Formatter(
            '[%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.propagate = False  # Don't propagate to root logger
    
    # Cache and return
    _loggers[logger_name] = logger
    return logger


def set_log_level(level: int | str) -> None:
    """Set the logging level for all Quantum Conduit loggers.
    
    Args:
        level: Logging level (logging.DEBUG, logging.INFO, etc.) or string
            ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
    
    Example:
        >>> from qconduit.logging import set_log_level
        >>> import logging
        >>> set_log_level(logging.DEBUG)
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.WARNING)
    
    # Update all cached loggers
    for logger in _loggers.values():
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
    
    # Update default for new loggers
    global _DEFAULT_LEVEL
    _DEFAULT_LEVEL = level


def configure_logging(
    level: int | str = logging.WARNING,
    format_string: Optional[str] = None,
    stream: Optional[object] = None,
) -> None:
    """Configure logging for Quantum Conduit.
    
    This function allows fine-grained control over logging configuration.
    It should typically be called once at application startup.
    
    Args:
        level: Logging level (default: WARNING).
        format_string: Custom format string. If None, uses default.
        stream: Output stream (default: sys.stderr).
    
    Example:
        >>> from qconduit.logging import configure_logging
        >>> import logging
        >>> configure_logging(level=logging.INFO)
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.WARNING)
    
    if stream is None:
        stream = sys.stderr
    
    if format_string is None:
        format_string = '[%(levelname)s] %(name)s: %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Update all existing loggers
    for logger in _loggers.values():
        logger.setLevel(level)
        # Remove old handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        # Add new handler
        handler = logging.StreamHandler(stream)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Update default
    global _DEFAULT_LEVEL
    _DEFAULT_LEVEL = level

