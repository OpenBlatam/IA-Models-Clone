"""
Logging Configuration - Centralized logging setup.

Provides:
- Standardized logging configuration
- Formatters and handlers
- Log levels and filters
- Context managers for logging
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

# ════════════════════════════════════════════════════════════════════════════════
# LOG LEVELS
# ════════════════════════════════════════════════════════════════════════════════

class LogLevel(str, Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

# ════════════════════════════════════════════════════════════════════════════════
# LOG FORMATS
# ════════════════════════════════════════════════════════════════════════════════

# Standard format
STANDARD_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Detailed format
DETAILED_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - "
    "%(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
)

# Simple format
SIMPLE_FORMAT = "%(levelname)s - %(message)s"

# JSON format (for structured logging)
JSON_FORMAT = {
    "timestamp": "%(asctime)s",
    "name": "%(name)s",
    "level": "%(levelname)s",
    "message": "%(message)s",
    "module": "%(module)s",
    "function": "%(funcName)s",
    "line": "%(lineno)d",
}

# ════════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ════════════════════════════════════════════════════════════════════════════════

def setup_logging(
    level: str = "INFO",
    format_str: str = STANDARD_FORMAT,
    log_file: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    console: bool = True,
    file_logging: bool = True,
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_str: Log format string
        log_file: Path to log file
        log_dir: Directory for log files
        console: Whether to log to console
        file_logging: Whether to log to file
    
    Returns:
        Configured logger
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_logging and (log_file or log_dir):
        if log_file is None:
            # Create log file in log_dir with timestamp
            if log_dir is None:
                log_dir = Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"benchmark_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with optional level override.
    
    Args:
        name: Logger name (usually __name__)
        level: Optional log level override
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    if level:
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
    
    return logger

def configure_module_logging(
    module_name: str,
    level: str = "INFO",
    propagate: bool = True,
) -> logging.Logger:
    """
    Configure logging for a specific module.
    
    Args:
        module_name: Module name
        level: Log level
        propagate: Whether to propagate to parent logger
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(module_name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logger.propagate = propagate
    return logger

# ════════════════════════════════════════════════════════════════════════════════
# CONTEXT MANAGERS
# ════════════════════════════════════════════════════════════════════════════════

class LoggingContext:
    """Context manager for temporary logging configuration."""
    
    def __init__(
        self,
        level: str = "DEBUG",
        format_str: str = DETAILED_FORMAT,
        log_file: Optional[Path] = None,
    ):
        self.level = level
        self.format_str = format_str
        self.log_file = log_file
        self.original_level = None
        self.original_handlers = None
    
    def __enter__(self):
        # Save original configuration
        root_logger = logging.getLogger()
        self.original_level = root_logger.level
        self.original_handlers = root_logger.handlers.copy()
        
        # Setup new configuration
        setup_logging(
            level=self.level,
            format_str=self.format_str,
            log_file=self.log_file,
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(self.original_level)
        root_logger.handlers.clear()
        root_logger.handlers.extend(self.original_handlers)

# ════════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def log_function_call(logger: logging.Logger, func_name: str, args: tuple, kwargs: dict):
    """Log a function call with arguments."""
    logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")

def log_performance(logger: logging.Logger, operation: str, duration_ms: float):
    """Log performance metrics."""
    logger.info(f"{operation} took {duration_ms:.2f}ms")

def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: Dict[str, Any],
):
    """Log an error with additional context."""
    context_str = ", ".join(f"{k}={v}" for k, v in context.items())
    logger.error(f"{type(error).__name__}: {error} | Context: {context_str}", exc_info=True)

__all__ = [
    "LogLevel",
    "STANDARD_FORMAT",
    "DETAILED_FORMAT",
    "SIMPLE_FORMAT",
    "JSON_FORMAT",
    "setup_logging",
    "get_logger",
    "configure_module_logging",
    "LoggingContext",
    "log_function_call",
    "log_performance",
    "log_error_with_context",
]












