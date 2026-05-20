"""
Logging configuration using Loguru

This module configures structured logging using Loguru with:
- Console output with colors for development
- File rotation for application logs
- Separate error logs with backtraces
- Configurable log levels from settings
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

try:
    from config.settings import settings
    LOG_LEVEL = getattr(settings, 'LOG_LEVEL', 'INFO')
except ImportError:
    LOG_LEVEL = 'INFO'


def setup_logging(
    log_level: Optional[str] = None,
    log_dir: Optional[Path] = None,
    enable_file_logging: bool = True
) -> None:
    """
    Configure Loguru logger with console and file handlers
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                   If None, uses LOG_LEVEL from settings or defaults to INFO
        log_dir: Directory for log files. If None, uses "logs" directory
        enable_file_logging: Whether to enable file logging (default: True)
    
    Returns:
        None (configures global logger)
    """
    # Remove default handler
    logger.remove()
    
    # Use provided log level or fallback to settings/default
    level = log_level or LOG_LEVEL
    
    # Add console handler with color
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    
    if not enable_file_logging:
        return
    
    # Set up log directory
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Add file handler for all logs
    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        rotation="00:00",  # Rotate at midnight
        retention="30 days",
        level="INFO",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} - {message}"
        ),
        compression="zip",
        backtrace=True,
        diagnose=True,
    )
    
    # Add file handler for errors only
    logger.add(
        log_dir / "error_{time:YYYY-MM-DD}.log",
        rotation="00:00",  # Rotate at midnight
        retention="90 days",  # Keep errors longer
        level="ERROR",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} - {message}"
        ),
        compression="zip",
        backtrace=True,
        diagnose=True,
    )
    
    logger.info(f"Logging configured with level: {level}")


def get_logger(name: Optional[str] = None):
    """
    Get a logger instance for a specific module
    
    Args:
        name: Module name (usually __name__)
        
    Returns:
        Logger instance configured for the module
    """
    if name:
        return logger.bind(name=name)
    return logger


# Initialize logger on module import
setup_logging()

