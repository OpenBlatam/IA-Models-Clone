"""
Centralized Logging Configuration
Structured logging setup with multiple handlers and formatters
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from core.config import settings

# Create logs directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup structured logging with multiple handlers
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        
    Returns:
        Configured logger instance
    """
    level = log_level or settings.log_level.upper()
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level, logging.INFO))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with colored output (if available)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level, logging.INFO))
    
    # Formatter
    formatter = logging.Formatter(
        settings.log_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler for general logs
    if log_file is None:
        log_file = LOG_DIR / "app.log"
    
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Error file handler (errors only)
    error_file_handler = logging.FileHandler(
        LOG_DIR / "errors.log",
        encoding="utf-8"
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(formatter)
    logger.addHandler(error_file_handler)
    
    # Set specific log levels for noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance for a specific module
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)