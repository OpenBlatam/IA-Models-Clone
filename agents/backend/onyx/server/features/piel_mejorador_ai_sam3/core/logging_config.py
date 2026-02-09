"""
Logging Configuration for Piel Mejorador AI SAM3
================================================

Advanced logging setup with structured logging and performance tracking.
"""

import logging
import sys
from typing import Optional
from pathlib import Path
from datetime import datetime


class PerformanceFilter(logging.Filter):
    """Filter for performance-related log messages."""
    
    def filter(self, record):
        """Add performance context to log records."""
        if not hasattr(record, "performance"):
            record.performance = False
        return True


class StructuredFormatter(logging.Formatter):
    """Structured formatter for better log parsing."""
    
    def format(self, record):
        """Format log record with structured data."""
        # Base format
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields
        if hasattr(record, "task_id"):
            log_data["task_id"] = record.task_id
        if hasattr(record, "file_path"):
            log_data["file_path"] = record.file_path
        if hasattr(record, "performance"):
            log_data["performance"] = record.performance
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Format as JSON-like string
        import json
        return json.dumps(log_data, ensure_ascii=False)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    structured: bool = False,
    performance_tracking: bool = True
) -> logging.Logger:
    """
    Setup advanced logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for file logging
        structured: Use structured JSON logging
        performance_tracking: Enable performance tracking
        
    Returns:
        Configured logger
    """
    # Get root logger
    logger = logging.getLogger("piel_mejorador_ai_sam3")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(formatter)
    
    if performance_tracking:
        console_handler.addFilter(PerformanceFilter())
    
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        if performance_tracking:
            file_handler.addFilter(PerformanceFilter())
        
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"piel_mejorador_ai_sam3.{name}")




