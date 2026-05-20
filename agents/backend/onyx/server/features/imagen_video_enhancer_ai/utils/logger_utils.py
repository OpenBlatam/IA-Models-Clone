"""
Logger Utilities
================

Advanced logging utilities.
"""

import logging
import sys
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import json


class StructuredLogger:
    """Logger with structured output support."""
    
    def __init__(self, name: str, level: int = logging.INFO):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
    
    def _log_structured(
        self,
        level: int,
        message: str,
        **context
    ):
        """Log with structured context."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "level": logging.getLevelName(level),
            **context
        }
        
        self.logger.log(level, json.dumps(log_data))
    
    def info(self, message: str, **context):
        """Log info with context."""
        self._log_structured(logging.INFO, message, **context)
    
    def error(self, message: str, **context):
        """Log error with context."""
        self._log_structured(logging.ERROR, message, **context)
    
    def warning(self, message: str, **context):
        """Log warning with context."""
        self._log_structured(logging.WARNING, message, **context)
    
    def debug(self, message: str, **context):
        """Log debug with context."""
        self._log_structured(logging.DEBUG, message, **context)


class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize performance logger.
        
        Args:
            logger: Base logger
        """
        self.logger = logger
    
    def log_operation(
        self,
        operation: str,
        duration: float,
        **metrics
    ):
        """
        Log operation performance.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            **metrics: Additional metrics
        """
        self.logger.info(
            f"Performance: {operation}",
            extra={
                "operation": operation,
                "duration_seconds": duration,
                **metrics
            }
        )


def setup_structured_logging(
    name: str,
    level: str = "INFO",
    output_file: Optional[Path] = None,
    json_format: bool = False
) -> StructuredLogger:
    """
    Setup structured logging.
    
    Args:
        name: Logger name
        level: Logging level
        output_file: Optional output file
        json_format: Whether to use JSON format
        
    Returns:
        StructuredLogger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if json_format:
        console_handler.setFormatter(logging.Formatter('%(message)s'))
    else:
        console_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
    logger.addHandler(console_handler)
    
    # File handler if specified
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_file, encoding='utf-8')
        if json_format:
            file_handler.setFormatter(logging.Formatter('%(message)s'))
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        logger.addHandler(file_handler)
    
    return StructuredLogger(name, getattr(logging, level.upper()))




