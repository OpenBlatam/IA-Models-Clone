"""
Advanced Logging
================

Advanced logging utilities with structured logging and performance tracking.
"""

import logging
import sys
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from enum import Enum


class LogLevel(Enum):
    """Log level."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class StructuredLogger:
    """Structured logger with JSON output."""
    
    def __init__(self, name: str, level: LogLevel = LogLevel.INFO):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            level: Log level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level.value)
        self.name = name
    
    def _log(self, level: LogLevel, message: str, **kwargs):
        """
        Log structured message.
        
        Args:
            level: Log level
            message: Log message
            **kwargs: Additional fields
        """
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": level.name,
            "logger": self.name,
            "message": message,
            **kwargs
        }
        
        log_message = json.dumps(log_data)
        self.logger.log(level.value, log_message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, **kwargs)


class PerformanceLogger:
    """Logger for performance tracking."""
    
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
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log operation performance.
        
        Args:
            operation: Operation name
            duration: Operation duration in seconds
            success: Whether operation was successful
            metadata: Optional metadata
        """
        log_data = {
            "type": "performance",
            "operation": operation,
            "duration": duration,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            log_data["metadata"] = metadata
        
        self.logger.info(json.dumps(log_data))
    
    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration: float,
        **kwargs
    ):
        """
        Log HTTP request.
        
        Args:
            method: HTTP method
            path: Request path
            status_code: Response status code
            duration: Request duration
            **kwargs: Additional fields
        """
        self.log_operation(
            f"{method} {path}",
            duration,
            success=(200 <= status_code < 400),
            metadata={
                "method": method,
                "path": path,
                "status_code": status_code,
                **kwargs
            }
        )


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: LogLevel = LogLevel.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    json_format: bool = False
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Log directory path
        log_level: Log level
        max_bytes: Maximum log file size
        backup_count: Number of backup files
        json_format: Use JSON format
        
    Returns:
        Root logger
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.value)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level.value)
    
    if json_format:
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s"}'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / "app.log"
        file_handler = RotatingFileHandler(
            str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(log_level.value)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger




