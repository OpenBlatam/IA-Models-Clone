"""
Structured Logging

Utilities for structured logging with JSON support.
"""

import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class StructuredLogger:
    """Structured logger with JSON support."""
    
    def __init__(
        self,
        name: str,
        log_dir: Optional[str] = None,
        level: int = logging.INFO,
        use_json: bool = False
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level
            use_json: Use JSON format
        """
        self.name = name
        self.use_json = use_json
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler()
        if use_json:
            console_handler.setFormatter(JsonFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            
            log_file = log_path / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            
            if use_json:
                file_handler.setFormatter(JsonFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                )
            
            self.logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file}")
    
    def log(
        self,
        level: int,
        message: str,
        **kwargs
    ) -> None:
        """
        Log message with additional context.
        
        Args:
            level: Logging level
            message: Log message
            **kwargs: Additional context
        """
        if self.use_json:
            log_data = {
                'message': message,
                'timestamp': datetime.now().isoformat(),
                **kwargs
            }
            self.logger.log(level, json.dumps(log_data))
        else:
            if kwargs:
                context = ', '.join(f"{k}={v}" for k, v in kwargs.items())
                self.logger.log(level, f"{message} | {context}")
            else:
                self.logger.log(level, message)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.log(logging.ERROR, message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.log(logging.DEBUG, message, **kwargs)


class JsonFormatter(logging.Formatter):
    """JSON formatter for logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        return json.dumps(log_data)


def setup_logging(
    name: str = "music_generation",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    use_json: bool = False
) -> StructuredLogger:
    """
    Setup structured logging.
    
    Args:
        name: Logger name
        log_dir: Log directory
        level: Logging level
        use_json: Use JSON format
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name, log_dir, level, use_json)


def get_logger(name: str) -> logging.Logger:
    """
    Get logger by name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)



