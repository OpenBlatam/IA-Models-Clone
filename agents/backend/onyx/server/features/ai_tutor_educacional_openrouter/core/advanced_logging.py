"""
Advanced logging system with structured logging and monitoring.
"""

import logging
import json
import sys
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data
        
        return json.dumps(log_data, ensure_ascii=False)


class AdvancedLogger:
    """
    Advanced logging system with structured logging and file rotation.
    """
    
    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        level: str = "INFO",
        enable_file_logging: bool = True,
        enable_console_logging: bool = True,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Console handler
        if enable_console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if enable_file_logging:
            file_handler = RotatingFileHandler(
                self.log_dir / f"{name}.log",
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, level.upper()))
            structured_formatter = StructuredFormatter()
            file_handler.setFormatter(structured_formatter)
            self.logger.addHandler(file_handler)
        
        # Error file handler
        if enable_file_logging:
            error_handler = RotatingFileHandler(
                self.log_dir / f"{name}_errors.log",
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            structured_formatter = StructuredFormatter()
            error_handler.setFormatter(structured_formatter)
            self.logger.addHandler(error_handler)
    
    def log_with_context(
        self,
        level: str,
        message: str,
        extra_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log with additional context data."""
        log_method = getattr(self.logger, level.lower())
        
        if extra_data:
            # Create a custom log record with extra data
            old_factory = logging.getLogRecordFactory()
            
            def record_factory(*args, **kwargs):
                record = old_factory(*args, **kwargs)
                record.extra_data = extra_data
                return record
            
            logging.setLogRecordFactory(record_factory)
            log_method(message, **kwargs)
            logging.setLogRecordFactory(old_factory)
        else:
            log_method(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, **kwargs)






