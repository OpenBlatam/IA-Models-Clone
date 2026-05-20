"""
Advanced Logging Utilities
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import sys

logger = logging.getLogger(__name__)


class JSONFormatter(logging.Formatter):
    """JSON formatter for logs"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in [
                "name", "msg", "args", "created", "filename",
                "funcName", "levelname", "levelno", "lineno",
                "module", "msecs", "message", "pathname", "process",
                "processName", "relativeCreated", "thread", "threadName",
                "exc_info", "exc_text", "stack_info"
            ]:
                log_data[key] = value
        
        return json.dumps(log_data)


class StructuredLogger:
    """Structured logger with context"""
    
    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        use_json: bool = False
    ):
        """
        Initialize structured logger
        
        Args:
            name: Logger name
            level: Log level
            log_file: Optional log file path
            use_json: Use JSON formatting
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if use_json:
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            if use_json:
                file_handler.setFormatter(JSONFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                )
            self.logger.addHandler(file_handler)
        
        self.context = {}
    
    def set_context(self, **kwargs):
        """Set logging context"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear logging context"""
        self.context.clear()
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log with context"""
        extra = {**self.context, **kwargs}
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log_with_context(logging.CRITICAL, message, **kwargs)


class LogAggregator:
    """Aggregate logs from multiple sources"""
    
    def __init__(self):
        """Initialize log aggregator"""
        self.logs: List[Dict[str, Any]] = []
        logger.info("LogAggregator initialized")
    
    def add_log(self, log_entry: Dict[str, Any]):
        """Add log entry"""
        self.logs.append({
            **log_entry,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def get_logs(
        self,
        level: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get logs
        
        Args:
            level: Optional level filter
            limit: Optional limit
        
        Returns:
            List of log entries
        """
        logs = self.logs
        
        if level:
            logs = [log for log in logs if log.get("level") == level]
        
        if limit:
            logs = logs[-limit:]
        
        return logs
    
    def clear_logs(self):
        """Clear all logs"""
        self.logs.clear()
