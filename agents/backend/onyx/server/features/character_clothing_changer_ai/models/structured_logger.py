"""
Structured Logger for Flux2 Clothing Changer
=============================================

Structured logging with context and metrics.
"""

import logging
import json
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    message: str
    context: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration: Optional[float] = None


class StructuredLogger:
    """Structured logger with context tracking."""
    
    def __init__(
        self,
        name: str = "Flux2ClothingChanger",
        log_file: Optional[Path] = None,
        enable_json: bool = True,
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            log_file: Optional log file path
            enable_json: Enable JSON logging
        """
        self.name = name
        self.log_file = log_file
        self.enable_json = enable_json
        self.logger = logging.getLogger(name)
        self.context_stack: list = []
        self.lock = threading.Lock()
        
        # Setup file handler if specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_file)
            if enable_json:
                handler.setFormatter(JsonFormatter())
            self.logger.addHandler(handler)
    
    def push_context(self, context: Dict[str, Any]) -> None:
        """Push context to stack."""
        with self.lock:
            self.context_stack.append(context)
    
    def pop_context(self) -> Optional[Dict[str, Any]]:
        """Pop context from stack."""
        with self.lock:
            if self.context_stack:
                return self.context_stack.pop()
            return None
    
    def get_context(self) -> Dict[str, Any]:
        """Get merged context from stack."""
        with self.lock:
            merged = {}
            for ctx in self.context_stack:
                merged.update(ctx)
            return merged
    
    def log(
        self,
        level: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
        duration: Optional[float] = None,
    ) -> None:
        """
        Log structured entry.
        
        Args:
            level: Log level (info, warning, error, debug)
            message: Log message
            context: Additional context
            metrics: Optional metrics
            error: Optional error
            duration: Optional duration in seconds
        """
        merged_context = self.get_context()
        if context:
            merged_context.update(context)
        
        entry = LogEntry(
            timestamp=datetime.utcnow().isoformat(),
            level=level,
            message=message,
            context=merged_context,
            metrics=metrics,
            error=str(error) if error else None,
            duration=duration,
        )
        
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        
        if self.enable_json:
            log_method(json.dumps(asdict(entry), default=str))
        else:
            log_method(f"{message} | Context: {merged_context}")
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.log("info", message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.log("warning", message, **kwargs)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs) -> None:
        """Log error message."""
        self.log("error", message, error=error, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.log("debug", message, **kwargs)
    
    def timed(self, message: str, **kwargs):
        """Context manager for timed operations."""
        return TimedContext(self, message, **kwargs)


class TimedContext:
    """Context manager for timed logging."""
    
    def __init__(self, logger: StructuredLogger, message: str, **kwargs):
        self.logger = logger
        self.message = message
        self.kwargs = kwargs
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(f"Starting: {self.message}", **self.kwargs)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type:
            self.logger.error(
                f"Failed: {self.message}",
                error=exc_val,
                duration=duration,
                **self.kwargs
            )
        else:
            self.logger.info(
                f"Completed: {self.message}",
                duration=duration,
                **self.kwargs
            )


class JsonFormatter(logging.Formatter):
    """JSON formatter for logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        try:
            # If already JSON, return as-is
            if record.msg.startswith("{"):
                return record.msg
        except:
            pass
        
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


