"""
Log Formatting Utilities for Piel Mejorador AI SAM3
===================================================

Unified log formatting and message utilities.
"""

import logging
import json
from typing import Any, Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass, field, asdict


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime = field(default_factory=datetime.now)
    level: str = "INFO"
    logger: str = ""
    message: str = ""
    task_id: Optional[str] = None
    file_path: Optional[str] = None
    performance: Optional[Dict[str, Any]] = None
    exception: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        # Remove None values
        return {k: v for k, v in data.items() if v is not None}
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


class LogFormattingUtils:
    """Unified log formatting utilities."""
    
    @staticmethod
    def format_structured(
        level: str,
        message: str,
        logger_name: str = "",
        task_id: Optional[str] = None,
        file_path: Optional[str] = None,
        performance: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        **metadata
    ) -> str:
        """
        Format structured log entry.
        
        Args:
            level: Log level
            message: Log message
            logger_name: Logger name
            task_id: Optional task ID
            file_path: Optional file path
            performance: Optional performance data
            exception: Optional exception
            **metadata: Additional metadata
            
        Returns:
            Formatted log string (JSON)
        """
        entry = LogEntry(
            level=level,
            logger=logger_name,
            message=message,
            task_id=task_id,
            file_path=file_path,
            performance=performance,
            exception=str(exception) if exception else None,
            metadata=metadata
        )
        return entry.to_json()
    
    @staticmethod
    def format_simple(
        level: str,
        message: str,
        logger_name: str = "",
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Format simple log entry.
        
        Args:
            level: Log level
            message: Log message
            logger_name: Logger name
            timestamp: Optional timestamp
            
        Returns:
            Formatted log string
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        return f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {logger_name} - {level} - {message}"
    
    @staticmethod
    def format_with_context(
        message: str,
        context: Dict[str, Any],
        level: str = "INFO"
    ) -> str:
        """
        Format log message with context.
        
        Args:
            message: Log message
            context: Context dictionary
            level: Log level
            
        Returns:
            Formatted log string
        """
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        return f"[{level}] {message} | Context: {context_str}"
    
    @staticmethod
    def format_performance(
        operation: str,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format performance log.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            metadata: Optional metadata
            
        Returns:
            Formatted performance log
        """
        perf_str = f"Operation: {operation}, Duration: {duration:.3f}s"
        if metadata:
            meta_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
            perf_str += f", Metadata: {meta_str}"
        return perf_str
    
    @staticmethod
    def format_error(
        error: Exception,
        operation: str = "operation",
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format error log.
        
        Args:
            error: Exception
            operation: Operation name
            context: Optional context
            
        Returns:
            Formatted error log
        """
        error_str = f"Error in {operation}: {type(error).__name__}: {str(error)}"
        if context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in context.items())
            error_str += f" | Context: {ctx_str}"
        return error_str


class StructuredFormatter(logging.Formatter):
    """Structured formatter for logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record.
        
        Args:
            record: Log record
            
        Returns:
            Formatted log string
        """
        entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created),
            level=record.levelname,
            logger=record.name,
            message=record.getMessage(),
        )
        
        # Add extra fields
        if hasattr(record, "task_id"):
            entry.task_id = record.task_id
        if hasattr(record, "file_path"):
            entry.file_path = record.file_path
        if hasattr(record, "performance"):
            entry.performance = record.performance
        if record.exc_info:
            entry.exception = self.formatException(record.exc_info)
        
        # Add extra metadata
        for key, value in record.__dict__.items():
            if key not in [
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "message", "pathname", "process", "processName", "relativeCreated",
                "thread", "threadName", "exc_info", "exc_text", "stack_info",
                "task_id", "file_path", "performance"
            ]:
                entry.metadata[key] = value
        
        return entry.to_json()


# Convenience functions
def format_structured(level: str, message: str, **kwargs) -> str:
    """Format structured log."""
    return LogFormattingUtils.format_structured(level, message, **kwargs)


def format_simple(level: str, message: str, **kwargs) -> str:
    """Format simple log."""
    return LogFormattingUtils.format_simple(level, message, **kwargs)


def format_error(error: Exception, **kwargs) -> str:
    """Format error log."""
    return LogFormattingUtils.format_error(error, **kwargs)




