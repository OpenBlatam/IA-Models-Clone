"""
Structured Logging
==================

Advanced structured logging system.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log level."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime
    level: LogLevel
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "context": self.context,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class StructuredLogger:
    """Structured logger."""
    
    def __init__(self, name: str, log_file: Optional[Path] = None):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            log_file: Optional log file path
        """
        self.name = name
        self.log_file = log_file
        self.logger = logging.getLogger(name)
        self.entries: List[LogEntry] = []
        self.max_entries = 10000
    
    def log(
        self,
        level: LogLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log structured entry.
        
        Args:
            level: Log level
            message: Log message
            context: Optional context
            metadata: Optional metadata
        """
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            context=context or {},
            metadata=metadata or {}
        )
        
        self.entries.append(entry)
        
        # Limit entries
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
        
        # Write to file if configured
        if self.log_file:
            self._write_to_file(entry)
        
        # Also log to standard logger
        log_func = getattr(self.logger, level.value, self.logger.info)
        log_func(f"{message} | Context: {context} | Metadata: {metadata}")
    
    def _write_to_file(self, entry: LogEntry):
        """Write log entry to file."""
        try:
            if self.log_file:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(entry.to_json() + '\n')
        except Exception as e:
            self.logger.error(f"Error writing log entry: {e}")
    
    def debug(self, message: str, context: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None):
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, context, metadata)
    
    def info(self, message: str, context: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None):
        """Log info message."""
        self.log(LogLevel.INFO, message, context, metadata)
    
    def warning(self, message: str, context: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None):
        """Log warning message."""
        self.log(LogLevel.WARNING, message, context, metadata)
    
    def error(self, message: str, context: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None):
        """Log error message."""
        self.log(LogLevel.ERROR, message, context, metadata)
    
    def critical(self, message: str, context: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None):
        """Log critical message."""
        self.log(LogLevel.CRITICAL, message, context, metadata)
    
    def get_entries(
        self,
        level: Optional[LogLevel] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[LogEntry]:
        """
        Get log entries with filters.
        
        Args:
            level: Optional level filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of entries
            
        Returns:
            List of log entries
        """
        entries = self.entries
        
        if level:
            entries = [e for e in entries if e.level == level]
        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]
        if end_time:
            entries = [e for e in entries if e.timestamp <= end_time]
        
        return entries[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        by_level = {}
        for entry in self.entries:
            level = entry.level.value
            by_level[level] = by_level.get(level, 0) + 1
        
        return {
            "total_entries": len(self.entries),
            "by_level": by_level
        }

