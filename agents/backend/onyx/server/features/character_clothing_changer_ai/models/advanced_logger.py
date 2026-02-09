"""
Advanced Logger for Flux2 Clothing Changer
==========================================

Advanced logging system with filtering and aggregation.
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import logging
from collections import deque

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """Log entry."""
    timestamp: float
    level: LogLevel
    message: str
    module: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AdvancedLogger:
    """Advanced logging system."""
    
    def __init__(
        self,
        log_file: Optional[Path] = None,
        max_history: int = 10000,
        enable_json: bool = True,
    ):
        """
        Initialize advanced logger.
        
        Args:
            log_file: Optional log file path
            max_history: Maximum log history
            enable_json: Enable JSON logging
        """
        self.log_file = log_file
        self.max_history = max_history
        self.enable_json = enable_json
        
        self.log_history: deque = deque(maxlen=max_history)
        self.filters: List[callable] = []
        self.handlers: List[callable] = []
    
    def add_filter(self, filter_func: callable) -> None:
        """
        Add log filter.
        
        Args:
            filter_func: Filter function
        """
        self.filters.append(filter_func)
    
    def add_handler(self, handler_func: callable) -> None:
        """
        Add log handler.
        
        Args:
            handler_func: Handler function
        """
        self.handlers.append(handler_func)
    
    def log(
        self,
        level: LogLevel,
        message: str,
        module: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log message.
        
        Args:
            level: Log level
            message: Log message
            module: Module name
            metadata: Optional metadata
        """
        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            message=message,
            module=module,
            metadata=metadata or {},
        )
        
        # Apply filters
        for filter_func in self.filters:
            if not filter_func(entry):
                return
        
        # Add to history
        self.log_history.append(entry)
        
        # Call handlers
        for handler_func in self.handlers:
            try:
                handler_func(entry)
            except Exception as e:
                logger.error(f"Error in log handler: {e}")
        
        # Write to file if configured
        if self.log_file:
            self._write_to_file(entry)
    
    def _write_to_file(self, entry: LogEntry) -> None:
        """Write log entry to file."""
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.log_file, "a", encoding="utf-8") as f:
                if self.enable_json:
                    log_data = {
                        "timestamp": entry.timestamp,
                        "level": entry.level.value,
                        "message": entry.message,
                        "module": entry.module,
                        "metadata": entry.metadata,
                    }
                    f.write(json.dumps(log_data) + "\n")
                else:
                    f.write(
                        f"[{entry.timestamp}] {entry.level.value} {entry.module}: {entry.message}\n"
                    )
        except Exception as e:
            logger.error(f"Failed to write log: {e}")
    
    def get_logs(
        self,
        level: Optional[LogLevel] = None,
        module: Optional[str] = None,
        time_range: Optional[float] = None,
        limit: int = 100,
    ) -> List[LogEntry]:
        """
        Get logs.
        
        Args:
            level: Optional level filter
            module: Optional module filter
            time_range: Optional time range in seconds
            limit: Maximum results
            
        Returns:
            List of log entries
        """
        logs = list(self.log_history)
        
        if level:
            logs = [l for l in logs if l.level == level]
        
        if module:
            logs = [l for l in logs if l.module == module]
        
        if time_range:
            cutoff_time = time.time() - time_range
            logs = [l for l in logs if l.timestamp >= cutoff_time]
        
        return logs[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logger statistics."""
        level_counts = {}
        for level in LogLevel:
            level_counts[level.value] = len([
                l for l in self.log_history if l.level == level
            ])
        
        return {
            "total_logs": len(self.log_history),
            "level_counts": level_counts,
            "filters": len(self.filters),
            "handlers": len(self.handlers),
        }


