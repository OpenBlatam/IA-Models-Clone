#!/usr/bin/env python3
"""
Advanced Logger - Comprehensive logging system
Provides structured logging, file logging, console logging, and monitoring
"""

import logging
import sys
import json
import time
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import threading
import queue
import traceback
from enum import Enum

class LogLevel(Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime
    level: LogLevel
    message: str
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None
    stack_trace: Optional[str] = None

class Logger:
    """Advanced logger with structured logging and monitoring."""
    
    def __init__(self, name: str, level: LogLevel = LogLevel.INFO, 
                 log_file: Optional[str] = None, max_file_size: int = 10 * 1024 * 1024,
                 backup_count: int = 5, use_json: bool = True):
        self.name = name
        self.level = level
        self.log_file = log_file
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.use_json = use_json
        
        # Logging state
        self.log_entries = []
        self.log_queue = queue.Queue()
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Initialize logger
        self._initialize_logger()
        self._start_monitoring()
    
    def _initialize_logger(self):
        """Initialize the logger."""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(getattr(logging, self.level.value))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.level.value))
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if self.log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_file, maxBytes=self.max_file_size, backupCount=self.backup_count
            )
            file_handler.setLevel(getattr(logging, self.level.value))
            
            if self.use_json:
                file_formatter = self._create_json_formatter()
            else:
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def _create_json_formatter(self):
        """Create JSON formatter for structured logging."""
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line_number": record.lineno,
                    "thread_id": record.thread,
                    "process_id": record.process,
                    "metadata": getattr(record, 'metadata', {}),
                    "exception": record.exc_text if record.exc_info else None,
                    "stack_trace": traceback.format_exc() if record.exc_info else None
                }
                return json.dumps(log_entry)
        
        return JSONFormatter()
    
    def _start_monitoring(self):
        """Start log monitoring thread."""
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_logs, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_logs(self):
        """Monitor and process log entries."""
        while self.is_monitoring:
            try:
                # Process log entries from queue
                while not self.log_queue.empty():
                    log_entry = self.log_queue.get_nowait()
                    self.log_entries.append(log_entry)
                    
                    # Keep only recent entries (limit memory usage)
                    if len(self.log_entries) > 10000:
                        self.log_entries = self.log_entries[-5000:]
                
                time.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                print(f"Log monitoring error: {e}")
                time.sleep(1)
    
    def _create_log_entry(self, level: LogLevel, message: str, metadata: Dict[str, Any] = None,
                         exception: Exception = None) -> LogEntry:
        """Create a structured log entry."""
        frame = sys._getframe(2)  # Get caller frame
        
        return LogEntry(
            timestamp=datetime.now(timezone.utc),
            level=level,
            message=message,
            module=frame.f_globals.get('__name__', 'unknown'),
            function=frame.f_code.co_name,
            line_number=frame.f_lineno,
            thread_id=threading.get_ident(),
            process_id=threading.get_ident(),
            metadata=metadata or {},
            exception=str(exception) if exception else None,
            stack_trace=traceback.format_exc() if exception else None
        )
    
    def _log(self, level: LogLevel, message: str, metadata: Dict[str, Any] = None,
             exception: Exception = None):
        """Internal logging method."""
        # Create log entry
        log_entry = self._create_log_entry(level, message, metadata, exception)
        
        # Add to queue for monitoring
        self.log_queue.put(log_entry)
        
        # Log using standard logger
        if exception:
            self.logger.log(getattr(logging, level.value), message, exc_info=exception)
        else:
            self.logger.log(getattr(logging, level.value), message)
    
    def debug(self, message: str, metadata: Dict[str, Any] = None):
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, metadata)
    
    def info(self, message: str, metadata: Dict[str, Any] = None):
        """Log info message."""
        self._log(LogLevel.INFO, message, metadata)
    
    def warning(self, message: str, metadata: Dict[str, Any] = None):
        """Log warning message."""
        self._log(LogLevel.WARNING, message, metadata)
    
    def error(self, message: str, metadata: Dict[str, Any] = None, exception: Exception = None):
        """Log error message."""
        self._log(LogLevel.ERROR, message, metadata, exception)
    
    def critical(self, message: str, metadata: Dict[str, Any] = None, exception: Exception = None):
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, metadata, exception)
    
    def get_log_entries(self, level: Optional[LogLevel] = None, 
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> List[LogEntry]:
        """Get log entries with optional filtering."""
        entries = self.log_entries.copy()
        
        # Filter by level
        if level:
            entries = [entry for entry in entries if entry.level == level]
        
        # Filter by time range
        if start_time:
            entries = [entry for entry in entries if entry.timestamp >= start_time]
        
        if end_time:
            entries = [entry for entry in entries if entry.timestamp <= end_time]
        
        return entries
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        if not self.log_entries:
            return {}
        
        # Count by level
        level_counts = {}
        for entry in self.log_entries:
            level_counts[entry.level.value] = level_counts.get(entry.level.value, 0) + 1
        
        # Time range
        timestamps = [entry.timestamp for entry in self.log_entries]
        start_time = min(timestamps) if timestamps else None
        end_time = max(timestamps) if timestamps else None
        
        return {
            "total_entries": len(self.log_entries),
            "level_counts": level_counts,
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "duration_seconds": (end_time - start_time).total_seconds() if start_time and end_time else 0
        }
    
    def export_logs(self, file_path: str, format: str = "json") -> bool:
        """Export logs to file."""
        try:
            entries = self.get_log_entries()
            
            if format.lower() == "json":
                with open(file_path, 'w') as f:
                    json.dump([entry.__dict__ for entry in entries], f, indent=2, default=str)
            elif format.lower() == "csv":
                import csv
                with open(file_path, 'w', newline='') as f:
                    if entries:
                        writer = csv.DictWriter(f, fieldnames=entries[0].__dict__.keys())
                        writer.writeheader()
                        for entry in entries:
                            writer.writerow(entry.__dict__)
            else:
                with open(file_path, 'w') as f:
                    for entry in entries:
                        f.write(f"{entry.timestamp} - {entry.level.value} - {entry.message}\n")
            
            return True
            
        except Exception as e:
            self.error(f"Failed to export logs: {e}")
            return False
    
    def clear_logs(self):
        """Clear stored log entries."""
        self.log_entries.clear()
        self.info("Log entries cleared")
    
    def stop_monitoring(self):
        """Stop log monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
    
    def cleanup(self):
        """Cleanup logger."""
        self.stop_monitoring()
        
        # Close all handlers
        for handler in self.logger.handlers:
            handler.close()
        
        self.logger.handlers.clear()

class FileLogger(Logger):
    """File-specific logger."""
    
    def __init__(self, name: str, log_file: str, level: LogLevel = LogLevel.INFO,
                 max_file_size: int = 10 * 1024 * 1024, backup_count: int = 5):
        super().__init__(name, level, log_file, max_file_size, backup_count, use_json=True)
    
    def log_to_file(self, message: str, level: LogLevel = LogLevel.INFO, 
                   metadata: Dict[str, Any] = None):
        """Log directly to file."""
        self._log(level, message, metadata)

class ConsoleLogger(Logger):
    """Console-specific logger."""
    
    def __init__(self, name: str, level: LogLevel = LogLevel.INFO):
        super().__init__(name, level, log_file=None, use_json=False)
    
    def log_to_console(self, message: str, level: LogLevel = LogLevel.INFO,
                      metadata: Dict[str, Any] = None):
        """Log directly to console."""
        self._log(level, message, metadata)

class StructuredLogger(Logger):
    """Structured logger with advanced features."""
    
    def __init__(self, name: str, level: LogLevel = LogLevel.INFO, 
                 log_file: Optional[str] = None, use_json: bool = True):
        super().__init__(name, level, log_file, use_json=use_json)
        self.context = {}
    
    def set_context(self, **kwargs):
        """Set logging context."""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear logging context."""
        self.context.clear()
    
    def log_with_context(self, message: str, level: LogLevel = LogLevel.INFO,
                        metadata: Dict[str, Any] = None):
        """Log with context."""
        combined_metadata = {**self.context, **(metadata or {})}
        self._log(level, message, combined_metadata)
    
    def log_performance(self, operation: str, duration: float, 
                       metadata: Dict[str, Any] = None):
        """Log performance metrics."""
        perf_metadata = {
            "operation": operation,
            "duration": duration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(metadata or {})
        }
        self._log(LogLevel.INFO, f"Performance: {operation}", perf_metadata)
    
    def log_error_with_context(self, message: str, exception: Exception,
                              metadata: Dict[str, Any] = None):
        """Log error with context."""
        error_metadata = {
            "error_type": type(exception).__name__,
            "error_message": str(exception),
            **(metadata or {})
        }
        self._log(LogLevel.ERROR, message, error_metadata, exception)
