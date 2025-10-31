"""
Refactored Logging System

Sistema de logging y auditoría refactorizado para el AI History Comparison System.
Maneja logging estructurado, auditoría completa, análisis de logs y alertas inteligentes.
"""

import asyncio
import logging
import json
import sys
import os
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Callable, Union, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
from contextlib import asynccontextmanager
import weakref
from collections import defaultdict, deque
import hashlib
import gzip
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar('T')


class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    AUDIT = "audit"
    SECURITY = "security"
    PERFORMANCE = "performance"


class LogCategory(Enum):
    """Log category enumeration"""
    SYSTEM = "system"
    APPLICATION = "application"
    USER = "user"
    SECURITY = "security"
    AUDIT = "audit"
    PERFORMANCE = "performance"
    ERROR = "error"
    API = "api"
    DATABASE = "database"
    CACHE = "cache"
    CUSTOM = "custom"


class LogFormat(Enum):
    """Log format enumeration"""
    JSON = "json"
    TEXT = "text"
    CSV = "csv"
    XML = "xml"
    STRUCTURED = "structured"


class LogRotation(Enum):
    """Log rotation enumeration"""
    SIZE = "size"
    TIME = "time"
    SIZE_AND_TIME = "size_and_time"


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    message: str
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    duration: Optional[float] = None
    memory_usage: Optional[int] = None
    cpu_usage: Optional[float] = None
    tags: Set[str] = field(default_factory=set)
    labels: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None
    stack_trace: Optional[str] = None


@dataclass
class AuditEntry:
    """Audit log entry"""
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    result: str
    ip_address: str
    user_agent: str
    session_id: str
    request_id: str
    duration: float
    details: Dict[str, Any] = field(default_factory=dict)
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None


@dataclass
class LogFilter:
    """Log filter configuration"""
    name: str
    level: Optional[LogLevel] = None
    category: Optional[LogCategory] = None
    module: Optional[str] = None
    function: Optional[str] = None
    user_id: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    labels: Dict[str, str] = field(default_factory=dict)
    time_range: Optional[Tuple[datetime, datetime]] = None
    custom_filter: Optional[Callable] = None


class LogHandler(ABC):
    """Abstract log handler"""
    
    @abstractmethod
    async def handle(self, entry: LogEntry) -> None:
        """Handle log entry"""
        pass
    
    @abstractmethod
    async def flush(self) -> None:
        """Flush buffered logs"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close handler"""
        pass


class ConsoleLogHandler(LogHandler):
    """Console log handler with colored output"""
    
    def __init__(self, format_type: LogFormat = LogFormat.TEXT, use_colors: bool = True):
        self._format_type = format_type
        self._use_colors = use_colors
        self._colors = {
            LogLevel.DEBUG: '\033[36m',      # Cyan
            LogLevel.INFO: '\033[32m',       # Green
            LogLevel.WARNING: '\033[33m',    # Yellow
            LogLevel.ERROR: '\033[31m',      # Red
            LogLevel.CRITICAL: '\033[35m',   # Magenta
            LogLevel.AUDIT: '\033[34m',      # Blue
            LogLevel.SECURITY: '\033[91m',   # Light Red
            LogLevel.PERFORMANCE: '\033[93m' # Light Yellow
        }
        self._reset_color = '\033[0m'
    
    async def handle(self, entry: LogEntry) -> None:
        """Handle log entry"""
        if self._format_type == LogFormat.JSON:
            output = json.dumps(asdict(entry), default=str, indent=2)
        elif self._format_type == LogFormat.TEXT:
            output = self._format_text(entry)
        else:
            output = str(entry)
        
        if self._use_colors:
            color = self._colors.get(entry.level, '')
            output = f"{color}{output}{self._reset_color}"
        
        print(output, file=sys.stdout if entry.level in [LogLevel.DEBUG, LogLevel.INFO] else sys.stderr)
    
    def _format_text(self, entry: LogEntry) -> str:
        """Format log entry as text"""
        return (f"{entry.timestamp.isoformat()} "
                f"[{entry.level.value.upper()}] "
                f"[{entry.category.value}] "
                f"{entry.module}:{entry.function}:{entry.line_number} - "
                f"{entry.message}")
    
    async def flush(self) -> None:
        """Flush console output"""
        sys.stdout.flush()
        sys.stderr.flush()
    
    async def close(self) -> None:
        """Close console handler"""
        await self.flush()


class FileLogHandler(LogHandler):
    """File log handler with rotation and compression"""
    
    def __init__(self, file_path: str, format_type: LogFormat = LogFormat.JSON,
                 rotation: LogRotation = LogRotation.SIZE_AND_TIME,
                 max_size_mb: int = 100, max_files: int = 10):
        self._file_path = Path(file_path)
        self._format_type = format_type
        self._rotation = rotation
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._max_files = max_files
        self._buffer: List[LogEntry] = []
        self._buffer_size = 1000
        self._lock = asyncio.Lock()
        
        # Create directory if it doesn't exist
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
    
    async def handle(self, entry: LogEntry) -> None:
        """Handle log entry"""
        async with self._lock:
            self._buffer.append(entry)
            
            if len(self._buffer) >= self._buffer_size:
                await self._flush_buffer()
    
    async def _flush_buffer(self) -> None:
        """Flush buffer to file"""
        if not self._buffer:
            return
        
        # Check if rotation is needed
        if await self._should_rotate():
            await self._rotate_file()
        
        # Write to file
        try:
            with open(self._file_path, 'a', encoding='utf-8') as f:
                for entry in self._buffer:
                    if self._format_type == LogFormat.JSON:
                        f.write(json.dumps(asdict(entry), default=str) + '\n')
                    else:
                        f.write(self._format_text(entry) + '\n')
            
            self._buffer.clear()
            
        except Exception as e:
            logger.error(f"Error writing to log file: {e}")
    
    async def _should_rotate(self) -> bool:
        """Check if file rotation is needed"""
        if not self._file_path.exists():
            return False
        
        file_size = self._file_path.stat().st_size
        return file_size >= self._max_size_bytes
    
    async def _rotate_file(self) -> None:
        """Rotate log file"""
        try:
            # Compress current file
            compressed_path = f"{self._file_path}.gz"
            with open(self._file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove original file
            self._file_path.unlink()
            
            # Clean up old files
            await self._cleanup_old_files()
            
        except Exception as e:
            logger.error(f"Error rotating log file: {e}")
    
    async def _cleanup_old_files(self) -> None:
        """Clean up old log files"""
        try:
            log_dir = self._file_path.parent
            pattern = f"{self._file_path.name}.*.gz"
            
            files = list(log_dir.glob(pattern))
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Keep only max_files
            for old_file in files[self._max_files:]:
                old_file.unlink()
                
        except Exception as e:
            logger.error(f"Error cleaning up old log files: {e}")
    
    def _format_text(self, entry: LogEntry) -> str:
        """Format log entry as text"""
        return (f"{entry.timestamp.isoformat()} "
                f"[{entry.level.value.upper()}] "
                f"[{entry.category.value}] "
                f"{entry.module}:{entry.function}:{entry.line_number} - "
                f"{entry.message}")
    
    async def flush(self) -> None:
        """Flush buffer to file"""
        async with self._lock:
            await self._flush_buffer()
    
    async def close(self) -> None:
        """Close file handler"""
        await self.flush()


class DatabaseLogHandler(LogHandler):
    """Database log handler for structured storage"""
    
    def __init__(self, connection_string: str, table_name: str = "logs"):
        self._connection_string = connection_string
        self._table_name = table_name
        self._buffer: List[LogEntry] = []
        self._buffer_size = 1000
        self._lock = asyncio.Lock()
    
    async def handle(self, entry: LogEntry) -> None:
        """Handle log entry"""
        async with self._lock:
            self._buffer.append(entry)
            
            if len(self._buffer) >= self._buffer_size:
                await self._flush_buffer()
    
    async def _flush_buffer(self) -> None:
        """Flush buffer to database"""
        if not self._buffer:
            return
        
        try:
            # This would implement actual database insertion
            # For now, just clear the buffer
            self._buffer.clear()
            
        except Exception as e:
            logger.error(f"Error writing to database: {e}")
    
    async def flush(self) -> None:
        """Flush buffer to database"""
        async with self._lock:
            await self._flush_buffer()
    
    async def close(self) -> None:
        """Close database handler"""
        await self.flush()


class LogAnalyzer:
    """Log analyzer for pattern detection and insights"""
    
    def __init__(self):
        self._patterns: Dict[str, re.Pattern] = {}
        self._alerts: List[Callable] = []
        self._stats: Dict[str, Any] = defaultdict(int)
    
    def add_pattern(self, name: str, pattern: str) -> None:
        """Add pattern for analysis"""
        self._patterns[name] = re.compile(pattern, re.IGNORECASE)
    
    async def analyze_entry(self, entry: LogEntry) -> Dict[str, Any]:
        """Analyze log entry for patterns"""
        analysis = {
            "patterns_matched": [],
            "anomalies": [],
            "insights": []
        }
        
        # Check patterns
        for pattern_name, pattern in self._patterns.items():
            if pattern.search(entry.message):
                analysis["patterns_matched"].append(pattern_name)
        
        # Check for anomalies
        if entry.level == LogLevel.ERROR:
            analysis["anomalies"].append("error_detected")
        
        if entry.duration and entry.duration > 5.0:  # 5 seconds
            analysis["anomalies"].append("slow_operation")
        
        if entry.memory_usage and entry.memory_usage > 100 * 1024 * 1024:  # 100MB
            analysis["anomalies"].append("high_memory_usage")
        
        # Generate insights
        if entry.category == LogCategory.SECURITY:
            analysis["insights"].append("security_event")
        
        if entry.category == LogCategory.PERFORMANCE:
            analysis["insights"].append("performance_metric")
        
        return analysis
    
    async def analyze_logs(self, logs: List[LogEntry]) -> Dict[str, Any]:
        """Analyze multiple log entries"""
        analysis = {
            "total_logs": len(logs),
            "level_distribution": defaultdict(int),
            "category_distribution": defaultdict(int),
            "error_rate": 0.0,
            "average_duration": 0.0,
            "patterns_found": defaultdict(int),
            "anomalies": []
        }
        
        total_duration = 0.0
        error_count = 0
        
        for log in logs:
            # Level distribution
            analysis["level_distribution"][log.level.value] += 1
            
            # Category distribution
            analysis["category_distribution"][log.category.value] += 1
            
            # Error rate
            if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                error_count += 1
            
            # Duration
            if log.duration:
                total_duration += log.duration
            
            # Pattern analysis
            entry_analysis = await self.analyze_entry(log)
            for pattern in entry_analysis["patterns_matched"]:
                analysis["patterns_found"][pattern] += 1
            
            analysis["anomalies"].extend(entry_analysis["anomalies"])
        
        # Calculate metrics
        if logs:
            analysis["error_rate"] = error_count / len(logs)
            analysis["average_duration"] = total_duration / len(logs)
        
        return analysis


class AuditLogger:
    """Audit logger for compliance and tracking"""
    
    def __init__(self):
        self._audit_entries: List[AuditEntry] = []
        self._lock = asyncio.Lock()
        self._callbacks: List[Callable] = []
    
    async def log_audit(self, entry: AuditEntry) -> None:
        """Log audit entry"""
        async with self._lock:
            self._audit_entries.append(entry)
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(entry)
                    else:
                        callback(entry)
                except Exception as e:
                    logger.error(f"Error in audit callback: {e}")
    
    async def get_audit_trail(self, user_id: str = None, start_time: datetime = None,
                             end_time: datetime = None) -> List[AuditEntry]:
        """Get audit trail"""
        async with self._lock:
            entries = self._audit_entries.copy()
            
            if user_id:
                entries = [e for e in entries if e.user_id == user_id]
            
            if start_time:
                entries = [e for e in entries if e.timestamp >= start_time]
            
            if end_time:
                entries = [e for e in entries if e.timestamp <= end_time]
            
            return entries
    
    def add_callback(self, callback: Callable) -> None:
        """Add audit callback"""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> None:
        """Remove audit callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)


class RefactoredLoggingManager:
    """Refactored logging manager with comprehensive features"""
    
    def __init__(self):
        self._handlers: List[LogHandler] = []
        self._filters: List[LogFilter] = []
        self._analyzer = LogAnalyzer()
        self._audit_logger = AuditLogger()
        self._buffer: List[LogEntry] = []
        self._buffer_size = 1000
        self._flush_interval = 5.0  # seconds
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval = 3600.0  # 1 hour
        self._callbacks: List[Callable] = []
    
    async def initialize(self) -> None:
        """Initialize logging manager"""
        # Add default handlers
        self.add_handler(ConsoleLogHandler())
        self.add_handler(FileLogHandler("logs/application.log"))
        
        # Start flush task
        self._flush_task = asyncio.create_task(self._flush_loop())
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Refactored logging manager initialized")
    
    async def _flush_loop(self) -> None:
        """Flush buffer loop"""
        while True:
            try:
                await asyncio.sleep(self._flush_interval)
                await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Cleanup loop for old logs"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_old_logs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_old_logs(self) -> None:
        """Cleanup old log entries"""
        # Implementation would clean up old log entries
        pass
    
    def add_handler(self, handler: LogHandler) -> None:
        """Add log handler"""
        self._handlers.append(handler)
    
    def remove_handler(self, handler: LogHandler) -> None:
        """Remove log handler"""
        if handler in self._handlers:
            self._handlers.remove(handler)
    
    def add_filter(self, filter_config: LogFilter) -> None:
        """Add log filter"""
        self._filters.append(filter_config)
    
    def remove_filter(self, filter_name: str) -> None:
        """Remove log filter"""
        self._filters = [f for f in self._filters if f.name != filter_name]
    
    async def log(self, level: LogLevel, category: LogCategory, message: str,
                  module: str = None, function: str = None, line_number: int = None,
                  user_id: str = None, session_id: str = None, request_id: str = None,
                  correlation_id: str = None, ip_address: str = None, user_agent: str = None,
                  duration: float = None, memory_usage: int = None, cpu_usage: float = None,
                  tags: Set[str] = None, labels: Dict[str, str] = None,
                  context: Dict[str, Any] = None, exception: Exception = None) -> None:
        """Log entry"""
        # Get caller information
        if not module or not function or not line_number:
            frame = sys._getframe(1)
            module = module or frame.f_globals.get('__name__', 'unknown')
            function = function or frame.f_code.co_name
            line_number = line_number or frame.f_lineno
        
        # Create log entry
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=level,
            category=category,
            message=message,
            module=module,
            function=function,
            line_number=line_number,
            thread_id=threading.get_ident(),
            process_id=os.getpid(),
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            correlation_id=correlation_id,
            ip_address=ip_address,
            user_agent=user_agent,
            duration=duration,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            tags=tags or set(),
            labels=labels or {},
            context=context or {},
            exception=str(exception) if exception else None,
            stack_trace=traceback.format_exc() if exception else None
        )
        
        # Apply filters
        if not await self._should_log(entry):
            return
        
        # Add to buffer
        async with self._lock:
            self._buffer.append(entry)
            
            if len(self._buffer) >= self._buffer_size:
                await self._flush_buffer()
        
        # Analyze entry
        analysis = await self._analyzer.analyze_entry(entry)
        if analysis["anomalies"]:
            await self._handle_anomalies(entry, analysis["anomalies"])
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(entry)
                else:
                    callback(entry)
            except Exception as e:
                logger.error(f"Error in logging callback: {e}")
    
    async def _should_log(self, entry: LogEntry) -> bool:
        """Check if entry should be logged based on filters"""
        for filter_config in self._filters:
            if filter_config.level and entry.level != filter_config.level:
                continue
            
            if filter_config.category and entry.category != filter_config.category:
                continue
            
            if filter_config.module and entry.module != filter_config.module:
                continue
            
            if filter_config.function and entry.function != filter_config.function:
                continue
            
            if filter_config.user_id and entry.user_id != filter_config.user_id:
                continue
            
            if filter_config.tags and not filter_config.tags.issubset(entry.tags):
                continue
            
            if filter_config.labels:
                for key, value in filter_config.labels.items():
                    if entry.labels.get(key) != value:
                        continue
            
            if filter_config.time_range:
                start_time, end_time = filter_config.time_range
                if not (start_time <= entry.timestamp <= end_time):
                    continue
            
            if filter_config.custom_filter and not filter_config.custom_filter(entry):
                continue
            
            return True
        
        return len(self._filters) == 0  # Log if no filters
    
    async def _flush_buffer(self) -> None:
        """Flush buffer to handlers"""
        if not self._buffer:
            return
        
        entries = self._buffer.copy()
        self._buffer.clear()
        
        # Send to handlers
        for handler in self._handlers:
            try:
                for entry in entries:
                    await handler.handle(entry)
                await handler.flush()
            except Exception as e:
                logger.error(f"Error in log handler: {e}")
    
    async def _handle_anomalies(self, entry: LogEntry, anomalies: List[str]) -> None:
        """Handle detected anomalies"""
        for anomaly in anomalies:
            if anomaly == "error_detected":
                await self.log(
                    LogLevel.WARNING,
                    LogCategory.SYSTEM,
                    f"Error detected in {entry.module}:{entry.function}",
                    tags={"anomaly", "error"}
                )
            elif anomaly == "slow_operation":
                await self.log(
                    LogLevel.WARNING,
                    LogCategory.PERFORMANCE,
                    f"Slow operation detected: {entry.duration}s",
                    tags={"anomaly", "performance"}
                )
            elif anomaly == "high_memory_usage":
                await self.log(
                    LogLevel.WARNING,
                    LogCategory.PERFORMANCE,
                    f"High memory usage detected: {entry.memory_usage} bytes",
                    tags={"anomaly", "memory"}
                )
    
    async def flush(self) -> None:
        """Flush all buffers"""
        async with self._lock:
            await self._flush_buffer()
    
    async def audit(self, user_id: str, action: str, resource: str, result: str,
                   ip_address: str, user_agent: str, session_id: str, request_id: str,
                   duration: float, details: Dict[str, Any] = None,
                   before_state: Dict[str, Any] = None, after_state: Dict[str, Any] = None) -> None:
        """Log audit entry"""
        entry = AuditEntry(
            timestamp=datetime.utcnow(),
            user_id=user_id,
            action=action,
            resource=resource,
            result=result,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            request_id=request_id,
            duration=duration,
            details=details or {},
            before_state=before_state,
            after_state=after_state
        )
        
        await self._audit_logger.log_audit(entry)
    
    async def get_logs(self, start_time: datetime = None, end_time: datetime = None,
                      level: LogLevel = None, category: LogCategory = None,
                      limit: int = 1000) -> List[LogEntry]:
        """Get log entries"""
        # This would implement log retrieval from storage
        return []
    
    async def get_audit_trail(self, **kwargs) -> List[AuditEntry]:
        """Get audit trail"""
        return await self._audit_logger.get_audit_trail(**kwargs)
    
    async def analyze_logs(self, logs: List[LogEntry]) -> Dict[str, Any]:
        """Analyze logs"""
        return await self._analyzer.analyze_logs(logs)
    
    def add_callback(self, callback: Callable) -> None:
        """Add logging callback"""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> None:
        """Remove logging callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get logging manager health status"""
        return {
            "handlers_count": len(self._handlers),
            "filters_count": len(self._filters),
            "buffer_size": len(self._buffer),
            "flush_interval": self._flush_interval,
            "cleanup_interval": self._cleanup_interval
        }
    
    async def shutdown(self) -> None:
        """Shutdown logging manager"""
        if self._flush_task:
            self._flush_task.cancel()
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Flush all handlers
        for handler in self._handlers:
            try:
                await handler.close()
            except Exception as e:
                logger.error(f"Error closing log handler: {e}")
        
        logger.info("Refactored logging manager shutdown")


# Global logging manager
logging_manager = RefactoredLoggingManager()


# Convenience functions
async def log_debug(message: str, **kwargs):
    """Log debug message"""
    await logging_manager.log(LogLevel.DEBUG, LogCategory.APPLICATION, message, **kwargs)


async def log_info(message: str, **kwargs):
    """Log info message"""
    await logging_manager.log(LogLevel.INFO, LogCategory.APPLICATION, message, **kwargs)


async def log_warning(message: str, **kwargs):
    """Log warning message"""
    await logging_manager.log(LogLevel.WARNING, LogCategory.APPLICATION, message, **kwargs)


async def log_error(message: str, **kwargs):
    """Log error message"""
    await logging_manager.log(LogLevel.ERROR, LogCategory.ERROR, message, **kwargs)


async def log_critical(message: str, **kwargs):
    """Log critical message"""
    await logging_manager.log(LogLevel.CRITICAL, LogCategory.ERROR, message, **kwargs)


async def log_audit(user_id: str, action: str, resource: str, result: str, **kwargs):
    """Log audit entry"""
    await logging_manager.audit(user_id, action, resource, result, **kwargs)


# Logging decorators
def log_function_call(level: LogLevel = LogLevel.INFO, category: LogCategory = LogCategory.APPLICATION):
    """Log function call decorator"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                await logging_manager.log(
                    level,
                    category,
                    f"Function {func.__name__} completed successfully",
                    function=func.__name__,
                    duration=duration
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                await logging_manager.log(
                    LogLevel.ERROR,
                    LogCategory.ERROR,
                    f"Function {func.__name__} failed: {str(e)}",
                    function=func.__name__,
                    duration=duration,
                    exception=e
                )
                
                raise
        
        return wrapper
    return decorator


def audit_function_call(action: str, resource: str):
    """Audit function call decorator"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            user_id = kwargs.get('user_id', 'system')
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                await logging_manager.audit(
                    user_id=user_id,
                    action=action,
                    resource=resource,
                    result="success",
                    ip_address=kwargs.get('ip_address', ''),
                    user_agent=kwargs.get('user_agent', ''),
                    session_id=kwargs.get('session_id', ''),
                    request_id=kwargs.get('request_id', ''),
                    duration=duration
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                await logging_manager.audit(
                    user_id=user_id,
                    action=action,
                    resource=resource,
                    result="failure",
                    ip_address=kwargs.get('ip_address', ''),
                    user_agent=kwargs.get('user_agent', ''),
                    session_id=kwargs.get('session_id', ''),
                    request_id=kwargs.get('request_id', ''),
                    duration=duration,
                    details={"error": str(e)}
                )
                
                raise
        
        return wrapper
    return decorator





















