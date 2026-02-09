"""
Advanced Logging Optimizations

Optimizations for:
- Structured logging
- Log rotation
- Log compression
- Log aggregation
- Performance logging
"""

import logging
import logging.handlers
import gzip
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import sys

logger = logging.getLogger(__name__)


class StructuredLogger:
    """Structured logging with JSON format."""
    
    def __init__(
        self,
        name: str,
        log_file: Optional[str] = None,
        level: str = "INFO",
        json_format: bool = True
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            log_file: Log file path
            level: Log level
            json_format: Use JSON format
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if json_format:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            if json_format:
                file_handler.setFormatter(StructuredFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                )
            self.logger.addHandler(file_handler)
    
    def log(self, level: str, message: str, **kwargs) -> None:
        """
        Log message with extra fields.
        
        Args:
            level: Log level
            message: Log message
            **kwargs: Extra fields
        """
        getattr(self.logger, level.lower())(
            message,
            extra=kwargs
        )


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'message', 'pathname', 'process', 'processName', 'relativeCreated',
                          'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info']:
                log_data[key] = value
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class LogRotator:
    """Optimized log rotation."""
    
    def __init__(
        self,
        log_file: str,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        compress: bool = True
    ):
        """
        Initialize log rotator.
        
        Args:
            log_file: Log file path
            max_bytes: Maximum file size
            backup_count: Number of backup files
            compress: Compress rotated logs
        """
        self.log_file = Path(log_file)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.compress = compress
    
    def rotate_if_needed(self) -> None:
        """Rotate log file if needed."""
        if not self.log_file.exists():
            return
        
        if self.log_file.stat().st_size >= self.max_bytes:
            self._rotate()
    
    def _rotate(self) -> None:
        """Perform log rotation."""
        # Rotate existing backups
        for i in range(self.backup_count - 1, 0, -1):
            old_backup = self.log_file.with_suffix(f'.{i}.log')
            if self.compress:
                old_backup = old_backup.with_suffix('.log.gz')
            new_backup = self.log_file.with_suffix(f'.{i + 1}.log')
            if self.compress:
                new_backup = new_backup.with_suffix('.log.gz')
            
            if old_backup.exists():
                if i == self.backup_count - 1:
                    old_backup.unlink()  # Delete oldest
                else:
                    old_backup.rename(new_backup)
        
        # Rotate current log
        backup = self.log_file.with_suffix('.1.log')
        if self.compress:
            backup = backup.with_suffix('.log.gz')
            with open(self.log_file, 'rb') as f_in:
                with gzip.open(backup, 'wb') as f_out:
                    f_out.writelines(f_in)
        else:
            self.log_file.rename(backup)
        
        # Create new log file
        self.log_file.touch()


class PerformanceLogger:
    """Performance logging optimization."""
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize performance logger.
        
        Args:
            logger: Base logger
        """
        self.logger = logger
        self.metrics: Dict[str, List[float]] = {}
    
    def log_performance(
        self,
        operation: str,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log performance metric.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            metadata: Optional metadata
        """
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append(duration)
        
        # Keep only last 1000 metrics
        if len(self.metrics[operation]) > 1000:
            self.metrics[operation] = self.metrics[operation][-1000:]
        
        # Log if slow
        if duration > 1.0:
            self.logger.warning(
                f"Slow operation: {operation} took {duration:.3f}s",
                extra={
                    'operation': operation,
                    'duration': duration,
                    'metadata': metadata or {}
                }
            )
    
    def get_stats(self, operation: str) -> Optional[Dict[str, float]]:
        """Get performance statistics."""
        if operation not in self.metrics or not self.metrics[operation]:
            return None
        
        durations = self.metrics[operation]
        return {
            'count': len(durations),
            'avg': sum(durations) / len(durations),
            'min': min(durations),
            'max': max(durations),
            'total': sum(durations)
        }








