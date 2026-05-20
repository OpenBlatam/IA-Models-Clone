"""
Advanced Logging System
=======================

Advanced logging system with rotation, filtering, and structured output.
"""

import logging
import logging.handlers
import json
import sys
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LogFormat(Enum):
    """Log formats."""
    JSON = "json"
    TEXT = "text"
    STRUCTURED = "structured"


@dataclass
class LogConfig:
    """Logging configuration."""
    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.STRUCTURED
    output_file: Optional[Path] = None
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_console: bool = True
    enable_file: bool = True
    filters: List[str] = field(default_factory=list)
    include_timestamp: bool = True
    include_level: bool = True
    include_module: bool = True
    include_function: bool = True
    include_line: bool = True


class StructuredFormatter(logging.Formatter):
    """Structured log formatter."""
    
    def __init__(self, config: LogConfig):
        """
        Initialize structured formatter.
        
        Args:
            config: Log configuration
        """
        super().__init__()
        self.config = config
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record.
        
        Args:
            record: Log record
            
        Returns:
            Formatted log string
        """
        if self.config.format == LogFormat.JSON:
            return self._format_json(record)
        elif self.config.format == LogFormat.STRUCTURED:
            return self._format_structured(record)
        else:
            return self._format_text(record)
    
    def _format_json(self, record: logging.LogRecord) -> str:
        """Format as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)
    
    def _format_structured(self, record: logging.LogRecord) -> str:
        """Format as structured text."""
        parts = []
        
        if self.config.include_timestamp:
            timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
            parts.append(f"[{timestamp}]")
        
        if self.config.include_level:
            parts.append(f"[{record.levelname}]")
        
        if self.config.include_module:
            parts.append(f"[{record.module}]")
        
        if self.config.include_function:
            parts.append(f"[{record.funcName}]")
        
        if self.config.include_line:
            parts.append(f"[L{record.lineno}]")
        
        parts.append(record.getMessage())
        
        # Add extra fields
        if hasattr(record, 'extra'):
            extra_str = " ".join(f"{k}={v}" for k, v in record.extra.items())
            if extra_str:
                parts.append(f"({extra_str})")
        
        # Add exception info
        if record.exc_info:
            parts.append(f"\n{self.formatException(record.exc_info)}")
        
        return " ".join(parts)
    
    def _format_text(self, record: logging.LogRecord) -> str:
        """Format as plain text."""
        return super().format(record)


class AdvancedLogger:
    """Advanced logger with rotation and filtering."""
    
    def __init__(self, name: str, config: LogConfig):
        """
        Initialize advanced logger.
        
        Args:
            name: Logger name
            config: Log configuration
        """
        self.name = name
        self.config = config
        self.logger = logging.getLogger(name)
        self.logger.setLevel(config.level.value)
        self.logger.handlers.clear()
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup log handlers."""
        formatter = StructuredFormatter(self.config)
        
        # Console handler
        if self.config.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.config.level.value)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.config.enable_file and self.config.output_file:
            file_handler = logging.handlers.RotatingFileHandler(
                self.config.output_file,
                maxBytes=self.config.max_bytes,
                backupCount=self.config.backup_count
            )
            file_handler.setLevel(self.config.level.value)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal log method."""
        # Apply filters
        if self.config.filters:
            if not any(filter_str in message for filter_str in self.config.filters):
                return
        
        # Add extra context
        extra = kwargs.copy()
        self.logger.log(level, message, extra=extra)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, extra=kwargs)
    
    def with_context(self, **context):
        """
        Create logger with additional context.
        
        Args:
            **context: Context variables
            
        Returns:
            Context logger
        """
        return ContextLogger(self.logger, context)


class ContextLogger:
    """Logger with additional context."""
    
    def __init__(self, logger: logging.Logger, context: Dict[str, Any]):
        """
        Initialize context logger.
        
        Args:
            logger: Base logger
            context: Context variables
        """
        self.logger = logger
        self.context = context
    
    def _add_context(self, kwargs: Dict[str, Any]):
        """Add context to kwargs."""
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        kwargs['extra'].update(self.context)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self._add_context(kwargs)
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self._add_context(kwargs)
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self._add_context(kwargs)
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context."""
        self._add_context(kwargs)
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with context."""
        self._add_context(kwargs)
        self.logger.critical(message, **kwargs)


class LogManager:
    """Manager for multiple loggers."""
    
    def __init__(self):
        """Initialize log manager."""
        self.loggers: Dict[str, AdvancedLogger] = {}
        self.default_config = LogConfig()
    
    def get_logger(self, name: str, config: Optional[LogConfig] = None) -> AdvancedLogger:
        """
        Get or create logger.
        
        Args:
            name: Logger name
            config: Optional log configuration
            
        Returns:
            Advanced logger
        """
        if name not in self.loggers:
            config = config or self.default_config
            self.loggers[name] = AdvancedLogger(name, config)
        return self.loggers[name]
    
    def set_default_config(self, config: LogConfig):
        """
        Set default log configuration.
        
        Args:
            config: Log configuration
        """
        self.default_config = config
    
    def get_all_loggers(self) -> Dict[str, AdvancedLogger]:
        """
        Get all loggers.
        
        Returns:
            Dictionary of logger name -> logger
        """
        return self.loggers.copy()



