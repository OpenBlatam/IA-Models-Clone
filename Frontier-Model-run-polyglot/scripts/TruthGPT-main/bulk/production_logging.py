#!/usr/bin/env python3
"""
Production Logging - Production-ready logging system
Handles structured logging, log aggregation, and monitoring integration
"""

import logging
import logging.handlers
import json
import time
import traceback
import sys
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
import threading
import queue
import os
from enum import Enum

class LogLevel(Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogFormat(Enum):
    """Log formats."""
    JSON = "json"
    TEXT = "text"
    STRUCTURED = "structured"

@dataclass
class LogContext:
    """Logging context information."""
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    operation_id: Optional[str] = None
    model_name: Optional[str] = None
    optimization_strategy: Optional[str] = None
    batch_id: Optional[str] = None
    worker_id: Optional[str] = None
    thread_id: Optional[str] = None
    process_id: Optional[str] = None
    hostname: Optional[str] = None
    service_name: str = "bulk_optimization"
    version: str = "1.0.0"

@dataclass
class PerformanceMetrics:
    """Performance metrics for logging."""
    operation_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    batch_size: int = 0
    success_rate: float = 0.0
    error_count: int = 0
    warning_count: int = 0

class StructuredFormatter(logging.Formatter):
    """Structured log formatter."""
    
    def __init__(self, format_type: LogFormat = LogFormat.JSON):
        self.format_type = format_type
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record."""
        if self.format_type == LogFormat.JSON:
            return self._format_json(record)
        elif self.format_type == LogFormat.STRUCTURED:
            return self._format_structured(record)
        else:
            return self._format_text(record)
    
    def _format_json(self, record: logging.LogRecord) -> str:
        """Format as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process
        }
        
        # Add context if available
        if hasattr(record, 'context'):
            log_data.update(record.context)
        
        # Add performance metrics if available
        if hasattr(record, 'metrics'):
            log_data['metrics'] = asdict(record.metrics)
        
        # Add exception info if available
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_data, default=str)
    
    def _format_structured(self, record: logging.LogRecord) -> str:
        """Format as structured text."""
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        level = record.levelname
        logger_name = record.name
        message = record.getMessage()
        module = record.module
        function = record.funcName
        line = record.lineno
        
        structured = f"[{timestamp}] {level} {logger_name}:{module}.{function}:{line} - {message}"
        
        # Add context
        if hasattr(record, 'context'):
            context_str = " ".join([f"{k}={v}" for k, v in record.context.items() if v])
            if context_str:
                structured += f" | {context_str}"
        
        # Add metrics
        if hasattr(record, 'metrics'):
            metrics_str = " ".join([f"{k}={v}" for k, v in asdict(record.metrics).items() if v])
            if metrics_str:
                structured += f" | {metrics_str}"
        
        return structured
    
    def _format_text(self, record: logging.LogRecord) -> str:
        """Format as plain text."""
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        return f"{timestamp} - {record.levelname} - {record.name} - {record.getMessage()}"

class ProductionLogger:
    """Production logging system."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(name)
        self.context = LogContext()
        self.metrics = PerformanceMetrics()
        self.log_queue = queue.Queue()
        self.background_thread = None
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger configuration."""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        log_level = getattr(logging, self.config.get('log_level', 'INFO'))
        self.logger.setLevel(log_level)
        
        # Console handler
        if self.config.get('enable_console', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = StructuredFormatter(
                LogFormat(self.config.get('console_format', 'text'))
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.get('enable_file', True):
            log_file = self.config.get('log_file', 'bulk_optimization.log')
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.get('max_file_size', 10 * 1024 * 1024),
                backupCount=self.config.get('backup_count', 5)
            )
            file_formatter = StructuredFormatter(
                LogFormat(self.config.get('file_format', 'json'))
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Syslog handler
        if self.config.get('enable_syslog', False):
            syslog_handler = logging.handlers.SysLogHandler(
                address=self.config.get('syslog_address', '/dev/log')
            )
            syslog_formatter = StructuredFormatter(LogFormat.JSON)
            syslog_handler.setFormatter(syslog_formatter)
            self.logger.addHandler(syslog_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def set_context(self, **kwargs):
        """Set logging context."""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
    
    def set_metrics(self, **kwargs):
        """Set performance metrics."""
        for key, value in kwargs.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log with context and metrics."""
        # Create log record
        record = self.logger.makeRecord(
            self.logger.name, level, "", 0, message, (), None
        )
        
        # Add context
        record.context = asdict(self.context)
        
        # Add metrics
        record.metrics = asdict(self.metrics)
        
        # Add additional fields
        for key, value in kwargs.items():
            setattr(record, key, value)
        
        # Handle the record
        self.logger.handle(record)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def log_operation_start(self, operation_id: str, operation_type: str, **kwargs):
        """Log operation start."""
        self.set_context(operation_id=operation_id)
        self.info(f"Operation started: {operation_type}", operation_type=operation_type, **kwargs)
    
    def log_operation_end(self, operation_id: str, success: bool, duration: float, **kwargs):
        """Log operation end."""
        self.set_context(operation_id=operation_id)
        self.set_metrics(operation_time=duration)
        
        if success:
            self.info(f"Operation completed successfully in {duration:.2f}s", **kwargs)
        else:
            self.error(f"Operation failed after {duration:.2f}s", **kwargs)
    
    def log_optimization_start(self, model_name: str, strategy: str, **kwargs):
        """Log optimization start."""
        self.set_context(model_name=model_name, optimization_strategy=strategy)
        self.info(f"Optimization started for {model_name} with strategy {strategy}", **kwargs)
    
    def log_optimization_end(self, model_name: str, success: bool, metrics: Dict[str, Any], **kwargs):
        """Log optimization end."""
        self.set_context(model_name=model_name)
        self.set_metrics(**metrics)
        
        if success:
            self.info(f"Optimization completed for {model_name}", **kwargs)
        else:
            self.error(f"Optimization failed for {model_name}", **kwargs)
    
    def log_performance_metrics(self, metrics: Dict[str, Any], **kwargs):
        """Log performance metrics."""
        self.set_metrics(**metrics)
        self.info("Performance metrics", **kwargs)
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Log error with full context."""
        error_context = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'error_traceback': traceback.format_exc()
        }
        
        if context:
            error_context.update(context)
        
        self.error(f"Error occurred: {error}", **error_context, **kwargs)
    
    def log_system_metrics(self, **kwargs):
        """Log system metrics."""
        import psutil
        
        system_metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'process_count': len(psutil.pids())
        }
        
        if hasattr(psutil, 'sensors_temperatures'):
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    system_metrics['temperature'] = temps
            except:
                pass
        
        self.info("System metrics", **system_metrics, **kwargs)

class LogAggregator:
    """Log aggregation system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.aggregation_queue = queue.Queue()
        self.background_thread = None
        self.running = False
    
    def start(self):
        """Start log aggregation."""
        if not self.running:
            self.running = True
            self.background_thread = threading.Thread(target=self._aggregate_logs)
            self.background_thread.start()
    
    def stop(self):
        """Stop log aggregation."""
        self.running = False
        if self.background_thread:
            self.background_thread.join()
    
    def _aggregate_logs(self):
        """Aggregate logs in background."""
        while self.running:
            try:
                # Process log aggregation
                time.sleep(1)
            except Exception as e:
                logging.error(f"Log aggregation error: {e}")

def create_production_logger(name: str, config: Optional[Dict[str, Any]] = None) -> ProductionLogger:
    """Create production logger."""
    return ProductionLogger(name, config)

def setup_production_logging(config: Optional[Dict[str, Any]] = None):
    """Setup production logging system."""
    config = config or {}
    
    # Create main logger
    main_logger = create_production_logger("bulk_optimization", config)
    
    # Create component loggers
    optimization_logger = create_production_logger("bulk_optimization.optimization", config)
    data_logger = create_production_logger("bulk_optimization.data", config)
    operation_logger = create_production_logger("bulk_optimization.operation", config)
    
    return {
        'main': main_logger,
        'optimization': optimization_logger,
        'data': data_logger,
        'operation': operation_logger
    }

if __name__ == "__main__":
    # Example usage
    config = {
        'log_level': 'INFO',
        'enable_console': True,
        'enable_file': True,
        'console_format': 'text',
        'file_format': 'json',
        'max_file_size': 10 * 1024 * 1024,
        'backup_count': 5
    }
    
    logger = create_production_logger("example", config)
    
    # Set context
    logger.set_context(
        request_id="req_123",
        user_id="user_456",
        operation_id="op_789"
    )
    
    # Log messages
    logger.info("Application started")
    logger.log_operation_start("op_789", "optimization")
    logger.log_optimization_start("model_1", "memory")
    logger.log_optimization_end("model_1", True, {"operation_time": 1.5, "memory_usage": 100})
    logger.log_operation_end("op_789", True, 2.0)
    logger.log_performance_metrics({"cpu_usage": 50.0, "memory_usage": 200.0})
    logger.log_system_metrics()
    
    print("✅ Production logging example completed")

