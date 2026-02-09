#!/usr/bin/env python3
"""
Enhanced Error Handling and Logging System for Frontier Model Training
Provides comprehensive error handling, structured logging, and monitoring capabilities.
"""

import os
import sys
import logging
import traceback
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import functools
import threading
from contextlib import contextmanager
import psutil
import torch
import numpy as np

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as rich_traceback_install
from loguru import logger
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

console = Console()

class LogLevel(Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class ErrorType(Enum):
    """Error types for categorization."""
    CONFIGURATION = "configuration"
    MODEL_LOADING = "model_loading"
    TRAINING = "training"
    VALIDATION = "validation"
    MEMORY = "memory"
    CUDA = "cuda"
    NETWORK = "network"
    DATA = "data"
    UNKNOWN = "unknown"

@dataclass
class ErrorContext:
    """Context information for errors."""
    error_type: ErrorType
    component: str
    operation: str
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    additional_data: Dict[str, Any] = None

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    gpu_memory_used: float
    gpu_memory_total: float
    gpu_utilization: float
    training_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    batch_time: Optional[float] = None
    throughput: Optional[float] = None

class StructuredLogger:
    """Enhanced structured logger with error handling and monitoring."""
    
    def __init__(self, 
                 log_dir: str = "./logs",
                 log_level: LogLevel = LogLevel.INFO,
                 enable_file_logging: bool = True,
                 enable_console_logging: bool = True,
                 enable_sentry: bool = False,
                 sentry_dsn: Optional[str] = None):
        
        self.log_dir = Path(log_dir)
        self.log_level = log_level
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        self.enable_sentry = enable_sentry
        self.sentry_dsn = sentry_dsn
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Setup Sentry if enabled
        if self.enable_sentry and self.sentry_dsn:
            self._setup_sentry()
            
        # Performance monitoring
        self.performance_metrics: List[PerformanceMetrics] = []
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        
    def _setup_logging(self):
        """Setup logging configuration."""
        # Remove default handlers
        logger.remove()
        
        # Console logging with Rich
        if self.enable_console_logging:
            logger.add(
                sys.stdout,
                level=self.log_level.value,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                colorize=True,
                backtrace=True,
                diagnose=True
            )
        
        # File logging
        if self.enable_file_logging:
            # Main log file
            logger.add(
                self.log_dir / "frontier_model.log",
                level=self.log_level.value,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                rotation="100 MB",
                retention="30 days",
                compression="zip",
                backtrace=True,
                diagnose=True
            )
            
            # Error log file
            logger.add(
                self.log_dir / "errors.log",
                level="ERROR",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                rotation="50 MB",
                retention="90 days",
                compression="zip",
                backtrace=True,
                diagnose=True,
                filter=lambda record: record["level"].name == "ERROR"
            )
            
            # Performance log file
            logger.add(
                self.log_dir / "performance.log",
                level="INFO",
                format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
                rotation="50 MB",
                retention="7 days",
                compression="zip",
                filter=lambda record: "PERFORMANCE" in record["message"]
            )
    
    def _setup_sentry(self):
        """Setup Sentry for error tracking."""
        try:
            sentry_logging = LoggingIntegration(
                level=logging.INFO,
                event_level=logging.ERROR
            )
            
            sentry_sdk.init(
                self.sentry_dsn,
                integrations=[sentry_logging],
                traces_sample_rate=1.0,
                environment="development",
                before_send=self._before_send
            )
            
            logger.info("Sentry error tracking initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Sentry: {e}")
    
    def _before_send(self, event, hint):
        """Process events before sending to Sentry."""
        # Add custom context
        event['tags'] = {
            'component': 'frontier_model',
            'version': '1.0.0'
        }
        
        # Add performance metrics if available
        if self.performance_metrics:
            latest_metrics = self.performance_metrics[-1]
            event['extra']['performance'] = asdict(latest_metrics)
            
        return event
    
    def log_error(self, 
                  error: Exception, 
                  error_type: ErrorType = ErrorType.UNKNOWN,
                  component: str = "unknown",
                  operation: str = "unknown",
                  additional_data: Optional[Dict[str, Any]] = None):
        """Log an error with context."""
        error_context = ErrorContext(
            error_type=error_type,
            component=component,
            operation=operation,
            timestamp=datetime.now(timezone.utc),
            additional_data=additional_data or {}
        )
        
        # Log structured error
        logger.error(
            f"Error in {component}.{operation}: {str(error)}",
            extra={
                "error_context": asdict(error_context),
                "error_type": error_type.value,
                "traceback": traceback.format_exc()
            }
        )
        
        # Send to Sentry if enabled
        if self.enable_sentry:
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("error_type", error_type.value)
                scope.set_tag("component", component)
                scope.set_tag("operation", operation)
                scope.set_context("error_context", asdict(error_context))
                if additional_data:
                    scope.set_context("additional_data", additional_data)
                sentry_sdk.capture_exception(error)
    
    def log_performance(self, metrics: PerformanceMetrics):
        """Log performance metrics."""
        self.performance_metrics.append(metrics)
        
        # Keep only last 1000 metrics in memory
        if len(self.performance_metrics) > 1000:
            self.performance_metrics = self.performance_metrics[-1000:]
        
        logger.info(
            f"PERFORMANCE: CPU: {metrics.cpu_percent:.1f}%, "
            f"Memory: {metrics.memory_percent:.1f}%, "
            f"GPU: {metrics.gpu_memory_used:.1f}MB/{metrics.gpu_memory_total:.1f}MB "
            f"({metrics.gpu_utilization:.1f}%)",
            extra={"metrics": asdict(metrics)}
        )
    
    def start_performance_monitoring(self, interval: float = 10.0):
        """Start performance monitoring in background thread."""
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_performance,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Performance monitoring started with {interval}s interval")
    
    def stop_performance_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Performance monitoring stopped")
    
    def _monitor_performance(self, interval: float):
        """Background performance monitoring."""
        while self.monitoring_active:
            try:
                metrics = self._collect_performance_metrics()
                self.log_performance(metrics)
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                time.sleep(interval)
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        # GPU metrics
        gpu_memory_used = 0.0
        gpu_memory_total = 0.0
        gpu_utilization = 0.0
        
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
            gpu_utilization = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0
        
        return PerformanceMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gpu_utilization=gpu_utilization
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.performance_metrics:
            return {}
        
        cpu_values = [m.cpu_percent for m in self.performance_metrics]
        memory_values = [m.memory_percent for m in self.performance_metrics]
        gpu_memory_values = [m.gpu_memory_used for m in self.performance_metrics]
        
        return {
            "cpu": {
                "mean": np.mean(cpu_values),
                "max": np.max(cpu_values),
                "min": np.min(cpu_values),
                "std": np.std(cpu_values)
            },
            "memory": {
                "mean": np.mean(memory_values),
                "max": np.max(memory_values),
                "min": np.min(memory_values),
                "std": np.std(memory_values)
            },
            "gpu_memory": {
                "mean": np.mean(gpu_memory_values),
                "max": np.max(gpu_memory_values),
                "min": np.min(gpu_memory_values),
                "std": np.std(gpu_memory_values)
            },
            "total_samples": len(self.performance_metrics)
        }

class ErrorHandler:
    """Centralized error handling with recovery strategies."""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[ErrorType, Callable] = {}
        
        # Setup default recovery strategies
        self._setup_default_recovery_strategies()
    
    def _setup_default_recovery_strategies(self):
        """Setup default recovery strategies for common errors."""
        self.recovery_strategies[ErrorType.MEMORY] = self._handle_memory_error
        self.recovery_strategies[ErrorType.CUDA] = self._handle_cuda_error
        self.recovery_strategies[ErrorType.MODEL_LOADING] = self._handle_model_loading_error
        self.recovery_strategies[ErrorType.TRAINING] = self._handle_training_error
    
    def handle_error(self, 
                    error: Exception, 
                    error_type: ErrorType = ErrorType.UNKNOWN,
                    component: str = "unknown",
                    operation: str = "unknown",
                    additional_data: Optional[Dict[str, Any]] = None,
                    retry_count: int = 0,
                    max_retries: int = 3) -> bool:
        """Handle an error with recovery strategies."""
        
        # Log the error
        self.logger.log_error(error, error_type, component, operation, additional_data)
        
        # Update error counts
        error_key = f"{error_type.value}:{component}:{operation}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Check if we should retry
        if retry_count < max_retries and error_key in self.recovery_strategies:
            try:
                recovery_success = self.recovery_strategies[error_type](error, additional_data)
                if recovery_success:
                    self.logger.info(f"Recovery successful for {error_key}, retrying...")
                    return True
            except Exception as recovery_error:
                self.logger.log_error(
                    recovery_error, 
                    ErrorType.UNKNOWN, 
                    "error_handler", 
                    "recovery",
                    {"original_error": str(error)}
                )
        
        return False
    
    def _handle_memory_error(self, error: Exception, additional_data: Optional[Dict[str, Any]]) -> bool:
        """Handle memory-related errors."""
        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            self.logger.info("Memory cleanup completed")
            return True
        except Exception as e:
            self.logger.log_error(e, ErrorType.MEMORY, "error_handler", "memory_cleanup")
            return False
    
    def _handle_cuda_error(self, error: Exception, additional_data: Optional[Dict[str, Any]]) -> bool:
        """Handle CUDA-related errors."""
        try:
            # Reset CUDA context
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Reset device if possible
                if hasattr(torch.cuda, 'reset'):
                    torch.cuda.reset()
            
            self.logger.info("CUDA reset completed")
            return True
        except Exception as e:
            self.logger.log_error(e, ErrorType.CUDA, "error_handler", "cuda_reset")
            return False
    
    def _handle_model_loading_error(self, error: Exception, additional_data: Optional[Dict[str, Any]]) -> bool:
        """Handle model loading errors."""
        try:
            # Clear model cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            self.logger.info("Model loading cleanup completed")
            return True
        except Exception as e:
            self.logger.log_error(e, ErrorType.MODEL_LOADING, "error_handler", "model_cleanup")
            return False
    
    def _handle_training_error(self, error: Exception, additional_data: Optional[Dict[str, Any]]) -> bool:
        """Handle training errors."""
        try:
            # Clear gradients
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            self.logger.info("Training cleanup completed")
            return True
        except Exception as e:
            self.logger.log_error(e, ErrorType.TRAINING, "error_handler", "training_cleanup")
            return False

def error_handler(error_type: ErrorType = ErrorType.UNKNOWN, 
                 component: str = "unknown",
                 operation: str = "unknown",
                 max_retries: int = 3):
    """Decorator for automatic error handling."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger from args or create default
            logger_instance = None
            for arg in args:
                if isinstance(arg, StructuredLogger):
                    logger_instance = arg
                    break
            
            if not logger_instance:
                logger_instance = StructuredLogger()
            
            error_handler_instance = ErrorHandler(logger_instance)
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries:
                        success = error_handler_instance.handle_error(
                            e, error_type, component, operation, 
                            {"attempt": attempt + 1, "max_retries": max_retries},
                            attempt, max_retries
                        )
                        if not success:
                            break
                    else:
                        error_handler_instance.handle_error(
                            e, error_type, component, operation,
                            {"attempt": attempt + 1, "max_retries": max_retries}
                        )
                        raise
            return None
        return wrapper
    return decorator

@contextmanager
def error_context(error_type: ErrorType, component: str, operation: str):
    """Context manager for error handling."""
    logger_instance = StructuredLogger()
    error_handler_instance = ErrorHandler(logger_instance)
    
    try:
        yield logger_instance, error_handler_instance
    except Exception as e:
        error_handler_instance.handle_error(e, error_type, component, operation)
        raise

# Global logger instance
_global_logger: Optional[StructuredLogger] = None

def get_logger() -> StructuredLogger:
    """Get global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = StructuredLogger()
    return _global_logger

def setup_logging(log_dir: str = "./logs",
                 log_level: LogLevel = LogLevel.INFO,
                 enable_sentry: bool = False,
                 sentry_dsn: Optional[str] = None) -> StructuredLogger:
    """Setup global logging configuration."""
    global _global_logger
    _global_logger = StructuredLogger(
        log_dir=log_dir,
        log_level=log_level,
        enable_sentry=enable_sentry,
        sentry_dsn=sentry_dsn
    )
    return _global_logger

if __name__ == "__main__":
    # Example usage
    logger_instance = setup_logging(
        log_dir="./logs",
        log_level=LogLevel.INFO,
        enable_sentry=False
    )
    
    # Start performance monitoring
    logger_instance.start_performance_monitoring(interval=5.0)
    
    # Example error handling
    try:
        # Simulate an error
        raise ValueError("Test error")
    except Exception as e:
        logger_instance.log_error(
            e, 
            ErrorType.UNKNOWN, 
            "test_component", 
            "test_operation",
            {"test_data": "example"}
        )
    
    # Wait a bit for performance monitoring
    time.sleep(10)
    
    # Stop monitoring and show summary
    logger_instance.stop_performance_monitoring()
    summary = logger_instance.get_performance_summary()
    console.print(f"Performance Summary: {summary}")
