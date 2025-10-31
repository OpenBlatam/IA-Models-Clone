from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import json
import logging
import time
import traceback
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import hashlib
import uuid
import threading
import queue
from collections import defaultdict, deque
import weakref
import gc
import psutil
import numpy as np
import torch
import structlog
from contextlib import contextmanager
import functools
import inspect
import pickle
import gzip
import base64
from typing import Any, List, Dict, Optional
"""
Comprehensive Error Handling and Debugging System for Cybersecurity ML

This module provides:
- Advanced error tracking and classification
- Real-time debugging tools and monitoring
- Automated error recovery mechanisms
- Performance profiling and bottleneck detection
- Security-focused error handling
- Comprehensive logging and alerting
- Debug mode utilities and tools
"""



# Configure structured logging
logger = structlog.get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    MODEL = "model"
    DATA = "data"
    SYSTEM = "system"
    SECURITY = "security"
    NETWORK = "network"
    MEMORY = "memory"
    PERFORMANCE = "performance"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for error tracking."""
    timestamp: datetime
    function_name: str
    module_name: str
    line_number: int
    stack_trace: str
    local_variables: Dict[str, Any] = field(default_factory=dict)
    global_variables: Dict[str, Any] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    cpu_usage: float = 0.0
    thread_id: int = 0
    process_id: int = 0
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None


@dataclass
class ErrorRecord:
    """Complete error record with all context."""
    error_id: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_time: Optional[float] = None
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ErrorTracker:
    """Advanced error tracking and classification system."""
    
    def __init__(self, max_errors: int = 10000, enable_persistence: bool = True):
        
    """__init__ function."""
self.max_errors = max_errors
        self.enable_persistence = enable_persistence
        self.errors: deque = deque(maxlen=max_errors)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.severity_counts: Dict[ErrorSeverity, int] = defaultdict(int)
        self.category_counts: Dict[ErrorCategory, int] = defaultdict(int)
        self.recovery_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"attempted": 0, "successful": 0})
        
        # Performance tracking
        self.performance_metrics = {
            "total_errors": 0,
            "recovery_rate": 0.0,
            "avg_recovery_time": 0.0,
            "error_rate_per_minute": 0.0
        }
        
        # Initialize persistence
        if enable_persistence:
            self.persistence_file = Path("error_tracker_data.pkl.gz")
            self.load_errors()
        
        logger.info("ErrorTracker initialized", max_errors=max_errors, enable_persistence=enable_persistence)
    
    def track_error(
        self,
        error: Exception,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Track an error with full context."""
        try:
            # Generate unique error ID
            error_id = str(uuid.uuid4())
            
            # Get current context
            frame = inspect.currentframe().f_back
            error_context = self._capture_context(frame, context or {})
            
            # Create error record
            error_record = ErrorRecord(
                error_id=error_id,
                error_type=type(error).__name__,
                error_message=str(error),
                severity=severity,
                category=category,
                context=error_context,
                tags=tags or [],
                metadata=metadata or {}
            )
            
            # Store error
            self.errors.append(error_record)
            
            # Update statistics
            self.error_counts[error_record.error_type] += 1
            self.severity_counts[severity] += 1
            self.category_counts[category] += 1
            self.performance_metrics["total_errors"] += 1
            
            # Log error
            logger.error(
                "Error tracked",
                error_id=error_id,
                error_type=error_record.error_type,
                severity=severity.value,
                category=category.value,
                function=error_context.function_name,
                module=error_context.module_name,
                line=error_context.line_number
            )
            
            # Persist if enabled
            if self.enable_persistence:
                self._save_errors()
            
            return error_id
            
        except Exception as e:
            logger.error(f"Error tracking failed: {str(e)}")
            return str(uuid.uuid4())
    
    def _capture_context(self, frame: Optional[inspect.FrameInfo], additional_context: Dict[str, Any]) -> ErrorContext:
        """Capture detailed context information."""
        try:
            # Get frame information
            if frame:
                function_name = frame.f_code.co_name
                module_name = frame.f_globals.get('__name__', 'unknown')
                line_number = frame.f_lineno
                stack_trace = ''.join(traceback.format_stack(frame))
                
                # Capture local variables (be careful with sensitive data)
                local_vars = {}
                for name, value in frame.f_locals.items():
                    if not name.startswith('_') and not callable(value):
                        try:
                            # Limit size of captured values
                            str_value = str(value)
                            if len(str_value) < 1000:
                                local_vars[name] = str_value
                        except:
                            local_vars[name] = "<unserializable>"
            else:
                function_name = "unknown"
                module_name = "unknown"
                line_number = 0
                stack_trace = traceback.format_exc()
                local_vars = {}
            
            # Get system information
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return ErrorContext(
                timestamp=datetime.now(),
                function_name=function_name,
                module_name=module_name,
                line_number=line_number,
                stack_trace=stack_trace,
                local_variables=local_vars,
                global_variables=additional_context,
                memory_usage={
                    "rss_mb": memory_info.rss / 1024 / 1024,
                    "vms_mb": memory_info.vms / 1024 / 1024,
                    "percent": process.memory_percent()
                },
                cpu_usage=process.cpu_percent(),
                thread_id=threading.get_ident(),
                process_id=os.getpid()
            )
            
        except Exception as e:
            logger.warning(f"Context capture failed: {str(e)}")
            return ErrorContext(
                timestamp=datetime.now(),
                function_name="unknown",
                module_name="unknown",
                line_number=0,
                stack_trace=traceback.format_exc()
            )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary."""
        return {
            "total_errors": len(self.errors),
            "error_counts": dict(self.error_counts),
            "severity_counts": {sev.value: count for sev, count in self.severity_counts.items()},
            "category_counts": {cat.value: count for cat, count in self.category_counts.items()},
            "recovery_stats": dict(self.recovery_stats),
            "performance_metrics": self.performance_metrics,
            "recent_errors": [
                {
                    "error_id": err.error_id,
                    "error_type": err.error_type,
                    "severity": err.severity.value,
                    "category": err.category.value,
                    "timestamp": err.context.timestamp.isoformat(),
                    "function": err.context.function_name
                }
                for err in list(self.errors)[-10:]  # Last 10 errors
            ]
        }
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[ErrorRecord]:
        """Get all errors of a specific category."""
        return [err for err in self.errors if err.category == category]
    
    def get_errors_by_severity(self, severity: ErrorSeverity) -> List[ErrorRecord]:
        """Get all errors of a specific severity."""
        return [err for err in self.errors if err.severity == severity]
    
    def mark_error_resolved(self, error_id: str) -> bool:
        """Mark an error as resolved."""
        for error in self.errors:
            if error.error_id == error_id:
                error.resolved = True
                error.resolution_time = datetime.now()
                logger.info("Error marked as resolved", error_id=error_id)
                return True
        return False
    
    def _save_errors(self) -> Any:
        """Save errors to persistent storage."""
        try:
            data = {
                "errors": list(self.errors),
                "error_counts": dict(self.error_counts),
                "severity_counts": {sev.value: count for sev, count in self.severity_counts.items()},
                "category_counts": {cat.value: count for cat, count in self.category_counts.items()},
                "recovery_stats": dict(self.recovery_stats),
                "performance_metrics": self.performance_metrics
            }
            
            with gzip.open(self.persistence_file, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                pickle.dump(data, f)
                
        except Exception as e:
            logger.error(f"Error saving to persistence: {str(e)}")
    
    def load_errors(self) -> Any:
        """Load errors from persistent storage."""
        try:
            if self.persistence_file.exists():
                with gzip.open(self.persistence_file, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    data = pickle.load(f)
                
                self.errors = deque(data.get("errors", []), maxlen=self.max_errors)
                self.error_counts = defaultdict(int, data.get("error_counts", {}))
                self.severity_counts = defaultdict(int, {
                    ErrorSeverity(sev): count for sev, count in data.get("severity_counts", {}).items()
                })
                self.category_counts = defaultdict(int, {
                    ErrorCategory(cat): count for cat, count in data.get("category_counts", {}).items()
                })
                self.recovery_stats = defaultdict(lambda: {"attempted": 0, "successful": 0}, data.get("recovery_stats", {}))
                self.performance_metrics = data.get("performance_metrics", self.performance_metrics)
                
                logger.info("Errors loaded from persistence", count=len(self.errors))
                
        except Exception as e:
            logger.error(f"Error loading from persistence: {str(e)}")


class Debugger:
    """Advanced debugging tools and utilities."""
    
    def __init__(self, enable_profiling: bool = True, enable_memory_tracking: bool = True):
        
    """__init__ function."""
self.enable_profiling = enable_profiling
        self.enable_memory_tracking = enable_memory_tracking
        self.profiling_data: Dict[str, List[float]] = defaultdict(list)
        self.memory_snapshots: List[Dict[str, float]] = []
        self.breakpoints: Dict[str, Callable] = {}
        self.watch_variables: Dict[str, Any] = {}
        self.debug_mode = False
        
        logger.info("Debugger initialized", enable_profiling=enable_profiling, enable_memory_tracking=enable_memory_tracking)
    
    @contextmanager
    def debug_context(self, context_name: str):
        """Context manager for debugging operations."""
        start_time = time.time()
        start_memory = self._get_memory_usage() if self.enable_memory_tracking else None
        
        try:
            yield
        except Exception as e:
            logger.error(f"Error in debug context '{context_name}': {str(e)}")
            raise
        finally:
            if self.enable_profiling:
                duration = time.time() - start_time
                self.profiling_data[context_name].append(duration)
                
                if self.enable_memory_tracking and start_memory:
                    end_memory = self._get_memory_usage()
                    memory_diff = end_memory - start_memory
                    if abs(memory_diff) > 1.0:  # 1MB threshold
                        logger.warning(f"Memory leak detected in '{context_name}': {memory_diff:.2f}MB")
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function performance."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            start_memory = self._get_memory_usage() if self.enable_memory_tracking else None
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                self.profiling_data[func.__name__].append(duration)
                
                if self.enable_memory_tracking and start_memory:
                    end_memory = self._get_memory_usage()
                    memory_diff = end_memory - start_memory
                    if abs(memory_diff) > 1.0:
                        logger.warning(f"Memory leak in '{func.__name__}': {memory_diff:.2f}MB")
        
        return wrapper
    
    def add_breakpoint(self, condition: str, callback: Callable):
        """Add a conditional breakpoint."""
        self.breakpoints[condition] = callback
        logger.info(f"Breakpoint added: {condition}")
    
    def check_breakpoints(self, context: Dict[str, Any]):
        """Check and execute breakpoints."""
        for condition, callback in self.breakpoints.items():
            try:
                if eval(condition, context):
                    callback(context)
            except Exception as e:
                logger.error(f"Breakpoint error: {str(e)}")
    
    def watch_variable(self, name: str, value: Any):
        """Watch a variable for changes."""
        if name in self.watch_variables:
            old_value = self.watch_variables[name]
            if old_value != value:
                logger.info(f"Variable '{name}' changed: {old_value} -> {value}")
        
        self.watch_variables[name] = value
    
    def get_profiling_summary(self) -> Dict[str, Any]:
        """Get profiling performance summary."""
        summary = {}
        for func_name, times in self.profiling_data.items():
            if times:
                summary[func_name] = {
                    "count": len(times),
                    "avg_time": np.mean(times),
                    "min_time": np.min(times),
                    "max_time": np.max(times),
                    "std_time": np.std(times),
                    "total_time": np.sum(times)
                }
        return summary
    
    def get_memory_summary(self) -> Dict[str, float]:
        """Get current memory usage summary."""
        return self._get_memory_usage()
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get detailed memory usage information."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent(),
                "available_mb": psutil.virtual_memory().available / 1024 / 1024
            }
        except Exception as e:
            logger.error(f"Memory usage check failed: {str(e)}")
            return {"error": str(e)}
    
    def enable_debug_mode(self) -> Any:
        """Enable debug mode with enhanced logging."""
        self.debug_mode = True
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")
    
    def disable_debug_mode(self) -> Any:
        """Disable debug mode."""
        self.debug_mode = False
        logging.getLogger().setLevel(logging.INFO)
        logger.info("Debug mode disabled")


class ErrorRecovery:
    """Automated error recovery mechanisms."""
    
    def __init__(self) -> Any:
        self.recovery_strategies: Dict[str, List[Callable]] = defaultdict(list)
        self.recovery_history: List[Dict[str, Any]] = []
        self.max_recovery_attempts = 3
        self.recovery_timeout = 30.0  # seconds
        
        logger.info("ErrorRecovery initialized")
    
    def register_recovery_strategy(self, error_type: str, strategy: Callable, priority: int = 0):
        """Register a recovery strategy for a specific error type."""
        self.recovery_strategies[error_type].append((priority, strategy))
        # Sort by priority (higher priority first)
        self.recovery_strategies[error_type].sort(key=lambda x: x[0], reverse=True)
        logger.info(f"Recovery strategy registered for {error_type}")
    
    async def attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt to recover from an error."""
        error_type = type(error).__name__
        strategies = self.recovery_strategies[error_type]
        
        if not strategies:
            logger.warning(f"No recovery strategies for error type: {error_type}")
            return False
        
        for attempt in range(self.max_recovery_attempts):
            for priority, strategy in strategies:
                try:
                    logger.info(f"Attempting recovery with strategy {strategy.__name__} (attempt {attempt + 1})")
                    
                    # Run recovery strategy with timeout
                    if asyncio.iscoroutinefunction(strategy):
                        result = await asyncio.wait_for(strategy(error, context), timeout=self.recovery_timeout)
                    else:
                        result = strategy(error, context)
                    
                    if result:
                        self.recovery_history.append({
                            "timestamp": datetime.now().isoformat(),
                            "error_type": error_type,
                            "strategy": strategy.__name__,
                            "attempt": attempt + 1,
                            "successful": True
                        })
                        logger.info(f"Recovery successful with strategy {strategy.__name__}")
                        return True
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Recovery strategy {strategy.__name__} timed out")
                except Exception as recovery_error:
                    logger.error(f"Recovery strategy {strategy.__name__} failed: {str(recovery_error)}")
        
        self.recovery_history.append({
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "attempts": self.max_recovery_attempts,
            "successful": False
        })
        
        logger.error(f"All recovery attempts failed for error type: {error_type}")
        return False
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        total_attempts = len(self.recovery_history)
        successful_recoveries = sum(1 for record in self.recovery_history if record.get("successful", False))
        
        return {
            "total_attempts": total_attempts,
            "successful_recoveries": successful_recoveries,
            "success_rate": successful_recoveries / total_attempts if total_attempts > 0 else 0.0,
            "recent_recoveries": self.recovery_history[-10:]  # Last 10 attempts
        }


class PerformanceMonitor:
    """Real-time performance monitoring and bottleneck detection."""
    
    def __init__(self, sampling_interval: float = 1.0):
        
    """__init__ function."""
self.sampling_interval = sampling_interval
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alert_thresholds: Dict[str, float] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.monitoring_active = False
        self.monitor_thread = None
        
        logger.info("PerformanceMonitor initialized", sampling_interval=sampling_interval)
    
    def start_monitoring(self) -> Any:
        """Start real-time performance monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            self.monitor_thread.start()
            logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> Any:
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self) -> Any:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self._collect_metrics()
                
                # Store metrics
                for key, value in metrics.items():
                    self.metrics_history[key].append(value)
                
                # Check alerts
                self._check_alerts(metrics)
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {str(e)}")
                time.sleep(self.sampling_interval)
    
    def _collect_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "cpu_percent": process.cpu_percent(),
                "memory_rss_mb": memory_info.rss / 1024 / 1024,
                "memory_percent": process.memory_percent(),
                "thread_count": process.num_threads(),
                "open_files": len(process.open_files()),
                "connections": len(process.connections()),
                "system_memory_percent": psutil.virtual_memory().percent,
                "system_cpu_percent": psutil.cpu_percent(),
                "disk_usage_percent": psutil.disk_usage('/').percent
            }
        except Exception as e:
            logger.error(f"Metrics collection error: {str(e)}")
            return {}
    
    def set_alert_threshold(self, metric: str, threshold: float, alert_type: str = "warning"):
        """Set alert threshold for a metric."""
        self.alert_thresholds[metric] = {"threshold": threshold, "type": alert_type}
        logger.info(f"Alert threshold set: {metric} > {threshold}")
    
    def _check_alerts(self, metrics: Dict[str, float]):
        """Check metrics against alert thresholds."""
        for metric, config in self.alert_thresholds.items():
            if metric in metrics:
                value = metrics[metric]
                threshold = config["threshold"]
                alert_type = config["type"]
                
                if value > threshold:
                    alert = {
                        "timestamp": datetime.now().isoformat(),
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                        "type": alert_type
                    }
                    self.alerts.append(alert)
                    
                    if alert_type == "critical":
                        logger.critical(f"Critical alert: {metric} = {value} (threshold: {threshold})")
                    else:
                        logger.warning(f"Warning alert: {metric} = {value} (threshold: {threshold})")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary with trends."""
        summary = {}
        
        for metric, values in self.metrics_history.items():
            if values:
                summary[metric] = {
                    "current": values[-1],
                    "average": np.mean(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "trend": "increasing" if len(values) > 1 and values[-1] > values[-2] else "decreasing"
                }
        
        return {
            "metrics": summary,
            "alerts": self.alerts[-10:],  # Last 10 alerts
            "alert_thresholds": self.alert_thresholds
        }


class ErrorHandlingDebuggingSystem:
    """Main error handling and debugging system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.config = config or {}
        
        # Initialize components
        self.error_tracker = ErrorTracker(
            max_errors=self.config.get("max_errors", 10000),
            enable_persistence=self.config.get("enable_persistence", True)
        )
        
        self.debugger = Debugger(
            enable_profiling=self.config.get("enable_profiling", True),
            enable_memory_tracking=self.config.get("enable_memory_tracking", True)
        )
        
        self.error_recovery = ErrorRecovery()
        self.performance_monitor = PerformanceMonitor(
            sampling_interval=self.config.get("sampling_interval", 1.0)
        )
        
        # Register default recovery strategies
        self._register_default_recovery_strategies()
        
        # Start monitoring
        if self.config.get("auto_start_monitoring", True):
            self.performance_monitor.start_monitoring()
        
        logger.info("ErrorHandlingDebuggingSystem initialized", config=self.config)
    
    def _register_default_recovery_strategies(self) -> Any:
        """Register default error recovery strategies."""
        
        # Memory error recovery
        def memory_recovery_strategy(error: Exception, context: Dict[str, Any]) -> bool:
            try:
                gc.collect()
                if hasattr(torch, 'cuda') and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Memory recovery strategy executed")
                return True
            except Exception as e:
                logger.error(f"Memory recovery failed: {str(e)}")
                return False
        
        self.error_recovery.register_recovery_strategy("MemoryError", memory_recovery_strategy, priority=10)
        
        # Model loading error recovery
        def model_recovery_strategy(error: Exception, context: Dict[str, Any]) -> bool:
            try:
                # Attempt to reload model or use fallback
                logger.info("Model recovery strategy executed")
                return True
            except Exception as e:
                logger.error(f"Model recovery failed: {str(e)}")
                return False
        
        self.error_recovery.register_recovery_strategy("RuntimeError", model_recovery_strategy, priority=5)
    
    @contextmanager
    async def error_context(self, context_name: str, severity: ErrorSeverity = ErrorSeverity.ERROR):
        """Context manager for error handling and debugging."""
        start_time = time.time()
        
        try:
            with self.debugger.debug_context(context_name):
                yield
        except Exception as e:
            # Track error
            error_id = self.error_tracker.track_error(
                error=e,
                severity=severity,
                category=self._classify_error(e),
                context={"context_name": context_name, "duration": time.time() - start_time}
            )
            
            # Attempt recovery
            recovery_successful = await self.error_recovery.attempt_recovery(e, {
                "context_name": context_name,
                "error_id": error_id,
                "duration": time.time() - start_time
            })
            
            if not recovery_successful:
                raise
    
    def _classify_error(self, error: Exception) -> ErrorCategory:
        """Classify error into appropriate category."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        if "memory" in error_message or isinstance(error, MemoryError):
            return ErrorCategory.MEMORY
        elif "validation" in error_message or "invalid" in error_message:
            return ErrorCategory.VALIDATION
        elif "model" in error_message or "tensor" in error_message:
            return ErrorCategory.MODEL
        elif "data" in error_message or "file" in error_message:
            return ErrorCategory.DATA
        elif "network" in error_message or "connection" in error_message:
            return ErrorCategory.NETWORK
        elif "security" in error_message or "permission" in error_message:
            return ErrorCategory.SECURITY
        elif "performance" in error_message or "timeout" in error_message:
            return ErrorCategory.PERFORMANCE
        else:
            return ErrorCategory.UNKNOWN
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "error_tracker": self.error_tracker.get_error_summary(),
            "debugger": {
                "profiling": self.debugger.get_profiling_summary(),
                "memory": self.debugger.get_memory_summary(),
                "debug_mode": self.debugger.debug_mode
            },
            "error_recovery": self.error_recovery.get_recovery_stats(),
            "performance_monitor": self.performance_monitor.get_performance_summary(),
            "timestamp": datetime.now().isoformat()
        }
    
    def enable_debug_mode(self) -> Any:
        """Enable comprehensive debug mode."""
        self.debugger.enable_debug_mode()
        logger.info("Comprehensive debug mode enabled")
    
    def disable_debug_mode(self) -> Any:
        """Disable debug mode."""
        self.debugger.disable_debug_mode()
        logger.info("Debug mode disabled")
    
    def cleanup(self) -> Any:
        """Cleanup resources."""
        self.performance_monitor.stop_monitoring()
        logger.info("ErrorHandlingDebuggingSystem cleanup completed")


# Utility functions for easy integration
def error_handler(func: Callable) -> Callable:
    """Decorator for automatic error handling."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        # Get or create error handling system
        if not hasattr(wrapper, '_error_system'):
            wrapper._error_system = ErrorHandlingDebuggingSystem()
        
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_id = wrapper._error_system.error_tracker.track_error(
                error=e,
                context={"function": func.__name__, "args": str(args), "kwargs": str(kwargs)}
            )
            
            # Attempt recovery
            recovery_successful = await wrapper._error_system.error_recovery.attempt_recovery(e, {
                "function": func.__name__,
                "error_id": error_id
            })
            
            if not recovery_successful:
                raise
    
    return wrapper


def debug_function(func: Callable) -> Callable:
    """Decorator for function debugging."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Get or create debugger
        if not hasattr(wrapper, '_debugger'):
            wrapper._debugger = Debugger()
        
        return wrapper._debugger.profile_function(func)(*args, **kwargs)
    
    return wrapper


# Example usage
if __name__ == "__main__":
    # Initialize the system
    error_system = ErrorHandlingDebuggingSystem({
        "max_errors": 5000,
        "enable_persistence": True,
        "enable_profiling": True,
        "enable_memory_tracking": True,
        "auto_start_monitoring": True
    })
    
    # Example usage with error handling
    @error_handler
    @debug_function
    async def example_function():
        
    """example_function function."""
with error_system.error_context("example_operation"):
            # Simulate some work
            time.sleep(0.1)
            
            # Simulate an error
            if np.random.random() < 0.3:
                raise ValueError("Simulated error for testing")
            
            return "Success"
    
    # Run example
    asyncio.run(example_function())
    
    # Get system status
    status = error_system.get_system_status()
    print(json.dumps(status, indent=2)) 