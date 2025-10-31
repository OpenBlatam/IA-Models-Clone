#!/usr/bin/env python3
"""
Logging and Monitoring Service
==============================

Comprehensive logging and monitoring system for the HeyGen AI system with:
- Structured logging with different levels
- Performance metrics collection
- Health monitoring
- Error tracking and reporting
- Log aggregation and analysis
"""

import logging
import logging.handlers
import time
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from contextlib import contextmanager
import threading
import queue
import traceback

# =============================================================================
# Logging Models
# =============================================================================

@dataclass
class LogEntry:
    """Structured log entry."""
    
    timestamp: datetime
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    extra_data: Dict[str, Any] = field(default_factory=dict)
    exception_info: Optional[str] = None

@dataclass
class PerformanceMetric:
    """Performance metric entry."""
    
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthStatus:
    """System health status."""
    
    component: str
    status: str  # healthy, warning, critical, unknown
    message: str
    timestamp: datetime
    metrics: Dict[str, Any] = field(default_factory=dict)
    last_check: datetime
    check_duration: float

# =============================================================================
# Custom Logging Handlers
# =============================================================================

class StructuredJSONHandler(logging.Handler):
    """Handler that outputs structured JSON logs."""
    
    def __init__(self, filename: str, max_bytes: int = 10*1024*1024, backup_count: int = 5):
        super().__init__()
        self.handler = logging.handlers.RotatingFileHandler(
            filename, maxBytes=max_bytes, backupCount=backup_count
        )
        self.handler.setFormatter(logging.Formatter())
    
    def emit(self, record):
        try:
            # Create structured log entry
            log_entry = LogEntry(
                timestamp=datetime.fromtimestamp(record.created),
                level=record.levelname,
                logger_name=record.name,
                message=record.getMessage(),
                module=record.module,
                function=record.funcName,
                line_number=record.lineno,
                thread_id=record.thread,
                process_id=record.process,
                extra_data=getattr(record, 'extra_data', {}),
                exception_info=self.formatException(record.exc_info) if record.exc_info else None
            )
            
            # Convert to JSON and write
            json_log = json.dumps(asdict(log_entry), default=str, ensure_ascii=False)
            self.handler.emit(logging.LogRecord(
                name=record.name,
                level=record.levelno,
                pathname=record.pathname,
                lineno=record.lineno,
                msg=json_log,
                args=(),
                exc_info=record.exc_info
            ))
            
        except Exception:
            self.handleError(record)

class PerformanceLoggingHandler(logging.Handler):
    """Handler for logging performance metrics."""
    
    def __init__(self, metrics_queue: queue.Queue):
        super().__init__()
        self.metrics_queue = metrics_queue
    
    def emit(self, record):
        try:
            if hasattr(record, 'performance_metric'):
                self.metrics_queue.put(record.performance_metric)
        except Exception:
            self.handleError(record)

# =============================================================================
# Performance Monitoring
# =============================================================================

class PerformanceMonitor:
    """Monitors and tracks performance metrics."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self.metrics_lock = threading.Lock()
        self.start_time = time.time()
    
    def record_metric(self, name: str, value: float, unit: str = "", 
                     tags: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            tags=tags or {},
            metadata=metadata or {}
        )
        
        with self.metrics_lock:
            self.metrics.append(metric)
    
    @contextmanager
    def measure_time(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for measuring operation time."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_metric(
                name=f"{operation_name}_duration",
                value=duration,
                unit="seconds",
                tags=tags
            )
    
    def get_metrics(self, name: Optional[str] = None, 
                   since: Optional[datetime] = None) -> List[PerformanceMetric]:
        """Get metrics filtered by name and time."""
        with self.metrics_lock:
            filtered_metrics = self.metrics
            
            if name:
                filtered_metrics = [m for m in filtered_metrics if m.name == name]
            
            if since:
                filtered_metrics = [m for m in filtered_metrics if m.timestamp >= since]
            
            return filtered_metrics.copy()
    
    def get_statistics(self, name: str, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get statistical summary of metrics."""
        metrics = self.get_metrics(name, since)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "latest": values[-1] if values else None
        }
    
    def clear_old_metrics(self, older_than_hours: int = 24):
        """Clear metrics older than specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        with self.metrics_lock:
            self.metrics = [m for m in self.metrics if m.timestamp > cutoff_time]

# =============================================================================
# Health Monitoring
# =============================================================================

class HealthMonitor:
    """Monitors system health and component status."""
    
    def __init__(self):
        self.components: Dict[str, HealthStatus] = {}
        self.health_checks: Dict[str, Callable] = {}
        self.monitoring_thread = None
        self.stop_monitoring = False
        self.check_interval = 30  # seconds
    
    def register_component(self, component_name: str, health_check_func: Callable):
        """Register a component for health monitoring."""
        self.health_checks[component_name] = health_check_func
        
        # Initialize component status
        self.components[component_name] = HealthStatus(
            component=component_name,
            status="unknown",
            message="Component registered, not yet checked",
            timestamp=datetime.now(),
            metrics={},
            last_check=datetime.now(),
            check_duration=0.0
        )
    
    def start_monitoring(self):
        """Start the health monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.stop_monitoring = False
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop the health monitoring thread."""
        self.stop_monitoring = True
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_monitoring:
            try:
                self._check_all_components()
                time.sleep(self.check_interval)
            except Exception as e:
                # Log error but continue monitoring
                print(f"Health monitoring error: {e}")
    
    def _check_all_components(self):
        """Check health of all registered components."""
        for component_name, health_check_func in self.health_checks.items():
            try:
                start_time = time.time()
                
                # Run health check
                result = health_check_func()
                
                check_duration = time.time() - start_time
                
                # Update component status
                self.components[component_name] = HealthStatus(
                    component=component_name,
                    status=result.get("status", "unknown"),
                    message=result.get("message", "Health check completed"),
                    timestamp=datetime.now(),
                    metrics=result.get("metrics", {}),
                    last_check=datetime.now(),
                    check_duration=check_duration
                )
                
            except Exception as e:
                # Mark component as critical if health check fails
                self.components[component_name] = HealthStatus(
                    component=component_name,
                    status="critical",
                    message=f"Health check failed: {str(e)}",
                    timestamp=datetime.now(),
                    metrics={},
                    last_check=datetime.now(),
                    check_duration=0.0
                )
    
    def get_component_health(self, component_name: str) -> Optional[HealthStatus]:
        """Get health status of a specific component."""
        return self.components.get(component_name)
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        if not self.components:
            return {"status": "unknown", "message": "No components registered"}
        
        # Count statuses
        status_counts = {}
        for status in self.components.values():
            status_counts[status.status] = status_counts.get(status.status, 0) + 1
        
        # Determine overall status
        if status_counts.get("critical", 0) > 0:
            overall_status = "critical"
        elif status_counts.get("warning", 0) > 0:
            overall_status = "warning"
        elif status_counts.get("healthy", 0) == len(self.components):
            overall_status = "healthy"
        else:
            overall_status = "unknown"
        
        return {
            "status": overall_status,
            "component_count": len(self.components),
            "status_breakdown": status_counts,
            "components": {name: asdict(status) for name, status in self.components.items()}
        }

# =============================================================================
# Main Logging Service
# =============================================================================

class LoggingService:
    """Main logging service that orchestrates all logging and monitoring."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = None
        self.performance_monitor = PerformanceMonitor()
        self.health_monitor = HealthMonitor()
        self.metrics_queue = queue.Queue()
        
        # Initialize logging
        self._setup_logging()
        
        # Start health monitoring
        self.health_monitor.start_monitoring()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Create main logger
        self.logger = logging.getLogger("heygen_ai")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        
        # File handler for all logs
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "heygen_ai.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        
        # JSON handler for structured logging
        json_handler = StructuredJSONHandler(
            log_dir / "heygen_ai_structured.json"
        )
        json_handler.setLevel(logging.INFO)
        self.logger.addHandler(json_handler)
        
        # Performance metrics handler
        perf_handler = PerformanceLoggingHandler(self.metrics_queue)
        perf_handler.setLevel(logging.INFO)
        self.logger.addHandler(perf_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name."""
        return logging.getLogger(f"heygen_ai.{name}")
    
    def log_performance(self, name: str, value: float, unit: str = "",
                       tags: Optional[Dict[str, str]] = None,
                       metadata: Optional[Dict[str, Any]] = None):
        """Log a performance metric."""
        # Record in performance monitor
        self.performance_monitor.record_metric(name, value, unit, tags, metadata)
        
        # Also log to logger
        logger = self.get_logger("performance")
        logger.info(f"Performance metric: {name}={value}{unit}", 
                   extra={"performance_metric": {
                       "name": name, "value": value, "unit": unit,
                       "tags": tags, "metadata": metadata
                   }})
    
    @contextmanager
    def measure_operation(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for measuring operation performance."""
        with self.performance_monitor.measure_time(operation_name, tags):
            yield
    
    def register_health_check(self, component_name: str, health_check_func: Callable):
        """Register a component for health monitoring."""
        self.health_monitor.register_component(component_name, health_check_func)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current system health status."""
        return self.health_monitor.get_overall_health()
    
    def get_component_health(self, component_name: str) -> Optional[HealthStatus]:
        """Get health status of a specific component."""
        return self.health_monitor.get_component_health(component_name)
    
    def get_performance_metrics(self, name: Optional[str] = None,
                              since: Optional[datetime] = None) -> List[PerformanceMetric]:
        """Get performance metrics."""
        return self.performance_monitor.get_metrics(name, since)
    
    def get_performance_statistics(self, name: str, 
                                 since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_monitor.get_statistics(name, since)
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log an error with context."""
        logger = self.get_logger("error")
        
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }
        
        logger.error(f"Error occurred: {error_info['error_message']}", 
                    extra={"error_info": error_info})
    
    def log_operation_start(self, operation: str, **kwargs):
        """Log the start of an operation."""
        logger = self.get_logger("operations")
        logger.info(f"Operation started: {operation}", extra={"operation": operation, **kwargs})
    
    def log_operation_complete(self, operation: str, duration: float, **kwargs):
        """Log the completion of an operation."""
        logger = self.get_logger("operations")
        logger.info(f"Operation completed: {operation} in {duration:.2f}s", 
                   extra={"operation": operation, "duration": duration, **kwargs})
    
    def cleanup(self):
        """Cleanup logging service resources."""
        try:
            # Stop health monitoring
            self.health_monitor.stop_monitoring()
            
            # Clear old metrics
            self.performance_monitor.clear_old_metrics()
            
            # Close all handlers
            for handler in self.logger.handlers:
                handler.close()
            
            self.logger.info("Logging service cleanup completed")
            
        except Exception as e:
            print(f"Error during logging service cleanup: {e}")

# =============================================================================
# Example Usage
# =============================================================================

def main():
    """Example usage of the logging service."""
    try:
        # Create logging service
        logging_service = LoggingService()
        
        # Get logger
        logger = logging_service.get_logger("example")
        
        # Log some messages
        logger.info("Application started")
        logger.debug("Debug information")
        logger.warning("Warning message")
        
        # Log performance metrics
        logging_service.log_performance("api_response_time", 0.15, "seconds", {"endpoint": "/generate"})
        logging_service.log_performance("memory_usage", 512.5, "MB", {"component": "avatar_generator"})
        
        # Measure operation time
        with logging_service.measure_operation("example_operation", {"type": "test"}):
            time.sleep(0.1)  # Simulate work
        
        # Register health check
        def example_health_check():
            return {
                "status": "healthy",
                "message": "Example component is working",
                "metrics": {"uptime": 3600}
            }
        
        logging_service.register_health_check("example_component", example_health_check)
        
        # Wait for health check
        time.sleep(2)
        
        # Get health status
        health = logging_service.get_health_status()
        print(f"Overall health: {health['status']}")
        
        # Get performance statistics
        stats = logging_service.get_performance_statistics("api_response_time")
        print(f"API response time stats: {stats}")
        
        # Cleanup
        logging_service.cleanup()
        
        print("Logging service example completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


