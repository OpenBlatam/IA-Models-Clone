from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import time
import psutil
import threading
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
from ..core.exceptions import PerformanceError
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Onyx AI Video System - Performance Monitor

Performance monitoring utilities for the Onyx AI Video system with
integration with Onyx's performance patterns and metrics collection.
"""




@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""
    metric_name: str
    threshold_value: float
    operator: str  # 'gt', 'lt', 'eq', 'gte', 'lte'
    action: str  # 'warn', 'error', 'alert'
    description: str = ""


class PerformanceMonitor:
    """
    Performance monitoring system for AI Video operations.
    
    Provides real-time monitoring, metrics collection, and performance
    analysis with Onyx integration.
    """
    
    def __init__(self, enable_monitoring: bool = True, metrics_interval: int = 60):
        
    """__init__ function."""
self.enable_monitoring = enable_monitoring
        self.metrics_interval = metrics_interval
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.thresholds: List[PerformanceThreshold] = []
        self.alerts: List[Dict[str, Any]] = []
        
        # System metrics
        self.system_metrics = {}
        self.last_system_check = None
        
        # Performance tracking
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.active_operations: Dict[str, float] = {}
        
        # Threading
        self._lock = threading.Lock()
        self._monitoring_thread = None
        self._stop_monitoring = False
        
        # Callbacks
        self.metric_callbacks: List[Callable] = []
        self.threshold_callbacks: List[Callable] = []
        
        if self.enable_monitoring:
            self._start_monitoring()
    
    def _start_monitoring(self) -> Any:
        """Start background monitoring thread."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_monitoring = False
            self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            self._monitoring_thread.start()
    
    def _monitoring_loop(self) -> Any:
        """Background monitoring loop."""
        while not self._stop_monitoring:
            try:
                self._collect_system_metrics()
                self._check_thresholds()
                time.sleep(self.metrics_interval)
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(5)  # Short delay on error
    
    def _collect_system_metrics(self) -> Any:
        """Collect system performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available / (1024 ** 3)  # GB
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free = disk.free / (1024 ** 3)  # GB
            
            # Network metrics
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # Process metrics
            process = psutil.Process()
            process_cpu_percent = process.cpu_percent()
            process_memory_percent = process.memory_percent()
            process_memory_info = process.memory_info()
            
            with self._lock:
                self.system_metrics = {
                    'cpu_percent': cpu_percent,
                    'cpu_count': cpu_count,
                    'memory_percent': memory_percent,
                    'memory_available_gb': memory_available,
                    'disk_percent': disk_percent,
                    'disk_free_gb': disk_free,
                    'network_bytes_sent': network_bytes_sent,
                    'network_bytes_recv': network_bytes_recv,
                    'process_cpu_percent': process_cpu_percent,
                    'process_memory_percent': process_memory_percent,
                    'process_memory_rss_mb': process_memory_info.rss / (1024 ** 2),
                    'process_memory_vms_mb': process_memory_info.vms / (1024 ** 2),
                    'timestamp': datetime.now()
                }
                
                self.last_system_check = datetime.now()
                
                # Store metrics
                for key, value in self.system_metrics.items():
                    if key != 'timestamp':
                        self.metrics_history[key].append(PerformanceMetric(
                            name=key,
                            value=value,
                            unit=self._get_metric_unit(key),
                            timestamp=datetime.now()
                        ))
                
        except Exception as e:
            print(f"Error collecting system metrics: {e}")
    
    def _get_metric_unit(self, metric_name: str) -> str:
        """Get unit for metric."""
        units = {
            'cpu_percent': '%',
            'memory_percent': '%',
            'disk_percent': '%',
            'memory_available_gb': 'GB',
            'disk_free_gb': 'GB',
            'network_bytes_sent': 'bytes',
            'network_bytes_recv': 'bytes',
            'process_cpu_percent': '%',
            'process_memory_percent': '%',
            'process_memory_rss_mb': 'MB',
            'process_memory_vms_mb': 'MB'
        }
        return units.get(metric_name, '')
    
    def _check_thresholds(self) -> Any:
        """Check performance thresholds."""
        with self._lock:
            for threshold in self.thresholds:
                current_value = self._get_current_metric_value(threshold.metric_name)
                if current_value is None:
                    continue
                
                if self._evaluate_threshold(current_value, threshold):
                    self._trigger_threshold_alert(threshold, current_value)
    
    def _get_current_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value for a metric."""
        if metric_name in self.system_metrics:
            return self.system_metrics[metric_name]
        
        # Check operation times
        if metric_name in self.operation_times and self.operation_times[metric_name]:
            return statistics.mean(self.operation_times[metric_name][-10:])  # Last 10 operations
        
        return None
    
    def _evaluate_threshold(self, value: float, threshold: PerformanceThreshold) -> bool:
        """Evaluate if threshold is exceeded."""
        operators = {
            'gt': lambda x, y: x > y,
            'lt': lambda x, y: x < y,
            'eq': lambda x, y: x == y,
            'gte': lambda x, y: x >= y,
            'lte': lambda x, y: x <= y
        }
        
        op_func = operators.get(threshold.operator)
        if op_func:
            return op_func(value, threshold.threshold_value)
        
        return False
    
    def _trigger_threshold_alert(self, threshold: PerformanceThreshold, current_value: float):
        """Trigger threshold alert."""
        alert = {
            'threshold': threshold,
            'current_value': current_value,
            'timestamp': datetime.now(),
            'message': f"Threshold exceeded: {threshold.metric_name} = {current_value} {threshold.operator} {threshold.threshold_value}"
        }
        
        self.alerts.append(alert)
        
        # Call threshold callbacks
        for callback in self.threshold_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Threshold callback error: {e}")
    
    def add_threshold(self, threshold: PerformanceThreshold):
        """Add performance threshold."""
        with self._lock:
            self.thresholds.append(threshold)
    
    def remove_threshold(self, metric_name: str, operator: str):
        """Remove performance threshold."""
        with self._lock:
            self.thresholds = [t for t in self.thresholds 
                             if not (t.metric_name == metric_name and t.operator == operator)]
    
    def start_operation(self, operation_name: str) -> str:
        """Start timing an operation."""
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        with self._lock:
            self.active_operations[operation_id] = time.time()
        return operation_id
    
    def end_operation(self, operation_id: str) -> Optional[float]:
        """End timing an operation."""
        with self._lock:
            if operation_id in self.active_operations:
                start_time = self.active_operations[operation_id]
                duration = time.time() - start_time
                
                # Extract operation name from ID
                operation_name = operation_id.rsplit('_', 1)[0]
                self.operation_times[operation_name].append(duration)
                
                # Keep only last 100 operations
                if len(self.operation_times[operation_name]) > 100:
                    self.operation_times[operation_name] = self.operation_times[operation_name][-100:]
                
                del self.active_operations[operation_id]
                
                # Store metric
                self.metrics_history[f"operation_{operation_name}"].append(PerformanceMetric(
                    name=f"operation_{operation_name}",
                    value=duration,
                    unit='seconds',
                    timestamp=datetime.now()
                ))
                
                return duration
        
        return None
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for an operation."""
        with self._lock:
            times = self.operation_times.get(operation_name, [])
            
            if not times:
                return {
                    'count': 0,
                    'avg_time': 0,
                    'min_time': 0,
                    'max_time': 0,
                    'total_time': 0
                }
            
            return {
                'count': len(times),
                'avg_time': statistics.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'total_time': sum(times),
                'median_time': statistics.median(times),
                'std_dev': statistics.stdev(times) if len(times) > 1 else 0
            }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        with self._lock:
            return self.system_metrics.copy()
    
    def get_metrics_history(self, metric_name: str, limit: int = 100) -> List[PerformanceMetric]:
        """Get metrics history for a specific metric."""
        with self._lock:
            metrics = list(self.metrics_history.get(metric_name, []))
            return metrics[-limit:] if limit else metrics
    
    def get_all_metrics(self) -> Dict[str, List[PerformanceMetric]]:
        """Get all metrics history."""
        with self._lock:
            return {name: list(metrics) for name, metrics in self.metrics_history.items()}
    
    def get_alerts(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get performance alerts."""
        with self._lock:
            if since:
                return [alert for alert in self.alerts if alert['timestamp'] >= since]
            return self.alerts.copy()
    
    def clear_alerts(self) -> Any:
        """Clear performance alerts."""
        with self._lock:
            self.alerts.clear()
    
    def add_metric_callback(self, callback: Callable[[PerformanceMetric], None]):
        """Add callback for new metrics."""
        self.metric_callbacks.append(callback)
    
    def add_threshold_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for threshold alerts."""
        self.threshold_callbacks.append(callback)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        with self._lock:
            summary = {
                'system_metrics': self.system_metrics.copy(),
                'operation_stats': {},
                'alerts_count': len(self.alerts),
                'active_operations': len(self.active_operations),
                'metrics_count': len(self.metrics_history),
                'timestamp': datetime.now()
            }
            
            # Add operation statistics
            for operation_name in self.operation_times:
                summary['operation_stats'][operation_name] = self.get_operation_stats(operation_name)
            
            return summary
    
    def reset_metrics(self) -> Any:
        """Reset all metrics."""
        with self._lock:
            self.metrics_history.clear()
            self.operation_times.clear()
            self.active_operations.clear()
            self.alerts.clear()
    
    def stop_monitoring(self) -> Any:
        """Stop performance monitoring."""
        self._stop_monitoring = True
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)


class PerformanceDecorator:
    """Decorator for automatic performance monitoring."""
    
    def __init__(self, monitor: PerformanceMonitor):
        
    """__init__ function."""
self.monitor = monitor
    
    def __call__(self, operation_name: str):
        """Decorator implementation."""
        def decorator(func) -> Any:
            def wrapper(*args, **kwargs) -> Any:
                operation_id = self.monitor.start_operation(operation_name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.monitor.end_operation(operation_id)
            return wrapper
        return decorator


class PerformanceContext:
    """Context manager for performance monitoring."""
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        
    """__init__ function."""
self.monitor = monitor
        self.operation_name = operation_name
        self.operation_id = None
    
    def __enter__(self) -> Any:
        self.operation_id = self.monitor.start_operation(self.operation_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> Any:
        if self.operation_id:
            self.monitor.end_operation(self.operation_id)


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor(enable_monitoring: bool = True, metrics_interval: int = 60) -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor(enable_monitoring, metrics_interval)
    return _performance_monitor


def monitor_performance(operation_name: str):
    """Decorator for performance monitoring."""
    monitor = get_performance_monitor()
    return PerformanceDecorator(monitor)(operation_name)


def performance_context(operation_name: str):
    """Context manager for performance monitoring."""
    monitor = get_performance_monitor()
    return PerformanceContext(monitor, operation_name)


# Utility functions
def get_system_performance() -> Dict[str, Any]:
    """Get current system performance metrics."""
    monitor = get_performance_monitor()
    return monitor.get_system_metrics()


def get_operation_performance(operation_name: str) -> Dict[str, Any]:
    """Get performance statistics for an operation."""
    monitor = get_performance_monitor()
    return monitor.get_operation_stats(operation_name)


def add_performance_threshold(metric_name: str, threshold_value: float, operator: str, action: str = "warn", description: str = ""):
    """Add performance threshold."""
    monitor = get_performance_monitor()
    threshold = PerformanceThreshold(
        metric_name=metric_name,
        threshold_value=threshold_value,
        operator=operator,
        action=action,
        description=description
    )
    monitor.add_threshold(threshold)


def get_performance_alerts(since: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Get performance alerts."""
    monitor = get_performance_monitor()
    return monitor.get_alerts(since)


def clear_performance_alerts():
    """Clear performance alerts."""
    monitor = get_performance_monitor()
    monitor.clear_alerts()


def get_performance_summary() -> Dict[str, Any]:
    """Get performance summary."""
    monitor = get_performance_monitor()
    return monitor.get_performance_summary()


def reset_performance_metrics():
    """Reset performance metrics."""
    monitor = get_performance_monitor()
    monitor.reset_metrics()


def stop_performance_monitoring():
    """Stop performance monitoring."""
    monitor = get_performance_monitor()
    monitor.stop_monitoring()


# Default performance thresholds
def setup_default_thresholds():
    """Setup default performance thresholds."""
    default_thresholds = [
        ("cpu_percent", 80, "gte", "warn", "High CPU usage"),
        ("memory_percent", 85, "gte", "warn", "High memory usage"),
        ("disk_percent", 90, "gte", "warn", "High disk usage"),
        ("process_memory_percent", 80, "gte", "warn", "High process memory usage"),
    ]
    
    for metric_name, threshold_value, operator, action, description in default_thresholds:
        add_performance_threshold(metric_name, threshold_value, operator, action, description) 