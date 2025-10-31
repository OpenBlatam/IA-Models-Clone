"""
Advanced performance utilities for Ultimate Enhanced Supreme Production system
Following Flask best practices with functional programming patterns
"""

import time
import psutil
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
from flask import request, g, current_app
import threading
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Advanced performance monitor with real-time metrics."""
    
    def __init__(self, max_samples: int = 1000):
        """Initialize performance monitor with early returns."""
        self.max_samples = max_samples
        self.metrics = defaultdict(lambda: deque(maxlen=max_samples))
        self.lock = threading.Lock()
        self.start_time = time.time()
    
    def record_metric(self, name: str, value: float, timestamp: float = None) -> None:
        """Record performance metric with early returns."""
        if not name or value is None:
            return
        
        timestamp = timestamp or time.time()
        
        with self.lock:
            self.metrics[name].append({
                'value': value,
                'timestamp': timestamp
            })
    
    def get_metric_stats(self, name: str) -> Dict[str, Any]:
        """Get metric statistics with early returns."""
        if not name or name not in self.metrics:
            return {}
        
        values = [m['value'] for m in self.metrics[name]]
        if not values:
            return {}
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0,
            'latest': values[-1] if values else None
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics with early returns."""
        with self.lock:
            return {
                name: self.get_metric_stats(name)
                for name in self.metrics.keys()
            }
    
    def clear_metrics(self) -> None:
        """Clear all metrics with early returns."""
        with self.lock:
            self.metrics.clear()
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics with early returns."""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
                'process_count': len(psutil.pids()),
                'uptime': time.time() - self.start_time
            }
        except Exception as e:
            logger.error(f"❌ System metrics error: {e}")
            return {}

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def init_performance_monitor(app) -> None:
    """Initialize performance monitor with app."""
    global performance_monitor
    performance_monitor = PerformanceMonitor(max_samples=app.config.get('PERFORMANCE_MAX_SAMPLES', 1000))
    app.logger.info("📊 Performance monitor initialized")

def performance_tracker(metric_name: str = None):
    """Decorator for performance tracking with early returns."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            request_id = getattr(g, 'request_id', 'unknown')
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time
                
                # Record metric
                metric = metric_name or func.__name__
                performance_monitor.record_metric(metric, execution_time)
                
                # Log performance
                logger.info(f"⚡ {func.__name__} executed in {execution_time:.3f}s [req:{request_id}]")
                
                # Store in request context
                if not hasattr(g, 'performance_metrics'):
                    g.performance_metrics = {}
                g.performance_metrics[func.__name__] = execution_time
                
                return result
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                logger.error(f"❌ {func.__name__} failed after {execution_time:.3f}s [req:{request_id}]: {e}")
                raise
        return wrapper
    return decorator

def memory_tracker(func: Callable) -> Callable:
    """Decorator for memory usage tracking with early returns."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_memory = process.memory_info().rss
        
        try:
            result = func(*args, **kwargs)
            end_memory = process.memory_info().rss
            memory_used = end_memory - start_memory
            
            # Record memory metric
            performance_monitor.record_metric(f"{func.__name__}_memory", memory_used)
            
            logger.debug(f"💾 {func.__name__} used {memory_used / 1024 / 1024:.2f} MB")
            
            return result
        except Exception as e:
            logger.error(f"❌ Memory tracking error in {func.__name__}: {e}")
            raise
    return wrapper

def cpu_tracker(func: Callable) -> Callable:
    """Decorator for CPU usage tracking with early returns."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        start_cpu = psutil.cpu_percent()
        
        try:
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            end_cpu = psutil.cpu_percent()
            
            execution_time = end_time - start_time
            cpu_usage = end_cpu - start_cpu
            
            # Record CPU metric
            performance_monitor.record_metric(f"{func.__name__}_cpu", cpu_usage)
            performance_monitor.record_metric(f"{func.__name__}_time", execution_time)
            
            logger.debug(f"🖥️ {func.__name__} CPU usage: {cpu_usage:.2f}%")
            
            return result
        except Exception as e:
            logger.error(f"❌ CPU tracking error in {func.__name__}: {e}")
            raise
    return wrapper

def throughput_tracker(func: Callable) -> Callable:
    """Decorator for throughput tracking with early returns."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            
            # Calculate throughput (operations per second)
            throughput = 1.0 / execution_time if execution_time > 0 else 0
            
            # Record throughput metric
            performance_monitor.record_metric(f"{func.__name__}_throughput", throughput)
            
            logger.debug(f"📈 {func.__name__} throughput: {throughput:.2f} ops/sec")
            
            return result
        except Exception as e:
            logger.error(f"❌ Throughput tracking error in {func.__name__}: {e}")
            raise
    return wrapper

def latency_tracker(func: Callable) -> Callable:
    """Decorator for latency tracking with early returns."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            latency = time.perf_counter() - start_time
            
            # Record latency metric
            performance_monitor.record_metric(f"{func.__name__}_latency", latency)
            
            logger.debug(f"⏱️ {func.__name__} latency: {latency:.3f}s")
            
            return result
        except Exception as e:
            logger.error(f"❌ Latency tracking error in {func.__name__}: {e}")
            raise
    return wrapper

def get_performance_metrics() -> Dict[str, Any]:
    """Get performance metrics with early returns."""
    return performance_monitor.get_all_metrics()

def get_system_metrics() -> Dict[str, Any]:
    """Get system metrics with early returns."""
    return performance_monitor.get_system_metrics()

def get_metric_stats(metric_name: str) -> Dict[str, Any]:
    """Get specific metric statistics with early returns."""
    return performance_monitor.get_metric_stats(metric_name)

def clear_performance_metrics() -> None:
    """Clear performance metrics with early returns."""
    performance_monitor.clear_metrics()

def record_custom_metric(name: str, value: float) -> None:
    """Record custom metric with early returns."""
    performance_monitor.record_metric(name, value)

def get_performance_summary() -> Dict[str, Any]:
    """Get performance summary with early returns."""
    metrics = get_performance_metrics()
    system_metrics = get_system_metrics()
    
    return {
        'application_metrics': metrics,
        'system_metrics': system_metrics,
        'timestamp': time.time(),
        'uptime': time.time() - performance_monitor.start_time
    }

def check_performance_thresholds(thresholds: Dict[str, float]) -> Dict[str, bool]:
    """Check performance thresholds with early returns."""
    if not thresholds:
        return {}
    
    results = {}
    metrics = get_performance_metrics()
    
    for metric_name, threshold in thresholds.items():
        if metric_name in metrics:
            latest_value = metrics[metric_name].get('latest', 0)
            results[metric_name] = latest_value > threshold
        else:
            results[metric_name] = False
    
    return results

def get_performance_alerts() -> List[Dict[str, Any]]:
    """Get performance alerts with early returns."""
    alerts = []
    metrics = get_performance_metrics()
    system_metrics = get_system_metrics()
    
    # Check CPU threshold
    if system_metrics.get('cpu_percent', 0) > 80:
        alerts.append({
            'type': 'cpu_high',
            'message': f"CPU usage is {system_metrics['cpu_percent']:.1f}%",
            'severity': 'warning'
        })
    
    # Check memory threshold
    if system_metrics.get('memory_percent', 0) > 85:
        alerts.append({
            'type': 'memory_high',
            'message': f"Memory usage is {system_metrics['memory_percent']:.1f}%",
            'severity': 'critical'
        })
    
    # Check disk threshold
    if system_metrics.get('disk_percent', 0) > 90:
        alerts.append({
            'type': 'disk_high',
            'message': f"Disk usage is {system_metrics['disk_percent']:.1f}%",
            'severity': 'critical'
        })
    
    # Check application metrics
    for metric_name, stats in metrics.items():
        if stats.get('mean', 0) > 5.0:  # 5 seconds threshold
            alerts.append({
                'type': 'slow_operation',
                'message': f"{metric_name} is slow (mean: {stats['mean']:.2f}s)",
                'severity': 'warning'
            })
    
    return alerts

def optimize_performance() -> Dict[str, Any]:
    """Optimize performance with early returns."""
    optimizations = []
    
    # Check for memory leaks
    system_metrics = get_system_metrics()
    if system_metrics.get('memory_percent', 0) > 70:
        optimizations.append({
            'type': 'memory_optimization',
            'message': 'Consider memory optimization',
            'priority': 'high'
        })
    
    # Check for CPU bottlenecks
    if system_metrics.get('cpu_percent', 0) > 80:
        optimizations.append({
            'type': 'cpu_optimization',
            'message': 'Consider CPU optimization',
            'priority': 'high'
        })
    
    # Check for slow operations
    metrics = get_performance_metrics()
    for metric_name, stats in metrics.items():
        if stats.get('mean', 0) > 2.0:
            optimizations.append({
                'type': 'operation_optimization',
                'message': f"Optimize {metric_name} (mean: {stats['mean']:.2f}s)",
                'priority': 'medium'
            })
    
    return {
        'optimizations': optimizations,
        'timestamp': time.time()
    }

def benchmark_function(func: Callable, iterations: int = 100, *args, **kwargs) -> Dict[str, Any]:
    """Benchmark function performance with early returns."""
    if not func or iterations <= 0:
        return {}
    
    times = []
    
    for _ in range(iterations):
        start_time = time.perf_counter()
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(f"❌ Benchmark error: {e}")
            continue
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    if not times:
        return {}
    
    return {
        'iterations': iterations,
        'min_time': min(times),
        'max_time': max(times),
        'mean_time': statistics.mean(times),
        'median_time': statistics.median(times),
        'std_time': statistics.stdev(times) if len(times) > 1 else 0,
        'total_time': sum(times),
        'ops_per_second': iterations / sum(times) if sum(times) > 0 else 0
    }

def profile_function(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Profile function with detailed metrics with early returns."""
    if not func:
        return {}
    
    start_time = time.perf_counter()
    start_memory = psutil.Process().memory_info().rss
    start_cpu = psutil.cpu_percent()
    
    try:
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss
        end_cpu = psutil.cpu_percent()
        
        return {
            'execution_time': end_time - start_time,
            'memory_used': end_memory - start_memory,
            'cpu_usage': end_cpu - start_cpu,
            'result': result,
            'timestamp': time.time()
        }
    except Exception as e:
        logger.error(f"❌ Profiling error: {e}")
        return {'error': str(e)}

def get_performance_report() -> Dict[str, Any]:
    """Get comprehensive performance report with early returns."""
    return {
        'summary': get_performance_summary(),
        'alerts': get_performance_alerts(),
        'optimizations': optimize_performance(),
        'timestamp': time.time()
    }









