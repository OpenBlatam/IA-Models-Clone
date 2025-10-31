"""
Advanced monitoring utilities for Ultimate Enhanced Supreme Production system
Following Flask best practices with functional programming patterns
"""

import time
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
from flask import request, g, current_app
from collections import defaultdict, deque
import threading
import json

logger = logging.getLogger(__name__)

class MonitoringManager:
    """Advanced monitoring manager with real-time metrics."""
    
    def __init__(self, max_samples: int = 1000):
        """Initialize monitoring manager with early returns."""
        self.max_samples = max_samples
        self.metrics = defaultdict(lambda: deque(maxlen=max_samples))
        self.alerts = deque(maxlen=1000)
        self.health_checks = {}
        self.lock = threading.Lock()
        self.start_time = time.time()
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Record monitoring metric with early returns."""
        if not name or value is None:
            return
        
        timestamp = time.time()
        metric_data = {
            'value': value,
            'timestamp': timestamp,
            'tags': tags or {}
        }
        
        with self.lock:
            self.metrics[name].append(metric_data)
    
    def record_alert(self, alert_type: str, message: str, severity: str = 'warning', 
                    tags: Dict[str, str] = None) -> None:
        """Record monitoring alert with early returns."""
        if not alert_type or not message:
            return
        
        alert_data = {
            'type': alert_type,
            'message': message,
            'severity': severity,
            'tags': tags or {},
            'timestamp': time.time()
        }
        
        with self.lock:
            self.alerts.append(alert_data)
        
        logger.warning(f"🚨 Alert: {alert_type} - {message}")
    
    def register_health_check(self, name: str, check_func: Callable) -> None:
        """Register health check with early returns."""
        if not name or not check_func:
            return
        
        self.health_checks[name] = check_func
        logger.info(f"✅ Health check registered: {name}")
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks with early returns."""
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                result = check_func()
                results[name] = {
                    'status': 'healthy' if result else 'unhealthy',
                    'result': result,
                    'timestamp': time.time()
                }
            except Exception as e:
                results[name] = {
                    'status': 'unhealthy',
                    'error': str(e),
                    'timestamp': time.time()
                }
                logger.error(f"❌ Health check failed: {name} - {e}")
        
        return results
    
    def get_metric_stats(self, name: str, time_window: int = 3600) -> Dict[str, Any]:
        """Get metric statistics with early returns."""
        if not name or name not in self.metrics:
            return {}
        
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        with self.lock:
            recent_metrics = [
                m for m in self.metrics[name] 
                if m['timestamp'] >= cutoff_time
            ]
        
        if not recent_metrics:
            return {}
        
        values = [m['value'] for m in recent_metrics]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'latest': values[-1] if values else None,
            'time_window': time_window
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics with early returns."""
        with self.lock:
            return {
                name: self.get_metric_stats(name)
                for name in self.metrics.keys()
            }
    
    def get_recent_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alerts with early returns."""
        with self.lock:
            return list(self.alerts)[-limit:]
    
    def clear_metrics(self) -> None:
        """Clear all metrics with early returns."""
        with self.lock:
            self.metrics.clear()
    
    def clear_alerts(self) -> None:
        """Clear all alerts with early returns."""
        with self.lock:
            self.alerts.clear()

# Global monitoring manager instance
monitoring_manager = MonitoringManager()

def init_monitoring(app) -> None:
    """Initialize monitoring with app."""
    global monitoring_manager
    monitoring_manager = MonitoringManager(max_samples=app.config.get('MONITORING_MAX_SAMPLES', 1000))
    app.logger.info("📊 Monitoring manager initialized")

def monitor_metric(metric_name: str, tags: Dict[str, str] = None):
    """Decorator for monitoring metrics with early returns."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time
                
                # Record metric
                monitoring_manager.record_metric(metric_name, execution_time, tags)
                
                return result
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                monitoring_manager.record_metric(f"{metric_name}_error", execution_time, tags)
                raise
        return wrapper
    return decorator

def monitor_errors(func: Callable) -> Callable:
    """Decorator for monitoring errors with early returns."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Record error metric
            monitoring_manager.record_metric('errors', 1, {'function': func.__name__})
            
            # Record alert
            monitoring_manager.record_alert(
                'function_error',
                f"Error in {func.__name__}: {str(e)}",
                'error',
                {'function': func.__name__}
            )
            
            raise
    return wrapper

def monitor_requests(func: Callable) -> Callable:
    """Decorator for monitoring requests with early returns."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        request_id = getattr(g, 'request_id', 'unknown')
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            
            # Record request metrics
            monitoring_manager.record_metric('request_duration', execution_time, {
                'endpoint': request.endpoint or 'unknown',
                'method': request.method,
                'status': 'success'
            })
            
            monitoring_manager.record_metric('requests', 1, {
                'endpoint': request.endpoint or 'unknown',
                'method': request.method
            })
            
            return result
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            
            # Record error metrics
            monitoring_manager.record_metric('request_duration', execution_time, {
                'endpoint': request.endpoint or 'unknown',
                'method': request.method,
                'status': 'error'
            })
            
            monitoring_manager.record_metric('request_errors', 1, {
                'endpoint': request.endpoint or 'unknown',
                'method': request.method
            })
            
            # Record alert
            monitoring_manager.record_alert(
                'request_error',
                f"Request error in {request.endpoint}: {str(e)}",
                'error',
                {'endpoint': request.endpoint, 'method': request.method}
            )
            
            raise
    return wrapper

def monitor_performance(func: Callable) -> Callable:
    """Decorator for monitoring performance with early returns."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            
            # Record performance metrics
            monitoring_manager.record_metric('performance', execution_time, {
                'function': func.__name__
            })
            
            # Check for performance issues
            if execution_time > 5.0:  # 5 second threshold
                monitoring_manager.record_alert(
                    'slow_performance',
                    f"Slow performance in {func.__name__}: {execution_time:.2f}s",
                    'warning',
                    {'function': func.__name__}
                )
            
            return result
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            monitoring_manager.record_metric('performance_error', execution_time, {
                'function': func.__name__
            })
            raise
    return wrapper

def monitor_memory(func: Callable) -> Callable:
    """Decorator for monitoring memory usage with early returns."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        import psutil
        process = psutil.Process()
        start_memory = process.memory_info().rss
        
        try:
            result = func(*args, **kwargs)
            end_memory = process.memory_info().rss
            memory_used = end_memory - start_memory
            
            # Record memory metric
            monitoring_manager.record_metric('memory_usage', memory_used, {
                'function': func.__name__
            })
            
            # Check for memory issues
            if memory_used > 100 * 1024 * 1024:  # 100MB threshold
                monitoring_manager.record_alert(
                    'high_memory_usage',
                    f"High memory usage in {func.__name__}: {memory_used / 1024 / 1024:.2f}MB",
                    'warning',
                    {'function': func.__name__}
                )
            
            return result
        except Exception as e:
            logger.error(f"❌ Memory monitoring error in {func.__name__}: {e}")
            raise
    return wrapper

def get_monitoring_metrics() -> Dict[str, Any]:
    """Get monitoring metrics with early returns."""
    return monitoring_manager.get_all_metrics()

def get_monitoring_alerts() -> List[Dict[str, Any]]:
    """Get monitoring alerts with early returns."""
    return monitoring_manager.get_recent_alerts()

def get_health_status() -> Dict[str, Any]:
    """Get health status with early returns."""
    health_checks = monitoring_manager.run_health_checks()
    
    overall_status = 'healthy'
    for check_name, check_result in health_checks.items():
        if check_result['status'] != 'healthy':
            overall_status = 'unhealthy'
            break
    
    return {
        'status': overall_status,
        'checks': health_checks,
        'timestamp': time.time()
    }

def register_health_check(name: str, check_func: Callable) -> None:
    """Register health check with early returns."""
    monitoring_manager.register_health_check(name, check_func)

def record_custom_metric(name: str, value: float, tags: Dict[str, str] = None) -> None:
    """Record custom metric with early returns."""
    monitoring_manager.record_metric(name, value, tags)

def record_alert(alert_type: str, message: str, severity: str = 'warning', 
                tags: Dict[str, str] = None) -> None:
    """Record alert with early returns."""
    monitoring_manager.record_alert(alert_type, message, severity, tags)

def get_monitoring_dashboard() -> Dict[str, Any]:
    """Get monitoring dashboard data with early returns."""
    return {
        'metrics': get_monitoring_metrics(),
        'alerts': get_monitoring_alerts(),
        'health': get_health_status(),
        'timestamp': time.time()
    }

def check_thresholds(thresholds: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    """Check monitoring thresholds with early returns."""
    alerts = []
    metrics = get_monitoring_metrics()
    
    for metric_name, threshold_config in thresholds.items():
        if metric_name not in metrics:
            continue
        
        metric_value = metrics[metric_name].get('latest', 0)
        threshold_value = threshold_config.get('value', 0)
        operator = threshold_config.get('operator', 'gt')
        
        is_alert = False
        if operator == 'gt' and metric_value > threshold_value:
            is_alert = True
        elif operator == 'lt' and metric_value < threshold_value:
            is_alert = True
        elif operator == 'eq' and metric_value == threshold_value:
            is_alert = True
        
        if is_alert:
            alerts.append({
                'metric': metric_name,
                'value': metric_value,
                'threshold': threshold_value,
                'operator': operator,
                'message': f"{metric_name} {operator} {threshold_value} (current: {metric_value})",
                'severity': threshold_config.get('severity', 'warning'),
                'timestamp': time.time()
            })
    
    return alerts

def get_monitoring_report() -> Dict[str, Any]:
    """Get comprehensive monitoring report with early returns."""
    return {
        'dashboard': get_monitoring_dashboard(),
        'thresholds': check_thresholds({}),  # Add threshold configuration
        'timestamp': time.time()
    }

def clear_monitoring_data() -> None:
    """Clear monitoring data with early returns."""
    monitoring_manager.clear_metrics()
    monitoring_manager.clear_alerts()

def export_monitoring_data() -> Dict[str, Any]:
    """Export monitoring data with early returns."""
    return {
        'metrics': dict(monitoring_manager.metrics),
        'alerts': list(monitoring_manager.alerts),
        'health_checks': list(monitoring_manager.health_checks.keys()),
        'timestamp': time.time()
    }

def import_monitoring_data(data: Dict[str, Any]) -> bool:
    """Import monitoring data with early returns."""
    if not data:
        return False
    
    try:
        if 'metrics' in data:
            monitoring_manager.metrics.update(data['metrics'])
        
        if 'alerts' in data:
            monitoring_manager.alerts.extend(data['alerts'])
        
        return True
    except Exception as e:
        logger.error(f"❌ Import monitoring data error: {e}")
        return False









