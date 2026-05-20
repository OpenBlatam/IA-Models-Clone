"""
Advanced analytics utilities for Ultimate Enhanced Supreme Production system
Following Flask best practices with functional programming patterns
"""

import time
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
from flask import request, g, current_app
from collections import defaultdict, deque
import threading
import statistics
import json

logger = logging.getLogger(__name__)

class AnalyticsManager:
    """Advanced analytics manager with real-time data processing."""
    
    def __init__(self, max_samples: int = 10000):
        """Initialize analytics manager with early returns."""
        self.max_samples = max_samples
        self.events = deque(maxlen=max_samples)
        self.metrics = defaultdict(lambda: deque(maxlen=max_samples))
        self.lock = threading.Lock()
        self.start_time = time.time()
    
    def record_event(self, event_type: str, data: Dict[str, Any], tags: Dict[str, str] = None) -> None:
        """Record analytics event with early returns."""
        if not event_type or not data:
            return
        
        event_data = {
            'type': event_type,
            'data': data,
            'tags': tags or {},
            'timestamp': time.time()
        }
        
        with self.lock:
            self.events.append(event_data)
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Record analytics metric with early returns."""
        if not name or value is None:
            return
        
        metric_data = {
            'value': value,
            'tags': tags or {},
            'timestamp': time.time()
        }
        
        with self.lock:
            self.metrics[name].append(metric_data)
    
    def get_event_stats(self, event_type: str, time_window: int = 3600) -> Dict[str, Any]:
        """Get event statistics with early returns."""
        if not event_type:
            return {}
        
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        with self.lock:
            recent_events = [
                e for e in self.events 
                if e['type'] == event_type and e['timestamp'] >= cutoff_time
            ]
        
        if not recent_events:
            return {}
        
        return {
            'count': len(recent_events),
            'time_window': time_window,
            'rate': len(recent_events) / (time_window / 3600) if time_window > 0 else 0
        }
    
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
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0,
            'latest': values[-1] if values else None,
            'time_window': time_window
        }
    
    def get_trends(self, name: str, time_window: int = 3600, interval: int = 300) -> List[Dict[str, Any]]:
        """Get trends data with early returns."""
        if not name or name not in self.metrics:
            return []
        
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        with self.lock:
            recent_metrics = [
                m for m in self.metrics[name] 
                if m['timestamp'] >= cutoff_time
            ]
        
        if not recent_metrics:
            return []
        
        # Group by time intervals
        intervals = {}
        for metric in recent_metrics:
            interval_time = int(metric['timestamp'] // interval) * interval
            if interval_time not in intervals:
                intervals[interval_time] = []
            intervals[interval_time].append(metric['value'])
        
        # Calculate statistics for each interval
        trends = []
        for interval_time in sorted(intervals.keys()):
            values = intervals[interval_time]
            trends.append({
                'timestamp': interval_time,
                'count': len(values),
                'mean': statistics.mean(values),
                'min': min(values),
                'max': max(values)
            })
        
        return trends
    
    def get_correlations(self, metric1: str, metric2: str, time_window: int = 3600) -> Dict[str, Any]:
        """Get correlation between metrics with early returns."""
        if not metric1 or not metric2 or metric1 not in self.metrics or metric2 not in self.metrics:
            return {}
        
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        with self.lock:
            recent_metrics1 = [
                m for m in self.metrics[metric1] 
                if m['timestamp'] >= cutoff_time
            ]
            recent_metrics2 = [
                m for m in self.metrics[metric2] 
                if m['timestamp'] >= cutoff_time
            ]
        
        if not recent_metrics1 or not recent_metrics2:
            return {}
        
        # Simple correlation calculation
        values1 = [m['value'] for m in recent_metrics1]
        values2 = [m['value'] for m in recent_metrics2]
        
        if len(values1) != len(values2):
            return {}
        
        # Calculate correlation coefficient
        n = len(values1)
        sum1 = sum(values1)
        sum2 = sum(values2)
        sum1_sq = sum(x * x for x in values1)
        sum2_sq = sum(x * x for x in values2)
        sum12 = sum(x * y for x, y in zip(values1, values2))
        
        numerator = n * sum12 - sum1 * sum2
        denominator = ((n * sum1_sq - sum1 * sum1) * (n * sum2_sq - sum2 * sum2)) ** 0.5
        
        correlation = numerator / denominator if denominator != 0 else 0
        
        return {
            'correlation': correlation,
            'sample_size': n,
            'time_window': time_window
        }
    
    def get_predictions(self, name: str, time_window: int = 3600, prediction_points: int = 10) -> List[Dict[str, Any]]:
        """Get predictions based on historical data with early returns."""
        if not name or name not in self.metrics:
            return []
        
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        with self.lock:
            recent_metrics = [
                m for m in self.metrics[name] 
                if m['timestamp'] >= cutoff_time
            ]
        
        if not recent_metrics or len(recent_metrics) < 2:
            return []
        
        values = [m['value'] for m in recent_metrics]
        
        # Simple linear regression for prediction
        n = len(values)
        x = list(range(n))
        y = values
        
        # Calculate slope and intercept
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x * y for x, y in zip(x, y))
        sum_x2 = sum(x * x for x in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0
        intercept = (sum_y - slope * sum_x) / n
        
        # Generate predictions
        predictions = []
        for i in range(1, prediction_points + 1):
            predicted_value = slope * (n + i) + intercept
            predicted_time = current_time + (i * 300)  # 5-minute intervals
            
            predictions.append({
                'timestamp': predicted_time,
                'value': predicted_value,
                'confidence': 0.8  # Mock confidence
            })
        
        return predictions
    
    def clear_data(self) -> None:
        """Clear analytics data with early returns."""
        with self.lock:
            self.events.clear()
            self.metrics.clear()

# Global analytics manager instance
analytics_manager = AnalyticsManager()

def init_analytics(app) -> None:
    """Initialize analytics with app."""
    global analytics_manager
    analytics_manager = AnalyticsManager(max_samples=app.config.get('ANALYTICS_MAX_SAMPLES', 10000))
    app.logger.info("📈 Analytics manager initialized")

def track_event(event_type: str, data: Dict[str, Any] = None, tags: Dict[str, str] = None):
    """Decorator for tracking events with early returns."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                
                # Record event
                event_data = {
                    'function': func.__name__,
                    'result': 'success',
                    'timestamp': time.time()
                }
                if data:
                    event_data.update(data)
                
                analytics_manager.record_event(event_type, event_data, tags)
                
                return result
            except Exception as e:
                # Record error event
                error_data = {
                    'function': func.__name__,
                    'result': 'error',
                    'error': str(e),
                    'timestamp': time.time()
                }
                if data:
                    error_data.update(data)
                
                analytics_manager.record_event(f"{event_type}_error", error_data, tags)
                
                raise
        return wrapper
    return decorator

def track_usage(func: Callable) -> Callable:
    """Decorator for tracking usage with early returns."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        request_id = getattr(g, 'request_id', 'unknown')
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            
            # Record usage metrics
            analytics_manager.record_metric('usage_duration', execution_time, {
                'function': func.__name__,
                'endpoint': request.endpoint or 'unknown',
                'method': request.method
            })
            
            analytics_manager.record_metric('usage_count', 1, {
                'function': func.__name__,
                'endpoint': request.endpoint or 'unknown',
                'method': request.method
            })
            
            # Record usage event
            analytics_manager.record_event('function_usage', {
                'function': func.__name__,
                'endpoint': request.endpoint or 'unknown',
                'method': request.method,
                'duration': execution_time,
                'request_id': request_id
            })
            
            return result
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            
            # Record error metrics
            analytics_manager.record_metric('usage_errors', 1, {
                'function': func.__name__,
                'endpoint': request.endpoint or 'unknown',
                'method': request.method
            })
            
            # Record error event
            analytics_manager.record_event('function_error', {
                'function': func.__name__,
                'endpoint': request.endpoint or 'unknown',
                'method': request.method,
                'duration': execution_time,
                'error': str(e),
                'request_id': request_id
            })
            
            raise
    return wrapper

def track_performance(func: Callable) -> Callable:
    """Decorator for tracking performance with early returns."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            
            # Record performance metrics
            analytics_manager.record_metric('performance', execution_time, {
                'function': func.__name__
            })
            
            # Record performance event
            analytics_manager.record_event('performance_measurement', {
                'function': func.__name__,
                'duration': execution_time,
                'timestamp': time.time()
            })
            
            return result
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            
            # Record error performance
            analytics_manager.record_metric('performance_error', execution_time, {
                'function': func.__name__
            })
            
            raise
    return wrapper

def get_analytics_data() -> Dict[str, Any]:
    """Get analytics data with early returns."""
    return {
        'events': list(analytics_manager.events),
        'metrics': dict(analytics_manager.metrics),
        'timestamp': time.time()
    }

def get_usage_analytics(time_window: int = 3600) -> Dict[str, Any]:
    """Get usage analytics with early returns."""
    return {
        'function_usage': analytics_manager.get_event_stats('function_usage', time_window),
        'function_errors': analytics_manager.get_event_stats('function_error', time_window),
        'usage_duration': analytics_manager.get_metric_stats('usage_duration', time_window),
        'usage_count': analytics_manager.get_metric_stats('usage_count', time_window),
        'time_window': time_window
    }

def get_performance_analytics(time_window: int = 3600) -> Dict[str, Any]:
    """Get performance analytics with early returns."""
    return {
        'performance': analytics_manager.get_metric_stats('performance', time_window),
        'performance_trends': analytics_manager.get_trends('performance', time_window),
        'performance_predictions': analytics_manager.get_predictions('performance', time_window),
        'time_window': time_window
    }

def get_optimization_analytics(time_window: int = 3600) -> Dict[str, Any]:
    """Get optimization analytics with early returns."""
    return {
        'optimization_events': analytics_manager.get_event_stats('optimization', time_window),
        'optimization_metrics': analytics_manager.get_metric_stats('optimization', time_window),
        'optimization_trends': analytics_manager.get_trends('optimization', time_window),
        'time_window': time_window
    }

def get_analytics_report(time_window: int = 3600) -> Dict[str, Any]:
    """Get comprehensive analytics report with early returns."""
    return {
        'usage': get_usage_analytics(time_window),
        'performance': get_performance_analytics(time_window),
        'optimization': get_optimization_analytics(time_window),
        'timestamp': time.time()
    }

def record_custom_event(event_type: str, data: Dict[str, Any], tags: Dict[str, str] = None) -> None:
    """Record custom event with early returns."""
    analytics_manager.record_event(event_type, data, tags)

def record_custom_metric(name: str, value: float, tags: Dict[str, str] = None) -> None:
    """Record custom metric with early returns."""
    analytics_manager.record_metric(name, value, tags)

def get_analytics_trends(metric_name: str, time_window: int = 3600, interval: int = 300) -> List[Dict[str, Any]]:
    """Get analytics trends with early returns."""
    return analytics_manager.get_trends(metric_name, time_window, interval)

def get_analytics_correlations(metric1: str, metric2: str, time_window: int = 3600) -> Dict[str, Any]:
    """Get analytics correlations with early returns."""
    return analytics_manager.get_correlations(metric1, metric2, time_window)

def get_analytics_predictions(metric_name: str, time_window: int = 3600, prediction_points: int = 10) -> List[Dict[str, Any]]:
    """Get analytics predictions with early returns."""
    return analytics_manager.get_predictions(metric_name, time_window, prediction_points)

def clear_analytics_data() -> None:
    """Clear analytics data with early returns."""
    analytics_manager.clear_data()

def export_analytics_data() -> Dict[str, Any]:
    """Export analytics data with early returns."""
    return {
        'events': list(analytics_manager.events),
        'metrics': dict(analytics_manager.metrics),
        'timestamp': time.time()
    }

def import_analytics_data(data: Dict[str, Any]) -> bool:
    """Import analytics data with early returns."""
    if not data:
        return False
    
    try:
        if 'events' in data:
            analytics_manager.events.extend(data['events'])
        
        if 'metrics' in data:
            analytics_manager.metrics.update(data['metrics'])
        
        return True
    except Exception as e:
        logger.error(f"❌ Import analytics data error: {e}")
        return False









