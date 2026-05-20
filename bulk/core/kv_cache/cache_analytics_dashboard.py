"""
Analytics dashboard for KV cache.

This module provides comprehensive analytics and reporting capabilities
for cache performance monitoring.
"""

import time
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """A metric."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class DashboardReport:
    """Dashboard report."""
    timestamp: float
    metrics: Dict[str, Any]
    alerts: List[Dict[str, Any]]
    recommendations: List[str]
    summary: Dict[str, Any]


class CacheAnalyticsDashboard:
    """Analytics dashboard for cache."""
    
    def __init__(self, cache: Any):
        self.cache = cache
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._alerts: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a metric."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=time.time(),
            tags=tags or {}
        )
        
        with self._lock:
            self._metrics[name].append(metric)
            
    def get_metric(self, name: str, window_seconds: Optional[float] = None) -> List[Metric]:
        """Get metrics for a name."""
        with self._lock:
            metrics = list(self._metrics.get(name, []))
            
        if window_seconds:
            cutoff_time = time.time() - window_seconds
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            
        return metrics
        
    def get_metric_statistics(self, name: str, window_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Get statistics for a metric."""
        metrics = self.get_metric(name, window_seconds)
        
        if not metrics:
            return {}
            
        values = [m.value for m in metrics]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'latest': values[-1] if values else None
        }
        
    def generate_report(self) -> DashboardReport:
        """Generate comprehensive dashboard report."""
        current_time = time.time()
        
        # Collect metrics from cache
        metrics = {}
        alerts = []
        recommendations = []
        
        if hasattr(self.cache, 'stats'):
            stats = self.cache.stats
            
            # Cache performance metrics
            metrics['hit_rate'] = getattr(stats, 'hit_rate', 0.0)
            metrics['miss_rate'] = getattr(stats, 'miss_rate', 0.0)
            metrics['total_requests'] = getattr(stats, 'total_requests', 0)
            metrics['cache_size'] = len(self.cache._cache) if hasattr(self.cache, '_cache') else 0
            
            # Check for alerts
            if metrics['hit_rate'] < 0.5:
                alerts.append({
                    'level': 'warning',
                    'message': 'Low hit rate detected',
                    'metric': 'hit_rate',
                    'value': metrics['hit_rate']
                })
                recommendations.append('Consider increasing cache size or adjusting eviction strategy')
                
            if metrics['miss_rate'] > 0.5:
                alerts.append({
                    'level': 'warning',
                    'message': 'High miss rate detected',
                    'metric': 'miss_rate',
                    'value': metrics['miss_rate']
                })
                recommendations.append('Review cache warming strategy and TTL settings')
                
        # Get custom metrics
        for metric_name in self._metrics.keys():
            stats = self.get_metric_statistics(metric_name, window_seconds=3600)
            if stats:
                metrics[f'{metric_name}_stats'] = stats
                
        summary = {
            'total_metrics': len(self._metrics),
            'total_alerts': len(alerts),
            'report_time': current_time
        }
        
        return DashboardReport(
            timestamp=current_time,
            metrics=metrics,
            alerts=alerts,
            recommendations=recommendations,
            summary=summary
        )
        
    def get_time_series(
        self,
        metric_name: str,
        start_time: float,
        end_time: float,
        interval: float = 60.0
    ) -> List[Tuple[float, float]]:
        """Get time series data for a metric."""
        metrics = self.get_metric(metric_name)
        
        # Filter by time range
        filtered = [
            m for m in metrics
            if start_time <= m.timestamp <= end_time
        ]
        
        # Aggregate by interval
        time_series = []
        current_interval = start_time
        
        while current_interval <= end_time:
            interval_end = current_interval + interval
            interval_metrics = [
                m for m in filtered
                if current_interval <= m.timestamp < interval_end
            ]
            
            if interval_metrics:
                avg_value = statistics.mean([m.value for m in interval_metrics])
                time_series.append((current_interval, avg_value))
            else:
                time_series.append((current_interval, 0.0))
                
            current_interval += interval
            
        return time_series
        
    def add_alert(
        self,
        level: str,
        message: str,
        metric: Optional[str] = None,
        value: Optional[float] = None
    ) -> None:
        """Add an alert."""
        alert = {
            'level': level,
            'message': message,
            'timestamp': time.time(),
            'metric': metric,
            'value': value
        }
        
        with self._lock:
            self._alerts.append(alert)
            
    def get_alerts(self, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get alerts, optionally filtered by level."""
        with self._lock:
            alerts = list(self._alerts)
            
        if level:
            alerts = [a for a in alerts if a.get('level') == level]
            
        return alerts
        
    def clear_alerts(self) -> None:
        """Clear all alerts."""
        with self._lock:
            self._alerts.clear()


class CacheMonitoringDashboard:
    """Monitoring dashboard wrapper."""
    
    def __init__(self, cache: Any):
        self.cache = cache
        self.dashboard = CacheAnalyticsDashboard(cache)
        
    def record_operation(self, operation: str, duration: float, success: bool = True) -> None:
        """Record a cache operation."""
        self.dashboard.record_metric(
            f'operation_{operation}_duration',
            duration,
            MetricType.TIMER
        )
        self.dashboard.record_metric(
            f'operation_{operation}_success',
            1.0 if success else 0.0,
            MetricType.COUNTER
        )
        
    def get_report(self) -> DashboardReport:
        """Get dashboard report."""
        return self.dashboard.generate_report()














