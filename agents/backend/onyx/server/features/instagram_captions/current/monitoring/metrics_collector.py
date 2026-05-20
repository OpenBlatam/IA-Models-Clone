"""
Metrics Collector for Instagram Captions API v10.0

Advanced metrics collection and aggregation.
"""

import time
import statistics
from typing import Dict, Any, List, Optional, Union
from collections import defaultdict, deque
from dataclasses import dataclass

@dataclass
class MetricPoint:
    """Individual metric data point."""
    name: str
    value: Union[int, float]
    timestamp: float
    tags: Dict[str, str]
    metadata: Optional[Dict[str, Any]] = None

class MetricsCollector:
    """Advanced metrics collection and aggregation system."""
    
    def __init__(self, max_metrics: int = 10000, retention_hours: int = 24):
        self.max_metrics = max_metrics
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics))
        self.aggregated_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.start_time = time.time()
    
    def record_metric(self, name: str, value: Union[int, float], 
                     tags: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """Record a new metric."""
        if tags is None:
            tags = {}
        
        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags,
            metadata=metadata
        )
        
        self.metrics[name].append(metric_point)
        
        # Update aggregated metrics
        self._update_aggregated_metric(name, value)
    
    def record_counter(self, name: str, increment: int = 1, 
                      tags: Optional[Dict[str, str]] = None):
        """Record a counter metric."""
        current_value = self.aggregated_metrics.get(name, {}).get('count', 0)
        self.record_metric(name, current_value + increment, tags)
    
    def record_gauge(self, name: str, value: Union[int, float], 
                    tags: Optional[Dict[str, str]] = None):
        """Record a gauge metric."""
        self.record_metric(name, value, tags)
    
    def record_histogram(self, name: str, value: Union[int, float], 
                        buckets: Optional[List[float]] = None,
                        tags: Optional[Dict[str, str]] = None):
        """Record a histogram metric."""
        if buckets is None:
            buckets = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
        
        # Record the value
        self.record_metric(name, value, tags)
        
        # Update histogram buckets
        histogram_key = f"{name}_histogram"
        if histogram_key not in self.aggregated_metrics:
            self.aggregated_metrics[histogram_key] = {
                'buckets': buckets,
                'bucket_counts': defaultdict(int),
                'min': float('inf'),
                'max': float('-inf')
            }
        
        hist = self.aggregated_metrics[histogram_key]
        hist['min'] = min(hist['min'], value)
        hist['max'] = max(hist['max'], value)
        
        # Count values in buckets
        for bucket in buckets:
            if value <= bucket:
                hist['bucket_counts'][bucket] += 1
                break
    
    def record_timing(self, name: str, duration: float, 
                     tags: Optional[Dict[str, str]] = None):
        """Record a timing metric."""
        self.record_histogram(f"{name}_duration", duration, tags=tags)
        self.record_metric(f"{name}_count", 1, tags)
    
    def _update_aggregated_metric(self, name: str, value: Union[int, float]):
        """Update aggregated metrics for a given metric name."""
        if name not in self.aggregated_metrics:
            self.aggregated_metrics[name] = {
                'count': 0,
                'sum': 0,
                'min': float('inf'),
                'max': float('-inf'),
                'values': []
            }
        
        agg = self.aggregated_metrics[name]
        agg['count'] += 1
        agg['sum'] += value
        agg['min'] = min(agg['min'], value)
        agg['max'] = max(agg['max'], value)
        agg['values'].append(value)
        
        # Keep only recent values for percentiles
        if len(agg['values']) > 1000:
            agg['values'] = agg['values'][-1000:]
    
    def get_metric_summary(self, name: str) -> Dict[str, Any]:
        """Get summary statistics for a specific metric."""
        if name not in self.aggregated_metrics:
            return {"error": "Metric not found"}
        
        agg = self.aggregated_metrics[name]
        values = agg['values']
        
        if not values:
            return {"error": "No data available"}
        
        # Calculate percentiles
        sorted_values = sorted(values)
        percentiles = {}
        for p in [50, 75, 90, 95, 99]:
            index = int((p / 100) * (len(sorted_values) - 1))
            percentiles[f'p{p}'] = sorted_values[index]
        
        return {
            'name': name,
            'count': agg['count'],
            'sum': agg['sum'],
            'min': agg['min'],
            'max': agg['max'],
            'mean': agg['sum'] / agg['count'],
            'median': statistics.median(values),
            'percentiles': percentiles,
            'last_updated': time.time()
        }
    
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summaries = {}
        for metric_name in self.aggregated_metrics.keys():
            if not metric_name.endswith('_histogram'):
                summaries[metric_name] = self.get_metric_summary(metric_name)
        
        return {
            'total_metrics': len(summaries),
            'metrics': summaries,
            'collection_start': self.start_time,
            'uptime_seconds': time.time() - self.start_time
        }
    
    def get_metrics_by_tag(self, tag_name: str, tag_value: str) -> Dict[str, Any]:
        """Get metrics filtered by specific tag."""
        filtered_metrics = {}
        
        for metric_name, metric_queue in self.metrics.items():
            matching_points = [
                point for point in metric_queue
                if point.tags.get(tag_name) == tag_value
            ]
            
            if matching_points:
                values = [point.value for point in matching_points]
                filtered_metrics[metric_name] = {
                    'count': len(values),
                    'sum': sum(values),
                    'min': min(values),
                    'max': max(values),
                    'mean': sum(values) / len(values),
                    'median': statistics.median(values)
                }
        
        return filtered_metrics
    
    def get_metrics_timeline(self, name: str, 
                           interval_minutes: int = 5,
                           hours_back: int = 24) -> Dict[str, Any]:
        """Get metrics timeline for a specific metric."""
        if name not in self.metrics:
            return {"error": "Metric not found"}
        
        current_time = time.time()
        start_time = current_time - (hours_back * 3600)
        interval_seconds = interval_minutes * 60
        
        timeline = []
        current_interval = start_time
        
        while current_interval < current_time:
            interval_end = current_interval + interval_seconds
            
            # Find metrics in this interval
            interval_metrics = [
                point.value for point in self.metrics[name]
                if current_interval <= point.timestamp < interval_end
            ]
            
            if interval_metrics:
                timeline.append({
                    'timestamp': current_interval,
                    'count': len(interval_metrics),
                    'sum': sum(interval_metrics),
                    'min': min(interval_metrics),
                    'max': max(interval_metrics),
                    'mean': sum(interval_metrics) / len(interval_metrics)
                })
            else:
                timeline.append({
                    'timestamp': current_interval,
                    'count': 0,
                    'sum': 0,
                    'min': 0,
                    'max': 0,
                    'mean': 0
                })
            
            current_interval = interval_end
        
        return {
            'metric_name': name,
            'interval_minutes': interval_minutes,
            'hours_back': hours_back,
            'timeline': timeline
        }
    
    def export_metrics(self, format: str = "json") -> str:
        """Export all metrics in specified format."""
        if format.lower() == "json":
            import json
            return json.dumps(self.get_all_metrics_summary(), indent=2, default=str)
        else:
            return str(self.get_all_metrics_summary())
    
    def clear_metrics(self, metric_name: Optional[str] = None):
        """Clear metrics for specific metric or all metrics."""
        if metric_name:
            if metric_name in self.metrics:
                self.metrics[metric_name].clear()
            if metric_name in self.aggregated_metrics:
                del self.aggregated_metrics[metric_name]
        else:
            self.metrics.clear()
            self.aggregated_metrics.clear()
    
    def cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        current_time = time.time()
        cutoff_time = current_time - (self.retention_hours * 3600)
        
        for metric_name, metric_queue in self.metrics.items():
            # Remove old metrics
            while metric_queue and metric_queue[0].timestamp < cutoff_time:
                metric_queue.popleft()






