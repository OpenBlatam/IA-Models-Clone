"""
Metrics Collector

Advanced metrics collection and aggregation.
"""

import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect and aggregate metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.metric_metadata: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.timestamps: Dict[str, List[datetime]] = defaultdict(list)
    
    def collect(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Collect metric.
        
        Args:
            metric_name: Metric name
            value: Metric value
            tags: Optional tags
        """
        self.metrics[metric_name].append(value)
        self.timestamps[metric_name].append(datetime.now())
        
        if tags:
            if 'tags' not in self.metric_metadata[metric_name]:
                self.metric_metadata[metric_name]['tags'] = []
            self.metric_metadata[metric_name]['tags'].append(tags)
    
    def get_summary(
        self,
        metric_name: str
    ) -> Dict[str, Any]:
        """
        Get metric summary.
        
        Args:
            metric_name: Metric name
            
        Returns:
            Metric summary
        """
        values = self.metrics.get(metric_name, [])
        
        if not values:
            return {}
        
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'max': max(values),
            'sum': sum(values)
        }
    
    def get_all_summaries(self) -> Dict[str, Dict[str, Any]]:
        """
        Get summaries for all metrics.
        
        Returns:
            Dictionary of metric summaries
        """
        return {
            name: self.get_summary(name)
            for name in self.metrics.keys()
        }
    
    def reset(self, metric_name: Optional[str] = None) -> None:
        """
        Reset metrics.
        
        Args:
            metric_name: Metric name (None = all)
        """
        if metric_name:
            self.metrics[metric_name].clear()
            self.timestamps[metric_name].clear()
            if metric_name in self.metric_metadata:
                del self.metric_metadata[metric_name]
        else:
            self.metrics.clear()
            self.timestamps.clear()
            self.metric_metadata.clear()
        
        logger.info(f"Reset metrics: {metric_name or 'all'}")


def collect_metric(
    collector: MetricsCollector,
    metric_name: str,
    value: float,
    **kwargs
) -> None:
    """Collect metric."""
    collector.collect(metric_name, value, **kwargs)


def get_metrics_summary(
    collector: MetricsCollector,
    metric_name: Optional[str] = None
) -> Dict[str, Any]:
    """Get metrics summary."""
    if metric_name:
        return collector.get_summary(metric_name)
    return collector.get_all_summaries()



