"""
Metrics Aggregator for Color Grading AI
========================================

Aggregates metrics from multiple sources and provides unified views.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class AggregatedMetric:
    """Aggregated metric."""
    name: str
    value: float
    count: int
    min_value: float
    max_value: float
    avg_value: float
    median_value: float
    p95_value: float
    p99_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsAggregator:
    """
    Metrics aggregator.
    
    Features:
    - Multi-source aggregation
    - Time-window aggregation
    - Statistical calculations
    - Metric grouping
    - Custom aggregators
    - Real-time aggregation
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics aggregator.
        
        Args:
            window_size: Size of aggregation window
        """
        self.window_size = window_size
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._metadata: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._lock = asyncio.Lock()
        self._aggregators: Dict[str, Callable] = {}
    
    def register_aggregator(self, name: str, aggregator: Callable):
        """
        Register custom aggregator.
        
        Args:
            name: Aggregator name
            aggregator: Aggregator function
        """
        self._aggregators[name] = aggregator
        logger.debug(f"Registered aggregator: {name}")
    
    async def add_metric(
        self,
        name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add metric value.
        
        Args:
            name: Metric name
            value: Metric value
            metadata: Optional metadata
        """
        async with self._lock:
            self._metrics[name].append(value)
            
            # Maintain window size
            if len(self._metrics[name]) > self.window_size:
                self._metrics[name].pop(0)
            
            # Update metadata
            if metadata:
                self._metadata[name].update(metadata)
    
    async def aggregate(
        self,
        name: Optional[str] = None,
        window_seconds: Optional[float] = None
    ) -> Union[AggregatedMetric, Dict[str, AggregatedMetric]]:
        """
        Aggregate metrics.
        
        Args:
            name: Optional metric name
            window_seconds: Optional time window
            
        Returns:
            Aggregated metric(s)
        """
        async with self._lock:
            if name:
                return self._aggregate_single(name)
            else:
                return {
                    metric_name: self._aggregate_single(metric_name)
                    for metric_name in self._metrics.keys()
                }
    
    def _aggregate_single(self, name: str) -> AggregatedMetric:
        """Aggregate a single metric."""
        values = self._metrics.get(name, [])
        
        if not values:
            return AggregatedMetric(
                name=name,
                value=0.0,
                count=0,
                min_value=0.0,
                max_value=0.0,
                avg_value=0.0,
                median_value=0.0,
                p95_value=0.0,
                p99_value=0.0,
                metadata=self._metadata.get(name, {})
            )
        
        sorted_values = sorted(values)
        
        return AggregatedMetric(
            name=name,
            value=values[-1] if values else 0.0,
            count=len(values),
            min_value=min(values),
            max_value=max(values),
            avg_value=statistics.mean(values),
            median_value=statistics.median(values),
            p95_value=self._percentile(sorted_values, 95),
            p99_value=self._percentile(sorted_values, 99),
            metadata=self._metadata.get(name, {})
        )
    
    def _percentile(self, sorted_data: List[float], percentile: int) -> float:
        """Calculate percentile."""
        if not sorted_data:
            return 0.0
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    async def aggregate_by_group(
        self,
        group_func: Callable[[str], str]
    ) -> Dict[str, AggregatedMetric]:
        """
        Aggregate metrics by group.
        
        Args:
            group_func: Function to determine group from metric name
            
        Returns:
            Aggregated metrics by group
        """
        async with self._lock:
            groups: Dict[str, List[float]] = defaultdict(list)
            
            for name, values in self._metrics.items():
                group = group_func(name)
                groups[group].extend(values)
            
            result = {}
            for group, values in groups.items():
                if values:
                    sorted_values = sorted(values)
                    result[group] = AggregatedMetric(
                        name=group,
                        value=values[-1] if values else 0.0,
                        count=len(values),
                        min_value=min(values),
                        max_value=max(values),
                        avg_value=statistics.mean(values),
                        median_value=statistics.median(values),
                        p95_value=self._percentile(sorted_values, 95),
                        p99_value=self._percentile(sorted_values, 99),
                    )
            
            return result
    
    async def clear(self, name: Optional[str] = None):
        """
        Clear metrics.
        
        Args:
            name: Optional metric name
        """
        async with self._lock:
            if name:
                if name in self._metrics:
                    del self._metrics[name]
                if name in self._metadata:
                    del self._metadata[name]
            else:
                self._metrics.clear()
                self._metadata.clear()
            logger.info(f"Cleared metrics: {name or 'all'}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregator statistics."""
        return {
            "metrics_count": len(self._metrics),
            "total_values": sum(len(v) for v in self._metrics.values()),
            "window_size": self.window_size,
            "aggregators_count": len(self._aggregators),
        }


