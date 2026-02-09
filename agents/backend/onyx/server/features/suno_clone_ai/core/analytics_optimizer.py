"""
Analytics and Metrics Optimizations

Optimizations for:
- Metrics collection
- Analytics processing
- Real-time metrics
- Performance tracking
- User analytics
"""

import logging
import time
from typing import Optional, Dict, Any, List
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Metric data structure."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Optimized metrics collection."""
    
    def __init__(self, max_metrics: int = 10000):
        """
        Initialize metrics collector.
        
        Args:
            max_metrics: Maximum metrics to keep in memory
        """
        self.metrics: deque = deque(maxlen=max_metrics)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
    
    def increment(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Increment counter metric.
        
        Args:
            name: Metric name
            value: Increment value
            tags: Optional tags
        """
        self.counters[name] += value
        self.metrics.append(Metric(name, value, tags=tags or {}))
    
    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Set gauge metric.
        
        Args:
            name: Metric name
            value: Gauge value
            tags: Optional tags
        """
        self.gauges[name] = value
        self.metrics.append(Metric(name, value, tags=tags or {}))
    
    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record histogram value.
        
        Args:
            name: Metric name
            value: Histogram value
            tags: Optional tags
        """
        self.histograms[name].append(value)
        if len(self.histograms[name]) > 1000:
            self.histograms[name] = self.histograms[name][-1000:]
        
        self.metrics.append(Metric(name, value, tags=tags or {}))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get all statistics."""
        histogram_stats = {}
        for name, values in self.histograms.items():
            if values:
                histogram_stats[name] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'p50': sorted(values)[len(values) // 2],
                    'p95': sorted(values)[int(len(values) * 0.95)],
                    'p99': sorted(values)[int(len(values) * 0.99)]
                }
        
        return {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'histograms': histogram_stats,
            'total_metrics': len(self.metrics)
        }


class RealTimeAnalytics:
    """Real-time analytics processing."""
    
    def __init__(self, window_seconds: int = 60):
        """
        Initialize real-time analytics.
        
        Args:
            window_seconds: Time window in seconds
        """
        self.window_seconds = window_seconds
        self.events: deque = deque()
    
    def record_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Record event.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        event = {
            'type': event_type,
            'data': data,
            'timestamp': time.time()
        }
        self.events.append(event)
    
    def get_recent_events(self, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent events.
        
        Args:
            event_type: Filter by event type
            
        Returns:
            List of recent events
        """
        now = time.time()
        cutoff = now - self.window_seconds
        
        recent = [
            event for event in self.events
            if event['timestamp'] > cutoff and
            (event_type is None or event['type'] == event_type)
        ]
        
        return recent
    
    def get_event_stats(self) -> Dict[str, int]:
        """Get event statistics."""
        stats = defaultdict(int)
        for event in self.get_recent_events():
            stats[event['type']] += 1
        return dict(stats)


class PerformanceTracker:
    """Performance tracking optimization."""
    
    def __init__(self):
        """Initialize performance tracker."""
        self.operations: Dict[str, List[float]] = defaultdict(list)
    
    def track_operation(self, name: str, duration: float) -> None:
        """
        Track operation performance.
        
        Args:
            name: Operation name
            duration: Duration in seconds
        """
        self.operations[name].append(duration)
        if len(self.operations[name]) > 1000:
            self.operations[name] = self.operations[name][-1000:]
    
    def get_performance_stats(self, name: str) -> Optional[Dict[str, float]]:
        """
        Get performance statistics for operation.
        
        Args:
            name: Operation name
            
        Returns:
            Performance statistics
        """
        if name not in self.operations or not self.operations[name]:
            return None
        
        durations = self.operations[name]
        sorted_durations = sorted(durations)
        
        return {
            'count': len(durations),
            'total': sum(durations),
            'avg': sum(durations) / len(durations),
            'min': min(durations),
            'max': max(durations),
            'p50': sorted_durations[len(sorted_durations) // 2],
            'p95': sorted_durations[int(len(sorted_durations) * 0.95)],
            'p99': sorted_durations[int(len(sorted_durations) * 0.99)]
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get all performance statistics."""
        return {
            name: self.get_performance_stats(name)
            for name in self.operations.keys()
        }


class UserAnalytics:
    """User analytics optimization."""
    
    def __init__(self):
        """Initialize user analytics."""
        self.user_actions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
    
    def track_action(self, user_id: str, action: str, data: Dict[str, Any]) -> None:
        """
        Track user action.
        
        Args:
            user_id: User ID
            action: Action type
            data: Action data
        """
        action_record = {
            'action': action,
            'data': data,
            'timestamp': time.time()
        }
        self.user_actions[user_id].append(action_record)
        
        # Keep only last 1000 actions per user
        if len(self.user_actions[user_id]) > 1000:
            self.user_actions[user_id] = self.user_actions[user_id][-1000:]
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get user statistics.
        
        Args:
            user_id: User ID
            
        Returns:
            User statistics
        """
        actions = self.user_actions.get(user_id, [])
        
        action_counts = defaultdict(int)
        for action in actions:
            action_counts[action['action']] += 1
        
        return {
            'total_actions': len(actions),
            'action_counts': dict(action_counts),
            'last_action': actions[-1] if actions else None
        }








