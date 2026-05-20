"""
Metrics and Performance Tracking
================================

Advanced metrics collection with observer pattern and exporters.

Refactored with:
- Observer pattern for metrics events
- MetricsExporter abstraction
- TimeWindow for sliding window calculations
- Enhanced performance percentiles
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable, Protocol
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class TaskMetrics:
    """Metrics for a single task."""
    task_id: str
    start_time: float
    end_time: Optional[float] = None
    generations: int = 0
    masks_found: int = 0
    status: str = "pending"
    error: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def is_completed(self) -> bool:
        """Check if task is completed."""
        return self.status == "completed"
    
    @property
    def is_failed(self) -> bool:
        """Check if task failed."""
        return self.status == "failed"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "generations": self.generations,
            "masks_found": self.masks_found,
            "status": self.status,
            "error": self.error,
            "labels": self.labels,
        }


class MetricsObserver(Protocol):
    """Protocol for metrics observers."""
    
    def on_task_start(self, task_id: str, labels: Dict[str, str] = None) -> None:
        """Called when task starts."""
        ...
    
    def on_task_complete(
        self, 
        task_id: str, 
        duration: float,
        generations: int,
        masks_found: int,
    ) -> None:
        """Called when task completes."""
        ...
    
    def on_task_fail(self, task_id: str, error: str) -> None:
        """Called when task fails."""
        ...


class MetricsExporter(ABC):
    """Abstract base class for metrics exporters."""
    
    @abstractmethod
    def export(self, metrics: Dict[str, Any]) -> None:
        """Export metrics to destination."""
        pass
    
    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered metrics."""
        pass


class LoggingMetricsExporter(MetricsExporter):
    """Exporter that logs metrics."""
    
    def export(self, metrics: Dict[str, Any]) -> None:
        """Log metrics."""
        logger.info(f"Metrics: {metrics}")
    
    def flush(self) -> None:
        """No-op for logging exporter."""
        pass


class PrometheusMetricsExporter(MetricsExporter):
    """Exporter for Prometheus format (stub for extension)."""
    
    def __init__(self, endpoint: str = "/metrics"):
        self.endpoint = endpoint
        self._buffer: List[str] = []
    
    def export(self, metrics: Dict[str, Any]) -> None:
        """Format metrics for Prometheus."""
        lines = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                lines.append(f"agent_{key} {value}")
        self._buffer.extend(lines)
    
    def flush(self) -> None:
        """Flush buffer (would typically send to push gateway)."""
        if self._buffer:
            logger.debug(f"Would export {len(self._buffer)} metrics to Prometheus")
            self._buffer.clear()
    
    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format."""
        return "\n".join(self._buffer)


@dataclass
class TimeWindow:
    """
    Sliding time window for metrics aggregation.
    
    Provides efficient sliding window calculations.
    """
    window_size_seconds: int
    _data: deque = field(default_factory=lambda: deque())
    
    def add(self, value: float, timestamp: Optional[float] = None):
        """Add value to window."""
        ts = timestamp or time.time()
        self._data.append((ts, value))
        self._cleanup()
    
    def _cleanup(self):
        """Remove expired entries."""
        cutoff = time.time() - self.window_size_seconds
        while self._data and self._data[0][0] < cutoff:
            self._data.popleft()
    
    @property
    def values(self) -> List[float]:
        """Get all values in window."""
        self._cleanup()
        return [v for _, v in self._data]
    
    @property
    def count(self) -> int:
        """Get count of values in window."""
        self._cleanup()
        return len(self._data)
    
    @property
    def sum(self) -> float:
        """Get sum of values in window."""
        return sum(self.values)
    
    @property
    def avg(self) -> float:
        """Get average of values in window."""
        vals = self.values
        return sum(vals) / len(vals) if vals else 0.0
    
    def percentile(self, p: float) -> float:
        """Get percentile of values in window."""
        vals = sorted(self.values)
        if not vals:
            return 0.0
        idx = int(len(vals) * p / 100)
        return vals[min(idx, len(vals) - 1)]


class ObserverRegistry:
    """Registry for metrics observers."""
    
    def __init__(self):
        self._observers: List[MetricsObserver] = []
    
    def register(self, observer: MetricsObserver):
        """Register an observer."""
        self._observers.append(observer)
    
    def unregister(self, observer: MetricsObserver):
        """Unregister an observer."""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def notify_task_start(self, task_id: str, labels: Dict[str, str] = None):
        """Notify all observers of task start."""
        for obs in self._observers:
            try:
                obs.on_task_start(task_id, labels)
            except Exception as e:
                logger.warning(f"Observer error on task start: {e}")
    
    def notify_task_complete(
        self,
        task_id: str,
        duration: float,
        generations: int,
        masks_found: int,
    ):
        """Notify all observers of task completion."""
        for obs in self._observers:
            try:
                obs.on_task_complete(task_id, duration, generations, masks_found)
            except Exception as e:
                logger.warning(f"Observer error on task complete: {e}")
    
    def notify_task_fail(self, task_id: str, error: str):
        """Notify all observers of task failure."""
        for obs in self._observers:
            try:
                obs.on_task_fail(task_id, error)
            except Exception as e:
                logger.warning(f"Observer error on task fail: {e}")


class MetricsCollector:
    """
    Collects and aggregates metrics for the agent.
    
    Refactored with:
    - Observer pattern for extensibility
    - TimeWindow for efficient sliding window calculations
    - Exporter abstraction for multiple output formats
    - Factory methods for common configurations
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics collector.
        
        Args:
            max_history: Maximum number of tasks to keep in history
        """
        self.max_history = max_history
        self.task_metrics: Dict[str, TaskMetrics] = {}
        self.task_history: deque = deque(maxlen=max_history)
        
        # Aggregated metrics
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_generations = 0
        self.total_masks = 0
        
        # Time windows for different periods
        self.hourly_durations = TimeWindow(3600)  # Last hour
        self.daily_durations = TimeWindow(86400)  # Last day
        
        # Timing metrics
        self.task_durations: deque = deque(maxlen=max_history)
        self.generation_times: deque = deque(maxlen=max_history)
        
        # Error tracking
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.recent_errors: deque = deque(maxlen=100)
        
        # Observer registry
        self._observers = ObserverRegistry()
        
        # Exporters
        self._exporters: List[MetricsExporter] = []
    
    # === Factory Methods ===
    
    @classmethod
    def create_with_logging(cls, max_history: int = 1000) -> "MetricsCollector":
        """Create collector with logging exporter."""
        collector = cls(max_history)
        collector.add_exporter(LoggingMetricsExporter())
        return collector
    
    @classmethod
    def create_with_prometheus(
        cls, 
        max_history: int = 1000,
        endpoint: str = "/metrics"
    ) -> "MetricsCollector":
        """Create collector with Prometheus exporter."""
        collector = cls(max_history)
        collector.add_exporter(PrometheusMetricsExporter(endpoint))
        return collector
    
    # === Observer Management ===
    
    def add_observer(self, observer: MetricsObserver):
        """Add a metrics observer."""
        self._observers.register(observer)
    
    def remove_observer(self, observer: MetricsObserver):
        """Remove a metrics observer."""
        self._observers.unregister(observer)
    
    # === Exporter Management ===
    
    def add_exporter(self, exporter: MetricsExporter):
        """Add a metrics exporter."""
        self._exporters.append(exporter)
    
    def remove_exporter(self, exporter: MetricsExporter):
        """Remove a metrics exporter."""
        if exporter in self._exporters:
            self._exporters.remove(exporter)
    
    def export_metrics(self):
        """Export current metrics to all exporters."""
        metrics = self.get_summary()
        for exporter in self._exporters:
            try:
                exporter.export(metrics)
            except Exception as e:
                logger.error(f"Export error: {e}")
    
    # === Core Metrics Operations ===
    
    def start_task(self, task_id: str, labels: Dict[str, str] = None) -> None:
        """Record task start."""
        self.task_metrics[task_id] = TaskMetrics(
            task_id=task_id,
            start_time=time.time(),
            status="processing",
            labels=labels or {},
        )
        self.total_tasks += 1
        
        # Notify observers
        self._observers.notify_task_start(task_id, labels)
    
    def complete_task(
        self,
        task_id: str,
        generations: int = 0,
        masks_found: int = 0
    ) -> None:
        """Record task completion."""
        if task_id not in self.task_metrics:
            logger.warning(f"Task {task_id} not found in metrics")
            return
        
        metric = self.task_metrics[task_id]
        metric.end_time = time.time()
        metric.status = "completed"
        metric.generations = generations
        metric.masks_found = masks_found
        
        # Update aggregates
        self.completed_tasks += 1
        self.total_generations += generations
        self.total_masks += masks_found
        
        # Record duration
        if metric.duration:
            self.task_durations.append(metric.duration)
            self.hourly_durations.add(metric.duration)
            self.daily_durations.add(metric.duration)
        
        # Add to history
        self.task_history.append(metric)
        
        # Notify observers
        self._observers.notify_task_complete(
            task_id, 
            metric.duration or 0, 
            generations, 
            masks_found
        )
    
    def fail_task(self, task_id: str, error: str) -> None:
        """Record task failure."""
        if task_id not in self.task_metrics:
            logger.warning(f"Task {task_id} not found in metrics")
            return
        
        metric = self.task_metrics[task_id]
        metric.end_time = time.time()
        metric.status = "failed"
        metric.error = error
        
        # Update aggregates
        self.failed_tasks += 1
        
        # Track error
        error_type = type(error).__name__ if hasattr(error, '__class__') else "Unknown"
        self.error_counts[error_type] += 1
        self.recent_errors.append({
            "task_id": task_id,
            "error": str(error),
            "timestamp": datetime.now().isoformat(),
        })
        
        # Add to history
        self.task_history.append(metric)
        
        # Notify observers
        self._observers.notify_task_fail(task_id, error)
    
    def record_generation_time(self, duration: float) -> None:
        """Record generation time."""
        self.generation_times.append(duration)
    
    # === Reporting Methods ===
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        success_rate = (
            self.completed_tasks / self.total_tasks
            if self.total_tasks > 0
            else 0.0
        )
        
        avg_duration = (
            sum(self.task_durations) / len(self.task_durations)
            if self.task_durations
            else 0.0
        )
        
        avg_generations = (
            self.total_generations / self.completed_tasks
            if self.completed_tasks > 0
            else 0.0
        )
        
        avg_masks = (
            self.total_masks / self.completed_tasks
            if self.completed_tasks > 0
            else 0.0
        )
        
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": success_rate,
            "total_generations": self.total_generations,
            "total_masks": self.total_masks,
            "avg_task_duration": avg_duration,
            "avg_generations_per_task": avg_generations,
            "avg_masks_per_task": avg_masks,
            "error_counts": dict(self.error_counts),
            "recent_errors_count": len(self.recent_errors),
            "hourly_avg_duration": self.hourly_durations.avg,
            "daily_avg_duration": self.daily_durations.avg,
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics with percentiles."""
        if not self.task_durations:
            return {
                "min_duration": 0.0,
                "max_duration": 0.0,
                "avg_duration": 0.0,
                "median_duration": 0.0,
                "p95_duration": 0.0,
                "p99_duration": 0.0,
                "hourly_p95": 0.0,
                "daily_p95": 0.0,
            }
        
        sorted_durations = sorted(self.task_durations)
        n = len(sorted_durations)
        
        return {
            "min_duration": min(sorted_durations),
            "max_duration": max(sorted_durations),
            "avg_duration": sum(sorted_durations) / n,
            "median_duration": sorted_durations[n // 2],
            "p95_duration": sorted_durations[int(n * 0.95)] if n > 0 else 0.0,
            "p99_duration": sorted_durations[int(n * 0.99)] if n > 0 else 0.0,
            "hourly_p95": self.hourly_durations.percentile(95),
            "daily_p95": self.daily_durations.percentile(95),
        }
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent errors."""
        return list(self.recent_errors)[-limit:]
    
    def clear_old_metrics(self, max_age_hours: int = 24) -> None:
        """Clear metrics older than specified hours."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        old_tasks = [
            task_id for task_id, metric in self.task_metrics.items()
            if metric.end_time and metric.end_time < cutoff_time
        ]
        for task_id in old_tasks:
            del self.task_metrics[task_id]
        
        logger.info(f"Cleared {len(old_tasks)} old task metrics")
