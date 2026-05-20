"""
Metrics Collector for Contabilidad Mexicana AI
==============================================

Refactored with:
- MetricType enum for categorization
- MetricObserver protocol for extensibility
- Dataclass for metric snapshots
- TimeWindow for sliding calculations
"""

import logging
import time
from typing import Dict, Any, Optional, List, Protocol, Callable
from collections import defaultdict, deque
from datetime import datetime, timedelta
from threading import Lock
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    SERVICE_CALL = "service_call"
    CACHE = "cache"
    REGIMEN = "regimen"
    IMPUESTO = "impuesto"
    ERROR = "error"


class MetricEvent(Enum):
    """Metric events."""
    SERVICE_CALLED = "service_called"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class ServiceMetricSnapshot:
    """Snapshot of service metrics."""
    count: int = 0
    success: int = 0
    errors: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        return (self.success / self.count * 100) if self.count > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "count": self.count,
            "success": self.success,
            "errors": self.errors,
            "success_rate": f"{self.success_rate:.2f}%",
            "avg_time": self.avg_time,
            "min_time": self.min_time if self.min_time != float('inf') else 0,
            "max_time": self.max_time,
        }


@dataclass
class CacheMetricSnapshot:
    """Snapshot of cache metrics."""
    hits: int = 0
    misses: int = 0
    
    @property
    def total_requests(self) -> int:
        return self.hits + self.misses
    
    @property
    def hit_rate(self) -> float:
        return (self.hits / self.total_requests * 100) if self.total_requests > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": self.total_requests,
            "hit_rate": f"{self.hit_rate:.2f}%",
        }


class MetricObserver(Protocol):
    """Protocol for metric observers."""
    
    def on_metric(self, metric_type: MetricType, data: Dict[str, Any]) -> None:
        """Called when a metric is recorded."""
        ...


class BaseMetricExporter(ABC):
    """Abstract base class for metric exporters."""
    
    @abstractmethod
    def export(self, metrics: Dict[str, Any]) -> None:
        """Export metrics."""
        pass


class LoggingMetricExporter(BaseMetricExporter):
    """Exports metrics to log."""
    
    def export(self, metrics: Dict[str, Any]) -> None:
        logger.info(f"Metrics: {metrics}")


class ObserverRegistry:
    """Registry for metric observers."""
    
    def __init__(self):
        self._observers: List[MetricObserver] = []
    
    def register(self, observer: MetricObserver):
        """Register an observer."""
        self._observers.append(observer)
    
    def notify(self, metric_type: MetricType, data: Dict[str, Any]):
        """Notify all observers."""
        for observer in self._observers:
            try:
                observer.on_metric(metric_type, data)
            except Exception as e:
                logger.warning(f"Observer error: {e}")


@dataclass
class TimeWindow:
    """Sliding time window for metric aggregation."""
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
    def count(self) -> int:
        self._cleanup()
        return len(self._data)
    
    @property
    def avg(self) -> float:
        self._cleanup()
        if not self._data:
            return 0.0
        return sum(v for _, v in self._data) / len(self._data)


class MetricsCollector:
    """
    Collector of metrics for the system.
    
    Refactored with:
    - Observer pattern for extensibility
    - Dataclass snapshots for cleaner data
    - TimeWindow for sliding calculations
    - Factory methods
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.lock = Lock()
        self.start_time = datetime.now()
        
        # Service metrics with snapshot pattern
        self._service_metrics: Dict[str, ServiceMetricSnapshot] = defaultdict(ServiceMetricSnapshot)
        self._response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        
        # Other metrics
        self._regimen_metrics: Dict[str, int] = defaultdict(int)
        self._impuesto_metrics: Dict[str, int] = defaultdict(int)
        self._cache_metrics = CacheMetricSnapshot()
        self._error_metrics: Dict[str, int] = defaultdict(int)
        self._hourly_usage: Dict[str, int] = defaultdict(int)
        
        # Observer registry
        self._observers = ObserverRegistry()
        
        # Exporters
        self._exporters: List[BaseMetricExporter] = []
        
        # Time windows for real-time metrics
        self._hourly_response_times = TimeWindow(3600)
    
    # === Factory Methods ===
    
    @classmethod
    def create_with_logging(cls, max_history: int = 1000) -> "MetricsCollector":
        """Create collector with logging exporter."""
        collector = cls(max_history)
        collector.add_exporter(LoggingMetricExporter())
        return collector
    
    # === Observer/Exporter Management ===
    
    def add_observer(self, observer: MetricObserver):
        """Add a metric observer."""
        self._observers.register(observer)
    
    def add_exporter(self, exporter: BaseMetricExporter):
        """Add a metric exporter."""
        self._exporters.append(exporter)
    
    def export_metrics(self):
        """Export current metrics to all exporters."""
        metrics = self.get_overall_stats()
        for exporter in self._exporters:
            try:
                exporter.export(metrics)
            except Exception as e:
                logger.error(f"Export error: {e}")
    
    # === Recording Methods ===
    
    def record_service_call(
        self,
        service_name: str,
        duration: float,
        success: bool = True,
        error_type: Optional[str] = None
    ) -> None:
        """Record a service call."""
        with self.lock:
            # Get or create snapshot
            if service_name not in self._service_metrics:
                self._service_metrics[service_name] = ServiceMetricSnapshot()
            
            snapshot = self._service_metrics[service_name]
            snapshot.count += 1
            snapshot.total_time += duration
            self._response_times[service_name].append(duration)
            
            if success:
                snapshot.success += 1
            else:
                snapshot.errors += 1
                if error_type:
                    self._error_metrics[error_type] += 1
            
            # Update statistics
            snapshot.avg_time = snapshot.total_time / snapshot.count
            snapshot.min_time = min(snapshot.min_time, duration)
            snapshot.max_time = max(snapshot.max_time, duration)
            
            # Record in time window
            self._hourly_response_times.add(duration)
            
            # Record hourly usage
            hour_key = datetime.now().strftime("%Y-%m-%d %H:00")
            self._hourly_usage[hour_key] += 1
        
        # Notify observers
        self._observers.notify(MetricType.SERVICE_CALL, {
            "service": service_name,
            "duration": duration,
            "success": success,
        })
    
    def record_regimen_usage(self, regimen: str) -> None:
        """Record regime usage."""
        with self.lock:
            self._regimen_metrics[regimen] += 1
        self._observers.notify(MetricType.REGIMEN, {"regimen": regimen})
    
    def record_impuesto_usage(self, tipo_impuesto: str) -> None:
        """Record tax type usage."""
        with self.lock:
            self._impuesto_metrics[tipo_impuesto] += 1
        self._observers.notify(MetricType.IMPUESTO, {"tipo": tipo_impuesto})
    
    def record_cache_hit(self) -> None:
        """Record cache hit."""
        with self.lock:
            self._cache_metrics.hits += 1
        self._observers.notify(MetricType.CACHE, {"hit": True})
    
    def record_cache_miss(self) -> None:
        """Record cache miss."""
        with self.lock:
            self._cache_metrics.misses += 1
        self._observers.notify(MetricType.CACHE, {"hit": False})
    
    # === Reporting Methods ===
    
    def get_service_stats(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """Get service statistics."""
        with self.lock:
            if service_name:
                snapshot = self._service_metrics.get(service_name)
                return snapshot.to_dict() if snapshot else {}
            
            return {
                name: snapshot.to_dict()
                for name, snapshot in self._service_metrics.items()
            }
    
    def get_regimen_stats(self) -> Dict[str, int]:
        """Get regime usage statistics."""
        with self.lock:
            return dict(self._regimen_metrics)
    
    def get_impuesto_stats(self) -> Dict[str, int]:
        """Get tax type usage statistics."""
        with self.lock:
            return dict(self._impuesto_metrics)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return self._cache_metrics.to_dict()
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics."""
        with self.lock:
            return dict(self._error_metrics)
    
    def get_hourly_usage(self, hours: int = 24) -> Dict[str, int]:
        """Get hourly usage."""
        with self.lock:
            cutoff = datetime.now() - timedelta(hours=hours)
            return {
                hour: count
                for hour, count in self._hourly_usage.items()
                if datetime.strptime(hour, "%Y-%m-%d %H:00") >= cutoff
            }
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall system statistics."""
        with self.lock:
            total_calls = sum(s.count for s in self._service_metrics.values())
            total_success = sum(s.success for s in self._service_metrics.values())
            total_errors = sum(s.errors for s in self._service_metrics.values())
            total_time = sum(s.total_time for s in self._service_metrics.values())
            
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            return {
                "uptime_seconds": uptime,
                "uptime_hours": uptime / 3600,
                "total_calls": total_calls,
                "total_success": total_success,
                "total_errors": total_errors,
                "success_rate": (total_success / total_calls * 100) if total_calls > 0 else 0,
                "avg_response_time": (total_time / total_calls) if total_calls > 0 else 0,
                "hourly_avg_response": self._hourly_response_times.avg,
                "services": len(self._service_metrics),
                "most_used_regimen": max(self._regimen_metrics.items(), key=lambda x: x[1])[0] if self._regimen_metrics else None,
                "most_used_impuesto": max(self._impuesto_metrics.items(), key=lambda x: x[1])[0] if self._impuesto_metrics else None,
            }
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self.lock:
            self._service_metrics.clear()
            self._response_times.clear()
            self._regimen_metrics.clear()
            self._impuesto_metrics.clear()
            self._cache_metrics = CacheMetricSnapshot()
            self._error_metrics.clear()
            self._hourly_usage.clear()
            self.start_time = datetime.now()
