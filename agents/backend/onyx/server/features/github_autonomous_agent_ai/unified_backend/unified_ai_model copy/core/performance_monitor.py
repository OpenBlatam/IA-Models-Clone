"""
Performance Monitor for Unified AI Model
Provides metrics, monitoring, and statistics
"""

import time
import logging
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class MetricEntry:
    """A single metric entry."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


class PerformanceMonitor:
    """
    Performance monitoring and metrics collection.
    
    Features:
    - Request latency tracking
    - Token usage tracking
    - Error rate monitoring
    - Cache hit rate tracking
    - Custom metric support
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.timers: Dict[str, float] = {}
        
        # Request tracking
        self.request_latencies: List[float] = []
        self.max_latencies = 1000  # Keep last 1000 entries
        
        # Error tracking
        self.errors: List[Dict[str, Any]] = []
        self.max_errors = 100
        
        logger.info("Performance Monitor initialized")
    
    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a metric value."""
        self.metrics[name].append(value)
        
        # Limit stored metrics
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        self.counters[name] += value
    
    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        self.gauges[name] = value
    
    def start_timer(self, name: str) -> None:
        """Start a timer."""
        self.timers[name] = time.time()
    
    def stop_timer(self, name: str) -> Optional[float]:
        """Stop a timer and return elapsed time in ms."""
        if name not in self.timers:
            return None
        
        elapsed = (time.time() - self.timers[name]) * 1000
        del self.timers[name]
        
        self.record_metric(f"{name}_ms", elapsed)
        return elapsed
    
    def record_request(
        self,
        model: str,
        latency_ms: float,
        tokens: int = 0,
        success: bool = True,
        cached: bool = False
    ) -> None:
        """Record a request."""
        self.increment_counter("total_requests")
        
        if success:
            self.increment_counter("successful_requests")
        else:
            self.increment_counter("failed_requests")
        
        if cached:
            self.increment_counter("cache_hits")
        else:
            self.increment_counter("cache_misses")
        
        # Track latency
        self.request_latencies.append(latency_ms)
        if len(self.request_latencies) > self.max_latencies:
            self.request_latencies = self.request_latencies[-self.max_latencies:]
        
        # Track by model
        self.increment_counter(f"requests_{model}")
        self.record_metric(f"latency_{model}", latency_ms)
        
        # Track tokens
        if tokens > 0:
            self.increment_counter("total_tokens", tokens)
            self.record_metric("tokens_per_request", tokens)
    
    def record_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record an error."""
        self.increment_counter("total_errors")
        self.increment_counter(f"error_{error_type}")
        
        self.errors.append({
            "type": error_type,
            "message": error_message,
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        })
        
        if len(self.errors) > self.max_errors:
            self.errors = self.errors[-self.max_errors:]
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        if not self.request_latencies:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "p99": 0.0
            }
        
        sorted_latencies = sorted(self.request_latencies)
        n = len(sorted_latencies)
        
        return {
            "min": min(sorted_latencies),
            "max": max(sorted_latencies),
            "mean": statistics.mean(sorted_latencies),
            "median": statistics.median(sorted_latencies),
            "p95": sorted_latencies[int(n * 0.95)] if n > 20 else sorted_latencies[-1],
            "p99": sorted_latencies[int(n * 0.99)] if n > 100 else sorted_latencies[-1]
        }
    
    def get_error_rate(self) -> float:
        """Get error rate as percentage."""
        total = self.counters.get("total_requests", 0)
        if total == 0:
            return 0.0
        
        failed = self.counters.get("failed_requests", 0)
        return (failed / total) * 100
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate as percentage."""
        hits = self.counters.get("cache_hits", 0)
        misses = self.counters.get("cache_misses", 0)
        total = hits + misses
        
        if total == 0:
            return 0.0
        
        return (hits / total) * 100
    
    def get_uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        return (datetime.now() - self.start_time).total_seconds()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        latency_stats = self.get_latency_stats()
        
        return {
            "uptime_seconds": self.get_uptime_seconds(),
            "requests": {
                "total": self.counters.get("total_requests", 0),
                "successful": self.counters.get("successful_requests", 0),
                "failed": self.counters.get("failed_requests", 0),
                "error_rate": self.get_error_rate()
            },
            "cache": {
                "hits": self.counters.get("cache_hits", 0),
                "misses": self.counters.get("cache_misses", 0),
                "hit_rate": self.get_cache_hit_rate()
            },
            "latency": latency_stats,
            "tokens": {
                "total": self.counters.get("total_tokens", 0),
                "average_per_request": (
                    self.counters.get("total_tokens", 0) / 
                    max(self.counters.get("total_requests", 1), 1)
                )
            },
            "errors": {
                "total": self.counters.get("total_errors", 0),
                "recent": self.errors[-5:] if self.errors else []
            },
            "counters": dict(self.counters),
            "gauges": dict(self.gauges)
        }
    
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a specific metric."""
        values = self.metrics.get(name, [])
        
        if not values:
            return {
                "count": 0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "sum": 0.0
            }
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "sum": sum(values)
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.counters.clear()
        self.gauges.clear()
        self.timers.clear()
        self.request_latencies.clear()
        self.errors.clear()
        self.start_time = datetime.now()
        
        logger.info("Performance Monitor reset")


# Singleton instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor



