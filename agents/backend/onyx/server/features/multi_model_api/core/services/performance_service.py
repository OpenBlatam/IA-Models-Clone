"""
Performance service for Multi-Model API
Tracks and optimizes performance metrics
"""

import time
import logging
from typing import Dict, Any, Optional, List
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics at a point in time"""
    timestamp: float
    requests_per_second: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    cache_hit_rate: float


class PerformanceService:
    """Service for tracking and optimizing performance"""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize performance service
        
        Args:
            window_size: Size of sliding window for metrics
        """
        self.window_size = window_size
        self.latency_history: deque = deque(maxlen=window_size)
        self.error_history: deque = deque(maxlen=window_size)
        self.cache_hit_history: deque = deque(maxlen=window_size)
        self.request_timestamps: deque = deque(maxlen=window_size * 10)
        self.snapshots: List[PerformanceSnapshot] = []
    
    def record_request(
        self,
        latency_ms: float,
        is_error: bool = False,
        cache_hit: bool = False
    ) -> None:
        """
        Record a request for performance tracking
        
        Args:
            latency_ms: Request latency in milliseconds
            is_error: Whether request resulted in error
            cache_hit: Whether cache was hit
        """
        current_time = time.time()
        
        self.latency_history.append(latency_ms)
        self.error_history.append(is_error)
        self.cache_hit_history.append(cache_hit)
        self.request_timestamps.append(current_time)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics
        
        Returns:
            Dictionary with current metrics
        """
        if not self.latency_history:
            return {
                "requests_per_second": 0.0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "error_rate": 0.0,
                "cache_hit_rate": 0.0,
                "total_requests": 0
            }
        
        # Calculate requests per second
        if len(self.request_timestamps) > 1:
            time_span = self.request_timestamps[-1] - self.request_timestamps[0]
            requests_per_second = len(self.request_timestamps) / time_span if time_span > 0 else 0.0
        else:
            requests_per_second = 0.0
        
        # Calculate latency percentiles
        sorted_latencies = sorted(self.latency_history)
        n = len(sorted_latencies)
        
        avg_latency = sum(sorted_latencies) / n if n > 0 else 0.0
        p95_latency = sorted_latencies[int(n * 0.95)] if n > 0 else 0.0
        p99_latency = sorted_latencies[int(n * 0.99)] if n > 0 else 0.0
        
        # Calculate error rate
        error_count = sum(self.error_history)
        error_rate = (error_count / len(self.error_history) * 100) if self.error_history else 0.0
        
        # Calculate cache hit rate
        cache_hit_count = sum(self.cache_hit_history)
        cache_hit_rate = (cache_hit_count / len(self.cache_hit_history) * 100) if self.cache_hit_history else 0.0
        
        return {
            "requests_per_second": round(requests_per_second, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "p99_latency_ms": round(p99_latency, 2),
            "error_rate": round(error_rate, 2),
            "cache_hit_rate": round(cache_hit_rate, 2),
            "total_requests": len(self.latency_history)
        }
    
    def take_snapshot(self) -> PerformanceSnapshot:
        """
        Take a snapshot of current performance metrics
        
        Returns:
            PerformanceSnapshot
        """
        metrics = self.get_current_metrics()
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            requests_per_second=metrics["requests_per_second"],
            avg_latency_ms=metrics["avg_latency_ms"],
            p95_latency_ms=metrics["p95_latency_ms"],
            p99_latency_ms=metrics["p99_latency_ms"],
            error_rate=metrics["error_rate"],
            cache_hit_rate=metrics["cache_hit_rate"]
        )
        self.snapshots.append(snapshot)
        
        # Keep only last 100 snapshots
        if len(self.snapshots) > 100:
            self.snapshots = self.snapshots[-100:]
        
        return snapshot
    
    def get_snapshots(self, last_n: int = 10) -> List[PerformanceSnapshot]:
        """
        Get recent performance snapshots
        
        Args:
            last_n: Number of recent snapshots to return
            
        Returns:
            List of PerformanceSnapshot
        """
        return self.snapshots[-last_n:] if len(self.snapshots) > last_n else self.snapshots
    
    def detect_performance_issues(self) -> List[Dict[str, Any]]:
        """
        Detect potential performance issues
        
        Returns:
            List of detected issues with recommendations
        """
        issues = []
        metrics = self.get_current_metrics()
        
        # High error rate
        if metrics["error_rate"] > 10.0:
            issues.append({
                "type": "high_error_rate",
                "severity": "high",
                "current_value": metrics["error_rate"],
                "threshold": 10.0,
                "recommendation": "Investigate error sources and consider circuit breakers"
            })
        
        # High latency
        if metrics["p95_latency_ms"] > 5000.0:
            issues.append({
                "type": "high_latency",
                "severity": "medium",
                "current_value": metrics["p95_latency_ms"],
                "threshold": 5000.0,
                "recommendation": "Consider optimizing model execution or increasing timeout"
            })
        
        # Low cache hit rate
        if metrics["cache_hit_rate"] < 5.0 and metrics["total_requests"] > 100:
            issues.append({
                "type": "low_cache_hit_rate",
                "severity": "low",
                "current_value": metrics["cache_hit_rate"],
                "threshold": 5.0,
                "recommendation": "Review cache strategy and TTL settings"
            })
        
        return issues
    
    def reset(self) -> None:
        """Reset all performance metrics"""
        self.latency_history.clear()
        self.error_history.clear()
        self.cache_hit_history.clear()
        self.request_timestamps.clear()
        self.snapshots.clear()
        logger.info("Performance metrics reset")


# Global performance service instance
_performance_service: Optional['PerformanceService'] = None


def get_performance_service() -> PerformanceService:
    """Get or create global performance service instance"""
    global _performance_service
    if _performance_service is None:
        _performance_service = PerformanceService()
    return _performance_service

