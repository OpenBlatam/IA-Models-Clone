"""
Performance metrics utilities for tracking system performance.
"""

from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)

# Global metrics instance
_metrics_instance: Optional['PerformanceMetrics'] = None


class PerformanceMetrics:
    """Performance metrics collector."""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize performance metrics.
        
        Args:
            max_history: Maximum number of historical records
        """
        self.max_history = max_history
        
        # Request metrics
        self.request_times: deque = deque(maxlen=max_history)
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.request_errors: Dict[str, int] = defaultdict(int)
        
        # Query metrics
        self.query_times: deque = deque(maxlen=max_history)
        self.query_counts: Dict[str, int] = defaultdict(int)
        self.query_errors: Dict[str, int] = defaultdict(int)
        
        # Cache metrics
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        
        # Error metrics
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.error_times: deque = deque(maxlen=max_history)
    
    def record_request(
        self,
        method: str,
        path: str,
        duration: float,
        status_code: int
    ) -> None:
        """
        Record request metrics.
        
        Args:
            method: HTTP method
            path: Request path
            duration: Request duration in seconds
            status_code: HTTP status code
        """
        self.request_times.append(duration)
        self.request_counts[f"{method} {path}"] += 1
        
        if status_code >= 400:
            self.request_errors[f"{method} {path}"] += 1
    
    def record_query(
        self,
        query_type: str,
        duration: float,
        success: bool = True
    ) -> None:
        """
        Record query metrics.
        
        Args:
            query_type: Query type identifier
            duration: Query duration in seconds
            success: Whether query succeeded
        """
        self.query_times.append(duration)
        self.query_counts[query_type] += 1
        
        if not success:
            self.query_errors[query_type] += 1
    
    def record_cache_hit(self) -> None:
        """Record cache hit."""
        self.cache_hits += 1
    
    def record_cache_miss(self) -> None:
        """Record cache miss."""
        self.cache_misses += 1
    
    def record_error(self, error_type: str, error_message: str) -> None:
        """
        Record error metrics.
        
        Args:
            error_type: Error type
            error_message: Error message
        """
        self.error_counts[error_type] += 1
        self.error_times.append({
            "type": error_type,
            "message": error_message,
            "timestamp": datetime.now()
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics.
        
        Returns:
            Dictionary with summary metrics
        """
        avg_request_time = (
            sum(self.request_times) / len(self.request_times)
            if self.request_times else 0
        )
        
        avg_query_time = (
            sum(self.query_times) / len(self.query_times)
            if self.query_times else 0
        )
        
        total_requests = sum(self.request_counts.values())
        total_queries = sum(self.query_counts.values())
        total_errors = sum(self.error_counts.values())
        
        cache_hit_rate = (
            self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0 else 0
        )
        
        return {
            "requests": {
                "total": total_requests,
                "average_time": round(avg_request_time, 3),
                "error_count": sum(self.request_errors.values())
            },
            "queries": {
                "total": total_queries,
                "average_time": round(avg_query_time, 3),
                "error_count": sum(self.query_errors.values())
            },
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": round(cache_hit_rate, 3)
            },
            "errors": {
                "total": total_errors,
                "by_type": dict(self.error_counts)
            }
        }
    
    def get_request_stats(self) -> Dict[str, Any]:
        """
        Get request statistics.
        
        Returns:
            Dictionary with request stats
        """
        return {
            "counts": dict(self.request_counts),
            "errors": dict(self.request_errors),
            "average_time": (
                sum(self.request_times) / len(self.request_times)
                if self.request_times else 0
            ),
            "min_time": min(self.request_times) if self.request_times else 0,
            "max_time": max(self.request_times) if self.request_times else 0
        }
    
    def get_query_stats(self) -> Dict[str, Any]:
        """
        Get query statistics.
        
        Returns:
            Dictionary with query stats
        """
        return {
            "counts": dict(self.query_counts),
            "errors": dict(self.query_errors),
            "average_time": (
                sum(self.query_times) / len(self.query_times)
                if self.query_times else 0
            ),
            "min_time": min(self.query_times) if self.query_times else 0,
            "max_time": max(self.query_times) if self.query_times else 0
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "total": total,
            "hit_rate": round(hit_rate, 3)
        }
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get error statistics.
        
        Returns:
            Dictionary with error stats
        """
        return {
            "counts": dict(self.error_counts),
            "total": sum(self.error_counts.values()),
            "recent": list(self.error_times)[-10:]  # Last 10 errors
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.request_times.clear()
        self.request_counts.clear()
        self.request_errors.clear()
        self.query_times.clear()
        self.query_counts.clear()
        self.query_errors.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.error_counts.clear()
        self.error_times.clear()


def get_metrics(max_history: int = 1000) -> PerformanceMetrics:
    """
    Get or create global metrics instance.
    
    Args:
        max_history: Maximum number of historical records
        
    Returns:
        Metrics instance
    """
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = PerformanceMetrics(max_history=max_history)
        logger.info(f"Performance metrics initialized: max_history={max_history}")
    return _metrics_instance




