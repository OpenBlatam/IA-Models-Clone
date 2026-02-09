"""
Metrics and monitoring utilities for Robot Maintenance AI.
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects metrics for API usage and system performance.
    """
    
    def __init__(self):
        self.request_count = defaultdict(int)
        self.request_times = defaultdict(list)
        self.error_count = defaultdict(int)
        self.cache_hits = 0
        self.cache_misses = 0
        self.start_time = datetime.now()
    
    def record_request(self, endpoint: str, duration: float, success: bool = True):
        """
        Record a request metric.
        
        Args:
            endpoint: API endpoint name
            duration: Request duration in seconds
            success: Whether request was successful
        """
        self.request_count[endpoint] += 1
        self.request_times[endpoint].append(duration)
        
        if not success:
            self.error_count[endpoint] += 1
        
        if len(self.request_times[endpoint]) > 1000:
            self.request_times[endpoint] = self.request_times[endpoint][-1000:]
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_misses += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get aggregated statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "total_requests": sum(self.request_count.values()),
            "total_errors": sum(self.error_count.values()),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0 else 0.0
            ),
            "endpoints": {}
        }
        
        for endpoint in self.request_count:
            times = self.request_times[endpoint]
            if times:
                stats["endpoints"][endpoint] = {
                    "count": self.request_count[endpoint],
                    "errors": self.error_count[endpoint],
                    "avg_duration": sum(times) / len(times),
                    "min_duration": min(times),
                    "max_duration": max(times),
                    "error_rate": (
                        self.error_count[endpoint] / self.request_count[endpoint]
                        if self.request_count[endpoint] > 0 else 0.0
                    )
                }
        
        return stats
    
    def reset(self):
        """Reset all metrics."""
        self.request_count.clear()
        self.request_times.clear()
        self.error_count.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.start_time = datetime.now()


metrics_collector = MetricsCollector()






