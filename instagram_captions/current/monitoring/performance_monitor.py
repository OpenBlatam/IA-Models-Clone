"""
Performance Monitor for Instagram Captions API v10.0

Optimized performance monitoring with reduced memory footprint.
"""

import time
import statistics
from typing import Dict, Any, List, Optional
from collections import deque
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    response_time: float
    timestamp: float
    endpoint: str
    method: str
    status_code: int
    user_agent: str

class PerformanceMonitor:
    """Optimized performance monitoring."""
    
    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.metrics: deque = deque(maxlen=max_samples)
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
    
    def record_request(self, response_time: float, endpoint: str, 
                      method: str, status_code: int, user_agent: str = ""):
        """Record a single request metric."""
        metric = PerformanceMetrics(
            response_time=response_time,
            timestamp=time.time(),
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            user_agent=user_agent
        )
        
        self.metrics.append(metric)
        self.request_count += 1
        
        if status_code >= 400:
            self.error_count += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics:
            return {"message": "No metrics available"}
        
        response_times = [m.response_time for m in self.metrics]
        
        return {
            "total_requests": self.request_count,
            "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0,
            "uptime_seconds": time.time() - self.start_time,
            "response_times": {
                "avg": statistics.mean(response_times),
                "median": statistics.median(response_times),
                "p95": self._percentile(response_times, 95),
                "p99": self._percentile(response_times, 99),
                "min": min(response_times),
                "max": max(response_times)
            },
            "endpoints": self._get_endpoint_stats()
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile efficiently."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index % 1)
    
    def _get_endpoint_stats(self) -> Dict[str, Any]:
        """Get endpoint-specific statistics."""
        endpoint_stats = {}
        
        for metric in self.metrics:
            if metric.endpoint not in endpoint_stats:
                endpoint_stats[metric.endpoint] = {
                    "count": 0,
                    "avg_response_time": 0,
                    "total_response_time": 0
                }
            
            stats = endpoint_stats[metric.endpoint]
            stats["count"] += 1
            stats["total_response_time"] += metric.response_time
            stats["avg_response_time"] = stats["total_response_time"] / stats["count"]
        
        return endpoint_stats
    
    def clear_metrics(self):
        """Clear all stored metrics."""
        self.metrics.clear()
        self.request_count = 0
        self.error_count = 0
        self.start_time = time.time()






