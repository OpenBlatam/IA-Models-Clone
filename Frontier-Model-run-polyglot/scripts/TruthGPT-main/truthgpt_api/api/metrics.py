"""
API Metrics
===========

Metrics collection and monitoring for the TruthGPT API.
"""

import time
from typing import Dict, Any
from collections import defaultdict
from datetime import datetime
import threading


class MetricsCollector:
    """Collects and stores API metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self._lock = threading.Lock()
        self._request_count = defaultdict(int)
        self._request_times = defaultdict(list)
        self._error_count = defaultdict(int)
        self._model_operations = defaultdict(int)
        self._start_time = datetime.now()
        self._total_requests = 0
        self._total_errors = 0
    
    def record_request(self, endpoint: str, method: str, duration: float, status_code: int):
        """
        Record a request metric.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            duration: Request duration in seconds
            status_code: HTTP status code
        """
        with self._lock:
            key = f"{method} {endpoint}"
            self._request_count[key] += 1
            self._request_times[key].append(duration)
            self._total_requests += 1
            
            if status_code >= 400:
                self._error_count[key] += 1
                self._total_errors += 1
    
    def record_model_operation(self, operation: str, model_id: str = None):
        """
        Record a model operation.
        
        Args:
            operation: Operation type (create, compile, train, etc.)
            model_id: Optional model ID
        """
        with self._lock:
            self._model_operations[operation] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            uptime = (datetime.now() - self._start_time).total_seconds()
            
            avg_times = {}
            for key, times in self._request_times.items():
                if times:
                    avg_times[key] = {
                        "avg": sum(times) / len(times),
                        "min": min(times),
                        "max": max(times),
                        "count": len(times)
                    }
            
            return {
                "uptime_seconds": uptime,
                "total_requests": self._total_requests,
                "total_errors": self._total_errors,
                "error_rate": self._total_errors / self._total_requests if self._total_requests > 0 else 0,
                "requests_per_second": self._total_requests / uptime if uptime > 0 else 0,
                "endpoint_stats": {
                    key: {
                        "count": self._request_count[key],
                        "errors": self._error_count.get(key, 0),
                        **avg_times.get(key, {})
                    }
                    for key in self._request_count.keys()
                },
                "model_operations": dict(self._model_operations),
                "start_time": self._start_time.isoformat()
            }
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._request_count.clear()
            self._request_times.clear()
            self._error_count.clear()
            self._model_operations.clear()
            self._total_requests = 0
            self._total_errors = 0
            self._start_time = datetime.now()


metrics_collector = MetricsCollector()

