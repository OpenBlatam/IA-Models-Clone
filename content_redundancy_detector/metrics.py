"""
Metrics and monitoring system
"""

import time
import threading
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Metrics for individual requests"""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class SystemMetrics:
    """System-wide metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    requests_per_minute: float = 0.0
    error_rate: float = 0.0


class MetricsCollector:
    """Collects and manages system metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._lock = threading.Lock()
        self._request_history: deque = deque(maxlen=max_history)
        self._endpoint_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'errors': 0,
            'last_request': 0.0
        })
        self._start_time = time.time()
    
    def record_request(self, metrics: RequestMetrics) -> None:
        """Record a request metric"""
        with self._lock:
            self._request_history.append(metrics)
            
            # Update endpoint metrics
            endpoint_key = f"{metrics.method}:{metrics.endpoint}"
            endpoint_data = self._endpoint_metrics[endpoint_key]
            endpoint_data['count'] += 1
            endpoint_data['total_time'] += metrics.response_time
            endpoint_data['last_request'] = metrics.timestamp
            
            if metrics.status_code >= 400:
                endpoint_data['errors'] += 1
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        with self._lock:
            if not self._request_history:
                return SystemMetrics()
            
            current_time = time.time()
            recent_requests = [
                req for req in self._request_history
                if current_time - req.timestamp <= 60  # Last minute
            ]
            
            total_requests = len(self._request_history)
            successful_requests = sum(1 for req in self._request_history if req.status_code < 400)
            failed_requests = total_requests - successful_requests
            
            total_response_time = sum(req.response_time for req in self._request_history)
            average_response_time = total_response_time / total_requests if total_requests > 0 else 0.0
            
            requests_per_minute = len(recent_requests)
            error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0.0
            
            return SystemMetrics(
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                average_response_time=average_response_time,
                requests_per_minute=requests_per_minute,
                error_rate=error_rate
            )
    
    def get_endpoint_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics by endpoint"""
        with self._lock:
            result = {}
            for endpoint, data in self._endpoint_metrics.items():
                avg_time = data['total_time'] / data['count'] if data['count'] > 0 else 0.0
                error_rate = (data['errors'] / data['count'] * 100) if data['count'] > 0 else 0.0
                
                result[endpoint] = {
                    'count': data['count'],
                    'average_response_time': avg_time,
                    'error_rate': error_rate,
                    'errors': data['errors'],
                    'last_request': data['last_request']
                }
            
            return result
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health-related metrics"""
        system_metrics = self.get_system_metrics()
        uptime = time.time() - self._start_time
        
        # Determine health status
        health_status = "healthy"
        if system_metrics.error_rate > 10:
            health_status = "degraded"
        if system_metrics.error_rate > 25:
            health_status = "unhealthy"
        
        return {
            "status": health_status,
            "uptime": uptime,
            "total_requests": system_metrics.total_requests,
            "error_rate": system_metrics.error_rate,
            "average_response_time": system_metrics.average_response_time,
            "requests_per_minute": system_metrics.requests_per_minute
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        with self._lock:
            self._request_history.clear()
            self._endpoint_metrics.clear()
            self._start_time = time.time()
        logger.info("Metrics reset")


# Global metrics collector
metrics_collector = MetricsCollector()


def record_request_metric(endpoint: str, method: str, status_code: int, response_time: float) -> None:
    """Record a request metric"""
    metrics = RequestMetrics(
        endpoint=endpoint,
        method=method,
        status_code=status_code,
        response_time=response_time
    )
    metrics_collector.record_request(metrics)


def get_system_metrics() -> Dict[str, Any]:
    """Get system metrics"""
    system_metrics = metrics_collector.get_system_metrics()
    return {
        "total_requests": system_metrics.total_requests,
        "successful_requests": system_metrics.successful_requests,
        "failed_requests": system_metrics.failed_requests,
        "average_response_time": round(system_metrics.average_response_time, 4),
        "requests_per_minute": system_metrics.requests_per_minute,
        "error_rate": round(system_metrics.error_rate, 2)
    }


def get_endpoint_metrics() -> Dict[str, Dict[str, Any]]:
    """Get endpoint metrics"""
    return metrics_collector.get_endpoint_metrics()


def get_health_metrics() -> Dict[str, Any]:
    """Get health metrics"""
    return metrics_collector.get_health_metrics()


def reset_metrics() -> None:
    """Reset all metrics"""
    metrics_collector.reset_metrics()


