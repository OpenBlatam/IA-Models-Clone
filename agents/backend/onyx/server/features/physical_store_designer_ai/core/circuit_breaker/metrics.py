"""
Circuit Breaker Metrics

Defines metrics collection and statistics for circuit breaker.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker observability"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    state_changes: int = 0
    last_state_change: Optional[datetime] = None
    current_failure_count: int = 0
    current_success_count: int = 0
    # Advanced metrics
    response_times: List[float] = field(default_factory=list)  # Track response times
    retry_count: int = 0  # Total retries attempted
    fallback_count: int = 0  # Total fallbacks used
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate"""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        # Calculate response time statistics
        response_time_stats = self._calculate_response_time_stats()
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "rejected_requests": self.rejected_requests,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "state_changes": self.state_changes,
            "last_state_change": self.last_state_change.isoformat() if self.last_state_change else None,
            "current_failure_count": self.current_failure_count,
            "current_success_count": self.current_success_count,
            "retry_count": self.retry_count,
            "fallback_count": self.fallback_count,
            **response_time_stats
        }
    
    def _calculate_response_time_stats(self) -> Dict[str, Any]:
        """Calculate response time statistics"""
        if not self.response_times:
            return {
                "avg_response_time": 0.0,
                "p50_response_time": 0.0,
                "p95_response_time": 0.0,
                "p99_response_time": 0.0,
                "min_response_time": 0.0,
                "max_response_time": 0.0
            }
        
        sorted_times = sorted(self.response_times)
        n = len(sorted_times)
        
        return {
            "avg_response_time": sum(sorted_times) / n,
            "p50_response_time": sorted_times[int(n * 0.50)] if n > 0 else 0.0,
            "p95_response_time": sorted_times[int(n * 0.95)] if n > 0 else 0.0,
            "p99_response_time": sorted_times[int(n * 0.99)] if n > 0 else 0.0,
            "min_response_time": sorted_times[0] if n > 0 else 0.0,
            "max_response_time": sorted_times[-1] if n > 0 else 0.0
        }
    
    def record_response_time(self, duration: float):
        """Record response time for metrics"""
        self.response_times.append(duration)
        # Keep only last 1000 response times to avoid memory issues
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]

