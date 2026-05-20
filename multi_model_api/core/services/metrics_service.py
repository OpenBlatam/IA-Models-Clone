"""
Metrics service for Multi-Model API
Tracks and aggregates metrics for observability
"""

import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    request_id: str
    strategy: str
    models_count: int
    success_count: int
    failure_count: int
    total_latency_ms: float
    cache_hit: bool
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ModelMetrics:
    """Metrics for a single model"""
    model_type: str
    call_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    last_called: Optional[datetime] = None


class MetricsService:
    """Service for tracking and aggregating metrics"""
    
    def __init__(self):
        """Initialize metrics service"""
        self.request_metrics: List[RequestMetrics] = []
        self.model_metrics: Dict[str, ModelMetrics] = defaultdict(lambda: ModelMetrics(model_type=""))
        self._max_requests = 1000  # Keep last 1000 requests
        self._lock = None  # Would use asyncio.Lock in async context
    
    def record_request(
        self,
        request_id: str,
        strategy: str,
        models_count: int,
        success_count: int,
        failure_count: int,
        total_latency_ms: float,
        cache_hit: bool = False
    ) -> None:
        """
        Record metrics for a request
        
        Args:
            request_id: Unique request identifier
            strategy: Execution strategy used
            models_count: Number of models executed
            success_count: Number of successful model executions
            failure_count: Number of failed model executions
            total_latency_ms: Total request latency in milliseconds
            cache_hit: Whether cache was hit
        """
        metrics = RequestMetrics(
            request_id=request_id,
            strategy=strategy,
            models_count=models_count,
            success_count=success_count,
            failure_count=failure_count,
            total_latency_ms=total_latency_ms,
            cache_hit=cache_hit
        )
        
        self.request_metrics.append(metrics)
        
        # Keep only last N requests
        if len(self.request_metrics) > self._max_requests:
            self.request_metrics = self.request_metrics[-self._max_requests:]
    
    def record_model_execution(
        self,
        model_type: str,
        success: bool,
        latency_ms: float
    ) -> None:
        """
        Record metrics for a model execution
        
        Args:
            model_type: Type of model
            success: Whether execution was successful
            latency_ms: Execution latency in milliseconds
        """
        metrics = self.model_metrics[model_type]
        metrics.model_type = model_type
        metrics.call_count += 1
        metrics.total_latency_ms += latency_ms
        metrics.last_called = datetime.now()
        
        if success:
            metrics.success_count += 1
        else:
            metrics.failure_count += 1
        
        # Update average latency
        if metrics.call_count > 0:
            metrics.avg_latency_ms = metrics.total_latency_ms / metrics.call_count
    
    def get_request_stats(self, last_n: int = 100) -> Dict[str, Any]:
        """
        Get statistics for recent requests
        
        Args:
            last_n: Number of recent requests to analyze
            
        Returns:
            Dictionary with request statistics
        """
        recent = self.request_metrics[-last_n:] if len(self.request_metrics) > last_n else self.request_metrics
        
        if not recent:
            return {
                "total_requests": 0,
                "avg_latency_ms": 0.0,
                "success_rate": 0.0,
                "cache_hit_rate": 0.0
            }
        
        total_requests = len(recent)
        total_latency = sum(m.total_latency_ms for m in recent)
        total_success = sum(m.success_count for m in recent)
        total_failures = sum(m.failure_count for m in recent)
        cache_hits = sum(1 for m in recent if m.cache_hit)
        
        # Calculate success rate (based on model executions, not requests)
        total_model_executions = total_success + total_failures
        success_rate = (total_success / total_model_executions * 100) if total_model_executions > 0 else 0.0
        
        return {
            "total_requests": total_requests,
            "avg_latency_ms": round(total_latency / total_requests, 2) if total_requests > 0 else 0.0,
            "success_rate": round(success_rate, 2),
            "cache_hit_rate": round((cache_hits / total_requests * 100), 2) if total_requests > 0 else 0.0,
            "total_model_executions": total_model_executions,
            "total_successful_executions": total_success,
            "total_failed_executions": total_failures
        }
    
    def get_model_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all models
        
        Returns:
            Dictionary mapping model_type to statistics
        """
        stats = {}
        for model_type, metrics in self.model_metrics.items():
            if metrics.call_count > 0:
                stats[model_type] = {
                    "call_count": metrics.call_count,
                    "success_count": metrics.success_count,
                    "failure_count": metrics.failure_count,
                    "success_rate": round((metrics.success_count / metrics.call_count * 100), 2),
                    "avg_latency_ms": round(metrics.avg_latency_ms, 2),
                    "total_latency_ms": round(metrics.total_latency_ms, 2),
                    "last_called": metrics.last_called.isoformat() if metrics.last_called else None
                }
        return stats
    
    def get_strategy_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics by strategy
        
        Returns:
            Dictionary mapping strategy to statistics
        """
        strategy_stats = defaultdict(lambda: {
            "count": 0,
            "total_latency_ms": 0.0,
            "total_success": 0,
            "total_failures": 0
        })
        
        for metrics in self.request_metrics:
            stats = strategy_stats[metrics.strategy]
            stats["count"] += 1
            stats["total_latency_ms"] += metrics.total_latency_ms
            stats["total_success"] += metrics.success_count
            stats["total_failures"] += metrics.failure_count
        
        # Calculate averages and rates
        result = {}
        for strategy, stats in strategy_stats.items():
            count = stats["count"]
            result[strategy] = {
                "request_count": count,
                "avg_latency_ms": round(stats["total_latency_ms"] / count, 2) if count > 0 else 0.0,
                "total_model_executions": stats["total_success"] + stats["total_failures"],
                "success_rate": round(
                    (stats["total_success"] / (stats["total_success"] + stats["total_failures"]) * 100)
                    if (stats["total_success"] + stats["total_failures"]) > 0 else 0.0,
                    2
                )
            }
        
        return result
    
    def reset(self) -> None:
        """Reset all metrics"""
        self.request_metrics.clear()
        self.model_metrics.clear()
        logger.info("Metrics reset")


# Global metrics service instance
_metrics_service: Optional[MetricsService] = None


def get_metrics_service() -> MetricsService:
    """Get or create global metrics service instance"""
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = MetricsService()
    return _metrics_service




