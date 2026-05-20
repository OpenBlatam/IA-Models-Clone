"""
Metrics and Monitoring Service
===============================

Service for tracking metrics, performance, and usage statistics.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class OperationMetric:
    """Metric for a single operation"""
    operation_type: str  # "clothing_change" or "face_swap"
    success: bool
    duration: float
    timestamp: datetime
    prompt_id: Optional[str] = None
    openrouter_used: bool = False
    truthgpt_used: bool = False
    error: Optional[str] = None


@dataclass
class ServiceMetrics:
    """Aggregated service metrics"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_duration: float = 0.0
    average_duration: float = 0.0
    openrouter_usage_count: int = 0
    truthgpt_usage_count: int = 0
    operations_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    recent_operations: deque = field(default_factory=lambda: deque(maxlen=100))


class MetricsService:
    """
    Service for tracking and aggregating metrics.
    
    Features:
    - Operation tracking
    - Performance metrics
    - Usage statistics
    - Error tracking
    - Time-based analytics
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics service.
        
        Args:
            max_history: Maximum number of operations to keep in history
        """
        self.max_history = max_history
        self.metrics = ServiceMetrics()
        self.operation_history: deque = deque(maxlen=max_history)
        self.hourly_metrics: Dict[str, Dict[str, Any]] = {}
        self.daily_metrics: Dict[str, Dict[str, Any]] = {}
    
    def record_operation(
        self,
        operation_type: str,
        success: bool,
        duration: float,
        prompt_id: Optional[str] = None,
        openrouter_used: bool = False,
        truthgpt_used: bool = False,
        error: Optional[str] = None
    ) -> None:
        """
        Record an operation metric.
        
        Args:
            operation_type: Type of operation ("clothing_change" or "face_swap")
            success: Whether operation succeeded
            duration: Operation duration in seconds
            prompt_id: ComfyUI prompt ID
            openrouter_used: Whether OpenRouter was used
            truthgpt_used: Whether TruthGPT was used
            error: Error message if failed
        """
        metric = OperationMetric(
            operation_type=operation_type,
            success=success,
            duration=duration,
            timestamp=datetime.now(),
            prompt_id=prompt_id,
            openrouter_used=openrouter_used,
            truthgpt_used=truthgpt_used,
            error=error
        )
        
        # Update aggregated metrics
        self.metrics.total_operations += 1
        self.metrics.operations_by_type[operation_type] += 1
        
        if success:
            self.metrics.successful_operations += 1
        else:
            self.metrics.failed_operations += 1
            if error:
                error_type = error.split(":")[0] if ":" in error else "unknown"
                self.metrics.errors_by_type[error_type] += 1
        
        self.metrics.total_duration += duration
        self.metrics.average_duration = (
            self.metrics.total_duration / self.metrics.total_operations
        )
        
        if openrouter_used:
            self.metrics.openrouter_usage_count += 1
        
        if truthgpt_used:
            self.metrics.truthgpt_usage_count += 1
        
        # Add to history
        self.operation_history.append(metric)
        self.metrics.recent_operations.append(metric)
        
        # Update time-based metrics
        self._update_time_based_metrics(metric)
    
    def _update_time_based_metrics(self, metric: OperationMetric) -> None:
        """Update hourly and daily metrics"""
        hour_key = metric.timestamp.strftime("%Y-%m-%d-%H")
        day_key = metric.timestamp.strftime("%Y-%m-%d")
        
        # Update hourly metrics
        if hour_key not in self.hourly_metrics:
            self.hourly_metrics[hour_key] = {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "total_duration": 0.0
            }
        
        self.hourly_metrics[hour_key]["total"] += 1
        if metric.success:
            self.hourly_metrics[hour_key]["successful"] += 1
        else:
            self.hourly_metrics[hour_key]["failed"] += 1
        self.hourly_metrics[hour_key]["total_duration"] += metric.duration
        
        # Update daily metrics
        if day_key not in self.daily_metrics:
            self.daily_metrics[day_key] = {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "total_duration": 0.0
            }
        
        self.daily_metrics[day_key]["total"] += 1
        if metric.success:
            self.daily_metrics[day_key]["successful"] += 1
        else:
            self.daily_metrics[day_key]["failed"] += 1
        self.daily_metrics[day_key]["total_duration"] += metric.duration
        
        # Cleanup old metrics (keep last 7 days)
        cutoff_date = datetime.now() - timedelta(days=7)
        self.hourly_metrics = {
            k: v for k, v in self.hourly_metrics.items()
            if datetime.strptime(k, "%Y-%m-%d-%H") > cutoff_date
        }
        self.daily_metrics = {
            k: v for k, v in self.daily_metrics.items()
            if datetime.strptime(k, "%Y-%m-%d") > cutoff_date
        }
    
    def get_metrics(self, time_range: Optional[str] = None) -> Dict[str, Any]:
        """
        Get aggregated metrics.
        
        Args:
            time_range: Optional time range ("hour", "day", "week", or None for all)
            
        Returns:
            Dictionary with metrics
        """
        base_metrics = {
            "total_operations": self.metrics.total_operations,
            "successful_operations": self.metrics.successful_operations,
            "failed_operations": self.metrics.failed_operations,
            "success_rate": (
                self.metrics.successful_operations / self.metrics.total_operations * 100
                if self.metrics.total_operations > 0 else 0
            ),
            "average_duration": self.metrics.average_duration,
            "total_duration": self.metrics.total_duration,
            "operations_by_type": dict(self.metrics.operations_by_type),
            "errors_by_type": dict(self.metrics.errors_by_type),
            "openrouter_usage": {
                "count": self.metrics.openrouter_usage_count,
                "percentage": (
                    self.metrics.openrouter_usage_count / self.metrics.total_operations * 100
                    if self.metrics.total_operations > 0 else 0
                )
            },
            "truthgpt_usage": {
                "count": self.metrics.truthgpt_usage_count,
                "percentage": (
                    self.metrics.truthgpt_usage_count / self.metrics.total_operations * 100
                    if self.metrics.total_operations > 0 else 0
                )
            }
        }
        
        if time_range:
            base_metrics["time_range"] = time_range
            base_metrics["time_based_metrics"] = self._get_time_range_metrics(time_range)
        
        return base_metrics
    
    def _get_time_range_metrics(self, time_range: str) -> Dict[str, Any]:
        """Get metrics for a specific time range"""
        now = datetime.now()
        
        if time_range == "hour":
            cutoff = now - timedelta(hours=1)
            relevant_metrics = {
                k: v for k, v in self.hourly_metrics.items()
                if datetime.strptime(k, "%Y-%m-%d-%H") >= cutoff
            }
        elif time_range == "day":
            cutoff = now - timedelta(days=1)
            relevant_metrics = {
                k: v for k, v in self.daily_metrics.items()
                if datetime.strptime(k, "%Y-%m-%d") >= cutoff
            }
        elif time_range == "week":
            cutoff = now - timedelta(days=7)
            relevant_metrics = {
                k: v for k, v in self.daily_metrics.items()
                if datetime.strptime(k, "%Y-%m-%d") >= cutoff
            }
        else:
            relevant_metrics = {}
        
        return relevant_metrics
    
    def get_recent_operations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent operations.
        
        Args:
            limit: Maximum number of operations to return
            
        Returns:
            List of recent operation metrics
        """
        recent = list(self.metrics.recent_operations)[-limit:]
        return [
            {
                "operation_type": m.operation_type,
                "success": m.success,
                "duration": m.duration,
                "timestamp": m.timestamp.isoformat(),
                "prompt_id": m.prompt_id,
                "openrouter_used": m.openrouter_used,
                "truthgpt_used": m.truthgpt_used,
                "error": m.error
            }
            for m in recent
        ]
    
    def reset_metrics(self) -> None:
        """Reset all metrics (use with caution)"""
        self.metrics = ServiceMetrics()
        self.operation_history.clear()
        self.hourly_metrics.clear()
        self.daily_metrics.clear()
        logger.warning("Metrics reset")


# Global metrics service instance
_metrics_service: Optional[MetricsService] = None


def get_metrics_service() -> MetricsService:
    """Get or create metrics service instance"""
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = MetricsService()
    return _metrics_service

