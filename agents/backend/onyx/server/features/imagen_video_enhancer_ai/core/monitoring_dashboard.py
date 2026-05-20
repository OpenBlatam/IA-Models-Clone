"""
Monitoring Dashboard for Imagen Video Enhancer AI
=================================================

Real-time monitoring dashboard data provider.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class DashboardMetrics:
    """Dashboard metrics."""
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    pending_tasks: int
    processing_tasks: int
    success_rate: float
    average_processing_time: float
    cache_hit_rate: float
    webhook_success_rate: float
    retry_success_rate: float
    active_workers: int
    queue_size: int
    uptime_seconds: float
    timestamp: str


class MonitoringDashboard:
    """
    Provides real-time monitoring data for dashboard.
    
    Features:
    - Real-time metrics
    - Historical data
    - Performance trends
    - System health
    """
    
    def __init__(self, agent):
        """
        Initialize monitoring dashboard.
        
        Args:
            agent: EnhancerAgent instance
        """
        self.agent = agent
        self.start_time = datetime.now()
        self._historical_metrics: List[Dict[str, Any]] = []
    
    def get_current_metrics(self) -> DashboardMetrics:
        """
        Get current system metrics.
        
        Returns:
            DashboardMetrics with current state
        """
        stats = self.agent.get_stats()
        executor_stats = stats.get("executor_stats", {})
        cache_stats = stats.get("cache_stats", {})
        webhook_stats = stats.get("webhook_stats", {})
        retry_stats = stats.get("retry_stats", {})
        performance_stats = stats.get("performance_stats", {})
        
        # Calculate totals
        total_tasks = executor_stats.get("total_tasks", 0)
        completed_tasks = executor_stats.get("completed_tasks", 0)
        failed_tasks = executor_stats.get("failed_tasks", 0)
        pending_tasks = executor_stats.get("pending_tasks", 0)
        processing_tasks = executor_stats.get("processing_tasks", 0)
        
        # Calculate rates
        success_rate = (
            completed_tasks / total_tasks
            if total_tasks > 0
            else 0.0
        )
        
        cache_hit_rate = cache_stats.get("hit_rate", 0.0)
        
        webhook_total = webhook_stats.get("total_sent", 0)
        webhook_successful = webhook_stats.get("successful", 0)
        webhook_success_rate = (
            webhook_successful / webhook_total
            if webhook_total > 0
            else 0.0
        )
        
        retry_total = retry_stats.get("total_retries", 0)
        retry_successful = retry_stats.get("successful_retries", 0)
        retry_success_rate = (
            retry_successful / retry_total
            if retry_total > 0
            else 0.0
        )
        
        # Calculate average processing time
        avg_time = 0.0
        if performance_stats:
            all_times = []
            for service_stats in performance_stats.values():
                if "average" in service_stats:
                    all_times.append(service_stats["average"])
            if all_times:
                avg_time = sum(all_times) / len(all_times)
        
        # Calculate uptime
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return DashboardMetrics(
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            pending_tasks=pending_tasks,
            processing_tasks=processing_tasks,
            success_rate=success_rate,
            average_processing_time=avg_time,
            cache_hit_rate=cache_hit_rate,
            webhook_success_rate=webhook_success_rate,
            retry_success_rate=retry_success_rate,
            active_workers=executor_stats.get("active_workers", 0),
            queue_size=pending_tasks,
            uptime_seconds=uptime,
            timestamp=datetime.now().isoformat()
        )
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get current metrics as dictionary."""
        metrics = self.get_current_metrics()
        return asdict(metrics)
    
    def get_historical_metrics(
        self,
        hours: int = 24,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get historical metrics.
        
        Args:
            hours: Number of hours to look back
            limit: Maximum number of records
            
        Returns:
            List of historical metric dictionaries
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Filter and limit
        filtered = [
            m for m in self._historical_metrics
            if datetime.fromisoformat(m["timestamp"]) >= cutoff
        ]
        
        return filtered[-limit:]
    
    def record_metrics(self):
        """Record current metrics to history."""
        metrics = self.get_metrics_dict()
        self._historical_metrics.append(metrics)
        
        # Keep only last 1000 records
        if len(self._historical_metrics) > 1000:
            self._historical_metrics = self._historical_metrics[-1000:]
    
    def get_performance_trends(self) -> Dict[str, Any]:
        """
        Get performance trends.
        
        Returns:
            Dictionary with trend data
        """
        if len(self._historical_metrics) < 2:
            return {
                "success_rate_trend": "stable",
                "processing_time_trend": "stable",
                "cache_hit_rate_trend": "stable"
            }
        
        recent = self._historical_metrics[-10:]
        
        # Calculate trends
        success_rates = [m["success_rate"] for m in recent]
        processing_times = [m["average_processing_time"] for m in recent]
        cache_rates = [m["cache_hit_rate"] for m in recent]
        
        def calculate_trend(values):
            if len(values) < 2:
                return "stable"
            first_half = sum(values[:len(values)//2]) / (len(values)//2)
            second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
            diff = second_half - first_half
            if abs(diff) < 0.01:
                return "stable"
            return "improving" if diff > 0 else "degrading"
        
        return {
            "success_rate_trend": calculate_trend(success_rates),
            "processing_time_trend": calculate_trend([-t for t in processing_times]),  # Negative because lower is better
            "cache_hit_rate_trend": calculate_trend(cache_rates)
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get system health status.
        
        Returns:
            Dictionary with health information
        """
        metrics = self.get_current_metrics()
        
        health_score = 100.0
        
        # Deduct points for issues
        if metrics.success_rate < 0.9:
            health_score -= 20
        if metrics.cache_hit_rate < 0.5:
            health_score -= 10
        if metrics.average_processing_time > 10.0:
            health_score -= 15
        if metrics.failed_tasks > metrics.completed_tasks * 0.1:
            health_score -= 25
        
        # Determine status
        if health_score >= 90:
            status = "excellent"
        elif health_score >= 75:
            status = "good"
        elif health_score >= 60:
            status = "fair"
        else:
            status = "poor"
        
        return {
            "status": status,
            "score": max(0, health_score),
            "metrics": asdict(metrics),
            "trends": self.get_performance_trends(),
            "timestamp": datetime.now().isoformat()
        }




