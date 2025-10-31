"""
Webhook Analytics - Advanced analytics and reporting
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsEvent:
    """Analytics event record"""
    timestamp: float
    event_type: str
    endpoint_id: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyticsReport:
    """Analytics report data"""
    period_start: float
    period_end: float
    total_events: int
    success_rate: float
    average_response_time: float
    top_endpoints: List[Tuple[str, int]]
    error_breakdown: Dict[str, int]
    hourly_distribution: Dict[int, int]
    performance_metrics: Dict[str, float]


class WebhookAnalytics:
    """
    Advanced analytics system for webhook operations
    
    Features:
    - Event tracking and analysis
    - Performance metrics
    - Success/failure rates
    - Endpoint analytics
    - Time-based analysis
    - Custom reporting
    """
    
    def __init__(self, retention_days: int = 30):
        """
        Initialize analytics system
        
        Args:
            retention_days: How long to keep analytics data
        """
        self.retention_days = retention_days
        self.events: List[AnalyticsEvent] = []
        self.metrics_cache: Dict[str, Any] = {}
        self.last_cleanup = time.time()
    
    def track_event(
        self,
        event_type: str,
        endpoint_id: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track analytics event
        
        Args:
            event_type: Type of event (delivery, error, etc.)
            endpoint_id: Endpoint identifier
            data: Event data
            metadata: Additional metadata
        """
        event = AnalyticsEvent(
            timestamp=time.time(),
            event_type=event_type,
            endpoint_id=endpoint_id,
            data=data,
            metadata=metadata or {}
        )
        
        self.events.append(event)
        
        # Cleanup old data periodically
        if time.time() - self.last_cleanup > 3600:  # Every hour
            self.cleanup_old_data()
            self.last_cleanup = time.time()
    
    def track_delivery(
        self,
        endpoint_id: str,
        status: str,
        duration: float,
        event_type: str,
        retry_count: int = 0
    ) -> None:
        """Track webhook delivery event"""
        self.track_event(
            event_type="delivery",
            endpoint_id=endpoint_id,
            data={
                "status": status,
                "duration": duration,
                "webhook_event": event_type,
                "retry_count": retry_count
            }
        )
    
    def track_error(
        self,
        endpoint_id: str,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track error event"""
        self.track_event(
            event_type="error",
            endpoint_id=endpoint_id,
            data={
                "error_type": error_type,
                "error_message": error_message,
                "context": context or {}
            }
        )
    
    def track_rate_limit(
        self,
        endpoint_id: str,
        limit_type: str,
        current_rate: float
    ) -> None:
        """Track rate limit event"""
        self.track_event(
            event_type="rate_limit",
            endpoint_id=endpoint_id,
            data={
                "limit_type": limit_type,
                "current_rate": current_rate
            }
        )
    
    def track_circuit_breaker(
        self,
        endpoint_id: str,
        state: str,
        failure_rate: float
    ) -> None:
        """Track circuit breaker event"""
        self.track_event(
            event_type="circuit_breaker",
            endpoint_id=endpoint_id,
            data={
                "state": state,
                "failure_rate": failure_rate
            }
        )
    
    def get_performance_report(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        endpoint_id: Optional[str] = None
    ) -> AnalyticsReport:
        """
        Generate comprehensive performance report
        
        Args:
            start_time: Report start time (default: 24 hours ago)
            end_time: Report end time (default: now)
            endpoint_id: Filter by endpoint (optional)
            
        Returns:
            AnalyticsReport with performance data
        """
        now = time.time()
        start_time = start_time or (now - 86400)  # 24 hours ago
        end_time = end_time or now
        
        # Filter events by time and endpoint
        filtered_events = [
            event for event in self.events
            if start_time <= event.timestamp <= end_time
            and (endpoint_id is None or event.endpoint_id == endpoint_id)
        ]
        
        if not filtered_events:
            return self._empty_report(start_time, end_time)
        
        # Calculate metrics
        delivery_events = [e for e in filtered_events if e.event_type == "delivery"]
        error_events = [e for e in filtered_events if e.event_type == "error"]
        
        # Success rate
        successful_deliveries = sum(
            1 for e in delivery_events
            if e.data.get("status") == "delivered"
        )
        total_deliveries = len(delivery_events)
        success_rate = successful_deliveries / total_deliveries if total_deliveries > 0 else 0
        
        # Average response time
        response_times = [
            e.data.get("duration", 0) for e in delivery_events
            if "duration" in e.data
        ]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Top endpoints
        endpoint_counts = Counter(e.endpoint_id for e in delivery_events)
        top_endpoints = endpoint_counts.most_common(10)
        
        # Error breakdown
        error_types = Counter(e.data.get("error_type", "unknown") for e in error_events)
        error_breakdown = dict(error_types)
        
        # Hourly distribution
        hourly_dist = defaultdict(int)
        for event in delivery_events:
            hour = datetime.fromtimestamp(event.timestamp).hour
            hourly_dist[hour] += 1
        
        # Performance metrics
        performance_metrics = self._calculate_performance_metrics(delivery_events)
        
        return AnalyticsReport(
            period_start=start_time,
            period_end=end_time,
            total_events=len(filtered_events),
            success_rate=success_rate,
            average_response_time=avg_response_time,
            top_endpoints=top_endpoints,
            error_breakdown=error_breakdown,
            hourly_distribution=dict(hourly_dist),
            performance_metrics=performance_metrics
        )
    
    def _calculate_performance_metrics(self, delivery_events: List[AnalyticsEvent]) -> Dict[str, float]:
        """Calculate detailed performance metrics"""
        if not delivery_events:
            return {}
        
        durations = [e.data.get("duration", 0) for e in delivery_events if "duration" in e.data]
        retry_counts = [e.data.get("retry_count", 0) for e in delivery_events]
        
        metrics = {}
        
        if durations:
            metrics.update({
                "min_response_time": min(durations),
                "max_response_time": max(durations),
                "p50_response_time": self._percentile(durations, 50),
                "p95_response_time": self._percentile(durations, 95),
                "p99_response_time": self._percentile(durations, 99)
            })
        
        if retry_counts:
            metrics.update({
                "average_retry_count": sum(retry_counts) / len(retry_counts),
                "max_retry_count": max(retry_counts),
                "retry_rate": sum(1 for r in retry_counts if r > 0) / len(retry_counts)
            })
        
        return metrics
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _empty_report(self, start_time: float, end_time: float) -> AnalyticsReport:
        """Create empty report for no data"""
        return AnalyticsReport(
            period_start=start_time,
            period_end=end_time,
            total_events=0,
            success_rate=0.0,
            average_response_time=0.0,
            top_endpoints=[],
            error_breakdown={},
            hourly_distribution={},
            performance_metrics={}
        )
    
    def get_endpoint_analytics(
        self,
        endpoint_id: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get detailed analytics for specific endpoint"""
        cutoff = time.time() - (days * 86400)
        endpoint_events = [
            event for event in self.events
            if event.endpoint_id == endpoint_id and event.timestamp >= cutoff
        ]
        
        if not endpoint_events:
            return {"error": "No data found for endpoint"}
        
        delivery_events = [e for e in endpoint_events if e.event_type == "delivery"]
        error_events = [e for e in endpoint_events if e.event_type == "error"]
        
        # Calculate metrics
        total_deliveries = len(delivery_events)
        successful_deliveries = sum(
            1 for e in delivery_events
            if e.data.get("status") == "delivered"
        )
        
        success_rate = successful_deliveries / total_deliveries if total_deliveries > 0 else 0
        
        # Response time analysis
        response_times = [
            e.data.get("duration", 0) for e in delivery_events
            if "duration" in e.data
        ]
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Error analysis
        error_types = Counter(e.data.get("error_type", "unknown") for e in error_events)
        
        # Daily breakdown
        daily_stats = defaultdict(lambda: {"deliveries": 0, "errors": 0, "success_rate": 0})
        
        for event in endpoint_events:
            date = datetime.fromtimestamp(event.timestamp).date()
            if event.event_type == "delivery":
                daily_stats[date]["deliveries"] += 1
                if event.data.get("status") == "delivered":
                    daily_stats[date]["success_rate"] += 1
            elif event.event_type == "error":
                daily_stats[date]["errors"] += 1
        
        # Calculate daily success rates
        for date, stats in daily_stats.items():
            if stats["deliveries"] > 0:
                stats["success_rate"] = stats["success_rate"] / stats["deliveries"]
        
        return {
            "endpoint_id": endpoint_id,
            "period_days": days,
            "total_deliveries": total_deliveries,
            "successful_deliveries": successful_deliveries,
            "success_rate": success_rate,
            "average_response_time": avg_response_time,
            "error_count": len(error_events),
            "error_types": dict(error_types),
            "daily_stats": {
                str(date): stats for date, stats in daily_stats.items()
            },
            "performance_trend": self._calculate_trend(delivery_events)
        }
    
    def _calculate_trend(self, events: List[AnalyticsEvent]) -> str:
        """Calculate performance trend (improving/declining/stable)"""
        if len(events) < 10:
            return "insufficient_data"
        
        # Split into two halves
        mid_point = len(events) // 2
        first_half = events[:mid_point]
        second_half = events[mid_point:]
        
        # Calculate success rates
        first_success_rate = sum(
            1 for e in first_half if e.data.get("status") == "delivered"
        ) / len(first_half)
        
        second_success_rate = sum(
            1 for e in second_half if e.data.get("status") == "delivered"
        ) / len(second_half)
        
        # Determine trend
        diff = second_success_rate - first_success_rate
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        else:
            return "stable"
    
    def cleanup_old_data(self) -> int:
        """Remove old analytics data"""
        cutoff = time.time() - (self.retention_days * 86400)
        old_count = len(self.events)
        
        self.events = [event for event in self.events if event.timestamp >= cutoff]
        
        removed_count = old_count - len(self.events)
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old analytics events")
        
        return removed_count
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        now = time.time()
        last_hour = now - 3600
        last_24h = now - 86400
        
        # Recent events
        recent_events = [e for e in self.events if e.timestamp >= last_hour]
        recent_deliveries = [e for e in recent_events if e.event_type == "delivery"]
        recent_errors = [e for e in recent_events if e.event_type == "error"]
        
        # Calculate health score
        health_score = 100.0
        
        if recent_deliveries:
            success_rate = sum(
                1 for e in recent_deliveries
                if e.data.get("status") == "delivered"
            ) / len(recent_deliveries)
            
            # Deduct points for low success rate
            health_score -= (1 - success_rate) * 50
            
            # Deduct points for high error rate
            error_rate = len(recent_errors) / len(recent_deliveries)
            health_score -= error_rate * 30
        
        # Deduct points for high response times
        if recent_deliveries:
            avg_response_time = sum(
                e.data.get("duration", 0) for e in recent_deliveries
                if "duration" in e.data
            ) / len([e for e in recent_deliveries if "duration" in e.data])
            
            if avg_response_time > 5.0:  # 5 seconds
                health_score -= min(20, (avg_response_time - 5) * 4)
        
        health_score = max(0, min(100, health_score))
        
        return {
            "health_score": health_score,
            "status": self._get_health_status(health_score),
            "recent_deliveries": len(recent_deliveries),
            "recent_errors": len(recent_errors),
            "success_rate": success_rate if recent_deliveries else 0,
            "average_response_time": avg_response_time if recent_deliveries else 0,
            "last_updated": now
        }
    
    def _get_health_status(self, score: float) -> str:
        """Get health status from score"""
        if score >= 90:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 50:
            return "fair"
        elif score >= 25:
            return "poor"
        else:
            return "critical"


# Global analytics instance
webhook_analytics = WebhookAnalytics()





