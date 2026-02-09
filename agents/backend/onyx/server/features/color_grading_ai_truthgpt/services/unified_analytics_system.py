"""
Unified Analytics System for Color Grading AI
==============================================

Consolidates all analytics services into a single unified system:
- AnalyticsService (usage analytics)
- TelemetryService (telemetry tracking)
- MetricsCollector (metrics collection)
- MetricsAggregator (metrics aggregation)
- AnalyticsDashboard (dashboard metrics)

Features:
- Unified analytics interface
- Multi-source data collection
- Real-time aggregation
- Custom reports
- Dashboard integration
- Export capabilities
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

from .analytics_service import AnalyticsService
from .telemetry_service import TelemetryService, TelemetryEvent
from .metrics_collector import MetricsCollector
from .metrics_aggregator import MetricsAggregator, AggregatedMetric
from .analytics_dashboard import AnalyticsDashboard, DashboardMetric, DashboardWidget

logger = logging.getLogger(__name__)


@dataclass
class UnifiedAnalyticsReport:
    """Unified analytics report."""
    timestamp: datetime
    usage_report: Dict[str, Any]
    performance_report: Dict[str, Any]
    telemetry_summary: Dict[str, Any]
    aggregated_metrics: Dict[str, AggregatedMetric]
    dashboard_metrics: List[DashboardMetric]
    recommendations: List[str] = field(default_factory=list)


class UnifiedAnalyticsSystem:
    """
    Unified analytics system.
    
    Consolidates:
    - AnalyticsService: Usage analytics
    - TelemetryService: Telemetry tracking
    - MetricsCollector: Metrics collection
    - MetricsAggregator: Metrics aggregation
    - AnalyticsDashboard: Dashboard metrics
    
    Features:
    - Unified interface for all analytics
    - Multi-source data collection
    - Real-time aggregation
    - Custom reports
    - Dashboard integration
    """
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        history_manager,
        telemetry_storage_dir: str = "telemetry",
        metrics_dir: str = "metrics"
    ):
        """
        Initialize unified analytics system.
        
        Args:
            metrics_collector: Metrics collector instance
            history_manager: History manager instance
            telemetry_storage_dir: Telemetry storage directory
            metrics_dir: Metrics storage directory
        """
        self.metrics_collector = metrics_collector
        self.history_manager = history_manager
        
        # Initialize components
        self.analytics_service = AnalyticsService(metrics_collector, history_manager)
        self.telemetry_service = TelemetryService(storage_dir=telemetry_storage_dir)
        self.metrics_aggregator = MetricsAggregator(window_size=100)
        self.analytics_dashboard = AnalyticsDashboard()
        
        logger.info("Initialized UnifiedAnalyticsSystem")
    
    def track_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Track analytics event.
        
        Args:
            event_type: Event type
            data: Event data
            user_id: Optional user ID
            session_id: Optional session ID
        """
        # Track in telemetry
        self.telemetry_service.track_event(
            event_type,
            data,
            user_id=user_id,
            session_id=session_id
        )
        
        # Track in metrics collector
        self.metrics_collector.record_metric(
            f"event.{event_type}",
            value=1.0,
            metadata=data
        )
        
        logger.debug(f"Tracked event: {event_type}")
    
    def track_metric(
        self,
        metric_name: str,
        value: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Track metric.
        
        Args:
            metric_name: Metric name
            value: Metric value
            metadata: Optional metadata
        """
        # Track in metrics collector
        self.metrics_collector.record_metric(
            metric_name,
            value=value,
            metadata=metadata or {}
        )
        
        # Aggregate in aggregator
        import asyncio
        asyncio.create_task(
            self.metrics_aggregator.add_metric(metric_name, value, metadata)
        )
        
        # Track in telemetry
        self.telemetry_service.track_metric(metric_name, value)
    
    async def get_usage_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get usage report.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Usage report
        """
        return self.analytics_service.get_usage_report(start_date, end_date)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report."""
        return self.analytics_service.get_performance_report()
    
    async def get_aggregated_metrics(
        self,
        metric_name: Optional[str] = None
    ) -> Dict[str, AggregatedMetric]:
        """
        Get aggregated metrics.
        
        Args:
            metric_name: Optional metric name
            
        Returns:
            Aggregated metrics
        """
        return await self.metrics_aggregator.aggregate(metric_name)
    
    def get_dashboard_metrics(self) -> List[DashboardMetric]:
        """Get dashboard metrics."""
        return self.analytics_dashboard.get_metrics()
    
    def generate_unified_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> UnifiedAnalyticsReport:
        """
        Generate unified analytics report.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Unified analytics report
        """
        # Get usage report
        usage_report = self.get_usage_report(start_date, end_date)
        
        # Get performance report
        performance_report = self.get_performance_report()
        
        # Get telemetry summary
        telemetry_summary = self.telemetry_service.get_summary()
        
        # Get aggregated metrics
        import asyncio
        aggregated_metrics = asyncio.run(self.get_aggregated_metrics())
        
        # Get dashboard metrics
        dashboard_metrics = self.get_dashboard_metrics()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            usage_report,
            performance_report
        )
        
        return UnifiedAnalyticsReport(
            timestamp=datetime.now(),
            usage_report=usage_report,
            performance_report=performance_report,
            telemetry_summary=telemetry_summary,
            aggregated_metrics=aggregated_metrics,
            dashboard_metrics=dashboard_metrics,
            recommendations=recommendations
        )
    
    def _generate_recommendations(
        self,
        usage_report: Dict[str, Any],
        performance_report: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations from analytics."""
        recommendations = []
        
        # Check success rate
        summary = usage_report.get("summary", {})
        success_rate = summary.get("success_rate", 1.0)
        if success_rate < 0.95:
            recommendations.append(
                f"Success rate is {success_rate:.1%}. Review error patterns."
            )
        
        # Check performance
        if performance_report:
            avg_duration = performance_report.get("avg_duration", 0)
            if avg_duration > 10.0:
                recommendations.append(
                    f"Average operation duration is {avg_duration:.2f}s. Consider optimization."
                )
        
        return recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analytics statistics."""
        return {
            "metrics_collector": self.metrics_collector.get_statistics(),
            "metrics_aggregator": self.metrics_aggregator.get_statistics(),
            "telemetry": self.telemetry_service.get_statistics(),
        }


