"""
Analytics Service
=================

Advanced analytics service for data analysis and reporting.
"""

from __future__ import annotations
import asyncio
import logging
import json
import statistics
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import numpy as np
from scipy import stats
import pandas as pd

from ..utils.helpers import DateTimeHelpers
from ..utils.decorators import log_execution


logger = logging.getLogger(__name__)


class AnalyticsEventType(str, Enum):
    """Analytics event type enumeration"""
    WORKFLOW_CREATED = "workflow_created"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    NODE_ADDED = "node_added"
    NODE_UPDATED = "node_updated"
    NODE_DELETED = "node_deleted"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    API_REQUEST = "api_request"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_METRIC = "performance_metric"
    CUSTOM_EVENT = "custom_event"


class AnalyticsMetricType(str, Enum):
    """Analytics metric type enumeration"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"
    PERCENTILE = "percentile"


class AnalyticsTimeRange(str, Enum):
    """Analytics time range enumeration"""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


@dataclass
class AnalyticsEvent:
    """Analytics event representation"""
    id: str
    event_type: AnalyticsEventType
    user_id: Optional[str]
    session_id: Optional[str]
    timestamp: datetime
    properties: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    dimensions: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyticsMetric:
    """Analytics metric representation"""
    name: str
    type: AnalyticsMetricType
    value: float
    timestamp: datetime
    dimensions: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyticsReport:
    """Analytics report representation"""
    id: str
    name: str
    description: str
    time_range: AnalyticsTimeRange
    start_date: datetime
    end_date: datetime
    metrics: List[str]
    dimensions: List[str]
    filters: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=DateTimeHelpers.now_utc)
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyticsInsight:
    """Analytics insight representation"""
    id: str
    type: str
    title: str
    description: str
    confidence: float
    impact: str
    recommendation: str
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=DateTimeHelpers.now_utc)


class AnalyticsService:
    """Advanced analytics service"""
    
    def __init__(self, max_events: int = 1000000, retention_days: int = 365):
        self.max_events = max_events
        self.retention_days = retention_days
        self.events: List[AnalyticsEvent] = []
        self.metrics: List[AnalyticsMetric] = []
        self.reports: List[AnalyticsReport] = []
        self.insights: List[AnalyticsInsight] = []
        self.is_running = False
        self.processor_task: Optional[asyncio.Task] = None
        self._initialize_analytics()
    
    def _initialize_analytics(self):
        """Initialize analytics service"""
        # Pre-defined metrics
        self.metric_definitions = {
            "workflow_creation_rate": AnalyticsMetricType.RATE,
            "workflow_completion_rate": AnalyticsMetricType.RATE,
            "workflow_failure_rate": AnalyticsMetricType.RATE,
            "average_workflow_duration": AnalyticsMetricType.TIMER,
            "user_activity": AnalyticsMetricType.COUNTER,
            "api_response_time": AnalyticsMetricType.HISTOGRAM,
            "error_rate": AnalyticsMetricType.RATE,
            "system_performance": AnalyticsMetricType.GAUGE
        }
        
        logger.info("Analytics service initialized")
    
    async def start(self):
        """Start the analytics service"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processor_task = asyncio.create_task(self._processor_worker())
        logger.info("Analytics service started")
    
    async def stop(self):
        """Stop the analytics service"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Analytics service stopped")
    
    def track_event(
        self,
        event_type: AnalyticsEventType,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        dimensions: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Track analytics event"""
        event_id = f"event_{int(DateTimeHelpers.now_utc().timestamp())}_{len(self.events)}"
        
        event = AnalyticsEvent(
            id=event_id,
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            timestamp=DateTimeHelpers.now_utc(),
            properties=properties or {},
            metrics=metrics or {},
            dimensions=dimensions or {},
            metadata=metadata or {}
        )
        
        self.events.append(event)
        
        # Keep only max_events
        if len(self.events) > self.max_events:
            self.events.pop(0)
        
        logger.debug(f"Tracked analytics event: {event_type.value}")
        
        return event_id
    
    def record_metric(
        self,
        name: str,
        value: float,
        dimensions: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record analytics metric"""
        metric = AnalyticsMetric(
            name=name,
            type=self.metric_definitions.get(name, AnalyticsMetricType.GAUGE),
            value=value,
            timestamp=DateTimeHelpers.now_utc(),
            dimensions=dimensions or {},
            metadata=metadata or {}
        )
        
        self.metrics.append(metric)
        
        # Keep only recent metrics
        if len(self.metrics) > 100000:
            self.metrics = self.metrics[-50000:]
        
        logger.debug(f"Recorded analytics metric: {name} = {value}")
    
    async def _processor_worker(self):
        """Analytics processor worker"""
        while self.is_running:
            try:
                await self._process_events()
                await self._generate_insights()
                await asyncio.sleep(60)  # Process every minute
            except Exception as e:
                logger.error(f"Analytics processor error: {e}")
                await asyncio.sleep(60)
    
    async def _process_events(self):
        """Process analytics events"""
        # Calculate derived metrics from events
        now = DateTimeHelpers.now_utc()
        
        # Workflow metrics
        workflow_events = [e for e in self.events if e.event_type in [
            AnalyticsEventType.WORKFLOW_CREATED,
            AnalyticsEventType.WORKFLOW_COMPLETED,
            AnalyticsEventType.WORKFLOW_FAILED
        ]]
        
        if workflow_events:
            # Calculate workflow creation rate
            created_events = [e for e in workflow_events if e.event_type == AnalyticsEventType.WORKFLOW_CREATED]
            if created_events:
                creation_rate = len(created_events) / 24  # per hour
                self.record_metric("workflow_creation_rate", creation_rate)
            
            # Calculate workflow completion rate
            completed_events = [e for e in workflow_events if e.event_type == AnalyticsEventType.WORKFLOW_COMPLETED]
            if completed_events:
                completion_rate = len(completed_events) / 24  # per hour
                self.record_metric("workflow_completion_rate", completion_rate)
            
            # Calculate workflow failure rate
            failed_events = [e for e in workflow_events if e.event_type == AnalyticsEventType.WORKFLOW_FAILED]
            if failed_events:
                failure_rate = len(failed_events) / 24  # per hour
                self.record_metric("workflow_failure_rate", failure_rate)
        
        # User activity metrics
        user_events = [e for e in self.events if e.user_id is not None]
        if user_events:
            unique_users = len(set(e.user_id for e in user_events))
            self.record_metric("user_activity", unique_users)
        
        # API request metrics
        api_events = [e for e in self.events if e.event_type == AnalyticsEventType.API_REQUEST]
        if api_events:
            response_times = [e.metrics.get("response_time", 0) for e in api_events if "response_time" in e.metrics]
            if response_times:
                avg_response_time = statistics.mean(response_times)
                self.record_metric("api_response_time", avg_response_time)
        
        # Error metrics
        error_events = [e for e in self.events if e.event_type == AnalyticsEventType.ERROR_OCCURRED]
        if error_events:
            error_rate = len(error_events) / 24  # per hour
            self.record_metric("error_rate", error_rate)
    
    async def _generate_insights(self):
        """Generate analytics insights"""
        # Generate insights based on recent data
        now = DateTimeHelpers.now_utc()
        recent_cutoff = now - timedelta(hours=24)
        
        recent_events = [e for e in self.events if e.timestamp > recent_cutoff]
        
        if not recent_events:
            return
        
        # Insight 1: Workflow performance
        workflow_events = [e for e in recent_events if e.event_type in [
            AnalyticsEventType.WORKFLOW_CREATED,
            AnalyticsEventType.WORKFLOW_COMPLETED,
            AnalyticsEventType.WORKFLOW_FAILED
        ]]
        
        if workflow_events:
            created_count = len([e for e in workflow_events if e.event_type == AnalyticsEventType.WORKFLOW_CREATED])
            completed_count = len([e for e in workflow_events if e.event_type == AnalyticsEventType.WORKFLOW_COMPLETED])
            failed_count = len([e for e in workflow_events if e.event_type == AnalyticsEventType.WORKFLOW_FAILED])
            
            if created_count > 0:
                completion_rate = completed_count / created_count
                failure_rate = failed_count / created_count
                
                if completion_rate < 0.8:
                    insight = AnalyticsInsight(
                        id=f"insight_{int(now.timestamp())}_1",
                        type="performance",
                        title="Low Workflow Completion Rate",
                        description=f"Workflow completion rate is {completion_rate:.1%}, below the 80% threshold",
                        confidence=0.8,
                        impact="high",
                        recommendation="Investigate workflow failures and optimize workflow logic",
                        data={
                            "completion_rate": completion_rate,
                            "failure_rate": failure_rate,
                            "total_workflows": created_count
                        }
                    )
                    self.insights.append(insight)
                
                if failure_rate > 0.1:
                    insight = AnalyticsInsight(
                        id=f"insight_{int(now.timestamp())}_2",
                        type="reliability",
                        title="High Workflow Failure Rate",
                        description=f"Workflow failure rate is {failure_rate:.1%}, above the 10% threshold",
                        confidence=0.9,
                        impact="critical",
                        recommendation="Review error logs and implement better error handling",
                        data={
                            "failure_rate": failure_rate,
                            "failed_workflows": failed_count
                        }
                    )
                    self.insights.append(insight)
        
        # Insight 2: User activity patterns
        user_events = [e for e in recent_events if e.user_id is not None]
        if user_events:
            user_activity = Counter(e.user_id for e in user_events)
            most_active_user = user_activity.most_common(1)[0]
            
            if most_active_user[1] > 100:  # More than 100 events in 24 hours
                insight = AnalyticsInsight(
                    id=f"insight_{int(now.timestamp())}_3",
                    type="usage",
                    title="High User Activity",
                    description=f"User {most_active_user[0]} has {most_active_user[1]} events in 24 hours",
                    confidence=0.7,
                    impact="medium",
                    recommendation="Consider if this is expected behavior or investigate for potential issues",
                    data={
                        "user_id": most_active_user[0],
                        "event_count": most_active_user[1]
                    }
                )
                self.insights.append(insight)
        
        # Keep only recent insights
        if len(self.insights) > 100:
            self.insights = self.insights[-50:]
    
    def get_analytics_summary(self, time_range: AnalyticsTimeRange = AnalyticsTimeRange.DAY) -> Dict[str, Any]:
        """Get analytics summary"""
        now = DateTimeHelpers.now_utc()
        
        # Calculate time range
        if time_range == AnalyticsTimeRange.HOUR:
            start_time = now - timedelta(hours=1)
        elif time_range == AnalyticsTimeRange.DAY:
            start_time = now - timedelta(days=1)
        elif time_range == AnalyticsTimeRange.WEEK:
            start_time = now - timedelta(weeks=1)
        elif time_range == AnalyticsTimeRange.MONTH:
            start_time = now - timedelta(days=30)
        elif time_range == AnalyticsTimeRange.QUARTER:
            start_time = now - timedelta(days=90)
        elif time_range == AnalyticsTimeRange.YEAR:
            start_time = now - timedelta(days=365)
        else:
            start_time = now - timedelta(days=1)
        
        # Filter events by time range
        filtered_events = [e for e in self.events if e.timestamp >= start_time]
        
        # Calculate summary statistics
        total_events = len(filtered_events)
        unique_users = len(set(e.user_id for e in filtered_events if e.user_id))
        unique_sessions = len(set(e.session_id for e in filtered_events if e.session_id))
        
        # Event type distribution
        event_types = Counter(e.event_type.value for e in filtered_events)
        
        # User activity
        user_activity = Counter(e.user_id for e in filtered_events if e.user_id)
        top_users = user_activity.most_common(5)
        
        # Recent insights
        recent_insights = [i for i in self.insights if i.created_at >= start_time]
        
        return {
            "time_range": time_range.value,
            "start_time": start_time.isoformat(),
            "end_time": now.isoformat(),
            "total_events": total_events,
            "unique_users": unique_users,
            "unique_sessions": unique_sessions,
            "event_types": dict(event_types),
            "top_users": [{"user_id": user_id, "event_count": count} for user_id, count in top_users],
            "recent_insights": len(recent_insights),
            "timestamp": now.isoformat()
        }
    
    def get_workflow_analytics(self, time_range: AnalyticsTimeRange = AnalyticsTimeRange.DAY) -> Dict[str, Any]:
        """Get workflow-specific analytics"""
        now = DateTimeHelpers.now_utc()
        
        # Calculate time range
        if time_range == AnalyticsTimeRange.HOUR:
            start_time = now - timedelta(hours=1)
        elif time_range == AnalyticsTimeRange.DAY:
            start_time = now - timedelta(days=1)
        elif time_range == AnalyticsTimeRange.WEEK:
            start_time = now - timedelta(weeks=1)
        elif time_range == AnalyticsTimeRange.MONTH:
            start_time = now - timedelta(days=30)
        else:
            start_time = now - timedelta(days=1)
        
        # Filter workflow events
        workflow_events = [e for e in self.events if e.timestamp >= start_time and e.event_type in [
            AnalyticsEventType.WORKFLOW_CREATED,
            AnalyticsEventType.WORKFLOW_COMPLETED,
            AnalyticsEventType.WORKFLOW_FAILED
        ]]
        
        created_count = len([e for e in workflow_events if e.event_type == AnalyticsEventType.WORKFLOW_CREATED])
        completed_count = len([e for e in workflow_events if e.event_type == AnalyticsEventType.WORKFLOW_COMPLETED])
        failed_count = len([e for e in workflow_events if e.event_type == AnalyticsEventType.WORKFLOW_FAILED])
        
        # Calculate rates
        completion_rate = completed_count / created_count if created_count > 0 else 0
        failure_rate = failed_count / created_count if created_count > 0 else 0
        
        # Calculate average duration
        durations = []
        for event in workflow_events:
            if event.event_type == AnalyticsEventType.WORKFLOW_COMPLETED and "duration" in event.metrics:
                durations.append(event.metrics["duration"])
        
        avg_duration = statistics.mean(durations) if durations else 0
        
        return {
            "time_range": time_range.value,
            "start_time": start_time.isoformat(),
            "end_time": now.isoformat(),
            "workflows_created": created_count,
            "workflows_completed": completed_count,
            "workflows_failed": failed_count,
            "completion_rate": completion_rate,
            "failure_rate": failure_rate,
            "average_duration": avg_duration,
            "timestamp": now.isoformat()
        }
    
    def get_user_analytics(self, time_range: AnalyticsTimeRange = AnalyticsTimeRange.DAY) -> Dict[str, Any]:
        """Get user-specific analytics"""
        now = DateTimeHelpers.now_utc()
        
        # Calculate time range
        if time_range == AnalyticsTimeRange.HOUR:
            start_time = now - timedelta(hours=1)
        elif time_range == AnalyticsTimeRange.DAY:
            start_time = now - timedelta(days=1)
        elif time_range == AnalyticsTimeRange.WEEK:
            start_time = now - timedelta(weeks=1)
        elif time_range == AnalyticsTimeRange.MONTH:
            start_time = now - timedelta(days=30)
        else:
            start_time = now - timedelta(days=1)
        
        # Filter user events
        user_events = [e for e in self.events if e.timestamp >= start_time and e.user_id is not None]
        
        unique_users = len(set(e.user_id for e in user_events))
        total_events = len(user_events)
        
        # User activity distribution
        user_activity = Counter(e.user_id for e in user_events)
        top_users = user_activity.most_common(10)
        
        # Login events
        login_events = [e for e in user_events if e.event_type == AnalyticsEventType.USER_LOGIN]
        unique_logins = len(set(e.user_id for e in login_events))
        
        return {
            "time_range": time_range.value,
            "start_time": start_time.isoformat(),
            "end_time": now.isoformat(),
            "unique_users": unique_users,
            "total_events": total_events,
            "unique_logins": unique_logins,
            "top_users": [{"user_id": user_id, "event_count": count} for user_id, count in top_users],
            "timestamp": now.isoformat()
        }
    
    def get_performance_analytics(self, time_range: AnalyticsTimeRange = AnalyticsTimeRange.DAY) -> Dict[str, Any]:
        """Get performance analytics"""
        now = DateTimeHelpers.now_utc()
        
        # Calculate time range
        if time_range == AnalyticsTimeRange.HOUR:
            start_time = now - timedelta(hours=1)
        elif time_range == AnalyticsTimeRange.DAY:
            start_time = now - timedelta(days=1)
        elif time_range == AnalyticsTimeRange.WEEK:
            start_time = now - timedelta(weeks=1)
        elif time_range == AnalyticsTimeRange.MONTH:
            start_time = now - timedelta(days=30)
        else:
            start_time = now - timedelta(days=1)
        
        # Filter performance events
        performance_events = [e for e in self.events if e.timestamp >= start_time and e.event_type == AnalyticsEventType.PERFORMANCE_METRIC]
        
        # Extract performance metrics
        response_times = []
        error_rates = []
        
        for event in performance_events:
            if "response_time" in event.metrics:
                response_times.append(event.metrics["response_time"])
            if "error_rate" in event.metrics:
                error_rates.append(event.metrics["error_rate"])
        
        # Calculate statistics
        avg_response_time = statistics.mean(response_times) if response_times else 0
        p95_response_time = np.percentile(response_times, 95) if response_times else 0
        p99_response_time = np.percentile(response_times, 99) if response_times else 0
        
        avg_error_rate = statistics.mean(error_rates) if error_rates else 0
        
        return {
            "time_range": time_range.value,
            "start_time": start_time.isoformat(),
            "end_time": now.isoformat(),
            "average_response_time": avg_response_time,
            "p95_response_time": p95_response_time,
            "p99_response_time": p99_response_time,
            "average_error_rate": avg_error_rate,
            "total_requests": len(performance_events),
            "timestamp": now.isoformat()
        }
    
    def get_insights(self, limit: int = 10) -> List[AnalyticsInsight]:
        """Get analytics insights"""
        return self.insights[-limit:]
    
    def export_analytics(self, time_range: AnalyticsTimeRange = AnalyticsTimeRange.DAY, format: str = "json") -> str:
        """Export analytics data"""
        summary = self.get_analytics_summary(time_range)
        workflow_analytics = self.get_workflow_analytics(time_range)
        user_analytics = self.get_user_analytics(time_range)
        performance_analytics = self.get_performance_analytics(time_range)
        insights = self.get_insights()
        
        data = {
            "summary": summary,
            "workflow_analytics": workflow_analytics,
            "user_analytics": user_analytics,
            "performance_analytics": performance_analytics,
            "insights": [
                {
                    "id": insight.id,
                    "type": insight.type,
                    "title": insight.title,
                    "description": insight.description,
                    "confidence": insight.confidence,
                    "impact": insight.impact,
                    "recommendation": insight.recommendation,
                    "data": insight.data,
                    "created_at": insight.created_at.isoformat()
                }
                for insight in insights
            ]
        }
        
        if format == "json":
            return json.dumps(data, indent=2)
        elif format == "csv":
            # Convert to CSV format (simplified)
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write summary data
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Total Events", summary["total_events"]])
            writer.writerow(["Unique Users", summary["unique_users"]])
            writer.writerow(["Workflows Created", workflow_analytics["workflows_created"]])
            writer.writerow(["Workflows Completed", workflow_analytics["workflows_completed"]])
            writer.writerow(["Completion Rate", workflow_analytics["completion_rate"]])
            writer.writerow(["Average Response Time", performance_analytics["average_response_time"]])
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global analytics service
analytics_service = AnalyticsService()


# Utility functions
async def start_analytics_service():
    """Start the analytics service"""
    await analytics_service.start()


async def stop_analytics_service():
    """Stop the analytics service"""
    await analytics_service.stop()


def track_event(
    event_type: AnalyticsEventType,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    properties: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
    dimensions: Optional[Dict[str, str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Track analytics event"""
    return analytics_service.track_event(
        event_type, user_id, session_id, properties, metrics, dimensions, metadata
    )


def record_metric(
    name: str,
    value: float,
    dimensions: Optional[Dict[str, str]] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Record analytics metric"""
    analytics_service.record_metric(name, value, dimensions, metadata)


def get_analytics_summary(time_range: AnalyticsTimeRange = AnalyticsTimeRange.DAY) -> Dict[str, Any]:
    """Get analytics summary"""
    return analytics_service.get_analytics_summary(time_range)


def get_workflow_analytics(time_range: AnalyticsTimeRange = AnalyticsTimeRange.DAY) -> Dict[str, Any]:
    """Get workflow analytics"""
    return analytics_service.get_workflow_analytics(time_range)


def get_user_analytics(time_range: AnalyticsTimeRange = AnalyticsTimeRange.DAY) -> Dict[str, Any]:
    """Get user analytics"""
    return analytics_service.get_user_analytics(time_range)


def get_performance_analytics(time_range: AnalyticsTimeRange = AnalyticsTimeRange.DAY) -> Dict[str, Any]:
    """Get performance analytics"""
    return analytics_service.get_performance_analytics(time_range)


def get_insights(limit: int = 10) -> List[AnalyticsInsight]:
    """Get analytics insights"""
    return analytics_service.get_insights(limit)


def export_analytics(time_range: AnalyticsTimeRange = AnalyticsTimeRange.DAY, format: str = "json") -> str:
    """Export analytics data"""
    return analytics_service.export_analytics(time_range, format)


# Common analytics tracking functions
def track_workflow_created(workflow_id: str, user_id: str, workflow_name: str):
    """Track workflow creation"""
    return track_event(
        AnalyticsEventType.WORKFLOW_CREATED,
        user_id=user_id,
        properties={"workflow_id": workflow_id, "workflow_name": workflow_name}
    )


def track_workflow_completed(workflow_id: str, user_id: str, duration: float):
    """Track workflow completion"""
    return track_event(
        AnalyticsEventType.WORKFLOW_COMPLETED,
        user_id=user_id,
        properties={"workflow_id": workflow_id},
        metrics={"duration": duration}
    )


def track_workflow_failed(workflow_id: str, user_id: str, error_message: str):
    """Track workflow failure"""
    return track_event(
        AnalyticsEventType.WORKFLOW_FAILED,
        user_id=user_id,
        properties={"workflow_id": workflow_id, "error_message": error_message}
    )


def track_user_login(user_id: str, session_id: str, login_method: str):
    """Track user login"""
    return track_event(
        AnalyticsEventType.USER_LOGIN,
        user_id=user_id,
        session_id=session_id,
        properties={"login_method": login_method}
    )


def track_api_request(endpoint: str, method: str, response_time: float, status_code: int):
    """Track API request"""
    return track_event(
        AnalyticsEventType.API_REQUEST,
        properties={"endpoint": endpoint, "method": method, "status_code": status_code},
        metrics={"response_time": response_time}
    )


def track_error(error_type: str, error_message: str, user_id: Optional[str] = None):
    """Track error occurrence"""
    return track_event(
        AnalyticsEventType.ERROR_OCCURRED,
        user_id=user_id,
        properties={"error_type": error_type, "error_message": error_message}
    )




