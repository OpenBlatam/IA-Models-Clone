"""
Analytics Service - Fast Implementation
=======================================

Fast analytics service with intelligent insights.
"""

from __future__ import annotations
import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)


class AnalyticsEventType(str, Enum):
    """Analytics event type enumeration"""
    WORKFLOW_CREATED = "workflow_created"
    WORKFLOW_UPDATED = "workflow_updated"
    WORKFLOW_DELETED = "workflow_deleted"
    WORKFLOW_EXECUTED = "workflow_executed"
    NODE_ADDED = "node_added"
    NODE_EXECUTED = "node_executed"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    API_REQUEST = "api_request"
    ERROR_OCCURRED = "error_occurred"


class AnalyticsMetricType(str, Enum):
    """Analytics metric type enumeration"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AnalyticsTimeRange(str, Enum):
    """Analytics time range enumeration"""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class AnalyticsService:
    """Fast analytics service with intelligent insights"""
    
    def __init__(self):
        self.events = []
        self.metrics = {}
        self.insights = []
        self.reports = {}
        self.stats = {
            "total_events": 0,
            "total_metrics": 0,
            "total_insights": 0,
            "by_event_type": {event_type.value: 0 for event_type in AnalyticsEventType}
        }
    
    async def track_event(
        self,
        event_type: AnalyticsEventType,
        data: Dict[str, Any],
        user_id: Optional[int] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Track analytics event"""
        try:
            # Create event record
            event = {
                "id": len(self.events) + 1,
                "event_type": event_type.value,
                "data": data,
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {
                    "source": "workflow_chain",
                    "version": "3.0.0"
                }
            }
            
            # Store event
            self.events.append(event)
            self.stats["total_events"] += 1
            self.stats["by_event_type"][event_type.value] += 1
            
            # Generate insights
            await self._generate_insights(event)
            
            # Update metrics
            await self._update_metrics(event)
            
            logger.info(f"Event tracked: {event_type.value}")
            return event
        
        except Exception as e:
            logger.error(f"Failed to track event: {e}")
            return {"error": str(e)}
    
    async def record_metric(
        self,
        name: str,
        value: Union[int, float],
        metric_type: AnalyticsMetricType = AnalyticsMetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Record analytics metric"""
        try:
            # Create metric record
            metric = {
                "id": len(self.metrics) + 1,
                "name": name,
                "value": value,
                "type": metric_type.value,
                "labels": labels or {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store metric
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(metric)
            self.stats["total_metrics"] += 1
            
            logger.info(f"Metric recorded: {name} = {value}")
            return metric
        
        except Exception as e:
            logger.error(f"Failed to record metric: {e}")
            return {"error": str(e)}
    
    async def get_analytics_summary(
        self,
        time_range: AnalyticsTimeRange = AnalyticsTimeRange.DAY
    ) -> Dict[str, Any]:
        """Get analytics summary"""
        try:
            # Calculate time range
            end_time = datetime.utcnow()
            start_time = self._calculate_start_time(end_time, time_range)
            
            # Filter events by time range
            filtered_events = [
                event for event in self.events
                if start_time <= datetime.fromisoformat(event["timestamp"]) <= end_time
            ]
            
            # Calculate summary
            summary = {
                "time_range": time_range.value,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_events": len(filtered_events),
                "events_by_type": {},
                "top_events": [],
                "user_activity": {},
                "workflow_activity": {},
                "error_rate": 0,
                "performance_metrics": {}
            }
            
            # Count events by type
            for event in filtered_events:
                event_type = event["event_type"]
                if event_type not in summary["events_by_type"]:
                    summary["events_by_type"][event_type] = 0
                summary["events_by_type"][event_type] += 1
            
            # Get top events
            summary["top_events"] = sorted(
                summary["events_by_type"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            # Calculate error rate
            error_events = summary["events_by_type"].get("error_occurred", 0)
            total_events = len(filtered_events)
            summary["error_rate"] = (error_events / total_events * 100) if total_events > 0 else 0
            
            # Calculate performance metrics
            summary["performance_metrics"] = await self._calculate_performance_metrics(filtered_events)
            
            logger.info(f"Analytics summary generated for {time_range.value}")
            return summary
        
        except Exception as e:
            logger.error(f"Failed to get analytics summary: {e}")
            return {"error": str(e)}
    
    async def get_workflow_analytics(
        self,
        workflow_id: Optional[int] = None,
        time_range: AnalyticsTimeRange = AnalyticsTimeRange.DAY
    ) -> Dict[str, Any]:
        """Get workflow-specific analytics"""
        try:
            # Calculate time range
            end_time = datetime.utcnow()
            start_time = self._calculate_start_time(end_time, time_range)
            
            # Filter events by workflow and time range
            filtered_events = [
                event for event in self.events
                if start_time <= datetime.fromisoformat(event["timestamp"]) <= end_time
                and (workflow_id is None or event["data"].get("workflow_id") == workflow_id)
            ]
            
            # Calculate workflow analytics
            analytics = {
                "workflow_id": workflow_id,
                "time_range": time_range.value,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_events": len(filtered_events),
                "workflow_events": {},
                "execution_stats": {},
                "performance_metrics": {},
                "user_activity": {}
            }
            
            # Count workflow events
            for event in filtered_events:
                event_type = event["event_type"]
                if event_type not in analytics["workflow_events"]:
                    analytics["workflow_events"][event_type] = 0
                analytics["workflow_events"][event_type] += 1
            
            # Calculate execution stats
            analytics["execution_stats"] = await self._calculate_execution_stats(filtered_events)
            
            # Calculate performance metrics
            analytics["performance_metrics"] = await self._calculate_performance_metrics(filtered_events)
            
            logger.info(f"Workflow analytics generated for workflow {workflow_id}")
            return analytics
        
        except Exception as e:
            logger.error(f"Failed to get workflow analytics: {e}")
            return {"error": str(e)}
    
    async def get_user_analytics(
        self,
        user_id: Optional[int] = None,
        time_range: AnalyticsTimeRange = AnalyticsTimeRange.DAY
    ) -> Dict[str, Any]:
        """Get user-specific analytics"""
        try:
            # Calculate time range
            end_time = datetime.utcnow()
            start_time = self._calculate_start_time(end_time, time_range)
            
            # Filter events by user and time range
            filtered_events = [
                event for event in self.events
                if start_time <= datetime.fromisoformat(event["timestamp"]) <= end_time
                and (user_id is None or event["user_id"] == user_id)
            ]
            
            # Calculate user analytics
            analytics = {
                "user_id": user_id,
                "time_range": time_range.value,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_events": len(filtered_events),
                "user_events": {},
                "activity_patterns": {},
                "workflow_usage": {},
                "session_stats": {}
            }
            
            # Count user events
            for event in filtered_events:
                event_type = event["event_type"]
                if event_type not in analytics["user_events"]:
                    analytics["user_events"][event_type] = 0
                analytics["user_events"][event_type] += 1
            
            # Calculate activity patterns
            analytics["activity_patterns"] = await self._calculate_activity_patterns(filtered_events)
            
            # Calculate workflow usage
            analytics["workflow_usage"] = await self._calculate_workflow_usage(filtered_events)
            
            logger.info(f"User analytics generated for user {user_id}")
            return analytics
        
        except Exception as e:
            logger.error(f"Failed to get user analytics: {e}")
            return {"error": str(e)}
    
    async def get_insights(
        self,
        time_range: AnalyticsTimeRange = AnalyticsTimeRange.DAY,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get intelligent insights"""
        try:
            # Calculate time range
            end_time = datetime.utcnow()
            start_time = self._calculate_start_time(end_time, time_range)
            
            # Filter insights by time range
            filtered_insights = [
                insight for insight in self.insights
                if start_time <= datetime.fromisoformat(insight["timestamp"]) <= end_time
            ]
            
            # Sort by importance and timestamp
            filtered_insights.sort(
                key=lambda x: (x.get("importance", 0), x["timestamp"]),
                reverse=True
            )
            
            # Limit results
            return filtered_insights[:limit]
        
        except Exception as e:
            logger.error(f"Failed to get insights: {e}")
            return []
    
    async def _generate_insights(self, event: Dict[str, Any]) -> None:
        """Generate intelligent insights from event"""
        try:
            # Simple insight generation
            if event["event_type"] == "workflow_created":
                insight = {
                    "id": len(self.insights) + 1,
                    "type": "workflow_activity",
                    "title": "New Workflow Created",
                    "description": f"Workflow '{event['data'].get('name', 'Unknown')}' was created",
                    "importance": 3,
                    "timestamp": event["timestamp"],
                    "metadata": event["data"]
                }
                self.insights.append(insight)
                self.stats["total_insights"] += 1
            
            elif event["event_type"] == "error_occurred":
                insight = {
                    "id": len(self.insights) + 1,
                    "type": "error_alert",
                    "title": "Error Occurred",
                    "description": f"Error in {event['data'].get('component', 'Unknown')}: {event['data'].get('error', 'Unknown error')}",
                    "importance": 5,
                    "timestamp": event["timestamp"],
                    "metadata": event["data"]
                }
                self.insights.append(insight)
                self.stats["total_insights"] += 1
        
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
    
    async def _update_metrics(self, event: Dict[str, Any]) -> None:
        """Update metrics based on event"""
        try:
            # Update event type counter
            await self.record_metric(
                name=f"events_{event['event_type']}",
                value=1,
                metric_type=AnalyticsMetricType.COUNTER
            )
            
            # Update specific metrics based on event type
            if event["event_type"] == "workflow_executed":
                await self.record_metric(
                    name="workflow_execution_time",
                    value=event["data"].get("execution_time", 0),
                    metric_type=AnalyticsMetricType.HISTOGRAM
                )
            
            elif event["event_type"] == "api_request":
                await self.record_metric(
                    name="api_response_time",
                    value=event["data"].get("response_time", 0),
                    metric_type=AnalyticsMetricType.HISTOGRAM
                )
        
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    def _calculate_start_time(self, end_time: datetime, time_range: AnalyticsTimeRange) -> datetime:
        """Calculate start time based on time range"""
        if time_range == AnalyticsTimeRange.HOUR:
            return end_time - timedelta(hours=1)
        elif time_range == AnalyticsTimeRange.DAY:
            return end_time - timedelta(days=1)
        elif time_range == AnalyticsTimeRange.WEEK:
            return end_time - timedelta(weeks=1)
        elif time_range == AnalyticsTimeRange.MONTH:
            return end_time - timedelta(days=30)
        elif time_range == AnalyticsTimeRange.QUARTER:
            return end_time - timedelta(days=90)
        elif time_range == AnalyticsTimeRange.YEAR:
            return end_time - timedelta(days=365)
        else:
            return end_time - timedelta(days=1)
    
    async def _calculate_performance_metrics(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics"""
        try:
            metrics = {
                "average_response_time": 0,
                "total_requests": 0,
                "error_rate": 0,
                "throughput": 0
            }
            
            # Calculate metrics from events
            response_times = []
            error_count = 0
            
            for event in events:
                if event["event_type"] == "api_request":
                    response_time = event["data"].get("response_time", 0)
                    if response_time > 0:
                        response_times.append(response_time)
                
                if event["event_type"] == "error_occurred":
                    error_count += 1
            
            # Calculate averages
            if response_times:
                metrics["average_response_time"] = sum(response_times) / len(response_times)
            
            metrics["total_requests"] = len(events)
            metrics["error_rate"] = (error_count / len(events) * 100) if events else 0
            metrics["throughput"] = len(events) / 3600  # events per hour
            
            return metrics
        
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {e}")
            return {}
    
    async def _calculate_execution_stats(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate execution statistics"""
        try:
            stats = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "average_execution_time": 0
            }
            
            execution_times = []
            
            for event in events:
                if event["event_type"] == "workflow_executed":
                    stats["total_executions"] += 1
                    
                    if event["data"].get("success", False):
                        stats["successful_executions"] += 1
                    else:
                        stats["failed_executions"] += 1
                    
                    execution_time = event["data"].get("execution_time", 0)
                    if execution_time > 0:
                        execution_times.append(execution_time)
            
            if execution_times:
                stats["average_execution_time"] = sum(execution_times) / len(execution_times)
            
            return stats
        
        except Exception as e:
            logger.error(f"Failed to calculate execution stats: {e}")
            return {}
    
    async def _calculate_activity_patterns(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate user activity patterns"""
        try:
            patterns = {
                "hourly_activity": {},
                "daily_activity": {},
                "most_active_hour": 0,
                "most_active_day": "unknown"
            }
            
            # Count activity by hour and day
            for event in events:
                timestamp = datetime.fromisoformat(event["timestamp"])
                hour = timestamp.hour
                day = timestamp.strftime("%A")
                
                patterns["hourly_activity"][hour] = patterns["hourly_activity"].get(hour, 0) + 1
                patterns["daily_activity"][day] = patterns["daily_activity"].get(day, 0) + 1
            
            # Find most active hour and day
            if patterns["hourly_activity"]:
                patterns["most_active_hour"] = max(patterns["hourly_activity"], key=patterns["hourly_activity"].get)
            
            if patterns["daily_activity"]:
                patterns["most_active_day"] = max(patterns["daily_activity"], key=patterns["daily_activity"].get)
            
            return patterns
        
        except Exception as e:
            logger.error(f"Failed to calculate activity patterns: {e}")
            return {}
    
    async def _calculate_workflow_usage(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate workflow usage statistics"""
        try:
            usage = {
                "workflows_used": {},
                "most_used_workflow": "unknown",
                "total_workflow_events": 0
            }
            
            # Count workflow usage
            for event in events:
                if "workflow_id" in event["data"]:
                    workflow_id = event["data"]["workflow_id"]
                    usage["workflows_used"][workflow_id] = usage["workflows_used"].get(workflow_id, 0) + 1
                    usage["total_workflow_events"] += 1
            
            # Find most used workflow
            if usage["workflows_used"]:
                usage["most_used_workflow"] = max(usage["workflows_used"], key=usage["workflows_used"].get)
            
            return usage
        
        except Exception as e:
            logger.error(f"Failed to calculate workflow usage: {e}")
            return {}
    
    async def get_analytics_stats(self) -> Dict[str, Any]:
        """Get analytics service statistics"""
        try:
            return {
                "total_events": self.stats["total_events"],
                "total_metrics": self.stats["total_metrics"],
                "total_insights": self.stats["total_insights"],
                "events_by_type": self.stats["by_event_type"],
                "available_reports": len(self.reports),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get analytics stats: {e}")
            return {"error": str(e)}


# Global analytics service instance
analytics_service = AnalyticsService()