"""
AI Integration System - Analytics and Reporting
Advanced analytics, reporting, and insights for integration performance
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
from collections import defaultdict, Counter

from .database import get_db_session
from .models import (
    IntegrationRequest, IntegrationResult, IntegrationLog, 
    IntegrationMetrics, WebhookEvent
)

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics"""
    SUCCESS_RATE = "success_rate"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    PLATFORM_HEALTH = "platform_health"
    CONTENT_DISTRIBUTION = "content_distribution"
    USER_ACTIVITY = "user_activity"

class TimeRange(Enum):
    """Time range options for analytics"""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"

@dataclass
class AnalyticsData:
    """Analytics data structure"""
    metric_type: MetricType
    time_range: TimeRange
    data_points: List[Dict[str, Any]]
    summary: Dict[str, Any]
    generated_at: datetime

@dataclass
class PerformanceMetrics:
    """Performance metrics structure"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    average_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput_per_hour: float
    error_rate: float

@dataclass
class PlatformAnalytics:
    """Platform-specific analytics"""
    platform: str
    total_integrations: int
    successful_integrations: int
    failed_integrations: int
    success_rate: float
    average_response_time: float
    last_health_check: Optional[datetime]
    health_status: bool
    error_types: Dict[str, int]

@dataclass
class ContentAnalytics:
    """Content-specific analytics"""
    content_type: str
    total_created: int
    platforms_distributed: Dict[str, int]
    average_processing_time: float
    success_rate: float
    popular_tags: List[Tuple[str, int]]

class AnalyticsEngine:
    """Advanced analytics engine for integration data"""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.real_time_metrics = defaultdict(list)
    
    async def get_performance_metrics(self, time_range: TimeRange = TimeRange.DAY) -> PerformanceMetrics:
        """Get overall performance metrics"""
        try:
            cache_key = f"performance_metrics_{time_range.value}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            with get_db_session() as session:
                # Calculate time range
                end_time = datetime.utcnow()
                start_time = self._get_start_time(end_time, time_range)
                
                # Get integration requests in time range
                requests = session.query(IntegrationRequest).filter(
                    IntegrationRequest.created_at >= start_time,
                    IntegrationRequest.created_at <= end_time
                ).all()
                
                # Get integration results
                request_ids = [req.id for req in requests]
                results = session.query(IntegrationResult).filter(
                    IntegrationResult.request_id.in_(request_ids)
                ).all()
                
                # Calculate metrics
                total_requests = len(requests)
                successful_requests = len([r for r in results if r.status == "completed"])
                failed_requests = len([r for r in results if r.status == "failed"])
                
                success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
                error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0
                
                # Calculate response times
                response_times = []
                for request in requests:
                    if request.completed_at and request.started_at:
                        response_time = (request.completed_at - request.started_at).total_seconds()
                        response_times.append(response_time)
                
                avg_response_time = statistics.mean(response_times) if response_times else 0
                median_response_time = statistics.median(response_times) if response_times else 0
                p95_response_time = self._calculate_percentile(response_times, 95) if response_times else 0
                p99_response_time = self._calculate_percentile(response_times, 99) if response_times else 0
                
                # Calculate throughput
                time_diff_hours = (end_time - start_time).total_seconds() / 3600
                throughput_per_hour = total_requests / time_diff_hours if time_diff_hours > 0 else 0
                
                metrics = PerformanceMetrics(
                    total_requests=total_requests,
                    successful_requests=successful_requests,
                    failed_requests=failed_requests,
                    success_rate=success_rate,
                    average_response_time=avg_response_time,
                    median_response_time=median_response_time,
                    p95_response_time=p95_response_time,
                    p99_response_time=p99_response_time,
                    throughput_per_hour=throughput_per_hour,
                    error_rate=error_rate
                )
                
                self._cache_data(cache_key, metrics)
                return metrics
                
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    async def get_platform_analytics(self, time_range: TimeRange = TimeRange.DAY) -> List[PlatformAnalytics]:
        """Get platform-specific analytics"""
        try:
            cache_key = f"platform_analytics_{time_range.value}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            with get_db_session() as session:
                # Calculate time range
                end_time = datetime.utcnow()
                start_time = self._get_start_time(end_time, time_range)
                
                # Get platform results
                results = session.query(IntegrationResult).filter(
                    IntegrationResult.created_at >= start_time,
                    IntegrationResult.created_at <= end_time
                ).all()
                
                # Group by platform
                platform_data = defaultdict(lambda: {
                    'total': 0,
                    'successful': 0,
                    'failed': 0,
                    'response_times': [],
                    'errors': Counter()
                })
                
                for result in results:
                    platform = result.platform
                    platform_data[platform]['total'] += 1
                    
                    if result.status == "completed":
                        platform_data[platform]['successful'] += 1
                    else:
                        platform_data[platform]['failed'] += 1
                        if result.error_message:
                            error_type = self._categorize_error(result.error_message)
                            platform_data[platform]['errors'][error_type] += 1
                
                # Get health status
                health_metrics = session.query(IntegrationMetrics).filter(
                    IntegrationMetrics.metric_type == "health_check",
                    IntegrationMetrics.timestamp >= start_time
                ).all()
                
                platform_health = {}
                for metric in health_metrics:
                    platform_health[metric.platform] = {
                        'status': metric.metric_value == "1",
                        'timestamp': metric.timestamp
                    }
                
                # Build analytics
                analytics = []
                for platform, data in platform_data.items():
                    success_rate = (data['successful'] / data['total'] * 100) if data['total'] > 0 else 0
                    avg_response_time = statistics.mean(data['response_times']) if data['response_times'] else 0
                    
                    health_info = platform_health.get(platform, {})
                    
                    analytics.append(PlatformAnalytics(
                        platform=platform,
                        total_integrations=data['total'],
                        successful_integrations=data['successful'],
                        failed_integrations=data['failed'],
                        success_rate=success_rate,
                        average_response_time=avg_response_time,
                        last_health_check=health_info.get('timestamp'),
                        health_status=health_info.get('status', False),
                        error_types=dict(data['errors'])
                    ))
                
                self._cache_data(cache_key, analytics)
                return analytics
                
        except Exception as e:
            logger.error(f"Error calculating platform analytics: {str(e)}")
            return []
    
    async def get_content_analytics(self, time_range: TimeRange = TimeRange.DAY) -> List[ContentAnalytics]:
        """Get content-specific analytics"""
        try:
            cache_key = f"content_analytics_{time_range.value}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            with get_db_session() as session:
                # Calculate time range
                end_time = datetime.utcnow()
                start_time = self._get_start_time(end_time, time_range)
                
                # Get integration requests
                requests = session.query(IntegrationRequest).filter(
                    IntegrationRequest.created_at >= start_time,
                    IntegrationRequest.created_at <= end_time
                ).all()
                
                # Group by content type
                content_data = defaultdict(lambda: {
                    'total': 0,
                    'platforms': Counter(),
                    'processing_times': [],
                    'successful': 0,
                    'tags': Counter()
                })
                
                for request in requests:
                    content_type = request.content_type.value
                    content_data[content_type]['total'] += 1
                    
                    # Extract tags
                    if request.content_data and 'tags' in request.content_data:
                        for tag in request.content_data['tags']:
                            content_data[content_type]['tags'][tag] += 1
                    
                    # Get platform distribution
                    for platform in request.target_platforms:
                        content_data[content_type]['platforms'][platform] += 1
                    
                    # Calculate processing time
                    if request.completed_at and request.started_at:
                        processing_time = (request.completed_at - request.started_at).total_seconds()
                        content_data[content_type]['processing_times'].append(processing_time)
                    
                    # Count successful requests
                    if request.status == "completed":
                        content_data[content_type]['successful'] += 1
                
                # Build analytics
                analytics = []
                for content_type, data in content_data.items():
                    success_rate = (data['successful'] / data['total'] * 100) if data['total'] > 0 else 0
                    avg_processing_time = statistics.mean(data['processing_times']) if data['processing_times'] else 0
                    popular_tags = data['tags'].most_common(10)
                    
                    analytics.append(ContentAnalytics(
                        content_type=content_type,
                        total_created=data['total'],
                        platforms_distributed=dict(data['platforms']),
                        average_processing_time=avg_processing_time,
                        success_rate=success_rate,
                        popular_tags=popular_tags
                    ))
                
                self._cache_data(cache_key, analytics)
                return analytics
                
        except Exception as e:
            logger.error(f"Error calculating content analytics: {str(e)}")
            return []
    
    async def get_trend_analysis(self, metric_type: MetricType, time_range: TimeRange = TimeRange.DAY) -> AnalyticsData:
        """Get trend analysis for specific metrics"""
        try:
            cache_key = f"trend_analysis_{metric_type.value}_{time_range.value}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            with get_db_session() as session:
                # Calculate time range
                end_time = datetime.utcnow()
                start_time = self._get_start_time(end_time, time_range)
                
                # Get data points based on metric type
                data_points = []
                
                if metric_type == MetricType.SUCCESS_RATE:
                    data_points = await self._get_success_rate_trend(session, start_time, end_time, time_range)
                elif metric_type == MetricType.RESPONSE_TIME:
                    data_points = await self._get_response_time_trend(session, start_time, end_time, time_range)
                elif metric_type == MetricType.THROUGHPUT:
                    data_points = await self._get_throughput_trend(session, start_time, end_time, time_range)
                elif metric_type == MetricType.ERROR_RATE:
                    data_points = await self._get_error_rate_trend(session, start_time, end_time, time_range)
                
                # Calculate summary statistics
                summary = self._calculate_trend_summary(data_points)
                
                analytics_data = AnalyticsData(
                    metric_type=metric_type,
                    time_range=time_range,
                    data_points=data_points,
                    summary=summary,
                    generated_at=datetime.utcnow()
                )
                
                self._cache_data(cache_key, analytics_data)
                return analytics_data
                
        except Exception as e:
            logger.error(f"Error calculating trend analysis: {str(e)}")
            return AnalyticsData(
                metric_type=metric_type,
                time_range=time_range,
                data_points=[],
                summary={},
                generated_at=datetime.utcnow()
            )
    
    async def get_user_activity_analytics(self, time_range: TimeRange = TimeRange.DAY) -> Dict[str, Any]:
        """Get user activity analytics"""
        try:
            cache_key = f"user_activity_{time_range.value}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            with get_db_session() as session:
                # Calculate time range
                end_time = datetime.utcnow()
                start_time = self._get_start_time(end_time, time_range)
                
                # Get user activity from logs
                logs = session.query(IntegrationLog).filter(
                    IntegrationLog.created_at >= start_time,
                    IntegrationLog.created_at <= end_time
                ).all()
                
                # Analyze user activity
                user_activity = {
                    'total_actions': len(logs),
                    'unique_users': len(set(log.user_id for log in logs if hasattr(log, 'user_id'))),
                    'action_types': Counter(log.action for log in logs),
                    'hourly_activity': defaultdict(int),
                    'daily_activity': defaultdict(int)
                }
                
                for log in logs:
                    # Hourly activity
                    hour = log.created_at.hour
                    user_activity['hourly_activity'][hour] += 1
                    
                    # Daily activity
                    day = log.created_at.date()
                    user_activity['daily_activity'][day] += 1
                
                # Convert to regular dicts
                user_activity['action_types'] = dict(user_activity['action_types'])
                user_activity['hourly_activity'] = dict(user_activity['hourly_activity'])
                user_activity['daily_activity'] = dict(user_activity['daily_activity'])
                
                self._cache_data(cache_key, user_activity)
                return user_activity
                
        except Exception as e:
            logger.error(f"Error calculating user activity analytics: {str(e)}")
            return {}
    
    async def get_webhook_analytics(self, time_range: TimeRange = TimeRange.DAY) -> Dict[str, Any]:
        """Get webhook analytics"""
        try:
            cache_key = f"webhook_analytics_{time_range.value}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            with get_db_session() as session:
                # Calculate time range
                end_time = datetime.utcnow()
                start_time = self._get_start_time(end_time, time_range)
                
                # Get webhook events
                webhooks = session.query(WebhookEvent).filter(
                    WebhookEvent.received_at >= start_time,
                    WebhookEvent.received_at <= end_time
                ).all()
                
                # Analyze webhook data
                webhook_analytics = {
                    'total_webhooks': len(webhooks),
                    'processed_webhooks': len([w for w in webhooks if w.processed]),
                    'failed_webhooks': len([w for w in webhooks if not w.processed]),
                    'platforms': Counter(w.platform for w in webhooks),
                    'event_types': Counter(w.event_type for w in webhooks),
                    'processing_times': []
                }
                
                for webhook in webhooks:
                    if webhook.processed_at:
                        processing_time = (webhook.processed_at - webhook.received_at).total_seconds()
                        webhook_analytics['processing_times'].append(processing_time)
                
                # Calculate average processing time
                if webhook_analytics['processing_times']:
                    webhook_analytics['average_processing_time'] = statistics.mean(webhook_analytics['processing_times'])
                else:
                    webhook_analytics['average_processing_time'] = 0
                
                # Convert counters to regular dicts
                webhook_analytics['platforms'] = dict(webhook_analytics['platforms'])
                webhook_analytics['event_types'] = dict(webhook_analytics['event_types'])
                
                self._cache_data(cache_key, webhook_analytics)
                return webhook_analytics
                
        except Exception as e:
            logger.error(f"Error calculating webhook analytics: {str(e)}")
            return {}
    
    async def generate_comprehensive_report(self, time_range: TimeRange = TimeRange.DAY) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        try:
            logger.info(f"Generating comprehensive report for {time_range.value}")
            
            # Gather all analytics data
            performance_metrics = await self.get_performance_metrics(time_range)
            platform_analytics = await self.get_platform_analytics(time_range)
            content_analytics = await self.get_content_analytics(time_range)
            user_activity = await self.get_user_activity_analytics(time_range)
            webhook_analytics = await self.get_webhook_analytics(time_range)
            
            # Generate trend analysis
            success_rate_trend = await self.get_trend_analysis(MetricType.SUCCESS_RATE, time_range)
            response_time_trend = await self.get_trend_analysis(MetricType.RESPONSE_TIME, time_range)
            
            # Compile comprehensive report
            report = {
                'report_metadata': {
                    'generated_at': datetime.utcnow().isoformat(),
                    'time_range': time_range.value,
                    'report_type': 'comprehensive'
                },
                'performance_metrics': asdict(performance_metrics),
                'platform_analytics': [asdict(pa) for pa in platform_analytics],
                'content_analytics': [asdict(ca) for ca in content_analytics],
                'user_activity': user_activity,
                'webhook_analytics': webhook_analytics,
                'trends': {
                    'success_rate': asdict(success_rate_trend),
                    'response_time': asdict(response_time_trend)
                },
                'insights': self._generate_insights(
                    performance_metrics, platform_analytics, content_analytics
                ),
                'recommendations': self._generate_recommendations(
                    performance_metrics, platform_analytics, content_analytics
                )
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {str(e)}")
            return {'error': str(e)}
    
    def _get_start_time(self, end_time: datetime, time_range: TimeRange) -> datetime:
        """Calculate start time based on time range"""
        if time_range == TimeRange.HOUR:
            return end_time - timedelta(hours=1)
        elif time_range == TimeRange.DAY:
            return end_time - timedelta(days=1)
        elif time_range == TimeRange.WEEK:
            return end_time - timedelta(weeks=1)
        elif time_range == TimeRange.MONTH:
            return end_time - timedelta(days=30)
        elif time_range == TimeRange.QUARTER:
            return end_time - timedelta(days=90)
        elif time_range == TimeRange.YEAR:
            return end_time - timedelta(days=365)
        else:
            return end_time - timedelta(days=1)
    
    def _calculate_percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _categorize_error(self, error_message: str) -> str:
        """Categorize error message"""
        error_lower = error_message.lower()
        if 'timeout' in error_lower:
            return 'timeout'
        elif 'authentication' in error_lower or 'auth' in error_lower:
            return 'authentication'
        elif 'network' in error_lower or 'connection' in error_lower:
            return 'network'
        elif 'rate limit' in error_lower:
            return 'rate_limit'
        elif 'validation' in error_lower:
            return 'validation'
        else:
            return 'other'
    
    def _get_cached_data(self, cache_key: str) -> Any:
        """Get data from cache"""
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if (datetime.utcnow() - timestamp).total_seconds() < self.cache_ttl:
                return data
            else:
                del self.cache[cache_key]
        return None
    
    def _cache_data(self, cache_key: str, data: Any):
        """Cache data with timestamp"""
        self.cache[cache_key] = (data, datetime.utcnow())
    
    def _calculate_trend_summary(self, data_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for trend data"""
        if not data_points:
            return {}
        
        values = [point.get('value', 0) for point in data_points]
        
        return {
            'min': min(values),
            'max': max(values),
            'average': statistics.mean(values),
            'median': statistics.median(values),
            'trend': 'increasing' if values[-1] > values[0] else 'decreasing' if values[-1] < values[0] else 'stable'
        }
    
    def _generate_insights(self, performance: PerformanceMetrics, platforms: List[PlatformAnalytics], content: List[ContentAnalytics]) -> List[str]:
        """Generate insights from analytics data"""
        insights = []
        
        # Performance insights
        if performance.success_rate < 90:
            insights.append(f"Success rate is below 90% ({performance.success_rate:.1f}%). Consider investigating failed integrations.")
        
        if performance.average_response_time > 30:
            insights.append(f"Average response time is high ({performance.average_response_time:.1f}s). Consider optimizing integration processes.")
        
        # Platform insights
        for platform in platforms:
            if platform.success_rate < 80:
                insights.append(f"{platform.platform} has low success rate ({platform.success_rate:.1f}%). Check platform health and configuration.")
        
        # Content insights
        if content:
            most_popular_type = max(content, key=lambda x: x.total_created)
            insights.append(f"Most popular content type is {most_popular_type.content_type} with {most_popular_type.total_created} creations.")
        
        return insights
    
    def _generate_recommendations(self, performance: PerformanceMetrics, platforms: List[PlatformAnalytics], content: List[ContentAnalytics]) -> List[str]:
        """Generate recommendations from analytics data"""
        recommendations = []
        
        # Performance recommendations
        if performance.error_rate > 10:
            recommendations.append("High error rate detected. Review error logs and implement better error handling.")
        
        if performance.throughput_per_hour < 10:
            recommendations.append("Low throughput detected. Consider scaling up workers or optimizing integration processes.")
        
        # Platform recommendations
        unhealthy_platforms = [p for p in platforms if not p.health_status]
        if unhealthy_platforms:
            recommendations.append(f"Unhealthy platforms detected: {', '.join(p.platform for p in unhealthy_platforms)}. Check platform configurations.")
        
        # Content recommendations
        if content:
            slow_content_types = [c for c in content if c.average_processing_time > 60]
            if slow_content_types:
                recommendations.append(f"Slow processing content types: {', '.join(c.content_type for c in slow_content_types)}. Consider optimization.")
        
        return recommendations
    
    async def _get_success_rate_trend(self, session, start_time: datetime, end_time: datetime, time_range: TimeRange) -> List[Dict[str, Any]]:
        """Get success rate trend data"""
        # Implementation for success rate trend
        return []
    
    async def _get_response_time_trend(self, session, start_time: datetime, end_time: datetime, time_range: TimeRange) -> List[Dict[str, Any]]:
        """Get response time trend data"""
        # Implementation for response time trend
        return []
    
    async def _get_throughput_trend(self, session, start_time: datetime, end_time: datetime, time_range: TimeRange) -> List[Dict[str, Any]]:
        """Get throughput trend data"""
        # Implementation for throughput trend
        return []
    
    async def _get_error_rate_trend(self, session, start_time: datetime, end_time: datetime, time_range: TimeRange) -> List[Dict[str, Any]]:
        """Get error rate trend data"""
        # Implementation for error rate trend
        return []

# Global analytics engine instance
analytics_engine = AnalyticsEngine()

# Export main components
__all__ = [
    "AnalyticsEngine",
    "AnalyticsData",
    "PerformanceMetrics",
    "PlatformAnalytics",
    "ContentAnalytics",
    "MetricType",
    "TimeRange",
    "analytics_engine"
]



























