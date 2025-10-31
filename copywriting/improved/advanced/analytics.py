"""
Advanced Analytics Engine
========================

Comprehensive analytics and insights for copywriting performance.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json

import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from ..schemas import CopywritingRequest, CopywritingVariant, FeedbackRequest
from ..services import CopywritingRecord, FeedbackRecord
from ..utils import performance_tracker

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics"""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    USAGE = "usage"
    ENGAGEMENT = "engagement"
    TREND = "trend"


class TimeRange(str, Enum):
    """Time range options"""
    HOUR = "1h"
    DAY = "1d"
    WEEK = "1w"
    MONTH = "1M"
    QUARTER = "1Q"
    YEAR = "1Y"


@dataclass
class MetricData:
    """Metric data structure"""
    metric_type: MetricType
    name: str
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    requests_per_minute: float
    error_rate: float
    cache_hit_rate: float


@dataclass
class QualityMetrics:
    """Content quality metrics"""
    average_confidence_score: float
    average_rating: float
    total_feedback_count: int
    positive_feedback_rate: float
    improvement_suggestions_count: int
    most_common_improvements: List[str]


@dataclass
class UsageMetrics:
    """Usage analytics"""
    total_variants_generated: int
    average_variants_per_request: float
    most_popular_tones: List[Tuple[str, int]]
    most_popular_styles: List[Tuple[str, int]]
    most_popular_purposes: List[Tuple[str, int]]
    average_word_count: float
    cta_inclusion_rate: float


@dataclass
class EngagementMetrics:
    """User engagement metrics"""
    unique_users: int
    repeat_users: int
    session_duration_avg: float
    requests_per_user_avg: float
    feedback_submission_rate: float
    user_retention_rate: float


@dataclass
class TrendData:
    """Trend analysis data"""
    time_period: str
    metric_name: str
    values: List[float]
    timestamps: List[datetime]
    trend_direction: str  # "up", "down", "stable"
    trend_percentage: float


class AnalyticsEngine:
    """Advanced analytics engine for copywriting insights"""
    
    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def get_performance_metrics(
        self,
        session: AsyncSession,
        time_range: TimeRange = TimeRange.DAY
    ) -> PerformanceMetrics:
        """Get performance metrics"""
        cache_key = f"performance_metrics_{time_range}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        start_time = self._get_start_time(time_range)
        
        # Get request statistics
        request_stats = await self._get_request_stats(session, start_time)
        
        # Get response time statistics
        response_times = await self._get_response_times(session, start_time)
        
        # Calculate metrics
        total_requests = request_stats['total']
        successful_requests = request_stats['successful']
        failed_requests = total_requests - successful_requests
        
        avg_response_time = np.mean(response_times) if response_times else 0
        p95_response_time = np.percentile(response_times, 95) if response_times else 0
        p99_response_time = np.percentile(response_times, 99) if response_times else 0
        
        time_window_hours = self._get_time_window_hours(time_range)
        requests_per_minute = (total_requests / (time_window_hours * 60)) if time_window_hours > 0 else 0
        
        error_rate = (failed_requests / total_requests) if total_requests > 0 else 0
        cache_hit_rate = await self._get_cache_hit_rate(session, start_time)
        
        metrics = PerformanceMetrics(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time_ms=avg_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            requests_per_minute=requests_per_minute,
            error_rate=error_rate,
            cache_hit_rate=cache_hit_rate
        )
        
        self._cache_data(cache_key, metrics)
        return metrics
    
    async def get_quality_metrics(
        self,
        session: AsyncSession,
        time_range: TimeRange = TimeRange.DAY
    ) -> QualityMetrics:
        """Get content quality metrics"""
        cache_key = f"quality_metrics_{time_range}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        start_time = self._get_start_time(time_range)
        
        # Get confidence scores
        confidence_scores = await self._get_confidence_scores(session, start_time)
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        # Get feedback data
        feedback_data = await self._get_feedback_data(session, start_time)
        
        ratings = [f['rating'] for f in feedback_data]
        avg_rating = np.mean(ratings) if ratings else 0
        
        total_feedback = len(feedback_data)
        positive_feedback = len([f for f in feedback_data if f['rating'] >= 4])
        positive_rate = (positive_feedback / total_feedback) if total_feedback > 0 else 0
        
        # Get improvement suggestions
        improvements = []
        for feedback in feedback_data:
            if feedback.get('improvements'):
                improvements.extend(feedback['improvements'])
        
        improvement_count = len(improvements)
        most_common_improvements = self._get_most_common_items(improvements, 5)
        
        metrics = QualityMetrics(
            average_confidence_score=avg_confidence,
            average_rating=avg_rating,
            total_feedback_count=total_feedback,
            positive_feedback_rate=positive_rate,
            improvement_suggestions_count=improvement_count,
            most_common_improvements=most_common_improvements
        )
        
        self._cache_data(cache_key, metrics)
        return metrics
    
    async def get_usage_metrics(
        self,
        session: AsyncSession,
        time_range: TimeRange = TimeRange.DAY
    ) -> UsageMetrics:
        """Get usage analytics"""
        cache_key = f"usage_metrics_{time_range}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        start_time = self._get_start_time(time_range)
        
        # Get usage data
        usage_data = await self._get_usage_data(session, start_time)
        
        total_variants = sum(record['variant_count'] for record in usage_data)
        avg_variants_per_request = (total_variants / len(usage_data)) if usage_data else 0
        
        # Analyze preferences
        tones = [record['tone'] for record in usage_data]
        styles = [record['style'] for record in usage_data]
        purposes = [record['purpose'] for record in usage_data]
        
        most_popular_tones = self._get_most_common_items(tones, 5)
        most_popular_styles = self._get_most_common_items(styles, 5)
        most_popular_purposes = self._get_most_common_items(purposes, 5)
        
        # Word count analysis
        word_counts = [record['avg_word_count'] for record in usage_data if record['avg_word_count']]
        avg_word_count = np.mean(word_counts) if word_counts else 0
        
        # CTA inclusion rate
        cta_included = len([r for r in usage_data if r.get('cta_included', False)])
        cta_rate = (cta_included / len(usage_data)) if usage_data else 0
        
        metrics = UsageMetrics(
            total_variants_generated=total_variants,
            average_variants_per_request=avg_variants_per_request,
            most_popular_tones=most_popular_tones,
            most_popular_styles=most_popular_styles,
            most_popular_purposes=most_popular_purposes,
            average_word_count=avg_word_count,
            cta_inclusion_rate=cta_rate
        )
        
        self._cache_data(cache_key, metrics)
        return metrics
    
    async def get_engagement_metrics(
        self,
        session: AsyncSession,
        time_range: TimeRange = TimeRange.DAY
    ) -> EngagementMetrics:
        """Get user engagement metrics"""
        cache_key = f"engagement_metrics_{time_range}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        start_time = self._get_start_time(time_range)
        
        # Get user data
        user_data = await self._get_user_data(session, start_time)
        
        unique_users = len(set(record['user_id'] for record in user_data))
        repeat_users = len([user for user, count in 
                           self._get_user_request_counts(user_data).items() if count > 1])
        
        # Session analysis
        session_durations = [record['session_duration'] for record in user_data 
                           if record.get('session_duration')]
        avg_session_duration = np.mean(session_durations) if session_durations else 0
        
        # Requests per user
        user_request_counts = self._get_user_request_counts(user_data)
        avg_requests_per_user = np.mean(list(user_request_counts.values())) if user_request_counts else 0
        
        # Feedback submission rate
        total_requests = len(user_data)
        feedback_submissions = len([r for r in user_data if r.get('feedback_submitted')])
        feedback_rate = (feedback_submissions / total_requests) if total_requests > 0 else 0
        
        # User retention (simplified calculation)
        retention_rate = (repeat_users / unique_users) if unique_users > 0 else 0
        
        metrics = EngagementMetrics(
            unique_users=unique_users,
            repeat_users=repeat_users,
            session_duration_avg=avg_session_duration,
            requests_per_user_avg=avg_requests_per_user,
            feedback_submission_rate=feedback_rate,
            user_retention_rate=retention_rate
        )
        
        self._cache_data(cache_key, metrics)
        return metrics
    
    async def get_trend_analysis(
        self,
        session: AsyncSession,
        metric_name: str,
        time_range: TimeRange = TimeRange.WEEK
    ) -> TrendData:
        """Get trend analysis for a specific metric"""
        cache_key = f"trend_{metric_name}_{time_range}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        # Get historical data points
        data_points = await self._get_historical_data(session, metric_name, time_range)
        
        if not data_points:
            return TrendData(
                time_period=time_range,
                metric_name=metric_name,
                values=[],
                timestamps=[],
                trend_direction="stable",
                trend_percentage=0.0
            )
        
        values = [point['value'] for point in data_points]
        timestamps = [point['timestamp'] for point in data_points]
        
        # Calculate trend
        if len(values) >= 2:
            first_half = values[:len(values)//2]
            second_half = values[len(values)//2:]
            
            first_avg = np.mean(first_half)
            second_avg = np.mean(second_half)
            
            if second_avg > first_avg * 1.05:  # 5% threshold
                trend_direction = "up"
                trend_percentage = ((second_avg - first_avg) / first_avg) * 100
            elif second_avg < first_avg * 0.95:
                trend_direction = "down"
                trend_percentage = ((first_avg - second_avg) / first_avg) * 100
            else:
                trend_direction = "stable"
                trend_percentage = 0.0
        else:
            trend_direction = "stable"
            trend_percentage = 0.0
        
        trend_data = TrendData(
            time_period=time_range,
            metric_name=metric_name,
            values=values,
            timestamps=timestamps,
            trend_direction=trend_direction,
            trend_percentage=trend_percentage
        )
        
        self._cache_data(cache_key, trend_data)
        return trend_data
    
    async def get_comprehensive_dashboard(
        self,
        session: AsyncSession,
        time_range: TimeRange = TimeRange.DAY
    ) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        # Run all analytics in parallel
        tasks = [
            self.get_performance_metrics(session, time_range),
            self.get_quality_metrics(session, time_range),
            self.get_usage_metrics(session, time_range),
            self.get_engagement_metrics(session, time_range),
        ]
        
        performance, quality, usage, engagement = await asyncio.gather(*tasks)
        
        # Get trend data for key metrics
        trend_tasks = [
            self.get_trend_analysis(session, "response_time", time_range),
            self.get_trend_analysis(session, "success_rate", time_range),
            self.get_trend_analysis(session, "user_engagement", time_range),
        ]
        
        response_trend, success_trend, engagement_trend = await asyncio.gather(*trend_tasks)
        
        return {
            "performance": asdict(performance),
            "quality": asdict(quality),
            "usage": asdict(usage),
            "engagement": asdict(engagement),
            "trends": {
                "response_time": asdict(response_trend),
                "success_rate": asdict(success_trend),
                "user_engagement": asdict(engagement_trend),
            },
            "generated_at": datetime.utcnow().isoformat(),
            "time_range": time_range
        }
    
    # Helper methods
    def _get_start_time(self, time_range: TimeRange) -> datetime:
        """Get start time for time range"""
        now = datetime.utcnow()
        if time_range == TimeRange.HOUR:
            return now - timedelta(hours=1)
        elif time_range == TimeRange.DAY:
            return now - timedelta(days=1)
        elif time_range == TimeRange.WEEK:
            return now - timedelta(weeks=1)
        elif time_range == TimeRange.MONTH:
            return now - timedelta(days=30)
        elif time_range == TimeRange.QUARTER:
            return now - timedelta(days=90)
        elif time_range == TimeRange.YEAR:
            return now - timedelta(days=365)
        else:
            return now - timedelta(days=1)
    
    def _get_time_window_hours(self, time_range: TimeRange) -> float:
        """Get time window in hours"""
        if time_range == TimeRange.HOUR:
            return 1.0
        elif time_range == TimeRange.DAY:
            return 24.0
        elif time_range == TimeRange.WEEK:
            return 168.0
        elif time_range == TimeRange.MONTH:
            return 720.0
        elif time_range == TimeRange.QUARTER:
            return 2160.0
        elif time_range == TimeRange.YEAR:
            return 8760.0
        else:
            return 24.0
    
    def _get_cached_data(self, key: str) -> Optional[Any]:
        """Get cached data"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.utcnow() - timestamp < timedelta(seconds=self.cache_ttl):
                return data
            else:
                del self.cache[key]
        return None
    
    def _cache_data(self, key: str, data: Any):
        """Cache data"""
        self.cache[key] = (data, datetime.utcnow())
    
    def _get_most_common_items(self, items: List[str], limit: int = 5) -> List[Tuple[str, int]]:
        """Get most common items with counts"""
        from collections import Counter
        counter = Counter(items)
        return counter.most_common(limit)
    
    def _get_user_request_counts(self, user_data: List[Dict]) -> Dict[str, int]:
        """Get request counts per user"""
        from collections import Counter
        user_ids = [record['user_id'] for record in user_data]
        return dict(Counter(user_ids))
    
    # Database query methods (simplified - would need actual implementation)
    async def _get_request_stats(self, session: AsyncSession, start_time: datetime) -> Dict[str, int]:
        """Get request statistics"""
        # This would query the actual database
        # For now, return mock data
        return {"total": 1000, "successful": 950}
    
    async def _get_response_times(self, session: AsyncSession, start_time: datetime) -> List[float]:
        """Get response times"""
        # This would query the actual database
        # For now, return mock data
        return [100.0, 150.0, 200.0, 120.0, 180.0]
    
    async def _get_cache_hit_rate(self, session: AsyncSession, start_time: datetime) -> float:
        """Get cache hit rate"""
        # This would query the actual database
        return 0.75
    
    async def _get_confidence_scores(self, session: AsyncSession, start_time: datetime) -> List[float]:
        """Get confidence scores"""
        # This would query the actual database
        return [0.8, 0.9, 0.7, 0.85, 0.75]
    
    async def _get_feedback_data(self, session: AsyncSession, start_time: datetime) -> List[Dict]:
        """Get feedback data"""
        # This would query the actual database
        return [
            {"rating": 4, "improvements": ["More examples"]},
            {"rating": 5, "improvements": []},
            {"rating": 3, "improvements": ["Better tone", "Shorter content"]},
        ]
    
    async def _get_usage_data(self, session: AsyncSession, start_time: datetime) -> List[Dict]:
        """Get usage data"""
        # This would query the actual database
        return [
            {"tone": "professional", "style": "direct_response", "purpose": "sales", 
             "variant_count": 3, "avg_word_count": 500, "cta_included": True},
        ]
    
    async def _get_user_data(self, session: AsyncSession, start_time: datetime) -> List[Dict]:
        """Get user data"""
        # This would query the actual database
        return [
            {"user_id": "user1", "session_duration": 300, "feedback_submitted": True},
            {"user_id": "user2", "session_duration": 450, "feedback_submitted": False},
        ]
    
    async def _get_historical_data(
        self, 
        session: AsyncSession, 
        metric_name: str, 
        time_range: TimeRange
    ) -> List[Dict]:
        """Get historical data for trend analysis"""
        # This would query the actual database
        return [
            {"value": 100.0, "timestamp": datetime.utcnow() - timedelta(hours=6)},
            {"value": 120.0, "timestamp": datetime.utcnow() - timedelta(hours=4)},
            {"value": 110.0, "timestamp": datetime.utcnow() - timedelta(hours=2)},
        ]


# Global analytics engine instance
analytics_engine = AnalyticsEngine()






























