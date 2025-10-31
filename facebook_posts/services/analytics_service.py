"""
Advanced Analytics Service for Facebook Posts API
Real-time analytics, insights, and performance tracking
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import structlog

from ..core.config import get_settings
from ..core.models import FacebookPost, PostStatus, ContentType, AudienceType
from ..infrastructure.cache import get_cache_manager
from ..infrastructure.database import get_db_manager, PostRepository, AnalyticsRepository
from ..infrastructure.monitoring import get_monitor, timed

logger = structlog.get_logger(__name__)


@dataclass
class PostMetrics:
    """Post performance metrics"""
    post_id: str
    views: int = 0
    likes: int = 0
    shares: int = 0
    comments: int = 0
    clicks: int = 0
    engagement_rate: float = 0.0
    reach: int = 0
    impressions: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AudienceInsight:
    """Audience insight data"""
    audience_type: str
    engagement_rate: float
    preferred_content_types: List[str]
    optimal_posting_times: List[str]
    top_performing_topics: List[str]
    demographics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentInsight:
    """Content performance insight"""
    content_type: str
    average_engagement: float
    best_performing_length: int
    optimal_hashtag_count: int
    sentiment_impact: float
    creativity_score_impact: float


@dataclass
class PerformanceReport:
    """Performance report data"""
    period_start: datetime
    period_end: datetime
    total_posts: int
    total_engagement: int
    average_engagement_rate: float
    top_performing_posts: List[Dict[str, Any]]
    audience_insights: List[AudienceInsight]
    content_insights: List[ContentInsight]
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.now)


class RealTimeAnalytics:
    """Real-time analytics tracking"""
    
    def __init__(self):
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.engagement_tracker: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.performance_cache: Dict[str, Any] = {}
        self.cache_manager = get_cache_manager()
        self.monitor = get_monitor()
    
    async def track_post_engagement(self, post_id: str, engagement_type: str, value: int = 1):
        """Track post engagement in real-time"""
        try:
            # Update engagement tracker
            self.engagement_tracker[post_id][engagement_type] += value
            
            # Add to metrics buffer
            metric_data = {
                "post_id": post_id,
                "engagement_type": engagement_type,
                "value": value,
                "timestamp": datetime.now()
            }
            self.metrics_buffer[post_id].append(metric_data)
            
            # Update cache
            cache_key = f"post_engagement:{post_id}"
            await self.cache_manager.cache.set(cache_key, dict(self.engagement_tracker[post_id]), ttl=3600)
            
            # Record metrics
            self.monitor.metrics_collector.increment_counter(
                "post_engagement_total",
                labels={"post_id": post_id, "engagement_type": engagement_type}
            )
            
            logger.debug("Post engagement tracked", post_id=post_id, engagement_type=engagement_type, value=value)
            
        except Exception as e:
            logger.error("Failed to track post engagement", post_id=post_id, error=str(e))
    
    async def get_post_metrics(self, post_id: str) -> Optional[PostMetrics]:
        """Get real-time post metrics"""
        try:
            # Check cache first
            cache_key = f"post_metrics:{post_id}"
            cached_metrics = await self.cache_manager.cache.get(cache_key)
            
            if cached_metrics:
                return PostMetrics(**cached_metrics)
            
            # Get from engagement tracker
            engagement_data = self.engagement_tracker.get(post_id, {})
            
            if not engagement_data:
                return None
            
            # Calculate engagement rate
            total_engagement = sum(engagement_data.values())
            views = engagement_data.get("views", 1)  # Avoid division by zero
            engagement_rate = total_engagement / views if views > 0 else 0.0
            
            metrics = PostMetrics(
                post_id=post_id,
                views=engagement_data.get("views", 0),
                likes=engagement_data.get("likes", 0),
                shares=engagement_data.get("shares", 0),
                comments=engagement_data.get("comments", 0),
                clicks=engagement_data.get("clicks", 0),
                engagement_rate=engagement_rate,
                reach=engagement_data.get("reach", 0),
                impressions=engagement_data.get("impressions", 0)
            )
            
            # Cache metrics
            await self.cache_manager.cache.set(cache_key, metrics.__dict__, ttl=300)
            
            return metrics
            
        except Exception as e:
            logger.error("Failed to get post metrics", post_id=post_id, error=str(e))
            return None
    
    async def get_trending_posts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get trending posts based on recent engagement"""
        try:
            trending_posts = []
            
            for post_id, engagement_data in self.engagement_tracker.items():
                total_engagement = sum(engagement_data.values())
                if total_engagement > 0:
                    trending_posts.append({
                        "post_id": post_id,
                        "total_engagement": total_engagement,
                        "engagement_rate": total_engagement / max(engagement_data.get("views", 1), 1),
                        "engagement_breakdown": dict(engagement_data)
                    })
            
            # Sort by total engagement
            trending_posts.sort(key=lambda x: x["total_engagement"], reverse=True)
            
            return trending_posts[:limit]
            
        except Exception as e:
            logger.error("Failed to get trending posts", error=str(e))
            return []


class AudienceAnalytics:
    """Audience analytics and insights"""
    
    def __init__(self):
        self.audience_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(list))
        self.cache_manager = get_cache_manager()
        self.db_manager = get_db_manager()
    
    async def analyze_audience_behavior(self, audience_type: str, days: int = 30) -> AudienceInsight:
        """Analyze audience behavior patterns"""
        try:
            # Check cache first
            cache_key = f"audience_analysis:{audience_type}:{days}"
            cached_insight = await self.cache_manager.cache.get(cache_key)
            
            if cached_insight:
                return AudienceInsight(**cached_insight)
            
            # Get posts for this audience type
            post_repo = PostRepository(self.db_manager)
            posts = await post_repo.list_posts(
                limit=1000,
                filters={"audience_type": audience_type}
            )
            
            if not posts:
                return self._default_audience_insight(audience_type)
            
            # Analyze content types
            content_type_performance = defaultdict(list)
            topic_performance = defaultdict(list)
            posting_times = []
            
            for post in posts:
                content_type = post.get("content_type", "unknown")
                content_type_performance[content_type].append(post.get("engagement_score", 0))
                
                # Extract topics from metadata
                metadata = post.get("metadata", {})
                topic = metadata.get("topic", "unknown")
                topic_performance[topic].append(post.get("engagement_score", 0))
                
                # Extract posting time
                created_at = post.get("created_at")
                if created_at:
                    posting_times.append(created_at.hour)
            
            # Calculate insights
            preferred_content_types = self._get_top_performing_content_types(content_type_performance)
            top_performing_topics = self._get_top_performing_topics(topic_performance)
            optimal_posting_times = self._get_optimal_posting_times(posting_times)
            
            # Calculate average engagement rate
            engagement_rates = [post.get("engagement_score", 0) for post in posts]
            average_engagement = sum(engagement_rates) / len(engagement_rates) if engagement_rates else 0.0
            
            insight = AudienceInsight(
                audience_type=audience_type,
                engagement_rate=average_engagement,
                preferred_content_types=preferred_content_types,
                optimal_posting_times=optimal_posting_times,
                top_performing_topics=top_performing_topics,
                demographics={
                    "total_posts_analyzed": len(posts),
                    "analysis_period_days": days,
                    "average_post_length": sum(len(post.get("content", "")) for post in posts) / len(posts)
                }
            )
            
            # Cache insight
            await self.cache_manager.cache.set(cache_key, insight.__dict__, ttl=3600)
            
            return insight
            
        except Exception as e:
            logger.error("Failed to analyze audience behavior", audience_type=audience_type, error=str(e))
            return self._default_audience_insight(audience_type)
    
    def _get_top_performing_content_types(self, content_type_performance: Dict[str, List[float]]) -> List[str]:
        """Get top performing content types"""
        avg_performance = {
            content_type: sum(scores) / len(scores) if scores else 0.0
            for content_type, scores in content_type_performance.items()
        }
        
        return sorted(avg_performance.items(), key=lambda x: x[1], reverse=True)[:3]
    
    def _get_top_performing_topics(self, topic_performance: Dict[str, List[float]]) -> List[str]:
        """Get top performing topics"""
        avg_performance = {
            topic: sum(scores) / len(scores) if scores else 0.0
            for topic, scores in topic_performance.items()
        }
        
        return sorted(avg_performance.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _get_optimal_posting_times(self, posting_times: List[int]) -> List[str]:
        """Get optimal posting times"""
        if not posting_times:
            return ["9:00 AM", "1:00 PM", "7:00 PM"]
        
        # Count posting times
        time_counts = defaultdict(int)
        for hour in posting_times:
            time_counts[hour] += 1
        
        # Get top 3 hours
        top_hours = sorted(time_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return [f"{hour}:00" for hour, _ in top_hours]
    
    def _default_audience_insight(self, audience_type: str) -> AudienceInsight:
        """Default audience insight when no data is available"""
        return AudienceInsight(
            audience_type=audience_type,
            engagement_rate=0.0,
            preferred_content_types=["educational", "entertainment"],
            optimal_posting_times=["9:00 AM", "1:00 PM", "7:00 PM"],
            top_performing_topics=["technology", "business", "lifestyle"]
        )


class ContentAnalytics:
    """Content performance analytics"""
    
    def __init__(self):
        self.content_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(list))
        self.cache_manager = get_cache_manager()
        self.db_manager = get_db_manager()
    
    async def analyze_content_performance(self, content_type: str, days: int = 30) -> ContentInsight:
        """Analyze content performance patterns"""
        try:
            # Check cache first
            cache_key = f"content_analysis:{content_type}:{days}"
            cached_insight = await self.cache_manager.cache.get(cache_key)
            
            if cached_insight:
                return ContentInsight(**cached_insight)
            
            # Get posts for this content type
            post_repo = PostRepository(self.db_manager)
            posts = await post_repo.list_posts(
                limit=1000,
                filters={"content_type": content_type}
            )
            
            if not posts:
                return self._default_content_insight(content_type)
            
            # Analyze content characteristics
            lengths = []
            hashtag_counts = []
            sentiment_scores = []
            creativity_scores = []
            engagement_rates = []
            
            for post in posts:
                content = post.get("content", "")
                lengths.append(len(content))
                
                # Count hashtags
                hashtag_count = content.count("#")
                hashtag_counts.append(hashtag_count)
                
                # Get sentiment and creativity scores
                sentiment_scores.append(post.get("sentiment_score", 0.0))
                creativity_scores.append(post.get("creativity_score", 0.0))
                engagement_rates.append(post.get("engagement_score", 0.0))
            
            # Calculate insights
            best_length = self._calculate_optimal_length(lengths, engagement_rates)
            optimal_hashtag_count = self._calculate_optimal_hashtag_count(hashtag_counts, engagement_rates)
            sentiment_impact = self._calculate_sentiment_impact(sentiment_scores, engagement_rates)
            creativity_impact = self._calculate_creativity_impact(creativity_scores, engagement_rates)
            
            insight = ContentInsight(
                content_type=content_type,
                average_engagement=sum(engagement_rates) / len(engagement_rates) if engagement_rates else 0.0,
                best_performing_length=best_length,
                optimal_hashtag_count=optimal_hashtag_count,
                sentiment_impact=sentiment_impact,
                creativity_score_impact=creativity_impact
            )
            
            # Cache insight
            await self.cache_manager.cache.set(cache_key, insight.__dict__, ttl=3600)
            
            return insight
            
        except Exception as e:
            logger.error("Failed to analyze content performance", content_type=content_type, error=str(e))
            return self._default_content_insight(content_type)
    
    def _calculate_optimal_length(self, lengths: List[int], engagement_rates: List[float]) -> int:
        """Calculate optimal content length"""
        if not lengths or not engagement_rates:
            return 150
        
        # Group by length ranges
        length_ranges = {
            "short": (0, 100),
            "medium": (100, 200),
            "long": (200, 300),
            "very_long": (300, 1000)
        }
        
        range_performance = {}
        for range_name, (min_len, max_len) in length_ranges.items():
            range_engagements = [
                engagement for length, engagement in zip(lengths, engagement_rates)
                if min_len <= length < max_len
            ]
            if range_engagements:
                range_performance[range_name] = sum(range_engagements) / len(range_engagements)
        
        # Return middle of best performing range
        if range_performance:
            best_range = max(range_performance.items(), key=lambda x: x[1])[0]
            min_len, max_len = length_ranges[best_range]
            return (min_len + max_len) // 2
        
        return 150
    
    def _calculate_optimal_hashtag_count(self, hashtag_counts: List[int], engagement_rates: List[float]) -> int:
        """Calculate optimal hashtag count"""
        if not hashtag_counts or not engagement_rates:
            return 3
        
        # Group by hashtag count ranges
        hashtag_ranges = {
            "none": (0, 1),
            "few": (1, 3),
            "moderate": (3, 5),
            "many": (5, 10),
            "too_many": (10, 100)
        }
        
        range_performance = {}
        for range_name, (min_count, max_count) in hashtag_ranges.items():
            range_engagements = [
                engagement for count, engagement in zip(hashtag_counts, engagement_rates)
                if min_count <= count < max_count
            ]
            if range_engagements:
                range_performance[range_name] = sum(range_engagements) / len(range_engagements)
        
        # Return middle of best performing range
        if range_performance:
            best_range = max(range_performance.items(), key=lambda x: x[1])[0]
            min_count, max_count = hashtag_ranges[best_range]
            return (min_count + max_count) // 2
        
        return 3
    
    def _calculate_sentiment_impact(self, sentiment_scores: List[float], engagement_rates: List[float]) -> float:
        """Calculate sentiment impact on engagement"""
        if not sentiment_scores or not engagement_rates:
            return 0.0
        
        # Calculate correlation between sentiment and engagement
        n = len(sentiment_scores)
        if n < 2:
            return 0.0
        
        # Simple correlation calculation
        mean_sentiment = sum(sentiment_scores) / n
        mean_engagement = sum(engagement_rates) / n
        
        numerator = sum((s - mean_sentiment) * (e - mean_engagement) for s, e in zip(sentiment_scores, engagement_rates))
        denominator = (sum((s - mean_sentiment) ** 2 for s in sentiment_scores) * 
                      sum((e - mean_engagement) ** 2 for e in engagement_rates)) ** 0.5
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def _calculate_creativity_impact(self, creativity_scores: List[float], engagement_rates: List[float]) -> float:
        """Calculate creativity impact on engagement"""
        return self._calculate_sentiment_impact(creativity_scores, engagement_rates)
    
    def _default_content_insight(self, content_type: str) -> ContentInsight:
        """Default content insight when no data is available"""
        return ContentInsight(
            content_type=content_type,
            average_engagement=0.0,
            best_performing_length=150,
            optimal_hashtag_count=3,
            sentiment_impact=0.0,
            creativity_score_impact=0.0
        )


class PerformanceReporter:
    """Performance reporting and insights"""
    
    def __init__(self):
        self.cache_manager = get_cache_manager()
        self.db_manager = get_db_manager()
        self.audience_analytics = AudienceAnalytics()
        self.content_analytics = ContentAnalytics()
    
    @timed("performance_report_generation")
    async def generate_performance_report(self, days: int = 30) -> PerformanceReport:
        """Generate comprehensive performance report"""
        try:
            # Check cache first
            cache_key = f"performance_report:{days}"
            cached_report = await self.cache_manager.cache.get(cache_key)
            
            if cached_report:
                return PerformanceReport(**cached_report)
            
            # Calculate period
            period_end = datetime.now()
            period_start = period_end - timedelta(days=days)
            
            # Get posts from the period
            post_repo = PostRepository(self.db_manager)
            posts = await post_repo.list_posts(limit=10000)  # Get all posts
            
            # Filter posts by period
            period_posts = [
                post for post in posts
                if post.get("created_at") and period_start <= post["created_at"] <= period_end
            ]
            
            if not period_posts:
                return self._default_performance_report(period_start, period_end)
            
            # Calculate metrics
            total_posts = len(period_posts)
            total_engagement = sum(post.get("engagement_score", 0) for post in period_posts)
            average_engagement_rate = total_engagement / total_posts if total_posts > 0 else 0.0
            
            # Get top performing posts
            top_posts = sorted(period_posts, key=lambda x: x.get("engagement_score", 0), reverse=True)[:10]
            
            # Get audience insights
            audience_types = set(post.get("audience_type") for post in period_posts)
            audience_insights = []
            for audience_type in audience_types:
                if audience_type:
                    insight = await self.audience_analytics.analyze_audience_behavior(audience_type, days)
                    audience_insights.append(insight)
            
            # Get content insights
            content_types = set(post.get("content_type") for post in period_posts)
            content_insights = []
            for content_type in content_types:
                if content_type:
                    insight = await self.content_analytics.analyze_content_performance(content_type, days)
                    content_insights.append(insight)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                total_posts, average_engagement_rate, audience_insights, content_insights
            )
            
            report = PerformanceReport(
                period_start=period_start,
                period_end=period_end,
                total_posts=total_posts,
                total_engagement=total_engagement,
                average_engagement_rate=average_engagement_rate,
                top_performing_posts=top_posts,
                audience_insights=audience_insights,
                content_insights=content_insights,
                recommendations=recommendations
            )
            
            # Cache report
            await self.cache_manager.cache.set(cache_key, report.__dict__, ttl=1800)
            
            return report
            
        except Exception as e:
            logger.error("Failed to generate performance report", error=str(e))
            return self._default_performance_report(
                datetime.now() - timedelta(days=days),
                datetime.now()
            )
    
    def _generate_recommendations(
        self,
        total_posts: int,
        average_engagement: float,
        audience_insights: List[AudienceInsight],
        content_insights: List[ContentInsight]
    ) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # Posting frequency recommendations
        if total_posts < 10:
            recommendations.append("Consider increasing posting frequency to improve reach")
        elif total_posts > 100:
            recommendations.append("Consider reducing posting frequency to focus on quality")
        
        # Engagement recommendations
        if average_engagement < 0.3:
            recommendations.append("Focus on improving content engagement through better calls-to-action")
        elif average_engagement > 0.8:
            recommendations.append("Excellent engagement! Consider scaling successful content strategies")
        
        # Audience-specific recommendations
        for insight in audience_insights:
            if insight.engagement_rate < 0.4:
                recommendations.append(f"Improve engagement for {insight.audience_type} audience")
        
        # Content-specific recommendations
        for insight in content_insights:
            if insight.average_engagement < 0.4:
                recommendations.append(f"Optimize {insight.content_type} content for better performance")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _default_performance_report(self, period_start: datetime, period_end: datetime) -> PerformanceReport:
        """Default performance report when no data is available"""
        return PerformanceReport(
            period_start=period_start,
            period_end=period_end,
            total_posts=0,
            total_engagement=0,
            average_engagement_rate=0.0,
            top_performing_posts=[],
            audience_insights=[],
            content_insights=[],
            recommendations=["Start creating content to generate performance insights"]
        )


class AnalyticsService:
    """Main analytics service orchestrator"""
    
    def __init__(self):
        self.real_time_analytics = RealTimeAnalytics()
        self.audience_analytics = AudienceAnalytics()
        self.content_analytics = ContentAnalytics()
        self.performance_reporter = PerformanceReporter()
        self.cache_manager = get_cache_manager()
        self.monitor = get_monitor()
    
    async def track_engagement(self, post_id: str, engagement_type: str, value: int = 1):
        """Track post engagement"""
        await self.real_time_analytics.track_post_engagement(post_id, engagement_type, value)
    
    async def get_post_analytics(self, post_id: str) -> Optional[PostMetrics]:
        """Get post analytics"""
        return await self.real_time_analytics.get_post_metrics(post_id)
    
    async def get_trending_posts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get trending posts"""
        return await self.real_time_analytics.get_trending_posts(limit)
    
    async def get_audience_insights(self, audience_type: str, days: int = 30) -> AudienceInsight:
        """Get audience insights"""
        return await self.audience_analytics.analyze_audience_behavior(audience_type, days)
    
    async def get_content_insights(self, content_type: str, days: int = 30) -> ContentInsight:
        """Get content insights"""
        return await self.content_analytics.analyze_content_performance(content_type, days)
    
    async def generate_performance_report(self, days: int = 30) -> PerformanceReport:
        """Generate performance report"""
        return await self.performance_reporter.generate_performance_report(days)


# Global analytics service instance
_analytics_service: Optional[AnalyticsService] = None


def get_analytics_service() -> AnalyticsService:
    """Get global analytics service instance"""
    global _analytics_service
    
    if _analytics_service is None:
        _analytics_service = AnalyticsService()
    
    return _analytics_service


# Export all classes and functions
__all__ = [
    # Data classes
    'PostMetrics',
    'AudienceInsight',
    'ContentInsight',
    'PerformanceReport',
    
    # Analytics services
    'RealTimeAnalytics',
    'AudienceAnalytics',
    'ContentAnalytics',
    'PerformanceReporter',
    'AnalyticsService',
    
    # Utility functions
    'get_analytics_service',
]






























