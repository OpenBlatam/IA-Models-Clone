from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from uuid import UUID, uuid4
import json
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Analytics domain entity for LinkedIn Posts system.
"""



@dataclass
class TimeSeriesData:
    """Time series data point."""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "metadata": self.metadata
        }


@dataclass
class EngagementBreakdown:
    """Detailed engagement breakdown."""
    likes: int = 0
    comments: int = 0
    shares: int = 0
    clicks: int = 0
    impressions: int = 0
    reach: int = 0
    
    # Advanced metrics
    profile_visits: int = 0
    follows: int = 0
    reactions: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "likes": self.likes,
            "comments": self.comments,
            "shares": self.shares,
            "clicks": self.clicks,
            "impressions": self.impressions,
            "reach": self.reach,
            "profile_visits": self.profile_visits,
            "follows": self.follows,
            "reactions": self.reactions
        }


@dataclass
class AudienceInsights:
    """Audience insights data."""
    demographics: Dict[str, float] = field(default_factory=dict)
    industries: Dict[str, int] = field(default_factory=dict)
    locations: Dict[str, int] = field(default_factory=dict)
    job_titles: Dict[str, int] = field(default_factory=dict)
    company_sizes: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "demographics": self.demographics,
            "industries": self.industries,
            "locations": self.locations,
            "job_titles": self.job_titles,
            "company_sizes": self.company_sizes
        }


@dataclass
class ContentPerformance:
    """Content performance metrics."""
    post_id: UUID
    title: str
    content_preview: str
    
    # Engagement metrics
    engagement: EngagementBreakdown = field(default_factory=EngagementBreakdown)
    
    # Performance scores
    engagement_rate: float = 0.0
    reach_rate: float = 0.0
    click_through_rate: float = 0.0
    
    # Timing
    published_at: Optional[datetime] = None
    peak_engagement_time: Optional[datetime] = None
    
    # Content analysis
    content_score: float = 0.0
    hashtag_performance: Dict[str, int] = field(default_factory=dict)
    mention_performance: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "post_id": str(self.post_id),
            "title": self.title,
            "content_preview": self.content_preview,
            "engagement": self.engagement.to_dict(),
            "engagement_rate": self.engagement_rate,
            "reach_rate": self.reach_rate,
            "click_through_rate": self.click_through_rate,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "peak_engagement_time": self.peak_engagement_time.isoformat() if self.peak_engagement_time else None,
            "content_score": self.content_score,
            "hashtag_performance": self.hashtag_performance,
            "mention_performance": self.mention_performance
        }


@dataclass
class Analytics:
    """
    Analytics domain entity for comprehensive insights.
    
    Features:
    - Time series tracking
    - Audience insights
    - Content performance
    - Trend analysis
    - Predictive analytics
    """
    
    # Core fields
    id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    
    # Time period
    start_date: datetime = field(default_factory=datetime.utcnow)
    end_date: datetime = field(default_factory=datetime.utcnow)
    
    # Overall metrics
    total_posts: int = 0
    total_engagement: int = 0
    total_impressions: int = 0
    total_reach: int = 0
    
    # Performance metrics
    average_engagement_rate: float = 0.0
    average_reach_rate: float = 0.0
    average_click_through_rate: float = 0.0
    
    # Growth metrics
    follower_growth: int = 0
    engagement_growth: float = 0.0
    reach_growth: float = 0.0
    
    # Time series data
    engagement_timeline: List[TimeSeriesData] = field(default_factory=list)
    reach_timeline: List[TimeSeriesData] = field(default_factory=list)
    follower_timeline: List[TimeSeriesData] = field(default_factory=list)
    
    # Content performance
    top_performing_posts: List[ContentPerformance] = field(default_factory=list)
    worst_performing_posts: List[ContentPerformance] = field(default_factory=list)
    
    # Audience insights
    audience_insights: AudienceInsights = field(default_factory=AudienceInsights)
    
    # Trend analysis
    trending_hashtags: List[Dict[str, Any]] = field(default_factory=list)
    trending_topics: List[Dict[str, Any]] = field(default_factory=list)
    best_posting_times: List[Dict[str, Any]] = field(default_factory=list)
    
    # Predictive analytics
    predicted_engagement: float = 0.0
    predicted_reach: float = 0.0
    recommended_posting_times: List[datetime] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> Any:
        """Post-initialization processing."""
        if isinstance(self.audience_insights, dict):
            self.audience_insights = AudienceInsights(**self.audience_insights)
        
        # Convert timeline data
        for timeline_name in ['engagement_timeline', 'reach_timeline', 'follower_timeline']:
            timeline = getattr(self, timeline_name)
            if isinstance(timeline, list):
                converted_timeline = []
                for item in timeline:
                    if isinstance(item, dict):
                        item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                        converted_timeline.append(TimeSeriesData(**item))
                    else:
                        converted_timeline.append(item)
                setattr(self, timeline_name, converted_timeline)
        
        # Convert content performance
        for content_list_name in ['top_performing_posts', 'worst_performing_posts']:
            content_list = getattr(self, content_list_name)
            if isinstance(content_list, list):
                converted_list = []
                for item in content_list:
                    if isinstance(item, dict):
                        if 'post_id' in item and isinstance(item['post_id'], str):
                            item['post_id'] = UUID(item['post_id'])
                        if 'published_at' in item and item['published_at']:
                            item['published_at'] = datetime.fromisoformat(item['published_at'])
                        if 'peak_engagement_time' in item and item['peak_engagement_time']:
                            item['peak_engagement_time'] = datetime.fromisoformat(item['peak_engagement_time'])
                        converted_list.append(ContentPerformance(**item))
                    else:
                        converted_list.append(item)
                setattr(self, content_list_name, converted_list)
    
    @property
    def period_days(self) -> int:
        """Get the number of days in the analytics period."""
        return (self.end_date - self.start_date).days
    
    @property
    def engagement_per_post(self) -> float:
        """Calculate average engagement per post."""
        return self.total_engagement / self.total_posts if self.total_posts > 0 else 0.0
    
    @property
    def reach_per_post(self) -> float:
        """Calculate average reach per post."""
        return self.total_reach / self.total_posts if self.total_posts > 0 else 0.0
    
    def add_timeline_data(self, timeline_name: str, timestamp: datetime, value: float, metadata: Dict[str, Any] = None) -> None:
        """Add data point to timeline."""
        if metadata is None:
            metadata = {}
        
        timeline = getattr(self, timeline_name)
        timeline.append(TimeSeriesData(timestamp, value, metadata))
        setattr(self, timeline_name, timeline)
    
    def get_timeline_data(self, timeline_name: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[TimeSeriesData]:
        """Get filtered timeline data."""
        timeline = getattr(self, timeline_name)
        
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date
        
        return [
            data for data in timeline
            if start_date <= data.timestamp <= end_date
        ]
    
    def update_performance_metrics(self, **metrics) -> None:
        """Update performance metrics."""
        for key, value in metrics.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def add_top_performing_post(self, post: ContentPerformance) -> None:
        """Add a top performing post."""
        self.top_performing_posts.append(post)
        # Keep only top 10
        self.top_performing_posts = sorted(
            self.top_performing_posts,
            key=lambda x: x.engagement_rate,
            reverse=True
        )[:10]
    
    def add_worst_performing_post(self, post: ContentPerformance) -> None:
        """Add a worst performing post."""
        self.worst_performing_posts.append(post)
        # Keep only bottom 10
        self.worst_performing_posts = sorted(
            self.worst_performing_posts,
            key=lambda x: x.engagement_rate
        )[:10]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_posts": self.total_posts,
            "total_engagement": self.total_engagement,
            "total_impressions": self.total_impressions,
            "total_reach": self.total_reach,
            "average_engagement_rate": self.average_engagement_rate,
            "average_reach_rate": self.average_reach_rate,
            "average_click_through_rate": self.average_click_through_rate,
            "follower_growth": self.follower_growth,
            "engagement_growth": self.engagement_growth,
            "reach_growth": self.reach_growth,
            "engagement_timeline": [data.to_dict() for data in self.engagement_timeline],
            "reach_timeline": [data.to_dict() for data in self.reach_timeline],
            "follower_timeline": [data.to_dict() for data in self.follower_timeline],
            "top_performing_posts": [post.to_dict() for post in self.top_performing_posts],
            "worst_performing_posts": [post.to_dict() for post in self.worst_performing_posts],
            "audience_insights": self.audience_insights.to_dict(),
            "trending_hashtags": self.trending_hashtags,
            "trending_topics": self.trending_topics,
            "best_posting_times": self.best_posting_times,
            "predicted_engagement": self.predicted_engagement,
            "predicted_reach": self.predicted_reach,
            "recommended_posting_times": [dt.isoformat() for dt in self.recommended_posting_times],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Analytics':
        """Create from dictionary."""
        # Convert string IDs to UUIDs
        if 'id' in data and isinstance(data['id'], str):
            data['id'] = UUID(data['id'])
        if 'user_id' in data and isinstance(data['user_id'], str):
            data['user_id'] = UUID(data['user_id'])
        
        # Convert string dates to datetime
        for date_field in ['start_date', 'end_date', 'created_at', 'updated_at']:
            if date_field in data and data[date_field]:
                if isinstance(data[date_field], str):
                    data[date_field] = datetime.fromisoformat(data[date_field])
        
        # Convert recommended posting times
        if 'recommended_posting_times' in data and data['recommended_posting_times']:
            data['recommended_posting_times'] = [
                datetime.fromisoformat(dt) if isinstance(dt, str) else dt
                for dt in data['recommended_posting_times']
            ]
        
        return cls(**data)
    
    def __str__(self) -> str:
        """String representation."""
        return f"Analytics(id={self.id}, user_id={self.user_id}, period={self.period_days} days)"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Analytics(id={self.id}, user_id={self.user_id}, posts={self.total_posts}, engagement={self.total_engagement})" 