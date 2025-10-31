from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Post Metadata Value Object
=========================

Value object for post metadata and analytics information.
"""



@dataclass
class PostMetadata:
    """
    Post metadata value object for storing analytics and tracking information.
    
    This value object encapsulates metadata related to post performance,
    optimization, and tracking.
    """
    
    # Scheduling
    scheduled_at: Optional[datetime] = None
    
    # NLP and AI
    nlp_optimized: bool = False
    nlp_processing_time: Optional[float] = None
    ai_generated: bool = False
    ai_model_used: Optional[str] = None
    
    # Analytics
    views: int = 0
    unique_views: int = 0
    impressions: int = 0
    reach: int = 0
    
    # Performance
    click_through_rate: float = 0.0
    engagement_rate: float = 0.0
    viral_coefficient: float = 0.0
    
    # SEO and Discovery
    keywords: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    
    # Timing
    best_time_to_post: Optional[datetime] = None
    time_zone: str = "UTC"
    
    # Platform specific
    linkedin_post_id: Optional[str] = None
    external_url: Optional[str] = None
    
    # Custom fields
    custom_data: Dict[str, Any] = field(default_factory=dict)
    
    def update_analytics(self, **kwargs) -> None:
        """Update analytics data."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def increment_views(self, count: int = 1) -> None:
        """Increment view count."""
        self.views += count
    
    def increment_unique_views(self, count: int = 1) -> None:
        """Increment unique view count."""
        self.unique_views += count
    
    def increment_impressions(self, count: int = 1) -> None:
        """Increment impression count."""
        self.impressions += count
    
    def increment_reach(self, count: int = 1) -> None:
        """Increment reach count."""
        self.reach += count
    
    def add_keyword(self, keyword: str) -> None:
        """Add a keyword to the post."""
        if keyword and keyword not in self.keywords:
            self.keywords.append(keyword)
    
    def add_hashtag(self, hashtag: str) -> None:
        """Add a hashtag to the post."""
        if hashtag and hashtag not in self.hashtags:
            self.hashtags.append(hashtag)
    
    def add_mention(self, mention: str) -> None:
        """Add a mention to the post."""
        if mention and mention not in self.mentions:
            self.mentions.append(mention)
    
    def add_link(self, link: str) -> None:
        """Add a link to the post."""
        if link and link not in self.links:
            self.links.append(link)
    
    def set_nlp_optimization(self, processing_time: float, model: Optional[str] = None) -> None:
        """Set NLP optimization metadata."""
        self.nlp_optimized = True
        self.nlp_processing_time = processing_time
        if model:
            self.ai_model_used = model
    
    def set_ai_generation(self, model: str) -> None:
        """Set AI generation metadata."""
        self.ai_generated = True
        self.ai_model_used = model
    
    def calculate_engagement_rate(self, likes: int, comments: int, shares: int) -> float:
        """Calculate engagement rate."""
        total_engagement = likes + comments + shares
        if self.impressions > 0:
            self.engagement_rate = (total_engagement / self.impressions) * 100
        return self.engagement_rate
    
    def calculate_click_through_rate(self, clicks: int) -> float:
        """Calculate click-through rate."""
        if self.impressions > 0:
            self.click_through_rate = (clicks / self.impressions) * 100
        return self.click_through_rate
    
    def calculate_viral_coefficient(self, shares: int, followers: int) -> float:
        """Calculate viral coefficient."""
        if followers > 0:
            self.viral_coefficient = shares / followers
        return self.viral_coefficient
    
    def get_performance_score(self) -> float:
        """Calculate overall performance score."""
        score = 0.0
        
        # Engagement rate weight: 40%
        score += self.engagement_rate * 0.4
        
        # Click-through rate weight: 30%
        score += self.click_through_rate * 0.3
        
        # Viral coefficient weight: 20%
        score += min(self.viral_coefficient * 100, 100) * 0.2
        
        # Reach weight: 10%
        reach_score = min(self.reach / 1000, 100)  # Normalize reach
        score += reach_score * 0.1
        
        return min(score, 100)  # Cap at 100
    
    def get_optimization_status(self) -> str:
        """Get optimization status."""
        if self.nlp_optimized and self.ai_generated:
            return "fully_optimized"
        elif self.nlp_optimized:
            return "nlp_optimized"
        elif self.ai_generated:
            return "ai_generated"
        else:
            return "manual"
    
    def is_scheduled(self) -> bool:
        """Check if post is scheduled."""
        return self.scheduled_at is not None
    
    def get_schedule_status(self) -> str:
        """Get schedule status."""
        if not self.scheduled_at:
            return "not_scheduled"
        
        now = datetime.utcnow()
        if self.scheduled_at > now:
            return "scheduled"
        else:
            return "overdue"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "nlp_optimized": self.nlp_optimized,
            "nlp_processing_time": self.nlp_processing_time,
            "ai_generated": self.ai_generated,
            "ai_model_used": self.ai_model_used,
            "views": self.views,
            "unique_views": self.unique_views,
            "impressions": self.impressions,
            "reach": self.reach,
            "click_through_rate": self.click_through_rate,
            "engagement_rate": self.engagement_rate,
            "viral_coefficient": self.viral_coefficient,
            "keywords": self.keywords,
            "hashtags": self.hashtags,
            "mentions": self.mentions,
            "links": self.links,
            "best_time_to_post": self.best_time_to_post.isoformat() if self.best_time_to_post else None,
            "time_zone": self.time_zone,
            "linkedin_post_id": self.linkedin_post_id,
            "external_url": self.external_url,
            "custom_data": self.custom_data,
            "performance_score": self.get_performance_score(),
            "optimization_status": self.get_optimization_status(),
            "schedule_status": self.get_schedule_status()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PostMetadata':
        """Create from dictionary."""
        return cls(
            scheduled_at=datetime.fromisoformat(data["scheduled_at"]) if data.get("scheduled_at") else None,
            nlp_optimized=data.get("nlp_optimized", False),
            nlp_processing_time=data.get("nlp_processing_time"),
            ai_generated=data.get("ai_generated", False),
            ai_model_used=data.get("ai_model_used"),
            views=data.get("views", 0),
            unique_views=data.get("unique_views", 0),
            impressions=data.get("impressions", 0),
            reach=data.get("reach", 0),
            click_through_rate=data.get("click_through_rate", 0.0),
            engagement_rate=data.get("engagement_rate", 0.0),
            viral_coefficient=data.get("viral_coefficient", 0.0),
            keywords=data.get("keywords", []),
            hashtags=data.get("hashtags", []),
            mentions=data.get("mentions", []),
            links=data.get("links", []),
            best_time_to_post=datetime.fromisoformat(data["best_time_to_post"]) if data.get("best_time_to_post") else None,
            time_zone=data.get("time_zone", "UTC"),
            linkedin_post_id=data.get("linkedin_post_id"),
            external_url=data.get("external_url"),
            custom_data=data.get("custom_data", {})
        )
    
    def __str__(self) -> str:
        return f"PostMetadata(optimized={self.nlp_optimized}, views={self.views}, engagement={self.engagement_rate:.2f}%)"
    
    def __repr__(self) -> str:
        return f"PostMetadata(views={self.views}, impressions={self.impressions}, engagement_rate={self.engagement_rate:.2f}%)" 