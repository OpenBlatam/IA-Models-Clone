from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from dataclasses import dataclass, field
from uuid import UUID, uuid4
import json
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
LinkedIn Post domain entity with advanced features and optimizations.
"""



class PostStatus(str, Enum):
    """Post status enumeration."""
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    PUBLISHED = "published"
    FAILED = "failed"
    ARCHIVED = "archived"


class PostType(str, Enum):
    """Post type enumeration."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    ARTICLE = "article"
    POLL = "poll"
    EVENT = "event"
    JOB = "job"


class PostTone(str, Enum):
    """Post tone enumeration."""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    INSPIRATIONAL = "inspirational"
    HUMOROUS = "humorous"


@dataclass
class EngagementMetrics:
    """Engagement metrics for a post."""
    likes: int = 0
    comments: int = 0
    shares: int = 0
    impressions: int = 0
    clicks: int = 0
    reach: int = 0
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            "likes": self.likes,
            "comments": self.comments,
            "shares": self.shares,
            "impressions": self.impressions,
            "clicks": self.clicks,
            "reach": self.reach
        }


@dataclass
class PostContent:
    """Post content with rich features."""
    text: str
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    media_urls: List[str] = field(default_factory=list)
    call_to_action: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "hashtags": self.hashtags,
            "mentions": self.mentions,
            "links": self.links,
            "media_urls": self.media_urls,
            "call_to_action": self.call_to_action
        }


@dataclass
class LinkedInPost:
    """
    LinkedIn Post domain entity with advanced features.
    
    Features:
    - Rich content management
    - Engagement tracking
    - AI optimization
    - Analytics integration
    - Multi-format support
    """
    
    # Core fields
    id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    title: str = ""
    content: PostContent = field(default_factory=lambda: PostContent(""))
    
    # Metadata
    post_type: PostType = PostType.TEXT
    tone: PostTone = PostTone.PROFESSIONAL
    status: PostStatus = PostStatus.DRAFT
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    published_at: Optional[datetime] = None
    
    # Analytics
    engagement: EngagementMetrics = field(default_factory=EngagementMetrics)
    
    # AI and optimization
    ai_score: float = 0.0
    optimization_suggestions: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # External data
    linkedin_post_id: Optional[str] = None
    external_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Performance tracking
    performance_score: float = 0.0
    reach_score: float = 0.0
    engagement_score: float = 0.0
    
    def __post_init__(self) -> Any:
        """Post-initialization processing."""
        if isinstance(self.content, dict):
            self.content = PostContent(**self.content)
        if isinstance(self.engagement, dict):
            self.engagement = EngagementMetrics(**self.engagement)
    
    @property
    def is_published(self) -> bool:
        """Check if post is published."""
        return self.status == PostStatus.PUBLISHED
    
    @property
    def is_scheduled(self) -> bool:
        """Check if post is scheduled."""
        return self.status == PostStatus.SCHEDULED
    
    @property
    def can_publish(self) -> bool:
        """Check if post can be published."""
        return (
            self.status in [PostStatus.DRAFT, PostStatus.SCHEDULED] and
            len(self.content.text.strip()) > 0
        )
    
    @property
    def total_engagement(self) -> int:
        """Calculate total engagement."""
        return (
            self.engagement.likes +
            self.engagement.comments +
            self.engagement.shares
        )
    
    @property
    def engagement_rate(self) -> float:
        """Calculate engagement rate."""
        if self.engagement.impressions == 0:
            return 0.0
        return (self.total_engagement / self.engagement.impressions) * 100
    
    def add_hashtag(self, hashtag: str) -> None:
        """Add a hashtag to the post."""
        if hashtag.startswith('#'):
            hashtag = hashtag[1:]
        if hashtag not in self.content.hashtags:
            self.content.hashtags.append(hashtag)
    
    def add_mention(self, mention: str) -> None:
        """Add a mention to the post."""
        if mention.startswith('@'):
            mention = mention[1:]
        if mention not in self.content.mentions:
            self.content.mentions.append(mention)
    
    def update_engagement(self, **metrics) -> None:
        """Update engagement metrics."""
        for key, value in metrics.items():
            if hasattr(self.engagement, key):
                setattr(self.engagement, key, value)
    
    def optimize_for_ai(self, score: float, suggestions: List[str]) -> None:
        """Update AI optimization data."""
        self.ai_score = score
        self.optimization_suggestions = suggestions
    
    def schedule(self, scheduled_time: datetime) -> None:
        """Schedule the post."""
        self.status = PostStatus.SCHEDULED
        self.scheduled_at = scheduled_time
        self.updated_at = datetime.utcnow()
    
    def publish(self) -> None:
        """Mark post as published."""
        self.status = PostStatus.PUBLISHED
        self.published_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def archive(self) -> None:
        """Archive the post."""
        self.status = PostStatus.ARCHIVED
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "title": self.title,
            "content": self.content.to_dict(),
            "post_type": self.post_type.value,
            "tone": self.tone.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "engagement": self.engagement.to_dict(),
            "ai_score": self.ai_score,
            "optimization_suggestions": self.optimization_suggestions,
            "keywords": self.keywords,
            "linkedin_post_id": self.linkedin_post_id,
            "external_metadata": self.external_metadata,
            "performance_score": self.performance_score,
            "reach_score": self.reach_score,
            "engagement_score": self.engagement_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LinkedInPost':
        """Create from dictionary."""
        # Convert string IDs to UUIDs
        if 'id' in data and isinstance(data['id'], str):
            data['id'] = UUID(data['id'])
        if 'user_id' in data and isinstance(data['user_id'], str):
            data['user_id'] = UUID(data['user_id'])
        
        # Convert string dates to datetime
        for date_field in ['created_at', 'updated_at', 'scheduled_at', 'published_at']:
            if date_field in data and data[date_field]:
                if isinstance(data[date_field], str):
                    data[date_field] = datetime.fromisoformat(data[date_field])
        
        # Convert enums
        if 'post_type' in data and isinstance(data['post_type'], str):
            data['post_type'] = PostType(data['post_type'])
        if 'tone' in data and isinstance(data['tone'], str):
            data['tone'] = PostTone(data['tone'])
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = PostStatus(data['status'])
        
        return cls(**data)
    
    def __str__(self) -> str:
        """String representation."""
        return f"LinkedInPost(id={self.id}, title='{self.title}', status={self.status.value})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"LinkedInPost(id={self.id}, user_id={self.user_id}, title='{self.title}', status={self.status.value}, engagement={self.total_engagement})" 