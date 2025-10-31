from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from enum import Enum
from ..value_objects.content import Content
from ..value_objects.author import Author
from ..value_objects.post_metadata import PostMetadata
from ..value_objects.engagement_metrics import EngagementMetrics
from ..events.post_events import (
from ..exceptions.post_exceptions import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
LinkedIn Post Domain Entity - Refactored
========================================

Rich domain entity with business logic, value objects, and domain events.
Following Domain-Driven Design principles.
"""


    PostCreatedEvent,
    PostPublishedEvent,
    PostOptimizedEvent,
    PostEngagementUpdatedEvent,
    PostDeletedEvent,
    EventFactory
)
    InvalidPostStateError,
    ContentValidationError,
    PostAlreadyPublishedError
)


class PostStatus(Enum):
    """Post status enumeration."""
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    DELETED = "deleted"


class PostType(Enum):
    """Post type enumeration."""
    ARTICLE = "article"
    SHARE = "share"
    POLL = "poll"
    VIDEO = "video"
    IMAGE = "image"
    TEXT = "text"


class PostTone(Enum):
    """Post tone enumeration."""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    INSPIRATIONAL = "inspirational"
    EDUCATIONAL = "educational"


@dataclass
class LinkedInPost:
    """
    LinkedIn Post domain entity with rich behavior.
    
    This entity encapsulates all business logic related to LinkedIn posts,
    including validation, state management, and domain events.
    """
    
    # Identity
    id: UUID = field(default_factory=uuid4, init=False)
    
    # Core attributes
    content: Content
    author: Author
    post_type: PostType = PostType.TEXT
    tone: PostTone = PostTone.PROFESSIONAL
    
    # Metadata
    metadata: PostMetadata = field(default_factory=PostMetadata)
    engagement_metrics: EngagementMetrics = field(default_factory=EngagementMetrics)
    
    # State
    status: PostStatus = PostStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    published_at: Optional[datetime] = None
    
    # Domain events
    _events: List[Any] = field(default_factory=list, init=False)
    _version: int = field(default=1, init=False)
    
    def __post_init__(self) -> Any:
        """Validate entity after initialization."""
        self._validate_initial_state()
    
    def _validate_initial_state(self) -> None:
        """Validate the initial state of the post."""
        if not self.content:
            raise ContentValidationError("Post content cannot be empty")
        
        if not self.author:
            raise ContentValidationError("Post author is required")
    
    # Business Logic Methods
    
    def publish(self) -> None:
        """
        Publish the post.
        
        Raises:
            PostAlreadyPublishedError: If post is already published
            InvalidPostStateError: If post cannot be published
        """
        if self.status == PostStatus.PUBLISHED:
            raise PostAlreadyPublishedError(f"Post {self.id} is already published")
        
        if self.status == PostStatus.DELETED:
            raise InvalidPostStateError("Cannot publish a deleted post")
        
        if not self.content.is_valid_for_publishing():
            raise ContentValidationError("Content is not valid for publishing")
        
        # Update state
        self.status = PostStatus.PUBLISHED
        self.published_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self._version += 1
        
        # Add domain event
        self._add_event(EventFactory.create_post_published_event(
            post_id=self.id,
            author_id=self.author.id,
            published_at=self.published_at
        ))
    
    def schedule(self, scheduled_at: datetime) -> None:
        """
        Schedule the post for future publication.
        
        Args:
            scheduled_at: When the post should be published
        """
        if self.status != PostStatus.DRAFT:
            raise InvalidPostStateError("Only draft posts can be scheduled")
        
        if scheduled_at <= datetime.utcnow():
            raise InvalidPostStateError("Scheduled time must be in the future")
        
        self.status = PostStatus.SCHEDULED
        self.metadata.scheduled_at = scheduled_at
        self.updated_at = datetime.utcnow()
        self._version += 1
    
    def update_content(self, new_content: str) -> None:
        """
        Update the post content.
        
        Args:
            new_content: New content for the post
        """
        if self.status == PostStatus.PUBLISHED:
            raise InvalidPostStateError("Cannot update content of published post")
        
        old_content = self.content.value
        self.content = Content(new_content)
        self.updated_at = datetime.utcnow()
        self._version += 1
        
        # Add domain event for content update
        self._add_event(PostOptimizedEvent(
            post_id=self.id,
            old_content=old_content,
            new_content=new_content,
            optimized_at=self.updated_at
        ))
    
    def optimize_with_nlp(self, nlp_service: 'NLPService') -> None:
        """
        Optimize the post content using NLP service.
        
        Args:
            nlp_service: NLP service for content optimization
        """
        if self.status == PostStatus.PUBLISHED:
            raise InvalidPostStateError("Cannot optimize published post")
        
        optimized_content = nlp_service.optimize(self.content.value)
        self.update_content(optimized_content)
        
        # Update metadata
        self.metadata.nlp_optimized = True
        self.metadata.nlp_processing_time = nlp_service.last_processing_time
    
    def update_engagement(self, likes: int = 0, comments: int = 0, shares: int = 0) -> None:
        """
        Update engagement metrics.
        
        Args:
            likes: Number of likes
            comments: Number of comments
            shares: Number of shares
        """
        old_metrics = self.engagement_metrics.copy()
        
        self.engagement_metrics.update(likes, comments, shares)
        self.updated_at = datetime.utcnow()
        self._version += 1
        
        # Add domain event for engagement update
        self._add_event(PostEngagementUpdatedEvent(
            post_id=self.id,
            old_metrics=old_metrics,
            new_metrics=self.engagement_metrics,
            updated_at=self.updated_at
        ))
    
    def archive(self) -> None:
        """Archive the post."""
        if self.status not in [PostStatus.PUBLISHED, PostStatus.SCHEDULED]:
            raise InvalidPostStateError("Only published or scheduled posts can be archived")
        
        self.status = PostStatus.ARCHIVED
        self.updated_at = datetime.utcnow()
        self._version += 1
    
    def delete(self) -> None:
        """Delete the post."""
        self.status = PostStatus.DELETED
        self.updated_at = datetime.utcnow()
        self._version += 1
        
        # Add domain event
        self._add_event(PostDeletedEvent(
            post_id=self.id,
            deleted_at=self.updated_at
        ))
    
    def restore(self) -> None:
        """Restore a deleted post."""
        if self.status != PostStatus.DELETED:
            raise InvalidPostStateError("Only deleted posts can be restored")
        
        self.status = PostStatus.DRAFT
        self.updated_at = datetime.utcnow()
        self._version += 1
    
    # Domain Event Methods
    
    def _add_event(self, event: Any) -> None:
        """Add a domain event to the entity."""
        self._events.append(event)
    
    def get_events(self) -> List[Any]:
        """Get all domain events and clear them."""
        events = self._events.copy()
        self._events.clear()
        return events
    
    def has_events(self) -> bool:
        """Check if the entity has pending events."""
        return len(self._events) > 0
    
    # Query Methods
    
    def is_published(self) -> bool:
        """Check if the post is published."""
        return self.status == PostStatus.PUBLISHED
    
    def is_draft(self) -> bool:
        """Check if the post is a draft."""
        return self.status == PostStatus.DRAFT
    
    def is_scheduled(self) -> bool:
        """Check if the post is scheduled."""
        return self.status == PostStatus.SCHEDULED
    
    def is_deleted(self) -> bool:
        """Check if the post is deleted."""
        return self.status == PostStatus.DELETED
    
    def can_be_published(self) -> bool:
        """Check if the post can be published."""
        return (self.status in [PostStatus.DRAFT, PostStatus.SCHEDULED] and 
                self.content.is_valid_for_publishing())
    
    def get_engagement_score(self) -> float:
        """Calculate engagement score."""
        return self.engagement_metrics.calculate_score()
    
    def get_age_in_days(self) -> int:
        """Get the age of the post in days."""
        return (datetime.utcnow() - self.created_at).days
    
    # Serialization Methods
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary."""
        return {
            "id": str(self.id),
            "content": self.content.to_dict(),
            "author": self.author.to_dict(),
            "post_type": self.post_type.value,
            "tone": self.tone.value,
            "metadata": self.metadata.to_dict(),
            "engagement_metrics": self.engagement_metrics.to_dict(),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "version": self._version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LinkedInPost':
        """Create entity from dictionary."""
        return cls(
            id=UUID(data["id"]),
            content=Content.from_dict(data["content"]),
            author=Author.from_dict(data["author"]),
            post_type=PostType(data["post_type"]),
            tone=PostTone(data["tone"]),
            metadata=PostMetadata.from_dict(data["metadata"]),
            engagement_metrics=EngagementMetrics.from_dict(data["engagement_metrics"]),
            status=PostStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            published_at=datetime.fromisoformat(data["published_at"]) if data["published_at"] else None
        )
    
    # Factory Methods
    
    @classmethod
    def create_draft(cls, content: str, author: Author, **kwargs) -> 'LinkedInPost':
        """Create a new draft post."""
        return cls(
            content=Content(content),
            author=author,
            status=PostStatus.DRAFT,
            **kwargs
        )
    
    @classmethod
    def create_scheduled(cls, content: str, author: Author, scheduled_at: datetime, **kwargs) -> 'LinkedInPost':
        """Create a new scheduled post."""
        post = cls.create_draft(content, author, **kwargs)
        post.schedule(scheduled_at)
        return post
    
    # String Representation
    
    def __str__(self) -> str:
        return f"LinkedInPost(id={self.id}, status={self.status.value}, content_length={len(self.content.value)})"
    
    def __repr__(self) -> str:
        return f"LinkedInPost(id={self.id}, status={self.status.value}, author={self.author.id})"


# Type hints for services
class NLPService:
    """NLP service interface for type hints."""
    def optimize(self, content: str) -> str:
        """Optimize content using NLP."""
        pass
    
    @property
    def last_processing_time(self) -> float:
        """Get last processing time."""
        pass 