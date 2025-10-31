from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID
from enum import Enum
        from uuid import uuid4
        from uuid import uuid4
        from uuid import uuid4
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Post Domain Events
=================

Domain events for LinkedIn posts following event sourcing patterns.
"""



class EventType(Enum):
    """Event type enumeration."""
    POST_CREATED = "post_created"
    POST_PUBLISHED = "post_published"
    POST_OPTIMIZED = "post_optimized"
    POST_ENGAGEMENT_UPDATED = "post_engagement_updated"
    POST_DELETED = "post_deleted"
    POST_ARCHIVED = "post_archived"
    POST_SCHEDULED = "post_scheduled"
    POST_CONTENT_UPDATED = "post_content_updated"


@dataclass
class DomainEvent:
    """Base domain event class."""
    
    event_id: UUID
    event_type: EventType
    aggregate_id: UUID
    occurred_at: datetime
    version: int
    metadata: Dict[str, Any]
    
    def __post_init__(self) -> Any:
        """Set default values after initialization."""
        if not self.occurred_at:
            self.occurred_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": str(self.event_id),
            "event_type": self.event_type.value,
            "aggregate_id": str(self.aggregate_id),
            "occurred_at": self.occurred_at.isoformat(),
            "version": self.version,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DomainEvent':
        """Create event from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]),
            event_type=EventType(data["event_type"]),
            aggregate_id=UUID(data["aggregate_id"]),
            occurred_at=datetime.fromisoformat(data["occurred_at"]),
            version=data["version"],
            metadata=data["metadata"]
        )


@dataclass
class PostCreatedEvent(DomainEvent):
    """Event raised when a post is created."""
    
    content: str
    author_id: UUID
    post_type: str
    tone: str
    
    def __post_init__(self) -> Any:
        """Initialize event with default values."""
        super().__post_init__()
        self.event_type = EventType.POST_CREATED
        self.metadata.update({
            "content_length": len(self.content),
            "post_type": self.post_type,
            "tone": self.tone
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "content": self.content,
            "author_id": str(self.author_id),
            "post_type": self.post_type,
            "tone": self.tone
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PostCreatedEvent':
        """Create from dictionary."""
        event = cls(
            event_id=UUID(data["event_id"]),
            aggregate_id=UUID(data["aggregate_id"]),
            content=data["content"],
            author_id=UUID(data["author_id"]),
            post_type=data["post_type"],
            tone=data["tone"],
            occurred_at=datetime.fromisoformat(data["occurred_at"]),
            version=data["version"],
            metadata=data["metadata"]
        )
        return event


@dataclass
class PostPublishedEvent(DomainEvent):
    """Event raised when a post is published."""
    
    author_id: UUID
    published_at: datetime
    
    def __post_init__(self) -> Any:
        """Initialize event with default values."""
        super().__post_init__()
        self.event_type = EventType.POST_PUBLISHED
        self.metadata.update({
            "published_at": self.published_at.isoformat()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "author_id": str(self.author_id),
            "published_at": self.published_at.isoformat()
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PostPublishedEvent':
        """Create from dictionary."""
        event = cls(
            event_id=UUID(data["event_id"]),
            aggregate_id=UUID(data["aggregate_id"]),
            author_id=UUID(data["author_id"]),
            published_at=datetime.fromisoformat(data["published_at"]),
            occurred_at=datetime.fromisoformat(data["occurred_at"]),
            version=data["version"],
            metadata=data["metadata"]
        )
        return event


@dataclass
class PostOptimizedEvent(DomainEvent):
    """Event raised when a post is optimized."""
    
    old_content: str
    new_content: str
    optimized_at: datetime
    nlp_processing_time: Optional[float] = None
    ai_model_used: Optional[str] = None
    
    def __post_init__(self) -> Any:
        """Initialize event with default values."""
        super().__post_init__()
        self.event_type = EventType.POST_OPTIMIZED
        self.metadata.update({
            "content_improvement": len(self.new_content) - len(self.old_content),
            "nlp_processing_time": self.nlp_processing_time,
            "ai_model_used": self.ai_model_used
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "old_content": self.old_content,
            "new_content": self.new_content,
            "optimized_at": self.optimized_at.isoformat(),
            "nlp_processing_time": self.nlp_processing_time,
            "ai_model_used": self.ai_model_used
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PostOptimizedEvent':
        """Create from dictionary."""
        event = cls(
            event_id=UUID(data["event_id"]),
            aggregate_id=UUID(data["aggregate_id"]),
            old_content=data["old_content"],
            new_content=data["new_content"],
            optimized_at=datetime.fromisoformat(data["optimized_at"]),
            nlp_processing_time=data.get("nlp_processing_time"),
            ai_model_used=data.get("ai_model_used"),
            occurred_at=datetime.fromisoformat(data["occurred_at"]),
            version=data["version"],
            metadata=data["metadata"]
        )
        return event


@dataclass
class PostEngagementUpdatedEvent(DomainEvent):
    """Event raised when post engagement is updated."""
    
    old_metrics: Dict[str, Any]
    new_metrics: Dict[str, Any]
    updated_at: datetime
    
    def __post_init__(self) -> Any:
        """Initialize event with default values."""
        super().__post_init__()
        self.event_type = EventType.POST_ENGAGEMENT_UPDATED
        
        # Calculate engagement changes
        engagement_changes = {}
        for key in ["likes", "comments", "shares", "saves", "clicks"]:
            old_value = self.old_metrics.get(key, 0)
            new_value = self.new_metrics.get(key, 0)
            engagement_changes[f"{key}_change"] = new_value - old_value
        
        self.metadata.update(engagement_changes)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "old_metrics": self.old_metrics,
            "new_metrics": self.new_metrics,
            "updated_at": self.updated_at.isoformat()
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PostEngagementUpdatedEvent':
        """Create from dictionary."""
        event = cls(
            event_id=UUID(data["event_id"]),
            aggregate_id=UUID(data["aggregate_id"]),
            old_metrics=data["old_metrics"],
            new_metrics=data["new_metrics"],
            updated_at=datetime.fromisoformat(data["updated_at"]),
            occurred_at=datetime.fromisoformat(data["occurred_at"]),
            version=data["version"],
            metadata=data["metadata"]
        )
        return event


@dataclass
class PostDeletedEvent(DomainEvent):
    """Event raised when a post is deleted."""
    
    deleted_at: datetime
    deletion_reason: Optional[str] = None
    
    def __post_init__(self) -> Any:
        """Initialize event with default values."""
        super().__post_init__()
        self.event_type = EventType.POST_DELETED
        self.metadata.update({
            "deletion_reason": self.deletion_reason
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "deleted_at": self.deleted_at.isoformat(),
            "deletion_reason": self.deletion_reason
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PostDeletedEvent':
        """Create from dictionary."""
        event = cls(
            event_id=UUID(data["event_id"]),
            aggregate_id=UUID(data["aggregate_id"]),
            deleted_at=datetime.fromisoformat(data["deleted_at"]),
            deletion_reason=data.get("deletion_reason"),
            occurred_at=datetime.fromisoformat(data["occurred_at"]),
            version=data["version"],
            metadata=data["metadata"]
        )
        return event


@dataclass
class PostScheduledEvent(DomainEvent):
    """Event raised when a post is scheduled."""
    
    scheduled_at: datetime
    scheduled_by: UUID
    
    def __post_init__(self) -> Any:
        """Initialize event with default values."""
        super().__post_init__()
        self.event_type = EventType.POST_SCHEDULED
        self.metadata.update({
            "scheduled_at": self.scheduled_at.isoformat()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "scheduled_at": self.scheduled_at.isoformat(),
            "scheduled_by": str(self.scheduled_by)
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PostScheduledEvent':
        """Create from dictionary."""
        event = cls(
            event_id=UUID(data["event_id"]),
            aggregate_id=UUID(data["aggregate_id"]),
            scheduled_at=datetime.fromisoformat(data["scheduled_at"]),
            scheduled_by=UUID(data["scheduled_by"]),
            occurred_at=datetime.fromisoformat(data["occurred_at"]),
            version=data["version"],
            metadata=data["metadata"]
        )
        return event


@dataclass
class PostContentUpdatedEvent(DomainEvent):
    """Event raised when post content is updated."""
    
    old_content: str
    new_content: str
    updated_by: UUID
    update_reason: Optional[str] = None
    
    def __post_init__(self) -> Any:
        """Initialize event with default values."""
        super().__post_init__()
        self.event_type = EventType.POST_CONTENT_UPDATED
        self.metadata.update({
            "content_length_change": len(self.new_content) - len(self.old_content),
            "update_reason": self.update_reason
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "old_content": self.old_content,
            "new_content": self.new_content,
            "updated_by": str(self.updated_by),
            "update_reason": self.update_reason
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PostContentUpdatedEvent':
        """Create from dictionary."""
        event = cls(
            event_id=UUID(data["event_id"]),
            aggregate_id=UUID(data["aggregate_id"]),
            old_content=data["old_content"],
            new_content=data["new_content"],
            updated_by=UUID(data["updated_by"]),
            update_reason=data.get("update_reason"),
            occurred_at=datetime.fromisoformat(data["occurred_at"]),
            version=data["version"],
            metadata=data["metadata"]
        )
        return event


# Event factory for creating events
class EventFactory:
    """Factory for creating domain events."""
    
    @staticmethod
    def create_post_created_event(post_id: UUID, content: str, author_id: UUID, 
                                 post_type: str, tone: str) -> PostCreatedEvent:
        """Create a post created event."""
        return PostCreatedEvent(
            event_id=uuid4(),
            aggregate_id=post_id,
            content=content,
            author_id=author_id,
            post_type=post_type,
            tone=tone,
            occurred_at=datetime.utcnow(),
            version=1,
            metadata={}
        )
    
    @staticmethod
    def create_post_published_event(post_id: UUID, author_id: UUID, 
                                   published_at: datetime) -> PostPublishedEvent:
        """Create a post published event."""
        return PostPublishedEvent(
            event_id=uuid4(),
            aggregate_id=post_id,
            author_id=author_id,
            published_at=published_at,
            occurred_at=datetime.utcnow(),
            version=1,
            metadata={}
        )
    
    @staticmethod
    def create_post_optimized_event(post_id: UUID, old_content: str, new_content: str,
                                   optimized_at: datetime, nlp_processing_time: Optional[float] = None,
                                   ai_model_used: Optional[str] = None) -> PostOptimizedEvent:
        """Create a post optimized event."""
        return PostOptimizedEvent(
            event_id=uuid4(),
            aggregate_id=post_id,
            old_content=old_content,
            new_content=new_content,
            optimized_at=optimized_at,
            nlp_processing_time=nlp_processing_time,
            ai_model_used=ai_model_used,
            occurred_at=datetime.utcnow(),
            version=1,
            metadata={}
        ) 