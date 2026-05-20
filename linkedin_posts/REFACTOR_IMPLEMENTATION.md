# LinkedIn Posts System - Refactor Implementation Guide
## Step-by-Step Implementation with Code Examples

### 🎯 Implementation Overview

This guide provides a comprehensive step-by-step implementation of the LinkedIn posts system refactor, following clean architecture principles and modern development practices.

### 📋 Implementation Checklist

#### Phase 1: Core Domain Layer ✅
- [x] Domain entities with rich behavior
- [x] Value objects for type safety
- [x] Domain events and exceptions
- [ ] Domain services

#### Phase 2: Application Layer
- [ ] CQRS commands and queries
- [ ] Application services
- [ ] Event handlers
- [ ] Use cases

#### Phase 3: Infrastructure Layer
- [ ] Repository implementations
- [ ] Event bus
- [ ] Caching layer
- [ ] External services

#### Phase 4: Presentation Layer
- [ ] API endpoints
- [ ] Validation
- [ ] Error handling
- [ ] Documentation

### 🚀 Phase 1: Core Domain Implementation

#### 1.1 Domain Entities

**File: `core/domain/entities/linkedin_post_refactored.py`**

```python
@dataclass
class LinkedInPost:
    """LinkedIn Post domain entity with rich behavior."""
    
    # Identity
    id: UUID = field(default_factory=uuid4)
    
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
    
    def publish(self) -> None:
        """Publish the post."""
        if self.status == PostStatus.PUBLISHED:
            raise PostAlreadyPublishedError(f"Post {self.id} is already published")
        
        if not self.content.is_valid_for_publishing():
            raise ContentValidationError("Content is not valid for publishing")
        
        self.status = PostStatus.PUBLISHED
        self.published_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self._version += 1
        
        self._add_event(PostPublishedEvent(
            post_id=self.id,
            author_id=self.author.id,
            published_at=self.published_at
        ))
    
    def optimize_with_nlp(self, nlp_service: 'NLPService') -> None:
        """Optimize the post content using NLP service."""
        if self.status == PostStatus.PUBLISHED:
            raise InvalidPostStateError("Cannot optimize published post")
        
        optimized_content = nlp_service.optimize(self.content.value)
        self.update_content(optimized_content)
        
        self.metadata.nlp_optimized = True
        self.metadata.nlp_processing_time = nlp_service.last_processing_time
```

#### 1.2 Value Objects

**File: `core/domain/value_objects/content.py`**

```python
@dataclass(frozen=True)
class Content:
    """Content value object with validation and business rules."""
    
    value: str
    
    def __post_init__(self):
        """Validate content after initialization."""
        self._validate_content()
    
    def _validate_content(self) -> None:
        """Validate content according to business rules."""
        if not self.value:
            raise ValueError("Content cannot be empty")
        
        if len(self.value.strip()) < 10:
            raise ValueError("Content must be at least 10 characters long")
        
        if len(self.value) > 3000:
            raise ValueError("Content cannot exceed 3000 characters")
    
    def is_valid_for_publishing(self) -> bool:
        """Check if content is valid for publishing."""
        try:
            self._validate_content()
            return True
        except ValueError:
            return False
    
    def get_word_count(self) -> int:
        """Get the word count of the content."""
        return len(self.value.split())
    
    def get_hashtags(self) -> List[str]:
        """Extract hashtags from content."""
        hashtag_pattern = r'#\w+'
        return re.findall(hashtag_pattern, self.value)
```

#### 1.3 Domain Events

**File: `core/domain/events/post_events.py`**

```python
@dataclass
class PostPublishedEvent(DomainEvent):
    """Event raised when a post is published."""
    
    author_id: UUID
    published_at: datetime
    
    def __post_init__(self):
        """Initialize event with default values."""
        super().__post_init__()
        self.event_type = EventType.POST_PUBLISHED
        self.metadata.update({
            "published_at": self.published_at.isoformat()
        })

@dataclass
class PostOptimizedEvent(DomainEvent):
    """Event raised when a post is optimized."""
    
    old_content: str
    new_content: str
    optimized_at: datetime
    nlp_processing_time: Optional[float] = None
    ai_model_used: Optional[str] = None
```

#### 1.4 Domain Exceptions

**File: `core/domain/exceptions/post_exceptions.py`**

```python
class PostDomainError(Exception):
    """Base exception for post domain errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "POST_DOMAIN_ERROR"
        self.details = details or {}

class InvalidPostStateError(PostDomainError):
    """Exception raised when post state is invalid for the requested operation."""
    
    def __init__(self, message: str, current_state: Optional[str] = None, 
                 required_state: Optional[str] = None):
        details = {}
        if current_state:
            details["current_state"] = current_state
        if required_state:
            details["required_state"] = required_state
        
        super().__init__(message, "INVALID_POST_STATE", details)

class ContentValidationError(PostDomainError):
    """Exception raised when post content validation fails."""
    
    def __init__(self, message: str, content: Optional[str] = None, 
                 validation_errors: Optional[list] = None):
        details = {}
        if content:
            details["content"] = content
        if validation_errors:
            details["validation_errors"] = validation_errors
        
        super().__init__(message, "CONTENT_VALIDATION_ERROR", details)
```

### 🔄 Phase 2: Application Layer Implementation

#### 2.1 CQRS Commands

**File: `core/application/commands/post_commands.py`**

```python
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

@dataclass
class CreatePostCommand:
    """Command to create a new post."""
    content: str
    author_id: UUID
    post_type: str
    tone: str
    target_audience: Optional[str] = None
    industry: Optional[str] = None

@dataclass
class PublishPostCommand:
    """Command to publish a post."""
    post_id: UUID
    author_id: UUID

@dataclass
class OptimizePostCommand:
    """Command to optimize a post."""
    post_id: UUID
    use_nlp: bool = True
    use_async: bool = False

@dataclass
class UpdatePostCommand:
    """Command to update a post."""
    post_id: UUID
    content: Optional[str] = None
    post_type: Optional[str] = None
    tone: Optional[str] = None
    target_audience: Optional[str] = None
    industry: Optional[str] = None

@dataclass
class DeletePostCommand:
    """Command to delete a post."""
    post_id: UUID
    author_id: UUID
    reason: Optional[str] = None
```

#### 2.2 CQRS Queries

**File: `core/application/queries/post_queries.py`**

```python
from dataclasses import dataclass
from typing import Optional, List
from uuid import UUID

@dataclass
class GetPostQuery:
    """Query to get a specific post."""
    post_id: UUID
    include_metrics: bool = True

@dataclass
class ListPostsQuery:
    """Query to list posts with filtering."""
    author_id: Optional[UUID] = None
    status: Optional[str] = None
    post_type: Optional[str] = None
    limit: int = 50
    offset: int = 0
    sort_by: str = "created_at"
    sort_order: str = "desc"

@dataclass
class GetPostAnalyticsQuery:
    """Query to get post analytics."""
    post_id: UUID
    time_range: str = "7d"
    include_engagement: bool = True
    include_performance: bool = True

@dataclass
class SearchPostsQuery:
    """Query to search posts."""
    query: str
    author_id: Optional[UUID] = None
    post_type: Optional[str] = None
    limit: int = 20
    offset: int = 0
```

#### 2.3 Command Handlers

**File: `core/application/commands/handlers/post_command_handlers.py`**

```python
from typing import Optional
from uuid import UUID
import asyncio

from ...domain.entities.linkedin_post_refactored import LinkedInPost
from ...domain.events.post_events import PostCreatedEvent, PostPublishedEvent
from ...domain.exceptions.post_exceptions import (
    PostNotFoundError, InvalidPostStateError, ContentValidationError
)
from ...interfaces.repositories.post_repository import PostRepository
from ...interfaces.services.event_bus import EventBus
from ...interfaces.services.nlp_service import NLPService
from ..post_commands import (
    CreatePostCommand, PublishPostCommand, OptimizePostCommand,
    UpdatePostCommand, DeletePostCommand
)

class CreatePostCommandHandler:
    """Handler for creating posts."""
    
    def __init__(self, repository: PostRepository, event_bus: EventBus):
        self.repository = repository
        self.event_bus = event_bus
    
    async def handle(self, command: CreatePostCommand) -> UUID:
        """Handle create post command."""
        # Create post entity
        post = LinkedInPost.create_draft(
            content=command.content,
            author=Author(command.author_id, "Author Name"),  # TODO: Get from user service
            post_type=PostType(command.post_type),
            tone=PostTone(command.tone)
        )
        
        # Save to repository
        await self.repository.save(post)
        
        # Publish domain events
        events = post.get_events()
        for event in events:
            await self.event_bus.publish(event)
        
        return post.id

class PublishPostCommandHandler:
    """Handler for publishing posts."""
    
    def __init__(self, repository: PostRepository, event_bus: EventBus):
        self.repository = repository
        self.event_bus = event_bus
    
    async def handle(self, command: PublishPostCommand) -> None:
        """Handle publish post command."""
        # Get post
        post = await self.repository.get_by_id(command.post_id)
        if not post:
            raise PostNotFoundError(str(command.post_id))
        
        # Publish post
        post.publish()
        
        # Save to repository
        await self.repository.save(post)
        
        # Publish domain events
        events = post.get_events()
        for event in events:
            await self.event_bus.publish(event)

class OptimizePostCommandHandler:
    """Handler for optimizing posts."""
    
    def __init__(self, repository: PostRepository, nlp_service: NLPService, event_bus: EventBus):
        self.repository = repository
        self.nlp_service = nlp_service
        self.event_bus = event_bus
    
    async def handle(self, command: OptimizePostCommand) -> None:
        """Handle optimize post command."""
        # Get post
        post = await self.repository.get_by_id(command.post_id)
        if not post:
            raise PostNotFoundError(str(command.post_id))
        
        # Optimize with NLP
        if command.use_nlp:
            if command.use_async:
                # Async optimization
                optimization_task = asyncio.create_task(
                    self.nlp_service.optimize_async(post.content.value)
                )
                optimized_content = await optimization_task
            else:
                # Sync optimization
                optimized_content = self.nlp_service.optimize(post.content.value)
            
            post.update_content(optimized_content)
        
        # Save to repository
        await self.repository.save(post)
        
        # Publish domain events
        events = post.get_events()
        for event in events:
            await self.event_bus.publish(event)
```

#### 2.4 Query Handlers

**File: `core/application/queries/handlers/post_query_handlers.py`**

```python
from typing import Optional, List
from uuid import UUID

from ...interfaces.repositories.post_repository import PostRepository
from ...interfaces.services.cache_service import CacheService
from ...shared.dtos.post_dto import PostDTO, PostListDTO
from ..post_queries import (
    GetPostQuery, ListPostsQuery, GetPostAnalyticsQuery, SearchPostsQuery
)

class GetPostQueryHandler:
    """Handler for getting a specific post."""
    
    def __init__(self, repository: PostRepository, cache: CacheService):
        self.repository = repository
        self.cache = cache
    
    async def handle(self, query: GetPostQuery) -> Optional[PostDTO]:
        """Handle get post query."""
        # Try cache first
        cache_key = f"post:{query.post_id}"
        cached = await self.cache.get(cache_key)
        if cached:
            return PostDTO.from_dict(cached)
        
        # Get from repository
        post = await self.repository.get_by_id(query.post_id)
        if not post:
            return None
        
        # Convert to DTO
        dto = PostDTO.from_entity(post, include_metrics=query.include_metrics)
        
        # Cache result
        await self.cache.set(cache_key, dto.to_dict(), ttl=3600)
        
        return dto

class ListPostsQueryHandler:
    """Handler for listing posts."""
    
    def __init__(self, repository: PostRepository, cache: CacheService):
        self.repository = repository
        self.cache = cache
    
    async def handle(self, query: ListPostsQuery) -> PostListDTO:
        """Handle list posts query."""
        # Generate cache key
        cache_key = f"posts:list:{hash(query)}"
        cached = await self.cache.get(cache_key)
        if cached:
            return PostListDTO.from_dict(cached)
        
        # Get from repository
        posts = await self.repository.list_posts(
            author_id=query.author_id,
            status=query.status,
            post_type=query.post_type,
            limit=query.limit,
            offset=query.offset,
            sort_by=query.sort_by,
            sort_order=query.sort_order
        )
        
        # Convert to DTOs
        post_dtos = [PostDTO.from_entity(post) for post in posts]
        result = PostListDTO(
            posts=post_dtos,
            total=len(post_dtos),
            limit=query.limit,
            offset=query.offset
        )
        
        # Cache result
        await self.cache.set(cache_key, result.to_dict(), ttl=300)
        
        return result
```

### 🏗️ Phase 3: Infrastructure Layer Implementation

#### 3.1 Repository Interface

**File: `core/interfaces/repositories/post_repository.py`**

```python
from abc import ABC, abstractmethod
from typing import Optional, List
from uuid import UUID

from ...domain.entities.linkedin_post_refactored import LinkedInPost

class PostRepository(ABC):
    """Repository interface for LinkedIn posts."""
    
    @abstractmethod
    async def save(self, post: LinkedInPost) -> None:
        """Save a post."""
        pass
    
    @abstractmethod
    async def get_by_id(self, post_id: UUID) -> Optional[LinkedInPost]:
        """Get a post by ID."""
        pass
    
    @abstractmethod
    async def list_posts(self, 
                        author_id: Optional[UUID] = None,
                        status: Optional[str] = None,
                        post_type: Optional[str] = None,
                        limit: int = 50,
                        offset: int = 0,
                        sort_by: str = "created_at",
                        sort_order: str = "desc") -> List[LinkedInPost]:
        """List posts with filtering."""
        pass
    
    @abstractmethod
    async def delete(self, post_id: UUID) -> bool:
        """Delete a post."""
        pass
    
    @abstractmethod
    async def exists(self, post_id: UUID) -> bool:
        """Check if a post exists."""
        pass
    
    @abstractmethod
    async def count(self, author_id: Optional[UUID] = None) -> int:
        """Count posts."""
        pass
```

#### 3.2 Repository Implementation

**File: `infrastructure/persistence/repositories/post_repository_impl.py`**

```python
from typing import Optional, List
from uuid import UUID
import asyncio

from ...core.interfaces.repositories.post_repository import PostRepository
from ...core.domain.entities.linkedin_post_refactored import LinkedInPost
from ...core.domain.exceptions.post_exceptions import PostNotFoundError
from .models.post_model import PostModel
from .database import get_database

class PostRepositoryImpl(PostRepository):
    """PostgreSQL implementation of post repository."""
    
    def __init__(self, database=None):
        self.db = database or get_database()
    
    async def save(self, post: LinkedInPost) -> None:
        """Save a post."""
        post_data = post.to_dict()
        
        # Convert to model
        post_model = PostModel(
            id=post_data["id"],
            content=post_data["content"]["value"],
            author_id=post_data["author"]["id"],
            post_type=post_data["post_type"],
            tone=post_data["tone"],
            status=post_data["status"],
            metadata=post_data["metadata"],
            engagement_metrics=post_data["engagement_metrics"],
            created_at=post_data["created_at"],
            updated_at=post_data["updated_at"],
            published_at=post_data["published_at"],
            version=post_data["version"]
        )
        
        # Save to database
        async with self.db.transaction():
            await self.db.execute(
                """
                INSERT INTO posts (id, content, author_id, post_type, tone, status, 
                                 metadata, engagement_metrics, created_at, updated_at, 
                                 published_at, version)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    post_type = EXCLUDED.post_type,
                    tone = EXCLUDED.tone,
                    status = EXCLUDED.status,
                    metadata = EXCLUDED.metadata,
                    engagement_metrics = EXCLUDED.engagement_metrics,
                    updated_at = EXCLUDED.updated_at,
                    published_at = EXCLUDED.published_at,
                    version = EXCLUDED.version
                """,
                post_model.id, post_model.content, post_model.author_id,
                post_model.post_type, post_model.tone, post_model.status,
                post_model.metadata, post_model.engagement_metrics,
                post_model.created_at, post_model.updated_at,
                post_model.published_at, post_model.version
            )
    
    async def get_by_id(self, post_id: UUID) -> Optional[LinkedInPost]:
        """Get a post by ID."""
        result = await self.db.fetchrow(
            "SELECT * FROM posts WHERE id = $1",
            str(post_id)
        )
        
        if not result:
            return None
        
        # Convert to entity
        return LinkedInPost.from_dict(dict(result))
    
    async def list_posts(self, 
                        author_id: Optional[UUID] = None,
                        status: Optional[str] = None,
                        post_type: Optional[str] = None,
                        limit: int = 50,
                        offset: int = 0,
                        sort_by: str = "created_at",
                        sort_order: str = "desc") -> List[LinkedInPost]:
        """List posts with filtering."""
        # Build query
        query = "SELECT * FROM posts WHERE 1=1"
        params = []
        param_count = 0
        
        if author_id:
            param_count += 1
            query += f" AND author_id = ${param_count}"
            params.append(str(author_id))
        
        if status:
            param_count += 1
            query += f" AND status = ${param_count}"
            params.append(status)
        
        if post_type:
            param_count += 1
            query += f" AND post_type = ${param_count}"
            params.append(post_type)
        
        # Add sorting and pagination
        query += f" ORDER BY {sort_by} {sort_order.upper()}"
        query += f" LIMIT ${param_count + 1} OFFSET ${param_count + 2}"
        params.extend([limit, offset])
        
        # Execute query
        results = await self.db.fetch(query, *params)
        
        # Convert to entities
        return [LinkedInPost.from_dict(dict(result)) for result in results]
    
    async def delete(self, post_id: UUID) -> bool:
        """Delete a post."""
        result = await self.db.execute(
            "DELETE FROM posts WHERE id = $1",
            str(post_id)
        )
        return result == "DELETE 1"
    
    async def exists(self, post_id: UUID) -> bool:
        """Check if a post exists."""
        result = await self.db.fetchval(
            "SELECT EXISTS(SELECT 1 FROM posts WHERE id = $1)",
            str(post_id)
        )
        return result
    
    async def count(self, author_id: Optional[UUID] = None) -> int:
        """Count posts."""
        if author_id:
            result = await self.db.fetchval(
                "SELECT COUNT(*) FROM posts WHERE author_id = $1",
                str(author_id)
            )
        else:
            result = await self.db.fetchval("SELECT COUNT(*) FROM posts")
        
        return result or 0
```

#### 3.3 Event Bus Implementation

**File: `infrastructure/messaging/event_bus_impl.py`**

```python
from typing import Dict, List, Type, Any
import asyncio
import json
from datetime import datetime

from ...core.interfaces.services.event_bus import EventBus
from ...core.domain.events.post_events import DomainEvent
from ...shared.logging import get_logger

logger = get_logger(__name__)

class EventBusImpl(EventBus):
    """Redis-based event bus implementation."""
    
    def __init__(self, redis_client, handlers: Dict[Type[DomainEvent], List[Any]] = None):
        self.redis = redis_client
        self.handlers = handlers or {}
        self.event_stream = "post_events"
    
    async def publish(self, event: DomainEvent) -> None:
        """Publish an event."""
        try:
            # Serialize event
            event_data = {
                "event_id": str(event.event_id),
                "event_type": event.event_type.value,
                "aggregate_id": str(event.aggregate_id),
                "occurred_at": event.occurred_at.isoformat(),
                "version": event.version,
                "metadata": event.metadata,
                "data": event.to_dict()
            }
            
            # Publish to Redis stream
            await self.redis.xadd(
                self.event_stream,
                event_data
            )
            
            # Notify handlers
            await self._notify_handlers(event)
            
            logger.info(f"Event published: {event.event_type.value} for {event.aggregate_id}")
            
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            raise
    
    async def subscribe(self, event_type: Type[DomainEvent], handler: Any) -> None:
        """Subscribe to an event type."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        
        self.handlers[event_type].append(handler)
        logger.info(f"Handler subscribed to {event_type.__name__}")
    
    async def _notify_handlers(self, event: DomainEvent) -> None:
        """Notify all handlers for an event."""
        handlers = self.handlers.get(type(event), [])
        
        if not handlers:
            return
        
        # Execute handlers concurrently
        tasks = [handler.handle(event) for handler in handlers]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def get_events(self, aggregate_id: str, from_version: int = 0) -> List[DomainEvent]:
        """Get events for an aggregate."""
        events = []
        
        # Get events from Redis stream
        stream_data = await self.redis.xread(
            {self.event_stream: from_version},
            count=100
        )
        
        for stream, messages in stream_data:
            for message_id, data in messages:
                if data[b"aggregate_id"].decode() == aggregate_id:
                    event = self._deserialize_event(data)
                    events.append(event)
        
        return events
    
    def _deserialize_event(self, data: Dict[bytes, bytes]) -> DomainEvent:
        """Deserialize event from Redis data."""
        event_type = data[b"event_type"].decode()
        event_data = json.loads(data[b"data"].decode())
        
        # Map event type to class
        event_classes = {
            "post_created": PostCreatedEvent,
            "post_published": PostPublishedEvent,
            "post_optimized": PostOptimizedEvent,
            "post_engagement_updated": PostEngagementUpdatedEvent,
            "post_deleted": PostDeletedEvent
        }
        
        event_class = event_classes.get(event_type)
        if not event_class:
            raise ValueError(f"Unknown event type: {event_type}")
        
        return event_class.from_dict(event_data)
```

### 🎨 Phase 4: Presentation Layer Implementation

#### 4.1 API Router

**File: `presentation/api/post_router.py`**

```python
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from uuid import UUID

from ...core.application.commands.post_commands import (
    CreatePostCommand, PublishPostCommand, OptimizePostCommand,
    UpdatePostCommand, DeletePostCommand
)
from ...core.application.queries.post_queries import (
    GetPostQuery, ListPostsQuery, GetPostAnalyticsQuery
)
from ...core.application.commands.handlers.post_command_handlers import (
    CreatePostCommandHandler, PublishPostCommandHandler, OptimizePostCommandHandler
)
from ...core.application.queries.handlers.post_query_handlers import (
    GetPostQueryHandler, ListPostsQueryHandler
)
from ...shared.schemas.post_schemas import (
    CreatePostRequest, UpdatePostRequest, PostResponse, PostListResponse
)
from ...shared.dependencies import get_command_handlers, get_query_handlers
from ...shared.exceptions import handle_domain_exception

router = APIRouter(prefix="/posts", tags=["Posts"])

@router.post("/", response_model=PostResponse)
async def create_post(
    request: CreatePostRequest,
    handlers: CreatePostCommandHandler = Depends(get_command_handlers)
):
    """Create a new post."""
    try:
        command = CreatePostCommand(
            content=request.content,
            author_id=request.author_id,
            post_type=request.post_type,
            tone=request.tone,
            target_audience=request.target_audience,
            industry=request.industry
        )
        
        post_id = await handlers.handle(command)
        
        # Get created post
        query_handler = GetPostQueryHandler(handlers.repository, handlers.cache)
        post = await query_handler.handle(GetPostQuery(post_id=post_id))
        
        return PostResponse.from_dto(post)
        
    except Exception as e:
        handle_domain_exception(e)

@router.get("/{post_id}", response_model=PostResponse)
async def get_post(
    post_id: UUID,
    include_metrics: bool = Query(True),
    handlers: GetPostQueryHandler = Depends(get_query_handlers)
):
    """Get a specific post."""
    try:
        query = GetPostQuery(post_id=post_id, include_metrics=include_metrics)
        post = await handlers.handle(query)
        
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        
        return PostResponse.from_dto(post)
        
    except Exception as e:
        handle_domain_exception(e)

@router.get("/", response_model=PostListResponse)
async def list_posts(
    author_id: Optional[UUID] = Query(None),
    status: Optional[str] = Query(None),
    post_type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    handlers: ListPostsQueryHandler = Depends(get_query_handlers)
):
    """List posts with filtering."""
    try:
        query = ListPostsQuery(
            author_id=author_id,
            status=status,
            post_type=post_type,
            limit=limit,
            offset=offset
        )
        
        result = await handlers.handle(query)
        return PostListResponse.from_dto(result)
        
    except Exception as e:
        handle_domain_exception(e)

@router.post("/{post_id}/publish")
async def publish_post(
    post_id: UUID,
    author_id: UUID = Query(...),
    handlers: PublishPostCommandHandler = Depends(get_command_handlers)
):
    """Publish a post."""
    try:
        command = PublishPostCommand(post_id=post_id, author_id=author_id)
        await handlers.handle(command)
        
        return {"message": "Post published successfully"}
        
    except Exception as e:
        handle_domain_exception(e)

@router.post("/{post_id}/optimize")
async def optimize_post(
    post_id: UUID,
    use_nlp: bool = Query(True),
    use_async: bool = Query(False),
    handlers: OptimizePostCommandHandler = Depends(get_command_handlers)
):
    """Optimize a post."""
    try:
        command = OptimizePostCommand(
            post_id=post_id,
            use_nlp=use_nlp,
            use_async=use_async
        )
        await handlers.handle(command)
        
        return {"message": "Post optimized successfully"}
        
    except Exception as e:
        handle_domain_exception(e)
```

### 🧪 Testing Implementation

#### 5.1 Unit Tests

**File: `tests/unit/test_linkedin_post.py`**

```python
import pytest
from datetime import datetime
from uuid import uuid4

from ...core.domain.entities.linkedin_post_refactored import LinkedInPost
from ...core.domain.value_objects.content import Content
from ...core.domain.value_objects.author import Author
from ...core.domain.exceptions.post_exceptions import (
    InvalidPostStateError, ContentValidationError, PostAlreadyPublishedError
)

class TestLinkedInPost:
    """Test cases for LinkedInPost entity."""
    
    def test_create_post(self):
        """Test post creation."""
        content = Content("This is a test post content")
        author = Author(uuid4(), "Test Author")
        
        post = LinkedInPost(content=content, author=author)
        
        assert post.content == content
        assert post.author == author
        assert post.status == PostStatus.DRAFT
        assert post.is_draft()
    
    def test_publish_post(self):
        """Test post publishing."""
        content = Content("This is a test post content")
        author = Author(uuid4(), "Test Author")
        
        post = LinkedInPost(content=content, author=author)
        post.publish()
        
        assert post.status == PostStatus.PUBLISHED
        assert post.is_published()
        assert post.published_at is not None
    
    def test_publish_already_published_post(self):
        """Test publishing an already published post."""
        content = Content("This is a test post content")
        author = Author(uuid4(), "Test Author")
        
        post = LinkedInPost(content=content, author=author)
        post.publish()
        
        with pytest.raises(PostAlreadyPublishedError):
            post.publish()
    
    def test_update_content_published_post(self):
        """Test updating content of published post."""
        content = Content("This is a test post content")
        author = Author(uuid4(), "Test Author")
        
        post = LinkedInPost(content=content, author=author)
        post.publish()
        
        with pytest.raises(InvalidPostStateError):
            post.update_content("New content")
    
    def test_invalid_content(self):
        """Test post creation with invalid content."""
        with pytest.raises(ValueError):
            Content("")  # Empty content
    
    def test_content_too_short(self):
        """Test post creation with content too short."""
        with pytest.raises(ValueError):
            Content("Short")  # Less than 10 characters
```

### 🚀 Deployment and Migration

#### 6.1 Database Migration

**File: `infrastructure/persistence/migrations/001_create_posts_table.sql`**

```sql
-- Create posts table
CREATE TABLE posts (
    id UUID PRIMARY KEY,
    content TEXT NOT NULL,
    author_id UUID NOT NULL,
    post_type VARCHAR(50) NOT NULL,
    tone VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'draft',
    metadata JSONB DEFAULT '{}',
    engagement_metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    published_at TIMESTAMP WITH TIME ZONE,
    version INTEGER DEFAULT 1
);

-- Create indexes
CREATE INDEX idx_posts_author_id ON posts(author_id);
CREATE INDEX idx_posts_status ON posts(status);
CREATE INDEX idx_posts_created_at ON posts(created_at);
CREATE INDEX idx_posts_post_type ON posts(post_type);

-- Create events table for event sourcing
CREATE TABLE post_events (
    id UUID PRIMARY KEY,
    aggregate_id UUID NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL,
    occurred_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    version INTEGER NOT NULL
);

CREATE INDEX idx_post_events_aggregate_id ON post_events(aggregate_id);
CREATE INDEX idx_post_events_occurred_at ON post_events(occurred_at);
```

#### 6.2 Configuration

**File: `shared/config/settings.py`**

```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings."""
    
    # Database
    database_url: str = "postgresql://user:password@localhost/linkedin_posts"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # NLP Services
    openai_api_key: Optional[str] = None
    nlp_model: str = "gpt-3.5-turbo"
    
    # Application
    debug: bool = False
    log_level: str = "INFO"
    
    # Performance
    cache_ttl: int = 3600
    max_concurrent_requests: int = 100
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### 📊 Performance Monitoring

#### 7.1 Metrics Collection

**File: `infrastructure/monitoring/metrics.py`**

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
POST_CREATED = Counter('post_created_total', 'Total posts created')
POST_PUBLISHED = Counter('post_published_total', 'Total posts published')
POST_OPTIMIZED = Counter('post_optimized_total', 'Total posts optimized')

POST_CREATION_DURATION = Histogram('post_creation_duration_seconds', 'Post creation duration')
POST_OPTIMIZATION_DURATION = Histogram('post_optimization_duration_seconds', 'Post optimization duration')

ACTIVE_POSTS = Gauge('active_posts_total', 'Total active posts')
CACHE_HIT_RATE = Gauge('cache_hit_rate', 'Cache hit rate')

def track_post_creation():
    """Track post creation metrics."""
    POST_CREATED.inc()

def track_post_publishing():
    """Track post publishing metrics."""
    POST_PUBLISHED.inc()

def track_post_optimization():
    """Track post optimization metrics."""
    POST_OPTIMIZED.inc()

def measure_post_creation_duration():
    """Measure post creation duration."""
    return POST_CREATION_DURATION.time()

def measure_post_optimization_duration():
    """Measure post optimization duration."""
    return POST_OPTIMIZATION_DURATION.time()

def update_active_posts_count(count: int):
    """Update active posts count."""
    ACTIVE_POSTS.set(count)

def update_cache_hit_rate(rate: float):
    """Update cache hit rate."""
    CACHE_HIT_RATE.set(rate)
```

### 🎯 Next Steps

1. **Implement remaining components** following the patterns shown
2. **Add comprehensive testing** for all layers
3. **Set up CI/CD pipeline** for automated deployment
4. **Configure monitoring and alerting**
5. **Performance testing and optimization**
6. **Documentation and training**

This refactor implementation provides a solid foundation for a scalable, maintainable, and performant LinkedIn posts system following modern architecture patterns. 