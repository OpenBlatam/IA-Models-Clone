from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import json
import datetime
from typing import Dict, Any, List, Optional, Union, Tuple, Callable, TypeVar, Generic, Literal
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator, ConfigDict, computed_field
from pydantic.types import conint, confloat, constr, EmailStr, HttpUrl
from pydantic import EmailStr, HttpUrl, IPvAnyAddress
import numpy as np
from pathlib import Path
from enum import Enum
from typing import Any, List, Dict, Optional
import logging
"""
Types Module Structure - Models, Schemas
=======================================

This file demonstrates how to organize the types module structure:
- Models with type hints and Pydantic validation
- Schemas with async/sync patterns
- Named exports for utilities
"""


# ============================================================================
# NAMED EXPORTS
# ============================================================================

__all__ = [
    # Models
    "UserModel",
    "PostModel",
    "CommentModel",
    "MediaModel",
    "AnalyticsModel",
    
    # Schemas
    "UserSchema",
    "PostSchema", 
    "CommentSchema",
    "MediaSchema",
    "AnalyticsSchema",
    
    # Common utilities
    "TypesResult",
    "TypesConfig",
    "ModelType"
]

# ============================================================================
# COMMON UTILITIES
# ============================================================================

class TypesResult(BaseModel):
    """Pydantic model for types results."""
    
    model_config = ConfigDict(extra="forbid")
    
    is_successful: bool = Field(description="Whether operation was successful")
    operation_type: str = Field(description="Type of operation performed")
    result: Optional[Any] = Field(default=None, description="Operation result")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Operation metadata")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    execution_time: Optional[float] = Field(default=None, description="Execution time in seconds")

class TypesConfig(BaseModel):
    """Pydantic model for types configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    validate_on_assignment: bool = Field(default=True, description="Validate on assignment")
    extra_fields: Literal["ignore", "forbid", "allow"] = Field(default="forbid", description="Extra fields handling")
    str_strip_whitespace: bool = Field(default=True, description="Strip whitespace from strings")
    validate_assignment: bool = Field(default=True, description="Validate on assignment")
    use_enum_values: bool = Field(default=False, description="Use enum values")

class ModelType(BaseModel):
    """Pydantic model for model type validation."""
    
    model_config = ConfigDict(extra="forbid")
    
    type_name: constr(strip_whitespace=True) = Field(
        pattern=r"^(user|post|comment|media|analytics|config|settings)$"
    )
    description: Optional[str] = Field(default=None)
    is_active: bool = Field(default=True)

# ============================================================================
# MODELS
# ============================================================================

class UserModel(BaseModel):
    """User model with comprehensive type hints and validation."""
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=False
    )
    
    # Required fields
    user_id: conint(gt=0) = Field(description="Unique user identifier")
    username: constr(strip_whitespace=True, min_length=3, max_length=50) = Field(
        description="Username (3-50 characters)",
        pattern=r"^[a-zA-Z0-9_]+$"
    )
    email: EmailStr = Field(description="Valid email address")
    
    # Optional fields with defaults
    first_name: Optional[constr(strip_whitespace=True, min_length=1, max_length=100)] = Field(
        default=None,
        description="First name (1-100 characters)"
    )
    last_name: Optional[constr(strip_whitespace=True, min_length=1, max_length=100)] = Field(
        default=None,
        description="Last name (1-100 characters)"
    )
    bio: Optional[constr(strip_whitespace=True, max_length=500)] = Field(
        default=None,
        description="User bio (max 500 characters)"
    )
    profile_picture: Optional[HttpUrl] = Field(
        default=None,
        description="Profile picture URL"
    )
    date_of_birth: Optional[datetime.date] = Field(
        default=None,
        description="Date of birth"
    )
    is_active: bool = Field(default=True, description="User account status")
    is_verified: bool = Field(default=False, description="Email verification status")
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Account creation timestamp"
    )
    updated_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Last update timestamp"
    )
    
    # Computed fields
    @computed_field
    @property
    def full_name(self) -> str:
        """Computed full name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        elif self.first_name:
            return self.first_name
        elif self.last_name:
            return self.last_name
        else:
            return self.username
    
    @computed_field
    @property
    def display_name(self) -> str:
        """Computed display name."""
        return self.full_name or self.username
    
    @computed_field
    @property
    def age(self) -> Optional[int]:
        """Computed age."""
        if self.date_of_birth:
            today = datetime.date.today()
            return today.year - self.date_of_birth.year - (
                (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
            )
        return None
    
    # Custom validators
    @validator('username')
    def validate_username(cls, v: str) -> str:
        """Validate username format."""
        if not v.isalnum() and '_' not in v:
            raise ValueError("Username must contain only letters, numbers, and underscores")
        return v.lower()
    
    @validator('email')
    def validate_email(cls, v: str) -> str:
        """Validate email format."""
        if '@' not in v or '.' not in v:
            raise ValueError("Invalid email format")
        return v.lower()
    
    @validator('date_of_birth')
    def validate_date_of_birth(cls, v: Optional[datetime.date]) -> Optional[datetime.date]:
        """Validate date of birth."""
        if v and v > datetime.date.today():
            raise ValueError("Date of birth cannot be in the future")
        return v

class PostModel(BaseModel):
    """Post model with comprehensive type hints and validation."""
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True
    )
    
    # Required fields
    post_id: conint(gt=0) = Field(description="Unique post identifier")
    author_id: conint(gt=0) = Field(description="Author user ID")
    content: constr(strip_whitespace=True, min_length=1, max_length=10000) = Field(
        description="Post content (1-10000 characters)"
    )
    post_type: Literal["text", "image", "video", "link", "poll"] = Field(
        default="text",
        description="Type of post"
    )
    
    # Optional fields
    title: Optional[constr(strip_whitespace=True, max_length=200)] = Field(
        default=None,
        description="Post title (max 200 characters)"
    )
    media_urls: List[HttpUrl] = Field(
        default_factory=list,
        description="List of media URLs"
    )
    tags: List[constr(strip_whitespace=True, min_length=1, max_length=50)] = Field(
        default_factory=list,
        description="Post tags"
    )
    location: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Location information"
    )
    is_public: bool = Field(default=True, description="Post visibility")
    is_pinned: bool = Field(default=False, description="Pinned post status")
    likes_count: conint(ge=0) = Field(default=0, description="Number of likes")
    comments_count: conint(ge=0) = Field(default=0, description="Number of comments")
    shares_count: conint(ge=0) = Field(default=0, description="Number of shares")
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Post creation timestamp"
    )
    updated_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Last update timestamp"
    )
    
    # Computed fields
    @computed_field
    @property
    def engagement_rate(self) -> float:
        """Computed engagement rate."""
        total_engagement = self.likes_count + self.comments_count + self.shares_count
        return total_engagement / max(self.likes_count, 1)
    
    @computed_field
    @property
    def word_count(self) -> int:
        """Computed word count."""
        return len(self.content.split())
    
    @computed_field
    @property
    def has_media(self) -> bool:
        """Check if post has media."""
        return len(self.media_urls) > 0
    
    # Custom validators
    @validator('content')
    def validate_content(cls, v: str) -> str:
        """Validate content length and format."""
        if len(v.strip()) == 0:
            raise ValueError("Content cannot be empty")
        return v
    
    @validator('tags')
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Validate tags."""
        if len(v) > 10:
            raise ValueError("Maximum 10 tags allowed")
        return [tag.lower() for tag in v if tag.strip()]

class CommentModel(BaseModel):
    """Comment model with comprehensive type hints and validation."""
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True
    )
    
    # Required fields
    comment_id: conint(gt=0) = Field(description="Unique comment identifier")
    post_id: conint(gt=0) = Field(description="Parent post ID")
    author_id: conint(gt=0) = Field(description="Comment author ID")
    content: constr(strip_whitespace=True, min_length=1, max_length=1000) = Field(
        description="Comment content (1-1000 characters)"
    )
    
    # Optional fields
    parent_comment_id: Optional[conint(gt=0)] = Field(
        default=None,
        description="Parent comment ID for replies"
    )
    likes_count: conint(ge=0) = Field(default=0, description="Number of likes")
    is_edited: bool = Field(default=False, description="Edit status")
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Comment creation timestamp"
    )
    updated_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Last update timestamp"
    )
    
    # Computed fields
    @computed_field
    @property
    def is_reply(self) -> bool:
        """Check if comment is a reply."""
        return self.parent_comment_id is not None
    
    @computed_field
    @property
    def word_count(self) -> int:
        """Computed word count."""
        return len(self.content.split())

class MediaModel(BaseModel):
    """Media model with comprehensive type hints and validation."""
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True
    )
    
    # Required fields
    media_id: conint(gt=0) = Field(description="Unique media identifier")
    url: HttpUrl = Field(description="Media URL")
    media_type: Literal["image", "video", "audio", "document"] = Field(
        description="Type of media"
    )
    
    # Optional fields
    filename: Optional[constr(strip_whitespace=True)] = Field(
        default=None,
        description="Original filename"
    )
    file_size: Optional[conint(gt=0)] = Field(
        default=None,
        description="File size in bytes"
    )
    mime_type: Optional[constr(strip_whitespace=True)] = Field(
        default=None,
        description="MIME type"
    )
    duration: Optional[confloat(gt=0.0)] = Field(
        default=None,
        description="Media duration in seconds"
    )
    width: Optional[conint(gt=0)] = Field(
        default=None,
        description="Media width in pixels"
    )
    height: Optional[conint(gt=0)] = Field(
        default=None,
        description="Media height in pixels"
    )
    thumbnail_url: Optional[HttpUrl] = Field(
        default=None,
        description="Thumbnail URL"
    )
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Media creation timestamp"
    )
    
    # Computed fields
    @computed_field
    @property
    def aspect_ratio(self) -> Optional[float]:
        """Computed aspect ratio."""
        if self.width and self.height:
            return self.width / self.height
        return None
    
    @computed_field
    @property
    def file_size_mb(self) -> Optional[float]:
        """File size in MB."""
        if self.file_size:
            return self.file_size / (1024 * 1024)
        return None
    
    @computed_field
    @property
    def is_image(self) -> bool:
        """Check if media is image."""
        return self.media_type == "image"
    
    @computed_field
    @property
    def is_video(self) -> bool:
        """Check if media is video."""
        return self.media_type == "video"

class AnalyticsModel(BaseModel):
    """Analytics model with comprehensive type hints and validation."""
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True
    )
    
    # Required fields
    analytics_id: conint(gt=0) = Field(description="Unique analytics identifier")
    entity_id: conint(gt=0) = Field(description="Entity ID (user, post, etc.)")
    entity_type: Literal["user", "post", "comment", "media"] = Field(
        description="Type of entity"
    )
    metric_name: constr(strip_whitespace=True) = Field(
        description="Metric name"
    )
    metric_value: Union[int, float] = Field(description="Metric value")
    
    # Optional fields
    metric_unit: Optional[constr(strip_whitespace=True)] = Field(
        default=None,
        description="Metric unit"
    )
    time_period: Optional[Literal["hour", "day", "week", "month", "year"]] = Field(
        default=None,
        description="Time period for metric"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    recorded_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Metric recording timestamp"
    )
    
    # Computed fields
    @computed_field
    @property
    def is_numeric(self) -> bool:
        """Check if metric is numeric."""
        return isinstance(self.metric_value, (int, float))
    
    @computed_field
    @property
    def formatted_value(self) -> str:
        """Formatted metric value."""
        if self.metric_unit:
            return f"{self.metric_value} {self.metric_unit}"
        return str(self.metric_value)

# ============================================================================
# SCHEMAS
# ============================================================================

class UserSchema(BaseModel):
    """User schema for API requests/responses."""
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        json_encoders={
            datetime.datetime: lambda v: v.isoformat(),
            datetime.date: lambda v: v.isoformat()
        }
    )
    
    # Create user schema
    class CreateUser(BaseModel):
        username: constr(strip_whitespace=True, min_length=3, max_length=50)
        email: EmailStr
        password: constr(strip_whitespace=True, min_length=8, max_length=128)
        first_name: Optional[constr(strip_whitespace=True, min_length=1, max_length=100)] = None
        last_name: Optional[constr(strip_whitespace=True, min_length=1, max_length=100)] = None
        bio: Optional[constr(strip_whitespace=True, max_length=500)] = None
        date_of_birth: Optional[datetime.date] = None
    
    # Update user schema
    class UpdateUser(BaseModel):
        first_name: Optional[constr(strip_whitespace=True, min_length=1, max_length=100)] = None
        last_name: Optional[constr(strip_whitespace=True, min_length=1, max_length=100)] = None
        bio: Optional[constr(strip_whitespace=True, max_length=500)] = None
        profile_picture: Optional[HttpUrl] = None
        date_of_birth: Optional[datetime.date] = None
    
    # User response schema
    class UserResponse(BaseModel):
        user_id: int
        username: str
        email: str
        first_name: Optional[str] = None
        last_name: Optional[str] = None
        bio: Optional[str] = None
        profile_picture: Optional[str] = None
        full_name: str
        display_name: str
        age: Optional[int] = None
        is_active: bool
        is_verified: bool
        created_at: str
        updated_at: str
    
    # User list response schema
    class UserListResponse(BaseModel):
        users: List[UserResponse]
        total_count: int
        page: int
        page_size: int
        has_next: bool
        has_previous: bool

class PostSchema(BaseModel):
    """Post schema for API requests/responses."""
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        json_encoders={
            datetime.datetime: lambda v: v.isoformat(),
            datetime.date: lambda v: v.isoformat()
        }
    )
    
    # Create post schema
    class CreatePost(BaseModel):
        content: constr(strip_whitespace=True, min_length=1, max_length=10000)
        post_type: Literal["text", "image", "video", "link", "poll"] = "text"
        title: Optional[constr(strip_whitespace=True, max_length=200)] = None
        media_urls: List[HttpUrl] = []
        tags: List[constr(strip_whitespace=True, min_length=1, max_length=50)] = []
        location: Optional[Dict[str, Any]] = None
        is_public: bool = True
    
    # Update post schema
    class UpdatePost(BaseModel):
        content: Optional[constr(strip_whitespace=True, min_length=1, max_length=10000)] = None
        title: Optional[constr(strip_whitespace=True, max_length=200)] = None
        media_urls: Optional[List[HttpUrl]] = None
        tags: Optional[List[constr(strip_whitespace=True, min_length=1, max_length=50)]] = None
        location: Optional[Dict[str, Any]] = None
        is_public: Optional[bool] = None
    
    # Post response schema
    class PostResponse(BaseModel):
        post_id: int
        author_id: int
        content: str
        post_type: str
        title: Optional[str] = None
        media_urls: List[str]
        tags: List[str]
        location: Optional[Dict[str, Any]] = None
        is_public: bool
        is_pinned: bool
        likes_count: int
        comments_count: int
        shares_count: int
        engagement_rate: float
        word_count: int
        has_media: bool
        created_at: str
        updated_at: str
    
    # Post list response schema
    class PostListResponse(BaseModel):
        posts: List[PostResponse]
        total_count: int
        page: int
        page_size: int
        has_next: bool
        has_previous: bool

class CommentSchema(BaseModel):
    """Comment schema for API requests/responses."""
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        json_encoders={
            datetime.datetime: lambda v: v.isoformat(),
            datetime.date: lambda v: v.isoformat()
        }
    )
    
    # Create comment schema
    class CreateComment(BaseModel):
        content: constr(strip_whitespace=True, min_length=1, max_length=1000)
        parent_comment_id: Optional[conint(gt=0)] = None
    
    # Update comment schema
    class UpdateComment(BaseModel):
        content: constr(strip_whitespace=True, min_length=1, max_length=1000)
    
    # Comment response schema
    class CommentResponse(BaseModel):
        comment_id: int
        post_id: int
        author_id: int
        content: str
        parent_comment_id: Optional[int] = None
        likes_count: int
        is_edited: bool
        is_reply: bool
        word_count: int
        created_at: str
        updated_at: str
    
    # Comment list response schema
    class CommentListResponse(BaseModel):
        comments: List[CommentResponse]
        total_count: int
        page: int
        page_size: int
        has_next: bool
        has_previous: bool

class MediaSchema(BaseModel):
    """Media schema for API requests/responses."""
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        json_encoders={
            datetime.datetime: lambda v: v.isoformat(),
            datetime.date: lambda v: v.isoformat()
        }
    )
    
    # Create media schema
    class CreateMedia(BaseModel):
        url: HttpUrl
        media_type: Literal["image", "video", "audio", "document"]
        filename: Optional[constr(strip_whitespace=True)] = None
        file_size: Optional[conint(gt=0)] = None
        mime_type: Optional[constr(strip_whitespace=True)] = None
        duration: Optional[confloat(gt=0.0)] = None
        width: Optional[conint(gt=0)] = None
        height: Optional[conint(gt=0)] = None
        thumbnail_url: Optional[HttpUrl] = None
    
    # Update media schema
    class UpdateMedia(BaseModel):
        filename: Optional[constr(strip_whitespace=True)] = None
        thumbnail_url: Optional[HttpUrl] = None
    
    # Media response schema
    class MediaResponse(BaseModel):
        media_id: int
        url: str
        media_type: str
        filename: Optional[str] = None
        file_size: Optional[int] = None
        mime_type: Optional[str] = None
        duration: Optional[float] = None
        width: Optional[int] = None
        height: Optional[int] = None
        thumbnail_url: Optional[str] = None
        aspect_ratio: Optional[float] = None
        file_size_mb: Optional[float] = None
        is_image: bool
        is_video: bool
        created_at: str
    
    # Media list response schema
    class MediaListResponse(BaseModel):
        media: List[MediaResponse]
        total_count: int
        page: int
        page_size: int
        has_next: bool
        has_previous: bool

class AnalyticsSchema(BaseModel):
    """Analytics schema for API requests/responses."""
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        json_encoders={
            datetime.datetime: lambda v: v.isoformat(),
            datetime.date: lambda v: v.isoformat()
        }
    )
    
    # Create analytics schema
    class CreateAnalytics(BaseModel):
        entity_id: conint(gt=0)
        entity_type: Literal["user", "post", "comment", "media"]
        metric_name: constr(strip_whitespace=True)
        metric_value: Union[int, float]
        metric_unit: Optional[constr(strip_whitespace=True)] = None
        time_period: Optional[Literal["hour", "day", "week", "month", "year"]] = None
        metadata: Optional[Dict[str, Any]] = None
    
    # Analytics response schema
    class AnalyticsResponse(BaseModel):
        analytics_id: int
        entity_id: int
        entity_type: str
        metric_name: str
        metric_value: Union[int, float]
        metric_unit: Optional[str] = None
        time_period: Optional[str] = None
        metadata: Dict[str, Any]
        is_numeric: bool
        formatted_value: str
        recorded_at: str
    
    # Analytics list response schema
    class AnalyticsListResponse(BaseModel):
        analytics: List[AnalyticsResponse]
        total_count: int
        page: int
        page_size: int
        has_next: bool
        has_previous: bool

# ============================================================================
# MAIN TYPES MODULE
# ============================================================================

class MainTypesModule:
    """Main types module with proper imports and exports."""
    
    # Define main exports
    __all__ = [
        # Models
        "UserModel",
        "PostModel",
        "CommentModel",
        "MediaModel",
        "AnalyticsModel",
        
        # Schemas
        "UserSchema",
        "PostSchema",
        "CommentSchema",
        "MediaSchema",
        "AnalyticsSchema",
        
        # Common utilities
        "TypesResult",
        "TypesConfig",
        "ModelType",
        
        # Main functions
        "create_user_model",
        "create_post_model",
        "validate_user_schema",
        "validate_post_schema"
    ]
    
    async def create_user_model(
        user_data: Dict[str, Any],
        config: TypesConfig
    ) -> TypesResult:
        """Create user model with all patterns integrated."""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Validate and create user model
            user_model = UserModel(**user_data)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return TypesResult(
                is_successful=True,
                operation_type="create_user_model",
                result=user_model,
                metadata={
                    "validate_on_assignment": config.validate_on_assignment,
                    "extra_fields": config.extra_fields
                },
                execution_time=execution_time
            )
            
        except Exception as exc:
            return TypesResult(
                is_successful=False,
                operation_type="create_user_model",
                result=None,
                errors=[str(exc)],
                execution_time=None
            )
    
    async def create_post_model(
        post_data: Dict[str, Any],
        config: TypesConfig
    ) -> TypesResult:
        """Create post model with all patterns integrated."""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Validate and create post model
            post_model = PostModel(**post_data)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return TypesResult(
                is_successful=True,
                operation_type="create_post_model",
                result=post_model,
                metadata={
                    "validate_on_assignment": config.validate_on_assignment,
                    "extra_fields": config.extra_fields
                },
                execution_time=execution_time
            )
            
        except Exception as exc:
            return TypesResult(
                is_successful=False,
                operation_type="create_post_model",
                result=None,
                errors=[str(exc)],
                execution_time=None
            )
    
    def validate_user_schema(
        schema_data: Dict[str, Any],
        schema_type: str
    ) -> TypesResult:
        """Validate user schema with all patterns integrated."""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Validate based on schema type
            if schema_type == "create":
                schema = UserSchema.CreateUser(**schema_data)
            elif schema_type == "update":
                schema = UserSchema.UpdateUser(**schema_data)
            elif schema_type == "response":
                schema = UserSchema.UserResponse(**schema_data)
            else:
                raise ValueError(f"Unknown schema type: {schema_type}")
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return TypesResult(
                is_successful=True,
                operation_type="validate_user_schema",
                result=schema,
                metadata={"schema_type": schema_type},
                execution_time=execution_time
            )
            
        except Exception as exc:
            return TypesResult(
                is_successful=False,
                operation_type="validate_user_schema",
                result=None,
                errors=[str(exc)],
                execution_time=None
            )
    
    def validate_post_schema(
        schema_data: Dict[str, Any],
        schema_type: str
    ) -> TypesResult:
        """Validate post schema with all patterns integrated."""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Validate based on schema type
            if schema_type == "create":
                schema = PostSchema.CreatePost(**schema_data)
            elif schema_type == "update":
                schema = PostSchema.UpdatePost(**schema_data)
            elif schema_type == "response":
                schema = PostSchema.PostResponse(**schema_data)
            else:
                raise ValueError(f"Unknown schema type: {schema_type}")
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return TypesResult(
                is_successful=True,
                operation_type="validate_post_schema",
                result=schema,
                metadata={"schema_type": schema_type},
                execution_time=execution_time
            )
            
        except Exception as exc:
            return TypesResult(
                is_successful=False,
                operation_type="validate_post_schema",
                result=None,
                errors=[str(exc)],
                execution_time=None
            )

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

async def demonstrate_types_structure():
    """Demonstrate the types structure with all patterns."""
    
    print("📋 Demonstrating Types Structure with All Patterns")
    print("=" * 60)
    
    # Example 1: User model
    print("\n👤 User Model:")
    user_data = {
        "user_id": 1,
        "username": "john_doe",
        "email": "john.doe@example.com",
        "first_name": "John",
        "last_name": "Doe",
        "bio": "Software developer and tech enthusiast",
        "date_of_birth": datetime.date(1990, 5, 15),
        "is_verified": True
    }
    
    user_model = UserModel(**user_data)
    print(f"User created: {user_model.full_name}")
    print(f"Age: {user_model.age}")
    print(f"Display name: {user_model.display_name}")
    
    # Example 2: Post model
    print("\n📝 Post Model:")
    post_data = {
        "post_id": 1,
        "author_id": 1,
        "content": "This is my first post about technology and innovation!",
        "post_type": "text",
        "title": "My First Post",
        "tags": ["technology", "innovation", "first-post"],
        "is_public": True
    }
    
    post_model = PostModel(**post_data)
    print(f"Post created: {post_model.title}")
    print(f"Word count: {post_model.word_count}")
    print(f"Engagement rate: {post_model.engagement_rate:.2f}")
    print(f"Has media: {post_model.has_media}")
    
    # Example 3: Comment model
    print("\n💬 Comment Model:")
    comment_data = {
        "comment_id": 1,
        "post_id": 1,
        "author_id": 2,
        "content": "Great post! Looking forward to more content.",
        "likes_count": 5
    }
    
    comment_model = CommentModel(**comment_data)
    print(f"Comment created: {comment_model.content[:50]}...")
    print(f"Word count: {comment_model.word_count}")
    print(f"Is reply: {comment_model.is_reply}")
    
    # Example 4: Media model
    print("\n🖼️ Media Model:")
    media_data = {
        "media_id": 1,
        "url": "https://example.com/image.jpg",
        "media_type": "image",
        "filename": "profile_picture.jpg",
        "file_size": 1024000,
        "width": 1920,
        "height": 1080,
        "mime_type": "image/jpeg"
    }
    
    media_model = MediaModel(**media_data)
    print(f"Media created: {media_model.filename}")
    print(f"File size: {media_model.file_size_mb:.2f} MB")
    print(f"Aspect ratio: {media_model.aspect_ratio:.2f}")
    print(f"Is image: {media_model.is_image}")
    
    # Example 5: Analytics model
    print("\n📊 Analytics Model:")
    analytics_data = {
        "analytics_id": 1,
        "entity_id": 1,
        "entity_type": "user",
        "metric_name": "profile_views",
        "metric_value": 150,
        "metric_unit": "views",
        "time_period": "day"
    }
    
    analytics_model = AnalyticsModel(**analytics_data)
    print(f"Analytics created: {analytics_model.metric_name}")
    print(f"Value: {analytics_model.formatted_value}")
    print(f"Is numeric: {analytics_model.is_numeric}")
    
    # Example 6: Schema validation
    print("\n✅ Schema Validation:")
    main_module = MainTypesModule()
    
    # Validate user create schema
    user_create_data = {
        "username": "jane_doe",
        "email": "jane.doe@example.com",
        "password": "securepassword123",
        "first_name": "Jane",
        "last_name": "Doe"
    }
    
    user_schema_result = main_module.validate_user_schema(user_create_data, "create")
    print(f"User schema validation: {user_schema_result.is_successful}")
    
    # Validate post create schema
    post_create_data = {
        "content": "Another great post about technology!",
        "post_type": "text",
        "title": "Technology Post",
        "tags": ["technology", "post"]
    }
    
    post_schema_result = main_module.validate_post_schema(post_create_data, "create")
    print(f"Post schema validation: {post_schema_result.is_successful}")

def show_types_benefits():
    """Show the benefits of types structure."""
    
    benefits = {
        "organization": [
            "Clear separation of models and schemas",
            "Logical grouping of related data structures",
            "Easy to navigate and understand",
            "Scalable architecture for new types"
        ],
        "type_safety": [
            "Type hints throughout all models and schemas",
            "Pydantic validation for data integrity",
            "Computed fields for derived data",
            "Custom validators for business logic"
        ],
        "validation": [
            "Comprehensive input validation",
            "Automatic data transformation",
            "Error handling with clear messages",
            "Schema validation for API requests/responses"
        ],
        "flexibility": [
            "Multiple schema types (create, update, response)",
            "Optional fields with defaults",
            "Computed fields for derived data",
            "Extensible architecture for new fields"
        ]
    }
    
    return benefits

if __name__ == "__main__":
    # Demonstrate types structure
    asyncio.run(demonstrate_types_structure())
    
    benefits = show_types_benefits()
    
    print("\n🎯 Key Types Structure Benefits:")
    for category, items in benefits.items():
        print(f"\n{category.title()}:")
        for item in items:
            print(f"  • {item}")
    
    print("\n✅ Types structure organization completed successfully!") 