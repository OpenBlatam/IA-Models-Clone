"""
Pydantic schemas for request/response validation
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, validator, computed_field, ConfigDict


class PostStatus(str, Enum):
    """Blog post status enumeration."""
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    SCHEDULED = "scheduled"
    REVIEW = "review"


class PostCategory(str, Enum):
    """Blog post category enumeration."""
    TECHNOLOGY = "technology"
    SCIENCE = "science"
    BUSINESS = "business"
    LIFESTYLE = "lifestyle"
    TRAVEL = "travel"
    FOOD = "food"
    HEALTH = "health"
    EDUCATION = "education"
    ENTERTAINMENT = "entertainment"
    AI_ML = "ai_ml"
    BLOCKCHAIN = "blockchain"
    OTHER = "other"


# Base Models
class BlogPostBase(BaseModel):
    """Base blog post model with common fields."""
    title: str = Field(..., min_length=1, max_length=500, description="Post title")
    content: str = Field(..., min_length=1, description="Post content")
    excerpt: Optional[str] = Field(None, max_length=1000, description="Post excerpt")
    category: PostCategory = Field(default=PostCategory.OTHER, description="Post category")
    tags: List[str] = Field(default_factory=list, description="Post tags")
    seo_title: Optional[str] = Field(None, max_length=500, description="SEO title")
    seo_description: Optional[str] = Field(None, max_length=1000, description="SEO description")
    seo_keywords: List[str] = Field(default_factory=list, description="SEO keywords")
    featured_image_url: Optional[str] = Field(None, description="Featured image URL")
    scheduled_at: Optional[datetime] = Field(None, description="Scheduled publication time")
    
    @validator('title')
    def validate_title(cls, v):
        """Validate title is not empty after stripping."""
        if not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip()
    
    @validator('content')
    def validate_content(cls, v):
        """Validate content is not empty after stripping."""
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()
    
    @validator('tags')
    def validate_tags(cls, v):
        """Validate tags are unique and not empty."""
        if v:
            # Remove duplicates and empty strings
            v = list(set(tag.strip() for tag in v if tag.strip()))
        return v


class BlogPostCreate(BlogPostBase):
    """Model for creating a new blog post."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Getting Started with FastAPI",
                "content": "FastAPI is a modern, fast web framework for building APIs with Python...",
                "excerpt": "Learn how to build modern APIs with FastAPI",
                "category": "technology",
                "tags": ["python", "fastapi", "web-development"],
                "seo_title": "FastAPI Tutorial - Build Modern APIs",
                "seo_description": "Complete guide to building APIs with FastAPI",
                "seo_keywords": ["fastapi", "python", "api", "tutorial"]
            }
        }
    )


class BlogPostUpdate(BaseModel):
    """Model for updating an existing blog post."""
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    content: Optional[str] = Field(None, min_length=1)
    excerpt: Optional[str] = Field(None, max_length=1000)
    status: Optional[PostStatus] = None
    category: Optional[PostCategory] = None
    tags: Optional[List[str]] = None
    seo_title: Optional[str] = Field(None, max_length=500)
    seo_description: Optional[str] = Field(None, max_length=1000)
    seo_keywords: Optional[List[str]] = None
    featured_image_url: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    
    @validator('title')
    def validate_title(cls, v):
        """Validate title if provided."""
        if v is not None and not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip() if v else v
    
    @validator('content')
    def validate_content(cls, v):
        """Validate content if provided."""
        if v is not None and not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip() if v else v


class BlogPostResponse(BlogPostBase):
    """Model for blog post response."""
    id: int
    uuid: str
    slug: str
    author_id: str
    status: PostStatus
    view_count: int
    like_count: int
    share_count: int
    comment_count: int
    word_count: int
    reading_time_minutes: int
    created_at: datetime
    updated_at: datetime
    published_at: Optional[datetime]
    scheduled_at: Optional[datetime]
    sentiment_score: Optional[float]
    readability_score: Optional[float]
    topic_tags: List[str]
    
    model_config = ConfigDict(from_attributes=True)


class BlogPostListResponse(BaseModel):
    """Model for blog post list response."""
    id: int
    uuid: str
    title: str
    slug: str
    excerpt: Optional[str]
    author_id: str
    status: PostStatus
    category: PostCategory
    tags: List[str]
    view_count: int
    like_count: int
    comment_count: int
    reading_time_minutes: int
    featured_image_url: Optional[str]
    created_at: datetime
    published_at: Optional[datetime]
    
    model_config = ConfigDict(from_attributes=True)


# User Models
class UserBase(BaseModel):
    """Base user model."""
    email: str = Field(..., description="User email address")
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    full_name: Optional[str] = Field(None, max_length=255, description="Full name")
    bio: Optional[str] = Field(None, max_length=1000, description="User bio")
    website_url: Optional[str] = Field(None, description="Website URL")
    
    @validator('email')
    def validate_email(cls, v):
        """Validate email format."""
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower().strip()
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username format."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username can only contain letters, numbers, hyphens, and underscores')
        return v.lower().strip()


class UserCreate(UserBase):
    """Model for creating a new user."""
    password: str = Field(..., min_length=8, description="Password")
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class UserUpdate(BaseModel):
    """Model for updating user information."""
    full_name: Optional[str] = Field(None, max_length=255)
    bio: Optional[str] = Field(None, max_length=1000)
    website_url: Optional[str] = None
    avatar_url: Optional[str] = None


class UserResponse(UserBase):
    """Model for user response."""
    id: str
    is_active: bool
    is_verified: bool
    roles: List[str]
    avatar_url: Optional[str]
    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime]
    
    model_config = ConfigDict(from_attributes=True)


# Comment Models
class CommentBase(BaseModel):
    """Base comment model."""
    content: str = Field(..., min_length=1, max_length=2000, description="Comment content")
    parent_id: Optional[int] = Field(None, description="Parent comment ID for replies")
    
    @validator('content')
    def validate_content(cls, v):
        """Validate comment content."""
        if not v.strip():
            raise ValueError('Comment content cannot be empty')
        return v.strip()


class CommentCreate(CommentBase):
    """Model for creating a new comment."""
    pass


class CommentResponse(CommentBase):
    """Model for comment response."""
    id: int
    uuid: str
    post_id: int
    author_id: str
    is_approved: bool
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


# Pagination Models
class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(default=1, ge=1, description="Page number")
    size: int = Field(default=20, ge=1, le=100, description="Page size")
    
    @property
    def offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.size


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int
    
    @computed_field
    @property
    def has_next(self) -> bool:
        """Check if there's a next page."""
        return self.page < self.pages
    
    @computed_field
    @property
    def has_prev(self) -> bool:
        """Check if there's a previous page."""
        return self.page > 1


# Search Models
class SearchParams(BaseModel):
    """Search parameters."""
    query: str = Field(..., min_length=1, max_length=200, description="Search query")
    category: Optional[PostCategory] = Field(None, description="Filter by category")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    author_id: Optional[str] = Field(None, description="Filter by author")
    status: Optional[PostStatus] = Field(None, description="Filter by status")
    date_from: Optional[datetime] = Field(None, description="Filter from date")
    date_to: Optional[datetime] = Field(None, description="Filter to date")
    sort_by: str = Field(default="created_at", description="Sort field")
    sort_order: str = Field(default="desc", description="Sort order (asc/desc)")
    
    @validator('sort_order')
    def validate_sort_order(cls, v):
        """Validate sort order."""
        if v.lower() not in ['asc', 'desc']:
            raise ValueError('Sort order must be "asc" or "desc"')
        return v.lower()


# Error Models
class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(description="Error message")
    error_code: Optional[str] = Field(description="Error code")
    detail: Optional[str] = Field(description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    path: Optional[str] = Field(description="Request path")
    request_id: Optional[str] = Field(description="Request ID for tracking")


# Health Check Models
class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    services: Dict[str, bool] = Field(description="Service health status")
    version: str = Field(description="API version")
    uptime: Optional[float] = Field(description="Service uptime in seconds")


# File Upload Models
class FileUploadResponse(BaseModel):
    """File upload response model."""
    id: int
    uuid: str
    filename: str
    original_filename: str
    file_size: int
    mime_type: str
    url: str
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)






























