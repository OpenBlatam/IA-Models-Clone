from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from fastapi import FastAPI, Query, Path, Body, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
from .declarative_routes import (
from ..utils.optimized_base_model import OptimizedBaseModel
from pydantic import BaseModel, Field, computed_field, field_validator
    from .declarative_routes import get_route_metrics
    import uvicorn
from typing import Any, List, Dict, Optional
import logging
"""
Example FastAPI Application with Declarative Routes
==================================================

This example demonstrates how to use the declarative route system with:
- Clear return type annotations
- Structured request/response models
- Performance monitoring
- Error handling
- OpenAPI documentation
"""



    get_route, post_route, put_route, delete_route, patch_route,
    BaseResponseModel, SuccessResponse, ErrorResponse, PaginatedResponse,
    DeclarativeRouter, RouteMetadata
)

logger = structlog.get_logger(__name__)

# ============================================================================
# Request/Response Models
# ============================================================================

class UserCreateRequest(BaseModel):
    """Request model for user creation."""
    
    name: str = Field(..., min_length=1, max_length=100, description="User name")
    email: str = Field(..., pattern=r"^[^@]+@[^@]+\.[^@]+$", description="User email")
    age: Optional[int] = Field(None, ge=0, le=150, description="User age")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate and clean name."""
        return v.strip().title()
    
    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate and normalize email."""
        return v.strip().lower()

class UserUpdateRequest(BaseModel):
    """Request model for user updates."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="User name")
    email: Optional[str] = Field(None, pattern=r"^[^@]+@[^@]+\.[^@]+$", description="User email")
    age: Optional[int] = Field(None, ge=0, le=150, description="User age")
    preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")

class UserResponse(BaseResponseModel):
    """Response model for user operations."""
    
    success: bool = Field(default=True)
    user_id: str = Field(..., description="User ID")
    name: str = Field(..., description="User name")
    email: str = Field(..., description="User email")
    age: Optional[int] = Field(None, description="User age")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @computed_field
    @property
    def display_name(self) -> str:
        """Computed display name."""
        return f"{self.name} ({self.email})"
    
    @computed_field
    @property
    def is_adult(self) -> bool:
        """Computed adult status."""
        return self.age is not None and self.age >= 18

class UserListResponse(PaginatedResponse[UserResponse]):
    """Paginated response for user lists."""
    pass

class BlogPostCreateRequest(BaseModel):
    """Request model for blog post creation."""
    
    title: str = Field(..., min_length=1, max_length=200, description="Blog post title")
    content: str = Field(..., min_length=10, description="Blog post content")
    author_id: str = Field(..., description="Author ID")
    tags: List[str] = Field(default_factory=list, description="Blog post tags")
    category: Optional[str] = Field(None, description="Blog post category")
    is_published: bool = Field(default=False, description="Publication status")
    
    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate and clean title."""
        return v.strip()
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate content length."""
        if len(v.strip()) < 10:
            raise ValueError("Content must be at least 10 characters long")
        return v.strip()

class BlogPostResponse(BaseResponseModel):
    """Response model for blog post operations."""
    
    success: bool = Field(default=True)
    post_id: str = Field(..., description="Blog post ID")
    title: str = Field(..., description="Blog post title")
    content: str = Field(..., description="Blog post content")
    author_id: str = Field(..., description="Author ID")
    tags: List[str] = Field(default_factory=list, description="Blog post tags")
    category: Optional[str] = Field(None, description="Blog post category")
    is_published: bool = Field(default=False, description="Publication status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    word_count: Optional[int] = Field(None, description="Word count")
    
    @computed_field
    @property
    def slug(self) -> str:
        """Generate URL slug from title."""
        return self.title.lower().replace(" ", "-").replace("_", "-")
    
    @computed_field
    @property
    def excerpt(self) -> str:
        """Generate excerpt from content."""
        return self.content[:150] + "..." if len(self.content) > 150 else self.content

# ============================================================================
# Mock Data Store
# ============================================================================

@dataclass
class MockUser:
    """Mock user data structure."""
    id: str
    name: str
    email: str
    age: Optional[int]
    preferences: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

@dataclass
class MockBlogPost:
    """Mock blog post data structure."""
    id: str
    title: str
    content: str
    author_id: str
    tags: List[str]
    category: Optional[str]
    is_published: bool
    created_at: datetime
    updated_at: datetime
    word_count: int

class MockDataStore:
    """Mock data store for demonstration."""
    
    def __init__(self) -> Any:
        self.users: Dict[str, MockUser] = {}
        self.blog_posts: Dict[str, MockBlogPost] = {}
        self._initialize_mock_data()
    
    def _initialize_mock_data(self) -> Any:
        """Initialize with some mock data."""
        # Create mock users
        for i in range(1, 6):
            user_id = str(uuid.uuid4())
            self.users[user_id] = MockUser(
                id=user_id,
                name=f"User {i}",
                email=f"user{i}@example.com",
                age=20 + i,
                preferences={"theme": "dark", "language": "en"},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        
        # Create mock blog posts
        for i in range(1, 4):
            post_id = str(uuid.uuid4())
            self.blog_posts[post_id] = MockBlogPost(
                id=post_id,
                title=f"Blog Post {i}",
                content=f"This is the content of blog post {i}. It contains multiple sentences to demonstrate the word count functionality and excerpt generation.",
                author_id=list(self.users.keys())[0],
                tags=["example", "demo"],
                category="Technology",
                is_published=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                word_count=20
            )

# Global data store instance
data_store = MockDataStore()

# ============================================================================
# Route Handlers with Clear Return Type Annotations
# ============================================================================

@get_route(
    path="/users",
    response_model=UserListResponse,
    tags=["users"],
    summary="Get all users",
    description="Retrieve a paginated list of all users in the system"
)
async def get_users(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search term"),
    sort_by: Optional[str] = Query("name", description="Sort field"),
    sort_order: str = Query("asc", pattern="^(asc|desc)$", description="Sort order")
) -> UserListResponse:
    """
    Get all users with pagination and filtering.
    
    Args:
        page: Page number for pagination
        per_page: Number of items per page
        search: Optional search term
        sort_by: Field to sort by
        sort_order: Sort order (asc/desc)
        
    Returns:
        UserListResponse: Paginated list of users
    """
    try:
        logger.info(
            "Getting users",
            page=page,
            per_page=per_page,
            search=search,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        # Get all users
        all_users = list(data_store.users.values())
        
        # Apply search filter
        if search:
            search_lower = search.lower()
            all_users = [
                user for user in all_users
                if search_lower in user.name.lower() or search_lower in user.email.lower()
            ]
        
        # Apply sorting
        reverse = sort_order.lower() == "desc"
        if sort_by == "name":
            all_users.sort(key=lambda u: u.name, reverse=reverse)
        elif sort_by == "email":
            all_users.sort(key=lambda u: u.email, reverse=reverse)
        elif sort_by == "age":
            all_users.sort(key=lambda u: u.age or 0, reverse=reverse)
        elif sort_by == "created_at":
            all_users.sort(key=lambda u: u.created_at, reverse=reverse)
        
        # Apply pagination
        total = len(all_users)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_users = all_users[start_idx:end_idx]
        
        # Convert to response models
        user_responses = [
            UserResponse(
                user_id=user.id,
                name=user.name,
                email=user.email,
                age=user.age,
                preferences=user.preferences,
                created_at=user.created_at,
                updated_at=user.updated_at
            )
            for user in paginated_users
        ]
        
        # Calculate pagination info
        pages = (total + per_page - 1) // per_page
        has_next = page < pages
        has_prev = page > 1
        
        logger.info(
            "Successfully retrieved users",
            total=total,
            returned=len(user_responses),
            page=page,
            pages=pages
        )
        
        return UserListResponse(
            success=True,
            data=user_responses,
            pagination={
                "page": page,
                "per_page": per_page,
                "total": total,
                "pages": pages,
                "has_next": has_next,
                "has_prev": has_prev
            }
        )
        
    except Exception as e:
        logger.error("Failed to get users", error=str(e))
        return UserListResponse(
            success=False,
            error=f"Failed to retrieve users: {str(e)}",
            data=[],
            pagination={}
        )

@get_route(
    path="/users/{user_id}",
    response_model=UserResponse,
    tags=["users"],
    summary="Get user by ID",
    description="Retrieve a specific user by their ID"
)
async def get_user(
    user_id: str = Path(..., description="User ID")
) -> UserResponse:
    """
    Get a specific user by ID.
    
    Args:
        user_id: The ID of the user to retrieve
        
    Returns:
        UserResponse: User information
    """
    try:
        logger.info("Getting user", user_id=user_id)
        
        # Get user from data store
        user = data_store.users.get(user_id)
        
        if not user:
            logger.warning("User not found", user_id=user_id)
            return UserResponse(
                success=False,
                error=f"User {user_id} not found",
                user_id=user_id,
                name="",
                email=""
            )
        
        logger.info("Successfully retrieved user", user_id=user_id)
        
        return UserResponse(
            success=True,
            user_id=user.id,
            name=user.name,
            email=user.email,
            age=user.age,
            preferences=user.preferences,
            created_at=user.created_at,
            updated_at=user.updated_at
        )
        
    except Exception as e:
        logger.error("Failed to get user", user_id=user_id, error=str(e))
        return UserResponse(
            success=False,
            error=f"Failed to retrieve user: {str(e)}",
            user_id=user_id,
            name="",
            email=""
        )

@post_route(
    path="/users",
    response_model=UserResponse,
    status_code=201,
    tags=["users"],
    summary="Create new user",
    description="Create a new user in the system"
)
async def create_user(
    user_data: UserCreateRequest = Body(..., description="User data")
) -> UserResponse:
    """
    Create a new user.
    
    Args:
        user_data: User creation data
        
    Returns:
        UserResponse: Created user information
    """
    try:
        logger.info("Creating user", name=user_data.name, email=user_data.email)
        
        # Check if email already exists
        existing_user = next(
            (user for user in data_store.users.values() if user.email == user_data.email),
            None
        )
        
        if existing_user:
            logger.warning("Email already exists", email=user_data.email)
            return UserResponse(
                success=False,
                error=f"Email {user_data.email} already exists",
                user_id="",
                name="",
                email=""
            )
        
        # Create new user
        user_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        new_user = MockUser(
            id=user_id,
            name=user_data.name,
            email=user_data.email,
            age=user_data.age,
            preferences=user_data.preferences,
            created_at=now,
            updated_at=now
        )
        
        data_store.users[user_id] = new_user
        
        logger.info("Successfully created user", user_id=user_id)
        
        return UserResponse(
            success=True,
            user_id=new_user.id,
            name=new_user.name,
            email=new_user.email,
            age=new_user.age,
            preferences=new_user.preferences,
            created_at=new_user.created_at,
            updated_at=new_user.updated_at
        )
        
    except Exception as e:
        logger.error("Failed to create user", error=str(e))
        return UserResponse(
            success=False,
            error=f"Failed to create user: {str(e)}",
            user_id="",
            name="",
            email=""
        )

@put_route(
    path="/users/{user_id}",
    response_model=UserResponse,
    tags=["users"],
    summary="Update user",
    description="Update an existing user's information"
)
async def update_user(
    user_id: str = Path(..., description="User ID"),
    user_data: UserCreateRequest = Body(..., description="Updated user data")
) -> UserResponse:
    """
    Update an existing user.
    
    Args:
        user_id: The ID of the user to update
        user_data: Updated user data
        
    Returns:
        UserResponse: Updated user information
    """
    try:
        logger.info("Updating user", user_id=user_id)
        
        # Get existing user
        existing_user = data_store.users.get(user_id)
        
        if not existing_user:
            logger.warning("User not found for update", user_id=user_id)
            return UserResponse(
                success=False,
                error=f"User {user_id} not found",
                user_id=user_id,
                name="",
                email=""
            )
        
        # Check if email is being changed and if it already exists
        if user_data.email != existing_user.email:
            email_exists = next(
                (user for user in data_store.users.values() 
                 if user.email == user_data.email and user.id != user_id),
                None
            )
            
            if email_exists:
                logger.warning("Email already exists", email=user_data.email)
                return UserResponse(
                    success=False,
                    error=f"Email {user_data.email} already exists",
                    user_id=user_id,
                    name="",
                    email=""
                )
        
        # Update user
        existing_user.name = user_data.name
        existing_user.email = user_data.email
        existing_user.age = user_data.age
        existing_user.preferences = user_data.preferences
        existing_user.updated_at = datetime.utcnow()
        
        logger.info("Successfully updated user", user_id=user_id)
        
        return UserResponse(
            success=True,
            user_id=existing_user.id,
            name=existing_user.name,
            email=existing_user.email,
            age=existing_user.age,
            preferences=existing_user.preferences,
            created_at=existing_user.created_at,
            updated_at=existing_user.updated_at
        )
        
    except Exception as e:
        logger.error("Failed to update user", user_id=user_id, error=str(e))
        return UserResponse(
            success=False,
            error=f"Failed to update user: {str(e)}",
            user_id=user_id,
            name="",
            email=""
        )

@delete_route(
    path="/users/{user_id}",
    response_model=SuccessResponse,
    status_code=204,
    tags=["users"],
    summary="Delete user",
    description="Delete a user from the system"
)
async def delete_user(
    user_id: str = Path(..., description="User ID")
) -> SuccessResponse:
    """
    Delete a user.
    
    Args:
        user_id: The ID of the user to delete
        
    Returns:
        SuccessResponse: Deletion confirmation
    """
    try:
        logger.info("Deleting user", user_id=user_id)
        
        # Check if user exists
        if user_id not in data_store.users:
            logger.warning("User not found for deletion", user_id=user_id)
            return SuccessResponse(
                success=False,
                error=f"User {user_id} not found"
            )
        
        # Delete user
        del data_store.users[user_id]
        
        logger.info("Successfully deleted user", user_id=user_id)
        
        return SuccessResponse(
            success=True,
            message=f"User {user_id} deleted successfully"
        )
        
    except Exception as e:
        logger.error("Failed to delete user", user_id=user_id, error=str(e))
        return SuccessResponse(
            success=False,
            error=f"Failed to delete user: {str(e)}"
        )

@post_route(
    path="/blog-posts",
    response_model=BlogPostResponse,
    status_code=201,
    tags=["blog"],
    summary="Create blog post",
    description="Create a new blog post"
)
async def create_blog_post(
    post_data: BlogPostCreateRequest = Body(..., description="Blog post data"),
    background_tasks: BackgroundTasks = Depends()
) -> BlogPostResponse:
    """
    Create a new blog post.
    
    Args:
        post_data: Blog post creation data
        background_tasks: Background tasks for async processing
        
    Returns:
        BlogPostResponse: Created blog post information
    """
    try:
        logger.info("Creating blog post", title=post_data.title, author_id=post_data.author_id)
        
        # Validate author exists
        if post_data.author_id not in data_store.users:
            logger.warning("Author not found", author_id=post_data.author_id)
            return BlogPostResponse(
                success=False,
                error=f"Author {post_data.author_id} not found",
                post_id="",
                title="",
                content="",
                author_id=""
            )
        
        # Calculate word count
        word_count = len(post_data.content.split())
        
        # Create blog post
        post_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        new_post = MockBlogPost(
            id=post_id,
            title=post_data.title,
            content=post_data.content,
            author_id=post_data.author_id,
            tags=post_data.tags,
            category=post_data.category,
            is_published=post_data.is_published,
            created_at=now,
            updated_at=now,
            word_count=word_count
        )
        
        data_store.blog_posts[post_id] = new_post
        
        # Add background task for processing
        background_tasks.add_task(
            _process_blog_post_async,
            post_id
        )
        
        logger.info("Successfully created blog post", post_id=post_id)
        
        return BlogPostResponse(
            success=True,
            post_id=new_post.id,
            title=new_post.title,
            content=new_post.content,
            author_id=new_post.author_id,
            tags=new_post.tags,
            category=new_post.category,
            is_published=new_post.is_published,
            created_at=new_post.created_at,
            updated_at=new_post.updated_at,
            word_count=new_post.word_count
        )
        
    except Exception as e:
        logger.error("Failed to create blog post", error=str(e))
        return BlogPostResponse(
            success=False,
            error=f"Failed to create blog post: {str(e)}",
            post_id="",
            title="",
            content="",
            author_id=""
        )

async def _process_blog_post_async(post_id: str):
    """Background task for blog post processing."""
    try:
        logger.info("Processing blog post", post_id=post_id)
        # Simulate async processing
        await asyncio.sleep(1)
        logger.info("Successfully processed blog post", post_id=post_id)
    except Exception as e:
        logger.error("Failed to process blog post", post_id=post_id, error=str(e))

@get_route(
    path="/health",
    response_model=SuccessResponse,
    tags=["health"],
    summary="Health check",
    description="Check if the service is healthy"
)
async def health_check() -> SuccessResponse:
    """
    Health check endpoint.
    
    Returns:
        SuccessResponse: Health status
    """
    return SuccessResponse(
        success=True,
        message="Service is healthy",
        data={
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "users_count": len(data_store.users),
            "blog_posts_count": len(data_store.blog_posts)
        }
    )

@get_route(
    path="/metrics",
    response_model=Dict[str, Any],
    tags=["monitoring"],
    summary="Get route metrics",
    description="Get performance metrics for all routes"
)
async def get_metrics() -> Dict[str, Any]:
    """
    Get route performance metrics.
    
    Returns:
        Dict[str, Any]: Route metrics
    """
    
    return {
        "route_metrics": get_route_metrics(),
        "timestamp": datetime.utcnow().isoformat(),
        "data_store_stats": {
            "users_count": len(data_store.users),
            "blog_posts_count": len(data_store.blog_posts)
        }
    }

# ============================================================================
# FastAPI Application Setup
# ============================================================================

def create_example_app() -> FastAPI:
    """Create the example FastAPI application."""
    
    app = FastAPI(
        title="Declarative Routes Example API",
        description="Example API demonstrating declarative routes with clear return type annotations",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Create declarative router
    router = DeclarativeRouter(prefix="/api/v1", tags=["api"])
    
    # Register routes
    router.register_route(get_users._route_metadata, get_users)
    router.register_route(get_user._route_metadata, get_user)
    router.register_route(create_user._route_metadata, create_user)
    router.register_route(update_user._route_metadata, update_user)
    router.register_route(delete_user._route_metadata, delete_user)
    router.register_route(create_blog_post._route_metadata, create_blog_post)
    router.register_route(health_check._route_metadata, health_check)
    router.register_route(get_metrics._route_metadata, get_metrics)
    
    # Include router
    app.include_router(router.get_router())
    
    # Add root endpoint
    @app.get("/", tags=["root"])
    async def root() -> Dict[str, str]:
        """Root endpoint."""
        return {
            "message": "Declarative Routes Example API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health"
        }
    
    return app

# Create app instance
app = create_example_app()

match __name__:
    case "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 