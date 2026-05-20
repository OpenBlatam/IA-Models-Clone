# FastAPI Route Structure and Dependencies Pattern

## Overview

This guide covers best practices for structuring FastAPI routes and dependencies following official FastAPI documentation for Data Models, Path Operations, and Middleware.

## Key Principles

### 1. **Data Models (Pydantic)**
- Use Pydantic BaseModel for request/response validation
- Separate input and output models
- Use Field for validation and documentation
- Implement model inheritance for common fields

### 2. **Path Operations**
- Group related endpoints in routers
- Use descriptive HTTP methods and status codes
- Implement proper error handling with HTTPException
- Use dependency injection for shared resources

### 3. **Dependencies**
- Create reusable dependency functions
- Use dependency injection for database connections
- Implement authentication and authorization dependencies
- Use dependency overrides for testing

### 4. **Middleware**
- Implement logging and monitoring middleware
- Use CORS middleware for cross-origin requests
- Add security headers middleware
- Implement rate limiting middleware

## Data Models Structure

### Base Models
```python
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class PostType(str, Enum):
    EDUCATIONAL = "educational"
    PROMOTIONAL = "promotional"
    PERSONAL = "personal"
    INDUSTRY = "industry"

class Tone(str, Enum):
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    ENTHUSIASTIC = "enthusiastic"
    THOUGHTFUL = "thoughtful"

class BasePostModel(BaseModel):
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        validate_assignment=True,
        extra="forbid"
    )

class PostCreate(BasePostModel):
    content: str = Field(..., min_length=10, max_length=3000, description="Post content")
    post_type: PostType = Field(default=PostType.EDUCATIONAL, description="Type of post")
    tone: Tone = Field(default=Tone.PROFESSIONAL, description="Tone of the post")
    target_audience: str = Field(default="general", description="Target audience")
    hashtags: Optional[List[str]] = Field(default_factory=list, max_items=10)
    call_to_action: Optional[str] = Field(None, max_length=200)

class PostUpdate(BasePostModel):
    content: Optional[str] = Field(None, min_length=10, max_length=3000)
    post_type: Optional[PostType] = None
    tone: Optional[Tone] = None
    target_audience: Optional[str] = None
    hashtags: Optional[List[str]] = Field(None, max_items=10)
    call_to_action: Optional[str] = Field(None, max_length=200)

class PostResponse(BasePostModel):
    id: str
    content: str
    post_type: PostType
    tone: Tone
    target_audience: str
    user_id: str
    hashtags: List[str]
    call_to_action: Optional[str]
    sentiment_score: Optional[float]
    readability_score: Optional[float]
    engagement_prediction: Optional[float]
    views_count: int = 0
    likes_count: int = 0
    comments_count: int = 0
    shares_count: int = 0
    status: str
    created_at: datetime
    updated_at: datetime

class PostListResponse(BasePostModel):
    posts: List[PostResponse]
    total: int
    page: int
    size: int
    has_next: bool
    has_prev: bool

class ErrorResponse(BasePostModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

## Dependency Injection Structure

### Database Dependencies
```python
from fastapi import Depends
from typing import AsyncGenerator
import asyncpg

async def get_database_pool() -> AsyncGenerator[asyncpg.Pool, None]:
    """Database connection pool dependency"""
    pool = await asyncpg.create_pool(
        "postgresql://user:pass@localhost/linkedin_posts",
        min_size=5,
        max_size=20
    )
    try:
        yield pool
    finally:
        await pool.close()

async def get_database_connection(
    pool: asyncpg.Pool = Depends(get_database_pool)
) -> AsyncGenerator[asyncpg.Connection, None]:
    """Database connection dependency"""
    async with pool.acquire() as connection:
        yield connection
```

### Authentication Dependencies
```python
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Dict[str, Any]:
    """Get current authenticated user"""
    try:
        payload = jwt.decode(
            credentials.credentials, 
            "secret_key", 
            algorithms=["HS256"]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_active_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get current active user"""
    if not current_user.get("is_active"):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
```

### Service Dependencies
```python
from services.linkedin_service import LinkedInService
from services.ai_service import AIService
from services.notification_service import NotificationService

def get_linkedin_service(
    db_pool: asyncpg.Pool = Depends(get_database_pool)
) -> LinkedInService:
    """LinkedIn service dependency"""
    return LinkedInService(db_pool)

def get_ai_service() -> AIService:
    """AI service dependency"""
    return AIService()

def get_notification_service() -> NotificationService:
    """Notification service dependency"""
    return NotificationService()
```

## Router Structure

### Main Router
```python
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

router = APIRouter(prefix="/api/v1", tags=["linkedin-posts"])

@router.post(
    "/posts",
    response_model=PostResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new LinkedIn post",
    description="Create a new LinkedIn post with AI analysis and optimization"
)
async def create_post(
    post_data: PostCreate,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    linkedin_service: LinkedInService = Depends(get_linkedin_service),
    ai_service: AIService = Depends(get_ai_service)
) -> PostResponse:
    """Create a new LinkedIn post"""
    try:
        post = await linkedin_service.create_post(post_data, current_user["id"])
        return PostResponse(**post)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create post: {str(e)}"
        )

@router.get(
    "/posts/{post_id}",
    response_model=PostResponse,
    summary="Get a specific post",
    description="Retrieve a LinkedIn post by its ID"
)
async def get_post(
    post_id: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    linkedin_service: LinkedInService = Depends(get_linkedin_service)
) -> PostResponse:
    """Get a specific post by ID"""
    post = await linkedin_service.get_post(post_id, current_user["id"])
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found"
        )
    return PostResponse(**post)

@router.put(
    "/posts/{post_id}",
    response_model=PostResponse,
    summary="Update a post",
    description="Update an existing LinkedIn post"
)
async def update_post(
    post_id: str,
    post_update: PostUpdate,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    linkedin_service: LinkedInService = Depends(get_linkedin_service)
) -> PostResponse:
    """Update a post"""
    post = await linkedin_service.update_post(
        post_id, 
        post_update, 
        current_user["id"]
    )
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found"
        )
    return PostResponse(**post)

@router.delete(
    "/posts/{post_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a post",
    description="Delete a LinkedIn post"
)
async def delete_post(
    post_id: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    linkedin_service: LinkedInService = Depends(get_linkedin_service)
):
    """Delete a post"""
    success = await linkedin_service.delete_post(post_id, current_user["id"])
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found"
        )

@router.get(
    "/posts",
    response_model=PostListResponse,
    summary="List posts",
    description="Get paginated list of LinkedIn posts"
)
async def list_posts(
    page: int = 1,
    size: int = 10,
    post_type: Optional[PostType] = None,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    linkedin_service: LinkedInService = Depends(get_linkedin_service)
) -> PostListResponse:
    """List posts with pagination"""
    posts, total = await linkedin_service.list_posts(
        user_id=current_user["id"],
        page=page,
        size=size,
        post_type=post_type
    )
    
    return PostListResponse(
        posts=[PostResponse(**post) for post in posts],
        total=total,
        page=page,
        size=size,
        has_next=page * size < total,
        has_prev=page > 1
    )
```

## Middleware Structure

### Logging Middleware
```python
import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logging.info(f"Request: {request.method} {request.url}")
        
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logging.info(
            f"Response: {response.status_code} - {process_time:.3f}s"
        )
        
        return response
```

### CORS Middleware
```python
from fastapi.middleware.cors import CORSMiddleware

def add_cors_middleware(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "https://yourdomain.com"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
```

### Security Middleware
```python
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

def add_security_middleware(app):
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "yourdomain.com"]
    )
    
    # Only in production
    if app.debug is False:
        app.add_middleware(HTTPSRedirectMiddleware)
```

## Application Structure

### Main Application
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting up LinkedIn Posts API")
    yield
    # Shutdown
    logging.info("Shutting down LinkedIn Posts API")

app = FastAPI(
    title="LinkedIn Posts API",
    description="API for managing LinkedIn posts with AI optimization",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
add_cors_middleware(app)
add_security_middleware(app)

# Include routers
app.include_router(router, prefix="/api/v1")
```

## Error Handling

### Custom Exception Handler
```python
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "message": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "Invalid request data",
            "details": exc.errors(),
            "timestamp": datetime.utcnow().isoformat()
        }
    )
```

## Testing Dependencies

### Test Dependencies
```python
from fastapi.testclient import TestClient
from unittest.mock import Mock

def get_test_database():
    """Test database dependency"""
    return Mock()

def get_test_linkedin_service():
    """Test LinkedIn service dependency"""
    return Mock()

# Override dependencies for testing
app.dependency_overrides[get_database_pool] = get_test_database
app.dependency_overrides[get_linkedin_service] = get_test_linkedin_service
```

## Best Practices Summary

1. **Use Pydantic models for all data validation**
2. **Separate input and output models**
3. **Use dependency injection for shared resources**
4. **Group related endpoints in routers**
5. **Implement proper error handling**
6. **Use middleware for cross-cutting concerns**
7. **Add comprehensive logging**
8. **Implement security headers**
9. **Use descriptive HTTP status codes**
10. **Add proper API documentation** 