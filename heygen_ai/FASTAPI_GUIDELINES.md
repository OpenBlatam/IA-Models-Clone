# FastAPI-Specific Guidelines for HeyGen AI API

Comprehensive guidelines for building robust, scalable, and maintainable FastAPI applications with modern best practices and optimizations.

## 🚀 Overview

FastAPI provides:
- **High Performance**: Built on Starlette and Pydantic
- **Type Safety**: Full Python type hints support
- **Automatic Documentation**: OpenAPI/Swagger generation
- **Async Support**: Native async/await throughout
- **Modern Python**: Python 3.7+ features
- **Production Ready**: Built-in security, validation, and testing

## 📋 Table of Contents

1. [Project Structure](#project-structure)
2. [Application Setup](#application-setup)
3. [Dependency Injection](#dependency-injection)
4. [Request/Response Models](#requestresponse-models)
5. [Error Handling](#error-handling)
6. [Authentication & Security](#authentication--security)
7. [Database Integration](#database-integration)
8. [Background Tasks](#background-tasks)
9. [Testing](#testing)
10. [Performance Optimization](#performance-optimization)
11. [Deployment](#deployment)
12. [Monitoring](#monitoring)

## 🏗️ Project Structure

### Recommended Structure

```
heygen_ai/
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app instance
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py         # Pydantic settings
│   │   └── database.py         # Database config
│   ├── core/
│   │   ├── __init__.py
│   │   ├── security.py         # Authentication
│   │   ├── database.py         # Database session
│   │   └── dependencies.py     # Common dependencies
│   ├── models/
│   │   ├── __init__.py
│   │   ├── sqlalchemy_models.py # SQLAlchemy models
│   │   └── pydantic_models.py  # Pydantic schemas
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── user.py             # User schemas
│   │   ├── video.py            # Video schemas
│   │   └── common.py           # Common schemas
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── endpoints/
│   │   │   │   ├── users.py
│   │   │   │   ├── videos.py
│   │   │   │   └── auth.py
│   │   │   └── api.py          # API router
│   │   └── deps.py             # Dependencies
│   ├── services/
│   │   ├── __init__.py
│   │   ├── user_service.py
│   │   └── video_service.py
│   └── utils/
│       ├── __init__.py
│       ├── security.py
│       └── helpers.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api/
│   └── test_services/
├── alembic/
│   ├── versions/
│   └── env.py
├── requirements.txt
├── main.py
└── README.md
```

## 🔧 Application Setup

### Main Application Factory

```python
# api/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import structlog

from .config.settings import get_settings
from .api.v1.api import api_router
from .core.database import init_db, close_db

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting HeyGen AI API...")
    await init_db()
    logger.info("✓ Database initialized")
    logger.info("🚀 HeyGen AI API started successfully!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down HeyGen AI API...")
    await close_db()
    logger.info("✓ HeyGen AI API shutdown complete")

def create_application() -> FastAPI:
    """Create FastAPI application with all configurations."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="HeyGen AI API - Advanced Video Generation",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Include routers
    app.include_router(api_router, prefix="/api/v1")
    
    # Health check
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    return app

app = create_application()
```

### Settings Configuration

```python
# api/config/settings.py
from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache

class Settings(BaseSettings):
    # Application
    PROJECT_NAME: str = "HeyGen AI API"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database
    DATABASE_URL: str
    
    # CORS
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    HEYGEN_API_KEY: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

## 🔄 Dependency Injection

### Database Dependencies

```python
# api/core/dependencies.py
from typing import AsyncGenerator
from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_session
from .security import get_current_user
from ..models.sqlalchemy_models import User

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Database session dependency."""
    async with get_session() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

async def get_current_superuser(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Get current superuser."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The user doesn't have enough privileges"
        )
    return current_user
```

### Service Dependencies

```python
# api/services/user_service.py
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import Depends

from ..core.dependencies import get_db
from ..models.sqlalchemy_models import User
from ..schemas.user import UserCreate, UserUpdate

class UserService:
    def __init__(self, db: AsyncSession = Depends(get_db)):
        self.db = db
    
    async def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        result = await self.db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        result = await self.db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()
    
    async def create_user(self, user_create: UserCreate) -> User:
        """Create new user."""
        user = User(**user_create.dict())
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user
    
    async def update_user(self, user_id: int, user_update: UserUpdate) -> Optional[User]:
        """Update user."""
        user = await self.get_user(user_id)
        if not user:
            return None
        
        for field, value in user_update.dict(exclude_unset=True).items():
            setattr(user, field, value)
        
        await self.db.commit()
        await self.db.refresh(user)
        return user

# Dependency factory
def get_user_service(db: AsyncSession = Depends(get_db)) -> UserService:
    return UserService(db)
```

## 📝 Request/Response Models

### Pydantic Schemas

```python
# api/schemas/user.py
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, validator

class UserBase(BaseModel):
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = True
    is_superuser: bool = False
    full_name: Optional[str] = None

class UserCreate(UserBase):
    email: EmailStr
    password: str = Field(..., min_length=8)
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v

class UserUpdate(UserBase):
    password: Optional[str] = Field(None, min_length=8)

class UserInDBBase(UserBase):
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class User(UserInDBBase):
    pass

class UserInDB(UserInDBBase):
    hashed_password: str

# Response models with different levels of detail
class UserSummary(BaseModel):
    id: int
    email: str
    full_name: Optional[str]
    is_active: bool
    
    class Config:
        from_attributes = True

class UserDetail(User):
    videos_count: Optional[int] = None
    last_login: Optional[datetime] = None
```

### Video Schemas

```python
# api/schemas/video.py
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum

class VideoStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class VideoQuality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

class VideoBase(BaseModel):
    script: str = Field(..., min_length=1, max_length=1000)
    voice_id: str = Field(..., min_length=1)
    language: str = Field(default="en", min_length=2, max_length=5)
    quality: VideoQuality = VideoQuality.MEDIUM
    
    @validator('script')
    def validate_script(cls, v):
        if not v.strip():
            raise ValueError('Script cannot be empty')
        return v.strip()

class VideoCreate(VideoBase):
    pass

class VideoUpdate(BaseModel):
    script: Optional[str] = Field(None, min_length=1, max_length=1000)
    voice_id: Optional[str] = None
    language: Optional[str] = Field(None, min_length=2, max_length=5)
    quality: Optional[VideoQuality] = None

class VideoInDBBase(VideoBase):
    id: int
    video_id: str
    user_id: int
    status: VideoStatus
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    processing_time: Optional[float] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class Video(VideoInDBBase):
    pass

class VideoWithUser(Video):
    user: UserSummary
```

## 🛡️ Error Handling

### Custom Exception Classes

```python
# api/core/exceptions.py
from fastapi import HTTPException, status
from typing import Any, Dict, Optional

class HeyGenException(HTTPException):
    """Base exception for HeyGen AI API."""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: Optional[str] = None,
        headers: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code

class VideoProcessingException(HeyGenException):
    """Exception for video processing errors."""
    
    def __init__(self, detail: str, error_code: str = "VIDEO_PROCESSING_ERROR"):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code=error_code
        )

class RateLimitException(HeyGenException):
    """Exception for rate limiting."""
    
    def __init__(self, retry_after: int = 60):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            error_code="RATE_LIMIT_EXCEEDED",
            headers={"Retry-After": str(retry_after)}
        )

class ValidationException(HeyGenException):
    """Exception for validation errors."""
    
    def __init__(self, detail: str, field: Optional[str] = None):
        error_code = f"VALIDATION_ERROR_{field.upper()}" if field else "VALIDATION_ERROR"
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code=error_code
        )
```

### Exception Handlers

```python
# api/core/error_handlers.py
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError
import structlog

from .exceptions import HeyGenException

logger = structlog.get_logger()

async def heygen_exception_handler(request: Request, exc: HeyGenException):
    """Handle custom HeyGen exceptions."""
    logger.error(
        "HeyGen exception",
        error_code=exc.error_code,
        detail=exc.detail,
        status_code=exc.status_code,
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.detail,
                "type": "HeyGenException"
            }
        },
        headers=exc.headers
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    logger.error(
        "Validation error",
        errors=exc.errors(),
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "type": "ValidationError",
                "details": exc.errors()
            }
        }
    )

async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    """Handle database errors."""
    logger.error(
        "Database error",
        error=str(exc),
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "DATABASE_ERROR",
                "message": "Database operation failed",
                "type": "DatabaseError"
            }
        }
    )

async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        error_type=type(exc).__name__,
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "type": "InternalServerError"
            }
        }
    )
```

## 🔐 Authentication & Security

### JWT Authentication

```python
# api/core/security.py
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..config.settings import get_settings

settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Get current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            credentials.credentials, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = await get_user_service().get_user(int(user_id))
    if user is None:
        raise credentials_exception
    return user
```

### API Key Authentication

```python
# api/core/api_key_auth.py
from fastapi import Depends, HTTPException, status, Header
from typing import Optional

async def get_api_key(
    x_api_key: Optional[str] = Header(None)
) -> str:
    """Validate API key."""
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    # Validate API key against database
    user = await get_user_service().get_user_by_api_key(x_api_key)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return x_api_key
```

## 🗄️ Database Integration

### SQLAlchemy Integration

```python
# api/core/database.py
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from ..config.settings import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_pre_ping=True,
    pool_recycle=300
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def get_session() -> AsyncSession:
    """Get database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def init_db():
    """Initialize database."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def close_db():
    """Close database connections."""
    await engine.dispose()
```

## 🔄 Background Tasks

### Celery Integration

```python
# api/core/celery_app.py
from celery import Celery
from ..config.settings import get_settings

settings = get_settings()

celery_app = Celery(
    "heygen_ai",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["api.tasks.video_tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
)

# Video processing task
@celery_app.task(bind=True)
def process_video_task(self, video_id: int, script: str, voice_id: str):
    """Process video generation task."""
    try:
        # Video processing logic here
        self.update_state(state="PROGRESS", meta={"progress": 50})
        
        # Update video status
        return {"status": "completed", "video_id": video_id}
    except Exception as exc:
        self.update_state(state="FAILURE", meta={"error": str(exc)})
        raise
```

### Background Task Usage

```python
# api/api/v1/endpoints/videos.py
from fastapi import BackgroundTasks
from ..core.celery_app import process_video_task

@router.post("/videos/", response_model=Video)
async def create_video(
    video_create: VideoCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    """Create new video with background processing."""
    # Create video record
    video = await video_service.create_video(video_create, current_user.id)
    
    # Add background task
    background_tasks.add_task(
        process_video_task.delay,
        video.id,
        video_create.script,
        video_create.voice_id
    )
    
    return video
```

## 🧪 Testing

### Test Configuration

```python
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from api.main import app
from api.core.database import get_session
from api.config.settings import get_settings

# Test database
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestingSessionLocal = async_sessionmaker(engine, class_=AsyncSession)

async def override_get_session():
    async with TestingSessionLocal() as session:
        yield session

app.dependency_overrides[get_session] = override_get_session

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
async def db_session():
    async with TestingSessionLocal() as session:
        yield session
        await session.rollback()

@pytest.fixture
def test_user():
    return {
        "email": "test@example.com",
        "password": "testpassword123",
        "full_name": "Test User"
    }
```

### API Tests

```python
# tests/test_api/test_users.py
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

def test_create_user(client: TestClient, test_user: dict):
    """Test user creation."""
    response = client.post("/api/v1/users/", json=test_user)
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == test_user["email"]
    assert "id" in data

def test_get_user(client: TestClient, db_session: AsyncSession):
    """Test user retrieval."""
    # Create user first
    user_data = {"email": "test@example.com", "password": "testpass123"}
    create_response = client.post("/api/v1/users/", json=user_data)
    user_id = create_response.json()["id"]
    
    # Get user
    response = client.get(f"/api/v1/users/{user_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == user_id

def test_create_user_invalid_email(client: TestClient):
    """Test user creation with invalid email."""
    user_data = {"email": "invalid-email", "password": "testpass123"}
    response = client.post("/api/v1/users/", json=user_data)
    assert response.status_code == 422
```

## ⚡ Performance Optimization

### Caching

```python
# api/core/cache.py
import redis.asyncio as redis
from typing import Optional, Any
import json

class RedisCache:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        value = await self.redis.get(key)
        if value:
            return json.loads(value)
        return None
    
    async def set(self, key: str, value: Any, expire: int = 3600):
        """Set value in cache."""
        await self.redis.set(key, json.dumps(value), ex=expire)
    
    async def delete(self, key: str):
        """Delete value from cache."""
        await self.redis.delete(key)

# Cache dependency
async def get_cache() -> RedisCache:
    return RedisCache(settings.REDIS_URL)
```

### Response Caching

```python
# api/api/v1/endpoints/videos.py
from fastapi import Depends
from ..core.cache import RedisCache, get_cache

@router.get("/videos/{video_id}", response_model=Video)
async def get_video(
    video_id: int,
    cache: RedisCache = Depends(get_cache),
    current_user: User = Depends(get_current_active_user)
):
    """Get video with caching."""
    # Try cache first
    cache_key = f"video:{video_id}:{current_user.id}"
    cached_video = await cache.get(cache_key)
    if cached_video:
        return cached_video
    
    # Get from database
    video = await video_service.get_video(video_id, current_user.id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Cache for 5 minutes
    await cache.set(cache_key, video.dict(), expire=300)
    return video
```

### Rate Limiting

```python
# api/core/rate_limiting.py
from fastapi import HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

def rate_limit(requests: int, window: int = 60):
    """Rate limiting decorator."""
    def decorator(func):
        return limiter.limit(f"{requests}/{window}minute")(func)
    return decorator

# Usage in endpoints
@router.post("/videos/")
@rate_limit(10, 60)  # 10 requests per minute
async def create_video(request: Request, video_create: VideoCreate):
    # Endpoint logic
    pass
```

## 🚀 Deployment

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://user:pass@db:5432/heygen_ai
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ./uploads:/app/uploads
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=heygen_ai
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

  celery:
    build: .
    command: celery -A api.core.celery_app worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql+asyncpg://user:pass@db:5432/heygen_ai
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

## 📊 Monitoring

### Health Checks

```python
# api/core/health.py
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from ..core.database import get_session
from ..core.cache import RedisCache, get_cache

router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "healthy"}

@router.get("/health/detailed")
async def detailed_health_check(
    db: AsyncSession = Depends(get_session),
    cache: RedisCache = Depends(get_cache)
):
    """Detailed health check with dependencies."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "dependencies": {}
    }
    
    # Check database
    try:
        await db.execute(text("SELECT 1"))
        health_status["dependencies"]["database"] = "healthy"
    except Exception as e:
        health_status["dependencies"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Check cache
    try:
        await cache.set("health_check", "ok", expire=60)
        await cache.get("health_check")
        health_status["dependencies"]["cache"] = "healthy"
    except Exception as e:
        health_status["dependencies"]["cache"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    return health_status
```

### Metrics

```python
# api/core/metrics.py
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Response

# Metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

VIDEO_PROCESSING_DURATION = Histogram(
    'video_processing_duration_seconds',
    'Video processing duration'
)

# Middleware for metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    return response

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")
```

## 🎯 Best Practices Summary

### Code Organization
- Use dependency injection for services and database sessions
- Separate business logic into service classes
- Use Pydantic models for request/response validation
- Implement proper error handling with custom exceptions

### Performance
- Use async/await throughout the application
- Implement caching for frequently accessed data
- Use connection pooling for database connections
- Implement rate limiting for API endpoints

### Security
- Use JWT tokens for authentication
- Implement proper password hashing
- Use HTTPS in production
- Validate all input data with Pydantic

### Testing
- Write unit tests for all business logic
- Use integration tests for API endpoints
- Mock external dependencies
- Use test databases for testing

### Monitoring
- Implement health checks for all dependencies
- Use structured logging with correlation IDs
- Collect metrics for performance monitoring
- Set up alerting for critical issues

This FastAPI-specific guidelines document provides a comprehensive framework for building robust, scalable, and maintainable FastAPI applications with modern best practices and optimizations. 