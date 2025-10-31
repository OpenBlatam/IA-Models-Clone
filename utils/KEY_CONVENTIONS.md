# 🎯 Key Conventions - Blatam Academy Backend

## Overview

This document outlines the essential coding standards, patterns, and best practices for the Blatam Academy backend system. These conventions ensure consistency, maintainability, and optimal performance across all components.

## 📋 Table of Contents

1. [General Coding Standards](#general-coding-standards)
2. [FastAPI Conventions](#fastapi-conventions)
3. [Database Conventions](#database-conventions)
4. [Async/Await Patterns](#asyncawait-patterns)
5. [Error Handling](#error-handling)
6. [Performance Optimization](#performance-optimization)
7. [Security Conventions](#security-conventions)
8. [Testing Conventions](#testing-conventions)
9. [Documentation Standards](#documentation-standards)
10. [File Organization](#file-organization)
11. [Naming Conventions](#naming-conventions)
12. [Code Quality](#code-quality)

---

## 🏗️ General Coding Standards

### 1. Python Version & Dependencies
```python
# Use Python 3.9+ features
# Minimum Python version: 3.9
# Target Python version: 3.11+

# Core dependencies
fastapi>=0.104.0
pydantic>=2.0.0
sqlalchemy>=2.0.0
asyncpg>=0.29.0
redis>=5.0.0
orjson>=3.9.0
structlog>=23.0.0
```

### 2. Import Organization
```python
# Standard library imports
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union

# Third-party imports
import orjson
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

# Local imports
from .models import User
from .schemas import UserCreate, UserResponse
from .services import UserService
from .utils import logger
```

### 3. Type Hints
```python
# Always use type hints
async def create_user(user_data: UserCreate, db: AsyncSession) -> UserResponse:
    """Create a new user."""
    pass

# Use Union for multiple types
def process_data(data: Union[str, bytes, Dict[str, Any]]) -> List[str]:
    """Process different data types."""
    pass

# Use Optional for nullable values
def get_user(user_id: Optional[int] = None) -> Optional[User]:
    """Get user by ID."""
    pass
```

---

## 🚀 FastAPI Conventions

### 1. Application Structure
```python
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import users, auth, api
from .core.config import settings

app = FastAPI(
    title="Blatam Academy API",
    version="1.0.0",
    description="Blatam Academy Backend API",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(api.router, prefix="/api/v1", tags=["api"])
```

### 2. Router Organization
```python
# routers/users.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from ..dependencies import get_db, get_current_user
from ..schemas import UserCreate, UserResponse, UserUpdate
from ..services import UserService
from ..models import User

router = APIRouter()

@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """Create a new user."""
    return await UserService.create_user(db, user_data)

@router.get("/", response_model=List[UserResponse])
async def get_users(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> List[UserResponse]:
    """Get list of users."""
    return await UserService.get_users(db, skip=skip, limit=limit)
```

### 3. Dependency Injection
```python
# dependencies.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from .database import get_db_session
from .services import AuthService
from .models import User

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db_session)
) -> User:
    """Get current authenticated user."""
    token = credentials.credentials
    user = await AuthService.verify_token(db, token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return user

async def get_db_session() -> AsyncSession:
    """Get database session."""
    async with get_db() as session:
        yield session
```

### 4. Response Models
```python
# schemas.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime

class BaseResponseModel(BaseModel):
    """Base response model with common fields."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        from_attributes=True
    )

class UserResponse(BaseResponseModel):
    """User response model."""
    id: int
    email: str
    username: str
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

class PaginatedResponse(BaseResponseModel):
    """Paginated response wrapper."""
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int
    has_next: bool
    has_prev: bool
```

---

## 🗄️ Database Conventions

### 1. Async Database Operations
```python
# Always use async database operations
async def get_user_by_id(db: AsyncSession, user_id: int) -> Optional[User]:
    """Get user by ID using async operations."""
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    return result.scalar_one_or_none()

async def create_user(db: AsyncSession, user_data: UserCreate) -> User:
    """Create user using async operations."""
    user = User(**user_data.model_dump())
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user
```

### 2. Database Models
```python
# models.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class User(Base):
    """User model."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}')>"
```

### 3. Database Sessions
```python
# database.py
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from .core.config import settings

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_pre_ping=True,
    pool_recycle=300
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def get_db() -> AsyncSession:
    """Get database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
```

---

## ⚡ Async/Await Patterns

### 1. Async Function Definitions
```python
# Always use async/await for I/O operations
async def fetch_user_data(user_id: int) -> Dict[str, Any]:
    """Fetch user data asynchronously."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"/api/users/{user_id}") as response:
            return await response.json()

async def process_users(users: List[User]) -> List[Dict[str, Any]]:
    """Process users concurrently."""
    tasks = [fetch_user_data(user.id) for user in users]
    return await asyncio.gather(*tasks)
```

### 2. Background Tasks
```python
from fastapi import BackgroundTasks

@router.post("/users/")
async def create_user(
    user_data: UserCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Create user with background tasks."""
    user = await UserService.create_user(db, user_data)
    
    # Add background tasks
    background_tasks.add_task(send_welcome_email, user.email)
    background_tasks.add_task(update_user_analytics, user.id)
    
    return user
```

### 3. Concurrent Operations
```python
# Use asyncio.gather for concurrent operations
async def load_user_data(user_id: int) -> Dict[str, Any]:
    """Load user data from multiple sources concurrently."""
    user_info, user_posts, user_followers = await asyncio.gather(
        fetch_user_info(user_id),
        fetch_user_posts(user_id),
        fetch_user_followers(user_id)
    )
    
    return {
        "info": user_info,
        "posts": user_posts,
        "followers": user_followers
    }
```

---

## 🛡️ Error Handling

### 1. Custom Exceptions
```python
# exceptions.py
class BlatamAcademyException(Exception):
    """Base exception for Blatam Academy."""
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class UserNotFoundException(BlatamAcademyException):
    """User not found exception."""
    pass

class ValidationException(BlatamAcademyException):
    """Validation exception."""
    pass

class AuthenticationException(BlatamAcademyException):
    """Authentication exception."""
    pass
```

### 2. HTTP Exception Handling
```python
# Always use proper HTTP status codes
from fastapi import HTTPException, status

@router.get("/users/{user_id}")
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    """Get user by ID."""
    user = await UserService.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found"
        )
    return user
```

### 3. Exception Middleware
```python
# middleware/exception_handler.py
from fastapi import Request
from fastapi.responses import JSONResponse
from .exceptions import BlatamAcademyException

async def exception_handler(request: Request, exc: BlatamAcademyException):
    """Handle custom exceptions."""
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.message,
            "error_code": exc.error_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
```

---

## 🚀 Performance Optimization

### 1. Database Query Optimization
```python
# Use select() for queries
from sqlalchemy import select

async def get_users_with_posts(db: AsyncSession) -> List[User]:
    """Get users with their posts using optimized query."""
    stmt = (
        select(User)
        .options(selectinload(User.posts))
        .where(User.is_active == True)
    )
    result = await db.execute(stmt)
    return result.scalars().all()

# Use bulk operations for large datasets
async def create_users_bulk(db: AsyncSession, users_data: List[UserCreate]):
    """Create multiple users using bulk insert."""
    users = [User(**user_data.model_dump()) for user_data in users_data]
    db.add_all(users)
    await db.commit()
```

### 2. Caching
```python
# Use Redis for caching
import redis.asyncio as redis
from .core.config import settings

redis_client = redis.from_url(settings.redis_url)

async def get_cached_user(user_id: int) -> Optional[Dict[str, Any]]:
    """Get user from cache."""
    cache_key = f"user:{user_id}"
    cached_data = await redis_client.get(cache_key)
    if cached_data:
        return orjson.loads(cached_data)
    return None

async def cache_user(user_id: int, user_data: Dict[str, Any], ttl: int = 3600):
    """Cache user data."""
    cache_key = f"user:{user_id}"
    await redis_client.setex(cache_key, ttl, orjson.dumps(user_data))
```

### 3. Lazy Loading
```python
# Use lazy loading for large datasets
from .utils.advanced_lazy_loading import AdvancedLazyLoader, LoadingStrategy

lazy_loader = AdvancedLazyLoader()

@lazy_load(LoadingStrategy.LAZY)
async def load_large_dataset():
    """Load large dataset lazily."""
    return await fetch_large_dataset()

@streaming_load(chunk_size=1024*1024)
async def stream_large_file(file_path: str):
    """Stream large file content."""
    async with aiofiles.open(file_path, 'r') as f:
        async for line in f:
            yield line.strip()
```

---

## 🔒 Security Conventions

### 1. Authentication
```python
# Use JWT tokens for authentication
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash password."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt
```

### 2. Input Validation
```python
# Always validate input data
from pydantic import BaseModel, Field, validator

class UserCreate(BaseModel):
    """User creation model with validation."""
    email: str = Field(..., regex=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    username: str = Field(..., min_length=3, max_length=50, regex=r"^[a-zA-Z0-9_]+$")
    password: str = Field(..., min_length=8, max_length=128)
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength."""
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v
```

### 3. Rate Limiting
```python
# Implement rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

@router.post("/login")
@limiter.limit("5/minute")
async def login(request: Request, user_data: UserLogin):
    """Login with rate limiting."""
    # Login logic here
    pass
```

---

## 🧪 Testing Conventions

### 1. Test Structure
```python
# tests/test_users.py
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from ..models import User
from ..schemas import UserCreate

@pytest.mark.asyncio
async def test_create_user(async_client: AsyncClient, db_session: AsyncSession):
    """Test user creation."""
    user_data = UserCreate(
        email="test@example.com",
        username="testuser",
        password="testpass123"
    )
    
    response = await async_client.post("/api/v1/users/", json=user_data.model_dump())
    assert response.status_code == 201
    
    data = response.json()
    assert data["email"] == user_data.email
    assert data["username"] == user_data.username
    assert "id" in data

@pytest.mark.asyncio
async def test_get_user_not_found(async_client: AsyncClient):
    """Test getting non-existent user."""
    response = await async_client.get("/api/v1/users/999")
    assert response.status_code == 404
```

### 2. Test Fixtures
```python
# conftest.py
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from httpx import AsyncClient
from ..main import app
from ..database import get_db

@pytest.fixture
async def db_session():
    """Database session fixture."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    TestingSessionLocal = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with TestingSessionLocal() as session:
        yield session

@pytest.fixture
async def async_client(db_session: AsyncSession):
    """Async client fixture."""
    async def override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
    
    app.dependency_overrides.clear()
```

---

## 📚 Documentation Standards

### 1. Docstrings
```python
def calculate_user_score(user_id: int, weights: Dict[str, float]) -> float:
    """
    Calculate user score based on various metrics.
    
    Args:
        user_id: The ID of the user to calculate score for
        weights: Dictionary of metric weights (e.g., {'posts': 0.3, 'followers': 0.7})
    
    Returns:
        float: Calculated user score between 0 and 100
        
    Raises:
        UserNotFoundException: If user with given ID doesn't exist
        ValueError: If weights don't sum to 1.0
        
    Example:
        >>> weights = {'posts': 0.3, 'followers': 0.7}
        >>> score = calculate_user_score(123, weights)
        >>> print(f"User score: {score}")
    """
    pass
```

### 2. API Documentation
```python
@router.post(
    "/users/",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user",
    description="Create a new user account with the provided information",
    response_description="User created successfully"
)
async def create_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """
    Create a new user account.
    
    This endpoint creates a new user account with the provided information.
    The email and username must be unique.
    
    - **email**: Valid email address
    - **username**: 3-50 characters, alphanumeric and underscore only
    - **password**: 8-128 characters, must contain uppercase, lowercase, and digit
    """
    return await UserService.create_user(db, user_data)
```

---

## 📁 File Organization

### 1. Project Structure
```
blatam-academy/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py          # Configuration settings
│   │   │   ├── security.py        # Security utilities
│   │   │   └── database.py        # Database configuration
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── v1/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── endpoints/
│   │   │   │   │   ├── users.py
│   │   │   │   │   ├── auth.py
│   │   │   │   │   └── posts.py
│   │   │   │   └── api.py
│   │   │   └── deps.py            # Dependencies
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── user.py
│   │   │   └── post.py
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── user.py
│   │   │   └── post.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── user_service.py
│   │   │   └── auth_service.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── logger.py
│   │   │   └── helpers.py
│   │   └── tests/
│   │       ├── __init__.py
│   │       ├── conftest.py
│   │       ├── test_users.py
│   │       └── test_auth.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
```

### 2. Import Organization
```python
# __init__.py files should export public APIs
# models/__init__.py
from .user import User
from .post import Post

__all__ = ["User", "Post"]

# schemas/__init__.py
from .user import UserCreate, UserResponse, UserUpdate
from .post import PostCreate, PostResponse, PostUpdate

__all__ = [
    "UserCreate", "UserResponse", "UserUpdate",
    "PostCreate", "PostResponse", "PostUpdate"
]
```

---

## 🏷️ Naming Conventions

### 1. Python Naming
```python
# Variables and functions: snake_case
user_name = "john_doe"
def get_user_by_id(user_id: int):
    pass

# Classes: PascalCase
class UserService:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_RETRY_ATTEMPTS = 3
DEFAULT_PAGE_SIZE = 100

# Private methods: _leading_underscore
def _internal_helper():
    pass

# Protected methods: _leading_underscore
def _validate_user_data(user_data):
    pass
```

### 2. Database Naming
```python
# Tables: snake_case, plural
users, posts, user_posts

# Columns: snake_case
user_id, created_at, is_active

# Indexes: idx_tablename_columnname
idx_users_email, idx_posts_created_at

# Foreign keys: tablename_id
user_id, post_id, category_id
```

### 3. API Naming
```python
# Endpoints: kebab-case
GET /api/v1/users
POST /api/v1/user-profiles
PUT /api/v1/user-settings

# Query parameters: snake_case
GET /api/v1/users?page=1&page_size=20&sort_by=created_at

# Request/Response bodies: camelCase (JSON convention)
{
    "userName": "john_doe",
    "emailAddress": "john@example.com",
    "isActive": true
}
```

---

## 🎯 Code Quality

### 1. Code Formatting
```python
# Use Black for code formatting
# black --line-length 88 --target-version py39 app/

# Use isort for import sorting
# isort --profile black app/

# Use flake8 for linting
# flake8 app/ --max-line-length=88 --extend-ignore=E203,W503
```

### 2. Type Checking
```python
# Use mypy for type checking
# mypy app/ --ignore-missing-imports

# Add type hints to all functions
def process_data(data: List[Dict[str, Any]]) -> List[str]:
    """Process data with type hints."""
    return [str(item) for item in data]

# Use TypeVar for generic types
T = TypeVar('T')

def get_first_item(items: List[T]) -> Optional[T]:
    """Get first item from list."""
    return items[0] if items else None
```

### 3. Code Complexity
```python
# Keep functions simple and focused
# Maximum cyclomatic complexity: 10
# Maximum function length: 50 lines

def process_user_data(user_data: Dict[str, Any]) -> User:
    """Process user data - keep it simple."""
    # Validate input
    validate_user_data(user_data)
    
    # Transform data
    transformed_data = transform_user_data(user_data)
    
    # Create user
    user = create_user(transformed_data)
    
    return user
```

### 4. Error Handling
```python
# Always handle exceptions appropriately
async def safe_database_operation():
    """Safe database operation with proper error handling."""
    try:
        result = await database.execute(query)
        return result
    except DatabaseConnectionError as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

---

## 🔧 Development Workflow

### 1. Git Conventions
```bash
# Branch naming: feature/description, bugfix/description, hotfix/description
git checkout -b feature/user-authentication
git checkout -b bugfix/login-validation
git checkout -b hotfix/security-patch

# Commit messages: Conventional Commits
git commit -m "feat: add user authentication system"
git commit -m "fix: resolve login validation issue"
git commit -m "docs: update API documentation"
git commit -m "test: add user authentication tests"
```

### 2. Environment Management
```python
# Use environment variables for configuration
# .env file
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/db
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-secret-key
DEBUG=True

# config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    redis_url: str
    secret_key: str
    debug: bool = False
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### 3. Logging
```python
# Use structured logging
import structlog

logger = structlog.get_logger()

# Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
logger.info("User created", user_id=123, email="user@example.com")
logger.warning("Rate limit exceeded", user_id=123, ip="192.168.1.1")
logger.error("Database connection failed", error=str(e))
```

---

## 📊 Performance Guidelines

### 1. Database Optimization
```python
# Use indexes for frequently queried columns
# Use select() instead of query()
# Use bulk operations for large datasets
# Use connection pooling
# Use async operations

# Good
async def get_active_users(db: AsyncSession) -> List[User]:
    stmt = select(User).where(User.is_active == True)
    result = await db.execute(stmt)
    return result.scalars().all()

# Bad
def get_active_users(db: Session) -> List[User]:
    return db.query(User).filter(User.is_active == True).all()
```

### 2. Caching Strategy
```python
# Cache frequently accessed data
# Use appropriate TTL
# Implement cache invalidation
# Use Redis for distributed caching

async def get_user_with_cache(user_id: int) -> Optional[User]:
    # Try cache first
    cached_user = await cache.get(f"user:{user_id}")
    if cached_user:
        return cached_user
    
    # Fetch from database
    user = await get_user_from_db(user_id)
    if user:
        await cache.set(f"user:{user_id}", user, ttl=3600)
    
    return user
```

### 3. Async Operations
```python
# Use asyncio.gather for concurrent operations
# Use background tasks for non-critical operations
# Avoid blocking operations in async functions

# Good
async def load_user_data(user_id: int):
    user_info, user_posts = await asyncio.gather(
        fetch_user_info(user_id),
        fetch_user_posts(user_id)
    )
    return {"info": user_info, "posts": user_posts}

# Bad
async def load_user_data(user_id: int):
    user_info = await fetch_user_info(user_id)
    user_posts = await fetch_user_posts(user_id)  # Sequential
    return {"info": user_info, "posts": user_posts}
```

---

## 🎯 Summary

These key conventions ensure:

1. **Consistency** - Uniform coding style across the project
2. **Maintainability** - Clear structure and organization
3. **Performance** - Optimized database queries and async operations
4. **Security** - Proper authentication and input validation
5. **Reliability** - Comprehensive error handling and testing
6. **Scalability** - Efficient caching and lazy loading strategies

Follow these conventions to maintain high code quality and ensure the success of the Blatam Academy backend system. 