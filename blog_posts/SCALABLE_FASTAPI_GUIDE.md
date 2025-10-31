# Scalable FastAPI System Guide

## Overview

This guide covers building scalable FastAPI applications with modern best practices, async operations, caching, middleware, database integration, monitoring, and production-ready features. It provides comprehensive implementations for high-performance, maintainable API systems.

## Key Features

### 1. Modern FastAPI Architecture
- **Async/Await Support**: Full asynchronous request handling
- **Dependency Injection**: Clean dependency management
- **Request/Response Models**: Pydantic-based data validation
- **OpenAPI Documentation**: Automatic API documentation
- **Type Safety**: Comprehensive type hints and validation

### 2. Database Integration
- **SQLAlchemy 2.0**: Modern ORM with async support
- **Multiple Databases**: Support for SQLite, PostgreSQL, MySQL
- **Connection Pooling**: Efficient database connection management
- **Migrations**: Alembic-based database migrations
- **Transaction Management**: ACID-compliant transactions

### 3. Caching System
- **Redis Integration**: High-performance caching with Redis
- **Memory Caching**: In-memory caching fallback
- **Cache Decorators**: Easy-to-use caching decorators
- **Cache Invalidation**: Smart cache invalidation strategies
- **Multi-level Caching**: Hierarchical caching system

### 4. Security Features
- **JWT Authentication**: Secure token-based authentication
- **Password Hashing**: Bcrypt-based password security
- **API Key Management**: Secure API key handling
- **Rate Limiting**: Request rate limiting and throttling
- **CORS Support**: Cross-origin resource sharing

### 5. Monitoring and Metrics
- **Prometheus Metrics**: Comprehensive application metrics
- **Request Logging**: Detailed request/response logging
- **Performance Monitoring**: Response time and throughput tracking
- **Health Checks**: Application health monitoring
- **Error Tracking**: Comprehensive error handling and logging

## Architecture Overview

```
Scalable FastAPI System
├── Application Layer
│   ├── FastAPI App
│   ├── Middleware Stack
│   ├── Route Handlers
│   └── Dependency Injection
├── Business Logic Layer
│   ├── Services
│   ├── Models
│   ├── Validators
│   └── Utilities
├── Data Layer
│   ├── Database Manager
│   ├── Cache Manager
│   ├── External APIs
│   └── File Storage
├── Infrastructure Layer
│   ├── Security Manager
│   ├── Metrics Manager
│   ├── Background Tasks
│   └── Monitoring
└── Configuration Layer
    ├── Settings Management
    ├── Environment Variables
    ├── Feature Flags
    └── Deployment Config
```

## Configuration Management

### Basic Configuration

```python
from scalable_fastapi_system import Settings, DatabaseSettings, RedisSettings, SecuritySettings, APISettings

# Basic configuration
settings = Settings(
    database=DatabaseSettings(
        database_url="postgresql://user:password@localhost/dbname",
        pool_size=20,
        max_overflow=30
    ),
    redis=RedisSettings(
        redis_url="redis://localhost:6379",
        redis_db=0
    ),
    security=SecuritySettings(
        secret_key="your-secret-key",
        access_token_expire_minutes=30
    ),
    api=APISettings(
        title="My API",
        version="1.0.0",
        debug=False
    )
)
```

### Environment-based Configuration

```python
# .env file
DATABASE__DATABASE_URL=postgresql://user:password@localhost/dbname
DATABASE__POOL_SIZE=20
REDIS__REDIS_URL=redis://localhost:6379
SECURITY__SECRET_KEY=your-secret-key
API__DEBUG=false
API__HOST=0.0.0.0
API__PORT=8000

# Load from environment
settings = Settings()
```

## Database Integration

### Database Manager Setup

```python
from scalable_fastapi_system import DatabaseManager

# Initialize database manager
db_manager = DatabaseManager(settings)

# Get database session
def get_db():
    db = db_manager.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Async database session
async def get_async_db():
    async with db_manager.AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
```

### Model Definition

```python
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

### Database Operations

```python
from sqlalchemy.orm import Session
from scalable_fastapi_system import User

# Create user
def create_user(db: Session, user_data: UserCreate):
    hashed_password = security_manager.get_password_hash(user_data.password)
    user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

# Get user by ID
def get_user(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()

# Update user
def update_user(db: Session, user_id: int, user_data: UserUpdate):
    user = get_user(db, user_id)
    if user:
        for field, value in user_data.dict(exclude_unset=True).items():
            setattr(user, field, value)
        user.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(user)
    return user
```

## Caching System

### Cache Manager Setup

```python
from scalable_fastapi_system import CacheManager

# Initialize cache manager
cache_manager = CacheManager(settings)

# Basic cache operations
async def cache_operations():
    # Set value
    await cache_manager.set("user:123", {"name": "John", "email": "john@example.com"}, ttl=3600)
    
    # Get value
    user_data = await cache_manager.get("user:123")
    
    # Delete value
    await cache_manager.delete("user:123")
    
    # Clear all cache
    await cache_manager.clear()
```

### Cache Decorators

```python
from scalable_fastapi_system import cache_response

# Cache API response
@cache_response(ttl=300, key_prefix="api")
async def get_user_profile(user_id: int):
    # Expensive operation
    user_data = await fetch_user_from_database(user_id)
    return user_data

# Cache with custom key
@cache_response(ttl=600, key_prefix="user")
async def get_user_posts(user_id: int, page: int = 1):
    posts = await fetch_user_posts(user_id, page)
    return posts
```

### Advanced Caching Strategies

```python
# Multi-level caching
async def get_user_data(user_id: int):
    # Try memory cache first
    user_data = await cache_manager.get(f"memory:user:{user_id}")
    if user_data:
        return user_data
    
    # Try Redis cache
    user_data = await cache_manager.get(f"redis:user:{user_id}")
    if user_data:
        # Update memory cache
        await cache_manager.set(f"memory:user:{user_id}", user_data, ttl=300)
        return user_data
    
    # Fetch from database
    user_data = await fetch_from_database(user_id)
    
    # Cache in both levels
    await cache_manager.set(f"redis:user:{user_id}", user_data, ttl=3600)
    await cache_manager.set(f"memory:user:{user_id}", user_data, ttl=300)
    
    return user_data
```

## Security Implementation

### Authentication Setup

```python
from scalable_fastapi_system import SecurityManager, get_current_user, get_current_active_user

# Initialize security manager
security_manager = SecurityManager(settings)

# Create access token
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    return security_manager.create_access_token(data, expires_delta)

# Verify token
def verify_token(token: str):
    return security_manager.verify_token(token)

# Password hashing
def hash_password(password: str):
    return security_manager.get_password_hash(password)

def verify_password(plain_password: str, hashed_password: str):
    return security_manager.verify_password(plain_password, hashed_password)
```

### Authentication Routes

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/auth/register")
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    existing_user = db.query(User).filter(
        (User.username == user_data.username) | (User.email == user_data.email)
    ).first()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered"
        )
    
    # Create user
    hashed_password = security_manager.get_password_hash(user_data.password)
    user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return {"message": "User registered successfully", "user": user}

@app.post("/auth/login")
async def login(username: str, password: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    
    if not user or not security_manager.verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Create tokens
    access_token = security_manager.create_access_token(
        data={"sub": user.username, "user_id": user.id}
    )
    refresh_token = security_manager.create_refresh_token(
        data={"sub": user.username, "user_id": user.id}
    )
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

@app.get("/users/me")
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    return current_user
```

### Rate Limiting

```python
from scalable_fastapi_system import RateLimitingMiddleware

# Add rate limiting middleware
app.add_middleware(
    RateLimitingMiddleware,
    cache_manager=cache_manager,
    settings=settings
)

# Custom rate limiting
@app.middleware("http")
async def custom_rate_limit(request: Request, call_next):
    client_ip = request.client.host
    rate_limit_key = f"rate_limit:{client_ip}"
    
    # Check rate limit
    current_requests = await cache_manager.get(rate_limit_key, 0)
    if current_requests >= 100:  # 100 requests per minute
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded"}
        )
    
    # Increment request count
    await cache_manager.set(rate_limit_key, current_requests + 1, ttl=60)
    
    response = await call_next(request)
    return response
```

## Monitoring and Metrics

### Metrics Setup

```python
from scalable_fastapi_system import MetricsManager
import prometheus_client

# Initialize metrics manager
metrics_manager = MetricsManager()

# Record request metrics
@app.middleware("http")
async def record_metrics(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record metrics
    duration = time.time() - start_time
    metrics_manager.record_request(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
        duration=duration
    )
    
    return response

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    return StreamingResponse(
        prometheus_client.generate_latest(),
        media_type="text/plain"
    )
```

### Health Checks

```python
@app.get("/health")
async def health_check():
    # Check database health
    try:
        db = next(get_db())
        db.execute("SELECT 1")
        db.close()
        database_status = "healthy"
    except Exception:
        database_status = "unhealthy"
    
    # Check Redis health
    redis_status = await cache_manager.health_check()
    
    # Get system metrics
    memory = psutil.virtual_memory()
    cpu_usage = psutil.cpu_percent()
    
    return {
        "status": "healthy" if database_status == "healthy" and redis_status == "healthy" else "degraded",
        "timestamp": datetime.utcnow(),
        "version": settings.api.version,
        "database": database_status,
        "redis": redis_status,
        "memory_usage": {
            "total": memory.total,
            "used": memory.used,
            "percent": memory.percent
        },
        "cpu_usage": cpu_usage
    }
```

### Request Logging

```python
from scalable_fastapi_system import RequestLoggingMiddleware
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlog.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Add request logging middleware
app.add_middleware(
    RequestLoggingMiddleware,
    db_manager=db_manager,
    metrics_manager=metrics_manager
)

# Custom logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    logger.info(
        "Request started",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host
    )
    
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    logger.info(
        "Request completed",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration=duration
    )
    
    return response
```

## Middleware Stack

### Complete Middleware Configuration

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.api.title,
        version=settings.api.version,
        description=settings.api.description,
        debug=settings.api.debug
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Trusted host
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    
    # Custom middleware
    app.add_middleware(RequestLoggingMiddleware, db_manager=db_manager, metrics_manager=metrics_manager)
    app.add_middleware(RateLimitingMiddleware, cache_manager=cache_manager, settings=settings)
    
    return app
```

### Custom Middleware

```python
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class CustomMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, custom_config: dict):
        super().__init__(app)
        self.custom_config = custom_config
    
    async def dispatch(self, request: Request, call_next) -> Response:
        # Pre-processing
        start_time = time.time()
        
        # Add custom headers
        request.state.custom_data = "custom_value"
        
        # Process request
        response = await call_next(request)
        
        # Post-processing
        duration = time.time() - start_time
        response.headers["X-Custom-Duration"] = str(duration)
        
        return response

# Add custom middleware
app.add_middleware(CustomMiddleware, custom_config={"key": "value"})
```

## Background Tasks

### Task Management

```python
import asyncio
from scalable_fastapi_system import background_task

# Background task execution
@app.post("/tasks/background")
async def create_background_task(
    task_data: dict,
    current_user: User = Depends(get_current_active_user)
):
    task_id = str(uuid.uuid4())
    
    # Create background task
    task = asyncio.create_task(background_task(task_id, task_data))
    app.state.background_tasks.append(task)
    
    return {"task_id": task_id, "status": "created"}

# Custom background task
async def process_data_task(data: dict):
    logger.info(f"Processing data: {data}")
    
    # Simulate processing
    await asyncio.sleep(5)
    
    # Process data
    result = await process_data(data)
    
    # Store result
    await cache_manager.set(f"task_result:{data['id']}", result, ttl=3600)
    
    logger.info(f"Data processing completed: {result}")

@app.post("/process-data")
async def process_data_endpoint(data: dict):
    # Start background processing
    asyncio.create_task(process_data_task(data))
    
    return {"message": "Data processing started", "data_id": data.get("id")}
```

## Error Handling

### Global Exception Handlers

```python
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "errors": [str(error) for error in exc.errors()],
            "request_id": getattr(request.state, 'request_id', None)
        }
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "request_id": getattr(request.state, 'request_id', None)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "message": "Internal server error",
            "request_id": getattr(request.state, 'request_id', None)
        }
    )
```

### Custom Error Classes

```python
class APIException(Exception):
    def __init__(self, message: str, status_code: int = 400, error_code: str = None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(self.message)

class ValidationException(APIException):
    def __init__(self, message: str, field: str = None):
        super().__init__(message, status_code=422, error_code="VALIDATION_ERROR")
        self.field = field

class AuthenticationException(APIException):
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401, error_code="AUTHENTICATION_ERROR")

class AuthorizationException(APIException):
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message, status_code=403, error_code="AUTHORIZATION_ERROR")

# Custom exception handler
@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.message,
            "error_code": exc.error_code,
            "request_id": getattr(request.state, 'request_id', None)
        }
    )
```

## Testing

### Unit Tests

```python
import pytest
from fastapi.testclient import TestClient
from scalable_fastapi_system import create_app, get_db

@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)

@pytest.fixture
def db_session():
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "database" in data
    assert "redis" in data

def test_user_registration(client):
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123"
    }
    
    response = client.post("/auth/register", json=user_data)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["username"] == user_data["username"]
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_complete_user_workflow(client):
    # 1. Register user
    user_data = {
        "username": "integrationuser",
        "email": "integration@example.com",
        "password": "integrationpass123"
    }
    
    register_response = client.post("/auth/register", json=user_data)
    assert register_response.status_code == 200
    
    # 2. Login user
    login_data = {
        "username": user_data["username"],
        "password": user_data["password"]
    }
    
    login_response = client.post("/auth/login", data=login_data)
    assert login_response.status_code == 200
    
    login_data = login_response.json()
    token = login_data["data"]["access_token"]
    
    # 3. Get user info
    headers = {"Authorization": f"Bearer {token}"}
    user_response = client.get("/users/me", headers=headers)
    assert user_response.status_code == 200
    
    # 4. Update user
    update_data = {"email": "updated@example.com"}
    update_response = client.put("/users/me", json=update_data, headers=headers)
    assert update_response.status_code == 200
```

## Deployment

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
COPY requirements_scalable_fastapi.txt .
RUN pip install --no-cache-dir -r requirements_scalable_fastapi.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "scalable_fastapi_system:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE__DATABASE_URL=postgresql://user:password@db:5432/appdb
      - REDIS__REDIS_URL=redis://redis:6379
      - SECURITY__SECRET_KEY=your-secret-key
    depends_on:
      - db
      - redis
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=appdb
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - app
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### Production Configuration

```python
# production.py
import os
from scalable_fastapi_system import Settings

# Production settings
production_settings = Settings(
    database=DatabaseSettings(
        database_url=os.getenv("DATABASE_URL"),
        pool_size=int(os.getenv("DB_POOL_SIZE", "20")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "30"))
    ),
    redis=RedisSettings(
        redis_url=os.getenv("REDIS_URL"),
        redis_ssl=os.getenv("REDIS_SSL", "false").lower() == "true"
    ),
    security=SecuritySettings(
        secret_key=os.getenv("SECRET_KEY"),
        access_token_expire_minutes=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    ),
    api=APISettings(
        debug=False,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        workers=int(os.getenv("WORKERS", "4"))
    )
)
```

## Performance Optimization

### Database Optimization

```python
# Database connection pooling
database_settings = DatabaseSettings(
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False
)

# Query optimization
def get_users_optimized(db: Session, skip: int = 0, limit: int = 100):
    return db.query(User)\
        .options(load_only(User.id, User.username, User.email))\
        .offset(skip)\
        .limit(limit)\
        .all()

# Batch operations
def create_users_batch(db: Session, users_data: List[UserCreate]):
    users = []
    for user_data in users_data:
        hashed_password = security_manager.get_password_hash(user_data.password)
        user = User(
            username=user_data.username,
            email=user_data.email,
            hashed_password=hashed_password
        )
        users.append(user)
    
    db.add_all(users)
    db.commit()
    
    return users
```

### Caching Optimization

```python
# Cache warming
async def warm_cache():
    """Warm up cache with frequently accessed data."""
    # Warm user cache
    users = await get_all_active_users()
    for user in users:
        await cache_manager.set(f"user:{user.id}", user.dict(), ttl=3600)
    
    # Warm configuration cache
    config = await get_application_config()
    await cache_manager.set("app:config", config, ttl=1800)

# Cache invalidation
async def invalidate_user_cache(user_id: int):
    """Invalidate user-related cache."""
    await cache_manager.delete(f"user:{user_id}")
    await cache_manager.delete(f"user:profile:{user_id}")
    await cache_manager.delete(f"user:posts:{user_id}")

# Cache patterns
async def get_user_with_cache(user_id: int):
    """Get user with intelligent caching."""
    cache_key = f"user:{user_id}"
    
    # Try cache first
    user_data = await cache_manager.get(cache_key)
    if user_data:
        return user_data
    
    # Fetch from database
    user = await get_user_from_db(user_id)
    if user:
        # Cache with appropriate TTL
        ttl = 3600 if user.is_active else 300  # Shorter TTL for inactive users
        await cache_manager.set(cache_key, user.dict(), ttl=ttl)
    
    return user
```

## Security Best Practices

### Input Validation

```python
from pydantic import BaseModel, validator, EmailStr
import re

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    
    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters')
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('Username can only contain letters, numbers, and underscores')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        return v
```

### SQL Injection Prevention

```python
# Use parameterized queries
def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

# Avoid string concatenation
def get_users_by_role(db: Session, role: str):
    # Good - parameterized
    return db.query(User).filter(User.role == role).all()
    
    # Bad - string concatenation (vulnerable to SQL injection)
    # return db.execute(f"SELECT * FROM users WHERE role = '{role}'")
```

### XSS Prevention

```python
from fastapi import Response
from fastapi.responses import HTMLResponse

@app.get("/user/{username}")
async def get_user_profile(username: str):
    # Sanitize input
    username = html.escape(username)
    
    # Return safe response
    return {"username": username, "message": "User profile"}

# Use Content Security Policy
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response
```

## Conclusion

This guide provides comprehensive coverage of building scalable FastAPI applications with modern best practices. Key takeaways include:

- **Architecture Design**: Modular, scalable architecture with clear separation of concerns
- **Database Integration**: Efficient database operations with connection pooling and migrations
- **Caching Strategy**: Multi-level caching with Redis and in-memory fallbacks
- **Security Implementation**: Comprehensive security with JWT, rate limiting, and input validation
- **Monitoring and Metrics**: Full observability with Prometheus metrics and structured logging
- **Error Handling**: Robust error handling with custom exceptions and global handlers
- **Testing Strategy**: Comprehensive testing with unit, integration, and performance tests
- **Deployment**: Production-ready deployment with Docker and container orchestration
- **Performance Optimization**: Database and cache optimization for high performance
- **Security Best Practices**: Input validation, SQL injection prevention, and XSS protection

These practices ensure production-ready, scalable, secure, and maintainable FastAPI applications. 