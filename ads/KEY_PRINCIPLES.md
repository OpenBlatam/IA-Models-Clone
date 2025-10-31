# Key Principles for Python, FastAPI, and Scalable API Development

## Table of Contents

1. [Python Best Practices](#python-best-practices)
2. [FastAPI Development Principles](#fastapi-development-principles)
3. [Scalable API Architecture](#scalable-api-architecture)
4. [Performance Optimization](#performance-optimization)
5. [Security Principles](#security-principles)
6. [Testing and Quality Assurance](#testing-and-quality-assurance)
7. [Monitoring and Observability](#monitoring-and-observability)
8. [Deployment and DevOps](#deployment-and-devops)

## Python Best Practices

### 1. Code Organization and Structure

#### **Principle: Modular Design**
```python
# ✅ Good: Modular structure
project/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── endpoints/
│   │   │   └── dependencies.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── security.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   └── services/
│       ├── __init__.py
│       └── business_logic.py
├── tests/
├── requirements.txt
└── README.md
```

#### **Principle: Single Responsibility**
```python
# ✅ Good: Each class/function has one responsibility
class UserService:
    def create_user(self, user_data: dict) -> User:
        """Only handles user creation logic."""
        pass

class UserValidator:
    def validate_user_data(self, user_data: dict) -> bool:
        """Only handles validation logic."""
        pass

# ❌ Bad: Multiple responsibilities
class UserManager:
    def create_user(self, user_data: dict) -> User:
        """Handles creation, validation, and database operations."""
        pass
```

### 2. Type Hints and Documentation

#### **Principle: Always Use Type Hints**
```python
# ✅ Good: Clear type hints
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class UserCreate(BaseModel):
    username: str
    email: str
    age: Optional[int] = None

async def create_user(
    user_data: UserCreate,
    db_session: AsyncSession
) -> User:
    """Create a new user with proper type hints."""
    pass

# ❌ Bad: No type hints
def create_user(user_data, db_session):
    """Unclear what types are expected."""
    pass
```

#### **Principle: Comprehensive Documentation**
```python
def process_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process a list of data dictionaries and return aggregated results.
    
    Args:
        data: List of dictionaries containing raw data
        
    Returns:
        Dictionary containing processed results with keys:
        - total_count: Total number of items processed
        - success_count: Number of successfully processed items
        - errors: List of error messages
        
    Raises:
        ValueError: If data is empty or malformed
        ProcessingError: If critical processing fails
        
    Example:
        >>> data = [{"id": 1, "value": 10}, {"id": 2, "value": 20}]
        >>> result = process_data(data)
        >>> print(result["total_count"])
        2
    """
    pass
```

### 3. Error Handling

#### **Principle: Explicit Exception Handling**
```python
# ✅ Good: Specific exception handling
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def safe_divide(a: float, b: float) -> Optional[float]:
    """Safely divide two numbers with proper error handling."""
    try:
        return a / b
    except ZeroDivisionError:
        logger.error(f"Division by zero attempted: {a} / {b}")
        return None
    except TypeError as e:
        logger.error(f"Invalid types for division: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in division: {e}")
        return None

# ❌ Bad: Generic exception handling
def divide(a, b):
    try:
        return a / b
    except:
        return None
```

### 4. Performance and Memory Management

#### **Principle: Use Generators for Large Data**
```python
# ✅ Good: Memory-efficient generator
def process_large_file(file_path: str) -> Iterator[str]:
    """Process large files without loading everything into memory."""
    with open(file_path, 'r') as file:
        for line in file:
            yield line.strip()

# ❌ Bad: Loads entire file into memory
def process_large_file_bad(file_path: str) -> List[str]:
    with open(file_path, 'r') as file:
        return file.readlines()  # Loads entire file
```

#### **Principle: Use Context Managers**
```python
# ✅ Good: Proper resource management
from contextlib import contextmanager
import time

@contextmanager
def timer(operation_name: str):
    """Context manager for timing operations."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"{operation_name} took {end_time - start_time:.2f} seconds")

# Usage
with timer("Data processing"):
    process_data(large_dataset)
```

## FastAPI Development Principles

### 1. API Design

#### **Principle: RESTful Design**
```python
from fastapi import FastAPI, HTTPException, status
from typing import List

app = FastAPI()

# ✅ Good: RESTful endpoints
@app.get("/users", response_model=List[UserResponse])
async def get_users(skip: int = 0, limit: int = 100):
    """Get list of users with pagination."""
    pass

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):
    """Get specific user by ID."""
    pass

@app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate):
    """Create new user."""
    pass

@app.put("/users/{user_id}", response_model=UserResponse)
async def update_user(user_id: int, user: UserUpdate):
    """Update existing user."""
    pass

@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: int):
    """Delete user."""
    pass
```

#### **Principle: Proper HTTP Status Codes**
```python
from fastapi import HTTPException, status

# ✅ Good: Appropriate status codes
async def create_user(user_data: UserCreate):
    if user_exists(user_data.email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User with this email already exists"
        )
    
    user = await save_user(user_data)
    return user

async def get_user(user_id: int):
    user = await find_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user
```

### 2. Data Validation

#### **Principle: Use Pydantic for Validation**
```python
from pydantic import BaseModel, Field, validator, EmailStr
from typing import Optional
import re

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, regex=r"^[a-zA-Z0-9_]+$")
    email: EmailStr
    password: str = Field(..., min_length=8)
    age: Optional[int] = Field(None, ge=0, le=120)
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength."""
        if not re.match(r"^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!%*#?&]{8,}$", v):
            raise ValueError('Password must contain letters, numbers, and special characters')
        return v
    
    @validator('username')
    def validate_username(cls, v):
        """Ensure username is not reserved."""
        reserved_names = ['admin', 'root', 'system']
        if v.lower() in reserved_names:
            raise ValueError('Username is reserved')
        return v

    class Config:
        schema_extra = {
            "example": {
                "username": "john_doe",
                "email": "john@example.com",
                "password": "SecurePass123!",
                "age": 25
            }
        }
```

### 3. Dependency Injection

#### **Principle: Use Dependencies for Reusability**
```python
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Generator

# Database dependency
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication dependency
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_user_by_username(db, username=username)
    if user is None:
        raise credentials_exception
    return user

# Usage in endpoints
@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user
```

## Scalable API Architecture

### 1. Microservices Principles

#### **Principle: Service Separation**
```python
# ✅ Good: Separate services for different concerns
# user_service.py
class UserService:
    def __init__(self, db: Database):
        self.db = db
    
    async def create_user(self, user_data: UserCreate) -> User:
        # User-specific business logic
        pass

# notification_service.py
class NotificationService:
    def __init__(self, email_client: EmailClient):
        self.email_client = email_client
    
    async def send_welcome_email(self, user: User):
        # Notification-specific logic
        pass

# api_gateway.py
@app.post("/users")
async def create_user(
    user_data: UserCreate,
    user_service: UserService = Depends(get_user_service),
    notification_service: NotificationService = Depends(get_notification_service)
):
    user = await user_service.create_user(user_data)
    await notification_service.send_welcome_email(user)
    return user
```

### 2. Caching Strategy

#### **Principle: Multi-Level Caching**
```python
from functools import lru_cache
import redis
from typing import Optional

class CacheManager:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def get(self, key: str) -> Optional[dict]:
        """Get from Redis cache."""
        try:
            value = await self.redis.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: dict, ttl: int = 3600):
        """Set in Redis cache."""
        try:
            await self.redis.setex(key, ttl, json.dumps(value))
        except Exception as e:
            logger.error(f"Cache set error: {e}")

# In-memory cache for frequently accessed data
@lru_cache(maxsize=128)
def get_config_value(key: str) -> str:
    """Get configuration value with in-memory caching."""
    return load_config_from_file(key)

# Usage in endpoints
@app.get("/users/{user_id}")
async def get_user(
    user_id: int,
    cache: CacheManager = Depends(get_cache_manager),
    db: Session = Depends(get_db)
):
    # Try cache first
    cache_key = f"user:{user_id}"
    cached_user = await cache.get(cache_key)
    if cached_user:
        return cached_user
    
    # Fallback to database
    user = get_user_from_db(db, user_id)
    if user:
        await cache.set(cache_key, user.dict(), ttl=1800)
    
    return user
```

### 3. Database Optimization

#### **Principle: Efficient Queries**
```python
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy import and_, or_

# ✅ Good: Optimized queries with eager loading
async def get_users_with_posts(skip: int = 0, limit: int = 100):
    query = (
        select(User)
        .options(selectinload(User.posts))  # Eager load posts
        .offset(skip)
        .limit(limit)
    )
    return await db.execute(query)

# ✅ Good: Use indexes and proper filtering
async def search_users(
    name: Optional[str] = None,
    email: Optional[str] = None,
    active: Optional[bool] = None
):
    conditions = []
    
    if name:
        conditions.append(User.name.ilike(f"%{name}%"))
    if email:
        conditions.append(User.email == email)
    if active is not None:
        conditions.append(User.is_active == active)
    
    query = select(User).where(and_(*conditions))
    return await db.execute(query)

# ❌ Bad: N+1 query problem
async def get_users_with_posts_bad():
    users = await db.execute(select(User))
    for user in users:
        # This creates N additional queries
        posts = await db.execute(select(Post).where(Post.user_id == user.id))
        user.posts = posts
```

## Performance Optimization

### 1. Async/Await Patterns

#### **Principle: Use Async for I/O Operations**
```python
import asyncio
import aiohttp
from typing import List

# ✅ Good: Async I/O operations
async def fetch_user_data(user_ids: List[int]) -> List[dict]:
    """Fetch user data concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_single_user(session, user_id)
            for user_id in user_ids
        ]
        return await asyncio.gather(*tasks)

async def fetch_single_user(session: aiohttp.ClientSession, user_id: int) -> dict:
    """Fetch single user data."""
    async with session.get(f"/api/users/{user_id}") as response:
        return await response.json()

# ✅ Good: Background tasks
@app.post("/users/bulk-import")
async def bulk_import_users(
    users: List[UserCreate],
    background_tasks: BackgroundTasks
):
    """Import users in background."""
    background_tasks.add_task(process_bulk_import, users)
    return {"message": "Import started", "count": len(users)}

async def process_bulk_import(users: List[UserCreate]):
    """Process bulk import in background."""
    for user in users:
        await create_user(user)
        await asyncio.sleep(0.1)  # Rate limiting
```

### 2. Connection Pooling

#### **Principle: Efficient Database Connections**
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.pool import QueuePool

# ✅ Good: Connection pooling configuration
engine = create_async_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,  # Maximum connections in pool
    max_overflow=30,  # Additional connections when pool is full
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=3600,  # Recycle connections after 1 hour
    echo=False  # Set to True for SQL logging
)

async def get_db() -> AsyncSession:
    async with engine.begin() as conn:
        async with AsyncSession(conn) as session:
            yield session
```

### 3. Rate Limiting

#### **Principle: Protect Against Abuse**
```python
from fastapi import HTTPException, Request
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > minute_ago
        ]
        
        # Check rate limit
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.requests[client_id].append(now)
        return True

rate_limiter = RateLimiter()

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    client_id = request.client.host
    
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded"
        )
    
    response = await call_next(request)
    return response
```

## Security Principles

### 1. Authentication and Authorization

#### **Principle: Secure Token Management**
```python
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class SecurityManager:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash password."""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: dict, expires_delta: timedelta = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> dict:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            raise HTTPException(
                status_code=401,
                detail="Invalid token"
            )
```

### 2. Input Validation

#### **Principle: Validate All Inputs**
```python
from pydantic import BaseModel, validator, Field
import re

class UserInput(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r"^[^@]+@[^@]+\.[^@]+$")
    password: str = Field(..., min_length=8)
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username format."""
        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError('Username must contain only letters, numbers, and underscores')
        return v
    
    @validator('password')
    def validate_password_strength(cls, v):
        """Validate password strength."""
        if not re.search(r"[A-Z]", v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r"[a-z]", v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r"\d", v):
            raise ValueError('Password must contain at least one digit')
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError('Password must contain at least one special character')
        return v
```

### 3. SQL Injection Prevention

#### **Principle: Use ORM and Parameterized Queries**
```python
# ✅ Good: Use SQLAlchemy ORM
from sqlalchemy.orm import Session
from sqlalchemy import select

async def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email using ORM."""
    query = select(User).where(User.email == email)
    result = await db.execute(query)
    return result.scalar_one_or_none()

# ✅ Good: Use parameterized queries if raw SQL is needed
async def get_users_by_role(db: Session, role: str) -> List[User]:
    """Get users by role using parameterized query."""
    query = text("SELECT * FROM users WHERE role = :role")
    result = await db.execute(query, {"role": role})
    return result.fetchall()

# ❌ Bad: String concatenation (SQL injection risk)
async def get_user_by_email_bad(db: Session, email: str) -> Optional[User]:
    query = f"SELECT * FROM users WHERE email = '{email}'"  # DANGEROUS!
    result = await db.execute(query)
    return result.fetchone()
```

## Testing and Quality Assurance

### 1. Unit Testing

#### **Principle: Comprehensive Test Coverage**
```python
import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

class TestUserService:
    def setup_method(self):
        """Setup test fixtures."""
        self.db_mock = Mock()
        self.user_service = UserService(self.db_mock)
    
    def test_create_user_success(self):
        """Test successful user creation."""
        user_data = UserCreate(
            username="testuser",
            email="test@example.com",
            password="SecurePass123!"
        )
        
        self.db_mock.add.return_value = None
        self.db_mock.commit.return_value = None
        
        result = self.user_service.create_user(user_data)
        
        assert result.username == "testuser"
        assert result.email == "test@example.com"
        self.db_mock.add.assert_called_once()
        self.db_mock.commit.assert_called_once()
    
    def test_create_user_duplicate_email(self):
        """Test user creation with duplicate email."""
        user_data = UserCreate(
            username="testuser",
            email="existing@example.com",
            password="SecurePass123!"
        )
        
        self.db_mock.query.return_value.filter.return_value.first.return_value = User()
        
        with pytest.raises(ValueError, match="Email already exists"):
            self.user_service.create_user(user_data)

class TestAPIEndpoints:
    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)
    
    def test_create_user_endpoint(self):
        """Test user creation endpoint."""
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "SecurePass123!"
        }
        
        response = self.client.post("/users", json=user_data)
        
        assert response.status_code == 201
        assert response.json()["username"] == "testuser"
    
    def test_create_user_invalid_data(self):
        """Test user creation with invalid data."""
        user_data = {
            "username": "t",  # Too short
            "email": "invalid-email",
            "password": "123"  # Too short
        }
        
        response = self.client.post("/users", json=user_data)
        
        assert response.status_code == 422
        errors = response.json()["detail"]
        assert len(errors) > 0
```

### 2. Integration Testing

#### **Principle: Test Complete Workflows**
```python
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

@pytest.mark.asyncio
async def test_user_workflow():
    """Test complete user workflow."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # 1. Create user
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "SecurePass123!"
        }
        
        create_response = await client.post("/users", json=user_data)
        assert create_response.status_code == 201
        user = create_response.json()
        
        # 2. Login
        login_data = {
            "username": "testuser",
            "password": "SecurePass123!"
        }
        
        login_response = await client.post("/auth/login", data=login_data)
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
        
        # 3. Access protected endpoint
        headers = {"Authorization": f"Bearer {token}"}
        me_response = await client.get("/users/me", headers=headers)
        assert me_response.status_code == 200
        assert me_response.json()["username"] == "testuser"
```

## Monitoring and Observability

### 1. Logging

#### **Principle: Structured Logging**
```python
import logging
import json
from datetime import datetime
from typing import Any, Dict

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
    
    def log_event(self, event_type: str, data: Dict[str, Any], level: str = "info"):
        """Log structured event."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "data": data,
            "level": level
        }
        
        getattr(self.logger, level)(json.dumps(log_entry))
    
    def log_request(self, method: str, path: str, status_code: int, duration: float):
        """Log HTTP request."""
        self.log_event("http_request", {
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration * 1000
        })
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context."""
        self.log_event("error", {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }, level="error")

# Usage
logger = StructuredLogger("api")

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    logger.log_request(
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration=duration
    )
    
    return response
```

### 2. Metrics

#### **Principle: Collect Key Metrics**
```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')
ERROR_COUNT = Counter('errors_total', 'Total errors', ['error_type'])

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    try:
        response = await call_next(request)
        
        # Record metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_DURATION.observe(time.time() - start_time)
        
        return response
        
    except Exception as e:
        ERROR_COUNT.labels(error_type=type(e).__name__).inc()
        raise
```

## Deployment and DevOps

### 1. Containerization

#### **Principle: Use Docker for Consistency**
```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Environment Configuration

#### **Principle: Environment-Based Configuration**
```python
from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database
    database_url: str
    database_pool_size: int = 20
    database_max_overflow: int = 30
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Monitoring
    log_level: str = "INFO"
    enable_metrics: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Load settings
settings = Settings()
```

### 3. CI/CD Pipeline

#### **Principle: Automated Testing and Deployment**
```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest tests/ --cov=app --cov-report=xml
    
    - name: Run linting
      run: |
        flake8 app/
        black --check app/
        isort --check-only app/
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
      with:
        file: ./coverage.xml

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to production
      run: |
        echo "Deploying to production..."
        # Add deployment steps here
```

## Summary

These key principles form the foundation for building robust, scalable, and maintainable Python applications with FastAPI:

1. **Modular Design**: Organize code into logical modules with clear responsibilities
2. **Type Safety**: Use type hints and Pydantic for validation
3. **Error Handling**: Implement comprehensive error handling with proper logging
4. **Performance**: Use async/await, connection pooling, and caching
5. **Security**: Validate inputs, use secure authentication, and prevent common vulnerabilities
6. **Testing**: Write comprehensive unit and integration tests
7. **Monitoring**: Implement structured logging and metrics collection
8. **DevOps**: Use containers, environment-based configuration, and automated pipelines

Following these principles ensures your applications are production-ready, maintainable, and scalable. 