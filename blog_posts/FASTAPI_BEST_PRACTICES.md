# 🚀 FastAPI Best Practices - Complete Guide

## 📋 Table of Contents

1. [Project Structure](#project-structure)
2. [Dependency Injection](#dependency-injection)
3. [Request/Response Models](#requestresponse-models)
4. [Error Handling](#error-handling)
5. [Security](#security)
6. [Performance](#performance)
7. [Testing](#testing)
8. [Documentation](#documentation)
9. [Production Deployment](#production-deployment)
10. [Monitoring](#monitoring)

## 🏗️ Project Structure

### Recommended Structure
```
app/
├── __init__.py
├── main.py                 # FastAPI app creation
├── config/
│   ├── __init__.py
│   ├── settings.py         # Pydantic settings
│   └── database.py         # Database configuration
├── api/
│   ├── __init__.py
│   ├── v1/
│   │   ├── __init__.py
│   │   ├── endpoints/
│   │   │   ├── __init__.py
│   │   │   ├── analyses.py
│   │   │   ├── batches.py
│   │   │   └── health.py
│   │   └── api.py          # API router
│   └── dependencies.py     # Shared dependencies
├── core/
│   ├── __init__.py
│   ├── security.py         # Authentication/authorization
│   ├── exceptions.py       # Custom exceptions
│   └── middleware.py       # Custom middleware
├── models/
│   ├── __init__.py
│   ├── database.py         # SQLAlchemy models
│   └── schemas.py          # Pydantic models
├── services/
│   ├── __init__.py
│   ├── analysis.py         # Business logic
│   └── database.py         # Database operations
├── utils/
│   ├── __init__.py
│   ├── logging.py          # Logging configuration
│   └── validators.py       # Custom validators
└── tests/
    ├── __init__.py
    ├── conftest.py         # Test configuration
    ├── test_api/
    │   ├── __init__.py
    │   ├── test_analyses.py
    │   └── test_batches.py
    └── test_services/
        ├── __init__.py
        └── test_analysis.py
```

### Example Implementation

```python
# app/main.py
from fastapi import FastAPI
from app.api.v1.api import api_router
from app.core.middleware import add_middleware
from app.config.settings import get_settings

def create_app() -> FastAPI:
    settings = get_settings()
    
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=settings.api_description
    )
    
    # Add middleware
    add_middleware(app)
    
    # Include routers
    app.include_router(api_router, prefix="/api/v1")
    
    return app

app = create_app()
```

## 🔧 Dependency Injection

### Database Dependencies

```python
# app/api/dependencies.py
from typing import Annotated, AsyncGenerator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.database import get_db_session
from app.services.analysis import AnalysisService

async def get_analysis_service(
    session: AsyncSession = Depends(get_db_session)
) -> AnalysisService:
    return AnalysisService(session)

# Type alias for cleaner code
AnalysisServiceDep = Annotated[AnalysisService, Depends(get_analysis_service)]
```

### Configuration Dependencies

```python
# app/config/settings.py
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    database_url: str
    api_title: str = "NLP API"
    api_version: str = "1.0.0"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

# Dependency for settings
def get_settings_dep() -> Settings:
    return get_settings()
```

### Service Dependencies

```python
# app/services/analysis.py
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.schemas import AnalysisCreate, AnalysisUpdate, Analysis

class AnalysisService:
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_analysis(self, data: AnalysisCreate) -> Analysis:
        # Implementation
        pass
    
    async def get_analysis(self, analysis_id: int) -> Optional[Analysis]:
        # Implementation
        pass
```

## 📊 Request/Response Models

### Pydantic Models Best Practices

```python
# app/models/schemas.py
from pydantic import BaseModel, Field, validator, ConfigDict
from typing import Optional, List
from datetime import datetime
from enum import Enum

class AnalysisType(str, Enum):
    SENTIMENT = "sentiment"
    QUALITY = "quality"
    EMOTION = "emotion"

class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

class AnalysisBase(BaseModel):
    """Base model with common fields."""
    text_content: str = Field(..., min_length=1, max_length=10000)
    analysis_type: AnalysisType
    
    @validator('text_content')
    def validate_text_content(cls, v):
        if not v.strip():
            raise ValueError('Text content cannot be empty')
        return v.strip()

class AnalysisCreate(AnalysisBase):
    """Model for creating analysis."""
    optimization_tier: str = Field(default="standard")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text_content": "This is a sample text for analysis.",
                "analysis_type": "sentiment",
                "optimization_tier": "standard"
            }
        }
    )

class AnalysisUpdate(BaseModel):
    """Model for updating analysis."""
    status: Optional[AnalysisStatus] = None
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    processing_time_ms: Optional[float] = Field(None, ge=0.0)
    error_message: Optional[str] = None

class AnalysisResponse(AnalysisBase):
    """Model for analysis response."""
    id: int
    status: AnalysisStatus
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(default=1, ge=1, description="Page number")
    size: int = Field(default=20, ge=1, le=100, description="Page size")
    
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size

class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    items: List[AnalysisResponse]
    total: int
    page: int
    size: int
    pages: int
    
    @property
    def has_next(self) -> bool:
        return self.page < self.pages
    
    @property
    def has_prev(self) -> bool:
        return self.page > 1
```

### Query Parameters

```python
# app/api/v1/endpoints/analyses.py
from fastapi import APIRouter, Depends, Query, Path
from typing import Optional, List
from app.models.schemas import AnalysisType, AnalysisStatus

router = APIRouter()

@router.get("/analyses")
async def list_analyses(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    analysis_type: Optional[AnalysisType] = Query(None, description="Filter by analysis type"),
    status: Optional[AnalysisStatus] = Query(None, description="Filter by status"),
    order_by: str = Query("created_at", description="Order by field"),
    order_desc: bool = Query(True, description="Descending order")
):
    # Implementation
    pass

@router.get("/analyses/{analysis_id}")
async def get_analysis(
    analysis_id: int = Path(..., description="Analysis ID", ge=1)
):
    # Implementation
    pass
```

## ⚠️ Error Handling

### Custom Exceptions

```python
# app/core/exceptions.py
from fastapi import HTTPException, status
from typing import Optional, Any, Dict

class CustomHTTPException(HTTPException):
    """Custom HTTP exception with additional context."""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code

class ValidationError(CustomHTTPException):
    """Validation error."""
    def __init__(self, detail: str, error_code: Optional[str] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code=error_code
        )

class NotFoundError(CustomHTTPException):
    """Resource not found error."""
    def __init__(self, resource: str, resource_id: Any):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource} with id {resource_id} not found"
        )

class DatabaseError(CustomHTTPException):
    """Database operation error."""
    def __init__(self, detail: str, error_code: Optional[str] = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code=error_code
        )
```

### Exception Handlers

```python
# app/core/handlers.py
from fastapi import Request
from fastapi.responses import JSONResponse
from app.core.exceptions import CustomHTTPException
from app.models.schemas import ErrorResponse
from datetime import datetime

async def custom_http_exception_handler(
    request: Request, 
    exc: CustomHTTPException
) -> JSONResponse:
    """Handle custom HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=exc.error_code,
            timestamp=datetime.now(),
            path=request.url.path
        ).model_dump(),
        headers=exc.headers
    )

async def validation_exception_handler(
    request: Request, 
    exc: ValidationError
) -> JSONResponse:
    """Handle validation exceptions."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Validation error",
            detail=str(exc),
            timestamp=datetime.now(),
            path=request.url.path
        ).model_dump()
    )

async def general_exception_handler(
    request: Request, 
    exc: Exception
) -> JSONResponse:
    """Handle general exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred",
            timestamp=datetime.now(),
            path=request.url.path
        ).model_dump()
    )
```

### Error Response Models

```python
# app/models/schemas.py
class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(description="Error message")
    error_code: Optional[str] = Field(description="Error code")
    detail: Optional[str] = Field(description="Detailed error information")
    timestamp: datetime = Field(description="Error timestamp")
    path: Optional[str] = Field(description="Request path")
    request_id: Optional[str] = Field(description="Request ID for tracking")
```

## 🔒 Security

### Authentication

```python
# app/core/security.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional

security = HTTPBearer()

class SecurityService:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    security_service: SecurityService = Depends()
) -> dict:
    """Get current authenticated user."""
    token = credentials.credentials
    payload = security_service.verify_token(token)
    return payload
```

### CORS Configuration

```python
# app/core/middleware.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config.settings import get_settings

def add_cors_middleware(app: FastAPI):
    """Add CORS middleware."""
    settings = get_settings()
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
```

### Rate Limiting

```python
# app/core/rate_limiting.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import FastAPI

limiter = Limiter(key_func=get_remote_address)

def add_rate_limiting(app: FastAPI):
    """Add rate limiting to application."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Usage in endpoints
@limiter.limit("100/minute")
async def rate_limited_endpoint(request: Request):
    return {"message": "Rate limited endpoint"}
```

## ⚡ Performance

### Database Optimization

```python
# app/services/database.py
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, func
from typing import List, Optional

class DatabaseService:
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_analyses_paginated(
        self,
        skip: int = 0,
        limit: int = 20,
        analysis_type: Optional[str] = None
    ) -> tuple[List[Analysis], int]:
        """Get analyses with pagination and count."""
        # Build query
        query = select(Analysis)
        count_query = select(func.count(Analysis.id))
        
        if analysis_type:
            query = query.where(Analysis.analysis_type == analysis_type)
            count_query = count_query.where(Analysis.analysis_type == analysis_type)
        
        # Get total count
        total_result = await self.session.execute(count_query)
        total = total_result.scalar()
        
        # Get paginated results
        query = query.offset(skip).limit(limit)
        result = await self.session.execute(query)
        analyses = result.scalars().all()
        
        return analyses, total
```

### Caching

```python
# app/core/caching.py
from functools import wraps
from typing import Optional, Any
import redis.asyncio as redis
import json
from datetime import timedelta

class CacheService:
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
        await self.redis.setex(key, expire, json.dumps(value))
    
    async def delete(self, key: str):
        """Delete value from cache."""
        await self.redis.delete(key)

def cache_result(expire: int = 3600):
    """Cache decorator for function results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_service = CacheService("redis://localhost:6379")
            
            # Generate cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = await cache_service.get(cache_key)
            if cached_result:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache_service.set(cache_key, result, expire)
            
            return result
        return wrapper
    return decorator
```

### Background Tasks

```python
# app/api/v1/endpoints/analyses.py
from fastapi import BackgroundTasks

@router.post("/analyses/batch")
async def create_batch_analysis(
    data: BatchAnalysisCreate,
    background_tasks: BackgroundTasks,
    analysis_service: AnalysisServiceDep
):
    """Create batch analysis with background processing."""
    # Create batch record
    batch = await analysis_service.create_batch(data)
    
    # Add background task
    background_tasks.add_task(
        process_batch_analyses,
        batch.id,
        data.texts,
        data.analysis_type
    )
    
    return batch

async def process_batch_analyses(
    batch_id: int,
    texts: List[str],
    analysis_type: str
):
    """Background task to process batch analyses."""
    # Implementation
    pass
```

## 🧪 Testing

### Test Configuration

```python
# tests/conftest.py
import pytest
import asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from app.main import create_app
from app.models.database import Base

# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=True)
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()

@pytest.fixture
async def test_session(test_engine):
    """Create test database session."""
    async_session = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session

@pytest.fixture
def client(test_session):
    """Create test client."""
    app = create_app()
    
    # Override database dependency
    def override_get_db():
        return test_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
```

### API Tests

```python
# tests/test_api/test_analyses.py
import pytest
from fastapi.testclient import TestClient
from app.models.schemas import AnalysisCreate, AnalysisType

def test_create_analysis(client: TestClient):
    """Test creating analysis."""
    data = {
        "text_content": "This is a test text.",
        "analysis_type": "sentiment"
    }
    
    response = client.post("/api/v1/analyses", json=data)
    
    assert response.status_code == 200
    result = response.json()
    assert result["text_content"] == data["text_content"]
    assert result["analysis_type"] == data["analysis_type"]
    assert "id" in result

def test_get_analysis(client: TestClient):
    """Test getting analysis by ID."""
    # First create an analysis
    create_data = {
        "text_content": "Test text for retrieval.",
        "analysis_type": "sentiment"
    }
    create_response = client.post("/api/v1/analyses", json=create_data)
    analysis_id = create_response.json()["id"]
    
    # Then retrieve it
    response = client.get(f"/api/v1/analyses/{analysis_id}")
    
    assert response.status_code == 200
    result = response.json()
    assert result["id"] == analysis_id
    assert result["text_content"] == create_data["text_content"]

def test_list_analyses(client: TestClient):
    """Test listing analyses with pagination."""
    # Create multiple analyses
    for i in range(5):
        data = {
            "text_content": f"Test text {i}.",
            "analysis_type": "sentiment"
        }
        client.post("/api/v1/analyses", json=data)
    
    # List analyses
    response = client.get("/api/v1/analyses?page=1&size=3")
    
    assert response.status_code == 200
    result = response.json()
    assert len(result["items"]) == 3
    assert result["total"] >= 5
    assert result["page"] == 1
    assert result["size"] == 3

def test_validation_error(client: TestClient):
    """Test validation error handling."""
    data = {
        "text_content": "",  # Empty content should fail
        "analysis_type": "sentiment"
    }
    
    response = client.post("/api/v1/analyses", json=data)
    
    assert response.status_code == 422
    result = response.json()
    assert "error" in result

def test_not_found_error(client: TestClient):
    """Test not found error handling."""
    response = client.get("/api/v1/analyses/99999")
    
    assert response.status_code == 404
    result = response.json()
    assert "error" in result
```

### Service Tests

```python
# tests/test_services/test_analysis.py
import pytest
from app.services.analysis import AnalysisService
from app.models.schemas import AnalysisCreate, AnalysisType

@pytest.mark.asyncio
async def test_create_analysis(test_session):
    """Test creating analysis in service."""
    service = AnalysisService(test_session)
    
    data = AnalysisCreate(
        text_content="Test text for service.",
        analysis_type=AnalysisType.SENTIMENT
    )
    
    analysis = await service.create_analysis(data)
    
    assert analysis.text_content == data.text_content
    assert analysis.analysis_type == data.analysis_type
    assert analysis.id is not None

@pytest.mark.asyncio
async def test_get_analysis(test_session):
    """Test getting analysis from service."""
    service = AnalysisService(test_session)
    
    # Create analysis first
    data = AnalysisCreate(
        text_content="Test text for retrieval.",
        analysis_type=AnalysisType.SENTIMENT
    )
    created = await service.create_analysis(data)
    
    # Retrieve it
    analysis = await service.get_analysis(created.id)
    
    assert analysis is not None
    assert analysis.id == created.id
    assert analysis.text_content == data.text_content
```

## 📚 Documentation

### OpenAPI Customization

```python
# app/main.py
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="NLP Analysis API",
        version="1.0.0",
        description="API for natural language processing analysis",
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    
    # Add server information
    openapi_schema["servers"] = [
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        },
        {
            "url": "https://api.example.com",
            "description": "Production server"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

### Response Examples

```python
# app/models/schemas.py
class AnalysisResponse(BaseModel):
    id: int
    text_content: str
    analysis_type: AnalysisType
    status: AnalysisStatus
    created_at: datetime
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 1,
                "text_content": "This is a sample text for analysis.",
                "analysis_type": "sentiment",
                "status": "completed",
                "created_at": "2024-01-15T10:30:00Z"
            }
        }
    )
```

## 🚀 Production Deployment

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
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
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
      - DATABASE_URL=postgresql+asyncpg://postgres:password@db/nlp_db
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=nlp_db
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped

volumes:
  postgres_data:
```

### Environment Configuration

```bash
# .env.production
DATABASE_URL=postgresql+asyncpg://user:password@localhost/nlp_db
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-production-secret-key
API_TITLE=NLP Analysis API
API_VERSION=1.0.0
CORS_ORIGINS=["https://yourdomain.com"]
LOG_LEVEL=INFO
ENABLE_CACHING=true
RATE_LIMIT_PER_MINUTE=100
```

## 📊 Monitoring

### Health Check Endpoint

```python
# app/api/v1/endpoints/health.py
from fastapi import APIRouter, Depends
from app.models.schemas import HealthResponse
from app.services.database import DatabaseService

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check(db_service: DatabaseService = Depends()):
    """Health check endpoint."""
    try:
        # Check database connection
        db_healthy = await db_service.check_health()
        
        # Check other services
        redis_healthy = await check_redis_health()
        
        overall_healthy = db_healthy and redis_healthy
        
        return HealthResponse(
            status="healthy" if overall_healthy else "unhealthy",
            timestamp=datetime.now(),
            services={
                "database": db_healthy,
                "redis": redis_healthy
            }
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            error=str(e)
        )
```

### Logging Configuration

```python
# app/utils/logging.py
import structlog
import logging
from typing import Any, Dict

def configure_logging(log_level: str = "INFO", log_format: str = "json"):
    """Configure structured logging."""
    
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))

def get_logger(name: str) -> structlog.BoundLogger:
    """Get structured logger."""
    return structlog.get_logger(name)
```

### Metrics Collection

```python
# app/core/metrics.py
from prometheus_client import Counter, Histogram, Gauge
from fastapi import Request
import time

# Define metrics
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

ACTIVE_REQUESTS = Gauge(
    'http_active_requests',
    'Active HTTP requests',
    ['method', 'endpoint']
)

class MetricsMiddleware:
    """Middleware for collecting metrics."""
    
    async def __call__(self, request: Request, call_next):
        start_time = time.time()
        
        # Increment active requests
        ACTIVE_REQUESTS.labels(
            method=request.method,
            endpoint=request.url.path
        ).inc()
        
        # Process request
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        # Decrement active requests
        ACTIVE_REQUESTS.labels(
            method=request.method,
            endpoint=request.url.path
        ).dec()
        
        return response
```

## 🎯 Best Practices Summary

### ✅ Do's
- Use dependency injection for services and database connections
- Implement proper error handling with custom exceptions
- Use Pydantic models for request/response validation
- Implement rate limiting and security middleware
- Write comprehensive tests for all endpoints
- Use structured logging for better observability
- Implement health checks and monitoring
- Use background tasks for long-running operations
- Cache frequently accessed data
- Document your API with proper examples

### ❌ Don'ts
- Don't use global variables for state management
- Don't ignore error handling in async operations
- Don't expose sensitive information in error messages
- Don't skip input validation
- Don't use synchronous database operations in async endpoints
- Don't forget to implement proper CORS configuration
- Don't skip rate limiting for public APIs
- Don't ignore logging and monitoring
- Don't use hardcoded configuration values
- Don't skip security headers and middleware

This comprehensive guide provides all the essential patterns and best practices for building production-ready FastAPI applications with SQLAlchemy 2.0. 