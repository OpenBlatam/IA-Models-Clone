# FastAPI Best Practices for Security Tooling

## Overview

This document outlines best practices for implementing FastAPI-based security tooling APIs, focusing on performance, security, and maintainability.

## Table of Contents

1. [Project Structure](#project-structure)
2. [API Design Principles](#api-design-principles)
3. [Security Implementation](#security-implementation)
4. [Performance Optimization](#performance-optimization)
5. [Testing Strategy](#testing-strategy)
6. [Documentation Standards](#documentation-standards)
7. [Deployment Guidelines](#deployment-guidelines)

## Project Structure

```
cybersecurity/
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entry point
│   ├── dependencies.py         # Dependency injection
│   ├── middleware.py           # Custom middleware
│   ├── exceptions.py           # Custom exceptions
│   └── routes/
│       ├── __init__.py
│       ├── auth.py            # Authentication routes
│       ├── scans.py           # Scanning routes
│       ├── reports.py         # Reporting routes
│       └── admin.py           # Admin routes
├── core/
│   ├── config.py              # Configuration management
│   ├── security.py            # Security utilities
│   └── database.py            # Database connections
├── models/
│   ├── requests.py            # Request models
│   ├── responses.py           # Response models
│   └── database.py            # Database models
├── services/
│   ├── scan_service.py        # Scanning business logic
│   ├── auth_service.py        # Authentication logic
│   └── report_service.py      # Reporting logic
└── tests/
    ├── test_api.py            # API tests
    ├── test_integration.py    # Integration tests
    └── conftest.py            # Test configuration
```

## API Design Principles

### 1. RESTful Design

```python
# Good: RESTful endpoints
GET    /api/v1/scans              # List scans
POST   /api/v1/scans              # Create scan
GET    /api/v1/scans/{scan_id}    # Get specific scan
PUT    /api/v1/scans/{scan_id}    # Update scan
DELETE /api/v1/scans/{scan_id}    # Delete scan

# Good: Nested resources
GET    /api/v1/scans/{scan_id}/results
POST   /api/v1/scans/{scan_id}/results
```

### 2. Consistent Response Format

```python
from pydantic import BaseModel
from typing import Optional, Any, Dict

class APIResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class PaginatedResponse(BaseModel):
    items: list
    total: int
    page: int
    size: int
    pages: int
```

### 3. Error Handling

```python
from fastapi import HTTPException, status
from typing import Union

class SecurityAPIException(HTTPException):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(status_code=status_code, detail=detail)

class ValidationError(SecurityAPIException):
    def __init__(self, detail: str):
        super().__init__(detail=detail, status_code=422)

class AuthenticationError(SecurityAPIException):
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(detail=detail, status_code=401)

class AuthorizationError(SecurityAPIException):
    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(detail=detail, status_code=403)
```

## Security Implementation

### 1. Authentication & Authorization

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional

# Security configuration
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class SecurityService:
    def __init__(self):
        self.secret_key = SECRET_KEY
        self.algorithm = ALGORITHM
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        return pwd_context.hash(password)
    
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
            raise AuthenticationError("Invalid token")

# Dependency for authentication
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    security_service = SecurityService()
    payload = security_service.verify_token(credentials.credentials)
    user_id: str = payload.get("sub")
    if user_id is None:
        raise AuthenticationError("Invalid token")
    return user_id

# Role-based authorization
async def require_role(required_role: str):
    async def role_checker(current_user: str = Depends(get_current_user)):
        # Check user role from database or token
        user_role = get_user_role(current_user)
        if user_role != required_role:
            raise AuthorizationError(f"Required role: {required_role}")
        return current_user
    return role_checker
```

### 2. Rate Limiting

```python
from fastapi import Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import time
from collections import defaultdict

limiter = Limiter(key_func=get_remote_address)

class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
        self.limits = {
            "default": {"requests": 100, "window": 3600},  # 100 requests per hour
            "scan": {"requests": 10, "window": 3600},      # 10 scans per hour
            "admin": {"requests": 1000, "window": 3600}    # 1000 requests per hour
        }
    
    def is_allowed(self, client_id: str, endpoint: str = "default") -> bool:
        now = time.time()
        limit = self.limits.get(endpoint, self.limits["default"])
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < limit["window"]
        ]
        
        # Check if limit exceeded
        if len(self.requests[client_id]) >= limit["requests"]:
            return False
        
        # Add current request
        self.requests[client_id].append(now)
        return True

# Rate limiting decorator
def rate_limit(endpoint: str = "default"):
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            client_id = get_remote_address(request)
            rate_limiter = RateLimiter()
            
            if not rate_limiter.is_allowed(client_id, endpoint):
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded for {endpoint}"
                )
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator
```

### 3. Input Validation

```python
from pydantic import BaseModel, validator, Field
from typing import List, Optional
import ipaddress

class ScanRequest(BaseModel):
    targets: List[str] = Field(..., min_items=1, max_items=100)
    ports: Optional[List[int]] = Field(None, ge=1, le=65535)
    scan_type: str = Field(..., regex="^(tcp|udp|syn)$")
    timeout: float = Field(5.0, ge=0.1, le=60.0)
    
    @validator('targets')
    def validate_targets(cls, v):
        for target in v:
            try:
                ipaddress.ip_address(target)
            except ValueError:
                # Check if it's a valid hostname
                if not target.replace('.', '').replace('-', '').isalnum():
                    raise ValueError(f"Invalid target: {target}")
        return v
    
    @validator('ports')
    def validate_ports(cls, v):
        if v:
            for port in v:
                if not (1 <= port <= 65535):
                    raise ValueError(f"Invalid port: {port}")
        return v

class ScanResponse(BaseModel):
    scan_id: str
    status: str
    progress: float = Field(..., ge=0.0, le=100.0)
    results: Optional[dict] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
```

## Performance Optimization

### 1. Async Operations

```python
from fastapi import BackgroundTasks
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ScanService:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.active_scans = {}
    
    async def start_scan(self, scan_request: ScanRequest) -> str:
        scan_id = generate_scan_id()
        
        # Start scan in background
        task = asyncio.create_task(
            self._run_scan(scan_id, scan_request)
        )
        
        self.active_scans[scan_id] = {
            "task": task,
            "status": "running",
            "progress": 0.0,
            "results": None,
            "error": None
        }
        
        return scan_id
    
    async def _run_scan(self, scan_id: str, scan_request: ScanRequest):
        try:
            # Run scan in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                self._perform_scan,
                scan_request
            )
            
            self.active_scans[scan_id].update({
                "status": "completed",
                "progress": 100.0,
                "results": results
            })
            
        except Exception as e:
            self.active_scans[scan_id].update({
                "status": "failed",
                "error": str(e)
            })
    
    def _perform_scan(self, scan_request: ScanRequest) -> dict:
        # CPU-intensive scanning logic
        results = {}
        total_targets = len(scan_request.targets)
        
        for i, target in enumerate(scan_request.targets):
            # Simulate scanning
            target_results = self._scan_target(target, scan_request.ports)
            results[target] = target_results
            
            # Update progress
            progress = (i + 1) / total_targets * 100
            self.active_scans[scan_id]["progress"] = progress
        
        return results
```

### 2. Caching

```python
from functools import lru_cache
import redis
import json

class CacheService:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.default_ttl = 3600  # 1 hour
    
    async def get(self, key: str) -> Optional[dict]:
        try:
            data = self.redis_client.get(key)
            return json.loads(data) if data else None
        except Exception:
            return None
    
    async def set(self, key: str, value: dict, ttl: int = None):
        try:
            ttl = ttl or self.default_ttl
            self.redis_client.setex(key, ttl, json.dumps(value))
        except Exception:
            pass  # Fail silently for caching
    
    async def delete(self, key: str):
        try:
            self.redis_client.delete(key)
        except Exception:
            pass

# Cache decorator
def cache_result(ttl: int = 3600):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache_service = CacheService()
            
            # Generate cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = await cache_service.get(cache_key)
            if cached_result:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache_service.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator
```

### 3. Database Optimization

```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Index, text

# Database configuration
DATABASE_URL = "postgresql+asyncpg://user:password@localhost/security_db"

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True
)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Database dependency
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Optimized queries
class ScanRepository:
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_scans_by_user(self, user_id: str, limit: int = 50, offset: int = 0):
        query = text("""
            SELECT s.*, COUNT(r.id) as result_count
            FROM scans s
            LEFT JOIN scan_results r ON s.id = r.scan_id
            WHERE s.user_id = :user_id
            GROUP BY s.id
            ORDER BY s.created_at DESC
            LIMIT :limit OFFSET :offset
        """)
        
        result = await self.db.execute(
            query,
            {"user_id": user_id, "limit": limit, "offset": offset}
        )
        return result.fetchall()
    
    async def get_scan_with_results(self, scan_id: str):
        query = text("""
            SELECT s.*, 
                   json_agg(
                       json_build_object(
                           'id', r.id,
                           'target', r.target,
                           'port', r.port,
                           'status', r.status
                       )
                   ) as results
            FROM scans s
            LEFT JOIN scan_results r ON s.id = r.scan_id
            WHERE s.id = :scan_id
            GROUP BY s.id
        """)
        
        result = await self.db.execute(query, {"scan_id": scan_id})
        return result.fetchone()
```

## Testing Strategy

### 1. Unit Tests

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from api.main import app

client = TestClient(app)

class TestScanAPI:
    @pytest.mark.asyncio
    async def test_create_scan_success(self):
        """Test successful scan creation."""
        scan_data = {
            "targets": ["192.168.1.1"],
            "ports": [80, 443],
            "scan_type": "tcp",
            "timeout": 5.0
        }
        
        with patch('services.scan_service.ScanService.start_scan') as mock_start:
            mock_start.return_value = "scan_123"
            
            response = client.post("/api/v1/scans", json=scan_data)
            
            assert response.status_code == 201
            assert response.json()["scan_id"] == "scan_123"
            mock_start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_scan_invalid_target(self):
        """Test scan creation with invalid target."""
        scan_data = {
            "targets": ["invalid-target"],
            "ports": [80],
            "scan_type": "tcp"
        }
        
        response = client.post("/api/v1/scans", json=scan_data)
        
        assert response.status_code == 422
        assert "Invalid target" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_get_scan_not_found(self):
        """Test getting non-existent scan."""
        response = client.get("/api/v1/scans/non-existent")
        
        assert response.status_code == 404
        assert "Scan not found" in response.json()["detail"]

class TestAuthentication:
    def test_valid_token(self):
        """Test authentication with valid token."""
        headers = {"Authorization": "Bearer valid_token"}
        
        with patch('core.security.verify_token') as mock_verify:
            mock_verify.return_value = {"sub": "user123"}
            
            response = client.get("/api/v1/profile", headers=headers)
            
            assert response.status_code == 200
            mock_verify.assert_called_once_with("valid_token")
    
    def test_invalid_token(self):
        """Test authentication with invalid token."""
        headers = {"Authorization": "Bearer invalid_token"}
        
        with patch('core.security.verify_token') as mock_verify:
            mock_verify.side_effect = AuthenticationError("Invalid token")
            
            response = client.get("/api/v1/profile", headers=headers)
            
            assert response.status_code == 401
            assert "Invalid token" in response.json()["detail"]
```

### 2. Integration Tests

```python
import pytest
from httpx import AsyncClient
from api.main import app
from core.database import get_db
from models.database import Base

@pytest.fixture
async def test_db():
    """Create test database."""
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield
    
    # Clean up
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest.fixture
async def client(test_db):
    """Create test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

class TestScanIntegration:
    @pytest.mark.asyncio
    async def test_full_scan_workflow(self, client):
        """Test complete scan workflow."""
        # 1. Create scan
        scan_data = {
            "targets": ["127.0.0.1"],
            "ports": [80, 443],
            "scan_type": "tcp"
        }
        
        response = await client.post("/api/v1/scans", json=scan_data)
        assert response.status_code == 201
        
        scan_id = response.json()["scan_id"]
        
        # 2. Check scan status
        response = await client.get(f"/api/v1/scans/{scan_id}")
        assert response.status_code == 200
        
        # 3. Wait for completion
        import asyncio
        for _ in range(10):  # Wait up to 10 seconds
            response = await client.get(f"/api/v1/scans/{scan_id}")
            status = response.json()["status"]
            
            if status == "completed":
                break
            
            await asyncio.sleep(1)
        
        assert response.json()["status"] == "completed"
        
        # 4. Get results
        response = await client.get(f"/api/v1/scans/{scan_id}/results")
        assert response.status_code == 200
        assert "results" in response.json()
```

### 3. Performance Tests

```python
import pytest
import asyncio
import time
from httpx import AsyncClient

class TestPerformance:
    @pytest.mark.asyncio
    async def test_concurrent_scans(self, client):
        """Test concurrent scan creation."""
        scan_data = {
            "targets": ["127.0.0.1"],
            "ports": [80],
            "scan_type": "tcp"
        }
        
        start_time = time.time()
        
        # Create 10 concurrent scans
        tasks = [
            client.post("/api/v1/scans", json=scan_data)
            for _ in range(10)
        ]
        
        responses = await asyncio.gather(*tasks)
        
        duration = time.time() - start_time
        
        # All should succeed
        assert all(r.status_code == 201 for r in responses)
        
        # Should complete within reasonable time
        assert duration < 5.0
    
    @pytest.mark.asyncio
    async def test_large_scan_creation(self, client):
        """Test creation of large scan."""
        scan_data = {
            "targets": [f"192.168.1.{i}" for i in range(1, 101)],  # 100 targets
            "ports": list(range(1, 1025)),  # First 1024 ports
            "scan_type": "tcp"
        }
        
        start_time = time.time()
        
        response = await client.post("/api/v1/scans", json=scan_data)
        
        duration = time.time() - start_time
        
        assert response.status_code == 201
        assert duration < 10.0  # Should handle large scans efficiently
```

## Documentation Standards

### 1. API Documentation

```python
from fastapi import APIRouter, Depends, HTTPException
from typing import List

router = APIRouter(prefix="/api/v1/scans", tags=["scans"])

@router.post(
    "/",
    response_model=ScanResponse,
    status_code=201,
    summary="Create a new security scan",
    description="""
    Create a new security scan with the specified targets and parameters.
    
    - **targets**: List of IP addresses or hostnames to scan
    - **ports**: Optional list of ports to scan (defaults to common ports)
    - **scan_type**: Type of scan (tcp, udp, or syn)
    - **timeout**: Scan timeout in seconds
    
    Returns a scan ID that can be used to track progress and retrieve results.
    """,
    responses={
        201: {
            "description": "Scan created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "scan_id": "scan_123",
                        "status": "running",
                        "progress": 0.0,
                        "created_at": "2023-01-01T00:00:00Z"
                    }
                }
            }
        },
        422: {
            "description": "Validation error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "targets"],
                                "msg": "Invalid target: invalid-host",
                                "type": "value_error"
                            }
                        ]
                    }
                }
            }
        }
    }
)
async def create_scan(
    scan_request: ScanRequest,
    current_user: str = Depends(get_current_user),
    scan_service: ScanService = Depends(get_scan_service)
):
    """
    Create a new security scan.
    
    This endpoint initiates a security scan with the provided parameters.
    The scan will run asynchronously and can be monitored using the returned scan ID.
    
    Args:
        scan_request: The scan configuration
        current_user: The authenticated user
        scan_service: The scan service dependency
    
    Returns:
        ScanResponse: The created scan information
    
    Raises:
        HTTPException: If the scan cannot be created
    """
    try:
        scan_id = await scan_service.start_scan(scan_request)
        return ScanResponse(
            scan_id=scan_id,
            status="running",
            progress=0.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 2. OpenAPI Configuration

```python
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Security Scanning API",
        version="1.0.0",
        description="""
        # Security Scanning API
        
        A comprehensive API for performing security scans and vulnerability assessments.
        
        ## Features
        
        - **Port Scanning**: TCP, UDP, and SYN scans
        - **Vulnerability Assessment**: Automated vulnerability detection
        - **Report Generation**: Detailed security reports
        - **Real-time Monitoring**: Live scan progress tracking
        
        ## Authentication
        
        All endpoints require Bearer token authentication.
        
        ## Rate Limiting
        
        - Default: 100 requests per hour
        - Scan operations: 10 scans per hour
        - Admin operations: 1000 requests per hour
        
        ## Error Codes
        
        - `400`: Bad Request
        - `401`: Unauthorized
        - `403`: Forbidden
        - `404`: Not Found
        - `422`: Validation Error
        - `429`: Rate Limit Exceeded
        - `500`: Internal Server Error
        """,
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    # Add global security requirement
    openapi_schema["security"] = [{"BearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app = FastAPI(
    title="Security Scanning API",
    description="Comprehensive security scanning and vulnerability assessment API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.openapi = custom_openapi
```

## Deployment Guidelines

### 1. Production Configuration

```python
# config/production.py
import os
from pydantic import BaseSettings

class ProductionSettings(BaseSettings):
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    DATABASE_POOL_SIZE: int = int(os.getenv("DATABASE_POOL_SIZE", "20"))
    DATABASE_MAX_OVERFLOW: int = int(os.getenv("DATABASE_MAX_OVERFLOW", "30"))
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Rate Limiting
    RATE_LIMIT_DEFAULT: int = int(os.getenv("RATE_LIMIT_DEFAULT", "100"))
    RATE_LIMIT_SCAN: int = int(os.getenv("RATE_LIMIT_SCAN", "10"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json")
    
    # Monitoring
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    METRICS_PORT: int = int(os.getenv("METRICS_PORT", "9090"))
    
    class Config:
        env_file = ".env"

settings = ProductionSettings()
```

### 2. Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
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

### 3. Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: security-api
  labels:
    app: security-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: security-api
  template:
    metadata:
      labels:
        app: security-api
    spec:
      containers:
      - name: security-api
        image: security-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: api-secret
              key: secret-key
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: security-api-service
spec:
  selector:
    app: security-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 4. Monitoring and Logging

```python
# monitoring/prometheus.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_SCANS = Gauge('active_scans', 'Number of active scans')
SCAN_DURATION = Histogram('scan_duration_seconds', 'Scan duration')

# Middleware for metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    REQUEST_DURATION.observe(duration)
    
    return response

# Logging configuration
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        
        if hasattr(record, 'scan_id'):
            log_entry['scan_id'] = record.scan_id
        
        return json.dumps(log_entry)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/security-api.log')
    ]
)

# Set formatter
for handler in logging.root.handlers:
    handler.setFormatter(JSONFormatter())
```

This comprehensive FastAPI documentation provides best practices for building secure, performant, and maintainable security tooling APIs. The implementation includes proper authentication, authorization, rate limiting, input validation, caching, database optimization, comprehensive testing, and production deployment guidelines. 