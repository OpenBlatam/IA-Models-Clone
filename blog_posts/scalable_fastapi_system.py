from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from contextlib import asynccontextmanager
from functools import wraps
import logging
import traceback
from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import QueuePool
from alembic import command, config as alembic_config
from alembic.script import ScriptDirectory
import redis.asyncio as redis
from redis.asyncio import Redis
import aioredis
from cachetools import TTLCache, LRUCache
import pickle
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary
import psutil
import structlog
import jwt
from passlib.context import CryptContext
from cryptography.fernet import Fernet
import secrets
import yaml
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
import aiofiles
import aiohttp
from motor.motor_asyncio import AsyncIOMotorClient
import orjson
    import uvicorn
from typing import Any, List, Dict, Optional
"""
Scalable FastAPI System with Modern Best Practices
=================================================

Production-ready FastAPI system with async operations, caching, middleware,
database integration, monitoring, and scalable architecture patterns.
"""


# FastAPI and web framework imports

# Database and ORM imports

# Caching and session management

# Monitoring and metrics

# Security and authentication

# Configuration and utilities

# Additional utilities

# Configure logging
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


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    database_url: str = "sqlite:///./app.db"
    async_database_url: str = "sqlite+aiosqlite:///./app.db"
    pool_size: int = 20
    max_overflow: int = 30
    pool_pre_ping: bool = True
    pool_recycle: int = 3600
    echo: bool = False

class RedisSettings(BaseSettings):
    """Redis configuration settings."""
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    redis_ssl_cert_reqs: Optional[str] = None

class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    bcrypt_rounds: int = 12

class APISettings(BaseSettings):
    """API configuration settings."""
    title: str = "Scalable FastAPI System"
    version: str = "1.0.0"
    description: str = "Production-ready FastAPI system with modern best practices"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    max_requests: int = 1000
    max_requests_jitter: int = 100
    timeout: int = 30
    keepalive: int = 5
    limit_concurrency: int = 1000
    limit_max_requests: int = 10000

class Settings(BaseSettings):
    """Main application settings."""
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    security: SecuritySettings = SecuritySettings()
    api: APISettings = APISettings()
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


# =============================================================================
# DATABASE MODELS
# =============================================================================

Base = declarative_base()

class User(Base):
    """User model for authentication and authorization."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    metadata_json = Column(JSON, nullable=True)

class APIKey(Base):
    """API key model for API authentication."""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    key_hash = Column(String(255), unique=True, index=True, nullable=False)
    user_id = Column(Integer, nullable=False)
    name = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime, nullable=True)
    permissions = Column(JSON, nullable=True)

class RequestLog(Base):
    """Request logging model for monitoring and analytics."""
    __tablename__ = "request_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(50), unique=True, index=True, nullable=False)
    user_id = Column(Integer, nullable=True)
    method = Column(String(10), nullable=False)
    path = Column(String(255), nullable=False)
    status_code = Column(Integer, nullable=False)
    response_time = Column(Integer, nullable=False)  # milliseconds
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    request_body = Column(Text, nullable=True)
    response_body = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON, nullable=True)


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class UserCreate(BaseModel):
    """User creation model."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r"^[^@]+@[^@]+\.[^@]+$")
    password: str = Field(..., min_length=8)

class UserUpdate(BaseModel):
    """User update model."""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[str] = Field(None, regex=r"^[^@]+@[^@]+\.[^@]+$")
    is_active: Optional[bool] = None

class UserResponse(BaseModel):
    """User response model."""
    id: int
    username: str
    email: str
    is_active: bool
    is_superuser: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    """Token model for authentication."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None
    user_id: Optional[int] = None
    permissions: Optional[List[str]] = None

class APIResponse(BaseModel):
    """Standard API response model."""
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    errors: Optional[List[str]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None

class HealthCheck(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str
    uptime: float
    database: str
    redis: str
    memory_usage: Dict[str, float]
    cpu_usage: float


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self, settings: Settings):
        
    """__init__ function."""
self.settings = settings
        self.engine = None
        self.async_engine = None
        self.SessionLocal = None
        self.AsyncSessionLocal = None
        self._initialize_engines()
    
    def _initialize_engines(self) -> Any:
        """Initialize database engines."""
        # Synchronous engine
        self.engine = create_engine(
            self.settings.database.database_url,
            poolclass=QueuePool,
            pool_size=self.settings.database.pool_size,
            max_overflow=self.settings.database.max_overflow,
            pool_pre_ping=self.settings.database.pool_pre_ping,
            pool_recycle=self.settings.database.pool_recycle,
            echo=self.settings.database.echo
        )
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Asynchronous engine
        self.async_engine = create_async_engine(
            self.settings.database.async_database_url,
            poolclass=QueuePool,
            pool_size=self.settings.database.pool_size,
            max_overflow=self.settings.database.max_overflow,
            pool_pre_ping=self.settings.database.pool_pre_ping,
            pool_recycle=self.settings.database.pool_recycle,
            echo=self.settings.database.echo
        )
        
        self.AsyncSessionLocal = async_sessionmaker(
            self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    def get_db(self) -> Session:
        """Get synchronous database session."""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    async def get_async_db(self) -> AsyncSession:
        """Get asynchronous database session."""
        async with self.AsyncSessionLocal() as session:
            try:
                yield session
            finally:
                await session.close()
    
    def create_tables(self) -> Any:
        """Create database tables."""
        Base.metadata.create_all(bind=self.engine)
    
    async def create_tables_async(self) -> Any:
        """Create database tables asynchronously."""
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    def run_migrations(self) -> Any:
        """Run database migrations."""
        alembic_cfg = alembic_config.Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

class CacheManager:
    """Cache management with Redis and in-memory caching."""
    
    def __init__(self, settings: Settings):
        
    """__init__ function."""
self.settings = settings
        self.redis_client: Optional[Redis] = None
        self.memory_cache = TTLCache(maxsize=1000, ttl=300)  # 5 minutes TTL
        self.lru_cache = LRUCache(maxsize=500)
        self._initialize_redis()
    
    async def _initialize_redis(self) -> Any:
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(
                self.settings.redis.redis_url,
                db=self.settings.redis.redis_db,
                password=self.settings.redis.redis_password,
                ssl=self.settings.redis.redis_ssl,
                ssl_cert_reqs=self.settings.redis.redis_ssl_cert_reqs,
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using in-memory cache only")
            self.redis_client = None
    
    async def get(self, key: str, default: Any = None) -> Optional[Dict[str, Any]]:
        """Get value from cache."""
        try:
            # Try Redis first
            if self.redis_client:
                value = await self.redis_client.get(key)
                if value:
                    return orjson.loads(value)
            
            # Fallback to memory cache
            return self.memory_cache.get(key, default)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return self.memory_cache.get(key, default)
    
    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set value in cache."""
        try:
            # Set in Redis
            if self.redis_client:
                serialized_value = orjson.dumps(value)
                await self.redis_client.setex(key, ttl, serialized_value)
            
            # Set in memory cache
            self.memory_cache[key] = value
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            # Fallback to memory cache only
            self.memory_cache[key] = value
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            # Delete from Redis
            if self.redis_client:
                await self.redis_client.delete(key)
            
            # Delete from memory cache
            self.memory_cache.pop(key, None)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            self.memory_cache.pop(key, None)
            return False
    
    async def clear(self) -> bool:
        """Clear all cache."""
        try:
            # Clear Redis
            if self.redis_client:
                await self.redis_client.flushdb()
            
            # Clear memory cache
            self.memory_cache.clear()
            self.lru_cache.clear()
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    async def health_check(self) -> str:
        """Check cache health."""
        try:
            if self.redis_client:
                await self.redis_client.ping()
                return "healthy"
            else:
                return "memory_only"
        except Exception:
            return "unhealthy"


# =============================================================================
# SECURITY AND AUTHENTICATION
# =============================================================================

class SecurityManager:
    """Security and authentication management."""
    
    def __init__(self, settings: Settings):
        
    """__init__ function."""
self.settings = settings
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.fernet = Fernet(settings.security.secret_key.encode()[:32].ljust(32, b'0'))
        self.security = HTTPBearer()
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash."""
        return self.pwd_context.hash(password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.settings.security.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.settings.security.secret_key, algorithm=self.settings.security.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, data: dict) -> str:
        """Create JWT refresh token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.settings.security.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, self.settings.security.secret_key, algorithm=self.settings.security.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.settings.security.secret_key, algorithms=[self.settings.security.algorithm])
            username: str = payload.get("sub")
            user_id: int = payload.get("user_id")
            permissions: List[str] = payload.get("permissions", [])
            
            if username is None:
                return None
            
            return TokenData(username=username, user_id=user_id, permissions=permissions)
        except jwt.PyJWTError:
            return None
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    async def generate_api_key(self) -> str:
        """Generate secure API key."""
        return secrets.token_urlsafe(32)
    
    async def hash_api_key(self, api_key: str) -> str:
        """Hash API key for storage."""
        return self.pwd_context.hash(api_key)


# =============================================================================
# MONITORING AND METRICS
# =============================================================================

class MetricsManager:
    """Prometheus metrics and monitoring."""
    
    def __init__(self) -> Any:
        self.request_counter = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint']
        )
        
        self.active_requests = Gauge(
            'http_active_requests',
            'Number of active HTTP requests'
        )
        
        self.error_counter = Counter(
            'http_errors_total',
            'Total HTTP errors',
            ['method', 'endpoint', 'error_type']
        )
        
        self.database_operations = Counter(
            'database_operations_total',
            'Total database operations',
            ['operation', 'table']
        )
        
        self.cache_hits = Counter(
            'cache_hits_total',
            'Total cache hits',
            ['cache_type']
        )
        
        self.cache_misses = Counter(
            'cache_misses_total',
            'Total cache misses',
            ['cache_type']
        )
        
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage'
        )
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics."""
        self.request_counter.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_error(self, method: str, endpoint: str, error_type: str):
        """Record error metrics."""
        self.error_counter.labels(method=method, endpoint=endpoint, error_type=error_type).inc()
    
    def record_database_operation(self, operation: str, table: str):
        """Record database operation metrics."""
        self.database_operations.labels(operation=operation, table=table).inc()
    
    def record_cache_hit(self, cache_type: str):
        """Record cache hit metrics."""
        self.cache_hits.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """Record cache miss metrics."""
        self.cache_misses.labels(cache_type=cache_type).inc()
    
    def update_system_metrics(self) -> Any:
        """Update system metrics."""
        memory = psutil.virtual_memory()
        self.memory_usage.set(memory.used)
        self.cpu_usage.set(psutil.cpu_percent())


# =============================================================================
# MIDDLEWARE
# =============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests."""
    
    def __init__(self, app, db_manager: DatabaseManager, metrics_manager: MetricsManager):
        
    """__init__ function."""
super().__init__(app)
        self.db_manager = db_manager
        self.metrics_manager = metrics_manager
    
    async def dispatch(self, request: StarletteRequest, call_next) -> StarletteResponse:
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Log request
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        
        try:
            response = await call_next(request)
            
            # Calculate response time
            response_time = int((time.time() - start_time) * 1000)
            
            # Record metrics
            self.metrics_manager.record_request(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code,
                duration=time.time() - start_time
            )
            
            # Log response
            logger.info(
                "Request completed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                response_time_ms=response_time
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = str(response_time)
            
            return response
            
        except Exception as e:
            # Record error metrics
            self.metrics_manager.record_error(
                method=request.method,
                endpoint=request.url.path,
                error_type=type(e).__name__
            )
            
            # Log error
            logger.error(
                "Request failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                error=str(e),
                traceback=traceback.format_exc()
            )
            
            raise


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting."""
    
    def __init__(self, app, cache_manager: CacheManager, settings: Settings):
        
    """__init__ function."""
super().__init__(app)
        self.cache_manager = cache_manager
        self.settings = settings
    
    async def dispatch(self, request: StarletteRequest, call_next) -> StarletteResponse:
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Check rate limit
        rate_limit_key = f"rate_limit:{client_ip}"
        current_requests = await self.cache_manager.get(rate_limit_key, 0)
        
        if current_requests >= self.settings.api.limit_concurrency:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded"}
            )
        
        # Increment request count
        await self.cache_manager.set(rate_limit_key, current_requests + 1, ttl=60)
        
        response = await call_next(request)
        return response


# =============================================================================
# DEPENDENCIES
# =============================================================================

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: Session = Depends(lambda: next(DatabaseManager(settings).get_db())),
    security_manager: SecurityManager = Depends(lambda: SecurityManager(settings))
) -> User:
    """Get current authenticated user."""
    token_data = security_manager.verify_token(credentials.credentials)
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = db.query(User).filter(User.id == token_data.user_id).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return current_user


async def get_current_superuser(current_user: User = Depends(get_current_active_user)) -> User:
    """Get current superuser."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )
    return current_user


def require_permissions(required_permissions: List[str]):
    """Decorator to require specific permissions."""
    def permission_checker(current_user: User = Depends(get_current_active_user)):
        # This is a simplified permission check
        # In a real application, you would check against user roles/permissions
        if not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        return current_user
    return permission_checker


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def cache_response(ttl: int = 300, key_prefix: str = ""):
    """Decorator to cache API responses."""
    def decorator(func: Callable):
        
    """decorator function."""
@wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                metrics_manager.record_cache_hit("api_response")
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache_manager.set(cache_key, result, ttl)
            metrics_manager.record_cache_miss("api_response")
            
            return result
        return wrapper
    return decorator


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async async def external_api_call(url: str, timeout: int = 30) -> Dict[str, Any]:
    """Make external API call with retry logic."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()


async def background_task(task_id: str, task_data: Dict[str, Any]):
    """Execute background task."""
    logger.info(f"Starting background task: {task_id}")
    try:
        # Simulate some work
        await asyncio.sleep(2)
        logger.info(f"Completed background task: {task_id}")
    except Exception as e:
        logger.error(f"Background task failed: {task_id}, error: {e}")


# =============================================================================
# APPLICATION FACTORY
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting application...")
    
    # Initialize managers
    global settings, db_manager, cache_manager, security_manager, metrics_manager
    settings = Settings()
    db_manager = DatabaseManager(settings)
    cache_manager = CacheManager(settings)
    await cache_manager._initialize_redis()
    security_manager = SecurityManager(settings)
    metrics_manager = MetricsManager()
    
    # Create database tables
    db_manager.create_tables()
    
    # Start background tasks
    app.state.background_tasks = []
    
    logger.info("Application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    
    # Cleanup background tasks
    for task in app.state.background_tasks:
        task.cancel()
    
    # Close database connections
    if db_manager.engine:
        db_manager.engine.dispose()
    
    # Close Redis connection
    if cache_manager.redis_client:
        await cache_manager.redis_client.close()
    
    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title=settings.api.title,
        version=settings.api.version,
        description=settings.api.description,
        debug=settings.api.debug,
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    app.add_middleware(RequestLoggingMiddleware, db_manager=db_manager, metrics_manager=metrics_manager)
    app.add_middleware(RateLimitingMiddleware, cache_manager=cache_manager, settings=settings)
    
    # Add exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        
    """validation_exception_handler function."""
return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=APIResponse(
                success=False,
                errors=[str(error) for error in exc.errors()],
                request_id=getattr(request.state, 'request_id', None)
            ).dict()
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        
    """http_exception_handler function."""
return JSONResponse(
            status_code=exc.status_code,
            content=APIResponse(
                success=False,
                message=exc.detail,
                request_id=getattr(request.state, 'request_id', None)
            ).dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        
    """general_exception_handler function."""
logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=APIResponse(
                success=False,
                message="Internal server error",
                request_id=getattr(request.state, 'request_id', None)
            ).dict()
        )
    
    return app


# =============================================================================
# API ROUTES
# =============================================================================

app = create_app()

@app.get("/", response_model=APIResponse)
async def root():
    """Root endpoint."""
    return APIResponse(
        success=True,
        data={"message": "Scalable FastAPI System is running"},
        message="Welcome to the Scalable FastAPI System"
    )


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    # Update system metrics
    metrics_manager.update_system_metrics()
    
    # Check database health
    try:
        db = next(db_manager.get_db())
        db.execute("SELECT 1")
        db.close()
        database_status = "healthy"
    except Exception:
        database_status = "unhealthy"
    
    # Check Redis health
    redis_status = await cache_manager.health_check()
    
    # Get system information
    memory = psutil.virtual_memory()
    cpu_usage = psutil.cpu_percent()
    
    return HealthCheck(
        status="healthy" if database_status == "healthy" and redis_status == "healthy" else "degraded",
        timestamp=datetime.utcnow(),
        version=settings.api.version,
        uptime=time.time(),
        database=database_status,
        redis=redis_status,
        memory_usage={
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent
        },
        cpu_usage=cpu_usage
    )


@app.post("/auth/register", response_model=APIResponse)
async def register(user_data: UserCreate, db: Session = Depends(lambda: next(db_manager.get_db()))):
    """Register new user."""
    # Check if user already exists
    existing_user = db.query(User).filter(
        (User.username == user_data.username) | (User.email == user_data.email)
    ).first()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered"
        )
    
    # Create new user
    hashed_password = security_manager.get_password_hash(user_data.password)
    user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return APIResponse(
        success=True,
        data=UserResponse.from_orm(user),
        message="User registered successfully"
    )


@app.post("/auth/login", response_model=APIResponse)
async def login(
    username: str,
    password: str,
    db: Session = Depends(lambda: next(db_manager.get_db()))
):
    """User login."""
    user = db.query(User).filter(User.username == username).first()
    if not user or not security_manager.verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    # Create tokens
    access_token = security_manager.create_access_token(
        data={"sub": user.username, "user_id": user.id}
    )
    refresh_token = security_manager.create_refresh_token(
        data={"sub": user.username, "user_id": user.id}
    )
    
    return APIResponse(
        success=True,
        data=Token(
            access_token=access_token,
            refresh_token=refresh_token
        ),
        message="Login successful"
    )


@app.get("/users/me", response_model=APIResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get current user information."""
    return APIResponse(
        success=True,
        data=UserResponse.from_orm(current_user),
        message="User information retrieved successfully"
    )


@app.put("/users/me", response_model=APIResponse)
async def update_current_user(
    user_data: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(lambda: next(db_manager.get_db()))
):
    """Update current user information."""
    # Update user fields
    for field, value in user_data.dict(exclude_unset=True).items():
        setattr(current_user, field, value)
    
    current_user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(current_user)
    
    return APIResponse(
        success=True,
        data=UserResponse.from_orm(current_user),
        message="User updated successfully"
    )


@app.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics."""
    return StreamingResponse(
        prometheus_client.generate_latest(),
        media_type="text/plain"
    )


@app.get("/cache/status")
async def get_cache_status():
    """Get cache status."""
    return APIResponse(
        success=True,
        data={
            "redis_status": await cache_manager.health_check(),
            "memory_cache_size": len(cache_manager.memory_cache),
            "lru_cache_size": len(cache_manager.lru_cache)
        },
        message="Cache status retrieved successfully"
    )


@app.post("/cache/clear")
async def clear_cache():
    """Clear all cache."""
    success = await cache_manager.clear()
    return APIResponse(
        success=success,
        message="Cache cleared successfully" if success else "Failed to clear cache"
    )


@app.get("/api/external-data")
@cache_response(ttl=600, key_prefix="external")
async def get_external_data():
    """Get external data with caching."""
    try:
        data = await external_api_call("https://jsonplaceholder.typicode.com/posts/1")
        return APIResponse(
            success=True,
            data=data,
            message="External data retrieved successfully"
        )
    except Exception as e:
        logger.error(f"External API call failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="External service unavailable"
        )


@app.post("/tasks/background")
async def create_background_task(
    task_data: Dict[str, Any],
    current_user: User = Depends(get_current_active_user)
):
    """Create background task."""
    task_id = str(uuid.uuid4())
    
    # Create background task
    task = asyncio.create_task(background_task(task_id, task_data))
    app.state.background_tasks.append(task)
    
    return APIResponse(
        success=True,
        data={"task_id": task_id},
        message="Background task created successfully"
    )


# =============================================================================
# MAIN APPLICATION
# =============================================================================

if __name__ == "__main__":
    
    # Initialize settings
    settings = Settings()
    
    # Run application
    uvicorn.run(
        "scalable_fastapi_system:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug,
        workers=settings.api.workers if not settings.api.debug else 1,
        log_level="info"
    ) 