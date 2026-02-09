from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import logging
from typing import Any, Optional, Dict, List, Callable, Awaitable, Union, Type, TypeVar
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from enum import Enum
import redis.asyncio as redis
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel, Field
import structlog
            from jose import JWTError, jwt
from typing import Any, List, Dict, Optional
"""
🔧 FastAPI Dependency Injection System
======================================

Comprehensive dependency injection system for managing state and shared resources:
- Database connections and sessions
- Authentication and authorization
- Caching and Redis connections
- External service clients
- Configuration management
- Logging and monitoring
- Background task management
- Rate limiting and throttling
"""



logger = structlog.get_logger(__name__)

T = TypeVar('T')

class ResourceType(Enum):
    """Resource types for dependency injection"""
    DATABASE = "database"
    CACHE = "cache"
    AUTH = "auth"
    CONFIG = "config"
    LOGGER = "logger"
    SERVICE = "service"
    CLIENT = "client"
    BACKGROUND = "background"

@dataclass
class DependencyConfig:
    """Configuration for dependency injection"""
    # Database settings
    database_url: str = "postgresql+asyncpg://user:pass@localhost/db"
    database_pool_size: int = 20
    database_max_overflow: int = 30
    database_pool_pre_ping: bool = True
    
    # Cache settings
    redis_url: str = "redis://localhost:6379"
    redis_pool_size: int = 10
    cache_ttl: int = 3600
    
    # Auth settings
    secret_key: str = "your-secret-key"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Service settings
    max_concurrent_requests: int = 100
    request_timeout: float = 30.0
    
    # Monitoring settings
    enable_metrics: bool = True
    enable_tracing: bool = True

class DatabaseManager:
    """
    Database connection manager using dependency injection.
    """
    
    def __init__(self, config: DependencyConfig):
        
    """__init__ function."""
self.config = config
        self.engine = None
        self.session_factory = None
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> Any:
        """Initialize database connections."""
        async with self._lock:
            if self.engine is None:
                self.engine = create_async_engine(
                    self.config.database_url,
                    pool_size=self.config.database_pool_size,
                    max_overflow=self.config.database_max_overflow,
                    pool_pre_ping=self.config.database_pool_pre_ping,
                    echo=False
                )
                
                self.session_factory = async_sessionmaker(
                    self.engine,
                    class_=AsyncSession,
                    expire_on_commit=False
                )
                
                logger.info("Database manager initialized")
    
    async def cleanup(self) -> Any:
        """Cleanup database connections."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database manager cleaned up")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        """Get database session with automatic cleanup."""
        if not self.session_factory:
            await self.initialize()
        
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def get_session_dependency(self) -> AsyncSession:
        """Dependency function for FastAPI."""
        async with self.get_session() as session:
            yield session

class CacheManager:
    """
    Redis cache manager using dependency injection.
    """
    
    def __init__(self, config: DependencyConfig):
        
    """__init__ function."""
self.config = config
        self.redis_client = None
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> Any:
        """Initialize Redis connection."""
        async with self._lock:
            if self.redis_client is None:
                self.redis_client = redis.from_url(
                    self.config.redis_url,
                    max_connections=self.config.redis_pool_size,
                    decode_responses=True
                )
                
                # Test connection
                await self.redis_client.ping()
                logger.info("Cache manager initialized")
    
    async def cleanup(self) -> Any:
        """Cleanup Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Cache manager cleaned up")
    
    async def get_client(self) -> redis.Redis:
        """Get Redis client."""
        if not self.redis_client:
            await self.initialize()
        return self.redis_client
    
    async def get_client_dependency(self) -> redis.Redis:
        """Dependency function for FastAPI."""
        return await self.get_client()

class AuthManager:
    """
    Authentication manager using dependency injection.
    """
    
    def __init__(self, config: DependencyConfig):
        
    """__init__ function."""
self.config = config
        self.security = HTTPBearer()
        self._lock = asyncio.Lock()
    
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(
                token, 
                self.config.secret_key, 
                algorithms=[self.config.algorithm]
            )
            return payload
        except JWTError:
            return None
    
    async def get_current_user(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
        db: AsyncSession = Depends(lambda: None)  # Will be overridden
    ) -> Optional[Dict[str, Any]]:
        """Get current authenticated user."""
        token = credentials.credentials
        payload = await self.verify_token(token)
        
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return payload

class ConfigManager:
    """
    Configuration manager using dependency injection.
    """
    
    def __init__(self, config: DependencyConfig):
        
    """__init__ function."""
self.config = config
        self._cache = {}
        self._lock = asyncio.Lock()
    
    def get_config(self, key: str, default: Any = None) -> Optional[Dict[str, Any]]:
        """Get configuration value."""
        return getattr(self.config, key, default)
    
    def set_config(self, key: str, value: Any):
        """Set configuration value."""
        setattr(self.config, key, value)
    
    async def get_config_dependency(self) -> DependencyConfig:
        """Dependency function for FastAPI."""
        return self.config

class ServiceManager:
    """
    Service manager for external services using dependency injection.
    """
    
    def __init__(self, config: DependencyConfig):
        
    """__init__ function."""
self.config = config
        self.services = {}
        self._lock = asyncio.Lock()
    
    async def register_service(self, name: str, service: Any):
        """Register a service."""
        async with self._lock:
            self.services[name] = service
            logger.info(f"Service registered: {name}")
    
    async def get_service(self, name: str) -> Optional[Any]:
        """Get a service by name."""
        return self.services.get(name)
    
    async def get_service_dependency(self, service_name: str):
        """Dependency function for FastAPI."""
        def dependency():
            
    """dependency function."""
return self.services.get(service_name)
        return dependency

class BackgroundTaskManager:
    """
    Background task manager using dependency injection.
    """
    
    def __init__(self, config: DependencyConfig):
        
    """__init__ function."""
self.config = config
        self.tasks = {}
        self._lock = asyncio.Lock()
    
    async def add_task(self, task_id: str, task_func: Callable, *args, **kwargs):
        """Add a background task."""
        async with self._lock:
            task = asyncio.create_task(task_func(*args, **kwargs))
            self.tasks[task_id] = task
            logger.info(f"Background task added: {task_id}")
    
    async def get_task(self, task_id: str) -> Optional[asyncio.Task]:
        """Get a background task."""
        return self.tasks.get(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a background task."""
        task = self.tasks.get(task_id)
        if task:
            task.cancel()
            del self.tasks[task_id]
            logger.info(f"Background task cancelled: {task_id}")
            return True
        return False
    
    async def get_task_manager_dependency(self) -> Optional[Dict[str, Any]]:
        """Dependency function for FastAPI."""
        return self

class DependencyContainer:
    """
    Main dependency container for managing all dependencies.
    """
    
    def __init__(self, config: DependencyConfig):
        
    """__init__ function."""
self.config = config
        self.db_manager = DatabaseManager(config)
        self.cache_manager = CacheManager(config)
        self.auth_manager = AuthManager(config)
        self.config_manager = ConfigManager(config)
        self.service_manager = ServiceManager(config)
        self.background_manager = BackgroundTaskManager(config)
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> Any:
        """Initialize all dependency managers."""
        async with self._lock:
            if not self._initialized:
                await self.db_manager.initialize()
                await self.cache_manager.initialize()
                self._initialized = True
                logger.info("Dependency container initialized")
    
    async def cleanup(self) -> Any:
        """Cleanup all dependency managers."""
        async with self._lock:
            if self._initialized:
                await self.db_manager.cleanup()
                await self.cache_manager.cleanup()
                self._initialized = False
                logger.info("Dependency container cleaned up")
    
    def get_db_session_dependency(self) -> Optional[Dict[str, Any]]:
        """Get database session dependency."""
        return self.db_manager.get_session_dependency
    
    def get_cache_client_dependency(self) -> Optional[Dict[str, Any]]:
        """Get cache client dependency."""
        return self.cache_manager.get_client_dependency
    
    def get_auth_dependency(self) -> Optional[Dict[str, Any]]:
        """Get authentication dependency."""
        return self.auth_manager.get_current_user
    
    def get_config_dependency(self) -> Optional[Dict[str, Any]]:
        """Get configuration dependency."""
        return self.config_manager.get_config_dependency
    
    def get_service_dependency(self, service_name: str):
        """Get service dependency."""
        return self.service_manager.get_service_dependency(service_name)
    
    def get_background_manager_dependency(self) -> Optional[Dict[str, Any]]:
        """Get background task manager dependency."""
        return self.background_manager.get_task_manager_dependency

# Global dependency container
_dependency_container: Optional[DependencyContainer] = None

def get_dependency_container() -> DependencyContainer:
    """Get global dependency container."""
    global _dependency_container
    if _dependency_container is None:
        config = DependencyConfig()
        _dependency_container = DependencyContainer(config)
    return _dependency_container

# Dependency functions for FastAPI
async def get_db_session() -> AsyncSession:
    """Get database session dependency."""
    container = get_dependency_container()
    async with container.db_manager.get_session() as session:
        yield session

async def get_cache_client() -> redis.Redis:
    """Get cache client dependency."""
    container = get_dependency_container()
    return await container.cache_manager.get_client()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """Get current user dependency."""
    container = get_dependency_container()
    return await container.auth_manager.get_current_user(credentials, db)

async def get_config() -> DependencyConfig:
    """Get configuration dependency."""
    container = get_dependency_container()
    return await container.config_manager.get_config_dependency()

async def get_background_manager() -> BackgroundTaskManager:
    """Get background task manager dependency."""
    container = get_dependency_container()
    return await container.background_manager.get_task_manager_dependency()

# Service-specific dependencies
class UserService:
    """User service with dependency injection."""
    
    def __init__(
        self,
        db: AsyncSession = Depends(get_db_session),
        cache: redis.Redis = Depends(get_cache_client),
        config: DependencyConfig = Depends(get_config)
    ):
        self.db = db
        self.cache = cache
        self.config = config
    
    async def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID with caching."""
        # Try cache first
        cache_key = f"user:{user_id}"
        cached_user = await self.cache.get(cache_key)
        if cached_user:
            return cached_user
        
        # Get from database
        # Implementation here...
        user = {"id": user_id, "name": "John Doe"}
        
        # Cache result
        await self.cache.setex(cache_key, self.config.cache_ttl, str(user))
        
        return user

async def get_user_service(
    db: AsyncSession = Depends(get_db_session),
    cache: redis.Redis = Depends(get_cache_client),
    config: DependencyConfig = Depends(get_config)
) -> UserService:
    """Get user service dependency."""
    return UserService(db, cache, config)

# Rate limiting dependency
class RateLimiter:
    """Rate limiter using dependency injection."""
    
    def __init__(self, cache: redis.Redis = Depends(get_cache_client)):
        self.cache = cache
    
    async def check_rate_limit(self, key: str, limit: int, window: int) -> bool:
        """Check rate limit."""
        current = await self.cache.incr(key)
        if current == 1:
            await self.cache.expire(key, window)
        return current <= limit

async def get_rate_limiter(
    cache: redis.Redis = Depends(get_cache_client)
) -> RateLimiter:
    """Get rate limiter dependency."""
    return RateLimiter(cache)

# Logging dependency
class RequestLogger:
    """Request logger using dependency injection."""
    
    def __init__(self, request: Request):
        
    """__init__ function."""
self.request = request
        self.start_time = time.time()
    
    def log_request(self, response_time: float, status_code: int):
        """Log request details."""
        logger.info(
            "Request processed",
            method=self.request.method,
            url=str(self.request.url),
            response_time=response_time,
            status_code=status_code
        )

async async def get_request_logger(request: Request) -> RequestLogger:
    """Get request logger dependency."""
    return RequestLogger(request)

# Middleware for dependency injection
class DependencyInjectionMiddleware:
    """Middleware for managing dependency injection lifecycle."""
    
    def __init__(self, app: FastAPI):
        
    """__init__ function."""
self.app = app
        self.container = get_dependency_container()
    
    async def __call__(self, request: Request, call_next):
        """Process request with dependency injection."""
        # Initialize dependencies if needed
        await self.container.initialize()
        
        # Process request
        response = await call_next(request)
        
        return response

# FastAPI application setup with dependency injection
def create_app_with_dependencies() -> FastAPI:
    """Create FastAPI app with dependency injection setup."""
    app = FastAPI(
        title="Blatam Academy API",
        version="1.0.0",
        description="API with comprehensive dependency injection"
    )
    
    # Initialize dependency container
    container = get_dependency_container()
    
    # Add startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        
    """startup_event function."""
await container.initialize()
    
    @app.on_event("shutdown")
    async def shutdown_event():
        
    """shutdown_event function."""
await container.cleanup()
    
    # Add middleware
    app.add_middleware(DependencyInjectionMiddleware)
    
    return app

# Example usage in FastAPI routes
def create_example_routes(app: FastAPI):
    """Create example routes demonstrating dependency injection."""
    
    @app.get("/users/{user_id}")
    async def get_user(
        user_id: int,
        user_service: UserService = Depends(get_user_service),
        current_user: Dict[str, Any] = Depends(get_current_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter),
        logger: RequestLogger = Depends(get_request_logger)
    ):
        """Get user with dependency injection."""
        # Check rate limit
        rate_limit_key = f"rate_limit:{current_user['id']}"
        if not await rate_limiter.check_rate_limit(rate_limit_key, 100, 3600):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        # Get user
        user = await user_service.get_user(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Log request
        response_time = time.time() - logger.start_time
        logger.log_request(response_time, 200)
        
        return user
    
    @app.post("/background-task")
    async def create_background_task(
        task_data: Dict[str, Any],
        background_manager: BackgroundTaskManager = Depends(get_background_manager),
        current_user: Dict[str, Any] = Depends(get_current_user)
    ):
        """Create background task with dependency injection."""
        task_id = f"task_{current_user['id']}_{int(time.time())}"
        
        async def background_task():
            
    """background_task function."""
# Simulate background work
            await asyncio.sleep(5)
            logger.info(f"Background task completed: {task_id}")
        
        await background_manager.add_task(task_id, background_task)
        
        return {"task_id": task_id, "status": "created"}

# Example of how to use the dependency injection system
async def example_usage():
    """Example usage of the dependency injection system."""
    
    # Create app with dependencies
    app = create_app_with_dependencies()
    create_example_routes(app)
    
    # Initialize container
    container = get_dependency_container()
    await container.initialize()
    
    try:
        # Use dependencies
        async with container.db_manager.get_session() as db:
            # Database operations
            pass
        
        cache_client = await container.cache_manager.get_client()
        await cache_client.set("test_key", "test_value")
        
        # Register services
        await container.service_manager.register_service("email", "email_service")
        await container.service_manager.register_service("payment", "payment_service")
        
        logger.info("Dependency injection system working correctly")
        
    finally:
        await container.cleanup()

match __name__:
    case "__main__":
    asyncio.run(example_usage()) 