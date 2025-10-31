from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import logging
from typing import Any, Optional, Dict, List, Callable, Awaitable, Union, Type, TypeVar, Generic
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from enum import Enum
from fastapi import FastAPI, Depends, HTTPException, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
import structlog
from typing import Any, List, Dict, Optional
"""
🎯 FastAPI Dependency Injection Patterns
========================================

Comprehensive patterns and best practices for FastAPI dependency injection:
- Singleton patterns
- Factory patterns
- Scoped dependencies
- Conditional dependencies
- Cached dependencies
- Async dependencies
- Testing dependencies
- Configuration patterns
"""



logger = structlog.get_logger(__name__)

T = TypeVar('T')

class DependencyScope(Enum):
    """Dependency scopes"""
    SINGLETON = "singleton"
    REQUEST = "request"
    SESSION = "session"
    TRANSIENT = "transient"

class DependencyLifecycle(Enum):
    """Dependency lifecycle stages"""
    INITIALIZATION = "initialization"
    REQUEST_START = "request_start"
    REQUEST_PROCESSING = "request_processing"
    REQUEST_END = "request_end"
    CLEANUP = "cleanup"

@dataclass
class DependencyMetadata:
    """Metadata for dependency injection"""
    name: str
    scope: DependencyScope
    lifecycle: DependencyLifecycle
    dependencies: List[str] = field(default_factory=list)
    cache_ttl: Optional[int] = None
    retry_count: int = 0
    timeout: Optional[float] = None

class SingletonDependency:
    """
    Singleton pattern for dependency injection.
    Ensures only one instance exists throughout the application lifecycle.
    """
    
    def __init__(self, factory_func: Callable):
        
    """__init__ function."""
self.factory_func = factory_func
        self._instance: Optional[Any] = None
        self._lock = asyncio.Lock()
    
    async def get_instance(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """Get singleton instance."""
        if self._instance is None:
            async with self._lock:
                if self._instance is None:
                    if asyncio.iscoroutinefunction(self.factory_func):
                        self._instance = await self.factory_func(*args, **kwargs)
                    else:
                        self._instance = self.factory_func(*args, **kwargs)
        return self._instance
    
    def reset(self) -> Any:
        """Reset singleton instance (useful for testing)."""
        self._instance = None

class FactoryDependency:
    """
    Factory pattern for dependency injection.
    Creates new instances for each request.
    """
    
    def __init__(self, factory_func: Callable):
        
    """__init__ function."""
self.factory_func = factory_func
    
    async def create_instance(self, *args, **kwargs) -> Any:
        """Create new instance."""
        if asyncio.iscoroutinefunction(self.factory_func):
            return await self.factory_func(*args, **kwargs)
        else:
            return self.factory_func(*args, **kwargs)

class ScopedDependency:
    """
    Scoped dependency pattern.
    Manages dependencies with different scopes (request, session, etc.).
    """
    
    def __init__(self, scope: DependencyScope):
        
    """__init__ function."""
self.scope = scope
        self._instances: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
    
    async def get_instance(self, key: str, factory_func: Callable, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """Get scoped instance."""
        if self.scope == DependencyScope.SINGLETON:
            if key not in self._instances:
                async with self._lock:
                    if key not in self._instances:
                        if asyncio.iscoroutinefunction(factory_func):
                            self._instances[key] = await factory_func(*args, **kwargs)
                        else:
                            self._instances[key] = factory_func(*args, **kwargs)
            return self._instances[key]
        
        elif self.scope == DependencyScope.REQUEST:
            # For request scope, create new instance each time
            if asyncio.iscoroutinefunction(factory_func):
                return await factory_func(*args, **kwargs)
            else:
                return factory_func(*args, **kwargs)
        
        elif self.scope == DependencyScope.TRANSIENT:
            # For transient scope, always create new instance
            if asyncio.iscoroutinefunction(factory_func):
                return await factory_func(*args, **kwargs)
            else:
                return factory_func(*args, **kwargs)

class CachedDependency:
    """
    Cached dependency pattern.
    Caches dependency instances with TTL.
    """
    
    def __init__(self, ttl: int = 3600):
        
    """__init__ function."""
self.ttl = ttl
        self._cache: Dict[str, tuple[Any, float]] = {}
        self._lock = asyncio.Lock()
    
    async def get_cached_instance(
        self, 
        key: str, 
        factory_func: Callable, 
        *args, 
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Get cached instance."""
        current_time = time.time()
        
        # Check cache
        if key in self._cache:
            instance, timestamp = self._cache[key]
            if current_time - timestamp < self.ttl:
                return instance
        
        # Create new instance
        async with self._lock:
            if key in self._cache:
                instance, timestamp = self._cache[key]
                if current_time - timestamp < self.ttl:
                    return instance
            
            if asyncio.iscoroutinefunction(factory_func):
                instance = await factory_func(*args, **kwargs)
            else:
                instance = factory_func(*args, **kwargs)
            
            self._cache[key] = (instance, current_time)
            return instance
    
    def clear_cache(self) -> Any:
        """Clear all cached instances."""
        self._cache.clear()

class ConditionalDependency:
    """
    Conditional dependency pattern.
    Provides different dependencies based on conditions.
    """
    
    def __init__(self) -> Any:
        self.conditions: List[tuple[Callable, Callable]] = []
        self.default_factory: Optional[Callable] = None
    
    def add_condition(self, condition: Callable, factory: Callable):
        """Add a condition and its corresponding factory."""
        self.conditions.append((condition, factory))
    
    def set_default(self, factory: Callable):
        """Set default factory when no conditions match."""
        self.default_factory = factory
    
    async def get_instance(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """Get instance based on conditions."""
        # Check conditions in order
        for condition, factory in self.conditions:
            if asyncio.iscoroutinefunction(condition):
                should_use = await condition(*args, **kwargs)
            else:
                should_use = condition(*args, **kwargs)
            
            if should_use:
                if asyncio.iscoroutinefunction(factory):
                    return await factory(*args, **kwargs)
                else:
                    return factory(*args, **kwargs)
        
        # Use default factory
        if self.default_factory:
            if asyncio.iscoroutinefunction(self.default_factory):
                return await self.default_factory(*args, **kwargs)
            else:
                return self.default_factory(*args, **kwargs)
        
        raise ValueError("No matching condition and no default factory")

class AsyncDependency:
    """
    Async dependency pattern.
    Manages async dependencies with proper lifecycle.
    """
    
    def __init__(self, factory_func: Callable):
        
    """__init__ function."""
self.factory_func = factory_func
        self._instance: Optional[Any] = None
        self._lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self) -> Any:
        """Initialize async dependency."""
        if not self._initialized:
            async with self._lock:
                if not self._initialized:
                    self._instance = await self.factory_func()
                    self._initialized = True
    
    async def get_instance(self) -> Optional[Dict[str, Any]]:
        """Get async instance."""
        await self.initialize()
        return self._instance
    
    async def cleanup(self) -> Any:
        """Cleanup async dependency."""
        if self._instance and hasattr(self._instance, 'close'):
            await self._instance.close()
        self._initialized = False
        self._instance = None

# Dependency decorators
def singleton_dependency(factory_func: Callable):
    """Decorator for singleton dependencies."""
    singleton = SingletonDependency(factory_func)
    
    @wraps(factory_func)
    async def wrapper(*args, **kwargs) -> Any:
        return await singleton.get_instance(*args, **kwargs)
    
    return wrapper

def cached_dependency(ttl: int = 3600):
    """Decorator for cached dependencies."""
    def decorator(factory_func: Callable):
        
    """decorator function."""
cached = CachedDependency(ttl)
        
        @wraps(factory_func)
        async def wrapper(*args, **kwargs) -> Any:
            key = f"{factory_func.__name__}:{hash(str(args) + str(kwargs))}"
            return await cached.get_cached_instance(key, factory_func, *args, **kwargs)
        
        return wrapper
    return decorator

def scoped_dependency(scope: DependencyScope):
    """Decorator for scoped dependencies."""
    def decorator(factory_func: Callable):
        
    """decorator function."""
scoped = ScopedDependency(scope)
        
        @wraps(factory_func)
        async def wrapper(*args, **kwargs) -> Any:
            key = f"{factory_func.__name__}:{hash(str(args) + str(kwargs))}"
            return await scoped.get_instance(key, factory_func, *args, **kwargs)
        
        return wrapper
    return decorator

def async_dependency(factory_func: Callable):
    """Decorator for async dependencies."""
    return AsyncDependency(factory_func)

# Practical dependency patterns

class DatabaseConnection:
    """Database connection with dependency injection."""
    
    def __init__(self, connection_string: str):
        
    """__init__ function."""
self.connection_string = connection_string
        self._connection = None
    
    async def connect(self) -> Any:
        """Connect to database."""
        # Simulate database connection
        await asyncio.sleep(0.1)
        self._connection = {"status": "connected", "url": self.connection_string}
    
    async def close(self) -> Any:
        """Close database connection."""
        self._connection = None
    
    async def execute(self, query: str):
        """Execute database query."""
        if not self._connection:
            await self.connect()
        return f"Executed: {query}"

@singleton_dependency
async def get_database_connection() -> DatabaseConnection:
    """Get singleton database connection."""
    connection = DatabaseConnection("postgresql://localhost/db")
    await connection.connect()
    return connection

class CacheService:
    """Cache service with dependency injection."""
    
    def __init__(self, redis_url: str):
        
    """__init__ function."""
self.redis_url = redis_url
        self._cache = {}
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        return self._cache.get(key)
    
    async def set(self, key: str, value: str, ttl: int = 3600):
        """Set value in cache."""
        self._cache[key] = value

@cached_dependency(ttl=1800)  # 30 minutes
async def get_cache_service() -> CacheService:
    """Get cached cache service."""
    return CacheService("redis://localhost:6379")

class EmailService:
    """Email service with dependency injection."""
    
    def __init__(self, api_key: str):
        
    """__init__ function."""
self.api_key = api_key
    
    async def send_email(self, to: str, subject: str, body: str):
        """Send email."""
        return f"Email sent to {to}: {subject}"

class SMSService:
    """SMS service with dependency injection."""
    
    def __init__(self, api_key: str):
        
    """__init__ function."""
self.api_key = api_key
    
    async def send_sms(self, to: str, message: str):
        """Send SMS."""
        return f"SMS sent to {to}: {message}"

# Conditional dependency example
notification_service = ConditionalDependency()

def should_use_email(notification_type: str) -> bool:
    """Condition for using email service."""
    return notification_type == "email"

def should_use_sms(notification_type: str) -> bool:
    """Condition for using SMS service."""
    return notification_type == "sms"

notification_service.add_condition(should_use_email, lambda: EmailService("email_api_key"))
notification_service.add_condition(should_use_sms, lambda: SMSService("sms_api_key"))

async def get_notification_service(notification_type: str):
    """Get notification service based on type."""
    return await notification_service.get_instance(notification_type)

# Configuration dependency
@dataclass
class AppConfig:
    """Application configuration."""
    database_url: str
    redis_url: str
    email_api_key: str
    sms_api_key: str
    debug: bool = False

@lru_cache()
def get_config() -> AppConfig:
    """Get application configuration (cached)."""
    return AppConfig(
        database_url="postgresql://localhost/db",
        redis_url="redis://localhost:6379",
        email_api_key="email_key",
        sms_api_key="sms_key",
        debug=True
    )

# Authentication dependency
class AuthService:
    """Authentication service with dependency injection."""
    
    def __init__(self, config: AppConfig):
        
    """__init__ function."""
self.config = config
        self.security = HTTPBearer()
    
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify authentication token."""
        # Simulate token verification
        if token == "valid_token":
            return {"user_id": 123, "email": "user@example.com"}
        return None

async def get_auth_service(config: AppConfig = Depends(get_config)) -> AuthService:
    """Get authentication service dependency."""
    return AuthService(config)

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    auth_service: AuthService = Depends(get_auth_service)
) -> Dict[str, Any]:
    """Get current authenticated user."""
    token = credentials.credentials
    user = await auth_service.verify_token(token)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    return user

# Rate limiting dependency
class RateLimiter:
    """Rate limiter with dependency injection."""
    
    def __init__(self, cache_service: CacheService):
        
    """__init__ function."""
self.cache_service = cache_service
    
    async def check_rate_limit(self, key: str, limit: int, window: int) -> bool:
        """Check rate limit."""
        current = await self.cache_service.get(key)
        if current is None:
            await self.cache_service.set(key, "1", window)
            return True
        
        count = int(current) + 1
        if count > limit:
            return False
        
        await self.cache_service.set(key, str(count), window)
        return True

async def get_rate_limiter(
    cache_service: CacheService = Depends(get_cache_service)
) -> RateLimiter:
    """Get rate limiter dependency."""
    return RateLimiter(cache_service)

# Logging dependency
class RequestLogger:
    """Request logger with dependency injection."""
    
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

# Background task dependency
class BackgroundTaskManager:
    """Background task manager with dependency injection."""
    
    def __init__(self) -> Any:
        self.tasks: Dict[str, asyncio.Task] = {}
    
    async def add_task(self, task_id: str, task_func: Callable, *args, **kwargs):
        """Add background task."""
        task = asyncio.create_task(task_func(*args, **kwargs))
        self.tasks[task_id] = task
        logger.info(f"Background task added: {task_id}")
    
    async def get_task_status(self, task_id: str) -> str:
        """Get task status."""
        task = self.tasks.get(task_id)
        if not task:
            return "not_found"
        
        if task.done():
            if task.exception():
                return "failed"
            return "completed"
        
        return "running"

async def get_background_manager() -> BackgroundTaskManager:
    """Get background task manager dependency."""
    return BackgroundTaskManager()

# Example FastAPI routes using dependency injection patterns
def create_dependency_example_routes(app: FastAPI):
    """Create example routes demonstrating dependency injection patterns."""
    
    @app.get("/users/{user_id}")
    async def get_user(
        user_id: int,
        db: DatabaseConnection = Depends(get_database_connection),
        cache: CacheService = Depends(get_cache_service),
        current_user: Dict[str, Any] = Depends(get_current_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter),
        logger: RequestLogger = Depends(get_request_logger)
    ):
        """Get user with comprehensive dependency injection."""
        # Check rate limit
        rate_limit_key = f"rate_limit:{current_user['user_id']}"
        if not await rate_limiter.check_rate_limit(rate_limit_key, 100, 3600):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        # Try cache first
        cache_key = f"user:{user_id}"
        cached_user = await cache.get(cache_key)
        if cached_user:
            return {"user": cached_user, "source": "cache"}
        
        # Get from database
        result = await db.execute(f"SELECT * FROM users WHERE id = {user_id}")
        
        # Cache result
        await cache.set(cache_key, result, 1800)
        
        # Log request
        response_time = time.time() - logger.start_time
        logger.log_request(response_time, 200)
        
        return {"user": result, "source": "database"}
    
    @app.post("/notifications")
    async def send_notification(
        notification_type: str,
        recipient: str,
        message: str,
        notification_service = Depends(get_notification_service),
        current_user: Dict[str, Any] = Depends(get_current_user)
    ):
        """Send notification using conditional dependency."""
        service = await notification_service(notification_type)
        
        if isinstance(service, EmailService):
            result = await service.send_email(recipient, "Notification", message)
        elif isinstance(service, SMSService):
            result = await service.send_sms(recipient, message)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid notification type"
            )
        
        return {"result": result, "type": notification_type}
    
    @app.post("/background-tasks")
    async def create_background_task(
        task_data: Dict[str, Any],
        background_manager: BackgroundTaskManager = Depends(get_background_manager),
        current_user: Dict[str, Any] = Depends(get_current_user)
    ):
        """Create background task with dependency injection."""
        task_id = f"task_{current_user['user_id']}_{int(time.time())}"
        
        async def background_task():
            
    """background_task function."""
await asyncio.sleep(5)
            logger.info(f"Background task completed: {task_id}")
        
        await background_manager.add_task(task_id, background_task)
        
        return {"task_id": task_id, "status": "created"}
    
    @app.get("/background-tasks/{task_id}")
    async def get_task_status(
        task_id: str,
        background_manager: BackgroundTaskManager = Depends(get_background_manager)
    ):
        """Get background task status."""
        status = await background_manager.get_task_status(task_id)
        return {"task_id": task_id, "status": status}

# Example usage
async def example_dependency_patterns():
    """Example usage of dependency injection patterns."""
    
    # Singleton dependency
    db1 = await get_database_connection()
    db2 = await get_database_connection()
    assert db1 is db2  # Same instance
    
    # Cached dependency
    cache1 = await get_cache_service()
    cache2 = await get_cache_service()
    # May be same instance due to caching
    
    # Conditional dependency
    email_service = await get_notification_service("email")
    sms_service = await get_notification_service("sms")
    
    # Configuration dependency
    config = get_config()
    assert config.database_url == "postgresql://localhost/db"
    
    logger.info("Dependency injection patterns working correctly")

match __name__:
    case "__main__":
    asyncio.run(example_dependency_patterns()) 