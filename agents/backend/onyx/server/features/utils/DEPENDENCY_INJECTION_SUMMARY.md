# 🔧 FastAPI Dependency Injection System - Complete Guide

## Overview

This comprehensive guide demonstrates how to properly use FastAPI's dependency injection system for managing state and shared resources in the Blatam Academy backend. Dependency injection ensures clean separation of concerns, testability, and efficient resource management.

## 🏗️ Architecture

### Core Components

1. **DependencyContainer** - Main container managing all dependencies
2. **DatabaseManager** - Database connection and session management
3. **CacheManager** - Redis cache connection management
4. **AuthManager** - Authentication and authorization
5. **ConfigManager** - Configuration management
6. **ServiceManager** - External service management
7. **BackgroundTaskManager** - Background task management

### Dependency Patterns

- **Singleton** - Single instance throughout application lifecycle
- **Factory** - New instance for each request
- **Scoped** - Different scopes (request, session, transient)
- **Cached** - Cached instances with TTL
- **Conditional** - Different dependencies based on conditions
- **Async** - Async dependencies with proper lifecycle

## 🎯 Key Features

### 1. Centralized Dependency Management
```python
# Single container manages all dependencies
container = DependencyContainer(config)
await container.initialize()

# Access dependencies through container
db_session = await container.db_manager.get_session()
cache_client = await container.cache_manager.get_client()
```

### 2. Automatic Resource Management
```python
# Dependencies are automatically initialized and cleaned up
@app.on_event("startup")
async def startup_event():
    await container.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    await container.cleanup()
```

### 3. Scoped Dependencies
```python
# Different scopes for different use cases
@singleton_dependency
async def get_database_connection() -> DatabaseConnection:
    """Singleton database connection."""
    pass

@cached_dependency(ttl=1800)
async def get_cache_service() -> CacheService:
    """Cached cache service."""
    pass

@scoped_dependency(DependencyScope.REQUEST)
async def get_user_service() -> UserService:
    """Request-scoped user service."""
    pass
```

### 4. Conditional Dependencies
```python
# Different services based on conditions
notification_service = ConditionalDependency()
notification_service.add_condition(
    lambda type: type == "email", 
    lambda: EmailService("email_api_key")
)
notification_service.add_condition(
    lambda type: type == "sms", 
    lambda: SMSService("sms_api_key")
)

async def get_notification_service(notification_type: str):
    return await notification_service.get_instance(notification_type)
```

## 📊 Dependency Injection Patterns

### 1. Singleton Pattern
```python
class SingletonDependency:
    """Ensures only one instance exists."""
    
    def __init__(self, factory_func: Callable):
        self.factory_func = factory_func
        self._instance: Optional[Any] = None
        self._lock = asyncio.Lock()
    
    async def get_instance(self, *args, **kwargs) -> Any:
        if self._instance is None:
            async with self._lock:
                if self._instance is None:
                    self._instance = await self.factory_func(*args, **kwargs)
        return self._instance

# Usage
@singleton_dependency
async def get_database_connection() -> DatabaseConnection:
    connection = DatabaseConnection("postgresql://localhost/db")
    await connection.connect()
    return connection
```

### 2. Factory Pattern
```python
class FactoryDependency:
    """Creates new instances for each request."""
    
    def __init__(self, factory_func: Callable):
        self.factory_func = factory_func
    
    async def create_instance(self, *args, **kwargs) -> Any:
        return await self.factory_func(*args, **kwargs)

# Usage
async def get_user_service() -> UserService:
    return UserService()  # New instance each time
```

### 3. Cached Pattern
```python
class CachedDependency:
    """Caches instances with TTL."""
    
    def __init__(self, ttl: int = 3600):
        self.ttl = ttl
        self._cache: Dict[str, tuple[Any, float]] = {}
    
    async def get_cached_instance(self, key: str, factory_func: Callable) -> Any:
        current_time = time.time()
        
        if key in self._cache:
            instance, timestamp = self._cache[key]
            if current_time - timestamp < self.ttl:
                return instance
        
        instance = await factory_func()
        self._cache[key] = (instance, current_time)
        return instance

# Usage
@cached_dependency(ttl=1800)  # 30 minutes
async def get_cache_service() -> CacheService:
    return CacheService("redis://localhost:6379")
```

### 4. Conditional Pattern
```python
class ConditionalDependency:
    """Provides different dependencies based on conditions."""
    
    def __init__(self):
        self.conditions: List[tuple[Callable, Callable]] = []
        self.default_factory: Optional[Callable] = None
    
    def add_condition(self, condition: Callable, factory: Callable):
        self.conditions.append((condition, factory))
    
    async def get_instance(self, *args, **kwargs) -> Any:
        for condition, factory in self.conditions:
            if await condition(*args, **kwargs):
                return await factory(*args, **kwargs)
        
        if self.default_factory:
            return await self.default_factory(*args, **kwargs)
        
        raise ValueError("No matching condition")

# Usage
notification_service = ConditionalDependency()
notification_service.add_condition(
    lambda type: type == "email", 
    lambda: EmailService("email_api_key")
)
notification_service.add_condition(
    lambda type: type == "sms", 
    lambda: SMSService("sms_api_key")
)
```

## 🔧 Implementation Examples

### 1. Database Dependencies
```python
class DatabaseManager:
    """Database connection manager."""
    
    def __init__(self, config: DependencyConfig):
        self.config = config
        self.engine = None
        self.session_factory = None
    
    async def initialize(self):
        """Initialize database connections."""
        self.engine = create_async_engine(
            self.config.database_url,
            pool_size=self.config.database_pool_size,
            max_overflow=self.config.database_max_overflow,
            pool_pre_ping=self.config.database_pool_pre_ping
        )
        
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        """Get database session with automatic cleanup."""
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

# Dependency function
async def get_db_session() -> AsyncSession:
    """Get database session dependency."""
    container = get_dependency_container()
    async with container.db_manager.get_session() as session:
        yield session
```

### 2. Cache Dependencies
```python
class CacheManager:
    """Redis cache manager."""
    
    def __init__(self, config: DependencyConfig):
        self.config = config
        self.redis_client = None
    
    async def initialize(self):
        """Initialize Redis connection."""
        self.redis_client = redis.from_url(
            self.config.redis_url,
            max_connections=self.config.redis_pool_size,
            decode_responses=True
        )
        await self.redis_client.ping()
    
    async def get_client(self) -> redis.Redis:
        """Get Redis client."""
        if not self.redis_client:
            await self.initialize()
        return self.redis_client

# Dependency function
async def get_cache_client() -> redis.Redis:
    """Get cache client dependency."""
    container = get_dependency_container()
    return await container.cache_manager.get_client()
```

### 3. Authentication Dependencies
```python
class AuthManager:
    """Authentication manager."""
    
    def __init__(self, config: DependencyConfig):
        self.config = config
        self.security = HTTPBearer()
    
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token."""
        try:
            from jose import JWTError, jwt
            payload = jwt.decode(
                token, 
                self.config.secret_key, 
                algorithms=[self.config.algorithm]
            )
            return payload
        except JWTError:
            return None

# Dependency function
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """Get current authenticated user."""
    container = get_dependency_container()
    return await container.auth_manager.get_current_user(credentials, db)
```

### 4. Service Dependencies
```python
class ServiceManager:
    """Service manager for external services."""
    
    def __init__(self, config: DependencyConfig):
        self.config = config
        self.services = {}
    
    async def register_service(self, name: str, service: Any):
        """Register a service."""
        self.services[name] = service
    
    async def get_service(self, name: str) -> Optional[Any]:
        """Get a service by name."""
        return self.services.get(name)

# Usage
async def get_email_service() -> EmailService:
    """Get email service dependency."""
    container = get_dependency_container()
    return await container.service_manager.get_service("email")
```

## 🚀 FastAPI Integration

### 1. Application Setup
```python
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
        await container.initialize()
    
    @app.on_event("shutdown")
    async def shutdown_event():
        await container.cleanup()
    
    return app
```

### 2. Route Dependencies
```python
@app.get("/users/{user_id}")
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db_session),
    cache: redis.Redis = Depends(get_cache_client),
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
    await cache.setex(cache_key, 1800, result)
    
    # Log request
    response_time = time.time() - logger.start_time
    logger.log_request(response_time, 200)
    
    return {"user": result, "source": "database"}
```

### 3. Background Task Dependencies
```python
@app.post("/background-tasks")
async def create_background_task(
    task_data: Dict[str, Any],
    background_manager: BackgroundTaskManager = Depends(get_background_manager),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Create background task with dependency injection."""
    task_id = f"task_{current_user['user_id']}_{int(time.time())}"
    
    async def background_task():
        await asyncio.sleep(5)
        logger.info(f"Background task completed: {task_id}")
    
    await background_manager.add_task(task_id, background_task)
    
    return {"task_id": task_id, "status": "created"}
```

## 📊 Dependency Scopes

### 1. Singleton Scope
- **Use Case**: Database connections, configuration, logging
- **Lifecycle**: Application lifetime
- **Memory**: Shared across all requests
- **Example**: Database connection pool

### 2. Request Scope
- **Use Case**: User-specific data, request context
- **Lifecycle**: Single request
- **Memory**: New instance per request
- **Example**: User service, request logger

### 3. Session Scope
- **Use Case**: User session data
- **Lifecycle**: User session
- **Memory**: Shared within session
- **Example**: User preferences, shopping cart

### 4. Transient Scope
- **Use Case**: Stateless services, utilities
- **Lifecycle**: Single use
- **Memory**: New instance each time
- **Example**: Email service, SMS service

## 🔧 Best Practices

### 1. Dependency Organization
```python
# Group related dependencies
class DatabaseDependencies:
    @staticmethod
    async def get_db_session() -> AsyncSession:
        pass
    
    @staticmethod
    async def get_db_connection() -> DatabaseConnection:
        pass

class AuthDependencies:
    @staticmethod
    async def get_current_user() -> Dict[str, Any]:
        pass
    
    @staticmethod
    async def get_auth_service() -> AuthService:
        pass
```

### 2. Error Handling
```python
async def get_db_session() -> AsyncSession:
    """Get database session with error handling."""
    try:
        container = get_dependency_container()
        async with container.db_manager.get_session() as session:
            yield session
    except Exception as e:
        logger.error(f"Database session error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable"
        )
```

### 3. Configuration Management
```python
@lru_cache()
def get_config() -> AppConfig:
    """Get application configuration (cached)."""
    return AppConfig(
        database_url=os.getenv("DATABASE_URL"),
        redis_url=os.getenv("REDIS_URL"),
        secret_key=os.getenv("SECRET_KEY"),
        debug=os.getenv("DEBUG", "False").lower() == "true"
    )
```

### 4. Testing Dependencies
```python
# Test fixtures for dependencies
@pytest.fixture
async def test_db_session():
    """Test database session."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    TestingSessionLocal = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with TestingSessionLocal() as session:
        yield session

@pytest.fixture
async def test_cache_client():
    """Test cache client."""
    return redis.from_url("redis://localhost:6379/1")  # Use separate DB

# Override dependencies in tests
def test_get_user(test_client, test_db_session, test_cache_client):
    """Test user endpoint with mocked dependencies."""
    app.dependency_overrides[get_db_session] = lambda: test_db_session
    app.dependency_overrides[get_cache_client] = lambda: test_cache_client
    
    response = test_client.get("/users/1")
    assert response.status_code == 200
```

## 📈 Performance Benefits

### 1. Resource Efficiency
- **Connection Pooling**: Reuse database connections
- **Caching**: Reduce redundant operations
- **Lazy Loading**: Load resources only when needed
- **Memory Management**: Proper cleanup of resources

### 2. Scalability
- **Horizontal Scaling**: Stateless dependencies
- **Load Balancing**: Shared resource management
- **Concurrent Processing**: Async dependency handling
- **Resource Limits**: Configurable connection pools

### 3. Maintainability
- **Separation of Concerns**: Clean dependency boundaries
- **Testability**: Easy to mock dependencies
- **Configuration**: Centralized configuration management
- **Error Handling**: Consistent error handling patterns

## 🔍 Monitoring and Debugging

### 1. Dependency Metrics
```python
class DependencyMetrics:
    """Track dependency usage and performance."""
    
    def __init__(self):
        self.usage_count = defaultdict(int)
        self.response_times = defaultdict(list)
        self.error_count = defaultdict(int)
    
    def record_usage(self, dependency_name: str, response_time: float, error: bool = False):
        """Record dependency usage."""
        self.usage_count[dependency_name] += 1
        self.response_times[dependency_name].append(response_time)
        if error:
            self.error_count[dependency_name] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get dependency metrics."""
        return {
            "usage_count": dict(self.usage_count),
            "average_response_times": {
                name: sum(times) / len(times) if times else 0
                for name, times in self.response_times.items()
            },
            "error_count": dict(self.error_count)
        }
```

### 2. Dependency Health Checks
```python
async def check_dependency_health() -> Dict[str, bool]:
    """Check health of all dependencies."""
    container = get_dependency_container()
    
    health_status = {}
    
    # Check database
    try:
        async with container.db_manager.get_session() as session:
            await session.execute("SELECT 1")
        health_status["database"] = True
    except Exception:
        health_status["database"] = False
    
    # Check cache
    try:
        cache_client = await container.cache_manager.get_client()
        await cache_client.ping()
        health_status["cache"] = True
    except Exception:
        health_status["cache"] = False
    
    return health_status
```

## 🎯 Summary

This comprehensive dependency injection system provides:

1. **Centralized Management** - Single container for all dependencies
2. **Resource Efficiency** - Connection pooling and caching
3. **Scalability** - Async patterns and resource limits
4. **Maintainability** - Clean separation and testability
5. **Performance** - Optimized resource usage
6. **Monitoring** - Comprehensive metrics and health checks

The system follows FastAPI best practices and ensures efficient resource management for the Blatam Academy backend. 