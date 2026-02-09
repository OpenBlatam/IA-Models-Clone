# 🔧 FastAPI Dependency Injection System - Complete Guide

## Overview

This comprehensive guide demonstrates how to properly use FastAPI's dependency injection system for managing state and shared resources in the Video-OpusClip AI video processing system. Dependency injection ensures clean separation of concerns, testability, and efficient resource management.

## 🏗️ Architecture

### Core Components

1. **DependencyContainer** - Main container managing all dependencies
2. **DatabaseManager** - Database connection and session management
3. **CacheManager** - Redis cache connection management
4. **ModelManager** - AI model loading and caching
5. **ServiceManager** - External service and HTTP client management
6. **PerformanceOptimizer** - Performance optimization utilities
7. **TrainingLogger** - Training progress logging
8. **ErrorHandler** - Error handling and recovery
9. **MixedPrecisionManager** - Mixed precision training
10. **GradientAccumulator** - Gradient accumulation
11. **MultiGPUTrainer** - Multi-GPU training
12. **CodeProfiler** - Code profiling and optimization

### Dependency Patterns

- **Singleton** - Single instance throughout application lifecycle
- **Request** - New instance for each request
- **Session** - Different scopes (request, session, transient)
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
model = await container.model_manager.get_model("video_processor")
```

### 2. Automatic Resource Management
```python
# Dependencies are automatically initialized and cleaned up
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Initialize dependency container
    config = get_app_config()
    container = DependencyContainer(config)
    set_dependency_container(container)
    
    try:
        await container.initialize()
        yield
    finally:
        await container.cleanup()

app = FastAPI(lifespan=lifespan)
```

### 3. Scoped Dependencies
```python
# Different scopes for different use cases
@singleton_dependency
async def get_database_manager() -> DatabaseManager:
    """Singleton database manager."""
    return DatabaseManager(config)

@cached_dependency(ttl=300)
async def get_model(model_name: str) -> nn.Module:
    """Cached model loading."""
    return await load_model(model_name)
```

## 🔧 Implementation Guide

### 1. Configuration Management

```python
class AppConfig(BaseModel):
    """Application configuration for dependency injection."""
    
    # Database settings
    database_url: str = Field(..., env="DATABASE_URL")
    database_pool_size: int = Field(20, env="DATABASE_POOL_SIZE")
    
    # Cache settings
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    cache_enabled: bool = Field(True, env="CACHE_ENABLED")
    
    # Model settings
    model_cache_dir: str = Field("./models", env="MODEL_CACHE_DIR")
    device: str = Field("auto", env="DEVICE")
    mixed_precision: bool = Field(True, env="MIXED_PRECISION")
    
    # Performance settings
    max_workers: int = Field(4, env="MAX_WORKERS")
    request_timeout: float = Field(60.0, env="REQUEST_TIMEOUT")
    
    # Monitoring settings
    enable_metrics: bool = Field(True, env="ENABLE_METRICS")
    enable_health_checks: bool = Field(True, env="ENABLE_HEALTH_CHECKS")

@lru_cache()
def get_app_config() -> AppConfig:
    """Get application configuration (cached)."""
    return AppConfig()
```

### 2. Resource Managers

#### Database Manager
```python
class DatabaseManager:
    """Database connection manager with connection pooling."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.engine = None
        self.session_factory = None
    
    async def initialize(self):
        """Initialize database connection pool."""
        self.engine = create_async_engine(
            self.config.database_url,
            poolclass=QueuePool,
            pool_size=self.config.database_pool_size,
            max_overflow=self.config.database_max_overflow,
            pool_pre_ping=True
        )
        
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def get_session(self) -> AsyncSession:
        """Get database session from pool."""
        return self.session_factory()
    
    async def health_check(self) -> bool:
        """Perform health check on database."""
        try:
            async with self.engine.begin() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception:
            return False
    
    async def cleanup(self):
        """Cleanup database connections."""
        if self.engine:
            await self.engine.dispose()
```

#### Cache Manager
```python
class CacheManager:
    """Redis cache manager with connection pooling."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = None
    
    async def initialize(self):
        """Initialize Redis connection."""
        if self.config.cache_enabled:
            self.client = redis.from_url(
                self.config.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20
            )
            await self.client.ping()
    
    async def get_client(self) -> Optional[redis.Redis]:
        """Get Redis client."""
        return self.client if self.config.cache_enabled else None
    
    async def health_check(self) -> bool:
        """Perform health check on cache."""
        if not self.client or not self.config.cache_enabled:
            return True
        
        try:
            await self.client.ping()
            return True
        except Exception:
            return False
    
    async def cleanup(self):
        """Cleanup cache connections."""
        if self.client:
            await self.client.close()
```

#### Model Manager
```python
class ModelManager:
    """AI model manager with caching and optimization."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.models: Dict[str, nn.Module] = {}
        self._device = self._get_device()
    
    def _get_device(self) -> torch.device:
        """Get the appropriate device for model execution."""
        if self.config.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif self.config.device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    async def load_model(self, model_name: str, config: ModelConfig) -> nn.Module:
        """Load and cache a model."""
        if model_name in self.models:
            return self.models[model_name]
        
        # Load model (your actual model loading logic)
        model = await self._load_model_async(model_name, config)
        
        # Move to device
        model = model.to(self._device)
        
        # Enable mixed precision if configured
        if self.config.mixed_precision and self._device.type == "cuda":
            model = model.half()
        
        # Cache model
        self.models[model_name] = model
        return model
    
    async def get_model(self, model_name: str) -> Optional[nn.Module]:
        """Get a cached model."""
        return self.models.get(model_name)
    
    async def health_check(self) -> bool:
        """Perform health check on models."""
        try:
            for name, model in self.models.items():
                if not hasattr(model, 'parameters'):
                    return False
            return True
        except Exception:
            return False
    
    async def cleanup(self):
        """Cleanup model resources."""
        for name, model in self.models.items():
            del model
        self.models.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### 3. Dependency Container

```python
class DependencyContainer:
    """Central dependency injection container for Video-OpusClip."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.db_manager = DatabaseManager(config)
        self.cache_manager = CacheManager(config)
        self.model_manager = ModelManager(config)
        self.service_manager = ServiceManager(config)
        
        # Video-OpusClip specific managers
        self.performance_optimizer = PerformanceOptimizer()
        self.training_logger = TrainingLogger()
        self.error_handler = ErrorHandler()
        self.mixed_precision_manager = MixedPrecisionManager()
        self.gradient_accumulator = GradientAccumulator()
        self.multi_gpu_trainer = MultiGPUTrainer()
        self.code_profiler = CodeProfiler()
        
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize all dependency managers."""
        async with self._lock:
            if self._initialized:
                return
            
            # Initialize core managers
            await self.db_manager.initialize()
            await self.cache_manager.initialize()
            await self.service_manager.initialize()
            
            # Initialize Video-OpusClip managers
            await self.performance_optimizer.initialize()
            await self.training_logger.initialize()
            await self.error_handler.initialize()
            await self.mixed_precision_manager.initialize()
            await self.gradient_accumulator.initialize()
            await self.multi_gpu_trainer.initialize()
            await self.code_profiler.initialize()
            
            self._initialized = True
    
    async def cleanup(self):
        """Cleanup all dependency managers."""
        async with self._lock:
            if not self._initialized:
                return
            
            # Cleanup managers
            await self.db_manager.cleanup()
            await self.cache_manager.cleanup()
            await self.service_manager.cleanup()
            
            await self.performance_optimizer.cleanup()
            await self.training_logger.cleanup()
            await self.error_handler.cleanup()
            await self.mixed_precision_manager.cleanup()
            await self.gradient_accumulator.cleanup()
            await self.multi_gpu_trainer.cleanup()
            await self.code_profiler.cleanup()
            
            self._initialized = False
```

### 4. FastAPI Integration

#### Application Factory
```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan with dependency injection."""
    # Initialize dependency container
    config = get_app_config()
    container = DependencyContainer(config)
    set_dependency_container(container)
    
    try:
        await container.initialize()
        logger.info("Application started successfully")
        yield
    finally:
        await container.cleanup()
        logger.info("Application shutdown complete")

def create_app() -> FastAPI:
    """Create FastAPI application with dependency injection."""
    
    app = FastAPI(
        title="Video-OpusClip API",
        description="AI-powered video processing system with dependency injection",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add middleware for request tracking
    @app.middleware("http")
    async def add_request_context(request: Request, call_next):
        start_time = time.time()
        request.state.request_id = f"req_{int(start_time * 1000)}"
        request.state.start_time = start_time
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        logger.info(
            "Request processed",
            request_id=request.state.request_id,
            method=request.method,
            path=request.url.path,
            duration=duration,
            status_code=response.status_code
        )
        
        return response
    
    return app
```

#### Dependency Injection Functions
```python
def get_dependency_container() -> DependencyContainer:
    """Get the global dependency container."""
    if _container is None:
        raise RuntimeError("Dependency container not initialized")
    return _container

# Database session dependency
def get_db_session_dependency():
    async def dependency() -> AsyncSession:
        container = get_dependency_container()
        async with container.db_manager.get_session() as session:
            yield session
    return dependency

# Cache client dependency
def get_cache_client_dependency():
    async def dependency() -> Optional[redis.Redis]:
        container = get_dependency_container()
        return await container.cache_manager.get_client()
    return dependency

# Model dependency
def get_model_dependency(model_name: str):
    async def dependency() -> nn.Module:
        container = get_dependency_container()
        model = await container.model_manager.get_model(model_name)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model {model_name} not available"
            )
        return model
    return dependency

# Performance optimizer dependency
def get_performance_optimizer_dependency():
    async def dependency() -> PerformanceOptimizer:
        container = get_dependency_container()
        return container.performance_optimizer
    return dependency

# Training logger dependency
def get_training_logger_dependency():
    async def dependency() -> TrainingLogger:
        container = get_dependency_container()
        return container.training_logger
    return dependency

# Error handler dependency
def get_error_handler_dependency():
    async def dependency() -> ErrorHandler:
        container = get_dependency_container()
        return container.error_handler
    return dependency
```

### 5. Route Implementation

#### Basic Route with Dependencies
```python
@app.post("/video/process")
async def process_video(
    video_data: Dict[str, Any],
    model: nn.Module = Depends(get_model_dependency("video_processor")),
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer_dependency()),
    error_handler: ErrorHandler = Depends(get_error_handler_dependency()),
    db_session: AsyncSession = Depends(get_db_session_dependency())
):
    """Process video using injected dependencies."""
    
    try:
        # Use injected dependencies
        with optimizer.optimize_context():
            result = await model(video_data)
        
        # Log to database
        await log_processing_result(db_session, video_data, result)
        
        return {"success": True, "result": result}
        
    except Exception as e:
        await error_handler.handle_error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Video processing failed"
        )
```

#### Advanced Route with Multiple Dependencies
```python
@app.post("/video/train")
async def train_video_model(
    training_data: Dict[str, Any],
    model: nn.Module = Depends(get_model_dependency("video_processor")),
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer_dependency()),
    training_logger: TrainingLogger = Depends(get_training_logger_dependency()),
    mixed_precision: MixedPrecisionManager = Depends(get_mixed_precision_manager_dependency()),
    gradient_accumulator: GradientAccumulator = Depends(get_gradient_accumulator_dependency()),
    multi_gpu: MultiGPUTrainer = Depends(get_multi_gpu_trainer_dependency()),
    error_handler: ErrorHandler = Depends(get_error_handler_dependency())
):
    """Train video model using multiple injected dependencies."""
    
    try:
        # Start training session
        await training_logger.start_training_session()
        
        # Configure mixed precision
        with mixed_precision.autocast():
            # Configure gradient accumulation
            with gradient_accumulator.accumulate_gradients():
                # Use multi-GPU training
                with multi_gpu.distributed_context():
                    # Optimize performance
                    with optimizer.optimize_context():
                        result = await model.train(training_data)
        
        # Log training completion
        await training_logger.log_training_completion(result)
        
        return {"success": True, "training_result": result}
        
    except Exception as e:
        await error_handler.handle_error(e)
        await training_logger.log_training_error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Training failed"
        )
```

### 6. Dependency Decorators

#### Singleton Dependency
```python
def singleton_dependency(func: Callable) -> Callable:
    """Decorator for singleton dependencies."""
    instance = None
    lock = asyncio.Lock()
    
    async def wrapper(*args, **kwargs):
        nonlocal instance
        if instance is None:
            async with lock:
                if instance is None:
                    instance = await func(*args, **kwargs)
        return instance
    
    return wrapper

@singleton_dependency
async def get_database_manager() -> DatabaseManager:
    """Singleton database manager."""
    config = get_app_config()
    manager = DatabaseManager(config)
    await manager.initialize()
    return manager
```

#### Cached Dependency
```python
def cached_dependency(ttl: int = 300):
    """Decorator for cached dependencies."""
    cache = {}
    
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl:
                    return result
            
            result = await func(*args, **kwargs)
            cache[cache_key] = (result, time.time())
            return result
        
        return wrapper
    return decorator

@cached_dependency(ttl=600)
async def get_model_config(model_name: str) -> ModelConfig:
    """Cached model configuration loading."""
    return await load_model_config(model_name)
```

#### Dependency Injection Decorator
```python
def inject_dependencies(*dependencies: Callable):
    """Decorator to inject dependencies into route functions."""
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            # Get container
            container = get_dependency_container()
            
            # Inject dependencies
            for dep_func in dependencies:
                dep_name = dep_func.__name__
                if dep_name not in kwargs:
                    kwargs[dep_name] = await dep_func()
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

@app.post("/video/process")
@inject_dependencies(
    get_model_dependency("video_processor"),
    get_performance_optimizer_dependency(),
    get_error_handler_dependency()
)
async def process_video(
    video_data: Dict[str, Any],
    model: nn.Module,
    optimizer: PerformanceOptimizer,
    error_handler: ErrorHandler
):
    """Process video with injected dependencies."""
    # Function implementation
    pass
```

## 🔍 Health Monitoring

### Health Check Endpoints
```python
def add_health_endpoints(app: FastAPI):
    """Add health check endpoints to FastAPI app."""
    
    @app.get("/health")
    async def health_check():
        """Basic health check."""
        container = get_dependency_container()
        
        return {
            "status": "healthy" if container._initialized else "unhealthy",
            "uptime": (datetime.now() - container._startup_time).total_seconds() if container._startup_time else 0,
            "timestamp": datetime.now().isoformat()
        }
    
    @app.get("/health/detailed")
    async def detailed_health_check():
        """Detailed health check for all dependencies."""
        container = get_dependency_container()
        
        if not container._initialized:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Dependency container not initialized"
            )
        
        # Check all dependencies
        health_status = {
            "database": await container.db_manager.health_check(),
            "cache": await container.cache_manager.health_check(),
            "models": await container.model_manager.health_check(),
            "services": await container.service_manager.health_check(),
            "overall": True
        }
        
        # Overall health
        health_status["overall"] = all(health_status.values())
        
        status_code = status.HTTP_200_OK if health_status["overall"] else status.HTTP_503_SERVICE_UNAVAILABLE
        
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "healthy" if health_status["overall"] else "unhealthy",
                "dependencies": health_status,
                "uptime": (datetime.now() - container._startup_time).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
        )
```

### Background Health Monitoring
```python
async def _health_check_loop(self):
    """Background health check loop."""
    while self._initialized:
        try:
            await asyncio.sleep(self.config.health_check_interval)
            
            # Perform health checks
            db_healthy = await self.db_manager.health_check()
            cache_healthy = await self.cache_manager.health_check()
            model_healthy = await self.model_manager.health_check()
            service_healthy = await self.service_manager.health_check()
            
            if not all([db_healthy, cache_healthy, model_healthy, service_healthy]):
                logger.warning("Some dependencies are unhealthy")
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Health check loop error: {e}")
```

## 🧪 Testing

### Test Configuration
```python
import pytest
from unittest.mock import Mock, AsyncMock
from fastapi.testclient import TestClient

@pytest.fixture
def mock_dependency_container():
    """Mock dependency container for testing."""
    container = Mock()
    
    # Mock database manager
    container.db_manager = Mock()
    container.db_manager.get_session = AsyncMock()
    container.db_manager.health_check = AsyncMock(return_value=True)
    
    # Mock cache manager
    container.cache_manager = Mock()
    container.cache_manager.get_client = AsyncMock(return_value=None)
    container.cache_manager.health_check = AsyncMock(return_value=True)
    
    # Mock model manager
    container.model_manager = Mock()
    container.model_manager.get_model = AsyncMock(return_value=Mock())
    container.model_manager.health_check = AsyncMock(return_value=True)
    
    # Mock other managers
    container.performance_optimizer = Mock()
    container.training_logger = Mock()
    container.error_handler = Mock()
    
    return container

@pytest.fixture
def test_app(mock_dependency_container):
    """Test FastAPI app with mocked dependencies."""
    app = create_app()
    
    # Override dependency container
    set_dependency_container(mock_dependency_container)
    
    return app

@pytest.fixture
def test_client(test_app):
    """Test client for FastAPI app."""
    return TestClient(test_app)
```

### Test Cases
```python
class TestDependencyInjection:
    """Test cases for dependency injection system."""
    
    def test_health_check(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_detailed_health_check(self, test_client):
        """Test detailed health check endpoint."""
        response = test_client.get("/health/detailed")
        assert response.status_code == 200
        assert response.json()["dependencies"]["overall"] == True
    
    def test_video_processing_with_dependencies(self, test_client):
        """Test video processing with injected dependencies."""
        video_data = {"input": "test_video.mp4"}
        
        response = test_client.post("/video/process", json=video_data)
        assert response.status_code == 200
        assert response.json()["success"] == True
    
    def test_dependency_failure_handling(self, test_client, mock_dependency_container):
        """Test handling of dependency failures."""
        # Mock database failure
        mock_dependency_container.db_manager.health_check = AsyncMock(return_value=False)
        
        response = test_client.get("/health/detailed")
        assert response.status_code == 503
        assert response.json()["dependencies"]["database"] == False
```

## 🚀 Performance Optimization

### Connection Pooling
```python
# Database connection pooling
engine = create_async_engine(
    database_url,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True
)

# Redis connection pooling
client = redis.from_url(
    redis_url,
    encoding="utf-8",
    decode_responses=True,
    max_connections=20
)

# HTTP client pooling
http_client = httpx.AsyncClient(
    timeout=60.0,
    limits=httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20
    )
)
```

### Caching Strategies
```python
# Model caching
models: Dict[str, nn.Module] = {}

async def get_model(model_name: str) -> nn.Module:
    if model_name not in models:
        models[model_name] = await load_model(model_name)
    return models[model_name]

# Configuration caching
@lru_cache()
def get_app_config() -> AppConfig:
    return AppConfig()

# Dependency caching
@cached_dependency(ttl=300)
async def get_expensive_resource() -> Any:
    return await load_expensive_resource()
```

### Memory Management
```python
# GPU memory management
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Model cleanup
async def cleanup_models():
    for name, model in models.items():
        del model
    models.clear()

# Connection cleanup
async def cleanup_connections():
    if engine:
        await engine.dispose()
    if http_client:
        await http_client.aclose()
```

## 🔒 Security Considerations

### Input Validation
```python
from pydantic import BaseModel, Field, validator

class VideoProcessingRequest(BaseModel):
    """Request model with validation."""
    video_url: str = Field(..., description="URL of the video to process")
    processing_type: str = Field(default="caption", description="Type of processing")
    max_length: int = Field(default=100, ge=1, le=500, description="Maximum output length")
    
    @validator("video_url")
    def validate_video_url(cls, v):
        """Validate video URL format and security."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Invalid URL format")
        return v
```

### Rate Limiting
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/video/process")
@limiter.limit("5/minute")
async def process_video(request: Request, ...):
    """Rate-limited video processing endpoint."""
    pass
```

### Authentication and Authorization
```python
from fastapi.security import OAuth2PasswordBearer, SecurityScopes

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    security_scopes: SecurityScopes = SecurityScopes()
) -> Dict[str, Any]:
    """Get current authenticated user."""
    # Validate token and return user info
    pass

def require_scope(required_scope: str):
    """Require specific scope for endpoint access."""
    def dependency(user: Dict[str, Any] = Depends(get_current_user)):
        if required_scope not in user.get("scopes", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return user
    return dependency

@app.post("/video/process")
async def process_video(
    user: Dict[str, Any] = Depends(require_scope("video:process")),
    ...
):
    """Protected video processing endpoint."""
    pass
```

## 📊 Monitoring and Metrics

### Request Tracking
```python
@app.middleware("http")
async def add_request_context(request: Request, call_next):
    """Add request context and performance monitoring."""
    start_time = time.time()
    request.state.request_id = f"req_{int(start_time * 1000)}"
    request.state.start_time = start_time
    
    # Process request
    response = await call_next(request)
    
    # Log request performance
    duration = time.time() - start_time
    logger.info(
        "Request processed",
        request_id=request.state.request_id,
        method=request.method,
        path=request.url.path,
        duration=duration,
        status_code=response.status_code
    )
    
    return response
```

### Dependency Metrics
```python
from prometheus_client import Counter, Histogram, Gauge

# Metrics
DEPENDENCY_REQUESTS = Counter('dependency_requests_total', 'Total dependency requests', ['dependency', 'status'])
DEPENDENCY_DURATION = Histogram('dependency_duration_seconds', 'Dependency request duration', ['dependency'])
DEPENDENCY_HEALTH = Gauge('dependency_health', 'Dependency health status', ['dependency'])

# Usage in dependencies
async def get_model_with_metrics(model_name: str) -> nn.Module:
    start_time = time.time()
    
    try:
        model = await get_model(model_name)
        DEPENDENCY_REQUESTS.labels(dependency="model", status="success").inc()
        DEPENDENCY_HEALTH.labels(dependency="model").set(1)
        return model
    except Exception as e:
        DEPENDENCY_REQUESTS.labels(dependency="model", status="error").inc()
        DEPENDENCY_HEALTH.labels(dependency="model").set(0)
        raise
    finally:
        duration = time.time() - start_time
        DEPENDENCY_DURATION.labels(dependency="model").observe(duration)
```

## 🔄 Error Handling

### Exception Handlers
```python
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "request_id": getattr(request.state, "request_id", None)
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled exception",
        request_id=getattr(request.state, "request_id", None),
        error=str(exc),
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "request_id": getattr(request.state, "request_id", None)
        }
    )
```

### Dependency Error Recovery
```python
async def get_database_session_with_retry() -> AsyncSession:
    """Get database session with retry logic."""
    container = get_dependency_container()
    
    for attempt in range(3):
        try:
            return await container.db_manager.get_session()
        except Exception as e:
            if attempt == 2:  # Last attempt
                raise
            logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
```

## 📋 Best Practices

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

class ModelDependencies:
    @staticmethod
    async def get_video_processor() -> nn.Module:
        pass
    
    @staticmethod
    async def get_caption_generator() -> nn.Module:
        pass
```

### 2. Configuration Management
```python
# Use environment variables for configuration
class AppConfig(BaseModel):
    database_url: str = Field(..., env="DATABASE_URL")
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    debug: bool = Field(False, env="DEBUG")

# Cache configuration
@lru_cache()
def get_config() -> AppConfig:
    return AppConfig()
```

### 3. Resource Lifecycle Management
```python
# Proper initialization and cleanup
async def initialize_dependencies():
    container = get_dependency_container()
    await container.initialize()

async def cleanup_dependencies():
    container = get_dependency_container()
    await container.cleanup()

# Use context managers
@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_dependencies()
    yield
    await cleanup_dependencies()
```

### 4. Testing Dependencies
```python
# Mock dependencies for testing
@pytest.fixture
def mock_dependencies():
    with patch('app.get_dependency_container') as mock_container:
        mock_container.return_value = Mock()
        yield mock_container.return_value

# Test dependency injection
def test_dependency_injection(mock_dependencies):
    # Test that dependencies are properly injected
    pass
```

### 5. Performance Monitoring
```python
# Monitor dependency performance
async def get_model_with_monitoring(model_name: str) -> nn.Module:
    start_time = time.time()
    
    try:
        model = await get_model(model_name)
        return model
    finally:
        duration = time.time() - start_time
        logger.info(f"Model {model_name} loaded in {duration:.3f}s")
```

## 🎯 Summary

This comprehensive FastAPI dependency injection system provides:

1. **Centralized Resource Management** - All dependencies managed in one place
2. **Automatic Lifecycle Management** - Proper initialization and cleanup
3. **Health Monitoring** - Real-time health checks for all dependencies
4. **Performance Optimization** - Connection pooling, caching, and monitoring
5. **Error Handling** - Comprehensive error handling and recovery
6. **Testing Support** - Easy mocking and testing of dependencies
7. **Security** - Input validation, rate limiting, and authentication
8. **Monitoring** - Request tracking and metrics collection

The system follows FastAPI best practices and provides a robust foundation for the Video-OpusClip AI video processing system. 