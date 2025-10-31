"""
🔧 FastAPI Dependency Injection System for Video-OpusClip

This module provides a comprehensive dependency injection system for managing
state and shared resources in the Video-OpusClip AI video processing system.

Features:
- Centralized dependency management with proper lifecycle
- Resource pooling and connection management
- Configuration-driven dependency injection
- Health monitoring and error handling
- Testing support with dependency mocking
- Performance monitoring and metrics
- Graceful degradation and fallbacks
- Type-safe dependency injection
"""

import asyncio
import time
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Callable, Awaitable, Union, Type, TypeVar, AsyncGenerator
from dataclasses import dataclass, field
from functools import lru_cache
from enum import Enum
from datetime import datetime, timedelta
import weakref

import torch
import torch.nn as nn
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import structlog
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import QueuePool
import httpx

# Import Video-OpusClip components
from .pydantic_models import VideoProcessingConfig, ModelConfig, PerformanceConfig
from .performance_optimizer import PerformanceOptimizer
from .training_logger import TrainingLogger
from .enhanced_error_handling import ErrorHandler
from .mixed_precision_training import MixedPrecisionManager
from .gradient_accumulation import GradientAccumulator
from .multi_gpu_training import MultiGPUTrainer
from .code_profiler import CodeProfiler

# Configure logging
logger = structlog.get_logger(__name__)

# Type variables
T = TypeVar('T')
DependencyT = TypeVar('DependencyT')

# =============================================================================
# Configuration Models
# =============================================================================

class DependencyScope(str, Enum):
    """Dependency scopes for lifecycle management."""
    SINGLETON = "singleton"      # Single instance for entire application
    REQUEST = "request"          # New instance per request
    SESSION = "session"          # Shared within user session
    TRANSIENT = "transient"      # New instance each time
    BACKGROUND = "background"    # Background task scope

class ResourceType(str, Enum):
    """Types of shared resources."""
    DATABASE = "database"
    CACHE = "cache"
    MODEL = "model"
    OPTIMIZER = "optimizer"
    LOGGER = "logger"
    MONITOR = "monitor"
    CLIENT = "client"
    SERVICE = "service"

class DependencyConfig(BaseModel):
    """Configuration for dependency injection."""
    
    # Core settings
    name: str = Field(..., description="Dependency name")
    type: ResourceType = Field(..., description="Resource type")
    scope: DependencyScope = Field(DependencyScope.SINGLETON, description="Dependency scope")
    
    # Performance settings
    cache_enabled: bool = Field(True, description="Enable caching")
    cache_ttl: int = Field(300, description="Cache TTL in seconds")
    max_instances: int = Field(10, description="Maximum instances for pooling")
    
    # Health monitoring
    health_check_enabled: bool = Field(True, description="Enable health checks")
    health_check_interval: int = Field(60, description="Health check interval in seconds")
    
    # Error handling
    retry_attempts: int = Field(3, description="Number of retry attempts")
    retry_delay: float = Field(1.0, description="Delay between retries in seconds")
    
    # Resource-specific settings
    resource_config: Dict[str, Any] = Field(default_factory=dict, description="Resource-specific configuration")

class AppConfig(BaseModel):
    """Application configuration for dependency injection."""
    
    # Database settings
    database_url: str = Field(..., env="DATABASE_URL")
    database_pool_size: int = Field(20, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(30, env="DATABASE_MAX_OVERFLOW")
    
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
    
    # Security settings
    secret_key: str = Field(..., env="SECRET_KEY")
    debug: bool = Field(False, env="DEBUG")
    
    @validator("device")
    def validate_device(cls, v):
        """Validate device configuration."""
        if v == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif v in ["cpu", "cuda", "mps"]:
            return v
        else:
            raise ValueError(f"Invalid device: {v}")

# =============================================================================
# Resource Managers
# =============================================================================

class DatabaseManager:
    """Database connection manager with connection pooling."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.engine = None
        self.session_factory = None
        self._health_status = True
        self._last_health_check = datetime.now()
    
    async def initialize(self):
        """Initialize database connection pool."""
        try:
            self.engine = create_async_engine(
                self.config.database_url,
                poolclass=QueuePool,
                pool_size=self.config.database_pool_size,
                max_overflow=self.config.database_max_overflow,
                pool_pre_ping=True,
                echo=self.config.debug
            )
            
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            logger.info("Database manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            raise
    
    async def get_session(self) -> AsyncSession:
        """Get database session from pool."""
        if not self.session_factory:
            raise RuntimeError("Database manager not initialized")
        
        return self.session_factory()
    
    async def health_check(self) -> bool:
        """Perform health check on database."""
        try:
            async with self.engine.begin() as conn:
                await conn.execute("SELECT 1")
            self._health_status = True
            self._last_health_check = datetime.now()
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            self._health_status = False
            return False
    
    async def cleanup(self):
        """Cleanup database connections."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database manager cleaned up")

class CacheManager:
    """Redis cache manager with connection pooling."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = None
        self._health_status = True
        self._last_health_check = datetime.now()
    
    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.client = redis.from_url(
                self.config.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20
            )
            
            # Test connection
            await self.client.ping()
            logger.info("Cache manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize cache manager: {e}")
            if not self.config.cache_enabled:
                logger.warning("Cache disabled, continuing without cache")
            else:
                raise
    
    async def get_client(self) -> Optional[redis.Redis]:
        """Get Redis client."""
        return self.client if self.config.cache_enabled else None
    
    async def health_check(self) -> bool:
        """Perform health check on cache."""
        if not self.client or not self.config.cache_enabled:
            return True
        
        try:
            await self.client.ping()
            self._health_status = True
            self._last_health_check = datetime.now()
            return True
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            self._health_status = False
            return False
    
    async def cleanup(self):
        """Cleanup cache connections."""
        if self.client:
            await self.client.close()
            logger.info("Cache manager cleaned up")

class ModelManager:
    """AI model manager with caching and optimization."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.models: Dict[str, nn.Module] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self._device = self._get_device()
        self._health_status = True
    
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
        
        try:
            # Load model (this would be your actual model loading logic)
            model = await self._load_model_async(model_name, config)
            
            # Move to device
            model = model.to(self._device)
            
            # Enable mixed precision if configured
            if self.config.mixed_precision and self._device.type == "cuda":
                model = model.half()
            
            # Cache model
            self.models[model_name] = model
            self.model_configs[model_name] = config
            
            logger.info(f"Model {model_name} loaded successfully on {self._device}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    async def _load_model_async(self, model_name: str, config: ModelConfig) -> nn.Module:
        """Async model loading implementation."""
        # This would be your actual model loading logic
        # For now, we'll create a dummy model
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        # Simulate async loading
        await asyncio.sleep(0.1)
        
        return model
    
    async def get_model(self, model_name: str) -> Optional[nn.Module]:
        """Get a cached model."""
        return self.models.get(model_name)
    
    async def health_check(self) -> bool:
        """Perform health check on models."""
        try:
            # Check if models are accessible
            for name, model in self.models.items():
                if not hasattr(model, 'parameters'):
                    self._health_status = False
                    return False
            
            self._health_status = True
            return True
        except Exception as e:
            logger.error(f"Model health check failed: {e}")
            self._health_status = False
            return False
    
    async def cleanup(self):
        """Cleanup model resources."""
        for name, model in self.models.items():
            del model
        self.models.clear()
        self.model_configs.clear()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model manager cleaned up")

class ServiceManager:
    """External service manager with HTTP client pooling."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.http_client = None
        self.services: Dict[str, Any] = {}
        self._health_status = True
    
    async def initialize(self):
        """Initialize HTTP client and services."""
        try:
            self.http_client = httpx.AsyncClient(
                timeout=self.config.request_timeout,
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20
                )
            )
            
            logger.info("Service manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize service manager: {e}")
            raise
    
    async def get_http_client(self) -> httpx.AsyncClient:
        """Get HTTP client."""
        return self.http_client
    
    async def register_service(self, name: str, service: Any):
        """Register an external service."""
        self.services[name] = service
        logger.info(f"Service {name} registered")
    
    async def get_service(self, name: str) -> Optional[Any]:
        """Get a registered service."""
        return self.services.get(name)
    
    async def health_check(self) -> bool:
        """Perform health check on services."""
        try:
            # Check HTTP client
            if self.http_client:
                response = await self.http_client.get("http://httpbin.org/status/200")
                if response.status_code != 200:
                    self._health_status = False
                    return False
            
            self._health_status = True
            return True
        except Exception as e:
            logger.error(f"Service health check failed: {e}")
            self._health_status = False
            return False
    
    async def cleanup(self):
        """Cleanup service connections."""
        if self.http_client:
            await self.http_client.aclose()
        
        self.services.clear()
        logger.info("Service manager cleaned up")

# =============================================================================
# Dependency Container
# =============================================================================

class DependencyContainer:
    """
    Central dependency injection container for Video-OpusClip.
    
    Manages the lifecycle of all shared resources and provides
    dependency injection functions for FastAPI routes.
    """
    
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
        
        # State management
        self._initialized = False
        self._startup_time = None
        self._health_check_task = None
        self._cleanup_task = None
        self._lock = asyncio.Lock()
        
        logger.info("Dependency container initialized")
    
    async def initialize(self):
        """Initialize all dependency managers."""
        async with self._lock:
            if self._initialized:
                return
            
            try:
                logger.info("Initializing dependency container...")
                
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
                
                # Start health check task
                if self.config.enable_health_checks:
                    self._health_check_task = asyncio.create_task(self._health_check_loop())
                
                self._initialized = True
                self._startup_time = datetime.now()
                
                logger.info("Dependency container initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize dependency container: {e}")
                await self.cleanup()
                raise
    
    async def cleanup(self):
        """Cleanup all dependency managers."""
        async with self._lock:
            if not self._initialized:
                return
            
            logger.info("Cleaning up dependency container...")
            
            # Cancel background tasks
            if self._health_check_task and not self._health_check_task.done():
                self._health_check_task.cancel()
                await self._health_check_task
            
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
            logger.info("Dependency container cleaned up")
    
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
    
    # Dependency injection functions
    def get_db_session_dependency(self):
        """Get database session dependency."""
        async def dependency() -> AsyncSession:
            async with self.db_manager.get_session() as session:
                yield session
        return dependency
    
    def get_cache_client_dependency(self):
        """Get cache client dependency."""
        async def dependency() -> Optional[redis.Redis]:
            return await self.cache_manager.get_client()
        return dependency
    
    def get_model_dependency(self, model_name: str):
        """Get model dependency."""
        async def dependency() -> nn.Module:
            model = await self.model_manager.get_model(model_name)
            if not model:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Model {model_name} not available"
                )
            return model
        return dependency
    
    def get_http_client_dependency(self):
        """Get HTTP client dependency."""
        async def dependency() -> httpx.AsyncClient:
            return await self.service_manager.get_http_client()
        return dependency
    
    def get_performance_optimizer_dependency(self):
        """Get performance optimizer dependency."""
        async def dependency() -> PerformanceOptimizer:
            return self.performance_optimizer
        return dependency
    
    def get_training_logger_dependency(self):
        """Get training logger dependency."""
        async def dependency() -> TrainingLogger:
            return self.training_logger
        return dependency
    
    def get_error_handler_dependency(self):
        """Get error handler dependency."""
        async def dependency() -> ErrorHandler:
            return self.error_handler
        return dependency
    
    def get_mixed_precision_manager_dependency(self):
        """Get mixed precision manager dependency."""
        async def dependency() -> MixedPrecisionManager:
            return self.mixed_precision_manager
        return dependency
    
    def get_gradient_accumulator_dependency(self):
        """Get gradient accumulator dependency."""
        async def dependency() -> GradientAccumulator:
            return self.gradient_accumulator
        return dependency
    
    def get_multi_gpu_trainer_dependency(self):
        """Get multi-GPU trainer dependency."""
        async def dependency() -> MultiGPUTrainer:
            return self.multi_gpu_trainer
        return dependency
    
    def get_code_profiler_dependency(self):
        """Get code profiler dependency."""
        async def dependency() -> CodeProfiler:
            return self.code_profiler
        return dependency

# =============================================================================
# Global Container Management
# =============================================================================

# Global container instance
_container: Optional[DependencyContainer] = None

def get_dependency_container() -> DependencyContainer:
    """Get the global dependency container."""
    if _container is None:
        raise RuntimeError("Dependency container not initialized")
    return _container

def set_dependency_container(container: DependencyContainer):
    """Set the global dependency container."""
    global _container
    _container = container

@lru_cache()
def get_app_config() -> AppConfig:
    """Get application configuration (cached)."""
    return AppConfig()

# =============================================================================
# FastAPI Application Factory
# =============================================================================

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
        """Add request context and performance monitoring."""
        start_time = time.time()
        
        # Add request ID
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
    
    # Add exception handlers
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
    
    return app

# =============================================================================
# Dependency Injection Decorators
# =============================================================================

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

# =============================================================================
# Health Check Endpoints
# =============================================================================

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

# =============================================================================
# Example Usage
# =============================================================================

def create_example_app() -> FastAPI:
    """Create example FastAPI app with dependency injection."""
    
    app = create_app()
    add_health_endpoints(app)
    
    # Example route using dependency injection
    @app.post("/video/process")
    @inject_dependencies(
        get_dependency_container().get_model_dependency("video_processor"),
        get_dependency_container().get_performance_optimizer_dependency(),
        get_dependency_container().get_error_handler_dependency()
    )
    async def process_video(
        video_data: Dict[str, Any],
        model: nn.Module = Depends(get_dependency_container().get_model_dependency("video_processor")),
        optimizer: PerformanceOptimizer = Depends(get_dependency_container().get_performance_optimizer_dependency()),
        error_handler: ErrorHandler = Depends(get_dependency_container().get_error_handler_dependency())
    ):
        """Process video using injected dependencies."""
        
        try:
            # Use injected dependencies
            with optimizer.optimize_context():
                result = await model(video_data)
            
            return {"success": True, "result": result}
            
        except Exception as e:
            await error_handler.handle_error(e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Video processing failed"
            )
    
    return app

if __name__ == "__main__":
    # Example usage
    app = create_example_app()
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 