from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import Annotated, Optional, List, Dict, Any, Protocol
from contextlib import asynccontextmanager
from abc import ABC, abstractmethod
import asyncio
import structlog
from datetime import datetime
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from fastapi import APIRouter
    import uvicorn
from typing import Any, List, Dict, Optional
import logging
﻿"""
Modular Product API - Enterprise Architecture with Specialized Libraries
========================================================================

Demonstrating advanced modular architecture with:

🏗️ MODULAR DESIGN:
- Clean Architecture with separated layers
- Dependency Injection with dependency-injector  
- Service-oriented architecture
- Plugin-based components
- Microservices-ready structure

📚 SPECIALIZED LIBRARIES:
- dependency-injector: IoC container
- structlog: Structured logging
- prometheus-client: Metrics & monitoring
- slowapi: Advanced rate limiting
- sqlalchemy 2.0: Async ORM with connection pooling
- pydantic-settings: Environment-based configuration
- redis-py: High-performance caching
- langchain: AI/LLM integration
- celery: Background tasks
- sentry-sdk: Error tracking
"""


# Dependency Injection

# FastAPI and async

# Monitoring and observability  

# Rate limiting

# Database and caching

# Configuration

# Structured logging setup
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    logger_factory=structlog.WriteLoggerFactory(),
    cache_logger_on_first_use=False,
)

logger = structlog.get_logger(__name__)

# ============================================================================
# CONFIGURATION MODULE - Environment-based settings
# ============================================================================

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)
    
    # Application
    app_name: str = "Modular Product API"
    app_version: str = "3.0.0"
    debug: bool = False
    
    # Database
    database_url: str = Field("postgresql+asyncpg://user:pass@localhost/products")
    db_pool_size: int = 20
    db_max_overflow: int = 30
    
    # Redis
    redis_url: str = Field("redis://localhost:6379/0")
    redis_max_connections: int = 50
    
    # Security
    secret_key: str = Field("your-secret-key-change-in-production")
    rate_limit_per_minute: int = 100
    rate_limit_per_hour: int = 1000
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    sentry_dsn: Optional[str] = None
    
    # AI
    openai_api_key: Optional[str] = None
    enable_ai_features: bool = False

settings = Settings()

# ============================================================================
# METRICS - Prometheus monitoring
# ============================================================================

# Application metrics
REQUEST_COUNT = Counter(
    'http_requests_total', 
    'Total HTTP requests', 
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'active_connections_total',
    'Number of active connections'
)

PRODUCTS_TOTAL = Gauge(
    'products_total',
    'Total number of products'
)

CACHE_OPERATIONS = Counter(
    'cache_operations_total',
    'Total cache operations',
    ['operation', 'status']
)

# ============================================================================
# INTERFACES - Contracts for modular components
# ============================================================================

class ICacheService(Protocol):
    async def get(self, key: str) -> Optional[Any]: ...
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool: ...
    async def delete(self, key: str) -> bool: ...
    async def health_check(self) -> bool: ...

class IProductRepository(Protocol):
    async def create(self, product_data: Dict[str, Any]) -> Dict[str, Any]: ...
    async def get_by_id(self, product_id: str) -> Optional[Dict[str, Any]]: ...
    async def search(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]: ...
    async def update(self, product_id: str, data: Dict[str, Any]) -> Dict[str, Any]: ...
    async def delete(self, product_id: str) -> bool: ...

class INotificationService(Protocol):
    async def send_email(self, to: str, subject: str, body: str) -> bool: ...
    async def send_slack_message(self, channel: str, message: str) -> bool: ...

class IAIService(Protocol):
    async def generate_description(self, product_name: str, features: List[str]) -> str: ...
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]: ...

# ============================================================================
# CORE SERVICES - Modular implementations
# ============================================================================

class RedisService:
    """High-performance Redis service with connection pooling."""
    
    def __init__(self, redis_url: str, max_connections: int = 50):
        
    """__init__ function."""
self.redis_url = redis_url
        self.max_connections = max_connections
        self.pool: Optional[redis.ConnectionPool] = None
        self.client: Optional[redis.Redis] = None
    
    async def initialize(self) -> None:
        """Initialize Redis connection pool."""
        try:
            self.pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                retry_on_timeout=True,
                socket_timeout=5
            )
            self.client = redis.Redis(connection_pool=self.pool)
            await self.client.ping()
            logger.info("✅ Redis service initialized", url=self.redis_url)
        except Exception as e:
            logger.error("❌ Redis initialization failed", error=str(e))
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis with metrics."""
        try:
            value = await self.client.get(key)
            CACHE_OPERATIONS.labels(operation="get", status="hit" if value else "miss").inc()
            return value
        except Exception as e:
            CACHE_OPERATIONS.labels(operation="get", status="error").inc()
            logger.error("Cache get error", key=key, error=str(e))
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis with metrics."""
        try:
            if ttl:
                await self.client.setex(key, ttl, value)
            else:
                await self.client.set(key, value)
            CACHE_OPERATIONS.labels(operation="set", status="success").inc()
            return True
        except Exception as e:
            CACHE_OPERATIONS.labels(operation="set", status="error").inc()
            logger.error("Cache set error", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        try:
            result = await self.client.delete(key)
            CACHE_OPERATIONS.labels(operation="delete", status="success").inc()
            return bool(result)
        except Exception as e:
            CACHE_OPERATIONS.labels(operation="delete", status="error").inc()
            logger.error("Cache delete error", key=key, error=str(e))
            return False
    
    async def health_check(self) -> bool:
        """Check Redis health."""
        try:
            await self.client.ping()
            return True
        except Exception:
            return False
    
    async def close(self) -> None:
        """Close Redis connections."""
        if self.client:
            await self.client.close()
        if self.pool:
            await self.pool.disconnect()
        logger.info("🔌 Redis connections closed")

class DatabaseService:
    """SQLAlchemy async database service."""
    
    def __init__(self, database_url: str, pool_size: int = 20, max_overflow: int = 30):
        
    """__init__ function."""
self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.engine = None
        self.async_session = None
    
    async def initialize(self) -> None:
        """Initialize database engine and session factory."""
        try:
            self.engine = create_async_engine(
                self.database_url,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_pre_ping=True,
                echo=settings.debug
            )
            
            self.async_session = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            logger.info("✅ Database service initialized", url=self.database_url)
        except Exception as e:
            logger.error("❌ Database initialization failed", error=str(e))
            raise
    
    async def get_session(self) -> AsyncSession:
        """Get database session."""
        return self.async_session()
    
    async def health_check(self) -> bool:
        """Check database health."""
        try:
            async with self.get_session() as session:
                await session.execute("SELECT 1")
            return True
        except Exception:
            return False
    
    async def close(self) -> None:
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
        logger.info("🔌 Database connections closed")

class ProductService:
    """Business logic service for products."""
    
    def __init__(self, cache_service: ICacheService, repository: IProductRepository):
        
    """__init__ function."""
self.cache = cache_service
        self.repository = repository
    
    async def create_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create product with caching."""
        product = await self.repository.create(product_data)
        
        # Cache the new product
        await self.cache.set(f"product:{product['id']}", product, ttl=3600)
        
        # Update metrics
        PRODUCTS_TOTAL.inc()
        
        logger.info("Product created", product_id=product["id"], name=product["name"])
        return product
    
    async def get_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get product with caching."""
        # Try cache first
        cached_product = await self.cache.get(f"product:{product_id}")
        if cached_product:
            return cached_product
        
        # Get from repository
        product = await self.repository.get_by_id(product_id)
        if product:
            # Cache for future requests
            await self.cache.set(f"product:{product_id}", product, ttl=3600)
        
        return product
    
    async def search_products(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search products with caching."""
        cache_key = f"search:{hash(str(sorted(filters.items())))}"
        
        # Try cache first
        cached_results = await self.cache.get(cache_key)
        if cached_results:
            return cached_results
        
        # Search in repository
        results = await self.repository.search(filters)
        
        # Cache results for 5 minutes
        await self.cache.set(cache_key, results, ttl=300)
        
        return results

class AIService:
    """AI/ML service for intelligent features."""
    
    def __init__(self, cache_service: ICacheService, openai_api_key: Optional[str] = None):
        
    """__init__ function."""
self.cache = cache_service
        self.openai_api_key = openai_api_key
        self.enabled = bool(openai_api_key)
    
    async def generate_description(self, product_name: str, features: List[str]) -> str:
        """Generate product description using AI."""
        if not self.enabled:
            return f"AI-generated description for {product_name} with features: {', '.join(features)}"
        
        cache_key = f"ai_desc:{hash(product_name + ''.join(features))}"
        
        # Try cache first
        cached_desc = await self.cache.get(cache_key)
        if cached_desc:
            return cached_desc
        
        # TODO: Integrate with OpenAI API
        # For now, return a mock description
        description = f"Innovative {product_name} featuring {', '.join(features[:3])}. Perfect for modern consumers seeking quality and reliability."
        
        # Cache for 24 hours
        await self.cache.set(cache_key, description, ttl=86400)
        
        logger.info("AI description generated", product=product_name)
        return description

# ============================================================================
# DEPENDENCY INJECTION CONTAINER
# ============================================================================

class Container(containers.DeclarativeContainer):
    """Main dependency injection container."""
    
    # Configuration
    config = providers.Configuration()
    
    # Core services
    redis_service = providers.Singleton(
        RedisService,
        redis_url=config.redis.url,
        max_connections=config.redis.max_connections
    )
    
    database_service = providers.Singleton(
        DatabaseService,
        database_url=config.database.url,
        pool_size=config.database.pool_size,
        max_overflow=config.database.max_overflow
    )
    
    # Business services
    product_service = providers.Factory(
        ProductService,
        cache_service=redis_service,
        repository=providers.Factory("ProductRepository", database_service=database_service)
    )
    
    ai_service = providers.Factory(
        AIService,
        cache_service=redis_service,
        openai_api_key=config.ai.openai_api_key
    )

# ============================================================================
# MIDDLEWARE - Cross-cutting concerns
# ============================================================================

class MetricsMiddleware:
    """Prometheus metrics middleware."""
    
    def __init__(self, app) -> Any:
        self.app = app
    
    async def __call__(self, scope, receive, send) -> Any:
        if scope["type"] == "http":
            method = scope["method"]
            path = scope["path"]
            
            ACTIVE_CONNECTIONS.inc()
            
            start_time = asyncio.get_event_loop().time()
            
            try:
                await self.app(scope, receive, send)
            finally:
                duration = asyncio.get_event_loop().time() - start_time
                REQUEST_DURATION.labels(method=method, endpoint=path).observe(duration)
                ACTIVE_CONNECTIONS.dec()
        else:
            await self.app(scope, receive, send)

# ============================================================================
# MODULAR ROUTERS
# ============================================================================


# Products router
products_router = APIRouter()

@products_router.post("/", status_code=status.HTTP_201_CREATED)
@inject
async def create_product(
    request: Dict[str, Any],
    product_service: ProductService = Depends(Provide[Container.product_service])
):
    """Create a new product."""
    try:
        product = await product_service.create_product(request)
        REQUEST_COUNT.labels(method="POST", endpoint="/products", status_code="201").inc()
        return {"success": True, "data": product}
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/products", status_code="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

@products_router.get("/{product_id}")
@inject
async def get_product(
    product_id: str,
    product_service: ProductService = Depends(Provide[Container.product_service])
):
    """Get product by ID."""
    product = await product_service.get_product(product_id)
    if not product:
        REQUEST_COUNT.labels(method="GET", endpoint="/products/{id}", status_code="404").inc()
        raise HTTPException(status_code=404, detail="Product not found")
    
    REQUEST_COUNT.labels(method="GET", endpoint="/products/{id}", status_code="200").inc()
    return {"success": True, "data": product}

@products_router.post("/search")
@inject
async def search_products(
    filters: Dict[str, Any],
    product_service: ProductService = Depends(Provide[Container.product_service])
):
    """Search products with filters."""
    results = await product_service.search_products(filters)
    REQUEST_COUNT.labels(method="POST", endpoint="/products/search", status_code="200").inc()
    return {"success": True, "data": results}

# AI router
ai_router = APIRouter()

@ai_router.post("/generate-description")
@inject
async def generate_description(
    request: Dict[str, Any],
    ai_service: AIService = Depends(Provide[Container.ai_service])
):
    """Generate AI product description."""
    description = await ai_service.generate_description(
        request["product_name"],
        request.get("features", [])
    )
    
    REQUEST_COUNT.labels(method="POST", endpoint="/ai/generate-description", status_code="200").inc()
    return {"success": True, "data": {"description": description}}

# Health router
health_router = APIRouter()

@health_router.get("/")
@inject
async def health_check(
    redis_service: RedisService = Depends(Provide[Container.redis_service]),
    database_service: DatabaseService = Depends(Provide[Container.database_service])
):
    """Comprehensive health check."""
    redis_healthy = await redis_service.health_check()
    db_healthy = await database_service.health_check()
    
    overall_status = "healthy" if redis_healthy and db_healthy else "degraded"
    
    return {
        "status": overall_status,
        "services": {
            "redis": "healthy" if redis_healthy else "unhealthy",
            "database": "healthy" if db_healthy else "unhealthy"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================================================
# APPLICATION FACTORY
# ============================================================================

def create_modular_app() -> FastAPI:
    """Create modular FastAPI application."""
    
    # Initialize Sentry if DSN provided
    if settings.sentry_dsn:
        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            integrations=[FastApiIntegration(auto_enable=False)],
            traces_sample_rate=0.1,
        )
    
    # Create container and configure
    container = Container()
    container.config.from_dict({
        "redis": {"url": settings.redis_url, "max_connections": settings.redis_max_connections},
        "database": {"url": settings.database_url, "pool_size": settings.db_pool_size, "max_overflow": settings.db_max_overflow},
        "ai": {"openai_api_key": settings.openai_api_key}
    })
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Modular Product API with enterprise architecture"
    )
    
    # Configure rate limiter
    limiter = Limiter(
        key_func=get_remote_address,
        storage_uri=settings.redis_url,
        default_limits=[f"{settings.rate_limit_per_hour}/hour", f"{settings.rate_limit_per_minute}/minute"]
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # Add middleware
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])
    app.add_middleware(MetricsMiddleware)
    
    # Include routers
    app.include_router(products_router, prefix="/api/v1/products", tags=["Products"])
    app.include_router(ai_router, prefix="/api/v1/ai", tags=["AI"])
    app.include_router(health_router, prefix="/health", tags=["Health"])
    
    # Wire container
    container.wire(modules=[__name__])
    app.container = container
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        redis_service = container.redis_service()
        database_service = container.database_service()
        
        await redis_service.initialize()
        await database_service.initialize()
        
        # Start Prometheus metrics server
        if settings.enable_metrics:
            start_http_server(settings.metrics_port)
        
        logger.info("🚀 Modular API started successfully")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        redis_service = container.redis_service()
        database_service = container.database_service()
        
        await redis_service.close()
        await database_service.close()
        
        logger.info("✅ Modular API shutdown complete")
    
    return app

# Create app instance
app = create_modular_app()

if __name__ == "__main__":
    
    print("🚀 Starting Modular Product API...")
    print("📊 Features enabled:")
    print("  ✅ Dependency Injection with dependency-injector")
    print("  ✅ Structured Logging with structlog")
    print("  ✅ Metrics with Prometheus")
    print("  ✅ Rate Limiting with slowapi")
    print("  ✅ Async Database with SQLAlchemy 2.0")
    print("  ✅ Redis Caching with connection pooling")
    print("  ✅ AI Integration ready")
    print("  ✅ Error Tracking with Sentry")
    print("  ✅ Health Checks comprehensive")
    print("  ✅ Clean Architecture patterns")
    print("")
    print("📡 Endpoints:")
    print("  POST /api/v1/products - Create product")
    print("  GET  /api/v1/products/{id} - Get product")
    print("  POST /api/v1/products/search - Search products")
    print("  POST /api/v1/ai/generate-description - AI description")
    print("  GET  /health - Health check")
    print("  GET  /metrics - Prometheus metrics (port 9090)")
    print("")
    
    uvicorn.run(
        "MODULAR_API_DEMO:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
