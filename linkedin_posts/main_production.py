from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, ORJSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import structlog
from .config.settings import Settings
from .api.routes.posts import router as posts_router
from .api.routes.analytics import router as analytics_router
from .api.routes.templates import router as templates_router
from .api.routes.health import router as health_router
from .api.middleware.performance import PerformanceMiddleware
from .api.middleware.security import SecurityMiddleware
from .api.middleware.rate_limit import RateLimitMiddleware
from .api.middleware.cache import CacheMiddleware
from .infrastructure.cache.redis_cache import RedisCache
from .infrastructure.database.session import DatabaseSession
from .core.services.post_service import PostService
from .core.services.ai_service import AIService
from .core.services.analytics_service import AnalyticsService
from .core.services.template_service import TemplateService
from typing import Any, List, Dict, Optional
"""
Production-ready LinkedIn Posts FastAPI application.
"""





# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'linkedin_posts_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status']
)
REQUEST_DURATION = Histogram(
    'linkedin_posts_request_duration_seconds',
    'Request duration',
    ['method', 'endpoint']
)
ACTIVE_CONNECTIONS = Gauge(
    'linkedin_posts_active_connections',
    'Active connections'
)
CACHE_HITS = Counter(
    'linkedin_posts_cache_hits_total',
    'Cache hits'
)
CACHE_MISSES = Counter(
    'linkedin_posts_cache_misses_total',
    'Cache misses'
)

# Global settings
settings = Settings()

# Global services
services: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting LinkedIn Posts application...")
    
    try:
        # Initialize database
        engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        async_session = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        
        # Initialize Redis cache
        redis_client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=settings.redis_max_connections
        )
        
        cache = RedisCache(redis_client)
        
        # Initialize database session
        db_session = DatabaseSession(async_session)
        
        # Initialize services
        ai_service = AIService()
        analytics_service = AnalyticsService(db_session, cache)
        template_service = TemplateService(db_session, cache)
        post_service = PostService(
            db_session, 
            ai_service, 
            analytics_service, 
            template_service
        )
        
        # Store services globally
        services.update({
            'db_session': db_session,
            'cache': cache,
            'ai_service': ai_service,
            'analytics_service': analytics_service,
            'template_service': template_service,
            'post_service': post_service,
            'redis_client': redis_client,
            'engine': engine
        })
        
        # Health check
        await cache.ping()
        logger.info("All services initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down LinkedIn Posts application...")
        
        if 'redis_client' in services:
            await services['redis_client'].close()
        
        if 'engine' in services:
            await services['engine'].dispose()
        
        logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="LinkedIn Posts API",
    description="Production-ready LinkedIn Posts generation and management system",
    version="2.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan,
    default_response_class=ORJSONResponse
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add custom middleware
app.add_middleware(SecurityMiddleware)
app.add_middleware(PerformanceMiddleware)
app.add_middleware(RateLimitMiddleware, redis_client=lambda: services.get('redis_client'))
app.add_middleware(CacheMiddleware, cache=lambda: services.get('cache'))

# Initialize Prometheus instrumentator
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=[".*admin.*", "/metrics"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="linkedin_posts_inprogress",
    inprogress_labels=True,
)

if settings.enable_metrics:
    instrumentator.instrument(app).expose(app)


# Dependency injection
async def get_post_service() -> PostService:
    """Get post service dependency."""
    return services.get('post_service')


async def get_ai_service() -> AIService:
    """Get AI service dependency."""
    return services.get('ai_service')


async def get_analytics_service() -> AnalyticsService:
    """Get analytics service dependency."""
    return services.get('analytics_service')


async def get_template_service() -> TemplateService:
    """Get template service dependency."""
    return services.get('template_service')


async def get_cache() -> RedisCache:
    """Get cache dependency."""
    return services.get('cache')


# Authentication
security = HTTPBearer()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user."""
    # Mock implementation - replace with actual authentication
    return {"user_id": "test-user", "email": "test@example.com"}


# Request/Response middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header."""
    start_time = time.time()
    
    # Track active connections
    ACTIVE_CONNECTIONS.inc()
    
    try:
        response = await call_next(request)
        
        # Add processing time
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # Record metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(process_time)
        
        return response
        
    except Exception as e:
        logger.error(f"Request failed: {e}")
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=500
        ).inc()
        raise
    
    finally:
        ACTIVE_CONNECTIONS.dec()


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "type": "http_exception"
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "type": "server_error"
            }
        }
    )


# Include routers
app.include_router(
    posts_router,
    prefix="/api/v1/posts",
    tags=["posts"],
    dependencies=[Depends(get_current_user)]
)

app.include_router(
    analytics_router,
    prefix="/api/v1/analytics",
    tags=["analytics"],
    dependencies=[Depends(get_current_user)]
)

app.include_router(
    templates_router,
    prefix="/api/v1/templates",
    tags=["templates"],
    dependencies=[Depends(get_current_user)]
)

app.include_router(
    health_router,
    prefix="/health",
    tags=["health"]
)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "LinkedIn Posts API",
        "version": "2.0.0",
        "status": "healthy",
        "timestamp": time.time()
    }


# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics."""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )


# Admin endpoints
@app.get("/admin/stats")
async def get_admin_stats(current_user: dict = Depends(get_current_user)):
    """Get admin statistics."""
    if not current_user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return {
        "active_connections": ACTIVE_CONNECTIONS._value._value,
        "cache_hits": CACHE_HITS._value._value,
        "cache_misses": CACHE_MISSES._value._value,
        "services": list(services.keys())
    }


@app.post("/admin/cache/clear")
async def clear_cache(current_user: dict = Depends(get_current_user)):
    """Clear application cache."""
    if not current_user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    cache = services.get('cache')
    if cache:
        await cache.clear()
        return {"message": "Cache cleared successfully"}
    
    return {"message": "Cache not available"}


# Production server configuration
if __name__ == "__main__":
    uvicorn.run(
        "main_production:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers if not settings.debug else 1,
        loop="uvloop",
        http="httptools",
        access_log=settings.debug,
        use_colors=settings.debug,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processor": structlog.dev.ConsoleRenderer(),
                },
            },
            "handlers": {
                "default": {
                    "level": "INFO",
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    ) 