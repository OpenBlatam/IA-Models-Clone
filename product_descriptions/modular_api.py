from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import logging.config
from contextlib import asynccontextmanager
from pathlib import Path
import structlog
import uvicorn
from dependency_injector.wiring import Provide, inject
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.middleware.sessions import SessionMiddleware
from .containers import Container
from .core.middleware import (
from .core.exceptions import setup_exception_handlers
from .routers import (
from .core.config import get_settings
from .core.monitoring import setup_monitoring
from .services.cache import CacheService
from .services.database import DatabaseService
    import subprocess
    import sys
    import subprocess
    import subprocess
    import sys
from typing import Any, List, Dict, Optional
"""
Modular Product API - Enterprise Architecture
============================================

Modern, modular FastAPI application with:
- Clean Architecture principles
- Dependency Injection with dependency-injector
- Advanced monitoring with Prometheus + Structlog
- High-performance caching with Redis + DiskCache
- Rate limiting with SlowAPI
- Database with SQLAlchemy 2.0 + async
- AI integration with LangChain + OpenAI
- Comprehensive error handling
- Type safety throughout
- Modular design patterns
"""



# Internal imports - modular structure
    LoggingMiddleware,
    PerformanceMiddleware,
    SecurityMiddleware,
    ErrorHandlingMiddleware
)
    products_router,
    health_router,
    analytics_router,
    ai_router,
    admin_router
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.WriteLoggerFactory(),
    cache_logger_on_first_use=False,
)

logger = structlog.get_logger(__name__)


class ModularAPIApplication:
    """
    Modular API Application with Enterprise Features
    
    Features:
    - Clean Architecture with DI Container
    - Advanced monitoring and observability
    - High-performance caching strategies
    - Rate limiting and security
    - AI-powered features
    - Comprehensive error handling
    - Type-safe throughout
    """
    
    def __init__(self) -> Any:
        self.settings = get_settings()
        self.container = Container()
        self.app: Optional[FastAPI] = None
        self.instrumentator: Optional[Instrumentator] = None
        
    def create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        
        # Create FastAPI app with lifespan
        self.app = FastAPI(
            title=self.settings.api.title,
            description=self.settings.api.description,
            version=self.settings.api.version,
            lifespan=self._lifespan,
            docs_url=self.settings.api.docs_url if not self.settings.is_production else None,
            redoc_url=self.settings.api.redoc_url if not self.settings.is_production else None,
            openapi_url=self.settings.api.openapi_url if not self.settings.is_production else None,
        )
        
        # Configure dependency injection
        self.container.config.from_pydantic(self.settings)
        self.container.wire(modules=[
            ".routers.products",
            ".routers.health", 
            ".routers.analytics",
            ".routers.ai",
            ".routers.admin",
            ".services.product_service",
            ".services.ai_service",
            ".services.analytics_service"
        ])
        
        # Setup components
        self._setup_middleware()
        self._setup_routers()
        self._setup_monitoring()
        self._setup_exception_handlers()
        
        logger.info("🚀 Modular API application created successfully")
        return self.app
    
    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """Application lifespan management."""
        # Startup
        logger.info("🔄 Starting application services...")
        
        try:
            # Initialize services
            await self._startup_services()
            logger.info("✅ All services started successfully")
            
            yield
            
        finally:
            # Shutdown
            logger.info("🛑 Shutting down application services...")
            await self._shutdown_services()
            logger.info("✅ All services shut down successfully")
    
    async def _startup_services(self) -> Any:
        """Initialize all application services."""
        
        # Initialize database
        db_service: DatabaseService = self.container.database_service()
        await db_service.initialize()
        
        # Initialize cache
        cache_service: CacheService = self.container.cache_service()
        await cache_service.initialize()
        
        # Warm up cache if needed
        if self.settings.is_production:
            await cache_service.warm_up()
        
        # Setup monitoring
        if self.settings.monitoring.enable_metrics:
            await setup_monitoring(self.app)
        
        logger.info("🎯 Services initialization completed")
    
    async def _shutdown_services(self) -> Any:
        """Cleanup all application services."""
        
        # Close database connections
        db_service: DatabaseService = self.container.database_service()
        await db_service.close()
        
        # Close cache connections
        cache_service: CacheService = self.container.cache_service()
        await cache_service.close()
        
        logger.info("🧹 Services cleanup completed")
    
    def _setup_middleware(self) -> Any:
        """Setup application middleware in correct order."""
        
        # Rate limiting (first to reject requests early)
        if self.settings.security.rate_limit_enabled:
            limiter = Limiter(
                key_func=get_remote_address,
                storage_uri=self.settings.redis.redis_url,
                default_limits=[f"{self.settings.security.rate_limit_requests}/hour"]
            )
            self.app.state.limiter = limiter
            self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
            self.app.add_middleware(SlowAPIMiddleware)
        
        # Security middleware
        self.app.add_middleware(SecurityMiddleware)
        
        # Trusted hosts (production)
        if self.settings.is_production:
            self.app.add_middleware(
                TrustedHostMiddleware, 
                allowed_hosts=["*.yourdomain.com", "yourdomain.com"]
            )
        
        # Session middleware (if needed)
        self.app.add_middleware(
            SessionMiddleware,
            secret_key=self.settings.security.jwt_secret_key
        )
        
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.settings.security.cors_origins,
            allow_credentials=self.settings.security.cors_allow_credentials,
            allow_methods=self.settings.security.cors_allow_methods,
            allow_headers=self.settings.security.cors_allow_headers,
        )
        
        # Compression
        if self.settings.api.enable_compression:
            self.app.add_middleware(
                GZipMiddleware, 
                minimum_size=self.settings.api.compression_minimum_size
            )
        
        # Custom middleware (order matters - last added runs first)
        self.app.add_middleware(PerformanceMiddleware)
        self.app.add_middleware(ErrorHandlingMiddleware)
        self.app.add_middleware(LoggingMiddleware)
        
        logger.info("🛡️ Middleware stack configured")
    
    def _setup_routers(self) -> Any:
        """Setup application routers."""
        
        # API v1 routes
        api_v1_prefix = "/api/v1"
        
        self.app.include_router(
            health_router.router,
            prefix="/health",
            tags=["Health & Monitoring"]
        )
        
        self.app.include_router(
            products_router.router,
            prefix=f"{api_v1_prefix}/products",
            tags=["Products"]
        )
        
        self.app.include_router(
            analytics_router.router,
            prefix=f"{api_v1_prefix}/analytics",
            tags=["Analytics"]
        )
        
        self.app.include_router(
            ai_router.router,
            prefix=f"{api_v1_prefix}/ai",
            tags=["AI & ML"]
        )
        
        # Admin routes (protected)
        if not self.settings.is_production or self.settings.debug:
            self.app.include_router(
                admin_router.router,
                prefix="/admin",
                tags=["Administration"]
            )
        
        logger.info("🔗 API routes configured")
    
    def _setup_monitoring(self) -> Any:
        """Setup monitoring and observability."""
        
        if self.settings.monitoring.enable_metrics:
            # Prometheus metrics
            self.instrumentator = Instrumentator(
                should_group_status_codes=False,
                should_ignore_untemplated=True,
                should_respect_env_var=True,
                should_instrument_requests_inprogress=True,
                excluded_handlers=["/health", "/metrics"],
                env_var_name="ENABLE_METRICS",
                inprogress_name="http_requests_inprogress",
                inprogress_labels=True,
            )
            
            self.instrumentator.instrument(self.app)
            self.instrumentator.expose(self.app, endpoint="/metrics")
            
            logger.info("📊 Prometheus metrics enabled")
        
        # Health check endpoint
        @self.app.get("/", include_in_schema=False)
        async def root():
            
    """root function."""
return {
                "service": self.settings.api.title,
                "version": self.settings.api.version,
                "status": "operational",
                "environment": self.settings.environment.value,
                "docs": self.settings.api.docs_url,
                "metrics": "/metrics" if self.settings.monitoring.enable_metrics else None
            }
    
    def _setup_exception_handlers(self) -> Any:
        """Setup global exception handlers."""
        setup_exception_handlers(self.app)
        logger.info("⚠️ Exception handlers configured")


# Dependency injection for request-scoped services
@inject
async def get_product_service(
    service: "ProductService" = Depends(Provide[Container.product_service])
):
    """Get product service instance."""
    return service


@inject  
async def get_cache_service(
    service: "CacheService" = Depends(Provide[Container.cache_service])
):
    """Get cache service instance."""
    return service


@inject
async def get_ai_service(
    service: "AIService" = Depends(Provide[Container.ai_service])
):
    """Get AI service instance."""
    return service


@inject
async def get_analytics_service(
    service: "AnalyticsService" = Depends(Provide[Container.analytics_service])
):
    """Get analytics service instance."""
    return service


# Factory function
def create_modular_app() -> FastAPI:
    """Create modular application instance."""
    app_factory = ModularAPIApplication()
    return app_factory.create_app()


# Application instance
app = create_modular_app()


# CLI Commands for development
def run_development_server():
    """Run development server with auto-reload."""
    settings = get_settings()
    
    uvicorn.run(
        "modular_api:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        log_config=settings.get_logging_config(),
        access_log=True,
        use_colors=True,
        loop="asyncio"
    )


def run_production_server():
    """Run production server with Gunicorn."""
    
    settings = get_settings()
    
    cmd = [
        "gunicorn",
        "modular_api:app",
        "-w", str(settings.api.workers),
        "-k", "uvicorn.workers.UvicornWorker",
        "--bind", f"{settings.api.host}:{settings.api.port}",
        "--access-logfile", "-",
        "--error-logfile", "-",
        "--log-level", settings.monitoring.log_level.value.lower(),
        "--timeout", "120",
        "--keep-alive", "5",
        "--max-requests", "1000",
        "--max-requests-jitter", "100",
        "--preload"
    ]
    
    subprocess.run(cmd)


def migrate_database():
    """Run database migrations."""
    
    settings = get_settings()
    env = {"DATABASE_URL": settings.database.async_database_url}
    
    subprocess.run(["alembic", "upgrade", "head"], env=env)


def create_migration(message: str):
    """Create new database migration."""
    
    settings = get_settings()
    env = {"DATABASE_URL": settings.database.async_database_url}
    
    subprocess.run(["alembic", "revision", "--autogenerate", "-m", message], env=env)


if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "dev":
            run_development_server()
        elif command == "prod":
            run_production_server()
        elif command == "migrate":
            migrate_database()
        elmatch command:
    case "makemigrations":
            message = sys.argv[2] if len(sys.argv) > 2 else "Auto-generated migration"
            create_migration(message)
        else:
            print("Unknown command. Available: dev, prod, migrate, makemigrations")
    else:
        run_development_server() 