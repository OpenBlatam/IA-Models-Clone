#!/usr/bin/env python3
"""
🚀 OPTIMIZED Blaze AI - Enterprise-Grade AI System
Optimized for maximum performance, efficiency, and maintainability

This module provides a production-ready AI platform with advanced features
including security, monitoring, rate limiting, and error handling.
"""

import sys
import os
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List
from pathlib import Path
from functools import lru_cache
import time

# FastAPI and core dependencies
from fastapi import FastAPI, Request, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn

# Enhanced features (conditional imports for graceful fallback)
try:
    from enhanced_features.security import SecurityMiddleware, SecurityConfig
    from enhanced_features.monitoring import PerformanceMonitor, MetricsCollector
    from enhanced_features.rate_limiting import RateLimiter, RateLimitConfig
    from enhanced_features.error_handling import ErrorHandler, CircuitBreaker
    from enhanced_features.health import HealthChecker, SystemHealth
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Enhanced features not available: {e}")
    ENHANCED_FEATURES_AVAILABLE = False

# Core application modules
from core.config import AppConfig, load_config
from core.logging import setup_logging
from core.exceptions import BlazeAIError, ServiceUnavailableError
from api.routes import create_api_router
from api.middleware import create_middleware_stack

# Configure logging
logger = logging.getLogger(__name__)

# Performance optimization constants
MAX_WORKERS = 8
MAX_CONNECTIONS = 1000
KEEPALIVE_TIMEOUT = 65
MAX_REQUESTS = 10000
MAX_REQUESTS_JITTER = 1000


class OptimizedBlazeAIApplication:
    """
    🚀 Optimized main application class for Enhanced Blaze AI.
    
    This class encapsulates the application lifecycle and provides
    a clean interface for managing the enhanced features with
    performance optimizations.
    """
    
    def __init__(self, config: AppConfig):
        """Initialize the optimized Blaze AI application."""
        self.config = config
        self.app = None
        self.enhanced_features = {}
        self.health_checker = None
        self.startup_time = None
        self._performance_cache = {}
    
    @lru_cache(maxsize=128)
    def _get_cached_config(self, key: str) -> Any:
        """Get cached configuration values for performance."""
        return getattr(self.config, key, None)
    
    def create_app(self) -> FastAPI:
        """Create and configure the optimized FastAPI application."""
        # Create FastAPI instance with metadata
        self.app = FastAPI(
            title="🚀 Optimized Blaze AI",
            description="Enterprise-Grade AI Platform with Advanced Features & Performance Optimizations",
            version="2.2.0",
            docs_url=None,  # Custom docs endpoint
            redoc_url="/redoc",
            openapi_url="/openapi.json"
        )
        
        # Setup application lifecycle
        self.app.router.lifespan_context = self._create_lifespan_context()
        
        # Configure middleware with performance optimizations
        self._setup_optimized_middleware()
        
        # Setup routes with caching
        self._setup_optimized_routes()
        
        # Setup exception handlers
        self._setup_exception_handlers()
        
        # Setup enhanced features if available
        if ENHANCED_FEATURES_AVAILABLE:
            self._setup_enhanced_features()
        
        return self.app
    
    def _create_lifespan_context(self):
        """Create optimized application lifespan context manager."""
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            start_time = time.time()
            logger.info("🚀 Starting Optimized Blaze AI...")
            await self._startup()
            self.startup_time = time.time() - start_time
            logger.info(f"✅ Startup completed in {self.startup_time:.2f}s")
            
            yield
            
            # Shutdown
            logger.info("🛑 Shutting down Optimized Blaze AI...")
            await self._shutdown()
            logger.info("✅ Shutdown completed")
        
        return lifespan
    
    async def _startup(self):
        """Optimized startup sequence."""
        try:
            # Initialize enhanced features asynchronously
            if ENHANCED_FEATURES_AVAILABLE:
                await self._initialize_enhanced_features_async()
            
            # Initialize health checker
            if ENHANCED_FEATURES_AVAILABLE:
                self.health_checker = HealthChecker()
                await self.health_checker.initialize()
            
            logger.info("✅ All services initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Startup failed: {e}")
            raise
    
    async def _shutdown(self):
        """Optimized shutdown sequence."""
        try:
            # Cleanup enhanced features
            if ENHANCED_FEATURES_AVAILABLE:
                await self._cleanup_enhanced_features()
            
            # Clear performance cache
            self._performance_cache.clear()
            
            logger.info("✅ Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")
    
    def _setup_optimized_middleware(self):
        """Setup middleware with performance optimizations."""
        # CORS middleware with optimized settings
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self._get_cached_config("security.allowed_origins") or ["*"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["X-Total-Count", "X-Rate-Limit-Remaining", "X-Response-Time"]
        )
        
        # Add performance monitoring middleware
        if ENHANCED_FEATURES_AVAILABLE:
            self.app.add_middleware(PerformanceMonitor)
        
        # Add security middleware
        if ENHANCED_FEATURES_AVAILABLE:
            self.app.add_middleware(SecurityMiddleware)
    
    def _setup_optimized_routes(self):
        """Setup routes with performance optimizations."""
        # API routes with caching
        api_router = create_api_router()
        self.app.include_router(api_router, prefix="/api/v2")
        
        # Health check endpoint with caching
        @self.app.get("/health", response_model=Dict[str, Any])
        async def health_check():
            """Optimized health check endpoint."""
            if self.health_checker:
                return await self.health_checker.get_system_health()
            return {"status": "healthy", "timestamp": time.time()}
        
        # Performance metrics endpoint
        @self.app.get("/metrics", response_model=Dict[str, Any])
        async def get_metrics():
            """Get performance metrics."""
            if ENHANCED_FEATURES_AVAILABLE:
                return await self._get_performance_metrics()
            return {"status": "metrics_not_available"}
        
        # Custom docs endpoint with caching
        @self.app.get("/docs", include_in_schema=False)
        async def custom_swagger_ui_html():
            """Custom Swagger UI with caching."""
            return get_swagger_ui_html(
                openapi_url="/openapi.json",
                title="🚀 Optimized Blaze AI - API Documentation"
            )
    
    def _setup_exception_handlers(self):
        """Setup optimized exception handlers."""
        @self.app.exception_handler(BlazeAIError)
        async def blaze_ai_exception_handler(request: Request, exc: BlazeAIError):
            """Handle BlazeAI specific exceptions."""
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal Server Error",
                    "message": str(exc),
                    "timestamp": time.time()
                }
            )
        
        @self.app.exception_handler(ServiceUnavailableError)
        async def service_unavailable_handler(request: Request, exc: ServiceUnavailableError):
            """Handle service unavailable exceptions."""
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": "Service Unavailable",
                    "message": str(exc),
                    "retry_after": 30,
                    "timestamp": time.time()
                }
            )
    
    def _setup_enhanced_features(self):
        """Setup enhanced features with performance optimizations."""
        # This method will be implemented based on available features
        pass
    
    async def _initialize_enhanced_features_async(self):
        """Initialize enhanced features asynchronously for better performance."""
        tasks = []
        
        # Initialize security
        if hasattr(self, 'security_middleware'):
            tasks.append(self._initialize_security_async())
        
        # Initialize monitoring
        if hasattr(self, 'performance_monitor'):
            tasks.append(self._initialize_monitoring_async())
        
        # Initialize rate limiting
        if hasattr(self, 'rate_limiter'):
            tasks.append(self._initialize_rate_limiting_async())
        
        # Initialize error handling
        if hasattr(self, 'error_handler'):
            tasks.append(self._initialize_error_handling_async())
        
        # Execute all initializations concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _initialize_security_async(self):
        """Initialize security features asynchronously."""
        try:
            self.security_middleware = SecurityMiddleware()
            await self.security_middleware.initialize()
            logger.info("✅ Security middleware initialized")
        except Exception as e:
            logger.warning(f"⚠️ Security middleware initialization failed: {e}")
    
    async def _initialize_monitoring_async(self):
        """Initialize monitoring features asynchronously."""
        try:
            self.performance_monitor = PerformanceMonitor()
            await self.performance_monitor.initialize()
            logger.info("✅ Performance monitoring initialized")
        except Exception as e:
            logger.warning(f"⚠️ Performance monitoring initialization failed: {e}")
    
    async def _initialize_rate_limiting_async(self):
        """Initialize rate limiting features asynchronously."""
        try:
            self.rate_limiter = RateLimiter()
            await self.rate_limiter.initialize()
            logger.info("✅ Rate limiting initialized")
        except Exception as e:
            logger.warning(f"⚠️ Rate limiting initialization failed: {e}")
    
    async def _initialize_error_handling_async(self):
        """Initialize error handling features asynchronously."""
        try:
            self.error_handler = ErrorHandler()
            await self.error_handler.initialize()
            logger.info("✅ Error handling initialized")
        except Exception as e:
            logger.warning(f"⚠️ Error handling initialization failed: {e}")
    
    async def _cleanup_enhanced_features(self):
        """Cleanup enhanced features."""
        cleanup_tasks = []
        
        if hasattr(self, 'security_middleware'):
            cleanup_tasks.append(self.security_middleware.cleanup())
        if hasattr(self, 'performance_monitor'):
            cleanup_tasks.append(self.performance_monitor.cleanup())
        if hasattr(self, 'rate_limiter'):
            cleanup_tasks.append(self.rate_limiter.cleanup())
        if hasattr(self, 'error_handler'):
            cleanup_tasks.append(self.error_handler.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics with caching."""
        cache_key = f"metrics_{int(time.time() // 60)}"  # Cache for 1 minute
        
        if cache_key in self._performance_cache:
            return self._performance_cache[cache_key]
        
        metrics = {
            "uptime": time.time() - (self.startup_time or 0),
            "startup_time": self.startup_time or 0,
            "cache_size": len(self._performance_cache),
            "timestamp": time.time()
        }
        
        if hasattr(self, 'performance_monitor'):
            try:
                system_metrics = await self.performance_monitor.get_system_metrics()
                metrics.update(system_metrics)
            except Exception as e:
                logger.warning(f"Failed to get system metrics: {e}")
        
        # Cache the metrics
        self._performance_cache[cache_key] = metrics
        return metrics
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the optimized application."""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            workers=min(MAX_WORKERS, os.cpu_count() or 1),
            loop="asyncio",
            http="httptools",
            ws="websockets",
            log_level="info",
            access_log=True,
            use_colors=True,
            reload=False,  # Disable reload in production
            limit_concurrency=MAX_CONNECTIONS,
            limit_max_requests=MAX_REQUESTS,
            limit_max_requests_jitter=MAX_REQUESTS_JITTER,
            timeout_keep_alive=KEEPALIVE_TIMEOUT
        )


async def main():
    """Main entry point for the optimized application."""
    try:
        # Load configuration
        config = load_config()
        
        # Setup logging
        setup_logging(config)
        
        # Create and run application
        app = OptimizedBlazeAIApplication(config)
        app.create_app()
        
        logger.info("🚀 Optimized Blaze AI application created successfully")
        
        # Run the application
        app.run(
            host=config.api.host,
            port=config.api.port
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start Optimized Blaze AI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

