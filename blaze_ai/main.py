#!/usr/bin/env python3
"""
Enhanced Blaze AI - Enterprise-Grade AI System
Refactored for better maintainability and organization

This module provides a production-ready AI platform with advanced features
including security, monitoring, rate limiting, and error handling.
"""

import sys
import os
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
from pathlib import Path

# FastAPI and core dependencies
from fastapi import FastAPI, Request, HTTPException, status
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


class BlazeAIApplication:
    """
    Main application class for Enhanced Blaze AI.
    
    This class encapsulates the application lifecycle and provides
    a clean interface for managing the enhanced features.
    """
    
    def __init__(self, config: AppConfig):
        """Initialize the Blaze AI application."""
        self.config = config
        self.app = None
        self.enhanced_features = {}
        self.health_checker = None
        
    def create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        # Create FastAPI instance with metadata
        self.app = FastAPI(
            title="Enhanced Blaze AI",
            description="Enterprise-Grade AI Platform with Advanced Features",
            version="2.1.0",
            docs_url=None,  # Custom docs endpoint
            redoc_url="/redoc",
            openapi_url="/openapi.json"
        )
        
        # Setup application lifecycle
        self.app.router.lifespan_context = self._create_lifespan_context()
        
        # Configure middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
        
        # Setup exception handlers
        self._setup_exception_handlers()
        
        # Setup enhanced features if available
        if ENHANCED_FEATURES_AVAILABLE:
            self._setup_enhanced_features()
        
        return self.app
    
    def _create_lifespan_context(self):
        """Create application lifespan context manager."""
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            logger.info("🚀 Starting Enhanced Blaze AI...")
            await self._startup()
            yield
            # Shutdown
            logger.info("🛑 Shutting down Enhanced Blaze AI...")
            await self._shutdown()
        
        return lifespan
    
    async def _startup(self):
        """Application startup tasks."""
        try:
            # Initialize enhanced features
            if ENHANCED_FEATURES_AVAILABLE:
                await self._initialize_enhanced_features()
            
            # Initialize health checker
            self.health_checker = SystemHealth() if ENHANCED_FEATURES_AVAILABLE else None
            
            logger.info("✅ Enhanced Blaze AI startup completed")
            
        except Exception as e:
            logger.error(f"❌ Startup failed: {e}")
            raise
    
    async def _shutdown(self):
        """Application shutdown tasks."""
        try:
            # Cleanup enhanced features
            if ENHANCED_FEATURES_AVAILABLE:
                await self._cleanup_enhanced_features()
            
            logger.info("✅ Enhanced Blaze AI shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Shutdown failed: {e}")
    
    def _setup_middleware(self):
        """Configure application middleware."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors.allow_origins,
            allow_credentials=self.config.cors.allow_credentials,
            allow_methods=self.config.cors.allow_methods,
            allow_headers=self.config.cors.allow_headers,
        )
        
        # Enhanced middleware if available
        if ENHANCED_FEATURES_AVAILABLE:
            middleware_stack = create_middleware_stack(self.config)
            for middleware in middleware_stack:
                self.app.add_middleware(middleware)
    
    def _setup_routes(self):
        """Configure application routes."""
        # API routes
        api_router = create_api_router(self.config)
        self.app.include_router(api_router, prefix="/api/v2")
        
        # Health check routes
        self._setup_health_routes()
        
        # Enhanced feature routes
        if ENHANCED_FEATURES_AVAILABLE:
            self._setup_enhanced_routes()
        
        # Documentation routes
        self._setup_docs_routes()
    
    def _setup_health_routes(self):
        """Setup health check endpoints."""
        @self.app.get("/health")
        async def health_check():
            """Basic health check endpoint."""
            return {
                "status": "healthy",
                "service": "Enhanced Blaze AI",
                "version": "2.1.0",
                "timestamp": asyncio.get_event_loop().time()
            }
        
        @self.app.get("/health/detailed")
        async def detailed_health_check():
            """Detailed health check with system status."""
            if self.health_checker:
                return await self.health_checker.get_detailed_health()
            else:
                return {
                    "status": "healthy",
                    "service": "Enhanced Blaze AI",
                    "version": "2.1.0",
                    "enhanced_features": False,
                    "timestamp": asyncio.get_event_loop().time()
                }
    
    def _setup_enhanced_routes(self):
        """Setup enhanced feature endpoints."""
        if not ENHANCED_FEATURES_AVAILABLE:
            return
        
        # Metrics endpoints
        @self.app.get("/metrics")
        async def get_metrics():
            """Get application metrics."""
            try:
                if 'monitor' in self.enhanced_features:
                    return await self.enhanced_features['monitor'].get_metrics()
                else:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Performance monitoring not available"
                    )
            except Exception as e:
                logger.error(f"Error getting metrics: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to retrieve metrics"
                )
        
        @self.app.get("/metrics/prometheus")
        async def get_prometheus_metrics():
            """Get Prometheus format metrics."""
            try:
                if 'monitor' in self.enhanced_features:
                    return await self.enhanced_features['monitor'].get_prometheus_metrics()
                else:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Performance monitoring not available"
                    )
            except Exception as e:
                logger.error(f"Error getting Prometheus metrics: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to retrieve Prometheus metrics"
                )
        
        # Security endpoints
        @self.app.get("/security/status")
        async def get_security_status():
            """Get security status and threats."""
            try:
                if 'security' in self.enhanced_features:
                    return await self.enhanced_features['security'].get_status()
                else:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Security monitoring not available"
                    )
            except Exception as e:
                logger.error(f"Error getting security status: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to retrieve security status"
                )
        
        # Error monitoring endpoints
        @self.app.get("/errors/summary")
        async def get_error_summary():
            """Get error summary and statistics."""
            try:
                if 'error_handler' in self.enhanced_features:
                    return await self.enhanced_features['error_handler'].get_summary()
                else:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Error monitoring not available"
                    )
            except Exception as e:
                logger.error(f"Error getting error summary: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to retrieve error summary"
                )
    
    def _setup_docs_routes(self):
        """Setup documentation endpoints."""
        @self.app.get("/docs", include_in_schema=False)
        async def custom_swagger_ui_html():
            """Custom Swagger UI with enhanced styling."""
            return get_swagger_ui_html(
                openapi_url=self.app.openapi_url,
                title=f"{self.app.title} - API Documentation",
                swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
                swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
            )
    
    def _setup_exception_handlers(self):
        """Configure exception handlers."""
        @self.app.exception_handler(BlazeAIError)
        async def blaze_ai_exception_handler(request: Request, exc: BlazeAIError):
            """Handle BlazeAI specific exceptions."""
            logger.error(f"BlazeAI Error: {exc.detail}")
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": "BlazeAI Error",
                    "detail": exc.detail,
                    "timestamp": asyncio.get_event_loop().time()
                }
            )
        
        @self.app.exception_handler(ServiceUnavailableError)
        async def service_unavailable_handler(request: Request, exc: ServiceUnavailableError):
            """Handle service unavailable exceptions."""
            logger.error(f"Service Unavailable: {exc.detail}")
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": "Service Unavailable",
                    "detail": exc.detail,
                    "timestamp": asyncio.get_event_loop().time()
                }
            )
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """Handle general exceptions."""
            logger.error(f"Unexpected error: {exc}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal Server Error",
                    "detail": "An unexpected error occurred",
                    "timestamp": asyncio.get_event_loop().time()
                }
            )
    
    async def _initialize_enhanced_features(self):
        """Initialize enhanced features."""
        try:
            # Security middleware
            if 'security' not in self.enhanced_features:
                security_config = SecurityConfig.from_app_config(self.config)
                self.enhanced_features['security'] = SecurityMiddleware(security_config)
            
            # Performance monitoring
            if 'monitor' not in self.enhanced_features:
                self.enhanced_features['monitor'] = PerformanceMonitor()
                await self.enhanced_features['monitor'].start()
            
            # Rate limiting
            if 'rate_limiter' not in self.enhanced_features:
                rate_limit_config = RateLimitConfig.from_app_config(self.config)
                self.enhanced_features['rate_limiter'] = RateLimiter(rate_limit_config)
            
            # Error handling
            if 'error_handler' not in self.enhanced_features:
                self.enhanced_features['error_handler'] = ErrorHandler()
            
            # Circuit breaker
            if 'circuit_breaker' not in self.enhanced_features:
                self.enhanced_features['circuit_breaker'] = CircuitBreaker()
            
            logger.info("✅ Enhanced features initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize enhanced features: {e}")
            # Continue without enhanced features
            self.enhanced_features = {}
    
    async def _cleanup_enhanced_features(self):
        """Cleanup enhanced features."""
        try:
            # Stop performance monitoring
            if 'monitor' in self.enhanced_features:
                await self.enhanced_features['monitor'].stop()
            
            # Cleanup other features
            for feature_name, feature in self.enhanced_features.items():
                if hasattr(feature, 'cleanup'):
                    await feature.cleanup()
            
            logger.info("✅ Enhanced features cleaned up successfully")
            
        except Exception as e:
            logger.error(f"❌ Error during feature cleanup: {e}")


def create_app(config: Optional[AppConfig] = None) -> FastAPI:
    """
    Factory function to create the Blaze AI application.
    
    Args:
        config: Application configuration. If None, loads from file.
    
    Returns:
        Configured FastAPI application instance.
    """
    if config is None:
        config = load_config()
    
    app_instance = BlazeAIApplication(config)
    return app_instance.create_app()


def main():
    """Main entry point for the application."""
    # Setup logging
    setup_logging()
    
    # Load configuration
    try:
        config = load_config()
        logger.info("✅ Configuration loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        sys.exit(1)
    
    # Create application
    app = create_app(config)
    
    # Determine server mode
    dev_mode = "--dev" in sys.argv
    
    if dev_mode:
        # Development mode with hot reload
        logger.info("🚀 Starting Enhanced Blaze AI in development mode...")
        uvicorn.run(
            "main:app",
            host=config.server.host,
            port=config.server.port,
            reload=True,
            log_level="debug"
        )
    else:
        # Production mode
        logger.info("🚀 Starting Enhanced Blaze AI in production mode...")
        uvicorn.run(
            app,
            host=config.server.host,
            port=config.server.port,
            workers=config.server.workers,
            log_level="info"
        )


if __name__ == "__main__":
    main()
