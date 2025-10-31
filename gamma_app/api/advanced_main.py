"""
Gamma App - Advanced Main API Application
Ultra-advanced FastAPI application with enterprise-grade features
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse
import redis
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import structlog
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.redis import RedisIntegration

# Import our enhanced services
from ..services.performance_service import performance_service
from ..services.security_service import security_service
from ..engines.ai_models_engine import AIModelsEngine
from ..services.cache_service import cache_service
from ..utils.config import get_settings
from ..utils.auth import verify_token, create_access_token
from ..models import ErrorResponse, SuccessResponse

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
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Gauge('http_active_connections', 'Active HTTP connections')
MODEL_INFERENCE_TIME = Histogram('model_inference_seconds', 'Model inference time', ['model_name'])
CACHE_HIT_RATE = Gauge('cache_hit_rate', 'Cache hit rate')
SECURITY_EVENTS = Counter('security_events_total', 'Security events', ['event_type', 'severity'])

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Global services
content_generator = None
collaboration_service = None
analytics_service = None
ai_models_engine = None
settings = None

class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring"""
    
    async def dispatch(self, request: StarletteRequest, call_next):
        start_time = time.time()
        
        # Track active connections
        ACTIVE_CONNECTIONS.inc()
        
        try:
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            REQUEST_DURATION.observe(duration)
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            # Add performance headers
            response.headers["X-Response-Time"] = str(duration)
            response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", "unknown")
            
            return response
            
        except Exception as e:
            # Record error metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=500
            ).inc()
            raise
        finally:
            ACTIVE_CONNECTIONS.dec()

class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for security monitoring"""
    
    async def dispatch(self, request: StarletteRequest, call_next):
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Security analysis
        request_data = {
            "method": request.method,
            "path": request.url.path,
            "headers": dict(request.headers),
            "query_params": dict(request.query_params)
        }
        
        # Analyze request security
        security_analysis = await security_service.analyze_request_security(
            request_data, client_ip
        )
        
        # Log security events
        if not security_analysis['is_safe']:
            for threat in security_analysis['threats_detected']:
                SECURITY_EVENTS.labels(
                    event_type=threat.split(':')[0] if ':' in threat else threat,
                    severity="high" if security_analysis['risk_score'] > 0.5 else "medium"
                ).inc()
        
        # Block high-risk requests
        if security_analysis['risk_score'] > 0.8:
            logger.warning("Blocking high-risk request", 
                         client_ip=client_ip, 
                         risk_score=security_analysis['risk_score'],
                         threats=security_analysis['threats_detected'])
            return JSONResponse(
                status_code=403,
                content={"error": "Request blocked due to security risk"}
            )
        
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Security-Score"] = str(security_analysis['risk_score'])
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        return response

class CacheMiddleware(BaseHTTPMiddleware):
    """Middleware for intelligent caching"""
    
    async def dispatch(self, request: StarletteRequest, call_next):
        # Skip caching for certain requests
        if request.method in ["POST", "PUT", "DELETE", "PATCH"]:
            return await call_next(request)
        
        # Generate cache key
        cache_key = f"api:{request.method}:{request.url.path}:{hash(str(request.query_params))}"
        
        # Try to get from cache
        cached_response = await cache_service.get(cache_key, namespace="api")
        if cached_response:
            CACHE_HIT_RATE.set(1.0)
            return JSONResponse(content=cached_response)
        
        # Process request
        response = await call_next(request)
        
        # Cache successful GET requests
        if request.method == "GET" and response.status_code == 200:
            try:
                response_body = response.body
                if response_body:
                    await cache_service.set(
                        cache_key, 
                        response_body.decode(), 
                        ttl=300,  # 5 minutes
                        namespace="api"
                    )
            except Exception as e:
                logger.error("Cache set error", error=str(e))
        
        CACHE_HIT_RATE.set(0.0)
        return response

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Advanced application lifespan manager"""
    global content_generator, collaboration_service, analytics_service, ai_models_engine, settings
    
    # Startup
    logger.info("Starting Gamma App Advanced API...")
    
    try:
        # Initialize settings
        settings = get_settings()
        
        # Initialize Sentry for error tracking
        if settings.sentry_dsn:
            sentry_sdk.init(
                dsn=settings.sentry_dsn,
                integrations=[
                    FastApiIntegration(auto_enabling_instrumentations=True),
                    RedisIntegration(),
                ],
                traces_sample_rate=0.1,
                environment=settings.environment
            )
        
        # Initialize services
        from ..core.content_generator import ContentGenerator
        from ..services.collaboration_service import CollaborationService
        from ..services.analytics_service import AnalyticsService
        
        content_generator = ContentGenerator({
            'openai_api_key': settings.openai_api_key,
            'anthropic_api_key': settings.anthropic_api_key,
            'openai_model': settings.openai_model,
            'anthropic_model': settings.anthropic_model
        })
        
        collaboration_service = CollaborationService()
        analytics_service = AnalyticsService()
        ai_models_engine = AIModelsEngine()
        
        # Start performance monitoring
        await performance_service.start_monitoring(interval=30)
        
        # Warm up cache
        await cache_service.warm_cache({
            "health_check": lambda: {"status": "healthy", "timestamp": datetime.now().isoformat()},
            "api_info": lambda: {"version": "2.0.0", "features": ["advanced", "enterprise"]}
        })
        
        logger.info("Gamma App Advanced API started successfully")
        
    except Exception as e:
        logger.error("Failed to start API", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Gamma App Advanced API...")
    
    try:
        # Stop performance monitoring
        await performance_service.stop_monitoring()
        
        # Close services
        await cache_service.close()
        await security_service.close()
        await performance_service.close()
        
        logger.info("Gamma App Advanced API shutdown complete")
        
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))

# Create FastAPI application with advanced configuration
app = FastAPI(
    title="Gamma App Advanced API",
    description="Ultra-Advanced AI-Powered Content Generation System with Enterprise Features",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    default_response_class=JSONResponse
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add middleware in order
app.add_middleware(SecurityMiddleware)
app.add_middleware(PerformanceMiddleware)
app.add_middleware(CacheMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins if settings else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts if settings else ["*"]
)

# Security
security = HTTPBearer()

# Include routers with advanced features
from .routes import content_router, collaboration_router, export_router, analytics_router
from .advanced_routes import (
    performance_router, security_router, ai_models_router, 
    monitoring_router, admin_router, webhook_router
)

app.include_router(content_router, prefix="/api/v2/content", tags=["content"])
app.include_router(collaboration_router, prefix="/api/v2/collaboration", tags=["collaboration"])
app.include_router(export_router, prefix="/api/v2/export", tags=["export"])
app.include_router(analytics_router, prefix="/api/v2/analytics", tags=["analytics"])
app.include_router(performance_router, prefix="/api/v2/performance", tags=["performance"])
app.include_router(security_router, prefix="/api/v2/security", tags=["security"])
app.include_router(ai_models_router, prefix="/api/v2/ai-models", tags=["ai-models"])
app.include_router(monitoring_router, prefix="/api/v2/monitoring", tags=["monitoring"])
app.include_router(admin_router, prefix="/api/v2/admin", tags=["admin"])
app.include_router(webhook_router, prefix="/api/v2/webhooks", tags=["webhooks"])

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Advanced root endpoint"""
    return {
        "message": "Welcome to Gamma App Advanced API",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "advanced-performance-monitoring",
            "ml-powered-security",
            "intelligent-caching",
            "real-time-analytics",
            "enterprise-grade-ai",
            "auto-scaling",
            "threat-detection",
            "performance-optimization"
        ],
        "docs": "/docs",
        "redoc": "/redoc",
        "metrics": "/metrics",
        "health": "/health"
    }

@app.get("/health", response_model=Dict[str, Any])
@limiter.limit("10/minute")
async def health_check(request: Request):
    """Advanced health check endpoint"""
    try:
        # Get cached health data
        health_data = await cache_service.get("health_check", namespace="api")
        if health_data:
            return health_data
        
        # Perform comprehensive health checks
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "uptime": time.time() - start_time if 'start_time' in globals() else 0,
            "services": {
                "content_generator": "healthy" if content_generator else "unavailable",
                "collaboration_service": "healthy" if collaboration_service else "unavailable",
                "analytics_service": "healthy" if analytics_service else "unavailable",
                "ai_models_engine": "healthy" if ai_models_engine else "unavailable",
                "cache_service": "healthy",
                "security_service": "healthy",
                "performance_service": "healthy"
            },
            "performance": {
                "active_connections": ACTIVE_CONNECTIONS._value._value,
                "cache_hit_rate": CACHE_HIT_RATE._value._value,
                "total_requests": sum(REQUEST_COUNT._metrics.values())
            },
            "security": {
                "threat_level": "low",
                "active_blocks": 0,
                "security_events": sum(SECURITY_EVENTS._metrics.values())
            }
        }
        
        # Cache health data for 30 seconds
        await cache_service.set("health_check", health_status, ttl=30, namespace="api")
        
        return health_status
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        generate_latest(),
        media_type="text/plain"
    )

@app.get("/api/v2/status", response_model=Dict[str, Any])
async def system_status():
    """Comprehensive system status"""
    try:
        # Get performance metrics
        performance_metrics = await performance_service.get_performance_dashboard()
        
        # Get security analytics
        security_analytics = await security_service.get_security_analytics()
        
        # Get AI models status
        ai_models_status = await ai_models_engine.get_memory_usage() if ai_models_engine else {}
        
        return {
            "system": {
                "status": "operational",
                "version": "2.0.0",
                "uptime": time.time() - start_time if 'start_time' in globals() else 0,
                "environment": settings.environment if settings else "development"
            },
            "performance": performance_metrics,
            "security": security_analytics,
            "ai_models": ai_models_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("System status check failed", error=str(e))
        raise HTTPException(status_code=500, detail="System status unavailable")

@app.post("/api/v2/auth/login", response_model=Dict[str, Any])
@limiter.limit("5/minute")
async def login(request: Request, credentials: Dict[str, str]):
    """Advanced authentication endpoint"""
    try:
        # Validate credentials
        username = credentials.get("username")
        password = credentials.get("password")
        
        if not username or not password:
            raise HTTPException(status_code=400, detail="Username and password required")
        
        # Check rate limiting for login attempts
        client_ip = request.client.host if request.client else "unknown"
        is_allowed, attempts = await security_service.check_login_attempts(client_ip)
        
        if not is_allowed:
            raise HTTPException(
                status_code=429, 
                detail=f"Too many login attempts. Try again in {settings.login_lockout_duration // 60} minutes"
            )
        
        # Authenticate user (simplified - would integrate with user service)
        if username == "admin" and password == "admin":  # Demo credentials
            # Record successful login
            await security_service.record_login_attempt(client_ip, success=True)
            
            # Create access token
            access_token = security_service.generate_jwt_token({
                "sub": username,
                "user_id": "1",
                "role": "admin"
            })
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": settings.session_timeout,
                "user": {
                    "id": "1",
                    "username": username,
                    "role": "admin"
                }
            }
        else:
            # Record failed login
            await security_service.record_login_attempt(client_ip, success=False)
            raise HTTPException(status_code=401, detail="Invalid credentials")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login failed", error=str(e))
        raise HTTPException(status_code=500, detail="Authentication service unavailable")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Advanced HTTP exception handler"""
    # Log the error
    logger.error("HTTP exception", 
                status_code=exc.status_code, 
                detail=exc.detail,
                path=request.url.path,
                method=request.method)
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            status_code=exc.status_code,
            timestamp=datetime.now().isoformat(),
            path=request.url.path,
            method=request.method
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Advanced general exception handler"""
    # Log the error with Sentry
    logger.error("Unhandled exception", 
                error=str(exc), 
                path=request.url.path,
                method=request.method,
                exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            status_code=500,
            timestamp=datetime.now().isoformat(),
            path=request.url.path,
            method=request.method
        ).dict()
    )

# Advanced dependencies
async def get_content_generator():
    """Get content generator instance with health check"""
    if not content_generator:
        raise HTTPException(
            status_code=503,
            detail="Content generator not available"
        )
    return content_generator

async def get_ai_models_engine():
    """Get AI models engine instance"""
    if not ai_models_engine:
        raise HTTPException(
            status_code=503,
            detail="AI models engine not available"
        )
    return ai_models_engine

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Advanced authentication dependency"""
    try:
        # Verify token
        is_valid, user_data = security_service.verify_jwt_token(credentials.credentials)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user_data
    except Exception as e:
        logger.error("Authentication failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def require_admin_role(current_user: dict = Depends(get_current_user)):
    """Require admin role"""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required"
        )
    return current_user

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Gamma App Advanced API",
        version="2.0.0",
        description="Ultra-Advanced AI-Powered Content Generation System",
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    
    # Add security to all endpoints
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            if method not in ["get", "head", "options"]:
                openapi_schema["paths"][path][method]["security"] = [{"BearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Set start time for uptime calculation
start_time = time.time()

if __name__ == "__main__":
    uvicorn.run(
        "advanced_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
        use_colors=True
    )
















