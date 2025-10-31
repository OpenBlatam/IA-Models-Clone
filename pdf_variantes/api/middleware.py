"""
PDF Variantes API - Middleware Configuration
Request/Response middleware for security, logging, and performance
"""

import asyncio
import logging
import uuid
import time
from typing import Dict, Any
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from ..utils.config import get_settings
from ..utils.response_helpers import get_request_id
from ..utils.structured_logging import set_request_id, set_user_id, get_logger

logger = get_logger(__name__)


# Rate limiting storage (in-memory, per-IP)
_rate_limit_bucket: Dict[str, list] = {}


def get_cors_config() -> Dict[str, Any]:
    """Get CORS configuration optimized for frontend"""
    import os
    settings = get_settings()
    cors_origins = getattr(settings, "CORS_ORIGINS", None)
    
    if cors_origins is None:
        # Defaults para desarrollo: localhost y puertos comunes
        cors_origins = [
            "http://localhost:3000",      # Next.js default
            "http://localhost:3001",      # Next.js alternate
            "http://localhost:5173",      # Vite default
            "http://localhost:4200",      # Angular default
            "http://localhost:8080",      # Vue CLI default
            "http://localhost:5000",      # Common dev port
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:4200",
            "http://127.0.0.1:8080",
            "http://127.0.0.1:5000",
        ]
    
    # In development, optionally allow all origins (not recommended for production)
    environment = os.getenv("ENVIRONMENT", "development").lower()
    if environment == "development" and os.getenv("ALLOW_ALL_ORIGINS", "false").lower() == "true":
        cors_origins = ["*"]
    
    return {
        "allow_origins": cors_origins if "*" not in cors_origins else ["*"],
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
        "allow_headers": [
            "Authorization",
            "Content-Type",
            "Accept",
            "Origin",
            "X-Requested-With",
            "X-User-Id",
            "User-Id",
            "X-API-Key",
            "*"
        ],
        "expose_headers": [
            "Content-Disposition",
            "Content-Type",
            "X-Total-Count",
            "X-Request-ID",
            "X-Response-Time",
            "X-Process-Time",
            "Authorization"
        ],
        "max_age": 3600,  # Cache preflight requests for 1 hour
    }


def setup_cors_middleware(app) -> None:
    """Setup CORS middleware"""
    cors_config = get_cors_config()
    app.add_middleware(
        CORSMiddleware,
        **cors_config
    )


def setup_trusted_host_middleware(app) -> None:
    """Setup trusted host middleware"""
    settings = get_settings()
    allowed_hosts = getattr(settings, "ALLOWED_HOSTS", ["localhost", "127.0.0.1", "*.yourdomain.com"])
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=allowed_hosts
    )


async def rate_limit_middleware(request: Request, call_next):
    """Basic per-IP rate limiting middleware (LEGACY - use OptimizedRateLimitMiddleware)
    
    Defaults: 100 requests per 60 seconds. Configurable via settings.
    """
    try:
        # Skip rate limiting for health checks (performance optimization)
        if request.url.path in ["/health", "/api/v1/health", "/"]:
            return await call_next(request)
        
        settings = get_settings()
        max_req_per_min = getattr(settings, "RATE_LIMIT_REQUESTS_PER_MINUTE", 100)
        window_seconds = 60

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()  # Use time.time() instead of event loop time for performance

        bucket = _rate_limit_bucket.setdefault(client_ip, [])
        # Drop old timestamps outside the window (optimized with list comprehension)
        cutoff = now - window_seconds
        bucket[:] = [ts for ts in bucket if ts > cutoff]

        if len(bucket) >= max_req_per_min:
            request_id = get_request_id(request)
            error_response = create_rate_limit_response(
                retry_after=60,
                request_id=request_id
            )
            return JSONResponse(
                status_code=429,
                content=error_response,
                headers={"Retry-After": "60"}
            )

        bucket.append(now)
    except Exception as e:
        # Fail open on any rate limit error (don't block requests)
        logger.warning(f"Rate limit middleware error: {e}")

    return await call_next(request)


async def request_logging_middleware(request: Request, call_next):
    """Log all requests with timing information and Request ID tracking"""
    from ..utils.structured_logging import set_request_context, log_performance, get_request_id as get_context_request_id
    from uuid import uuid4
    
    # Generate and set request ID if not already set
    request_id = get_request_id(request)
    if not request_id:
        request_id = str(uuid4())
        request.state.request_id = request_id
    set_request_id(request_id)
    start_time = time.perf_counter()  # Use perf_counter for better precision
    
    # Set request context for structured logging
    user_id = request.headers.get("X-User-Id") or request.headers.get("User-Id")
    correlation_id = request.headers.get("X-Correlation-ID")
    set_request_context(
        request_id=request_id,
        user_id=user_id,
        correlation_id=correlation_id
    )
    
    # Store start time in request state
    request.state.start_time = start_time
    request.state.request_id = request_id
    
    # Log request with structured context
    logger.info(
        f"Request: {request.method} {request.url.path} from {request.client.host if request.client else 'unknown'}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
        }
    )
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate process time
        process_time = time.perf_counter() - start_time
        
        # Add request ID and performance headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.3f}"
        response.headers["X-Response-Time"] = f"{process_time:.3f}"
        
        # Log performance metrics
        log_performance(
            operation=f"{request.method} {request.url.path}",
            duration=process_time,
            logger=logger,
            status_code=response.status_code
        )
        
        # Log response
        if process_time > 1.0:
            logger.warning(
                f"Slow request: {request.method} {request.url.path} - {process_time:.3f}s",
                extra={"status_code": response.status_code, "duration": process_time}
            )
        
        return response
        
    except Exception as e:
        # Log error with request ID
        process_time = time.perf_counter() - start_time
        logger.error(
            f"Request failed: {request.method} {request.url.path} - {str(e)}",
            exc_info=True,
            extra={
                "duration": process_time,
                "error_type": type(e).__name__
            }
        )
        raise
    finally:
        # Clear request context
        from ..utils.structured_logging import clear_request_context
        clear_request_context()


async def performance_monitoring_middleware(request: Request, call_next, services: Dict[str, Any] = None):
    """Monitor request performance using performance service"""
    # Get services from parameter or dependency system
    if not services:
        try:
            from .dependencies import get_services as _get_services
            services = _get_services()
        except:
            services = {}
    
    performance_service = services.get("performance_service") if services else None
    
    if performance_service and hasattr(performance_service, "start_request_monitoring"):
        try:
            await performance_service.start_request_monitoring(request)
        except Exception as e:
            logger.warning(f"Performance monitoring start error: {e}")
    
    response = await call_next(request)
    
    if performance_service and hasattr(performance_service, "end_request_monitoring"):
        try:
            await performance_service.end_request_monitoring(request, response)
        except Exception as e:
            logger.warning(f"Performance monitoring end error: {e}")
    
    return response


async def security_middleware(request: Request, call_next, services: Dict[str, Any] = None):
    """Apply security checks using security service"""
    # Get services from parameter or dependency system
    if not services:
        try:
            from .dependencies import get_services as _get_services
            services = _get_services()
        except:
            services = {}
    
    security_service = services.get("security_service") if services else None
    
    if security_service:
        try:
            if hasattr(security_service, "check_request"):
                security_result = await security_service.check_request(request)
                if security_result and hasattr(security_result, "is_safe") and not security_result.is_safe:
                    return JSONResponse(
                        status_code=403,
                        content={
                            "error": "Request blocked by security policy",
                            "reason": getattr(security_result, "reason", "Unknown reason")
                        }
                    )
        except Exception as e:
            logger.warning(f"Security check error: {e}")
    
    response = await call_next(request)
    
    if security_service:
        try:
            if hasattr(security_service, "log_request"):
                await security_service.log_request(request, response)
        except Exception as e:
            logger.warning(f"Security logging error: {e}")
    
    return response

