"""
Advanced Middleware for Robot Movement AI
==========================================

Implements:
- OpenTelemetry distributed tracing
- Rate limiting
- Circuit breakers
- Request/response logging
- Security headers
- Request ID tracking
"""

import time
import uuid
import logging
from typing import Callable
from fastapi import Request, Response, HTTPException, status
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import BaseHTTPMiddleware as StarletteBaseHTTPMiddleware
from starlette.types import ASGIApp
import redis.asyncio as aioredis
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from circuitbreaker import circuit
import os

logger = logging.getLogger(__name__)

# Initialize OpenTelemetry
def setup_opentelemetry(service_name: str = "robot-movement-ai"):
    """Setup OpenTelemetry tracing."""
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
    })
    
    # Use OTLP endpoint from environment or default to localhost
    otlp_endpoint = os.getenv("OTLP_ENDPOINT", "http://localhost:4317")
    
    trace.set_tracer_provider(TracerProvider(resource=resource))
    tracer = trace.get_tracer(__name__)
    
    otlp_exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        insecure=os.getenv("OTLP_INSECURE", "true").lower() == "true"
    )
    
    span_processor = BatchSpanProcessor(otlp_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Instrument FastAPI and HTTP clients
    FastAPIInstrumentor.instrument()
    HTTPXClientInstrumentor.instrument()
    
    return tracer

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)

# Circuit Breaker Configuration
CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = int(os.getenv("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "60"))
CIRCUIT_BREAKER_EXPECTED_EXCEPTION = Exception


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add request ID to all requests for tracing."""
    
    async def dispatch(self, request: Request, call_next: Callable):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next: Callable):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Structured logging for all requests."""
    
    async def dispatch(self, request: Request, call_next: Callable):
        start_time = time.time()
        request_id = getattr(request.state, "request_id", "unknown")
        
        # Log request
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
            }
        )
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log successful response
            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "process_time": process_time,
                }
            )
            
            response.headers["X-Process-Time"] = str(process_time)
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            # Log error
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "process_time": process_time,
                },
                exc_info=True
            )
            raise


class CircuitBreakerMiddleware(BaseHTTPMiddleware):
    """Circuit breaker middleware for external service calls."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.failure_count = {}
        self.last_failure_time = {}
        self.circuit_open = {}
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Only apply circuit breaker to external service calls
        if request.url.path.startswith("/api/v1/external/"):
            service_name = request.url.path.split("/")[3]
            
            if self._is_circuit_open(service_name):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Circuit breaker is open for {service_name}"
                )
            
            try:
                response = await call_next(request)
                
                # Reset failure count on success
                if service_name in self.failure_count:
                    self.failure_count[service_name] = 0
                
                return response
                
            except Exception as e:
                self._record_failure(service_name)
                raise
        
        return await call_next(request)
    
    def _is_circuit_open(self, service_name: str) -> bool:
        """Check if circuit breaker is open."""
        if service_name not in self.circuit_open:
            return False
        
        if not self.circuit_open[service_name]:
            return False
        
        # Check if recovery timeout has passed
        if service_name in self.last_failure_time:
            recovery_timeout = CIRCUIT_BREAKER_RECOVERY_TIMEOUT
            if time.time() - self.last_failure_time[service_name] > recovery_timeout:
                # Try to close circuit
                self.circuit_open[service_name] = False
                self.failure_count[service_name] = 0
                return False
        
        return True
    
    def _record_failure(self, service_name: str):
        """Record a failure for circuit breaker."""
        if service_name not in self.failure_count:
            self.failure_count[service_name] = 0
        
        self.failure_count[service_name] += 1
        self.last_failure_time[service_name] = time.time()
        
        if self.failure_count[service_name] >= CIRCUIT_BREAKER_FAILURE_THRESHOLD:
            self.circuit_open[service_name] = True
            logger.warning(
                f"Circuit breaker opened for {service_name}",
                extra={"service_name": service_name, "failure_count": self.failure_count[service_name]}
            )


class RedisCacheMiddleware(BaseHTTPMiddleware):
    """Redis caching middleware for GET requests."""
    
    def __init__(self, app: ASGIApp, redis_url: str = None):
        super().__init__(app)
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_client = None
        self.cache_ttl = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes default
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Skip caching for certain paths
        skip_paths = ["/health", "/metrics", "/docs", "/openapi.json"]
        if any(request.url.path.startswith(path) for path in skip_paths):
            return await call_next(request)
        
        # Initialize Redis client if needed
        if self.redis_client is None:
            try:
                self.redis_client = await aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Caching disabled.")
                return await call_next(request)
        
        # Generate cache key
        cache_key = f"cache:{request.url.path}:{str(request.query_params)}"
        
        try:
            # Try to get from cache
            cached_response = await self.redis_client.get(cache_key)
            if cached_response:
                logger.debug(f"Cache hit for {cache_key}")
                return Response(
                    content=cached_response,
                    media_type="application/json",
                    headers={"X-Cache": "HIT"}
                )
            
            # Cache miss - process request
            response = await call_next(request)
            
            # Cache successful responses
            if response.status_code == 200:
                try:
                    body = b""
                    async for chunk in response.body_iterator:
                        body += chunk
                    
                    await self.redis_client.setex(
                        cache_key,
                        self.cache_ttl,
                        body.decode("utf-8")
                    )
                    
                    logger.debug(f"Cached response for {cache_key}")
                    
                    return Response(
                        content=body,
                        status_code=response.status_code,
                        headers={**dict(response.headers), "X-Cache": "MISS"}
                    )
                except Exception as e:
                    logger.warning(f"Failed to cache response: {e}")
                    return response
            
            return response
            
        except Exception as e:
            logger.error(f"Cache error: {e}")
            return await call_next(request)


def setup_advanced_middleware(app, config):
    """Setup all advanced middleware."""
    # OpenTelemetry
    if os.getenv("ENABLE_TRACING", "true").lower() == "true":
        setup_opentelemetry("robot-movement-ai")
    
    # Rate limiting
    if os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true":
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        app.add_middleware(SlowAPIMiddleware)
    
    # Request ID
    app.add_middleware(RequestIDMiddleware)
    
    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Structured logging
    app.add_middleware(StructuredLoggingMiddleware)
    
    # Circuit breaker
    if os.getenv("ENABLE_CIRCUIT_BREAKER", "true").lower() == "true":
        app.add_middleware(CircuitBreakerMiddleware)
    
    # Redis caching
    if os.getenv("ENABLE_REDIS_CACHE", "true").lower() == "true":
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            app.add_middleware(RedisCacheMiddleware, redis_url=redis_url)
    
    return app















