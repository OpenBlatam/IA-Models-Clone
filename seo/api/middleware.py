from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import time
import json
import hashlib
import asyncio
from typing import Dict, Any, Optional, Callable
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import prometheus_client
from loguru import logger
import orjson
from tenacity import retry, stop_after_attempt, wait_exponential
            import psutil
from typing import Any, List, Dict, Optional
import logging
"""
Production middleware for Ultra-Optimized SEO Service.
Includes security, monitoring, rate limiting, and logging middleware.
"""


# Prometheus metrics
REQUEST_COUNT = prometheus_client.Counter(
    'seo_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = prometheus_client.Histogram(
    'seo_request_duration_seconds',
    'Request duration',
    ['method', 'endpoint']
)

ACTIVE_REQUESTS = prometheus_client.Gauge(
    'seo_active_requests',
    'Active requests'
)

ERROR_COUNT = prometheus_client.Counter(
    'seo_errors_total',
    'Total errors',
    ['type']
)

CACHE_HIT_RATE = prometheus_client.Gauge(
    'seo_cache_hit_rate',
    'Cache hit rate'
)

MEMORY_USAGE = prometheus_client.Gauge(
    'seo_memory_usage_bytes',
    'Memory usage in bytes'
)

CPU_USAGE = prometheus_client.Gauge(
    'seo_cpu_usage_percent',
    'CPU usage percentage'
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request logging."""
    
    def __init__(self, app: ASGIApp, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(app)
        self.config = config or {}
        self.log_requests = self.config.get('log_requests', True)
        self.log_responses = self.config.get('log_responses', True)
        self.log_headers = self.config.get('log_headers', False)
        self.log_body = self.config.get('log_body', False)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        
        # Log request
        if self.log_requests:
            await self._log_request(request)
        
        # Process request
        try:
            response = await call_next(request)
            processing_time = time.perf_counter() - start_time
            
            # Log response
            if self.log_responses:
                await self._log_response(request, response, processing_time)
            
            return response
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            await self._log_error(request, e, processing_time)
            raise
    
    async def _log_request(self, request: Request):
        """Log incoming request."""
        log_data = {
            'type': 'request',
            'method': request.method,
            'url': str(request.url),
            'path': request.url.path,
            'query_params': dict(request.query_params),
            'client_ip': self._get_client_ip(request),
            'user_agent': request.headers.get('user-agent'),
            'timestamp': time.time()
        }
        
        if self.log_headers:
            log_data['headers'] = dict(request.headers)
        
        if self.log_body and request.method in ['POST', 'PUT', 'PATCH']:
            try:
                body = await request.body()
                if body:
                    log_data['body_size'] = len(body)
                    if len(body) < 1024:  # Only log small bodies
                        log_data['body'] = body.decode('utf-8')
            except Exception:
                pass
        
        logger.info(f"Request: {request.method} {request.url.path}", extra=log_data)
    
    async def _log_response(self, request: Request, response: Response, processing_time: float):
        """Log response."""
        log_data = {
            'type': 'response',
            'method': request.method,
            'path': request.url.path,
            'status_code': response.status_code,
            'processing_time': processing_time,
            'content_length': response.headers.get('content-length'),
            'timestamp': time.time()
        }
        
        logger.info(
            f"Response: {request.method} {request.url.path} - "
            f"{response.status_code} - {processing_time:.3f}s",
            extra=log_data
        )
    
    async def _log_error(self, request: Request, error: Exception, processing_time: float):
        """Log error."""
        log_data = {
            'type': 'error',
            'method': request.method,
            'path': request.url.path,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'processing_time': processing_time,
            'timestamp': time.time()
        }
        
        logger.error(
            f"Error: {request.method} {request.url.path} - "
            f"{type(error).__name__}: {error}",
            extra=log_data
        )
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        return request.client.host if request.client else 'unknown'


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting metrics."""
    
    def __init__(self, app: ASGIApp, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(app)
        self.config = config or {}
        self.enable_metrics = self.config.get('enable_metrics', True)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enable_metrics:
            return await call_next(request)
        
        # Increment active requests
        ACTIVE_REQUESTS.inc()
        
        start_time = time.perf_counter()
        
        try:
            response = await call_next(request)
            processing_time = time.perf_counter() - start_time
            
            # Record metrics
            self._record_metrics(request, response, processing_time)
            
            return response
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            
            # Record error metrics
            self._record_error_metrics(request, e, processing_time)
            raise
        
        finally:
            # Decrement active requests
            ACTIVE_REQUESTS.dec()
    
    def _record_metrics(self, request: Request, response: Response, processing_time: float):
        """Record request metrics."""
        method = request.method
        endpoint = request.url.path
        
        # Record request count
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status=response.status_code
        ).inc()
        
        # Record request duration
        REQUEST_DURATION.labels(
            method=method,
            endpoint=endpoint
        ).observe(processing_time)
        
        # Record memory usage
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            MEMORY_USAGE.set(memory_info.rss)
            
            cpu_percent = process.cpu_percent()
            CPU_USAGE.set(cpu_percent)
        except ImportError:
            pass
    
    def _record_error_metrics(self, request: Request, error: Exception, processing_time: float):
        """Record error metrics."""
        method = request.method
        endpoint = request.url.path
        
        # Record error count
        ERROR_COUNT.labels(type=type(error).__name__).inc()
        
        # Record request duration for failed requests
        REQUEST_DURATION.labels(
            method=method,
            endpoint=endpoint
        ).observe(processing_time)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting."""
    
    def __init__(self, app: ASGIApp, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(app)
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.requests_per_minute = self.config.get('requests_per_minute', 100)
        self.requests_per_hour = self.config.get('requests_per_hour', 1000)
        self.by_ip = self.config.get('by_ip', True)
        
        # Rate limiting storage
        self.request_counts = {}
        self.ip_whitelist = set(self.config.get('ip_whitelist', []))
        self.ip_blacklist = set(self.config.get('ip_blacklist', []))
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled:
            return await call_next(request)
        
        client_ip = self._get_client_ip(request)
        
        # Check blacklist
        if client_ip in self.ip_blacklist:
            raise HTTPException(status_code=403, detail="IP address blocked")
        
        # Skip rate limiting for whitelisted IPs
        if client_ip in self.ip_whitelist:
            return await call_next(request)
        
        # Check rate limits
        if self.by_ip:
            if not await self._check_rate_limit(client_ip):
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded"
                )
        
        return await call_next(request)
    
    async def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client is within rate limits."""
        current_time = time.time()
        
        # Initialize client data
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = {
                'minute': {'count': 0, 'reset_time': current_time + 60},
                'hour': {'count': 0, 'reset_time': current_time + 3600}
            }
        
        client_data = self.request_counts[client_ip]
        
        # Reset counters if needed
        if current_time > client_data['minute']['reset_time']:
            client_data['minute'] = {'count': 0, 'reset_time': current_time + 60}
        
        if current_time > client_data['hour']['reset_time']:
            client_data['hour'] = {'count': 0, 'reset_time': current_time + 3600}
        
        # Check limits
        if (client_data['minute']['count'] >= self.requests_per_minute or
            client_data['hour']['count'] >= self.requests_per_hour):
            return False
        
        # Increment counters
        client_data['minute']['count'] += 1
        client_data['hour']['count'] += 1
        
        return True
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        return request.client.host if request.client else 'unknown'


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for security headers and validation."""
    
    def __init__(self, app: ASGIApp, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(app)
        self.config = config or {}
        self.enable_security_headers = self.config.get('enable_security_headers', True)
        self.enable_cors = self.config.get('enable_cors', True)
        self.max_request_size = self.config.get('max_request_size', 10 * 1024 * 1024)  # 10MB
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check request size
        content_length = request.headers.get('content-length')
        if content_length and int(content_length) > self.max_request_size:
            raise HTTPException(
                status_code=413,
                detail="Request too large"
            )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        if self.enable_security_headers:
            self._add_security_headers(response)
        
        return response
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response."""
        security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value


class CacheMiddleware(BaseHTTPMiddleware):
    """Middleware for response caching."""
    
    def __init__(self, app: ASGIApp, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(app)
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes
        self.cacheable_methods = {'GET', 'HEAD'}
        self.cacheable_paths = self.config.get('cacheable_paths', ['/health', '/metrics'])
        
        # Simple in-memory cache
        self.cache = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled:
            return await call_next(request)
        
        # Check if request is cacheable
        if not self._is_cacheable(request):
            return await call_next(request)
        
        # Generate cache key
        cache_key = self._generate_cache_key(request)
        
        # Check cache
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            CACHE_HIT_RATE.inc()
            return cached_response
        
        # Process request
        response = await call_next(request)
        
        # Cache response
        if response.status_code == 200:
            self._cache_response(cache_key, response)
        
        return response
    
    def _is_cacheable(self, request: Request) -> bool:
        """Check if request is cacheable."""
        return (request.method in self.cacheable_methods and
                request.url.path in self.cacheable_paths)
    
    def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key for request."""
        key_data = {
            'method': request.method,
            'path': request.url.path,
            'query': str(request.query_params),
            'headers': dict(request.headers)
        }
        
        key_string = orjson.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[Response]:
        """Get cached response."""
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if time.time() < cached_data['expires']:
                return cached_data['response']
            else:
                del self.cache[cache_key]
        
        return None
    
    def _cache_response(self, cache_key: str, response: Response):
        """Cache response."""
        # Create a copy of the response
        cached_response = Response(
            content=response.body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type
        )
        
        self.cache[cache_key] = {
            'response': cached_response,
            'expires': time.time() + self.cache_ttl
        }


class CircuitBreakerMiddleware(BaseHTTPMiddleware):
    """Middleware for circuit breaker pattern."""
    
    def __init__(self, app: ASGIApp, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(app)
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.failure_threshold = self.config.get('failure_threshold', 5)
        self.recovery_timeout = self.config.get('recovery_timeout', 60)
        self.expected_exception = self.config.get('expected_exception', Exception)
        
        # Circuit breaker state
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled:
            return await call_next(request)
        
        # Check circuit breaker state
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'HALF_OPEN'
            else:
                raise HTTPException(
                    status_code=503,
                    detail="Service temporarily unavailable"
                )
        
        try:
            response = await call_next(request)
            
            # Success - reset circuit breaker
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                self.failure_count = 0
            
            return response
            
        except self.expected_exception as e:
            # Failure - update circuit breaker
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
            
            raise


class RetryMiddleware(BaseHTTPMiddleware):
    """Middleware for automatic retries."""
    
    def __init__(self, app: ASGIApp, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(app)
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.max_attempts = self.config.get('max_attempts', 3)
        self.initial_delay = self.config.get('initial_delay', 1)
        self.max_delay = self.config.get('max_delay', 10)
        self.backoff_factor = self.config.get('backoff_factor', 2)
        self.retryable_status_codes = self.config.get('retryable_status_codes', [500, 502, 503, 504])
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled:
            return await call_next(request)
        
        # Don't retry non-idempotent methods
        if request.method not in {'GET', 'HEAD', 'OPTIONS'}:
            return await call_next(request)
        
        return await self._retry_request(request, call_next)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async async def _retry_request(self, request: Request, call_next: Callable) -> Response:
        """Retry request with exponential backoff."""
        response = await call_next(request)
        
        if response.status_code in self.retryable_status_codes:
            raise HTTPException(
                status_code=response.status_code,
                detail="Retryable error"
            )
        
        return response 