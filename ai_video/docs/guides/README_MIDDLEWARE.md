# 🔧 Middleware for Logging, Error Monitoring & Performance

## Overview

This guide covers implementing comprehensive middleware for FastAPI applications, including request/response logging, error monitoring, performance optimization, security headers, rate limiting, and caching.

## Why Use Middleware?

### Benefits
- **Centralized logging**: Consistent request/response logging across all endpoints
- **Error tracking**: Monitor and alert on errors in real-time
- **Performance monitoring**: Track response times and resource usage
- **Security**: Add security headers and rate limiting
- **Caching**: Improve response times with intelligent caching
- **Observability**: Better understanding of application behavior

### Common Use Cases
- Request/response logging
- Error monitoring and alerting
- Performance metrics collection
- Rate limiting
- Response caching
- Security headers
- Authentication/authorization
- Request validation

## Middleware Patterns

### 1. Request Logging Middleware

```python
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Comprehensive request/response logging middleware."""
    
    def __init__(self, app, 
                 log_requests: bool = True, 
                 log_responses: bool = True,
                 log_errors: bool = True,
                 sensitive_headers: set = None):
        super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.log_errors = log_errors
        self.sensitive_headers = sensitive_headers or {'authorization', 'cookie', 'x-api-key'}
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        request.state.start_time = time.time()
        
        # Log request
        if self.log_requests:
            await self.log_request(request, request_id)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - request.state.start_time
            
            # Log response
            if self.log_responses:
                await self.log_response(response, request_id, duration)
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = f"{duration:.3f}s"
            
            return response
            
        except Exception as e:
            # Log error
            duration = time.time() - request.state.start_time
            if self.log_errors:
                await self.log_error(request, request_id, e, duration)
            raise
```

### 2. Error Monitoring Middleware

```python
class ErrorMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive error monitoring and tracking."""
    
    def __init__(self, app, 
                 error_tracking: bool = True,
                 alert_threshold: int = 10,
                 alert_window: int = 300):
        super().__init__(app)
        self.error_tracking = error_tracking
        self.alert_threshold = alert_threshold
        self.alert_window = alert_window
        self.error_counts = {}
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            response = await call_next(request)
            
            # Track 4xx and 5xx responses
            if 400 <= response.status_code < 600:
                await self.track_error(response.status_code, request)
            
            return response
            
        except Exception as e:
            # Track exceptions
            await self.track_exception(e, request)
            
            # Check if we should alert
            await self.check_alert_threshold()
            
            # Return error response
            return await self.create_error_response(e, request)
    
    async def track_error(self, status_code: int, request: Request):
        """Track HTTP error responses."""
        error_key = f"error:{status_code}:{request.url.path}"
        current_time = time.time()
        
        if error_key not in self.error_counts:
            self.error_counts[error_key] = []
        
        self.error_counts[error_key].append(current_time)
        
        # Clean old entries
        cutoff_time = current_time - self.alert_window
        self.error_counts[error_key] = [
            t for t in self.error_counts[error_key] 
            if t > cutoff_time
        ]
    
    async def check_alert_threshold(self):
        """Check if error threshold has been exceeded."""
        current_time = time.time()
        cutoff_time = current_time - self.alert_window
        
        for error_key, timestamps in self.error_counts.items():
            recent_errors = [t for t in timestamps if t > cutoff_time]
            
            if len(recent_errors) >= self.alert_threshold:
                logger.critical(f"🚨 ALERT: {len(recent_errors)} errors for {error_key} in {self.alert_window}s")
                # Send alerts (email, Slack, etc.)
```

### 3. Performance Monitoring Middleware

```python
class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring and optimization."""
    
    def __init__(self, app, 
                 track_performance: bool = True,
                 slow_request_threshold: float = 1.0,
                 memory_tracking: bool = True):
        super().__init__(app)
        self.track_performance = track_performance
        self.slow_request_threshold = slow_request_threshold
        self.memory_tracking = memory_tracking
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Record initial state
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss if self.memory_tracking else 0
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate metrics
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss if self.memory_tracking else 0
            memory_delta = end_memory - start_memory
            
            # Track performance
            if self.track_performance:
                await self.track_performance_metrics(
                    request, response, duration, memory_delta
                )
            
            # Add performance headers
            response.headers["X-Processing-Time"] = f"{duration:.3f}s"
            if self.memory_tracking:
                response.headers["X-Memory-Delta"] = f"{memory_delta / 1024 / 1024:.2f}MB"
            
            return response
            
        except Exception as e:
            # Calculate metrics even for errors
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss if self.memory_tracking else 0
            memory_delta = end_memory - start_memory
            
            # Track error performance
            if self.track_performance:
                await self.track_performance_metrics(
                    request, None, duration, memory_delta, error=True
                )
            
            raise
    
    async def track_performance_metrics(self, request: Request, response: Optional[Response], 
                                      duration: float, memory_delta: int, error: bool = False):
        """Track performance metrics."""
        metrics = {
            "endpoint": f"{request.method}:{request.url.path}",
            "duration": duration,
            "memory_delta": memory_delta,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        if response:
            metrics["status_code"] = response.status_code
            metrics["content_length"] = response.headers.get("content-length")
        
        # Log slow requests
        if duration > self.slow_request_threshold:
            logger.warning(f"🐌 Slow request: {duration:.3f}s for {request.url.path}")
        
        # Store metrics
        await self.store_metrics(metrics)
```

### 4. Security Middleware

```python
class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for security headers and protection."""
    
    def __init__(self, app):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response
```

### 5. Rate Limiting Middleware

```python
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting requests."""
    
    def __init__(self, app, 
                 rate_limit: int = 100,
                 window: int = 3600,
                 redis_client: Optional[redis.Redis] = None):
        super().__init__(app)
        self.rate_limit = rate_limit
        self.window = window
        self.redis_client = redis_client
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Get client identifier
        client_id = self.get_client_id(request)
        
        # Check rate limit
        if not await self.check_rate_limit(client_id):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": self.window
                },
                headers={
                    "Retry-After": str(self.window),
                    "X-RateLimit-Limit": str(self.rate_limit),
                    "X-RateLimit-Remaining": "0"
                }
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = await self.get_remaining_requests(client_id)
        response.headers["X-RateLimit-Limit"] = str(self.rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        return response
    
    def get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        return request.client.host if request.client else "unknown"
    
    async def check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit."""
        if not self.redis_client:
            return True  # Skip rate limiting if no Redis
        
        key = f"rate_limit:{client_id}"
        
        try:
            current = await self.redis_client.get(key)
            
            if current is None:
                await self.redis_client.setex(key, self.window, 1)
                return True
            
            current_count = int(current)
            if current_count >= self.rate_limit:
                return False
            
            await self.redis_client.incr(key)
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True
```

### 6. Caching Middleware

```python
class CacheMiddleware(BaseHTTPMiddleware):
    """Middleware for response caching."""
    
    def __init__(self, app, 
                 cache_ttl: int = 3600,
                 redis_client: Optional[redis.Redis] = None,
                 cacheable_paths: set = None):
        super().__init__(app)
        self.cache_ttl = cache_ttl
        self.redis_client = redis_client
        self.cacheable_paths = cacheable_paths or {"/health", "/metrics"}
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Check if path is cacheable
        if request.url.path not in self.cacheable_paths:
            return await call_next(request)
        
        # Generate cache key
        cache_key = self.generate_cache_key(request)
        
        # Try to get cached response
        cached_response = await self.get_cached_response(cache_key)
        if cached_response:
            cached_response.headers["X-Cache"] = "HIT"
            return cached_response
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if 200 <= response.status_code < 300:
            await self.cache_response(cache_key, response)
            response.headers["X-Cache"] = "MISS"
        
        return response
    
    def generate_cache_key(self, request: Request) -> str:
        """Generate cache key for request."""
        key_parts = [request.url.path]
        
        if request.query_params:
            sorted_params = sorted(request.query_params.items())
            key_parts.append("&".join(f"{k}={v}" for k, v in sorted_params))
        
        return f"cache:{':'.join(key_parts)}"
    
    async def get_cached_response(self, cache_key: str) -> Optional[Response]:
        """Get cached response from Redis."""
        if not self.redis_client:
            return None
        
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                return JSONResponse(
                    content=data["content"],
                    status_code=data["status_code"],
                    headers=data["headers"]
                )
        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
        
        return None
    
    async def cache_response(self, cache_key: str, response: Response):
        """Cache response in Redis."""
        if not self.redis_client:
            return
        
        try:
            if hasattr(response, 'body'):
                content = response.body.decode()
            else:
                content = ""
            
            cache_data = {
                "content": content,
                "status_code": response.status_code,
                "headers": dict(response.headers)
            }
            
            await self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(cache_data)
            )
            
        except Exception as e:
            logger.error(f"Cache storage failed: {e}")
```

## Middleware Stack Creation

### Comprehensive Middleware Stack

```python
def create_middleware_stack(app, 
                          redis_client: Optional[redis.Redis] = None,
                          enable_logging: bool = True,
                          enable_monitoring: bool = True,
                          enable_security: bool = True,
                          enable_rate_limiting: bool = True,
                          enable_caching: bool = True) -> None:
    """Create and configure middleware stack."""
    
    # Add middleware in order (last added = first executed)
    
    # 1. Security middleware (first)
    if enable_security:
        app.add_middleware(SecurityMiddleware)
    
    # 2. Rate limiting
    if enable_rate_limiting:
        app.add_middleware(RateLimitMiddleware, 
                          rate_limit=100, 
                          window=3600,
                          redis_client=redis_client)
    
    # 3. Caching
    if enable_caching:
        app.add_middleware(CacheMiddleware,
                          cache_ttl=3600,
                          redis_client=redis_client)
    
    # 4. Performance monitoring
    if enable_monitoring:
        app.add_middleware(PerformanceMiddleware,
                          track_performance=True,
                          slow_request_threshold=1.0,
                          memory_tracking=True)
    
    # 5. Error monitoring
    if enable_monitoring:
        app.add_middleware(ErrorMonitoringMiddleware,
                          error_tracking=True,
                          alert_threshold=10,
                          alert_window=300)
    
    # 6. Request logging (last)
    if enable_logging:
        app.add_middleware(RequestLoggingMiddleware,
                          log_requests=True,
                          log_responses=True,
                          log_errors=True)
    
    # 7. Standard FastAPI middleware
    app.add_middleware(CORSMiddleware,
                      allow_origins=["*"],
                      allow_credentials=True,
                      allow_methods=["*"],
                      allow_headers=["*"])
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
```

## Usage Examples

### Basic FastAPI App with Middleware

```python
from fastapi import FastAPI
import redis.asyncio as redis

app = FastAPI()

# Create Redis client
redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)

# Create middleware stack
create_middleware_stack(
    app,
    redis_client=redis_client,
    enable_logging=True,
    enable_monitoring=True,
    enable_security=True,
    enable_rate_limiting=True,
    enable_caching=True
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/metrics")
async def get_metrics():
    return {"metrics": "data"}

@app.post("/api/data")
async def create_data():
    return {"message": "Data created"}
```

### AI Video API with Specialized Middleware

```python
def create_ai_video_app() -> FastAPI:
    """Create FastAPI application with AI Video specific middleware."""
    
    app = FastAPI(
        title="AI Video Generation API",
        description="API for AI video generation with comprehensive middleware",
        version="1.0.0"
    )
    
    # Create Redis client
    redis_client = redis.Redis(
        host="localhost",
        port=6379,
        db=0,
        decode_responses=True
    )
    
    # Add AI Video specific middleware
    app.add_middleware(AIVideoRateLimitMiddleware, redis_client=redis_client)
    app.add_middleware(AIVideoCacheMiddleware, redis_client=redis_client)
    app.add_middleware(PerformanceMonitoringMiddleware,
                      slow_threshold=2.0,
                      memory_threshold=100 * 1024 * 1024)
    app.add_middleware(ErrorMonitoringMiddleware,
                      alert_threshold=5,
                      alert_window=300)
    app.add_middleware(AIVideoLoggingMiddleware)
    
    # Standard middleware
    app.add_middleware(CORSMiddleware,
                      allow_origins=["*"],
                      allow_credentials=True,
                      allow_methods=["*"],
                      allow_headers=["*"])
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    return app
```

## Best Practices

### 1. Middleware Order
- **Security first**: Add security middleware first
- **Rate limiting early**: Check rate limits before processing
- **Caching before processing**: Check cache before expensive operations
- **Monitoring throughout**: Track performance and errors
- **Logging last**: Log after all processing is complete

### 2. Error Handling
- Always handle exceptions in middleware
- Don't let middleware errors break the application
- Log errors appropriately
- Provide fallback behavior when possible

### 3. Performance Considerations
- Keep middleware lightweight
- Use async operations when possible
- Cache expensive operations
- Monitor middleware performance

### 4. Configuration
- Make middleware configurable
- Use environment variables for settings
- Provide sensible defaults
- Allow disabling middleware in development

### 5. Testing
- Test middleware independently
- Mock external dependencies (Redis, etc.)
- Test error scenarios
- Verify middleware order

## Summary

- **Use middleware** for centralized cross-cutting concerns
- **Implement logging** for request/response tracking
- **Add error monitoring** for real-time alerting
- **Track performance** for optimization
- **Add security headers** for protection
- **Implement rate limiting** for API protection
- **Use caching** for performance improvement
- **Order middleware** correctly for optimal behavior
- **Test thoroughly** to ensure reliability

This approach provides comprehensive observability, security, and performance optimization for FastAPI applications. 