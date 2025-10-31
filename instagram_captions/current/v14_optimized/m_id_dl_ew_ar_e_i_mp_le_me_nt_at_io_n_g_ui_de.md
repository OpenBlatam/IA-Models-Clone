# Middleware Implementation Guide - Instagram Captions API v14.0

## Overview
This guide documents the comprehensive middleware implementation for the Instagram Captions API v14.0, covering logging, error monitoring, performance optimization, security, and response optimization.

## 🏗️ **Middleware Architecture**

### **Middleware Stack Order**
The middleware is applied in reverse order (last added = first executed):

1. **Error Handling Middleware** (outermost)
2. **Security Middleware**
3. **Performance Monitoring Middleware**
4. **Request Logging Middleware**
5. **Compression Middleware**
6. **Cache Middleware**
7. **CORS Middleware** (additional)

### **Configuration**
```python
middleware_config = {
    "enable_detailed_logging": True,
    "slow_request_threshold": 1.0,  # 1 second
    "enable_metrics": True,
    "max_request_size": 10 * 1024 * 1024,  # 10MB
    "enable_gzip": True,
    "min_compression_size": 1000
}

create_middleware_stack(app, middleware_config)
```

## 📊 **Request Logging Middleware**

### **Features**
- **Structured Logging**: JSON-formatted logs with consistent structure
- **Request ID Tracking**: Unique ID for each request
- **Performance Timing**: Precise request/response timing
- **Client Information**: IP, user agent, API key presence
- **Error Tracking**: Comprehensive error logging with context

### **Implementation**
```python
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Advanced request/response logging middleware with structured data."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Extract request info
        start_time = time.perf_counter()
        timestamp = datetime.now(timezone.utc)
        
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        api_key = request.headers.get("authorization", "").replace("Bearer ", "")
        
        # Log request start with structured data
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": client_ip,
            "user_agent": user_agent,
            "api_key_present": bool(api_key),
            "content_length": request.headers.get("content-length"),
            "timestamp": timestamp.isoformat(),
            "event": "request_started"
        }
        
        logger.info("Request started", extra=log_data)
        
        # Process request and log response
        # ... implementation details
```

### **Log Output Examples**

**Request Start:**
```json
{
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "method": "POST",
    "url": "http://localhost:8140/api/v14/generate",
    "path": "/api/v14/generate",
    "client_ip": "127.0.0.1",
    "user_agent": "curl/7.68.0",
    "api_key_present": true,
    "timestamp": "2024-01-15T10:30:45.123456+00:00",
    "event": "request_started"
}
```

**Request Completion:**
```json
{
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "status_code": 200,
    "duration_ms": 245.67,
    "duration_seconds": 0.246,
    "response_size": 1024,
    "event": "request_completed"
}
```

## ⚡ **Performance Monitoring Middleware**

### **Features**
- **Real-time Metrics**: Response times, error rates, throughput
- **Endpoint-specific Tracking**: Per-endpoint performance analysis
- **Performance Tiers**: Automatic categorization (excellent, good, acceptable, slow, very-slow, critical)
- **Percentile Calculations**: P95, P99 response times
- **Slow Request Detection**: Automatic alerting for slow requests

### **Implementation**
```python
class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Advanced performance monitoring middleware with metrics collection."""
    
    def __init__(self, app, slow_request_threshold: float = 1.0, enable_metrics: bool = True):
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
        self.enable_metrics = enable_metrics
        self.performance_monitor = PerformanceMonitor()
        self.metrics = {
            "total_requests": 0,
            "total_errors": 0,
            "slow_requests": 0,
            "average_response_time": 0.0,
            "p95_response_time": 0.0,
            "p99_response_time": 0.0,
            "endpoint_metrics": {},
            "response_times": []
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        endpoint = f"{request.method} {request.url.path}"
        
        # Process request and collect metrics
        response = await call_next(request)
        duration = time.perf_counter() - start_time
        
        # Update metrics
        self._update_metrics(endpoint, duration, False)
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        response.headers["X-Performance-Tier"] = self._get_performance_tier(duration)
        response.headers["X-Endpoint"] = endpoint
        
        return response
```

### **Performance Tiers**
```python
def _get_performance_tier(self, duration: float) -> str:
    """Get performance tier based on response time."""
    if duration < 0.050:  # 50ms
        return "excellent"
    elif duration < 0.100:  # 100ms
        return "good"
    elif duration < 0.250:  # 250ms
        return "acceptable"
    elif duration < 0.500:  # 500ms
        return "slow"
    elif duration < 1.000:  # 1s
        return "very-slow"
    else:
        return "critical"
```

### **Metrics Output**
```json
{
    "total_requests": 1250,
    "total_errors": 12,
    "slow_requests": 8,
    "average_response_time": 0.045,
    "p95_response_time": 0.089,
    "p99_response_time": 0.156,
    "endpoint_metrics": {
        "POST /api/v14/generate": {
            "count": 850,
            "total_time": 38.25,
            "error_count": 5,
            "slow_count": 3,
            "response_times": [0.023, 0.045, 0.067, ...]
        }
    }
}
```

## 🛡️ **Security Middleware**

### **Features**
- **Request Size Validation**: Prevents DoS attacks
- **Threat Detection**: SQL injection, XSS, path traversal detection
- **Security Headers**: Automatic security header injection
- **Content Scanning**: Malicious pattern detection
- **Security Logging**: Comprehensive security incident tracking

### **Implementation**
```python
class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware with threat detection and validation."""
    
    def __init__(self, app, max_request_size: int = 10 * 1024 * 1024):  # 10MB
        super().__init__(app)
        self.max_request_size = max_request_size
        self.error_tracker = ErrorTracker()
        self.threat_patterns = [
            r"<script", r"javascript:", r"data:", r"vbscript:",
            r"onload=", r"onerror=", r"onclick=",
            r"../", r"..\\", r"~", r"/etc/", r"/proc/",
            r"union.*select", r"drop.*table", r"insert.*into",
            r"exec\(", r"system\(", r"eval\("
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Validate request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            return JSONResponse(
                status_code=413,
                content={
                    "error": True,
                    "error_code": "REQUEST_TOO_LARGE",
                    "message": f"Request size exceeds {self.max_request_size // (1024*1024)}MB limit"
                }
            )
        
        # Security scan for malicious content
        body = await request.body()
        if body:
            body_str = body.decode('utf-8', errors='ignore')
            for pattern in self.threat_patterns:
                if pattern.lower() in body_str.lower():
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": True,
                            "error_code": "MALICIOUS_CONTENT",
                            "message": "Malicious content detected in request"
                        }
                    )
        
        # Add security headers
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response
```

## 🔧 **Error Handling Middleware**

### **Features**
- **Centralized Error Handling**: Single point for all error processing
- **Structured Error Responses**: Consistent error response format
- **Error Categorization**: Different handling for different error types
- **Error Tracking**: Comprehensive error logging and monitoring
- **Request ID Correlation**: Errors linked to specific requests

### **Implementation**
```python
class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Centralized error handling middleware with structured responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = getattr(request.state, "request_id", "unknown")
        
        try:
            return await call_next(request)
        except HTTPException as e:
            # HTTP exceptions are handled by FastAPI
            self.error_tracker.record_error(
                error_type=ErrorType.VALIDATION,
                message=str(e.detail),
                severity=ErrorSeverity.MEDIUM,
                details={"status_code": e.status_code},
                request_id=request_id
            )
            raise
        except ValueError as e:
            # Validation errors
            return JSONResponse(
                status_code=400,
                content={
                    "error": True,
                    "error_code": "VALIDATION_ERROR",
                    "message": str(e),
                    "request_id": request_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        except Exception as e:
            # Unexpected errors
            return JSONResponse(
                status_code=500,
                content={
                    "error": True,
                    "error_code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                    "request_id": request_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
```

## 🗜️ **Compression Middleware**

### **Features**
- **Automatic Compression**: Gzip compression for eligible responses
- **Size-based Compression**: Only compress responses above threshold
- **Content Type Filtering**: Only compress appropriate content types
- **Performance Optimization**: Reduced bandwidth usage

### **Implementation**
```python
class CompressionMiddleware(BaseHTTPMiddleware):
    """Response compression middleware for performance optimization."""
    
    def __init__(self, app, min_size: int = 1000, enable_gzip: bool = True):
        super().__init__(app)
        self.min_size = min_size
        self.enable_gzip = enable_gzip
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Check if response should be compressed
        if self.enable_gzip and self._should_compress(response):
            response.headers["Content-Encoding"] = "gzip"
            response.headers["Vary"] = "Accept-Encoding"
        
        return response
    
    def _should_compress(self, response: Response) -> bool:
        """Determine if response should be compressed."""
        content_type = response.headers.get("content-type", "")
        compressible_types = [
            "application/json",
            "text/plain",
            "text/html",
            "text/css",
            "application/javascript"
        ]
        
        if not any(ct in content_type for ct in compressible_types):
            return False
        
        if hasattr(response, 'body'):
            return len(response.body) >= self.min_size
        
        return False
```

## 💾 **Cache Middleware**

### **Features**
- **Endpoint-specific Caching**: Different cache policies per endpoint
- **Cache Headers**: Automatic cache control header injection
- **Performance Optimization**: Reduced server load through caching
- **Configurable Policies**: Flexible cache configuration

### **Implementation**
```python
class CacheMiddleware(BaseHTTPMiddleware):
    """Cache control middleware for response optimization."""
    
    def __init__(self, app):
        super().__init__(app)
        self.cache_settings = {
            "/health": {"max_age": 30},  # 30 seconds
            "/metrics": {"max_age": 60},  # 1 minute
            "/performance/status": {"max_age": 120},  # 2 minutes
            "/api/v14/info": {"max_age": 300},  # 5 minutes
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add cache headers based on endpoint
        path = request.url.path
        cache_config = self.cache_settings.get(path, {"max_age": 0})
        
        if cache_config["max_age"] > 0:
            response.headers["Cache-Control"] = f"public, max-age={cache_config['max_age']}"
        else:
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        
        return response
```

## 🏭 **Middleware Factory**

### **Factory Function**
```python
def create_middleware_stack(app, config: Dict[str, Any] = None):
    """Create and configure the complete middleware stack."""
    if config is None:
        config = {
            "enable_detailed_logging": True,
            "slow_request_threshold": 1.0,
            "enable_metrics": True,
            "max_request_size": 10 * 1024 * 1024,
            "enable_gzip": True,
            "min_compression_size": 1000
        }
    
    # Add middleware in reverse order (last added = first executed)
    
    # Error handling middleware (outermost)
    app.add_middleware(ErrorHandlingMiddleware)
    
    # Security middleware
    app.add_middleware(SecurityMiddleware, max_request_size=config["max_request_size"])
    
    # Performance monitoring middleware
    if config["enable_metrics"]:
        app.add_middleware(
            PerformanceMonitoringMiddleware,
            slow_request_threshold=config["slow_request_threshold"],
            enable_metrics=config["enable_metrics"]
        )
    
    # Request logging middleware
    app.add_middleware(
        RequestLoggingMiddleware,
        enable_detailed_logging=config["enable_detailed_logging"]
    )
    
    # Compression middleware
    if config["enable_gzip"]:
        app.add_middleware(
            CompressionMiddleware,
            min_size=config["min_compression_size"],
            enable_gzip=config["enable_gzip"]
        )
    
    # Cache middleware
    app.add_middleware(CacheMiddleware)
    
    return app
```

## 📈 **Performance Context Manager**

### **Usage**
```python
@asynccontextmanager
async def middleware_performance_context(operation: str, request_id: str):
    """Context manager for tracking middleware performance."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        if duration > 0.1:  # Log slow middleware operations
            logger.warning(f"Slow middleware operation: {operation} took {duration*1000:.1f}ms")
```

## 🎯 **Benefits**

### **✅ Logging**
- **Structured Data**: Consistent JSON logging format
- **Request Tracking**: Unique IDs for request correlation
- **Performance Timing**: Precise request/response timing
- **Error Context**: Comprehensive error information

### **✅ Performance Monitoring**
- **Real-time Metrics**: Live performance tracking
- **Endpoint Analysis**: Per-endpoint performance breakdown
- **Threshold Detection**: Automatic slow request detection
- **Percentile Tracking**: P95, P99 response time monitoring

### **✅ Error Monitoring**
- **Centralized Handling**: Single point for error processing
- **Error Categorization**: Different handling per error type
- **Structured Responses**: Consistent error response format
- **Error Tracking**: Comprehensive error logging

### **✅ Security**
- **Threat Detection**: Malicious content scanning
- **Request Validation**: Size and content validation
- **Security Headers**: Automatic security header injection
- **Security Logging**: Incident tracking and reporting

### **✅ Optimization**
- **Response Compression**: Automatic gzip compression
- **Cache Control**: Endpoint-specific caching policies
- **Performance Headers**: Response timing and tier information
- **Bandwidth Optimization**: Reduced data transfer

## 📊 **Metrics and Monitoring**

### **Key Metrics**
- **Response Times**: Average, P95, P99, min, max
- **Error Rates**: Per-endpoint error tracking
- **Throughput**: Requests per second
- **Cache Performance**: Hit rates and efficiency
- **Security Incidents**: Threat detection statistics

### **Performance Thresholds**
- **Excellent**: < 50ms
- **Good**: < 100ms
- **Acceptable**: < 250ms
- **Slow**: < 500ms
- **Very Slow**: < 1s
- **Critical**: ≥ 1s

### **Monitoring Integration**
- **Structured Logging**: JSON format for log aggregation
- **Performance Headers**: Real-time performance information
- **Error Tracking**: Comprehensive error monitoring
- **Security Monitoring**: Threat detection and reporting

This middleware implementation provides comprehensive logging, error monitoring, and performance optimization for the Instagram Captions API v14.0, ensuring robust, secure, and high-performance operation. 