# 🚀 ERROR MIDDLEWARE SYSTEM GUIDE

## Overview

This guide covers the comprehensive error middleware system for AI Video applications, providing:
- **Global error handling** for unexpected errors
- **Structured logging** with request/response correlation
- **Error monitoring** and alerting with circuit breakers
- **Performance tracking** and resource monitoring
- **Error recovery strategies** with retry and fallback patterns
- **Real-time alerting** and system health monitoring

## Table of Contents

1. [Error Types and Categorization](#error-types-and-categorization)
2. [Error Tracking and Monitoring](#error-tracking-and-monitoring)
3. [Structured Logging Middleware](#structured-logging-middleware)
4. [Error Handling Middleware](#error-handling-middleware)
5. [Performance Monitoring Middleware](#performance-monitoring-middleware)
6. [Circuit Breaker Patterns](#circuit-breaker-patterns)
7. [Alerting and Monitoring](#alerting-and-monitoring)
8. [Middleware Stack](#middleware-stack)
9. [Best Practices](#best-practices)
10. [Examples](#examples)

## Error Types and Categorization

### ErrorType Enum

```python
class ErrorType(Enum):
    VALIDATION = "validation"                    # Input validation errors
    AUTHENTICATION = "authentication"            # Authentication failures
    AUTHORIZATION = "authorization"              # Authorization failures
    NOT_FOUND = "not_found"                     # Resource not found
    RATE_LIMIT = "rate_limit"                   # Rate limiting exceeded
    TIMEOUT = "timeout"                         # Request timeouts
    DATABASE = "database"                       # Database errors
    CACHE = "cache"                             # Cache errors
    EXTERNAL_SERVICE = "external_service"       # External service errors
    MODEL = "model"                             # AI model errors
    MEMORY = "memory"                           # Memory errors
    SYSTEM = "system"                           # System errors
    UNKNOWN = "unknown"                         # Unknown errors
```

### ErrorAction Enum

```python
class ErrorAction(Enum):
    LOG = "log"                 # Just log the error
    ALERT = "alert"             # Send alert
    RETRY = "retry"             # Retry the operation
    FALLBACK = "fallback"       # Use fallback mechanism
    CIRCUIT_BREAK = "circuit_break"  # Trigger circuit breaker
    IGNORE = "ignore"           # Ignore the error
```

### ErrorInfo Class

```python
@dataclass
class ErrorInfo:
    error_type: ErrorType
    severity: ErrorSeverity
    action: ErrorAction
    retry_count: int = 0
    max_retries: int = 3
    retry_delay: float = 1.0
    alert_threshold: int = 5
    circuit_break_threshold: int = 10
```

## Error Tracking and Monitoring

### ErrorTracker Class

```python
class ErrorTracker:
    def __init__(self):
        self.error_counts = {}
        self.error_timestamps = []
        self.circuit_breakers = {}
        self.alert_history = []
        self.max_history = 1000
    
    def record_error(self, error_type: ErrorType, error_info: ErrorInfo, context: Dict[str, Any]):
        """Record an error for monitoring."""
        # Update error counts
        # Record timestamps
        # Check circuit breakers
        # Check alert thresholds
    
    def is_circuit_broken(self, error_type: ErrorType) -> bool:
        """Check if circuit breaker is active for error type."""
        return error_type.value in self.circuit_breakers
    
    def get_error_stats(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get error statistics for the last N minutes."""
        return {
            "total_errors": len(recent_errors),
            "errors_by_type": self._group_errors_by_type(recent_errors),
            "errors_by_severity": self._group_errors_by_severity(recent_errors),
            "error_rate": len(recent_errors) / window_minutes,
            "circuit_breakers": self.circuit_breakers,
            "recent_alerts": self.alert_history[-10:]
        }
```

### Circuit Breaker Implementation

```python
def _check_circuit_breaker(self, error_type: ErrorType, error_info: ErrorInfo):
    """Check if circuit breaker should be triggered."""
    if error_info.circuit_break_threshold <= 0:
        return
    
    recent_errors = [
        e for e in self.error_timestamps[-100:]  # Last 100 errors
        if e["error_type"] == error_type.value
    ]
    
    if len(recent_errors) >= error_info.circuit_break_threshold:
        self.circuit_breakers[error_type.value] = {
            "triggered_at": time.time(),
            "error_count": len(recent_errors),
            "threshold": error_info.circuit_break_threshold
        }
        
        logger.critical(f"Circuit breaker triggered for {error_type.value}")
```

## Structured Logging Middleware

### RequestLog Class

```python
@dataclass
class RequestLog:
    request_id: str
    method: str
    url: str
    client_ip: str
    user_agent: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    user_id: Optional[str] = None
    video_id: Optional[str] = None
    model_name: Optional[str] = None
    request_size: Optional[int] = None
    response_size: Optional[int] = None
    memory_usage: Optional[float] = None
```

### StructuredLoggingMiddleware

```python
class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, log_level: str = "INFO"):
        super().__init__(app)
        self.log_level = getattr(logging, log_level.upper())
        self.logger = logging.getLogger("request_logger")
        self.logger.setLevel(self.log_level)
    
    async def dispatch(self, request: Request, call_next):
        """Process request and log structured information."""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Extract request information
        log_entry = RequestLog(
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            client_ip=self._get_client_ip(request),
            user_agent=request.headers.get("user-agent", ""),
            start_time=start_time,
            user_id=request.headers.get("x-user-id"),
            video_id=self._extract_video_id(request),
            model_name=request.query_params.get("model_name"),
            request_size=self._get_request_size(request)
        )
        
        # Log request start
        self._log_request_start(log_entry)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Update log entry
            end_time = time.time()
            log_entry.end_time = end_time
            log_entry.duration = end_time - start_time
            log_entry.status_code = response.status_code
            log_entry.response_size = self._get_response_size(response)
            log_entry.memory_usage = self._get_memory_usage()
            
            # Log request completion
            self._log_request_complete(log_entry)
            
            return response
            
        except Exception as exc:
            # Update log entry with error
            end_time = time.time()
            log_entry.end_time = end_time
            log_entry.duration = end_time - start_time
            log_entry.error = str(exc)
            log_entry.error_type = exc.__class__.__name__
            log_entry.memory_usage = self._get_memory_usage()
            
            # Log request error
            self._log_request_error(log_entry)
            
            # Re-raise exception
            raise
```

### Logging Examples

```python
# Request start log
{
    "event": "request_start",
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "method": "POST",
    "url": "http://localhost:8000/videos/generate",
    "client_ip": "127.0.0.1",
    "user_agent": "Mozilla/5.0...",
    "timestamp": "2024-01-01T12:00:00",
    "user_id": "user_123",
    "video_id": "video_456",
    "model_name": "stable-diffusion",
    "request_size": 1024
}

# Request completion log
{
    "event": "request_complete",
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "method": "POST",
    "url": "http://localhost:8000/videos/generate",
    "duration": 2.5,
    "status_code": 200,
    "timestamp": "2024-01-01T12:00:02.5",
    "response_size": 512,
    "memory_usage": 768.5
}

# Request error log
{
    "event": "request_error",
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "method": "POST",
    "url": "http://localhost:8000/videos/generate",
    "duration": 1.2,
    "error": "Model loading failed",
    "error_type": "ModelLoadError",
    "timestamp": "2024-01-01T12:00:01.2",
    "memory_usage": 1024.0
}
```

## Error Handling Middleware

### ErrorHandlingMiddleware

```python
class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, error_tracker: ErrorTracker):
        super().__init__(app)
        self.error_tracker = error_tracker
        self.error_handler = HTTPExceptionHandler()
        self.error_monitor = ErrorMonitor()
        
        # Error type mapping for different exception types
        self.error_type_mapping = {
            RequestValidationError: ErrorType.VALIDATION,
            StarletteHTTPException: ErrorType.SYSTEM,
            AIVideoHTTPException: ErrorType.SYSTEM,
            TimeoutError: ErrorType.TIMEOUT,
            MemoryError: ErrorType.MEMORY,
            ConnectionError: ErrorType.EXTERNAL_SERVICE,
            # ... more mappings
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process request with error handling."""
        try:
            response = await call_next(request)
            return response
            
        except Exception as exc:
            # Determine error type
            error_type = self._determine_error_type(exc)
            error_info = self._get_error_info(error_type, exc)
            
            # Create error context
            context = self._create_error_context(request, exc)
            
            # Record error for monitoring
            self.error_tracker.record_error(error_type, error_info, context)
            self.error_monitor.record_error(self._convert_to_ai_video_exception(exc, context))
            
            # Handle error based on type
            return await self._handle_error(exc, error_type, error_info, context)
```

### Error Type Determination

```python
def _determine_error_type(self, exc: Exception) -> ErrorType:
    """Determine the type of error."""
    exc_type = type(exc)
    
    # Check direct type mapping
    if exc_type in self.error_type_mapping:
        return self.error_type_mapping[exc_type]
    
    # Check for specific error patterns
    error_message = str(exc).lower()
    
    if any(word in error_message for word in ["timeout", "timed out"]):
        return ErrorType.TIMEOUT
    elif any(word in error_message for word in ["memory", "out of memory"]):
        return ErrorType.MEMORY
    elif any(word in error_message for word in ["database", "sql", "connection"]):
        return ErrorType.DATABASE
    elif any(word in error_message for word in ["cache", "redis"]):
        return ErrorType.CACHE
    elif any(word in error_message for word in ["model", "inference", "gpu"]):
        return ErrorType.MODEL
    elif any(word in error_message for word in ["not found", "missing", "404"]):
        return ErrorType.NOT_FOUND
    elif any(word in error_message for word in ["permission", "access denied", "unauthorized"]):
        return ErrorType.AUTHORIZATION
    elif any(word in error_message for word in ["rate limit", "too many requests"]):
        return ErrorType.RATE_LIMIT
    elif any(word in error_message for word in ["validation", "invalid", "bad request"]):
        return ErrorType.VALIDATION
    
    return ErrorType.UNKNOWN
```

### Error Handling Strategies

```python
def _get_error_info(self, error_type: ErrorType, exc: Exception) -> ErrorInfo:
    """Get error information based on error type."""
    if error_type == ErrorType.TIMEOUT:
        return ErrorInfo(
            error_type=error_type,
            severity=ErrorSeverity.MEDIUM,
            action=ErrorAction.RETRY,
            max_retries=3,
            retry_delay=2.0,
            alert_threshold=10
        )
    elif error_type == ErrorType.MEMORY:
        return ErrorInfo(
            error_type=error_type,
            severity=ErrorSeverity.HIGH,
            action=ErrorAction.ALERT,
            alert_threshold=3,
            circuit_break_threshold=5
        )
    elif error_type == ErrorType.DATABASE:
        return ErrorInfo(
            error_type=error_type,
            severity=ErrorSeverity.HIGH,
            action=ErrorAction.RETRY,
            max_retries=5,
            retry_delay=1.0,
            alert_threshold=5,
            circuit_break_threshold=10
        )
    elif error_type == ErrorType.CACHE:
        return ErrorInfo(
            error_type=error_type,
            severity=ErrorSeverity.MEDIUM,
            action=ErrorAction.FALLBACK,
            alert_threshold=10
        )
    elif error_type == ErrorType.MODEL:
        return ErrorInfo(
            error_type=error_type,
            severity=ErrorSeverity.HIGH,
            action=ErrorAction.ALERT,
            alert_threshold=3,
            circuit_break_threshold=5
        )
    elif error_type == ErrorType.EXTERNAL_SERVICE:
        return ErrorInfo(
            error_type=error_type,
            severity=ErrorSeverity.HIGH,
            action=ErrorAction.RETRY,
            max_retries=3,
            retry_delay=5.0,
            alert_threshold=5,
            circuit_break_threshold=10
        )
    else:
        return ErrorInfo(
            error_type=error_type,
            severity=ErrorSeverity.MEDIUM,
            action=ErrorAction.LOG,
            alert_threshold=20
        )
```

## Performance Monitoring Middleware

### PerformanceMetrics Class

```python
@dataclass
class PerformanceMetrics:
    request_id: str
    method: str
    url: str
    duration: float
    memory_before: float
    memory_after: float
    memory_delta: float
    cpu_usage: float
    status_code: int
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
```

### PerformanceMonitoringMiddleware

```python
class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.metrics = []
        self.max_metrics = 1000
        self.slow_request_threshold = 5.0  # seconds
        self.high_memory_threshold = 1024  # MB
    
    async def dispatch(self, request: Request, call_next):
        """Process request with performance monitoring."""
        start_time = time.time()
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
        # Get initial metrics
        memory_before = self._get_memory_usage()
        cpu_before = self._get_cpu_usage()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate metrics
            end_time = time.time()
            duration = end_time - start_time
            memory_after = self._get_memory_usage()
            memory_delta = memory_after - memory_before
            
            # Create metrics
            metrics = PerformanceMetrics(
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                duration=duration,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_delta=memory_delta,
                cpu_usage=cpu_before,
                status_code=response.status_code
            )
            
            # Record metrics
            self._record_metrics(metrics)
            
            # Check for performance issues
            self._check_performance_issues(metrics)
            
            return response
            
        except Exception as exc:
            # Calculate metrics for error case
            end_time = time.time()
            duration = end_time - start_time
            memory_after = self._get_memory_usage()
            memory_delta = memory_after - memory_before
            
            metrics = PerformanceMetrics(
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                duration=duration,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_delta=memory_delta,
                cpu_usage=cpu_before,
                status_code=500,
                error=str(exc)
            )
            
            self._record_metrics(metrics)
            self._check_performance_issues(metrics)
            
            raise
```

### Performance Issue Detection

```python
def _check_performance_issues(self, metrics: PerformanceMetrics):
    """Check for performance issues and log warnings."""
    if metrics.duration > self.slow_request_threshold:
        logger.warning(f"Slow request detected: {metrics.duration:.2f}s for {metrics.url}")
    
    if metrics.memory_delta > 100:  # 100MB increase
        logger.warning(f"High memory usage detected: {metrics.memory_delta:.2f}MB increase")
    
    if metrics.memory_after > self.high_memory_threshold:
        logger.warning(f"High memory usage: {metrics.memory_after:.2f}MB")

def get_performance_stats(self, window_minutes: int = 5) -> Dict[str, Any]:
    """Get performance statistics for the last N minutes."""
    cutoff_time = time.time() - (window_minutes * 60)
    recent_metrics = [
        m for m in self.metrics
        if m.timestamp > cutoff_time
    ]
    
    if not recent_metrics:
        return {"message": "No metrics available"}
    
    durations = [m.duration for m in recent_metrics]
    memory_deltas = [m.memory_delta for m in recent_metrics]
    cpu_usage = [m.cpu_usage for m in recent_metrics]
    
    return {
        "total_requests": len(recent_metrics),
        "avg_duration": sum(durations) / len(durations),
        "max_duration": max(durations),
        "min_duration": min(durations),
        "avg_memory_delta": sum(memory_deltas) / len(memory_deltas),
        "max_memory_delta": max(memory_deltas),
        "avg_cpu_usage": sum(cpu_usage) / len(cpu_usage),
        "max_cpu_usage": max(cpu_usage),
        "slow_requests": len([d for d in durations if d > self.slow_request_threshold]),
        "high_memory_requests": len([m for m in memory_deltas if m > 100])
    }
```

## Circuit Breaker Patterns

### Circuit Breaker Implementation

```python
class CircuitBreakerExample:
    def __init__(self):
        self.error_tracker = ErrorTracker()
        self.request_count = 0
    
    async def process_video_with_circuit_breaker(self, video_id: str) -> Dict[str, Any]:
        """Process video with circuit breaker protection."""
        
        # Check if circuit breaker is active for video processing
        if self.error_tracker.is_circuit_broken(ErrorType.MODEL):
            return {
                "error": "Service temporarily unavailable due to model errors",
                "status": "circuit_broken",
                "video_id": video_id
            }
        
        try:
            # Simulate video processing
            self.request_count += 1
            
            # Simulate errors for demonstration
            if self.request_count % 3 == 0:  # Every 3rd request fails
                raise ModelLoadError("stable-diffusion", "Model loading failed")
            
            # Simulate successful processing
            await asyncio.sleep(1)
            
            return {
                "video_id": video_id,
                "status": "processed",
                "processing_time": 1.0
            }
            
        except Exception as exc:
            # Record error for circuit breaker
            error_info = ErrorInfo(
                error_type=ErrorType.MODEL,
                severity=ErrorSeverity.HIGH,
                action=ErrorAction.ALERT,
                alert_threshold=3,
                circuit_break_threshold=5
            )
            
            context = {
                "video_id": video_id,
                "request_count": self.request_count,
                "error_type": exc.__class__.__name__
            }
            
            self.error_tracker.record_error(ErrorType.MODEL, error_info, context)
            
            # Check if circuit breaker should be triggered
            if self.error_tracker.is_circuit_broken(ErrorType.MODEL):
                logger.critical(f"Circuit breaker triggered for video {video_id}")
            
            raise
```

### Circuit Breaker States

```python
# Circuit breaker states
CIRCUIT_CLOSED = "closed"      # Normal operation
CIRCUIT_OPEN = "open"          # Circuit breaker triggered
CIRCUIT_HALF_OPEN = "half_open"  # Testing if service is back

# Circuit breaker response
{
    "error": {
        "type": "CircuitBreakerError",
        "message": "Service temporarily unavailable due to model errors",
        "category": "system_error",
        "severity": "high",
        "status_code": 503,
        "timestamp": 1640995200.0,
        "context": {
            "error_type": "model",
            "triggered_at": 1640995200.0,
            "error_count": 10,
            "threshold": 5
        }
    }
}
```

## Alerting and Monitoring

### Alert System

```python
class AlertingExample:
    def __init__(self):
        self.error_tracker = ErrorTracker()
        self.alert_history = []
    
    def send_alert(self, alert_type: str, message: str, severity: str):
        """Send alert (simulated)."""
        alert = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": time.time()
        }
        
        self.alert_history.append(alert)
        logger.warning(f"ALERT: {alert_type} - {message}")
        
        return alert
    
    async def monitor_error_rates(self):
        """Monitor error rates and send alerts."""
        
        # Simulate error recording
        for i in range(10):
            error_info = ErrorInfo(
                error_type=ErrorType.MODEL,
                severity=ErrorSeverity.HIGH,
                action=ErrorAction.ALERT,
                alert_threshold=5
            )
            
            context = {
                "error_count": i + 1,
                "timestamp": time.time()
            }
            
            self.error_tracker.record_error(ErrorType.MODEL, error_info, context)
            
            # Check for alerts
            stats = self.error_tracker.get_error_stats(window_minutes=1)
            
            if stats["error_rate"] > 5:  # More than 5 errors per minute
                self.send_alert(
                    "HIGH_ERROR_RATE",
                    f"Error rate is {stats['error_rate']:.2f} errors/minute",
                    "high"
                )
            
            await asyncio.sleep(0.1)  # Small delay
```

### Alert Types

```python
# Common alert types
ALERT_TYPES = {
    "HIGH_ERROR_RATE": "Error rate exceeds threshold",
    "CIRCUIT_BREAKER_TRIGGERED": "Circuit breaker activated",
    "MEMORY_USAGE_HIGH": "Memory usage exceeds threshold",
    "SLOW_RESPONSE_TIME": "Response time exceeds threshold",
    "DATABASE_CONNECTION_FAILED": "Database connection issues",
    "MODEL_LOADING_FAILED": "AI model loading failures",
    "EXTERNAL_SERVICE_DOWN": "External service unavailable",
    "RATE_LIMIT_EXCEEDED": "Rate limit exceeded",
    "AUTHENTICATION_FAILURES": "Multiple authentication failures",
    "PERMISSION_DENIED": "Authorization failures"
}
```

## Middleware Stack

### MiddlewareStack Class

```python
class MiddlewareStack:
    """Stack of middleware for the application."""
    
    def __init__(self):
        self.error_tracker = ErrorTracker()
        self.middleware_stack = []
    
    def add_middleware(self, middleware_class: type, **kwargs):
        """Add middleware to the stack."""
        self.middleware_stack.append((middleware_class, kwargs))
    
    def create_middleware_stack(self, app: ASGIApp) -> ASGIApp:
        """Create the middleware stack."""
        # Add default middleware
        if not any(m[0] == StructuredLoggingMiddleware for m in self.middleware_stack):
            self.add_middleware(StructuredLoggingMiddleware, log_level="INFO")
        
        if not any(m[0] == ErrorHandlingMiddleware for m in self.middleware_stack):
            self.add_middleware(ErrorHandlingMiddleware, error_tracker=self.error_tracker)
        
        if not any(m[0] == PerformanceMonitoringMiddleware for m in self.middleware_stack):
            self.add_middleware(PerformanceMonitoringMiddleware)
        
        # Apply middleware in reverse order (last added is innermost)
        for middleware_class, kwargs in reversed(self.middleware_stack):
            app = middleware_class(app, **kwargs)
        
        return app
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return self.error_tracker.get_error_stats()
```

### FastAPI Integration

```python
def create_app_with_middleware() -> ASGIApp:
    """Create FastAPI app with comprehensive middleware."""
    from fastapi import FastAPI
    
    app = FastAPI(title="AI Video API with Middleware")
    
    # Create middleware stack
    middleware_stack = MiddlewareStack()
    
    # Add custom middleware if needed
    # middleware_stack.add_middleware(CustomMiddleware, custom_param="value")
    
    # Apply middleware stack
    app = middleware_stack.create_middleware_stack(app)
    
    # Store middleware stack for access in routes
    app.state.middleware_stack = middleware_stack
    
    # Add routes
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "timestamp": time.time()}
    
    @app.get("/errors/stats")
    async def get_error_stats():
        return middleware_stack.get_error_stats()
    
    @app.get("/performance/stats")
    async def get_performance_stats():
        return {"message": "Performance stats endpoint"}
    
    return app
```

## Best Practices

### 1. Error Categorization

```python
# ✅ GOOD: Proper error categorization
def _determine_error_type(self, exc: Exception) -> ErrorType:
    exc_type = type(exc)
    
    # Check direct type mapping first
    if exc_type in self.error_type_mapping:
        return self.error_type_mapping[exc_type]
    
    # Check error message patterns
    error_message = str(exc).lower()
    
    if "timeout" in error_message:
        return ErrorType.TIMEOUT
    elif "memory" in error_message:
        return ErrorType.MEMORY
    elif "database" in error_message:
        return ErrorType.DATABASE
    
    return ErrorType.UNKNOWN

# ❌ BAD: Generic error handling
def handle_error(self, exc: Exception):
    logger.error(f"Error occurred: {exc}")
    return {"error": "Internal server error"}
```

### 2. Circuit Breaker Configuration

```python
# ✅ GOOD: Appropriate circuit breaker thresholds
error_info = ErrorInfo(
    error_type=ErrorType.MODEL,
    severity=ErrorSeverity.HIGH,
    action=ErrorAction.ALERT,
    alert_threshold=3,        # Alert after 3 errors
    circuit_break_threshold=5  # Break circuit after 5 errors
)

# ❌ BAD: Too aggressive circuit breaker
error_info = ErrorInfo(
    error_type=ErrorType.MODEL,
    severity=ErrorSeverity.HIGH,
    action=ErrorAction.ALERT,
    alert_threshold=1,        # Too sensitive
    circuit_break_threshold=2  # Too aggressive
)
```

### 3. Performance Monitoring

```python
# ✅ GOOD: Comprehensive performance monitoring
def _check_performance_issues(self, metrics: PerformanceMetrics):
    # Check response time
    if metrics.duration > self.slow_request_threshold:
        logger.warning(f"Slow request: {metrics.duration:.2f}s")
    
    # Check memory usage
    if metrics.memory_delta > 100:
        logger.warning(f"High memory usage: {metrics.memory_delta:.2f}MB")
    
    # Check CPU usage
    if metrics.cpu_usage > 80:
        logger.warning(f"High CPU usage: {metrics.cpu_usage}%")

# ❌ BAD: No performance monitoring
def process_request(self, request):
    # No performance tracking
    return process_request_internal(request)
```

### 4. Structured Logging

```python
# ✅ GOOD: Structured logging with context
def _log_request_start(self, log_entry: RequestLog):
    log_data = {
        "event": "request_start",
        "request_id": log_entry.request_id,
        "method": log_entry.method,
        "url": log_entry.url,
        "client_ip": log_entry.client_ip,
        "user_id": log_entry.user_id,
        "video_id": log_entry.video_id,
        "timestamp": datetime.fromtimestamp(log_entry.start_time).isoformat()
    }
    
    self.logger.info(json.dumps(log_data))

# ❌ BAD: Unstructured logging
def log_request(self, request):
    logger.info(f"Request: {request.method} {request.url}")
```

### 5. Error Recovery

```python
# ✅ GOOD: Error recovery with retry and fallback
async def process_with_recovery(self, operation: str, video_id: str):
    try:
        # Try primary operation
        return await self.process_with_retry(operation, video_id)
    except Exception as exc:
        logger.warning(f"Primary operation failed: {exc}")
        
        try:
            # Try fallback operation
            return await self.process_with_retry("fallback_" + operation, video_id)
        except Exception as fallback_exc:
            logger.error(f"Both operations failed: {fallback_exc}")
            raise

# ❌ BAD: No error recovery
async def process_video(self, video_id: str):
    # No retry or fallback
    return await process_video_internal(video_id)
```

## Examples

### Complete Error Handling System

```python
class IntegratedErrorHandlingSystem:
    def __init__(self):
        self.error_tracker = ErrorTracker()
        self.performance_monitor = PerformanceMonitoringExample()
        self.error_recovery = ErrorRecoveryExample()
        self.alerting = AlertingExample()
    
    async def process_video_request(self, video_id: str, prompt: str, model_name: str):
        start_time = time.time()
        
        try:
            # Step 1: Load model with retry
            model_result = await self.error_recovery.process_with_retry(
                "load_model", 
                video_id
            )
            
            # Step 2: Generate video with fallback
            video_result = await self.error_recovery.process_with_fallback(
                "generate_video",
                "generate_video_fallback",
                video_id
            )
            
            # Step 3: Monitor performance
            processing_time = time.time() - start_time
            performance_result = await self.performance_monitor.monitor_video_processing(
                video_id, 
                processing_time
            )
            
            return {
                "video_id": video_id,
                "status": "completed",
                "model_loading": model_result,
                "video_generation": video_result,
                "performance": performance_result,
                "total_time": processing_time
            }
            
        except Exception as exc:
            # Record error
            error_info = ErrorInfo(
                error_type=ErrorType.SYSTEM,
                severity=ErrorSeverity.HIGH,
                action=ErrorAction.ALERT,
                alert_threshold=3
            )
            
            context = {
                "video_id": video_id,
                "prompt": prompt,
                "model_name": model_name,
                "error": str(exc)
            }
            
            self.error_tracker.record_error(ErrorType.SYSTEM, error_info, context)
            
            # Send alert
            self.alerting.send_alert(
                "VIDEO_PROCESSING_FAILED",
                f"Video processing failed for {video_id}: {exc}",
                "high"
            )
            
            raise
    
    def get_system_status(self):
        return {
            "error_stats": self.error_tracker.get_error_stats(),
            "alert_summary": self.alerting.get_alert_summary(),
            "circuit_breakers": self.error_tracker.circuit_breakers,
            "timestamp": time.time()
        }
```

### Testing Error Scenarios

```python
async def test_error_scenarios():
    app = create_ai_video_app_with_middleware()
    client = TestClient(app)
    
    # Test normal request
    response = client.get("/health")
    print(f"Health check: {response.status_code}")
    
    # Test validation error
    response = client.get("/videos/error_video")
    print(f"Validation error: {response.status_code}")
    
    # Test memory error
    response = client.get("/videos/memory_video")
    print(f"Memory error: {response.status_code}")
    
    # Test database error
    response = client.get("/videos/database_video")
    print(f"Database error: {response.status_code}")
    
    # Test model error
    response = client.get("/videos/model_video")
    print(f"Model error: {response.status_code}")
    
    # Get error statistics
    response = client.get("/errors/stats")
    print(f"Error stats: {response.json()}")
```

This comprehensive error middleware system provides robust error handling, monitoring, and recovery capabilities for AI Video applications, ensuring high availability and observability. 