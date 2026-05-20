# Middleware and Decorators - Implementation Summary

## Overview

This implementation provides comprehensive middleware and decorator implementations for centralized logging, metrics collection, and exception handling. The system follows the established patterns of guard clauses, early returns, structured logging, and modular design.

## Key Features

### 1. Centralized Logging Middleware
- **Structured Logging**: JSON and standard logging formats
- **Request/Response Tracking**: Complete HTTP request/response logging
- **Function Call Logging**: Detailed function execution logging
- **Context Information**: Request IDs, user IDs, correlation IDs
- **Performance Data**: Execution time and resource usage tracking
- **Exception Logging**: Comprehensive exception information

### 2. Metrics Collection Middleware
- **Multiple Backends**: Memory, Prometheus, Redis support
- **Request Metrics**: HTTP request/response metrics
- **Function Metrics**: Function call and performance metrics
- **Exception Metrics**: Exception tracking and categorization
- **Custom Metrics**: Extensible metric collection system
- **Real-time Monitoring**: Live metrics and statistics

### 3. Exception Handling Middleware
- **Custom Handlers**: Register custom exception handlers
- **Severity Levels**: Automatic severity determination
- **Context Preservation**: Maintain context during exception handling
- **Default Handlers**: Built-in handlers for common exceptions
- **Exception Reporting**: Configurable exception reporting
- **Stack Trace Analysis**: Detailed stack trace information

### 4. Performance Monitoring Middleware
- **Execution Time Tracking**: Precise execution time measurement
- **Slow Function Detection**: Automatic detection of slow functions
- **Resource Usage**: Memory and CPU usage monitoring
- **Performance Thresholds**: Configurable performance alerts
- **Performance Summary**: Statistical performance analysis
- **Async Support**: Performance monitoring for async functions

### 5. Decorators and Context Managers
- **Function Decorators**: Easy-to-use decorators for any function
- **Async Decorators**: Support for async functions
- **Performance Decorators**: Automatic performance monitoring
- **Exception Decorators**: Centralized exception handling
- **Context Managers**: Performance and logging context managers
- **Async Context Managers**: Async-compatible context managers

## Core Classes

### LoggingMiddleware
```python
class LoggingMiddleware:
    """Centralized logging middleware."""
    
    def log_request(self, request: Request, response: Response, execution_time: float):
        """Log HTTP request/response."""
        # Guard clause for disabled logging
        if not self.config.enable_logging:
            return
        
        # Extract context information
        request_id = request.headers.get(self.config.request_id_header)
        user_id = request.headers.get(self.config.user_id_header)
        
        # Create structured log entry
        log_entry = LogEntry(
            level=LogLevel.INFO,
            message=f"{request.method} {request.url.path} - {response.status_code}",
            request_id=request_id,
            user_id=user_id,
            extra_data={
                'method': request.method,
                'url': str(request.url),
                'status_code': response.status_code,
                'execution_time': execution_time
            }
        )
        
        # Happy path - log entry
        self._log_entry(log_entry)
```

### MetricsMiddleware
```python
class MetricsMiddleware:
    """Centralized metrics middleware."""
    
    def record_request_metric(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metric."""
        # Guard clause for disabled metrics
        if not self.config.enable_metrics:
            return
        
        # Create metric entry
        metric_entry = MetricEntry(
            name="http_request",
            value=1,
            metric_type=MetricType.COUNTER,
            labels={
                'method': method,
                'endpoint': endpoint,
                'status_code': str(status_code)
            }
        )
        
        # Happy path - record metric
        self._record_metric(metric_entry)
        
        # Record duration metric
        duration_metric = MetricEntry(
            name="http_request_duration",
            value=duration,
            metric_type=MetricType.HISTOGRAM,
            labels={'method': method, 'endpoint': endpoint}
        )
        
        self._record_metric(duration_metric)
```

### ExceptionHandlingMiddleware
```python
class ExceptionHandlingMiddleware:
    """Centralized exception handling middleware."""
    
    def handle_exception(self, exception: Exception, function_name: str, module_name: str,
                        line_number: int, context_data: Dict[str, Any] = None) -> ExceptionInfo:
        """Handle exception and return exception info."""
        # Guard clause for disabled exception handling
        if not self.config.enable_exception_handling:
            return None
        
        # Determine severity
        severity = self._determine_severity(exception)
        
        # Create exception info
        exception_info = ExceptionInfo(
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            severity=severity,
            function_name=function_name,
            module_name=module_name,
            line_number=line_number,
            stack_trace=traceback.format_exc(),
            context_data=context_data or {}
        )
        
        # Happy path - store and handle exception
        with self._lock:
            self.exceptions.append(exception_info)
        
        # Call custom handler if available
        handler = self._get_handler(exception)
        if handler:
            handler(exception_info)
        
        return exception_info
```

### MiddlewareManager
```python
class MiddlewareManager:
    """Centralized middleware manager."""
    
    def log_function_call(self, function_name: str, module_name: str, line_number: int,
                         args: tuple, kwargs: dict, result: Any, execution_time: float,
                         exception: Optional[Exception] = None):
        """Log function call with all middleware."""
        # Guard clauses for early returns
        if not self.config.enable_logging and not self.config.enable_metrics:
            return
        
        # Log function call
        if self.config.enable_logging:
            self.logging_middleware.log_function_call(
                function_name, module_name, line_number, args, kwargs, result, execution_time, exception
            )
        
        # Record metrics
        if self.config.enable_metrics:
            self.metrics_middleware.record_function_metric(
                function_name, module_name, execution_time, exception is None
            )
        
        # Record performance
        if self.config.enable_performance_monitoring:
            self.performance_middleware.record_performance(function_name, execution_time)
        
        # Handle exception
        if exception and self.config.enable_exception_handling:
            self.exception_middleware.handle_exception(
                exception, function_name, module_name, line_number
            )
```

## Design Patterns Applied

### 1. Middleware Pattern
- **Chain of Responsibility**: Middleware components in a chain
- **Separation of Concerns**: Each middleware handles specific functionality
- **Composability**: Middleware can be combined and configured
- **Extensibility**: Easy to add new middleware components

### 2. Decorator Pattern
- **Function Wrapping**: Wrap functions with additional behavior
- **Non-intrusive**: Add functionality without modifying original code
- **Composable**: Multiple decorators can be combined
- **Type Safety**: Preserve function signatures and types

### 3. Context Manager Pattern
- **Resource Management**: Automatic resource cleanup
- **Performance Monitoring**: Context-based performance tracking
- **Async Support**: Async context managers for async operations
- **Error Handling**: Automatic error handling in context

### 4. Registry Pattern
- **Exception Handlers**: Registry of custom exception handlers
- **Middleware Registration**: Dynamic middleware registration
- **Configuration Management**: Centralized configuration

### 5. Observer Pattern
- **Event Logging**: Log events as they occur
- **Metrics Collection**: Collect metrics during execution
- **Exception Reporting**: Report exceptions to multiple handlers

## Decorators

### Function Logging Decorator
```python
def log_function_call(middleware_manager: MiddlewareManager):
    """Decorator to log function calls."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function info
            function_name = func.__name__
            module_name = func.__module__
            line_number = inspect.getsourcelines(func)[1]
            
            # Record start time
            start_time = time.time()
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log successful execution
                middleware_manager.log_function_call(
                    function_name, module_name, line_number,
                    args, kwargs, result, execution_time
                )
                
                return result
            
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Log failed execution
                middleware_manager.log_function_call(
                    function_name, module_name, line_number,
                    args, kwargs, None, execution_time, e
                )
                
                raise
        
        return wrapper
    return decorator
```

### Performance Monitoring Decorator
```python
def monitor_performance(threshold: float = 1.0):
    """Decorator to monitor function performance."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log slow functions
                if execution_time > threshold:
                    logger.warning(f"Slow function: {func.__name__} took {execution_time:.2f}s")
                
                return result
            
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Function {func.__name__} failed after {execution_time:.2f}s: {e}")
                raise
        
        return wrapper
    return decorator
```

### Exception Handling Decorator
```python
def handle_exceptions(exception_handlers: Dict[Type[Exception], Callable] = None):
    """Decorator to handle exceptions."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Call custom handler if available
                if exception_handlers and type(e) in exception_handlers:
                    exception_handlers[type(e)](e)
                else:
                    # Default handling
                    logger.error(f"Exception in {func.__name__}: {e}")
                    raise
        
        return wrapper
    return decorator
```

## Context Managers

### Performance Context Manager
```python
@contextmanager
def performance_context(function_name: str, middleware_manager: MiddlewareManager):
    """Context manager for performance monitoring."""
    start_time = time.time()
    
    try:
        yield
    finally:
        execution_time = time.time() - start_time
        middleware_manager.performance_middleware.record_performance(
            function_name, execution_time
        )
```

### Async Performance Context Manager
```python
@asynccontextmanager
async def async_performance_context(function_name: str, middleware_manager: MiddlewareManager):
    """Async context manager for performance monitoring."""
    start_time = time.time()
    
    try:
        yield
    finally:
        execution_time = time.time() - start_time
        middleware_manager.performance_middleware.record_performance(
            function_name, execution_time
        )
```

## FastAPI Middleware

### HTTP Middleware
```python
class FastAPIMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for logging, metrics, and exception handling."""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Record start time
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            execution_time = time.time() - start_time
            
            # Log request
            self.middleware_manager.log_request(request, response, execution_time)
            
            return response
        
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Create error response
            error_response = Response(
                content=json.dumps({"error": str(e)}),
                status_code=500,
                media_type="application/json"
            )
            
            # Log error request
            self.middleware_manager.log_request(request, error_response, execution_time)
            
            return error_response
```

## Usage Examples

### Basic Middleware Usage
```python
# Create middleware configuration
config = MiddlewareConfig(
    enable_logging=True,
    enable_metrics=True,
    enable_exception_handling=True,
    enable_performance_monitoring=True,
    log_level=LogLevel.INFO,
    log_format="json"
)

# Create middleware manager
middleware_manager = MiddlewareManager(config)

# Use decorators
@log_function_call(middleware_manager)
@monitor_performance(threshold=0.1)
@handle_exceptions()
def example_function(x: int, y: int) -> int:
    if x < 0:
        raise ValueError("x must be positive")
    time.sleep(0.05)  # Simulate work
    return x + y

# Test function
result = example_function(5, 3)
```

### FastAPI Integration
```python
# Create FastAPI app with middleware
app = FastAPI()
app.add_middleware(FastAPIMiddleware, middleware_manager=middleware_manager)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/slow")
async def slow_endpoint():
    await asyncio.sleep(2)  # Simulate slow operation
    return {"message": "Slow response"}
```

### Context Manager Usage
```python
# Performance context manager
with performance_context("example_operation", middleware_manager):
    time.sleep(0.1)  # Simulate work
    print("Operation completed")

# Async context manager
async def async_operation():
    async with async_performance_context("async_operation", middleware_manager):
        await asyncio.sleep(0.1)  # Simulate async work
        print("Async operation completed")

asyncio.run(async_operation())
```

## Configuration Options

### MiddlewareConfig
```python
@dataclass
class MiddlewareConfig:
    enable_logging: bool = True
    enable_metrics: bool = True
    enable_exception_handling: bool = True
    enable_performance_monitoring: bool = True
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"
    metrics_backend: str = "memory"  # memory, prometheus, redis
    exception_reporting: bool = True
    performance_threshold: float = 1.0  # seconds
    request_id_header: str = "X-Request-ID"
    correlation_id_header: str = "X-Correlation-ID"
    user_id_header: str = "X-User-ID"
    session_id_header: str = "X-Session-ID"
```

## Performance Considerations

### 1. Minimal Overhead
- **Conditional Execution**: Only execute when enabled
- **Efficient Data Structures**: Use appropriate data structures
- **Lazy Evaluation**: Evaluate only when needed
- **Thread Safety**: Proper locking for concurrent access

### 2. Memory Management
- **Bounded Collections**: Prevent memory leaks
- **Weak References**: Use weak references where appropriate
- **Garbage Collection**: Proper cleanup of resources
- **Memory Monitoring**: Track memory usage

### 3. Async Support
- **Non-blocking Operations**: Async-compatible operations
- **Context Preservation**: Maintain context in async operations
- **Async Decorators**: Support for async functions
- **Async Context Managers**: Async context management

### 4. Scalability
- **Horizontal Scaling**: Support for distributed systems
- **Backend Integration**: Redis and Prometheus support
- **Load Balancing**: Compatible with load balancers
- **Caching**: Efficient caching strategies

## Security Features

### 1. Input Validation
- **Parameter Validation**: Validate all input parameters
- **Sanitization**: Clean and sanitize inputs
- **Type Safety**: Strong type checking
- **Boundary Checking**: Check input boundaries

### 2. Sensitive Data Handling
- **Data Masking**: Mask sensitive data in logs
- **Access Control**: Control access to sensitive information
- **Audit Trails**: Maintain audit trails
- **Compliance**: Support for compliance requirements

### 3. Error Handling
- **Secure Error Messages**: Don't expose sensitive information
- **Exception Sanitization**: Clean exception information
- **Error Reporting**: Secure error reporting
- **Fallback Mechanisms**: Graceful degradation

## Best Practices

### 1. Middleware Usage
- **Minimal Middleware**: Use only necessary middleware
- **Order Matters**: Consider middleware order
- **Configuration**: Use environment-based configuration
- **Testing**: Test middleware thoroughly

### 2. Decorator Usage
- **Composition**: Combine decorators effectively
- **Performance**: Consider decorator overhead
- **Readability**: Keep decorators simple
- **Documentation**: Document custom decorators

### 3. Context Managers
- **Resource Management**: Proper resource cleanup
- **Error Handling**: Handle errors in context
- **Async Support**: Use async context managers for async operations
- **Nesting**: Support nested context managers

### 4. Monitoring
- **Metrics Collection**: Collect relevant metrics
- **Alerting**: Set up appropriate alerts
- **Dashboard**: Create monitoring dashboards
- **Trends**: Monitor trends over time

## Integration Examples

### Django Integration
```python
from django.http import JsonResponse
from middleware_decorators_examples import MiddlewareManager, log_function_call

middleware_manager = MiddlewareManager(MiddlewareConfig())

@log_function_call(middleware_manager)
def django_view(request):
    return JsonResponse({"message": "Hello Django"})
```

### Flask Integration
```python
from flask import Flask, jsonify
from middleware_decorators_examples import MiddlewareManager, log_function_call

app = Flask(__name__)
middleware_manager = MiddlewareManager(MiddlewareConfig())

@app.route('/')
@log_function_call(middleware_manager)
def flask_view():
    return jsonify({"message": "Hello Flask"})
```

### Celery Integration
```python
from celery import Celery
from middleware_decorators_examples import MiddlewareManager, log_function_call

app = Celery('tasks')
middleware_manager = MiddlewareManager(MiddlewareConfig())

@app.task
@log_function_call(middleware_manager)
def celery_task(x, y):
    return x + y
```

## Conclusion

This implementation provides a robust, scalable, and secure foundation for centralized logging, metrics collection, and exception handling. The modular design, comprehensive features, and multiple integration options make it suitable for production use while maintaining flexibility and ease of use.

The system follows established patterns and best practices, ensuring maintainability, testability, and extensibility. The middleware and decorator approach provides clean separation of concerns while enabling powerful monitoring and debugging capabilities.

Key benefits:
- **Centralized Management**: All logging, metrics, and exception handling in one place
- **Flexible Configuration**: Easy to configure and customize
- **Multiple Interfaces**: Decorators, context managers, and middleware
- **Performance Optimized**: Minimal overhead with maximum functionality
- **Production Ready**: Comprehensive features for production environments
- **Extensible**: Easy to add new functionality and integrations 