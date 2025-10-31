# Middleware System Guide

A comprehensive guide for using middleware for handling unexpected errors, logging, and error monitoring in the HeyGen AI FastAPI application.

## 🎯 Overview

This guide covers:
- **Error Handling Middleware**: Comprehensive error handling with recovery strategies
- **Logging Middleware**: Structured logging with performance monitoring and security
- **Error Monitoring Middleware**: Real-time error tracking, alerting, and analysis
- **Integration Patterns**: How to combine all middleware components
- **Best Practices**: Guidelines for effective middleware usage

## 📋 Table of Contents

1. [Error Handling Middleware](#error-handling-middleware)
2. [Logging Middleware](#logging-middleware)
3. [Error Monitoring Middleware](#error-monitoring-middleware)
4. [Integration Guide](#integration-guide)
5. [Best Practices](#best-practices)
6. [Monitoring and Alerting](#monitoring-and-alerting)
7. [Performance Considerations](#performance-considerations)

## 🚨 Error Handling Middleware

### Overview

The Error Handling Middleware provides comprehensive error handling with:
- **Error Classification**: Automatic classification of different error types
- **Error Recovery**: Attempts to recover from errors using fallback strategies
- **Error Monitoring**: Tracks error patterns and metrics
- **Structured Logging**: Detailed error logging with context

### Basic Usage

```python
from fastapi import FastAPI
from api.middleware.error_handling_middleware import (
    ErrorHandlingMiddleware, create_error_handling_middleware
)

app = FastAPI()

# Create error handling middleware
error_middleware = create_error_handling_middleware(
    enable_recovery=True,
    enable_monitoring=True,
    log_request_body=False,
    log_headers=False
)

# Add middleware to app
app.add_middleware(ErrorHandlingMiddleware, **error_middleware.__dict__)
```

### Advanced Configuration

```python
from api.middleware.error_handling_middleware import (
    ErrorHandlingMiddleware, ErrorClassifier, ErrorMonitor, ErrorRecovery
)

# Custom error classifier
classifier = ErrorClassifier()

# Custom error monitor with alerting
monitor = ErrorMonitor()
monitor.alert_thresholds = {
    "critical": 1,    # Alert immediately
    "high": 5,        # Alert after 5 errors
    "medium": 20,     # Alert after 20 errors
    "low": 100        # Alert after 100 errors
}

# Custom recovery strategies
recovery = ErrorRecovery()

# Create middleware with custom components
middleware = ErrorHandlingMiddleware(
    app=None,
    enable_recovery=True,
    enable_monitoring=True,
    log_request_body=True,
    log_headers=True
)
```

### Error Recovery Examples

```python
# Database error recovery
async def database_recovery_strategy(exception, context):
    """Custom database recovery strategy."""
    try:
        # Try to reconnect to database
        await database.reconnect()
        return {"status": "recovered", "action": "database_reconnected"}
    except Exception:
        # Fall back to cached data
        cached_data = await cache.get("fallback_data")
        return {"status": "degraded", "action": "using_cached_data"}

# External service recovery
async def external_service_recovery_strategy(exception, context):
    """Custom external service recovery strategy."""
    try:
        # Try alternative service
        result = await alternative_service.call()
        return {"status": "recovered", "action": "alternative_service"}
    except Exception:
        # Return degraded response
        return {"status": "degraded", "action": "degraded_response"}

# Register custom recovery strategies
recovery.recovery_strategies[ErrorType.DATABASE_ERROR] = database_recovery_strategy
recovery.recovery_strategies[ErrorType.EXTERNAL_SERVICE_ERROR] = external_service_recovery_strategy
```

## 📊 Logging Middleware

### Overview

The Logging Middleware provides comprehensive logging with:
- **Request/Response Logging**: Detailed logging of all requests and responses
- **Performance Monitoring**: Track response times and performance metrics
- **Security Monitoring**: Detect and log security issues
- **Business Logic Logging**: Log business events and user actions
- **Audit Logging**: Compliance and security audit trails

### Basic Usage

```python
from api.middleware.logging_middleware import (
    LoggingMiddleware, create_logging_middleware
)

# Create logging middleware
logging_middleware = create_logging_middleware(
    log_request_body=False,
    log_response_body=False,
    log_headers=False,
    enable_performance_monitoring=True,
    enable_security_monitoring=True,
    enable_business_logging=True,
    enable_audit_logging=True
)

# Add middleware to app
app.add_middleware(LoggingMiddleware, **logging_middleware.__dict__)
```

### Structured Logging Setup

```python
from api.middleware.logging_middleware import LoggingUtilities

# Setup structured logging
LoggingUtilities.setup_structured_logging(
    log_level="INFO",
    log_format="json",
    include_timestamp=True
)

# Example log output
{
    "timestamp": "2024-01-01T12:00:00Z",
    "level": "info",
    "logger": "api.middleware.logging_middleware",
    "event": "Request received",
    "request_id": "123e4567-e89b-12d3-a456-426614174000",
    "method": "POST",
    "url": "/videos",
    "endpoint": "/videos",
    "ip_address": "192.168.1.1",
    "user_agent": "Mozilla/5.0..."
}
```

### Performance Monitoring

```python
from api.middleware.logging_middleware import PerformanceMonitor

# Create performance monitor
performance_monitor = PerformanceMonitor()
performance_monitor.slow_request_threshold_ms = 1000  # 1 second

# Get performance statistics
stats = performance_monitor.get_performance_stats()
print(f"Average response time: {stats['average_duration_ms']}ms")
print(f"Slow requests: {stats['slow_requests']}")
print(f"Error requests: {stats['error_requests']}")
```

### Security Monitoring

```python
from api.middleware.logging_middleware import SecurityMonitor

# Create security monitor
security_monitor = SecurityMonitor()
security_monitor.max_requests_per_minute = 100

# Check security for request
security_result = await security_monitor.check_security(request, context)

if security_result["issues_found"] > 0:
    logger.warning(
        "Security issues detected",
        issues=security_result["issues"]
    )
```

### Business Logic Logging

```python
from api.middleware.logging_middleware import BusinessLogger

# Create business logger
business_logger = BusinessLogger()

# Log video creation
await business_logger.log_video_creation(
    video_id="vid_123",
    template_id="template_1",
    script_length=500,
    context=log_context,
    user_id="user_456"
)

# Log user actions
await business_logger.log_user_action(
    action="update",
    resource_type="video",
    resource_id="vid_123",
    context=log_context,
    user_id="user_456"
)
```

### Audit Logging

```python
from api.middleware.logging_middleware import AuditLogger

# Create audit logger
audit_logger = AuditLogger()

# Log authentication
await audit_logger.log_authentication(
    success=True,
    user_id="user_456",
    context=log_context
)

# Log authorization
await audit_logger.log_authorization(
    resource="/videos/123",
    action="update",
    granted=True,
    context=log_context,
    user_id="user_456"
)
```

## 🔍 Error Monitoring Middleware

### Overview

The Error Monitoring Middleware provides real-time error tracking with:
- **Error Metrics Collection**: Track error rates, patterns, and trends
- **Alerting System**: Real-time alerts for critical errors
- **Error Analysis**: Pattern recognition and root cause analysis
- **Recovery Strategies**: Automatic error recovery attempts

### Basic Usage

```python
from api.middleware.error_monitoring_middleware import (
    ErrorMonitoringMiddleware, create_error_monitoring_middleware
)

# Create error monitoring middleware
monitoring_middleware = create_error_monitoring_middleware(
    enable_metrics=True,
    enable_alerting=True,
    enable_analysis=True,
    enable_recovery=True,
    retention_hours=24
)

# Add middleware to app
app.add_middleware(ErrorMonitoringMiddleware, **monitoring_middleware.__dict__)
```

### Error Metrics Collection

```python
from api.middleware.error_monitoring_middleware import ErrorMetricsCollector

# Create metrics collector
metrics_collector = ErrorMetricsCollector(retention_hours=24)

# Get current metrics
metrics = metrics_collector.get_metrics()
print(f"Total errors: {metrics['total_errors']}")
print(f"Recent errors: {metrics['recent_errors']}")
print(f"Error rate: {metrics['error_rate_per_minute']} per minute")

# Get error trends
trends = metrics_collector.get_error_trends(hours=1)
print(f"Hourly counts: {trends['hourly_counts']}")
print(f"Average errors per hour: {trends['average_errors_per_hour']}")
```

### Alerting System

```python
from api.middleware.error_monitoring_middleware import (
    ErrorAlertingSystem, AlertLevel
)

# Create alerting system
alerting_system = ErrorAlertingSystem()

# Add custom alert rule
def critical_error_condition(error_event):
    """Check if error is critical."""
    return (
        error_event.severity == ErrorSeverity.CRITICAL or
        error_event.error_type == ErrorType.DATABASE_ERROR
    )

alerting_system.add_alert_rule(
    name="critical_errors",
    condition=critical_error_condition,
    alert_level=AlertLevel.CRITICAL,
    message_template="Critical error: {error_type} at {endpoint}",
    cooldown_minutes=1
)

# Add alert channel (e.g., Slack)
async def slack_alert_channel(alert):
    """Send alert to Slack."""
    message = {
        "text": f"🚨 {alert['message']}",
        "attachments": [{
            "fields": [
                {"title": "Error Type", "value": alert['error_event']['error_type']},
                {"title": "Endpoint", "value": alert['error_event']['endpoint']},
                {"title": "User ID", "value": alert['error_event']['user_id'] or "Unknown"}
            ]
        }]
    }
    
    # Send to Slack webhook
    await send_slack_message(message)

alerting_system.add_alert_channel("slack", slack_alert_channel)
```

### Error Analysis

```python
from api.middleware.error_monitoring_middleware import ErrorAnalysisEngine

# Create analysis engine
analysis_engine = ErrorAnalysisEngine()

# Get error insights
insights = analysis_engine.get_error_insights()
print(f"Total patterns: {insights['total_patterns']}")
print(f"Most common errors: {insights['most_common_errors']}")
print(f"Recommendations: {insights['recommendations']}")
```

### Recovery Strategies

```python
from api.middleware.error_monitoring_middleware import (
    ErrorRecoveryStrategies, ErrorType
)

# Create recovery strategies
recovery_strategies = ErrorRecoveryStrategies()

# Custom database recovery
async def custom_database_recovery(error_event):
    """Custom database recovery strategy."""
    try:
        # Try to reconnect
        await database.reconnect()
        return {"status": "recovered", "action": "database_reconnected"}
    except Exception:
        # Use read replica
        await database.use_read_replica()
        return {"status": "degraded", "action": "using_read_replica"}

# Register custom recovery strategy
recovery_strategies.recovery_strategies[ErrorType.DATABASE_ERROR] = custom_database_recovery
```

## 🔗 Integration Guide

### Complete Middleware Setup

```python
from fastapi import FastAPI
from api.middleware.error_handling_middleware import create_error_handling_middleware
from api.middleware.logging_middleware import create_logging_middleware
from api.middleware.error_monitoring_middleware import create_error_monitoring_middleware

def create_app() -> FastAPI:
    """Create FastAPI application with comprehensive middleware."""
    app = FastAPI(
        title="HeyGen AI API",
        description="AI-powered video creation API",
        version="1.0.0"
    )
    
    # Add error handling middleware
    error_middleware = create_error_handling_middleware(
        enable_recovery=True,
        enable_monitoring=True,
        log_request_body=False,
        log_headers=False
    )
    app.add_middleware(ErrorHandlingMiddleware, **error_middleware.__dict__)
    
    # Add logging middleware
    logging_middleware = create_logging_middleware(
        log_request_body=False,
        log_response_body=False,
        log_headers=False,
        enable_performance_monitoring=True,
        enable_security_monitoring=True,
        enable_business_logging=True,
        enable_audit_logging=True
    )
    app.add_middleware(LoggingMiddleware, **logging_middleware.__dict__)
    
    # Add error monitoring middleware
    monitoring_middleware = create_error_monitoring_middleware(
        enable_metrics=True,
        enable_alerting=True,
        enable_analysis=True,
        enable_recovery=True,
        retention_hours=24
    )
    app.add_middleware(ErrorMonitoringMiddleware, **monitoring_middleware.__dict__)
    
    return app

app = create_app()
```

### Middleware Order

The order of middleware is important:

1. **Error Handling Middleware** (first - catches all errors)
2. **Logging Middleware** (second - logs requests and responses)
3. **Error Monitoring Middleware** (third - monitors and analyzes errors)

### Environment-Specific Configuration

```python
import os
from api.middleware.logging_middleware import LoggingUtilities

def configure_middleware_for_environment():
    """Configure middleware based on environment."""
    environment = os.getenv("ENVIRONMENT", "development")
    
    if environment == "production":
        # Production configuration
        LoggingUtilities.setup_structured_logging(
            log_level="INFO",
            log_format="json",
            include_timestamp=True
        )
        
        return {
            "error_handling": {
                "enable_recovery": True,
                "enable_monitoring": True,
                "log_request_body": False,
                "log_headers": False
            },
            "logging": {
                "log_request_body": False,
                "log_response_body": False,
                "log_headers": False,
                "enable_performance_monitoring": True,
                "enable_security_monitoring": True,
                "enable_business_logging": True,
                "enable_audit_logging": True
            },
            "monitoring": {
                "enable_metrics": True,
                "enable_alerting": True,
                "enable_analysis": True,
                "enable_recovery": True,
                "retention_hours": 24
            }
        }
    
    elif environment == "development":
        # Development configuration
        LoggingUtilities.setup_structured_logging(
            log_level="DEBUG",
            log_format="console",
            include_timestamp=True
        )
        
        return {
            "error_handling": {
                "enable_recovery": False,
                "enable_monitoring": True,
                "log_request_body": True,
                "log_headers": True
            },
            "logging": {
                "log_request_body": True,
                "log_response_body": True,
                "log_headers": True,
                "enable_performance_monitoring": True,
                "enable_security_monitoring": True,
                "enable_business_logging": True,
                "enable_audit_logging": True
            },
            "monitoring": {
                "enable_metrics": True,
                "enable_alerting": False,
                "enable_analysis": True,
                "enable_recovery": False,
                "retention_hours": 1
            }
        }
    
    else:
        # Default configuration
        return {
            "error_handling": {"enable_recovery": True, "enable_monitoring": True},
            "logging": {"enable_performance_monitoring": True},
            "monitoring": {"enable_metrics": True, "enable_alerting": True}
        }
```

## 🏆 Best Practices

### 1. Middleware Configuration

```python
# ✅ Good: Environment-specific configuration
config = configure_middleware_for_environment()
app.add_middleware(ErrorHandlingMiddleware, **config["error_handling"])

# ❌ Bad: Hard-coded configuration
app.add_middleware(ErrorHandlingMiddleware, enable_recovery=True, enable_monitoring=True)
```

### 2. Error Handling

```python
# ✅ Good: Use specific error types
from api.exceptions.http_exceptions import ValidationError, DatabaseError

if not user_data.email:
    raise ValidationError(
        message="Email is required",
        details=[{"field": "email", "message": "Email cannot be empty"}]
    )

# ❌ Bad: Generic error handling
if not user_data.email:
    raise HTTPException(status_code=400, detail="Bad request")
```

### 3. Logging

```python
# ✅ Good: Structured logging with context
logger.info(
    "Video created successfully",
    video_id=video.id,
    user_id=user.id,
    template_id=template.id,
    duration_ms=processing_time
)

# ❌ Bad: Unstructured logging
print(f"Video {video.id} created by user {user.id}")
```

### 4. Performance Monitoring

```python
# ✅ Good: Monitor specific endpoints
@router.post("/videos")
async def create_video(video_data: VideoCreateRequest):
    start_time = time.time()
    try:
        video = await video_service.create_video(video_data)
        duration = (time.time() - start_time) * 1000
        
        # Log performance
        logger.info(
            "Video creation completed",
            duration_ms=duration,
            video_id=video.id
        )
        
        return video
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        logger.error(
            "Video creation failed",
            duration_ms=duration,
            error=str(e)
        )
        raise

# ❌ Bad: No performance monitoring
@router.post("/videos")
async def create_video(video_data: VideoCreateRequest):
    return await video_service.create_video(video_data)
```

### 5. Security Monitoring

```python
# ✅ Good: Monitor for security issues
async def check_security_issues(request: Request):
    security_issues = []
    
    # Check for suspicious patterns
    if "<script>" in str(request.url):
        security_issues.append("XSS attempt detected")
    
    # Check rate limiting
    if await is_rate_limited(request.client.host):
        security_issues.append("Rate limit exceeded")
    
    if security_issues:
        logger.warning(
            "Security issues detected",
            issues=security_issues,
            ip_address=request.client.host
        )

# ❌ Bad: No security monitoring
# No security checks implemented
```

## 📊 Monitoring and Alerting

### Metrics Dashboard

```python
from api.middleware.error_monitoring_middleware import ErrorMonitoringEndpoints

# Create monitoring endpoints
monitoring_endpoints = ErrorMonitoringEndpoints(middleware)

@router.get("/metrics/errors")
async def get_error_metrics():
    """Get error metrics."""
    return monitoring_endpoints.get_metrics()

@router.get("/metrics/trends")
async def get_error_trends(hours: int = 1):
    """Get error trends."""
    return monitoring_endpoints.get_trends(hours)

@router.get("/metrics/insights")
async def get_error_insights():
    """Get error insights."""
    return monitoring_endpoints.get_insights()

@router.get("/metrics/alerts")
async def get_alerts(hours: int = 24):
    """Get alert history."""
    return monitoring_endpoints.get_alerts(hours)
```

### Alert Channels

```python
# Slack alerting
async def slack_alert(alert):
    """Send alert to Slack."""
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if webhook_url:
        message = {
            "text": f"🚨 {alert['message']}",
            "attachments": [{
                "color": "danger" if alert["alert_level"] == "critical" else "warning",
                "fields": [
                    {"title": "Error Type", "value": alert["error_event"]["error_type"]},
                    {"title": "Endpoint", "value": alert["error_event"]["endpoint"]},
                    {"title": "User ID", "value": alert["error_event"]["user_id"] or "Unknown"}
                ]
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            await session.post(webhook_url, json=message)

# Email alerting
async def email_alert(alert):
    """Send alert via email."""
    if alert["alert_level"] == "critical":
        # Send immediate email for critical alerts
        await send_critical_alert_email(alert)

# Add alert channels
monitoring_endpoints.add_alert_channel("slack", slack_alert)
monitoring_endpoints.add_alert_channel("email", email_alert)
```

### Health Checks

```python
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    # Check database connectivity
    try:
        await database.ping()
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"
    
    # Check cache connectivity
    try:
        await cache.ping()
        cache_status = "healthy"
    except Exception:
        cache_status = "unhealthy"
    
    # Check external services
    try:
        await heygen_api.ping()
        api_status = "healthy"
    except Exception:
        api_status = "unhealthy"
    
    overall_status = "healthy" if all([
        db_status == "healthy",
        cache_status == "healthy",
        api_status == "healthy"
    ]) else "unhealthy"
    
    return {
        "status": overall_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "database": db_status,
            "cache": cache_status,
            "heygen_api": api_status
        }
    }
```

## ⚡ Performance Considerations

### 1. Middleware Performance

```python
# ✅ Good: Efficient middleware
class EfficientLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = structlog.get_logger()
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Only log essential information
        response = await call_next(request)
        
        duration = time.time() - start_time
        if duration > 1.0:  # Only log slow requests
            self.logger.warning(
                "Slow request",
                duration=duration,
                path=request.url.path
            )
        
        return response

# ❌ Bad: Inefficient middleware
class InefficientLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Log everything (expensive)
        await self.log_full_request(request)
        
        response = await call_next(request)
        
        # Log everything (expensive)
        await self.log_full_response(response)
        
        return response
```

### 2. Async Operations

```python
# ✅ Good: Async logging
async def log_async(event_data):
    """Async logging operation."""
    await logger.ainfo("Event occurred", **event_data)

# ❌ Bad: Blocking logging
def log_sync(event_data):
    """Blocking logging operation."""
    logger.info("Event occurred", **event_data)
```

### 3. Memory Management

```python
# ✅ Good: Limited history
class LimitedHistoryMonitor:
    def __init__(self, max_history=1000):
        self.history = deque(maxlen=max_history)

# ❌ Bad: Unlimited history
class UnlimitedHistoryMonitor:
    def __init__(self):
        self.history = []  # Can grow indefinitely
```

### 4. Caching

```python
# ✅ Good: Cache expensive operations
class CachedErrorClassifier:
    def __init__(self):
        self.cache = {}
    
    def classify_error(self, exception):
        cache_key = f"{type(exception).__name__}:{str(exception)[:100]}"
        
        if cache_key not in self.cache:
            self.cache[cache_key] = self._classify_error(exception)
        
        return self.cache[cache_key]

# ❌ Bad: No caching
class UncachedErrorClassifier:
    def classify_error(self, exception):
        return self._classify_error(exception)  # Expensive operation every time
```

## 📚 Additional Resources

- [FastAPI Middleware](https://fastapi.tiangolo.com/tutorial/middleware/)
- [Structured Logging](https://structlog.readthedocs.io/)
- [Error Handling Best Practices](https://docs.python.org/3/tutorial/errors.html)
- [Monitoring and Observability](https://opentelemetry.io/)

## 🚀 Next Steps

1. **Implement the middleware system** in your FastAPI application
2. **Configure environment-specific settings** for development and production
3. **Set up monitoring and alerting** channels (Slack, email, etc.)
4. **Create custom recovery strategies** for your specific error scenarios
5. **Monitor performance** and optimize based on metrics
6. **Set up dashboards** for error tracking and analysis
7. **Document error patterns** and create runbooks for common issues

This comprehensive middleware system provides your HeyGen AI API with:
- **Robust error handling** with automatic recovery
- **Comprehensive logging** with performance and security monitoring
- **Real-time error tracking** with alerting and analysis
- **Production-ready patterns** for scalable error management
- **Flexible configuration** for different environments

The system is designed to handle all error scenarios while providing clear insights into system health, performance, and security. 