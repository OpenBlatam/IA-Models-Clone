# 🚨 Error Monitoring Middleware Guide - Instagram Captions API v14.0

## 📋 Overview

This guide documents the comprehensive error monitoring middleware system implemented in v14.0, featuring advanced error handling, structured logging, real-time monitoring, alerting capabilities, and graceful error recovery.

## 🎯 **Error Monitoring Architecture**

### **1. Core Components**

#### **ErrorMonitor Class**
```python
class ErrorMonitor:
    """Advanced error monitoring and alerting system"""
    
    def __init__(self, max_errors: int = 1000, alert_threshold: int = 10):
        self.errors: deque = deque(maxlen=max_errors)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.category_counts: Dict[ErrorCategory, int] = defaultdict(int)
        self.priority_counts: Dict[ErrorPriority, int] = defaultdict(int)
        self.endpoint_errors: Dict[str, int] = defaultdict(int)
        self.alerts: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.total_requests = 0
        self.error_requests = 0
        self.avg_response_time = 0.0
```

#### **ErrorRecord Data Structure**
```python
@dataclass
class ErrorRecord:
    """Structured error record for monitoring"""
    timestamp: datetime
    request_id: str
    error_type: str
    error_category: ErrorCategory
    priority: ErrorPriority
    message: str
    details: Dict[str, Any]
    stack_trace: Optional[str]
    endpoint: str
    method: str
    client_ip: str
    user_agent: str
    response_time: float
    status_code: int
```

### **2. Error Categories**

```python
class ErrorCategory(Enum):
    """Error categories for monitoring and alerting"""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    AI_MODEL = "ai_model"
    CACHE = "cache"
    DATABASE = "database"
    EXTERNAL_SERVICE = "external_service"
    NETWORK = "network"
    SYSTEM = "system"
    UNKNOWN = "unknown"
```

### **3. Error Priority Levels**

```python
class ErrorPriority(Enum):
    """Error priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
```

## 🔧 **Middleware Implementation**

### **1. ErrorMonitoringMiddleware**

#### **Core Features**
- **Comprehensive Error Tracking**: Records all errors with full context
- **Performance Impact Analysis**: Tracks response times and error rates
- **Real-time Alerting**: Automatic alerts for critical issues
- **Structured Logging**: JSON-formatted logs with consistent structure
- **Error Categorization**: Automatic classification of error types

#### **Implementation**
```python
class ErrorMonitoringMiddleware(BaseHTTPMiddleware):
    """Comprehensive error monitoring middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID if not present
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        request.state.request_id = request_id
        
        # Extract request context
        start_time = time.perf_counter()
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        endpoint = f"{request.method} {request.url.path}"
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            response_time = time.perf_counter() - start_time
            
            # Record successful request
            self.error_monitor.record_request(response_time, False)
            
            # Add monitoring headers
            response.headers["X-Error-Rate"] = f"{self.error_monitor.error_requests / max(self.error_monitor.total_requests, 1):.3f}"
            response.headers["X-Avg-Response-Time"] = f"{self.error_monitor.avg_response_time:.3f}s"
            
            return response
            
        except Exception as e:
            # Handle unexpected errors
            response_time = time.perf_counter() - start_time
            self.error_monitor.record_request(response_time, True)
            
            # Create comprehensive error record
            error_record = self._create_error_record(
                error_type=type(e).__name__,
                error_category=self._categorize_error(e),
                priority=self._determine_priority(e),
                message=str(e),
                details={"exception_type": type(e).__name__},
                stack_trace=traceback.format_exc() if self.enable_detailed_logging else None,
                endpoint=endpoint,
                method=request.method,
                client_ip=client_ip,
                user_agent=user_agent,
                response_time=response_time,
                status_code=500,
                request_id=request_id
            )
            
            self.error_monitor.record_error(error_record)
            
            # Return structured error response
            return JSONResponse(
                status_code=500,
                content=create_error_response(
                    error_code="INTERNAL_ERROR",
                    message="An unexpected error occurred",
                    status_code=500,
                    details={"error_type": type(e).__name__, "request_id": request_id},
                    request_id=request_id,
                    path=str(request.url.path),
                    method=request.method
                ).model_dump(),
                headers={"X-Request-ID": request_id}
            )
```

### **2. ErrorRecoveryMiddleware**

#### **Core Features**
- **Graceful Degradation**: Automatic fallback strategies
- **Error Recovery**: Attempts to recover from specific error types
- **Service Continuity**: Ensures API remains functional during issues
- **Recovery Strategies**: Category-specific recovery mechanisms

#### **Implementation**
```python
class ErrorRecoveryMiddleware(BaseHTTPMiddleware):
    """Middleware for error recovery and graceful degradation"""
    
    def __init__(self, app):
        super().__init__(app)
        self.recovery_strategies = {
            "ai_model": self._recover_ai_model,
            "cache": self._recover_cache,
            "database": self._recover_database
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            # Try to recover based on error type
            recovery_result = await self._attempt_recovery(e, request)
            if recovery_result:
                return recovery_result
            
            # If recovery fails, re-raise the error
            raise
    
    async def _attempt_recovery(self, error: Exception, request: Request) -> Optional[Response]:
        """Attempt to recover from error"""
        error_category = self._categorize_error(error)
        
        if error_category.value in self.recovery_strategies:
            try:
                return await self.recovery_strategies[error_category.value](error, request)
            except Exception as recovery_error:
                logger.error(f"Recovery failed: {recovery_error}")
        
        return None
```

## 📊 **Error Categorization & Prioritization**

### **1. Automatic Error Categorization**

```python
def _categorize_error(self, error: Exception) -> ErrorCategory:
    """Categorize error based on type and content"""
    error_type = type(error).__name__
    error_message = str(error).lower()
    
    # HTTP exceptions
    if isinstance(error, HTTPException):
        if error.status_code == 401:
            return ErrorCategory.AUTHENTICATION
        elif error.status_code == 403:
            return ErrorCategory.AUTHORIZATION
        elif error.status_code == 429:
            return ErrorCategory.RATE_LIMIT
        elif error.status_code == 400:
            return ErrorCategory.VALIDATION
        else:
            return ErrorCategory.SYSTEM
    
    # AI-related errors
    if isinstance(error, (AIGenerationError, ModelLoadingError)):
        return ErrorCategory.AI_MODEL
    
    # Cache errors
    if isinstance(error, CacheError):
        return ErrorCategory.CACHE
    
    # Network-related errors
    if any(keyword in error_message for keyword in ["connection", "timeout", "network"]):
        return ErrorCategory.NETWORK
    
    # External service errors
    if any(keyword in error_message for keyword in ["api", "service", "external"]):
        return ErrorCategory.EXTERNAL_SERVICE
    
    # Database errors
    if any(keyword in error_message for keyword in ["database", "sql", "db"]):
        return ErrorCategory.DATABASE
    
    return ErrorCategory.UNKNOWN
```

### **2. Priority Determination**

```python
def _determine_priority(self, error: Exception) -> ErrorPriority:
    """Determine error priority based on type and impact"""
    error_type = type(error).__name__
    error_message = str(error).lower()
    
    # Critical errors
    if isinstance(error, (AIGenerationError, ModelLoadingError)):
        return ErrorPriority.CRITICAL
    
    # High priority errors
    if any(keyword in error_message for keyword in ["memory", "disk", "database", "connection"]):
        return ErrorPriority.HIGH
    
    # Medium priority errors
    if isinstance(error, HTTPException) and error.status_code >= 500:
        return ErrorPriority.HIGH
    elif isinstance(error, HTTPException) and error.status_code >= 400:
        return ErrorPriority.MEDIUM
    
    return ErrorPriority.LOW
```

## 🚨 **Alerting System**

### **1. Alert Rules Configuration**

```python
self.alert_rules = {
    "error_rate_threshold": 0.05,  # 5% error rate
    "consecutive_errors_threshold": 5,
    "critical_error_threshold": 1,
    "response_time_threshold": 5.0  # 5 seconds
}
```

### **2. Alert Types**

#### **High Error Rate Alert**
```python
# Check error rate
error_rate = self.error_requests / max(self.total_requests, 1)
if error_rate > self.alert_rules["error_rate_threshold"]:
    self._create_alert(
        "HIGH_ERROR_RATE",
        f"Error rate is {error_rate:.2%} (threshold: {self.alert_rules['error_rate_threshold']:.2%})",
        error_record
    )
```

#### **Consecutive Errors Alert**
```python
# Check consecutive errors
recent_errors = [e for e in self.errors if 
                (current_time - e.timestamp.timestamp()) < 60]  # Last minute
if len(recent_errors) >= self.alert_rules["consecutive_errors_threshold"]:
    self._create_alert(
        "CONSECUTIVE_ERRORS",
        f"{len(recent_errors)} consecutive errors in the last minute",
        error_record
    )
```

#### **Critical Error Alert**
```python
# Check critical errors
if error_record.priority == ErrorPriority.CRITICAL:
    self._create_alert(
        "CRITICAL_ERROR",
        f"Critical error: {error_record.message}",
        error_record
    )
```

#### **Slow Response Alert**
```python
# Check response time
if error_record.response_time > self.alert_rules["response_time_threshold"]:
    self._create_alert(
        "SLOW_RESPONSE",
        f"Slow response time: {error_record.response_time:.2f}s",
        error_record
    )
```

### **3. Alert Creation**

```python
def _create_alert(self, alert_type: str, message: str, error_record: ErrorRecord):
    """Create and log an alert"""
    alert = {
        "timestamp": datetime.now(timezone.utc),
        "alert_type": alert_type,
        "message": message,
        "error_record": asdict(error_record),
        "severity": "HIGH" if alert_type in ["CRITICAL_ERROR", "HIGH_ERROR_RATE"] else "MEDIUM"
    }
    
    self.alerts.append(alert)
    
    # Log alert
    logger.critical(f"🚨 ALERT [{alert_type}]: {message}")
    logger.critical(f"Error details: {error_record.message}")
    logger.critical(f"Endpoint: {error_record.endpoint}")
    logger.critical(f"Request ID: {error_record.request_id}")
```

## 📈 **Error Statistics & Monitoring**

### **1. Comprehensive Error Statistics**

```python
def get_error_stats(self) -> Dict[str, Any]:
    """Get comprehensive error statistics"""
    return {
        "total_requests": self.total_requests,
        "error_requests": self.error_requests,
        "error_rate": self.error_requests / max(self.total_requests, 1),
        "avg_response_time": self.avg_response_time,
        "uptime": time.time() - self.start_time,
        "error_counts": dict(self.error_counts),
        "category_counts": {cat.value: count for cat, count in self.category_counts.items()},
        "priority_counts": {pri.value: count for pri, count in self.priority_counts.items()},
        "endpoint_errors": dict(self.endpoint_errors),
        "recent_alerts": len([a for a in self.alerts if 
                            (time.time() - a["timestamp"].timestamp()) < 3600]),  # Last hour
        "total_alerts": len(self.alerts)
    }
```

### **2. Recent Error Retrieval**

```python
def get_recent_errors(self, minutes: int = 60) -> List[Dict[str, Any]]:
    """Get errors from the last N minutes"""
    cutoff_time = time.time() - (minutes * 60)
    recent_errors = [
        asdict(error) for error in self.errors 
        if error.timestamp.timestamp() > cutoff_time
    ]
    return recent_errors
```

## 🔧 **Integration with FastAPI**

### **1. Middleware Stack Configuration**

```python
# Add comprehensive middleware stack
app.add_middleware(ErrorMonitoringMiddleware, enable_detailed_logging=True)
app.add_middleware(ErrorRecoveryMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(PerformanceMiddleware)
app.add_middleware(SecurityMiddleware)
app.add_middleware(CompressionMiddleware)
app.add_middleware(CachingMiddleware)
```

### **2. Error Statistics Endpoint**

```python
@app.get("/api/v14/error-stats", response_model=Dict[str, Any])
async def get_error_stats(
    api_key: str = Depends(validate_api_key_dependency)
):
    """Get comprehensive error statistics and monitoring data"""
    try:
        error_monitor = get_error_monitor()
        error_stats = error_monitor.get_error_stats()
        recent_errors = error_monitor.get_recent_errors(minutes=60)
        
        return {
            "error_statistics": error_stats,
            "recent_errors": recent_errors[:50],  # Limit to last 50 errors
            "alerts": error_monitor.alerts[-10:],  # Last 10 alerts
            "timestamp": time.time(),
            "api_version": "14.0.0"
        }
        
    except Exception as e:
        logger.error(f"Failed to get error stats: {e}")
        raise AIGenerationError(
            message="Failed to retrieve error statistics",
            details={"error": str(e)},
            path="/api/v14/error-stats",
            method="GET"
        )
```

### **3. Utility Functions**

```python
async def log_error_with_context(
    error: Exception,
    request: Request,
    response_time: float,
    additional_context: Optional[Dict[str, Any]] = None
):
    """Utility function to log errors with full context"""
    request_id = getattr(request.state, "request_id", "unknown")
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    endpoint = f"{request.method} {request.url.path}"
    
    error_record = ErrorRecord(
        timestamp=datetime.now(timezone.utc),
        request_id=request_id,
        error_type=type(error).__name__,
        error_category=ErrorCategory.UNKNOWN,
        priority=ErrorPriority.MEDIUM,
        message=str(error),
        details=additional_context or {},
        stack_trace=traceback.format_exc(),
        endpoint=endpoint,
        method=request.method,
        client_ip=client_ip,
        user_agent=user_agent,
        response_time=response_time,
        status_code=500
    )
    
    error_monitor.record_error(error_record)
```

## 📊 **Error Response Examples**

### **1. Structured Error Response**

```json
{
    "error": true,
    "error_code": "INTERNAL_ERROR",
    "message": "An unexpected error occurred",
    "details": {
        "error_type": "AIGenerationError",
        "request_id": "req-123456",
        "endpoint": "/api/v14/generate"
    },
    "timestamp": "2024-01-15T10:30:00.123Z",
    "request_id": "req-123456",
    "path": "/api/v14/generate",
    "method": "POST",
    "status_code": 500
}
```

### **2. Error Statistics Response**

```json
{
    "error_statistics": {
        "total_requests": 1500,
        "error_requests": 45,
        "error_rate": 0.03,
        "avg_response_time": 0.125,
        "uptime": 86400.0,
        "error_counts": {
            "AIGenerationError": 15,
            "ValidationError": 20,
            "CacheError": 10
        },
        "category_counts": {
            "ai_model": 15,
            "validation": 20,
            "cache": 10
        },
        "priority_counts": {
            "critical": 5,
            "high": 15,
            "medium": 20,
            "low": 5
        },
        "endpoint_errors": {
            "POST /api/v14/generate": 25,
            "POST /api/v14/batch": 20
        },
        "recent_alerts": 3,
        "total_alerts": 12
    },
    "recent_errors": [
        {
            "timestamp": "2024-01-15T10:30:00.123Z",
            "request_id": "req-123456",
            "error_type": "AIGenerationError",
            "category": "ai_model",
            "priority": "critical",
            "message": "AI model generation failed",
            "endpoint": "POST /api/v14/generate",
            "response_time": 2.5,
            "status_code": 500
        }
    ],
    "alerts": [
        {
            "timestamp": "2024-01-15T10:30:00.123Z",
            "alert_type": "CRITICAL_ERROR",
            "message": "Critical error: AI model generation failed",
            "severity": "HIGH"
        }
    ],
    "timestamp": 1705312200.123,
    "api_version": "14.0.0"
}
```

## 🎯 **Best Practices**

### **1. Error Handling Patterns**

#### **Do:**
```python
# Use error monitoring in endpoints
@app.post("/api/v14/generate")
async def generate_caption(request: OptimizedRequest):
    start_time = time.time()
    try:
        response = await optimized_engine.generate_caption(request)
        return response
    except Exception as e:
        response_time = time.time() - start_time
        await log_error_with_context(e, request, response_time)
        raise

# Use error recovery strategies
async def _recover_ai_model(self, error: Exception, request: Request):
    """Recover from AI model errors"""
    logger.warning("Attempting AI model recovery...")
    # Implement fallback logic
    return fallback_response
```

#### **Don't:**
```python
# Don't ignore errors
try:
    result = await some_operation()
except Exception:
    pass  # ❌ Silent failure

# Don't log without context
except Exception as e:
    logger.error(str(e))  # ❌ Missing context
```

### **2. Monitoring Configuration**

```python
# Configure alert thresholds appropriately
alert_rules = {
    "error_rate_threshold": 0.05,  # 5% for production
    "consecutive_errors_threshold": 5,
    "critical_error_threshold": 1,
    "response_time_threshold": 5.0
}

# Enable detailed logging in development
enable_detailed_logging = True  # Set to False in production
```

### **3. Performance Considerations**

```python
# Use efficient data structures
self.errors: deque = deque(maxlen=1000)  # Fixed size to prevent memory leaks

# Limit error history
recent_errors = recent_errors[:50]  # Limit to last 50 errors

# Use async operations for recovery
async def _attempt_recovery(self, error: Exception, request: Request):
    # Async recovery strategies
    pass
```

## 📈 **Benefits**

### **✅ Comprehensive Error Tracking**
- **Full Context**: Request ID, endpoint, client info, timing
- **Error Categorization**: Automatic classification by type
- **Priority Levels**: Critical, high, medium, low priority handling
- **Performance Impact**: Response time and error rate tracking

### **✅ Real-time Alerting**
- **Automatic Detection**: Error rate, consecutive errors, critical issues
- **Immediate Notification**: Real-time alerts for critical problems
- **Configurable Thresholds**: Adjustable alert rules
- **Severity Levels**: High and medium priority alerts

### **✅ Graceful Error Recovery**
- **Fallback Strategies**: Automatic recovery for different error types
- **Service Continuity**: API remains functional during issues
- **Recovery Mechanisms**: Category-specific recovery strategies
- **Graceful Degradation**: Reduced functionality instead of complete failure

### **✅ Structured Logging**
- **JSON Format**: Consistent, parseable log structure
- **Request Correlation**: Error tracking with request IDs
- **Performance Metrics**: Response time and error rate logging
- **Context Preservation**: Full error context for debugging

### **✅ Monitoring & Analytics**
- **Error Statistics**: Comprehensive error metrics
- **Trend Analysis**: Error patterns over time
- **Endpoint Analysis**: Per-endpoint error tracking
- **Alert History**: Historical alert tracking

---

This comprehensive error monitoring middleware system ensures robust error handling, real-time monitoring, and graceful recovery capabilities for the Instagram Captions API v14.0, providing maximum reliability and observability. 