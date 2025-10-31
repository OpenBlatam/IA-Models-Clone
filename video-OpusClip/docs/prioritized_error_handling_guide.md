# Prioritized Error Handling Guide

## Overview

The Video-OpusClip system implements a comprehensive, prioritized error handling system that categorizes errors by severity and impact, ensuring critical issues are addressed immediately while maintaining system stability.

## Error Priority Levels

### 1. CRITICAL (1000-1999) - System-Breaking Errors
**Immediate attention required - may cause system failure**

- `SYSTEM_CRASH` (1001): Complete system failure
- `DATABASE_CONNECTION_LOST` (1002): Database connectivity issues
- `REDIS_CONNECTION_LOST` (1003): Cache connectivity issues
- `GPU_MEMORY_EXHAUSTED` (1004): GPU memory completely full
- `DISK_SPACE_CRITICAL` (1005): Disk space below 5%
- `MODEL_LOADING_FAILED` (1006): AI models cannot be loaded
- `PIPELINE_INITIALIZATION_FAILED` (1007): Processing pipeline failure
- `CRITICAL_SERVICE_UNAVAILABLE` (1008): Essential service down

### 2. HIGH (2000-2999) - Processing Failures
**Affects user experience - processing cannot complete**

- `VIDEO_PROCESSING_FAILED` (2001): Video processing errors
- `LANGCHAIN_PROCESSING_FAILED` (2002): AI analysis failures
- `VIRAL_ANALYSIS_FAILED` (2003): Viral score calculation errors
- `BATCH_PROCESSING_FAILED` (2004): Batch operation failures
- `MODEL_INFERENCE_FAILED` (2007): AI model inference errors
- `VIDEO_ENCODING_FAILED` (2008): Video encoding issues
- `AUDIO_EXTRACTION_FAILED` (2009): Audio processing errors
- `FRAME_EXTRACTION_FAILED` (2010): Frame extraction issues

### 3. MEDIUM (4000-5999) - Resource & Configuration Issues
**System degradation - may affect performance**

- `INSUFFICIENT_MEMORY` (4001): Low system memory
- `GPU_NOT_AVAILABLE` (4002): GPU not accessible
- `DISK_SPACE_FULL` (4003): Disk space low
- `RATE_LIMIT_EXCEEDED` (4004): API rate limits hit
- `TIMEOUT_ERROR` (4005): Operation timeouts
- `CPU_OVERLOADED` (4006): High CPU usage
- `MISSING_CONFIG` (5001): Configuration missing
- `INVALID_CONFIG` (5002): Invalid configuration
- `API_KEY_MISSING` (5005): Missing API credentials

### 4. LOW (6000-7999) - Validation & Security Issues
**User input issues - can be resolved by user**

- `INVALID_YOUTUBE_URL` (6001): Invalid YouTube URLs
- `INVALID_LANGUAGE_CODE` (6002): Unsupported languages
- `INVALID_CLIP_LENGTH` (6003): Invalid video lengths
- `UNAUTHORIZED_ACCESS` (7001): Authentication failures
- `INVALID_TOKEN` (7002): Invalid authentication tokens
- `MALICIOUS_INPUT_DETECTED` (7004): Security threats detected

## Critical Error Handling

### System Health Monitoring

```python
from validation import validate_system_health, validate_gpu_health

# Check system resources before processing
validate_system_health()
validate_gpu_health()
```

### Critical Error Thresholds

The system automatically alerts when critical errors exceed thresholds:

- **Critical errors**: Alert after 5 occurrences
- **High priority errors**: Alert after 20 occurrences  
- **Medium priority errors**: Alert after 50 occurrences

### Automatic Alerting

```python
# Critical errors trigger immediate alerts
if critical_error_count >= error_thresholds['critical']:
    _send_critical_alert(error, request_id)
```

## Security Error Handling

### Malicious Input Detection

The system detects and blocks malicious input patterns:

```python
malicious_patterns = [
    "javascript:", "data:", "vbscript:", "file://", "ftp://",
    "eval(", "exec(", "system(", "shell_exec("
]

# Check for malicious patterns in URLs
for pattern in malicious_patterns:
    if pattern in url.lower():
        raise SecurityError(f"Malicious pattern detected: {pattern}")
```

### Threat Response

- **Injection attempts**: Automatic IP blocking
- **Malicious input**: Request rejection with security logging
- **Excessive requests**: Rate limiting and temporary blocking

## Edge Case Validation

### URL Validation Edge Cases

```python
def validate_youtube_url(url: str, field_name: str = "youtube_url") -> None:
    # Edge case: Empty or None URL
    if not url: 
        raise create_validation_error("YouTube URL is required", field_name, url)
    
    # Edge case: Wrong data type
    if not isinstance(url, str): 
        raise create_validation_error("YouTube URL must be a string", field_name, url)
    
    # Edge case: Extremely long URL (potential DoS)
    if len(url) > 2048: 
        raise create_validation_error("YouTube URL too long (max 2048 characters)", field_name, url)
    
    # Edge case: Malicious patterns
    malicious_patterns = ["javascript:", "data:", "eval("]
    for pattern in malicious_patterns:
        if pattern in url.lower():
            raise create_validation_error(f"Malicious URL pattern detected: {pattern}", field_name, url)
```

### Clip Length Edge Cases

```python
def validate_clip_length(length: int, field_name: str = "clip_length") -> None:
    # Edge case: Negative values
    if length < 0: 
        raise create_validation_error("Clip length cannot be negative", field_name, length)
    
    # Edge case: Zero length
    if length == 0: 
        raise create_validation_error("Clip length cannot be zero", field_name, length)
    
    # Edge case: Unrealistic values (potential overflow)
    if length > 86400:  # 24 hours
        raise create_validation_error("Clip length exceeds maximum allowed duration", field_name, length)
```

### Batch Size Edge Cases

```python
def validate_batch_size(size: int, field_name: str = "batch_size") -> None:
    # Edge case: Negative values
    if size < 0: 
        raise create_validation_error("Batch size cannot be negative", field_name, size)
    
    # Edge case: Zero batch size
    if size == 0: 
        raise create_validation_error("Batch size cannot be zero", field_name, size)
    
    # Edge case: Unrealistic batch sizes (potential DoS)
    if size > 1000: 
        raise create_validation_error("Batch size exceeds maximum allowed limit", field_name, size)
```

## System Health Monitoring

### Resource Monitoring

```python
def check_system_resources() -> Dict[str, Any]:
    """Check system resources and return health status."""
    health_status = {
        "cpu_usage": psutil.cpu_percent(interval=1),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "network_status": "healthy"
    }
    
    # Check for critical resource levels
    if health_status["memory_usage"] > 90:
        health_status["memory_critical"] = True
        health_status["warnings"] = health_status.get("warnings", []) + ["High memory usage"]
    
    if health_status["disk_usage"] > 95:
        health_status["disk_critical"] = True
        health_status["warnings"] = health_status.get("warnings", []) + ["Critical disk space"]
    
    return health_status
```

### GPU Health Monitoring

```python
def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU availability and status."""
    gpu_status = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None
    }
    
    if gpu_status["available"]:
        gpu_status["device_name"] = torch.cuda.get_device_name(0)
        gpu_status["memory_allocated"] = torch.cuda.memory_allocated(0) / 1024**3  # GB
        gpu_status["memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        # Check for critical GPU memory usage
        if gpu_status["memory_allocated"] / gpu_status["memory_total"] > 0.95:
            gpu_status["memory_critical"] = True
            gpu_status["warnings"] = gpu_status.get("warnings", []) + ["Critical GPU memory usage"]
    
    return gpu_status
```

## Error Pattern Analysis

The system automatically analyzes unknown errors to categorize them:

```python
def _analyze_error_pattern(self, error: Exception) -> str:
    """Analyze error patterns to help with debugging."""
    error_str = str(error).lower()
    
    if "memory" in error_str or "out of memory" in error_str:
        return "memory_related"
    elif "timeout" in error_str or "timed out" in error_str:
        return "timeout_related"
    elif "connection" in error_str or "network" in error_str:
        return "network_related"
    elif "permission" in error_str or "access" in error_str:
        return "permission_related"
    else:
        return "unknown_pattern"
```

## Configuration Error Handling

### Fallback Strategies

```python
def _get_fallback_config(self, config_key: str) -> Optional[Dict[str, Any]]:
    """Get fallback configuration for critical settings."""
    fallback_configs = {
        "model_path": "/default/models/",
        "api_key": "default_key",
        "max_workers": 1,
        "timeout": 30
    }
    return fallback_configs.get(config_key)
```

### Configuration Validation

```python
def validate_configuration_error(error: ConfigurationError, request_id: Optional[str] = None) -> ErrorResponse:
    """Handle configuration errors with fallback strategies."""
    # Try to use default configuration
    fallback_config = _get_fallback_config(error.details.get("config_key"))
    if fallback_config:
        logger.info("Using fallback configuration", config_key=error.details.get("config_key"))
    
    return ErrorResponse(
        error_code=error.error_code,
        message=error.message,
        details=error.details,
        timestamp=_get_timestamp(),
        request_id=request_id
    )
```

## API Integration

### Prioritized Request Processing

```python
@app.post("/api/v1/video/process")
async def process_video(request: VideoClipRequest, req: Request = None):
    """Process video with prioritized error handling."""
    request_id = getattr(req.state, 'request_id', None) if req else None
    
    try:
        # PRIORITY 1: System health validation (critical)
        validate_system_health()
        validate_gpu_health()
        
        # PRIORITY 2: Security validation (high)
        if any(pattern in request.youtube_url.lower() for pattern in ["javascript:", "data:", "eval("]):
            raise SecurityError("Malicious input detected in YouTube URL", "malicious_input")
        
        # PRIORITY 3: Request validation (medium)
        validate_video_request_data(request)
        
        # PRIORITY 4: Processing with monitoring
        response = processor.process_video(request)
        
        return response
        
    except CriticalSystemError as e:
        # Handle critical system errors
        error_response = error_handler.handle_critical_system_error(e, request_id)
        return JSONResponse(status_code=500, content=error_response.to_dict())
    
    except SecurityError as e:
        # Handle security errors
        error_response = error_handler.handle_security_error(e, request_id)
        return JSONResponse(status_code=403, content=error_response.to_dict())
```

### Enhanced Health Check

```python
@app.get("/health")
async def health_check():
    """Enhanced health check with system monitoring."""
    try:
        # Check system resources
        system_health = check_system_resources()
        gpu_health = check_gpu_availability()
        
        # Determine overall health status
        health_status = "healthy"
        warnings = []
        
        # Check for critical conditions
        if system_health.get("memory_critical"):
            health_status = "degraded"
            warnings.append("Critical memory usage")
        
        if system_health.get("disk_critical"):
            health_status = "critical"
            warnings.append("Critical disk space")
        
        return {
            "status": health_status,
            "system_health": system_health,
            "gpu_health": gpu_health,
            "warnings": warnings,
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": "Health check failed",
            "timestamp": time.time()
        }
```

## Testing Edge Cases

### Critical Error Tests

```python
def test_critical_system_error_handler(self):
    """Test critical system error handler with alerting."""
    handler = ErrorHandler()
    error = create_critical_system_error(
        "GPU memory exhausted",
        "gpu",
        ErrorCode.GPU_MEMORY_EXHAUSTED
    )
    
    with patch.object(handler, '_send_critical_alert') as mock_alert:
        response = handler.handle_critical_system_error(error, "test-request-id")
        
        assert response.error_code == ErrorCode.GPU_MEMORY_EXHAUSTED
        assert "GPU memory exhausted" in response.message
        assert response.request_id == "test-request-id"
        
        # Should not alert on first error
        mock_alert.assert_not_called()
```

### Security Error Tests

```python
def test_malicious_url_validation(self):
    """Test validation against malicious URL patterns."""
    malicious_urls = [
        "javascript:alert('xss')",
        "data:text/html,<script>alert('xss')</script>",
        "file:///etc/passwd",
        "eval('malicious_code')"
    ]
    
    for url in malicious_urls:
        with pytest.raises(ValidationError) as exc_info:
            validate_youtube_url(url)
        
        assert "Malicious URL pattern detected" in str(exc_info.value)
```

### Edge Case Tests

```python
def test_extremely_long_url(self):
    """Test validation of extremely long URLs."""
    long_url = "https://youtube.com/watch?v=" + "a" * 2000
    
    with pytest.raises(ValidationError) as exc_info:
        validate_youtube_url(long_url)
    
    assert "too long" in str(exc_info.value)

def test_negative_clip_length(self):
    """Test validation of negative clip lengths."""
    with pytest.raises(ValidationError) as exc_info:
        validate_clip_length(-5)
    
    assert "cannot be negative" in str(exc_info.value)
```

## Best Practices

### 1. Always Check System Health First
```python
# Before any processing operation
validate_system_health()
validate_gpu_health()
```

### 2. Validate Input Security
```python
# Check for malicious patterns in all user inputs
if any(pattern in user_input.lower() for pattern in malicious_patterns):
    raise SecurityError("Malicious input detected", "malicious_input")
```

### 3. Use Appropriate Error Codes
```python
# Use specific error codes for better monitoring
raise create_critical_system_error("GPU memory exhausted", "gpu", ErrorCode.GPU_MEMORY_EXHAUSTED)
```

### 4. Implement Fallback Strategies
```python
# Always have fallback configurations for critical settings
fallback_config = _get_fallback_config(config_key)
if fallback_config:
    logger.info("Using fallback configuration")
```

### 5. Monitor Error Patterns
```python
# Analyze unknown errors to improve system resilience
error_pattern = _analyze_error_pattern(error)
logger.info("Error pattern detected", pattern=error_pattern)
```

## Monitoring and Alerting

### Error Thresholds
- **Critical errors**: Alert after 5 occurrences
- **High priority errors**: Alert after 20 occurrences
- **Medium priority errors**: Alert after 50 occurrences

### Alert Integration
The system can integrate with:
- PagerDuty for critical alerts
- Slack for team notifications
- Email for detailed reports
- Custom webhooks for specific actions

### Metrics to Monitor
- Error rate by priority level
- System resource usage
- GPU memory utilization
- Response times by endpoint
- Security threat detection rate

## Troubleshooting

### Common Critical Errors

1. **GPU Memory Exhausted**
   - Check GPU memory usage
   - Reduce batch sizes
   - Clear GPU cache

2. **Disk Space Critical**
   - Clean up temporary files
   - Archive old videos
   - Increase disk space

3. **Database Connection Lost**
   - Check database service
   - Verify network connectivity
   - Review connection pool settings

### Debugging Unknown Errors

1. Check error pattern analysis
2. Review system logs with request ID
3. Monitor system resources
4. Test with minimal configuration

## Migration Guide

### From Basic Error Handling

1. Replace generic exceptions with specific error types
2. Add system health checks to critical endpoints
3. Implement security validation for all inputs
4. Add error pattern analysis for unknown errors
5. Set up monitoring and alerting thresholds

### Configuration Updates

```python
# Old configuration
app = FastAPI(title="Video Processing API")

# New configuration with error handling
app = FastAPI(
    title="Video Processing API",
    description="Advanced video processing with prioritized error handling",
    version="3.0.0"
)

# Add error handlers
@app.exception_handler(CriticalSystemError)
async def critical_system_exception_handler(request: Request, exc: CriticalSystemError):
    # Handle critical errors
    pass
```

This prioritized error handling system ensures that critical issues are addressed immediately while maintaining system stability and providing comprehensive monitoring and alerting capabilities. 