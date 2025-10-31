# Error Handling and Edge Case Priorities

## 1. Critical Security Errors (P0)

```python
# Authentication/Authorization failures
async def validate_user_permissions(user_id: str, required_role: str) -> bool:
    try:
        user = await get_user(user_id)
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        if required_role not in user.roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return True
    except Exception as e:
        logger.error(f"Auth error for user {user_id}: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

# Input sanitization and injection prevention
def sanitize_sql_input(user_input: str) -> str:
    dangerous_patterns = ["'", ";", "--", "/*", "*/", "xp_", "sp_"]
    for pattern in dangerous_patterns:
        if pattern in user_input.lower():
            raise HTTPException(status_code=400, detail="Invalid input detected")
    return user_input.strip()
```

## 2. Data Validation Errors (P1)

```python
# Required field validation
class PostRequest(BaseModel):
    content: str = Field(..., min_length=10, max_length=3000)
    post_type: str = Field(..., regex="^(educational|promotional|personal|industry)$")
    
    @validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        if len(v.split()) < 5:
            raise ValueError('Content must have at least 5 words')
        return v.strip()

# Business logic validation
async def validate_post_business_rules(post: PostRequest, user_id: str) -> None:
    # Check user's daily post limit
    daily_posts = await get_user_daily_posts(user_id)
    if daily_posts >= 5:
        raise HTTPException(status_code=429, detail="Daily post limit exceeded")
    
    # Check for duplicate content
    if await is_duplicate_content(post.content, user_id):
        raise HTTPException(status_code=400, detail="Duplicate content detected")
```

## 3. Rate Limiting and Resource Protection (P1)

```python
from fastapi import HTTPException, status
import asyncio
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self):
        self.requests = {}
    
    async def check_rate_limit(self, user_id: str, limit: int = 100, window: int = 3600) -> bool:
        now = datetime.now()
        user_requests = self.requests.get(user_id, [])
        
        # Remove old requests outside window
        user_requests = [req for req in user_requests if now - req < timedelta(seconds=window)]
        
        if len(user_requests) >= limit:
            return False
        
        user_requests.append(now)
        self.requests[user_id] = user_requests
        return True

async def rate_limit_middleware(user_id: str = Depends(get_current_user)):
    rate_limiter = RateLimiter()
    if not await rate_limiter.check_rate_limit(user_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later."
        )
    return user_id
```

## 4. Database and External Service Errors (P2)

```python
import asyncio
from typing import Optional

async def safe_database_operation(operation_func, *args, **kwargs):
    """Wrapper for database operations with retry logic"""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            return await operation_func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Database operation failed after {max_retries} attempts: {e}")
                raise HTTPException(status_code=500, detail="Database operation failed")
            
            logger.warning(f"Database operation attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(retry_delay * (2 ** attempt))

async def safe_external_api_call(url: str, timeout: int = 10) -> Optional[dict]:
    """Safe external API calls with timeout and error handling"""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        logger.error(f"External API timeout: {url}")
        raise HTTPException(status_code=503, detail="External service timeout")
    except httpx.HTTPStatusError as e:
        logger.error(f"External API error {e.response.status_code}: {url}")
        raise HTTPException(status_code=502, detail="External service error")
    except Exception as e:
        logger.error(f"External API unexpected error: {e}")
        raise HTTPException(status_code=500, detail="External service unavailable")
```

## 5. Edge Cases and Boundary Conditions (P2)

```python
# Handle empty or null responses
async def get_user_posts(user_id: str) -> List[dict]:
    posts = await safe_database_operation(fetch_user_posts, user_id)
    return posts if posts else []

# Handle malformed data
def validate_json_structure(data: dict, required_fields: List[str]) -> dict:
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise HTTPException(
            status_code=400, 
            detail=f"Missing required fields: {', '.join(missing_fields)}"
        )
    return data

# Handle concurrent operations
import asyncio
from asyncio import Lock

class ConcurrentOperationHandler:
    def __init__(self):
        self.locks = {}
    
    async def execute_with_lock(self, key: str, operation_func, *args, **kwargs):
        if key not in self.locks:
            self.locks[key] = Lock()
        
        async with self.locks[key]:
            return await operation_func(*args, **kwargs)

# Handle large payloads
async def validate_payload_size(request: Request, max_size: int = 1024 * 1024) -> None:
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > max_size:
        raise HTTPException(
            status_code=413, 
            detail=f"Payload too large. Maximum size: {max_size} bytes"
        )
```

## 6. Performance and Resource Management (P3)

```python
# Memory management for large operations
async def process_large_dataset(data: List[dict], batch_size: int = 1000) -> List[dict]:
    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_result = await process_batch(batch)
        results.extend(batch_result)
        
        # Allow other tasks to run
        await asyncio.sleep(0)
    
    return results

# Timeout handling for long-running operations
async def execute_with_timeout(operation_func, timeout: int = 30, *args, **kwargs):
    try:
        return await asyncio.wait_for(operation_func(*args, **kwargs), timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(f"Operation timed out after {timeout} seconds")
        raise HTTPException(status_code=408, detail="Operation timed out")

# Resource cleanup
class ResourceManager:
    def __init__(self):
        self.resources = []
    
    async def add_resource(self, resource):
        self.resources.append(resource)
    
    async def cleanup(self):
        for resource in self.resources:
            try:
                await resource.close()
            except Exception as e:
                logger.error(f"Error cleaning up resource: {e}")
        self.resources.clear()
```

## 7. Monitoring and Alerting (P3)

```python
import logging
from datetime import datetime
from typing import Dict, Any

class ErrorMonitor:
    def __init__(self):
        self.error_counts = {}
        self.alert_threshold = 10
    
    def log_error(self, error_type: str, context: Dict[str, Any] = None):
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        
        self.error_counts[error_type] += 1
        
        # Log error with context
        logger.error(f"Error {error_type}: {context}")
        
        # Alert if threshold exceeded
        if self.error_counts[error_type] >= self.alert_threshold:
            self.send_alert(error_type, self.error_counts[error_type])
    
    def send_alert(self, error_type: str, count: int):
        # Send alert to monitoring system
        alert_message = f"High error rate detected: {error_type} - {count} occurrences"
        logger.critical(alert_message)
        # Implement actual alerting logic here

# Usage in error handlers
error_monitor = ErrorMonitor()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_monitor.log_error(
        type(exc).__name__,
        {
            "url": str(request.url),
            "method": request.method,
            "client_ip": request.client.host,
            "timestamp": datetime.now().isoformat()
        }
    )
    raise exc
```

## 8. Graceful Degradation (P3)

```python
# Fallback mechanisms
async def get_post_with_fallback(post_id: str) -> dict:
    try:
        # Try primary data source
        post = await get_post_from_database(post_id)
        return post
    except Exception as e:
        logger.warning(f"Primary data source failed: {e}")
        try:
            # Try cache
            post = await get_post_from_cache(post_id)
            return post
        except Exception as e:
            logger.warning(f"Cache also failed: {e}")
            # Return minimal response
            return {"id": post_id, "status": "unavailable", "error": "Data temporarily unavailable"}

# Circuit breaker pattern
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout):
                self.state = "HALF_OPEN"
            else:
                raise HTTPException(status_code=503, detail="Service temporarily unavailable")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e
```

## 9. Edge Case Testing Scenarios

### Security Testing
```python
# Test SQL injection attempts
test_cases = [
    "'; DROP TABLE users; --",
    "1' OR '1'='1",
    "admin'--",
    "'; EXEC xp_cmdshell('dir'); --",
    "UNION SELECT * FROM users",
]

# Test XSS attempts
xss_test_cases = [
    "<script>alert('xss')</script>",
    "javascript:alert('xss')",
    "<img src=x onerror=alert('xss')>",
    "<iframe src='javascript:alert(\"xss\")'></iframe>",
]

# Test file upload attacks
file_test_cases = [
    ("malicious.php", "application/x-php"),
    ("script.js", "application/javascript"),
    ("large_file.txt", "text/plain", 10 * 1024 * 1024),  # 10MB
]
```

### Data Validation Testing
```python
# Test content validation edge cases
content_test_cases = [
    "",  # Empty content
    "a",  # Too short
    "a" * 3001,  # Too long
    "   ",  # Only whitespace
    "a" * 1000,  # Excessive repetition
    "word1 word2 word3 word4",  # Too few words
    "word " * 501,  # Too many words
]

# Test hashtag validation
hashtag_test_cases = [
    [],  # Empty list
    ["#"] * 31,  # Too many hashtags
    ["#a" * 26],  # Too long hashtag
    ["#invalid-hashtag"],  # Invalid characters
    ["duplicate", "duplicate"],  # Duplicates
]
```

### Rate Limiting Testing
```python
# Test rate limiting scenarios
async def test_rate_limiting():
    rate_limiter = RateLimiter()
    user_id = "test_user"
    
    # Test normal usage
    for i in range(50):
        assert await rate_limiter.check_rate_limit(user_id, limit=100)
    
    # Test limit exceeded
    for i in range(51):
        if i < 100:
            assert await rate_limiter.check_rate_limit(user_id, limit=100)
        else:
            assert not await rate_limiter.check_rate_limit(user_id, limit=100)
    
    # Test window expiration
    await asyncio.sleep(1)  # Wait for window to expire
    assert await rate_limiter.check_rate_limit(user_id, limit=100)
```

### Database Error Testing
```python
# Test database error scenarios
async def test_database_errors():
    # Test connection failure
    try:
        await safe_database_operation(failing_operation)
    except HTTPException as e:
        assert e.status_code == 500
    
    # Test retry logic
    retry_count = 0
    async def failing_then_succeeding_operation():
        nonlocal retry_count
        retry_count += 1
        if retry_count < 3:
            raise Exception("Database error")
        return "success"
    
    result = await safe_database_operation(failing_then_succeeding_operation)
    assert result == "success"
    assert retry_count == 3
```

### Performance Testing
```python
# Test performance edge cases
async def test_performance_scenarios():
    # Test large payload handling
    large_data = [{"id": i, "content": f"content_{i}"} for i in range(5000)]
    result = await handle_large_payload(large_data, batch_size=1000)
    assert len(result) == 5000
    
    # Test timeout handling
    async def slow_operation():
        await asyncio.sleep(2)
        return "result"
    
    try:
        await execute_with_timeout(slow_operation, timeout=1)
    except HTTPException as e:
        assert e.status_code == 408
```

## 10. Error Response Examples

### Standard Error Responses
```python
# 400 Bad Request
{
    "error": "validation_error",
    "detail": "Content must have at least 5 words",
    "field": "content",
    "timestamp": "2024-01-15T10:30:00Z"
}

# 401 Unauthorized
{
    "error": "authentication_error",
    "detail": "Invalid credentials",
    "timestamp": "2024-01-15T10:30:00Z"
}

# 429 Too Many Requests
{
    "error": "rate_limit_exceeded",
    "detail": "Rate limit exceeded. Try again later.",
    "retry_after": 3600,
    "timestamp": "2024-01-15T10:30:00Z"
}

# 500 Internal Server Error
{
    "error": "internal_error",
    "detail": "An unexpected error occurred",
    "request_id": "req_123456789",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

## Priority Summary

**P0 (Critical):**
- Authentication/Authorization failures
- Input sanitization and security
- Data corruption prevention

**P1 (High):**
- Data validation errors
- Rate limiting violations
- Business rule violations

**P2 (Medium):**
- Database connection issues
- External service failures
- Edge cases and boundary conditions

**P3 (Low):**
- Performance degradation
- Resource management
- Monitoring and alerting
- Graceful degradation 