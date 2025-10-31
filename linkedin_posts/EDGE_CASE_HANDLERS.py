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

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from fastapi import HTTPException, Request, status
from pydantic import BaseModel, Field, validator
import httpx
import json
import re
from typing import Any, List, Dict, Optional
"""
Edge Case Handlers for LinkedIn Posts System
Comprehensive handling of edge cases and error scenarios
"""


logger = logging.getLogger(__name__)

# ============================================================================
# CRITICAL SECURITY EDGE CASES (P0)
# ============================================================================

class SecurityValidator:
    """Handles security-related edge cases"""
    
    @staticmethod
    def validate_sql_injection(user_input: str) -> str:
        """Prevent SQL injection attacks"""
        dangerous_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
            r"(\b(UNION|OR|AND)\b\s+\d+)",
            r"(--|/\*|\*/|xp_|sp_)",
            r"(\b(WAITFOR|DELAY)\b)",
            r"(\b(SLEEP|BENCHMARK)\b)",
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                logger.warning(f"Potential SQL injection detected: {user_input}")
                raise HTTPException(
                    status_code=400,
                    detail="Invalid input detected"
                )
        
        return user_input.strip()
    
    @staticmethod
    def validate_xss_attack(content: str) -> str:
        """Prevent XSS attacks"""
        xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                logger.warning(f"Potential XSS attack detected: {content}")
                raise HTTPException(
                    status_code=400,
                    detail="Invalid content detected"
                )
        
        return content
    
    @staticmethod
    async def validate_file_upload(filename: str, content_type: str, max_size: int = 5 * 1024 * 1024) -> bool:
        """Validate file uploads"""
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.pdf', '.doc', '.docx'}
        allowed_types = {
            'image/jpeg', 'image/png', 'image/gif', 
            'application/pdf', 'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        }
        
        file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
        if f'.{file_ext}' not in allowed_extensions:
            raise HTTPException(status_code=400, detail="File type not allowed")
        
        if content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Content type not allowed")
        
        return True

# ============================================================================
# DATA VALIDATION EDGE CASES (P1)
# ============================================================================

class ContentValidator:
    """Handles content validation edge cases"""
    
    @staticmethod
    def validate_post_content(content: str) -> str:
        """Validate post content with comprehensive checks"""
        if not content or not content.strip():
            raise HTTPException(status_code=400, detail="Content cannot be empty")
        
        # Length validation
        if len(content) < 10:
            raise HTTPException(status_code=400, detail="Content too short (minimum 10 characters)")
        
        if len(content) > 3000:
            raise HTTPException(status_code=400, detail="Content too long (maximum 3000 characters)")
        
        # Word count validation
        words = content.split()
        if len(words) < 5:
            raise HTTPException(status_code=400, detail="Content must have at least 5 words")
        
        if len(words) > 500:
            raise HTTPException(status_code=400, detail="Content too long (maximum 500 words)")
        
        # Check for excessive whitespace
        if len(content) - len(content.strip()) > len(content) * 0.1:
            raise HTTPException(status_code=400, detail="Excessive whitespace detected")
        
        # Check for repeated characters
        for char in content:
            if content.count(char) > len(content) * 0.3:
                raise HTTPException(status_code=400, detail="Excessive character repetition")
        
        return content.strip()
    
    @staticmethod
    def validate_hashtags(hashtags: List[str]) -> List[str]:
        """Validate hashtag format and count"""
        if not hashtags:
            return []
        
        if len(hashtags) > 30:
            raise HTTPException(status_code=400, detail="Too many hashtags (maximum 30)")
        
        validated_hashtags = []
        for hashtag in hashtags:
            if not hashtag.startswith('#'):
                hashtag = f"#{hashtag}"
            
            if len(hashtag) < 2:
                continue
            
            if len(hashtag) > 50:
                raise HTTPException(status_code=400, detail=f"Hashtag too long: {hashtag}")
            
            # Check for valid characters
            if not re.match(r'^#[a-zA-Z0-9_]+$', hashtag):
                raise HTTPException(status_code=400, detail=f"Invalid hashtag format: {hashtag}")
            
            validated_hashtags.append(hashtag.lower())
        
        return list(set(validated_hashtags))  # Remove duplicates

class BusinessRuleValidator:
    """Handles business rule validation edge cases"""
    
    def __init__(self) -> Any:
        self.user_post_counts = {}
        self.duplicate_cache = {}
    
    async def validate_user_post_limit(self, user_id: str, daily_limit: int = 5) -> bool:
        """Check if user has exceeded daily post limit"""
        today = datetime.now().date()
        key = f"{user_id}_{today}"
        
        if key not in self.user_post_counts:
            # In production, this would query the database
            self.user_post_counts[key] = await self.get_user_daily_posts(user_id, today)
        
        if self.user_post_counts[key] >= daily_limit:
            raise HTTPException(
                status_code=429,
                detail=f"Daily post limit exceeded ({daily_limit} posts per day)"
            )
        
        return True
    
    async def validate_duplicate_content(self, content: str, user_id: str, time_window: int = 3600) -> bool:
        """Check for duplicate content within time window"""
        content_hash = hash(content.lower().strip())
        key = f"{user_id}_{content_hash}"
        
        if key in self.duplicate_cache:
            last_post_time = self.duplicate_cache[key]
            if datetime.now() - last_post_time < timedelta(seconds=time_window):
                raise HTTPException(
                    status_code=400,
                    detail="Duplicate content detected within time window"
                )
        
        return True
    
    async def get_user_daily_posts(self, user_id: str, date: datetime.date) -> int:
        """Get user's post count for a specific date"""
        # This would query the database in production
        return 0

# ============================================================================
# RATE LIMITING EDGE CASES (P1)
# ============================================================================

class RateLimitHandler:
    """Handles rate limiting edge cases"""
    
    def __init__(self) -> Any:
        self.request_history = {}
        self.blocked_users = {}
    
    async def check_rate_limit(
        self, 
        user_id: str, 
        endpoint: str,
        limit: int = 100, 
        window: int = 3600
    ) -> bool:
        """Check rate limit with endpoint-specific rules"""
        now = datetime.now()
        key = f"{user_id}_{endpoint}"
        
        # Check if user is blocked
        if user_id in self.blocked_users:
            block_until = self.blocked_users[user_id]
            if now < block_until:
                raise HTTPException(
                    status_code=429,
                    detail=f"Account temporarily blocked until {block_until}"
                )
            else:
                del self.blocked_users[user_id]
        
        # Get request history
        if key not in self.request_history:
            self.request_history[key] = []
        
        requests = self.request_history[key]
        
        # Remove old requests outside window
        requests = [req for req in requests if now - req < timedelta(seconds=window)]
        
        # Check limit
        if len(requests) >= limit:
            # Block user for 1 hour if they exceed limit significantly
            if len(requests) >= limit * 2:
                self.blocked_users[user_id] = now + timedelta(hours=1)
                raise HTTPException(
                    status_code=429,
                    detail="Account temporarily blocked due to excessive requests"
                )
            
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Limit: {limit} requests per hour"
            )
        
        # Add current request
        requests.append(now)
        self.request_history[key] = requests
        
        return True
    
    async def get_rate_limit_info(self, user_id: str, endpoint: str) -> Dict[str, Any]:
        """Get rate limit information for user"""
        key = f"{user_id}_{endpoint}"
        requests = self.request_history.get(key, [])
        
        now = datetime.now()
        recent_requests = [req for req in requests if now - req < timedelta(seconds=3600)]
        
        return {
            "requests_used": len(recent_requests),
            "requests_remaining": max(0, 100 - len(recent_requests)),
            "reset_time": (now + timedelta(seconds=3600)).isoformat() if recent_requests else None
        }

# ============================================================================
# DATABASE EDGE CASES (P2)
# ============================================================================

class DatabaseErrorHandler:
    """Handles database-related edge cases"""
    
    @staticmethod
    async def safe_database_operation(
        operation_func, 
        *args, 
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        """Execute database operation with retry logic and error handling"""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return await operation_func(*args, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"Database operation attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
        
        # All retries failed
        logger.error(f"Database operation failed after {max_retries} attempts: {last_error}")
        raise HTTPException(
            status_code=500,
            detail="Database operation failed. Please try again later."
        )
    
    @staticmethod
    async def handle_connection_pool_exhaustion():
        """Handle database connection pool exhaustion"""
        logger.error("Database connection pool exhausted")
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable due to high load"
        )
    
    @staticmethod
    async def handle_deadlock():
        """Handle database deadlock scenarios"""
        logger.warning("Database deadlock detected")
        raise HTTPException(
            status_code=409,
            detail="Resource conflict. Please try again."
        )

# ============================================================================
# EXTERNAL SERVICE EDGE CASES (P2)
# ============================================================================

class ExternalServiceHandler:
    """Handles external service edge cases"""
    
    def __init__(self) -> Any:
        self.circuit_breakers = {}
        self.fallback_responses = {}
    
    async def safe_external_call(
        self,
        service_name: str,
        call_func,
        *args,
        timeout: int = 10,
        max_retries: int = 2,
        **kwargs
    ):
        """Make safe external service calls with circuit breaker"""
        circuit_breaker = self.circuit_breakers.get(service_name)
        if not circuit_breaker:
            circuit_breaker = CircuitBreaker()
            self.circuit_breakers[service_name] = circuit_breaker
        
        try:
            return await circuit_breaker.call(
                self._make_call_with_timeout,
                call_func,
                timeout,
                max_retries,
                *args,
                **kwargs
            )
        except Exception as e:
            logger.error(f"External service {service_name} failed: {e}")
            return await self.get_fallback_response(service_name)
    
    async def _make_call_with_timeout(
        self,
        call_func,
        timeout: int,
        max_retries: int,
        *args,
        **kwargs
    ):
        """Make call with timeout and retry logic"""
        for attempt in range(max_retries):
            try:
                return await asyncio.wait_for(call_func(*args, **kwargs), timeout=timeout)
            except asyncio.TimeoutError:
                if attempt == max_retries - 1:
                    raise HTTPException(status_code=504, detail="External service timeout")
                await asyncio.sleep(1)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(1)
    
    async def get_fallback_response(self, service_name: str) -> Dict[str, Any]:
        """Get fallback response when external service fails"""
        fallback = self.fallback_responses.get(service_name, {
            "status": "service_unavailable",
            "message": "External service temporarily unavailable"
        })
        return fallback

class CircuitBreaker:
    """Circuit breaker pattern for external services"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        
    """__init__ function."""
self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs) -> Any:
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

# ============================================================================
# PERFORMANCE EDGE CASES (P3)
# ============================================================================

class PerformanceHandler:
    """Handles performance-related edge cases"""
    
    def __init__(self) -> Any:
        self.operation_times = {}
        self.slow_operation_threshold = 5.0  # seconds
    
    async def monitor_operation_time(self, operation_name: str, operation_func, *args, **kwargs):
        """Monitor operation execution time"""
        start_time = datetime.now()
        
        try:
            result = await operation_func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Log slow operations
            if execution_time > self.slow_operation_threshold:
                logger.warning(f"Slow operation detected: {operation_name} took {execution_time}s")
            
            # Store timing data
            if operation_name not in self.operation_times:
                self.operation_times[operation_name] = []
            self.operation_times[operation_name].append(execution_time)
            
            return result
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Operation {operation_name} failed after {execution_time}s: {e}")
            raise
    
    async def handle_large_payload(self, data: List[Dict], batch_size: int = 1000):
        """Handle large payloads in batches"""
        if len(data) <= batch_size:
            return await self.process_data(data)
        
        results = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_result = await self.process_data(batch)
            results.extend(batch_result)
            
            # Allow other tasks to run
            await asyncio.sleep(0)
        
        return results
    
    async def process_data(self, data: List[Dict]) -> List[Dict]:
        """Process data batch"""
        # Implementation would depend on specific use case
        return data

# ============================================================================
# RESOURCE MANAGEMENT EDGE CASES (P3)
# ============================================================================

class ResourceManager:
    """Handles resource management edge cases"""
    
    def __init__(self) -> Any:
        self.resources = []
        self.max_resources = 100
    
    async def acquire_resource(self, resource_type: str):
        """Acquire resource with limits"""
        if len(self.resources) >= self.max_resources:
            raise HTTPException(
                status_code=503,
                detail="Resource limit reached. Please try again later."
            )
        
        resource = await self.create_resource(resource_type)
        self.resources.append(resource)
        return resource
    
    async def release_resource(self, resource) -> Any:
        """Release resource safely"""
        try:
            if hasattr(resource, 'close'):
                await resource.close()
            elif hasattr(resource, 'cleanup'):
                await resource.cleanup()
        except Exception as e:
            logger.error(f"Error releasing resource: {e}")
        finally:
            if resource in self.resources:
                self.resources.remove(resource)
    
    async def create_resource(self, resource_type: str):
        """Create resource based on type"""
        # Implementation would depend on resource type
        return {"type": resource_type, "id": len(self.resources)}
    
    async def cleanup_all_resources(self) -> Any:
        """Cleanup all resources"""
        for resource in self.resources[:]:
            await self.release_resource(resource)

# ============================================================================
# MONITORING AND ALERTING (P3)
# ============================================================================

class ErrorMonitor:
    """Monitors and alerts on errors"""
    
    def __init__(self) -> Any:
        self.error_counts = {}
        self.alert_threshold = 10
        self.error_window = timedelta(minutes=5)
        self.error_history = []
    
    def log_error(self, error_type: str, context: Dict[str, Any] = None):
        """Log error with context"""
        now = datetime.now()
        
        # Clean old errors
        self.error_history = [
            error for error in self.error_history 
            if now - error['timestamp'] < self.error_window
        ]
        
        # Add new error
        error_entry = {
            'type': error_type,
            'timestamp': now,
            'context': context or {}
        }
        self.error_history.append(error_entry)
        
        # Count errors by type
        error_count = len([e for e in self.error_history if e['type'] == error_type])
        
        # Log error
        logger.error(f"Error {error_type}: {context}")
        
        # Alert if threshold exceeded
        if error_count >= self.alert_threshold:
            self.send_alert(error_type, error_count, context)
    
    def send_alert(self, error_type: str, count: int, context: Dict[str, Any]):
        """Send alert for high error rate"""
        alert_message = {
            "type": "high_error_rate",
            "error_type": error_type,
            "count": count,
            "window_minutes": self.error_window.total_seconds() / 60,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.critical(f"ALERT: {json.dumps(alert_message)}")
        # In production, this would send to monitoring system

# ============================================================================
# MAIN EDGE CASE HANDLER
# ============================================================================

class EdgeCaseHandler:
    """Main handler for all edge cases"""
    
    def __init__(self) -> Any:
        self.security_validator = SecurityValidator()
        self.content_validator = ContentValidator()
        self.business_validator = BusinessRuleValidator()
        self.rate_limiter = RateLimitHandler()
        self.db_handler = DatabaseErrorHandler()
        self.external_handler = ExternalServiceHandler()
        self.performance_handler = PerformanceHandler()
        self.resource_manager = ResourceManager()
        self.error_monitor = ErrorMonitor()
    
    async def handle_post_creation(
        self,
        user_id: str,
        content: str,
        hashtags: List[str] = None,
        request: Request = None
    ) -> Dict[str, Any]:
        """Handle post creation with comprehensive edge case checking"""
        try:
            # P0: Security validation
            content = self.security_validator.validate_sql_injection(content)
            content = self.security_validator.validate_xss_attack(content)
            
            # P1: Content validation
            content = self.content_validator.validate_post_content(content)
            hashtags = self.content_validator.validate_hashtags(hashtags or [])
            
            # P1: Business rules
            await self.business_validator.validate_user_post_limit(user_id)
            await self.business_validator.validate_duplicate_content(content, user_id)
            
            # P1: Rate limiting
            await self.rate_limiter.check_rate_limit(user_id, "post_creation")
            
            # P2: Database operation
            post_data = await self.db_handler.safe_database_operation(
                self.create_post_in_db,
                user_id,
                content,
                hashtags
            )
            
            # P3: Performance monitoring
            await self.performance_handler.monitor_operation_time(
                "post_creation",
                self.finalize_post_creation,
                post_data
            )
            
            return {
                "status": "success",
                "post_id": post_data["id"],
                "message": "Post created successfully"
            }
            
        except HTTPException:
            raise
        except Exception as e:
            self.error_monitor.log_error("post_creation_failed", {
                "user_id": user_id,
                "error": str(e)
            })
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def create_post_in_db(self, user_id: str, content: str, hashtags: List[str]):
        """Create post in database"""
        # Implementation would interact with actual database
        return {
            "id": f"post_{len(hashtags)}",
            "user_id": user_id,
            "content": content,
            "hashtags": hashtags,
            "created_at": datetime.now().isoformat()
        }
    
    async def finalize_post_creation(self, post_data: Dict[str, Any]):
        """Finalize post creation process"""
        # Additional processing like notifications, analytics, etc.
        await asyncio.sleep(0.1)  # Simulate processing time
        return post_data

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def example_usage():
    """Example of using the edge case handler"""
    handler = EdgeCaseHandler()
    
    try:
        result = await handler.handle_post_creation(
            user_id="user123",
            content="This is a test post with #hashtag",
            hashtags=["test", "example"]
        )
        print(f"Success: {result}")
        
    except HTTPException as e:
        print(f"HTTP Error {e.status_code}: {e.detail}")
    except Exception as e:
        print(f"Unexpected error: {e}")

match __name__:
    case "__main__":
    asyncio.run(example_usage()) 