"""
PDF Variantes API - Real-World Utilities
Practical utilities for production-ready error handling and resilience
"""

import asyncio
import time
import functools
from typing import Any, Callable, Optional, Dict, List, TypeVar, Awaitable
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorCode(str, Enum):
    """Real-world error codes for better error handling"""
    # General errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    
    # PDF specific errors
    PDF_INVALID_FORMAT = "PDF_INVALID_FORMAT"
    PDF_TOO_LARGE = "PDF_TOO_LARGE"
    PDF_PROCESSING_FAILED = "PDF_PROCESSING_FAILED"
    PDF_ENCRYPTED = "PDF_ENCRYPTED"
    PDF_CORRUPTED = "PDF_CORRUPTED"
    
    # Service errors
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    SERVICE_TIMEOUT = "SERVICE_TIMEOUT"
    DATABASE_ERROR = "DATABASE_ERROR"
    CACHE_ERROR = "CACHE_ERROR"
    STORAGE_ERROR = "STORAGE_ERROR"
    
    # AI/Processing errors
    AI_SERVICE_ERROR = "AI_SERVICE_ERROR"
    PROCESSING_QUEUE_FULL = "PROCESSING_QUEUE_FULL"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"


class RetryStrategy(str, Enum):
    """Retry strategies for different error types"""
    EXPONENTIAL = "exponential"  # 1s, 2s, 4s, 8s...
    LINEAR = "linear"  # 1s, 2s, 3s, 4s...
    FIXED = "fixed"  # 1s, 1s, 1s, 1s...
    IMMEDIATE = "immediate"  # No delay, immediate retry


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    retryable_exceptions: tuple = (Exception,),
    logger_instance: Optional[logging.Logger] = None
):
    """
    Real-world retry decorator with exponential backoff
    
    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        strategy: Retry strategy (exponential, linear, fixed)
        retryable_exceptions: Tuple of exceptions that should trigger retry
        logger_instance: Optional logger instance
    """
    log = logger_instance or logger
    
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        log.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {str(e)}"
                        )
                        raise
                    
                    # Calculate delay based on strategy
                    if strategy == RetryStrategy.EXPONENTIAL:
                        delay = min(initial_delay * (2 ** (attempt - 1)), max_delay)
                    elif strategy == RetryStrategy.LINEAR:
                        delay = min(initial_delay * attempt, max_delay)
                    elif strategy == RetryStrategy.FIXED:
                        delay = initial_delay
                    else:  # IMMEDIATE
                        delay = 0
                    
                    log.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {str(e)}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    if delay > 0:
                        await asyncio.sleep(delay)
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern for resilient service calls
    Prevents cascading failures by temporarily stopping calls to failing services
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half_open
        self.success_count = 0
    
    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        # Check if circuit should be opened/closed
        if self.state == "open":
            if self.last_failure_time and \
               (datetime.utcnow() - self.last_failure_time).total_seconds() > self.recovery_timeout:
                self.state = "half_open"
                self.success_count = 0
                logger.info(f"Circuit breaker moving to half_open state")
            else:
                raise Exception(
                    f"Circuit breaker is OPEN. Service unavailable. "
                    f"Will retry after {self.recovery_timeout}s"
                )
        
        try:
            result = await func(*args, **kwargs)
            
            # Success - reset if in half_open
            if self.state == "half_open":
                self.success_count += 1
                if self.success_count >= 2:  # Need 2 successes to close
                    self.state = "closed"
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info("Circuit breaker CLOSED - service recovered")
            else:
                # Reset failure count on success
                self.failure_count = 0
            
            return result
            
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(
                    f"Circuit breaker OPENED after {self.failure_count} failures. "
                    f"Service will be unavailable for {self.recovery_timeout}s"
                )
            
            raise


def with_timeout(
    timeout_seconds: float = 30.0,
    timeout_error: type = asyncio.TimeoutError
):
    """
    Add timeout to async function
    
    Args:
        timeout_seconds: Timeout in seconds
        timeout_error: Exception to raise on timeout
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"{func.__name__} timed out after {timeout_seconds}s"
                )
                raise timeout_error(
                    f"Operation timed out after {timeout_seconds} seconds"
                )
        
        return wrapper
    return decorator


class HealthCheck:
    """Real-world health check with dependency validation"""
    
    def __init__(self):
        self.checks: Dict[str, Callable[[], Awaitable[bool]]] = {}
        self.last_check: Dict[str, datetime] = {}
        self.check_cache_ttl = timedelta(seconds=30)
    
    def register_check(self, name: str, check_func: Callable[[], Awaitable[bool]]):
        """Register a health check"""
        self.checks[name] = check_func
    
    async def check_all(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }
        
        all_healthy = True
        
        for name, check_func in self.checks.items():
            try:
                # Use cached result if available
                if name in self.last_check:
                    cache_age = datetime.utcnow() - self.last_check[name]
                    if cache_age < self.check_cache_ttl:
                        continue  # Skip if cached and fresh
                
                is_healthy = await check_func()
                self.last_check[name] = datetime.utcnow()
                
                results["checks"][name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "checked_at": self.last_check[name].isoformat()
                }
                
                if not is_healthy:
                    all_healthy = False
                    
            except Exception as e:
                results["checks"][name] = {
                    "status": "error",
                    "error": str(e),
                    "checked_at": datetime.utcnow().isoformat()
                }
                all_healthy = False
        
        if not all_healthy:
            results["status"] = "degraded" if any(
                check.get("status") == "healthy" 
                for check in results["checks"].values()
            ) else "unhealthy"
        
        return results
    
    async def check_single(self, name: str) -> bool:
        """Check a single dependency"""
        if name not in self.checks:
            return False
        
        try:
            return await self.checks[name]()
        except Exception:
            return False


class RateLimiter:
    """Rate limiting per user/API key"""
    
    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        per_user: bool = True
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.per_user = per_user
        self.requests: Dict[str, List[float]] = {}
    
    def is_allowed(self, identifier: str):
        """
        Check if request is allowed
        
        Returns:
            (is_allowed, retry_after_seconds)
        """
        now = time.time()
        cutoff = now - self.window_seconds
        
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Clean old requests
        self.requests[identifier] = [
            ts for ts in self.requests[identifier] 
            if ts > cutoff
        ]
        
        # Check limit
        if len(self.requests[identifier]) >= self.max_requests:
            # Calculate retry after
            oldest_request = min(self.requests[identifier])
            retry_after = int(self.window_seconds - (now - oldest_request)) + 1
            return False, retry_after
        
        # Add current request
        self.requests[identifier].append(now)
        return True, None
    
    def reset(self, identifier: str):
        """Reset rate limit for identifier"""
        self.requests.pop(identifier, None)


def validate_pdf_file(file_content: bytes, max_size_mb: int = 100) -> tuple:
    """
    Real-world PDF validation
    
    Returns:
        (is_valid, error_message)
    """
    # Check size
    max_size_bytes = max_size_mb * 1024 * 1024
    if len(file_content) > max_size_bytes:
        return False, f"PDF file too large. Maximum size: {max_size_mb}MB"
    
    # Check PDF signature
    if not file_content.startswith(b'%PDF-'):
        return False, "Invalid PDF format. File does not start with PDF signature"
    
    # Check minimum size
    if len(file_content) < 100:
        return False, "PDF file too small. File may be corrupted"
    
    # Check for encryption (basic check)
    if b'/Encrypt' in file_content[:1024]:
        return False, "PDF file is encrypted. Encrypted PDFs are not supported"
    
    return True, None


def format_error_response(
    error_code: ErrorCode,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Format error response with real-world error codes"""
    response = {
        "success": False,
        "error": {
            "code": error_code.value,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
    }
    
    if details:
        response["error"]["details"] = details
    
    if request_id:
        response["request_id"] = request_id
    
    return response


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that handles zero denominator"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

