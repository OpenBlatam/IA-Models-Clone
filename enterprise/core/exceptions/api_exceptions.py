from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Dict, Any, Optional
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
API Domain Exceptions
====================

Custom exceptions for the enterprise API domain layer.
"""



class EnterpriseAPIException(Exception):
    """Base exception for enterprise API."""
    
    def __init__(self, message: str, code: str = "ENTERPRISE_ERROR", 
                 status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "status_code": self.status_code,
                "details": self.details
            }
        }


class RateLimitExceededException(EnterpriseAPIException):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, retry_after: int, message: str = "Rate limit exceeded"):
        
    """__init__ function."""
super().__init__(
            message=message,
            code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details={"retry_after": retry_after}
        )
        self.retry_after = retry_after


class CircuitBreakerOpenException(EnterpriseAPIException):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, service: str, message: str = "Service temporarily unavailable"):
        
    """__init__ function."""
super().__init__(
            message=message,
            code="CIRCUIT_BREAKER_OPEN",
            status_code=503,
            details={"service": service}
        )
        self.service = service


class CacheException(EnterpriseAPIException):
    """Exception raised for cache operations."""
    
    def __init__(self, operation: str, message: str = "Cache operation failed"):
        
    """__init__ function."""
super().__init__(
            message=message,
            code="CACHE_ERROR",
            status_code=500,
            details={"operation": operation}
        )
        self.operation = operation


class HealthCheckException(EnterpriseAPIException):
    """Exception raised during health checks."""
    
    def __init__(self, component: str, message: str = "Health check failed"):
        
    """__init__ function."""
super().__init__(
            message=message,
            code="HEALTH_CHECK_FAILED",
            status_code=503,
            details={"component": component}
        )
        self.component = component 