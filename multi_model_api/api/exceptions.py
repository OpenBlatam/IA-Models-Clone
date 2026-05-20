"""
Custom exceptions for Multi-Model API
Centralized exception handling for better error management
"""


class MultiModelAPIException(Exception):
    """Base exception for Multi-Model API"""
    
    def __init__(self, message: str, status_code: int = 500, details: dict = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class ModelExecutionException(MultiModelAPIException):
    """Exception during model execution"""
    
    def __init__(self, message: str, model_type: str = None, details: dict = None):
        super().__init__(
            message=message,
            status_code=500,
            details=details or {}
        )
        self.model_type = model_type


class RateLimitExceededException(MultiModelAPIException):
    """Rate limit exceeded exception"""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int = 60,
        limit: int = None,
        remaining: int = 0,
        reset_at: float = None
    ):
        super().__init__(
            message=message,
            status_code=429,
            details={
                "retry_after": retry_after,
                "limit": limit,
                "remaining": remaining,
                "reset_at": reset_at
            }
        )
        self.retry_after = retry_after
        self.limit = limit
        self.remaining = remaining
        self.reset_at = reset_at


class CacheException(MultiModelAPIException):
    """Cache operation failed"""
    
    def __init__(self, message: str, operation: str = None, details: dict = None):
        super().__init__(
            message=message,
            status_code=500,
            details=details or {}
        )
        self.operation = operation


class ValidationException(MultiModelAPIException):
    """Request validation failed"""
    
    def __init__(self, message: str, field: str = None, details: dict = None):
        super().__init__(
            message=message,
            status_code=400,
            details=details or {}
        )
        self.field = field


class ModelNotFoundException(MultiModelAPIException):
    """Model not found in registry"""
    
    def __init__(self, model_type: str, details: dict = None):
        super().__init__(
            message=f"Model {model_type} not found",
            status_code=404,
            details=details or {}
        )
        self.model_type = model_type


class StrategyNotFoundException(MultiModelAPIException):
    """Execution strategy not found"""
    
    def __init__(self, strategy: str, details: dict = None):
        super().__init__(
            message=f"Unknown execution strategy: {strategy}",
            status_code=400,
            details=details or {}
        )
        self.strategy = strategy


class TimeoutException(MultiModelAPIException):
    """Request timeout exception"""
    
    def __init__(self, timeout: float, details: dict = None):
        super().__init__(
            message=f"Request timed out after {timeout} seconds",
            status_code=504,
            details=details or {}
        )
        self.timeout = timeout




