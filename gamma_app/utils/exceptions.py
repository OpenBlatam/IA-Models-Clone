"""
Common Exceptions
"""


class BaseAppException(Exception):
    """Base exception for application"""
    def __init__(self, message: str, code: str = None, details: dict = None):
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(BaseAppException):
    """Validation error"""
    pass


class ConfigurationError(BaseAppException):
    """Configuration error"""
    pass


class AuthenticationError(BaseAppException):
    """Authentication error"""
    pass


class AuthorizationError(BaseAppException):
    """Authorization error"""
    pass


class NotFoundError(BaseAppException):
    """Resource not found error"""
    pass


class ConflictError(BaseAppException):
    """Resource conflict error"""
    pass


class ServiceUnavailableError(BaseAppException):
    """Service unavailable error"""
    pass


class RateLimitError(BaseAppException):
    """Rate limit error"""
    pass


class DatabaseError(BaseAppException):
    """Database error"""
    pass


class ExternalServiceError(BaseAppException):
    """External service error"""
    pass

