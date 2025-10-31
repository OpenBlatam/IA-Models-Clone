"""
Custom Exceptions
================

Ultra-advanced custom exceptions for Flask applications.
"""

class BaseException(Exception):
    """Base exception class."""
    def __init__(self, message: str, status_code: int = 500, details: dict = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class AuthenticationError(BaseException):
    """Authentication error."""
    def __init__(self, message: str = "Authentication failed", details: dict = None):
        super().__init__(message, 401, details)

class ValidationError(BaseException):
    """Validation error."""
    def __init__(self, message: str = "Validation failed", details: dict = None):
        super().__init__(message, 400, details)

class OptimizationError(BaseException):
    """Optimization error."""
    def __init__(self, message: str = "Optimization failed", details: dict = None):
        super().__init__(message, 500, details)

class PerformanceError(BaseException):
    """Performance error."""
    def __init__(self, message: str = "Performance issue", details: dict = None):
        super().__init__(message, 500, details)

class SecurityError(BaseException):
    """Security error."""
    def __init__(self, message: str = "Security violation", details: dict = None):
        super().__init__(message, 403, details)

class AIError(BaseException):
    """AI system error."""
    def __init__(self, message: str = "AI system error", details: dict = None):
        super().__init__(message, 500, details)

class QuantumError(BaseException):
    """Quantum computing error."""
    def __init__(self, message: str = "Quantum computing error", details: dict = None):
        super().__init__(message, 500, details)

class EdgeError(BaseException):
    """Edge computing error."""
    def __init__(self, message: str = "Edge computing error", details: dict = None):
        super().__init__(message, 500, details)