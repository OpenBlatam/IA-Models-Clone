"""
Middleware for Physical Store Designer AI

This module re-exports all middleware from the refactored middleware package
for backward compatibility.
"""

# Import from refactored modules
from .middleware import (
    TimeoutMiddleware,
    ErrorHandlerMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    RequestLoggingMiddleware,
    CompressionMiddleware,
)

# Re-export for backward compatibility
__all__ = [
    "TimeoutMiddleware",
    "ErrorHandlerMiddleware",
    "RateLimitMiddleware",
    "SecurityHeadersMiddleware",
    "RequestLoggingMiddleware",
    "CompressionMiddleware",
]

# Note: All middleware classes are now imported from the refactored middleware package.
# The original definitions have been removed.
