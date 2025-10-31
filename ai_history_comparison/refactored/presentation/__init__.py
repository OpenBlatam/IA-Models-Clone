"""
Presentation Layer
==================

This module contains the presentation layer components that handle
user interface concerns, API endpoints, and external communication.

The presentation layer includes:
- REST API endpoints
- WebSocket connections for real-time updates
- Request/response handling
- Authentication and authorization
- API documentation
- Error handling and validation
"""

from .api import create_app, get_app
from .middleware import (
    ErrorHandlerMiddleware,
    LoggingMiddleware,
    AuthenticationMiddleware,
    RateLimitMiddleware
)
from .endpoints import (
    AnalysisEndpoints,
    ComparisonEndpoints,
    ReportEndpoints,
    TrendEndpoints,
    SystemEndpoints
)

__all__ = [
    # Main app
    "create_app",
    "get_app",
    
    # Middleware
    "ErrorHandlerMiddleware",
    "LoggingMiddleware", 
    "AuthenticationMiddleware",
    "RateLimitMiddleware",
    
    # Endpoints
    "AnalysisEndpoints",
    "ComparisonEndpoints",
    "ReportEndpoints",
    "TrendEndpoints",
    "SystemEndpoints"
]




