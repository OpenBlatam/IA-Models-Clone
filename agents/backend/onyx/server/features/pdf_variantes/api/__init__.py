"""
PDF Variantes API Package
Main API module exports
"""

from .main import app
from .dependencies import get_services, get_pdf_service
from .exceptions import (
    BaseAPIException,
    ValidationError,
    NotFoundError,
    ConflictError,
    UnauthorizedError,
    ForbiddenError,
    RateLimitError,
    ServiceUnavailableError,
    InternalServerError,
)

__all__ = [
    "app",
    "get_services",
    "get_pdf_service",
    "BaseAPIException",
    "ValidationError",
    "NotFoundError",
    "ConflictError",
    "UnauthorizedError",
    "ForbiddenError",
    "RateLimitError",
    "ServiceUnavailableError",
    "InternalServerError",
]
