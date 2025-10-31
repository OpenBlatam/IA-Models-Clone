"""
API Module - FastAPI Application
==============================

Modern FastAPI application with clean architecture and comprehensive endpoints.
"""

from .app import create_app
from .routes import (
    documents_router,
    processing_router,
    health_router,
    metrics_router
)
from .middleware import (
    setup_cors,
    setup_logging,
    setup_error_handlers,
    setup_rate_limiting
)
from .dependencies import (
    get_document_service,
    get_config_manager,
    get_current_user
)

__all__ = [
    "create_app",
    "documents_router",
    "processing_router", 
    "health_router",
    "metrics_router",
    "setup_cors",
    "setup_logging",
    "setup_error_handlers",
    "setup_rate_limiting",
    "get_document_service",
    "get_config_manager",
    "get_current_user",
]

















