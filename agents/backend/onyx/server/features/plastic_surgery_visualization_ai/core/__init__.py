"""Core application modules."""

from core.app_factory import create_app
from core.factories import (
    create_cache,
    create_image_processor,
    create_ai_processor,
    create_visualization_service
)
from core.lifespan import lifespan, startup, shutdown
from core.middleware_config import setup_middleware
from core.exceptions_config import setup_exception_handlers
from core.routes_config import setup_routes

__all__ = [
    "create_app",
    "create_cache",
    "create_image_processor",
    "create_ai_processor",
    "create_visualization_service",
    "lifespan",
    "startup",
    "shutdown",
    "setup_middleware",
    "setup_exception_handlers",
    "setup_routes",
]
