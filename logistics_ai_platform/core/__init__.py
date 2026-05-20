"""
Core application setup module

This module provides functions to create and configure the FastAPI application.
"""

from .app_factory import create_app
from .middleware_setup import setup_middleware
from .exception_handlers_setup import setup_exception_handlers
from .router_setup import setup_routers
from .root_endpoints import setup_root_endpoints

__all__ = [
    "create_app",
    "setup_middleware",
    "setup_exception_handlers",
    "setup_routers",
    "setup_root_endpoints",
]
