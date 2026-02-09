"""Application configuration (deprecated - use app_factory instead)."""

# This module is kept for backward compatibility
# New code should use core.app_factory instead

from core.app_factory import create_app
from core.middleware_config import setup_middleware
from core.exceptions_config import setup_exception_handlers
from core.routes_config import setup_routes
from core.constants import API_VERSION, SERVICE_NAME


def create_app_config() -> dict:
    """Get FastAPI app configuration."""
    return {
        "title": f"{SERVICE_NAME} API",
        "description": "AI system that visualizes how you'll look after plastic surgery procedures",
        "version": API_VERSION,
    }

