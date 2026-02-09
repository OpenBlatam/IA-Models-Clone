from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .base import router as base_router
from .product_descriptions import router as product_descriptions_router
from .version_control import router as version_control_router
from .performance import router as performance_router
from .health import router as health_router
from .admin import router as admin_router
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Product Descriptions API Routes Package

This package contains all route modules organized by functionality.
Provides a clean interface for importing and registering routes.
"""


# Export all routers for easy registration
__all__ = [
    "base_router",
    "product_descriptions_router", 
    "version_control_router",
    "performance_router",
    "health_router",
    "admin_router"
]

# Router registry for automatic registration
ROUTER_REGISTRY = {
    "base": base_router,
    "product_descriptions": product_descriptions_router,
    "version_control": version_control_router,
    "performance": performance_router,
    "health": health_router,
    "admin": admin_router
}

def get_all_routers():
    """Get all registered routers."""
    return ROUTER_REGISTRY.values()

def get_router_by_name(name: str):
    """Get a specific router by name."""
    return ROUTER_REGISTRY.get(name)

def register_routers(app) -> Any:
    """Register all routers with the FastAPI app."""
    for router in get_all_routers():
        app.include_router(router) 