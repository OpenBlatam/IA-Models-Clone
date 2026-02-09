from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import List
from fastapi import APIRouter
from .captions import captions_router
from .performance import performance_router
from .async_flow_routes import async_flow_router
from .enhanced_async_routes import enhanced_async_router
from .shared_resources_routes import shared_resources_router
from .lazy_loading_routes import lazy_loading_router
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Routes Module for Instagram Captions API v14.0

Well-structured routing system with clear dependencies and organization:
- Modular route organization
- Clear dependency injection
- Consistent patterns and conventions
- Easy maintenance and testing
"""


# Import all route modules

# Create main router registry
class RouterRegistry:
    """Central router registry for managing all application routes"""
    
    def __init__(self) -> Any:
        self.routers: List[APIRouter] = []
        self.router_configs: dict = {}
    
    def register_router(
        self, 
        router: APIRouter, 
        prefix: str = "", 
        tags: List[str] = None,
        dependencies: List = None,
        description: str = ""
    ):
        """Register a router with configuration"""
        self.routers.append(router)
        self.router_configs[router] = {
            "prefix": prefix,
            "tags": tags or [],
            "dependencies": dependencies or [],
            "description": description
        }
    
    def get_all_routers(self) -> List[tuple]:
        """Get all routers with their configurations"""
        return [
            (router, config) 
            for router, config in self.router_configs.items()
        ]
    
    def get_router_by_tag(self, tag: str) -> List[APIRouter]:
        """Get routers by tag"""
        return [
            router for router, config in self.router_configs.items()
            if tag in config["tags"]
        ]

# Create router registry instance
router_registry = RouterRegistry()

# Register all routers with clear organization
router_registry.register_router(
    router=captions_router,
    prefix="/captions",
    tags=["captions", "core"],
    description="Core caption generation endpoints"
)

router_registry.register_router(
    router=performance_router,
    prefix="/performance",
    tags=["performance", "monitoring"],
    description="Performance monitoring and analytics endpoints"
)

router_registry.register_router(
    router=async_flow_router,
    prefix="/async-flows",
    tags=["async-flows", "advanced"],
    description="Advanced async flow management endpoints"
)

router_registry.register_router(
    router=enhanced_async_router,
    prefix="/enhanced-async",
    tags=["enhanced-async", "advanced"],
    description="Enhanced async database and API operations"
)

router_registry.register_router(
    router=shared_resources_router,
    prefix="/shared-resources",
    tags=["shared-resources", "infrastructure"],
    description="Shared resources management endpoints"
)

router_registry.register_router(
    router=lazy_loading_router,
    prefix="/lazy-loading",
    tags=["lazy-loading", "optimization"],
    description="Lazy loading and optimization endpoints"
)

# Export routers for easy access
__all__ = [
    "router_registry",
    "captions_router",
    "performance_router", 
    "async_flow_router",
    "enhanced_async_router",
    "shared_resources_router",
    "lazy_loading_router"
] 