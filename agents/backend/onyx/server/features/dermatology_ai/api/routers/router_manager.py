"""
Router Manager - Centralized management of all modular routers
"""

from fastapi import APIRouter
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class RouterManager:
    """Manages registration and organization of all API routers"""
    
    def __init__(self):
        self.routers: List[APIRouter] = []
        self.router_registry: dict = {}
    
    def register_router(
        self,
        router: APIRouter,
        name: str,
        prefix: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None
    ):
        """Register a router with metadata"""
        if prefix:
            router.prefix = prefix
        
        self.routers.append(router)
        self.router_registry[name] = {
            "router": router,
            "prefix": prefix or router.prefix,
            "tags": tags or router.tags,
            "description": description
        }
        logger.info(f"Registered router: {name} with prefix: {prefix or router.prefix}")
    
    def get_all_routers(self) -> List[APIRouter]:
        """Get all registered routers"""
        return self.routers
    
    def get_router_info(self) -> dict:
        """Get information about all registered routers"""
        return {
            name: {
                "prefix": info["prefix"],
                "tags": info["tags"],
                "description": info.get("description", "")
            }
            for name, info in self.router_registry.items()
        }
    
    def get_endpoints_summary(self) -> dict:
        """Get a summary of all endpoints from registered routers"""
        summary = {}
        for name, info in self.router_registry.items():
            router = info["router"]
            routes = []
            for route in router.routes:
                if hasattr(route, "path") and hasattr(route, "methods"):
                    routes.append({
                        "path": route.path,
                        "methods": list(route.methods),
                        "name": getattr(route, "name", "unknown")
                    })
            summary[name] = {
                "prefix": info["prefix"],
                "routes": routes
            }
        return summary


# Global router manager instance
_router_manager: Optional[RouterManager] = None


def get_router_manager() -> RouterManager:
    """Get or create the global router manager instance"""
    global _router_manager
    if _router_manager is None:
        _router_manager = RouterManager()
    return _router_manager




