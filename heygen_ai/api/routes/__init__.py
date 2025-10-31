from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from fastapi import APIRouter, FastAPI
from typing import Dict, List, Any, Optional
import structlog
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
HeyGen AI FastAPI Routes Package
Main routes package initialization with dependency injection and route registration.
"""


logger = structlog.get_logger()

# =============================================================================
# Route Registry
# =============================================================================

class RouteRegistry:
    """Central route registry for managing all API routes."""
    
    def __init__(self) -> Any:
        self.routes: Dict[str, APIRouter] = {}
        self.dependencies: Dict[str, Any] = {}
        self.middleware: List[Any] = []
        self._is_initialized = False
    
    def register_route(self, name: str, router: APIRouter, prefix: str = "", tags: Optional[List[str]] = None):
        """Register a route with the registry."""
        if name in self.routes:
            logger.warning(f"Route {name} already registered, overwriting")
        
        self.routes[name] = {
            "router": router,
            "prefix": prefix,
            "tags": tags or []
        }
        logger.info(f"Registered route: {name} with prefix: {prefix}")
    
    def register_dependency(self, name: str, dependency: Any):
        """Register a dependency with the registry."""
        self.dependencies[name] = dependency
        logger.info(f"Registered dependency: {name}")
    
    def register_middleware(self, middleware: Any):
        """Register middleware with the registry."""
        self.middleware.append(middleware)
        logger.info(f"Registered middleware: {middleware.__class__.__name__}")
    
    def get_route(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a registered route."""
        return self.routes.get(name)
    
    def get_dependency(self, name: str) -> Optional[Any]:
        """Get a registered dependency."""
        return self.dependencies.get(name)
    
    def get_all_routes(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered routes."""
        return self.routes.copy()
    
    def get_all_dependencies(self) -> Dict[str, Any]:
        """Get all registered dependencies."""
        return self.dependencies.copy()
    
    def setup_app(self, app: FastAPI):
        """Setup all routes and dependencies on the FastAPI app."""
        if self._is_initialized:
            return
        
        # Register all routes
        for name, route_info in self.routes.items():
            router = route_info["router"]
            prefix = route_info["prefix"]
            tags = route_info["tags"]
            
            app.include_router(
                router,
                prefix=prefix,
                tags=tags
            )
            logger.info(f"Included router {name} with prefix {prefix}")
        
        # Register middleware
        for middleware in self.middleware:
            app.add_middleware(middleware)
            logger.info(f"Added middleware: {middleware.__class__.__name__}")
        
        self._is_initialized = True
        logger.info("Route registry setup completed")

# Global route registry instance
route_registry = RouteRegistry()

# =============================================================================
# Dependency Injection Container
# =============================================================================

class DependencyContainer:
    """Dependency injection container for managing service dependencies."""
    
    def __init__(self) -> Any:
        self.services: Dict[str, Any] = {}
        self.singletons: Dict[str, Any] = {}
        self._is_initialized = False
    
    def register_service(self, name: str, service_class: type, *args, **kwargs):
        """Register a service with the container."""
        self.services[name] = {
            "class": service_class,
            "args": args,
            "kwargs": kwargs,
            "instance": None
        }
        logger.info(f"Registered service: {name}")
    
    def register_singleton(self, name: str, instance: Any):
        """Register a singleton instance."""
        self.singletons[name] = instance
        logger.info(f"Registered singleton: {name}")
    
    def get_service(self, name: str) -> Optional[Any]:
        """Get a service instance, creating it if necessary."""
        if name in self.singletons:
            return self.singletons[name]
        
        if name in self.services:
            service_info = self.services[name]
            if service_info["instance"] is None:
                service_info["instance"] = service_info["class"](*service_info["args"], **service_info["kwargs"])
            return service_info["instance"]
        
        return None
    
    def initialize_services(self) -> Any:
        """Initialize all registered services."""
        if self._is_initialized:
            return
        
        for name, service_info in self.services.items():
            if service_info["instance"] is None:
                service_info["instance"] = service_info["class"](*service_info["args"], **service_info["kwargs"])
                logger.info(f"Initialized service: {name}")
        
        self._is_initialized = True
        logger.info("Dependency container initialization completed")
    
    def cleanup_services(self) -> Any:
        """Cleanup all services."""
        for name, service_info in self.services.items():
            if service_info["instance"] and hasattr(service_info["instance"], "cleanup"):
                service_info["instance"].cleanup()
                logger.info(f"Cleaned up service: {name}")
        
        for name, instance in self.singletons.items():
            if hasattr(instance, "cleanup"):
                instance.cleanup()
                logger.info(f"Cleaned up singleton: {name}")

# Global dependency container instance
dependency_container = DependencyContainer()

# =============================================================================
# Route Categories
# =============================================================================

# Route categories for organization
ROUTE_CATEGORIES = {
    "AUTH": "Authentication and Authorization",
    "USERS": "User Management",
    "VIDEOS": "Video Processing",
    "AI": "AI and Machine Learning",
    "ANALYTICS": "Analytics and Reporting",
    "SYSTEM": "System and Health",
    "EXTERNAL": "External API Integration",
    "UTILS": "Utility and Helper Functions"
}

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "RouteRegistry",
    "DependencyContainer",
    "route_registry",
    "dependency_container",
    "ROUTE_CATEGORIES"
] 