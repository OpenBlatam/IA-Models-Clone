from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import time
import logging
import functools
import inspect
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, TypeVar, Generic, Awaitable, Type
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import weakref
import contextlib
from abc import ABC, abstractmethod
import structlog
from pydantic import BaseModel, Field
import numpy as np
from fastapi import FastAPI, APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
            import yaml
from typing import Any, List, Dict, Optional
"""
📁 Route Organization System
============================

Comprehensive route organization system for FastAPI:
- Modular route organization with clear separation of concerns
- Route grouping and categorization
- API versioning and documentation
- Route validation and testing utilities
- Performance monitoring and metrics
- Code generation and scaffolding
- Best practices enforcement
- Documentation generation
- Route discovery and analysis
- Dependency mapping and visualization
"""



logger = structlog.get_logger(__name__)

T = TypeVar('T')

class RouteGroup(Enum):
    """Route groups for organization"""
    AUTH = "authentication"
    USERS = "users"
    CONTENT = "content"
    ANALYTICS = "analytics"
    ADMIN = "admin"
    SYSTEM = "system"
    HEALTH = "health"
    MONITORING = "monitoring"
    API = "api"
    WEBHOOK = "webhook"
    FILE = "file"
    SEARCH = "search"
    NOTIFICATION = "notification"
    PAYMENT = "payment"
    INTEGRATION = "integration"

class RouteVersion(Enum):
    """API versions"""
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"
    BETA = "beta"
    ALPHA = "alpha"

class RouteEnvironment(Enum):
    """Route environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

@dataclass
class RouteInfo:
    """Route information"""
    path: str
    method: str
    handler: Callable
    tags: List[str] = field(default_factory=list)
    summary: str = ""
    description: str = ""
    response_model: Type[BaseModel] = None
    status_code: int = 200
    dependencies: List[Callable] = field(default_factory=list)
    group: RouteGroup = RouteGroup.API
    version: RouteVersion = RouteVersion.V1
    environment: RouteEnvironment = RouteEnvironment.PRODUCTION
    deprecated: bool = False
    rate_limit: Optional[int] = None
    cache_ttl: Optional[int] = None
    validation_schema: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class RouterInfo:
    """Router information"""
    prefix: str
    tags: List[str] = field(default_factory=list)
    group: RouteGroup = RouteGroup.API
    version: RouteVersion = RouteVersion.V1
    environment: RouteEnvironment = RouteEnvironment.PRODUCTION
    routes: List[RouteInfo] = field(default_factory=list)
    dependencies: List[Callable] = field(default_factory=list)
    middleware: List[Callable] = field(default_factory=list)
    include_in_schema: bool = True
    deprecated: bool = False

class RouteOrganizer:
    """Main route organizer"""
    
    def __init__(self, app: FastAPI):
        
    """__init__ function."""
self.app = app
        self.routers: Dict[str, APIRouter] = {}
        self.routes: Dict[str, RouteInfo] = {}
        self.route_groups: Dict[RouteGroup, List[str]] = defaultdict(list)
        self.route_versions: Dict[RouteVersion, List[str]] = defaultdict(list)
        self.route_environments: Dict[RouteEnvironment, List[str]] = defaultdict(list)
        
        # Performance tracking
        self.route_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Documentation
        self.route_docs: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Route Organizer initialized")
    
    def create_router(self, info: RouterInfo) -> APIRouter:
        """Create a new router with organization info"""
        router = APIRouter(
            prefix=info.prefix,
            tags=info.tags,
            include_in_schema=info.include_in_schema,
            deprecated=info.deprecated
        )
        
        # Store router info
        router_name = f"{info.prefix}_{info.version.value}"
        self.routers[router_name] = router
        
        logger.info(f"Created router: {router_name} in group: {info.group.value}")
        return router
    
    def register_route(self, router: APIRouter, info: RouteInfo) -> APIRouter:
        """Register a route with organization info"""
        # Add route to router
        router.add_api_route(
            path=info.path,
            endpoint=info.handler,
            methods=[info.method.upper()],
            tags=info.tags,
            summary=info.summary,
            description=info.description,
            response_model=info.response_model,
            status_code=info.status_code,
            dependencies=info.dependencies,
            deprecated=info.deprecated
        )
        
        # Store route info
        route_id = f"{info.method.upper()}_{info.path}"
        self.routes[route_id] = info
        
        # Organize routes
        self.route_groups[info.group].append(route_id)
        self.route_versions[info.version].append(route_id)
        self.route_environments[info.environment].append(route_id)
        
        # Generate documentation
        self._generate_route_documentation(info)
        
        logger.info(f"Registered route: {route_id} in group: {info.group.value}")
        return router
    
    def _generate_route_documentation(self, info: RouteInfo) -> None:
        """Generate route documentation"""
        route_id = f"{info.method.upper()}_{info.path}"
        
        self.route_docs[route_id] = {
            "path": info.path,
            "method": info.method,
            "tags": info.tags,
            "summary": info.summary,
            "description": info.description,
            "group": info.group.value,
            "version": info.version.value,
            "environment": info.environment.value,
            "deprecated": info.deprecated,
            "dependencies": [dep.__name__ for dep in info.dependencies],
            "rate_limit": info.rate_limit,
            "cache_ttl": info.cache_ttl,
            "created_at": info.created_at.isoformat(),
            "updated_at": info.updated_at.isoformat()
        }
    
    def get_routes_by_group(self, group: RouteGroup) -> List[RouteInfo]:
        """Get all routes in a specific group"""
        route_ids = self.route_groups.get(group, [])
        return [self.routes[route_id] for route_id in route_ids]
    
    def get_routes_by_version(self, version: RouteVersion) -> List[RouteInfo]:
        """Get all routes in a specific version"""
        route_ids = self.route_versions.get(version, [])
        return [self.routes[route_id] for route_id in route_ids]
    
    def get_routes_by_environment(self, environment: RouteEnvironment) -> List[RouteInfo]:
        """Get all routes in a specific environment"""
        route_ids = self.route_environments.get(environment, [])
        return [self.routes[route_id] for route_id in route_ids]
    
    def get_route_summary(self) -> Dict[str, Any]:
        """Get comprehensive route summary"""
        return {
            "total_routes": len(self.routes),
            "total_routers": len(self.routers),
            "groups": {
                group.value: len(routes) for group, routes in self.route_groups.items()
            },
            "versions": {
                version.value: len(routes) for version, routes in self.route_versions.items()
            },
            "environments": {
                env.value: len(routes) for env, routes in self.route_environments.items()
            }
        }
    
    def export_organization(self, format: str = "json") -> str:
        """Export route organization"""
        organization = {
            "routers": {
                name: {
                    "prefix": router.prefix,
                    "tags": router.tags,
                    "routes": [
                        {
                            "path": route.path,
                            "methods": route.methods,
                            "tags": route.tags,
                            "summary": route.summary,
                            "description": route.description
                        }
                        for route in router.routes
                    ]
                }
                for name, router in self.routers.items()
            },
            "routes": self.route_docs,
            "summary": self.get_route_summary()
        }
        
        if format.lower() == "yaml":
            return yaml.dump(organization, default_flow_style=False)
        else:
            return json.dumps(organization, indent=2, default=str)

# Route Builder with Organization

class OrganizedRouteBuilder:
    """Builder for creating organized routes"""
    
    def __init__(self, organizer: RouteOrganizer):
        
    """__init__ function."""
self.organizer = organizer
        self.info = RouteInfo(
            path="",
            method="GET",
            handler=None,
            tags=[],
            summary="",
            description="",
            response_model=None,
            status_code=200,
            dependencies=[],
            group=RouteGroup.API,
            version=RouteVersion.V1,
            environment=RouteEnvironment.PRODUCTION,
            deprecated=False,
            rate_limit=None,
            cache_ttl=None,
            validation_schema=None
        )
    
    def path(self, path: str) -> 'OrganizedRouteBuilder':
        """Set route path"""
        self.info.path = path
        return self
    
    def method(self, method: str) -> 'OrganizedRouteBuilder':
        """Set HTTP method"""
        self.info.method = method.upper()
        return self
    
    def handler(self, handler: Callable) -> 'OrganizedRouteBuilder':
        """Set route handler"""
        self.info.handler = handler
        return self
    
    def tags(self, tags: List[str]) -> 'OrganizedRouteBuilder':
        """Set route tags"""
        self.info.tags = tags
        return self
    
    def summary(self, summary: str) -> 'OrganizedRouteBuilder':
        """Set route summary"""
        self.info.summary = summary
        return self
    
    def description(self, description: str) -> 'OrganizedRouteBuilder':
        """Set route description"""
        self.info.description = description
        return self
    
    def response_model(self, model: Type[BaseModel]) -> 'OrganizedRouteBuilder':
        """Set response model"""
        self.info.response_model = model
        return self
    
    def status_code(self, code: int) -> 'OrganizedRouteBuilder':
        """Set status code"""
        self.info.status_code = code
        return self
    
    def dependencies(self, deps: List[Callable]) -> 'OrganizedRouteBuilder':
        """Set dependencies"""
        self.info.dependencies = deps
        return self
    
    def group(self, group: RouteGroup) -> 'OrganizedRouteBuilder':
        """Set route group"""
        self.info.group = group
        return self
    
    def version(self, version: RouteVersion) -> 'OrganizedRouteBuilder':
        """Set API version"""
        self.info.version = version
        return self
    
    def environment(self, environment: RouteEnvironment) -> 'OrganizedRouteBuilder':
        """Set environment"""
        self.info.environment = environment
        return self
    
    def deprecated(self, deprecated: bool = True) -> 'OrganizedRouteBuilder':
        """Mark route as deprecated"""
        self.info.deprecated = deprecated
        return self
    
    def rate_limit(self, limit: int) -> 'OrganizedRouteBuilder':
        """Set rate limit"""
        self.info.rate_limit = limit
        return self
    
    def cache_ttl(self, ttl: int) -> 'OrganizedRouteBuilder':
        """Set cache TTL"""
        self.info.cache_ttl = ttl
        return self
    
    def validation_schema(self, schema: Dict[str, Any]) -> 'OrganizedRouteBuilder':
        """Set validation schema"""
        self.info.validation_schema = schema
        return self
    
    def build(self, router: APIRouter) -> APIRouter:
        """Build and register organized route"""
        if not self.info.handler:
            raise ValueError("Handler must be set before building route")
        
        return self.organizer.register_route(router, self.info)

# Router Factory with Organization

class OrganizedRouterFactory:
    """Factory for creating organized routers"""
    
    def __init__(self, organizer: RouteOrganizer):
        
    """__init__ function."""
self.organizer = organizer
    
    def create_auth_router(self, prefix: str = "/auth", version: RouteVersion = RouteVersion.V1) -> APIRouter:
        """Create authentication router"""
        info = RouterInfo(
            prefix=prefix,
            tags=["authentication"],
            group=RouteGroup.AUTH,
            version=version
        )
        return self.organizer.create_router(info)
    
    def create_user_router(self, prefix: str = "/users", version: RouteVersion = RouteVersion.V1) -> APIRouter:
        """Create user management router"""
        info = RouterInfo(
            prefix=prefix,
            tags=["users"],
            group=RouteGroup.USERS,
            version=version
        )
        return self.organizer.create_router(info)
    
    def create_content_router(self, prefix: str = "/content", version: RouteVersion = RouteVersion.V1) -> APIRouter:
        """Create content management router"""
        info = RouterInfo(
            prefix=prefix,
            tags=["content"],
            group=RouteGroup.CONTENT,
            version=version
        )
        return self.organizer.create_router(info)
    
    def create_admin_router(self, prefix: str = "/admin", version: RouteVersion = RouteVersion.V1) -> APIRouter:
        """Create admin router"""
        info = RouterInfo(
            prefix=prefix,
            tags=["admin"],
            group=RouteGroup.ADMIN,
            version=version
        )
        return self.organizer.create_router(info)
    
    def create_system_router(self, prefix: str = "/system", version: RouteVersion = RouteVersion.V1) -> APIRouter:
        """Create system router"""
        info = RouterInfo(
            prefix=prefix,
            tags=["system"],
            group=RouteGroup.SYSTEM,
            version=version
        )
        return self.organizer.create_router(info)
    
    def create_health_router(self, prefix: str = "/health", version: RouteVersion = RouteVersion.V1) -> APIRouter:
        """Create health check router"""
        info = RouterInfo(
            prefix=prefix,
            tags=["health"],
            group=RouteGroup.HEALTH,
            version=version
        )
        return self.organizer.create_router(info)
    
    def create_monitoring_router(self, prefix: str = "/monitoring", version: RouteVersion = RouteVersion.V1) -> APIRouter:
        """Create monitoring router"""
        info = RouterInfo(
            prefix=prefix,
            tags=["monitoring"],
            group=RouteGroup.MONITORING,
            version=version
        )
        return self.organizer.create_router(info)
    
    def create_analytics_router(self, prefix: str = "/analytics", version: RouteVersion = RouteVersion.V1) -> APIRouter:
        """Create analytics router"""
        info = RouterInfo(
            prefix=prefix,
            tags=["analytics"],
            group=RouteGroup.ANALYTICS,
            version=version
        )
        return self.organizer.create_router(info)

# Route Discovery and Analysis

class RouteAnalyzer:
    """Analyze and discover routes"""
    
    def __init__(self, app: FastAPI):
        
    """__init__ function."""
self.app = app
    
    def discover_routes(self) -> List[Dict[str, Any]]:
        """Discover all routes in the application"""
        routes = []
        
        for route in self.app.routes:
            if isinstance(route, APIRoute):
                route_info = {
                    "path": route.path,
                    "methods": list(route.methods),
                    "tags": route.tags,
                    "summary": route.summary,
                    "description": route.description,
                    "response_model": str(route.response_model) if route.response_model else None,
                    "dependencies": [str(dep) for dep in route.dependencies],
                    "deprecated": route.deprecated
                }
                routes.append(route_info)
        
        return routes
    
    def analyze_dependencies(self) -> Dict[str, List[str]]:
        """Analyze route dependencies"""
        dependency_map = defaultdict(list)
        
        for route in self.app.routes:
            if isinstance(route, APIRoute):
                for dep in route.dependencies:
                    dep_name = dep.dependency.__name__ if hasattr(dep.dependency, '__name__') else str(dep.dependency)
                    dependency_map[dep_name].append(route.path)
        
        return dict(dependency_map)
    
    def find_unused_dependencies(self) -> List[str]:
        """Find unused dependencies"""
        all_dependencies = set()
        used_dependencies = set()
        
        # Collect all dependencies
        for route in self.app.routes:
            if isinstance(route, APIRoute):
                for dep in route.dependencies:
                    dep_name = dep.dependency.__name__ if hasattr(dep.dependency, '__name__') else str(dep.dependency)
                    all_dependencies.add(dep_name)
                    used_dependencies.add(dep_name)
        
        return list(all_dependencies - used_dependencies)
    
    def generate_route_map(self) -> Dict[str, Any]:
        """Generate comprehensive route map"""
        return {
            "routes": self.discover_routes(),
            "dependencies": self.analyze_dependencies(),
            "unused_dependencies": self.find_unused_dependencies(),
            "total_routes": len([r for r in self.app.routes if isinstance(r, APIRoute)]),
            "total_dependencies": len(set().union(*[set(dep.dependency.__name__ for dep in r.dependencies) for r in self.app.routes if isinstance(r, APIRoute)]))
        }

# Example usage

def create_organized_app() -> FastAPI:
    """Create a FastAPI app with organized routes"""
    app = FastAPI(
        title="Organized API",
        description="API with clear route organization",
        version="1.0.0"
    )
    
    # Create route organizer
    organizer = RouteOrganizer(app)
    
    # Create router factory
    router_factory = OrganizedRouterFactory(organizer)
    
    # Create route builder
    builder = OrganizedRouteBuilder(organizer)
    
    # Create organized routers
    auth_router = router_factory.create_auth_router()
    user_router = router_factory.create_user_router()
    content_router = router_factory.create_content_router()
    admin_router = router_factory.create_admin_router()
    system_router = router_factory.create_system_router()
    health_router = router_factory.create_health_router()
    monitoring_router = router_factory.create_monitoring_router()
    analytics_router = router_factory.create_analytics_router()
    
    # Example organized route definitions
    @builder.path("/login").method("POST").group(RouteGroup.AUTH).summary("User login")
    async def login():
        
    """login function."""
return {"message": "Login endpoint"}
    
    auth_router = builder.build(auth_router)
    
    @builder.path("/profile").method("GET").group(RouteGroup.USERS).summary("Get user profile")
    async def get_profile():
        
    """get_profile function."""
return {"message": "User profile"}
    
    user_router = builder.build(user_router)
    
    @builder.path("/posts").method("POST").group(RouteGroup.CONTENT).summary("Create post")
    async def create_post():
        
    """create_post function."""
return {"message": "Post created"}
    
    content_router = builder.build(content_router)
    
    @builder.path("/").method("GET").group(RouteGroup.HEALTH).summary("Health check")
    async def health_check():
        
    """health_check function."""
return {"status": "healthy"}
    
    health_router = builder.build(health_router)
    
    @builder.path("/metrics").method("GET").group(RouteGroup.MONITORING).summary("Get metrics")
    async def get_metrics():
        
    """get_metrics function."""
return {"metrics": "available"}
    
    monitoring_router = builder.build(monitoring_router)
    
    @builder.path("/analytics").method("GET").group(RouteGroup.ANALYTICS).summary("Get analytics")
    async def get_analytics():
        
    """get_analytics function."""
return {"analytics": "available"}
    
    analytics_router = builder.build(analytics_router)
    
    # Include organized routers in app
    app.include_router(auth_router)
    app.include_router(user_router)
    app.include_router(content_router)
    app.include_router(admin_router)
    app.include_router(system_router)
    app.include_router(health_router)
    app.include_router(monitoring_router)
    app.include_router(analytics_router)
    
    # Add organization endpoints
    @app.get("/organization/summary")
    async def get_organization_summary():
        """Get route organization summary"""
        return organizer.get_route_summary()
    
    @app.get("/organization/routes")
    async def get_organized_routes():
        """Get all organized routes"""
        return organizer.route_docs
    
    @app.get("/organization/groups/{group}")
    async def get_routes_by_group(group: str):
        """Get routes by group"""
        try:
            route_group = RouteGroup(group)
            routes = organizer.get_routes_by_group(route_group)
            return [{"path": route.path, "method": route.method, "summary": route.summary} for route in routes]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid group: {group}")
    
    @app.get("/organization/versions/{version}")
    async def get_routes_by_version(version: str):
        """Get routes by version"""
        try:
            route_version = RouteVersion(version)
            routes = organizer.get_routes_by_version(route_version)
            return [{"path": route.path, "method": route.method, "summary": route.summary} for route in routes]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid version: {version}")
    
    @app.get("/organization/export")
    async def export_organization(format: str = "json"):
        """Export route organization"""
        return organizer.export_organization(format)
    
    # Add route analysis
    analyzer = RouteAnalyzer(app)
    
    @app.get("/analysis/routes")
    async def analyze_routes():
        """Analyze application routes"""
        return analyzer.generate_route_map()
    
    @app.get("/analysis/dependencies")
    async def analyze_dependencies():
        """Analyze route dependencies"""
        return analyzer.analyze_dependencies()
    
    @app.get("/analysis/unused-dependencies")
    async def find_unused_dependencies():
        """Find unused dependencies"""
        return analyzer.find_unused_dependencies()
    
    return app

if __name__ == "__main__":
    app = create_organized_app()
    print("Organized FastAPI app created successfully") 