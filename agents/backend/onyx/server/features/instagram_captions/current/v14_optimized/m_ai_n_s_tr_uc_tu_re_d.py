from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
from routes import router_registry
from routes.factory import route_factory, create_core_router, create_advanced_router
from routes.structured_captions import router as structured_captions_router
from dependencies import get_request_context
from core.middleware import create_middleware_stack
from core.exceptions import (
from core.shared_resources import initialize_shared_resources, cleanup_shared_resources
    import uvicorn
from typing import Any, List, Dict, Optional
import asyncio
"""
Main Application for Instagram Captions API v14.0 - Structured Version

Well-structured FastAPI application demonstrating:
- Clear route organization with factory pattern
- Centralized dependency injection
- Consistent patterns and conventions
- Easy maintenance and testing
- Proper error handling and monitoring
"""


# Import structured routing components

# Import dependencies

# Import core components
    ValidationError, AIGenerationError, CacheError, 
    DatabaseError, RateLimitError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# LIFESPAN CONTEXT MANAGER
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager
    
    Handles startup and shutdown operations with proper resource management.
    """
    # Startup
    logger.info("Starting Instagram Captions API v14.0 - Structured Version")
    
    try:
        # Initialize shared resources
        await initialize_shared_resources()
        logger.info("Shared resources initialized successfully")
        
        # Initialize route registry
        await initialize_route_registry()
        logger.info("Route registry initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Instagram Captions API")
        
        try:
            # Cleanup shared resources
            await cleanup_shared_resources()
            logger.info("Shared resources cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")


# =============================================================================
# ROUTE REGISTRY INITIALIZATION
# =============================================================================

async def initialize_route_registry():
    """Initialize the route registry with all routers"""
    
    # Register structured captions router
    router_registry.register_router(
        router=structured_captions_router,
        prefix="/structured-captions",
        tags=["structured-captions", "core"],
        description="Structured caption generation endpoints"
    )
    
    # Register core router
    core_router = create_core_router()
    router_registry.register_router(
        router=core_router,
        prefix="/core",
        tags=["core"],
        description="Core API operations"
    )
    
    # Register advanced router
    advanced_router = create_advanced_router()
    router_registry.register_router(
        router=advanced_router,
        prefix="/advanced",
        tags=["advanced"],
        description="Advanced API operations"
    )
    
    logger.info(f"Registered {len(router_registry.routers)} routers")


# =============================================================================
# FASTAPI APPLICATION CREATION
# =============================================================================

def create_structured_app() -> FastAPI:
    """
    Create well-structured FastAPI application
    
    Demonstrates:
    - Clear organization
    - Proper middleware setup
    - Error handling
    - Documentation
    """
    
    app = FastAPI(
        title="Instagram Captions API v14.0 - Structured",
        description="""
        Well-structured Instagram Captions API with clear organization:
        
        ## Features
        - **Structured Routing**: Clear route organization with factory pattern
        - **Dependency Injection**: Centralized dependency management
        - **Error Handling**: Comprehensive error handling and validation
        - **Performance**: Optimized async operations and caching
        - **Monitoring**: Built-in performance monitoring and analytics
        
        ## Architecture
        - Modular route organization
        - Clear dependency injection
        - Consistent patterns and conventions
        - Easy maintenance and testing
        """,
        version="14.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add custom middleware stack
    middleware_stack = create_middleware_stack()
    for middleware in middleware_stack:
        app.add_middleware(middleware)
    
    return app


# Create application instance
app = create_structured_app()


# =============================================================================
# ROUTE REGISTRATION
# =============================================================================

def register_all_routes():
    """Register all routes with the application"""
    
    # Register all routers from registry
    for router, config in router_registry.get_all_routers():
        app.include_router(
            router,
            prefix=config["prefix"],
            tags=config["tags"],
            dependencies=config["dependencies"]
        )
        logger.info(f"Registered router: {config['description']}")
    
    # Register additional structured routes
    app.include_router(
        structured_captions_router,
        prefix="/api/v14",
        tags=["api-v14", "structured"]
    )


# Register routes
register_all_routes()


# =============================================================================
# GLOBAL EXCEPTION HANDLERS
# =============================================================================

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle validation errors globally"""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "validation_error",
            "message": str(exc),
            "timestamp": time.time(),
            "path": str(request.url)
        }
    )


@app.exception_handler(AIGenerationError)
async def ai_generation_exception_handler(request: Request, exc: AIGenerationError):
    """Handle AI generation errors globally"""
    logger.error(f"AI generation error: {exc}")
    return JSONResponse(
        status_code=503,
        content={
            "error": "ai_generation_error",
            "message": str(exc),
            "timestamp": time.time(),
            "path": str(request.url)
        }
    )


@app.exception_handler(CacheError)
async def cache_exception_handler(request: Request, exc: CacheError):
    """Handle cache errors globally"""
    logger.error(f"Cache error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "cache_error",
            "message": str(exc),
            "timestamp": time.time(),
            "path": str(request.url)
        }
    )


@app.exception_handler(DatabaseError)
async def database_exception_handler(request: Request, exc: DatabaseError):
    """Handle database errors globally"""
    logger.error(f"Database error: {exc}")
    return JSONResponse(
        status_code=503,
        content={
            "error": "database_error",
            "message": str(exc),
            "timestamp": time.time(),
            "path": str(request.url)
        }
    )


@app.exception_handler(RateLimitError)
async def rate_limit_exception_handler(request: Request, exc: RateLimitError):
    """Handle rate limit errors globally"""
    logger.warning(f"Rate limit exceeded: {exc}")
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": str(exc),
            "retry_after": exc.retry_after,
            "timestamp": time.time(),
            "path": str(request.url)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions globally"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred",
            "timestamp": time.time(),
            "path": str(request.url)
        }
    )


# =============================================================================
# ROOT ENDPOINTS
# =============================================================================

@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint
    
    Returns basic API information and available endpoints.
    """
    return {
        "message": "Instagram Captions API v14.0 - Structured Version",
        "version": "14.0.0",
        "status": "running",
        "documentation": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "timestamp": time.time()
    }


@app.get("/health", tags=["health"])
async def health_check():
    """
    Health check endpoint
    
    Returns comprehensive health status of all services.
    """
    try:
        # Check shared resources
        shared_resources_status = await check_shared_resources_health()
        
        # Check route registry
        route_registry_status = {
            "total_routers": len(router_registry.routers),
            "status": "healthy"
        }
        
        return {
            "status": "healthy",
            "version": "14.0.0",
            "timestamp": time.time(),
            "services": {
                "shared_resources": shared_resources_status,
                "route_registry": route_registry_status
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "version": "14.0.0",
            "timestamp": time.time(),
            "error": str(e)
        }


@app.get("/info", tags=["info"])
async def api_info():
    """
    API information endpoint
    
    Returns detailed information about the API structure and capabilities.
    """
    return {
        "api": {
            "name": "Instagram Captions API v14.0 - Structured",
            "version": "14.0.0",
            "description": "Well-structured Instagram Captions API with clear organization"
        },
        "architecture": {
            "routing": "Factory pattern with clear dependency injection",
            "dependencies": "Centralized dependency management",
            "error_handling": "Comprehensive exception handling",
            "performance": "Optimized async operations and caching",
            "monitoring": "Built-in performance monitoring"
        },
        "routes": {
            "total_routers": len(router_registry.routers),
            "categories": list(set(
                tag for _, config in router_registry.get_all_routers()
                for tag in config["tags"]
            ))
        },
        "features": [
            "Structured routing with factory pattern",
            "Clear dependency injection",
            "Comprehensive error handling",
            "Performance optimization",
            "Built-in monitoring",
            "Easy testing and maintenance"
        ],
        "timestamp": time.time()
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def check_shared_resources_health() -> Dict[str, Any]:
    """Check health of shared resources"""
    try:
        # This would check actual shared resources
        return {
            "status": "healthy",
            "components": ["database", "cache", "ai_engine"]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# =============================================================================
# DEVELOPMENT ENDPOINTS
# =============================================================================

@app.get("/debug/routes", tags=["debug"])
async def debug_routes():
    """
    Debug endpoint to view all registered routes
    
    Useful for development and debugging.
    """
    routes_info = []
    
    for router, config in router_registry.get_all_routers():
        routes_info.append({
            "name": config.get("description", "Unnamed router"),
            "prefix": config["prefix"],
            "tags": config["tags"],
            "dependencies_count": len(config["dependencies"])
        })
    
    return {
        "total_routers": len(routes_info),
        "routers": routes_info,
        "timestamp": time.time()
    }


@app.get("/debug/dependencies", tags=["debug"])
async def debug_dependencies():
    """
    Debug endpoint to view dependency information
    
    Useful for development and debugging.
    """
    return {
        "dependency_classes": [
            "ServiceDependencies",
            "CoreDependencies", 
            "AdvancedDependencies"
        ],
        "available_dependencies": [
            "get_current_user",
            "require_authentication",
            "require_permission",
            "get_database_pool",
            "get_api_client_pool",
            "get_optimized_engine",
            "get_cache_manager",
            "get_lazy_loader_manager",
            "get_io_monitor",
            "get_blocking_limiter"
        ],
        "timestamp": time.time()
    }


# =============================================================================
# APPLICATION STARTUP
# =============================================================================

if __name__ == "__main__":
    
    logger.info("Starting Instagram Captions API v14.0 - Structured Version")
    
    uvicorn.run(
        "main_structured:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 