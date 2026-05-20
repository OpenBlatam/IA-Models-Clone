"""
Modular Dermatology AI API - Refactored with service locator and router pattern
This is the new modular version that replaces the monolithic dermatology_api.py
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from typing import Dict, Any
import logging

from .services_locator import get_service_locator, ServiceLocator
from .routers.router_manager import RouterManager, get_router_manager
from .routers import (
    analysis_router,
    recommendations_router,
    tracking_router,
    products_router,
    ml_router,
    integrations_router,
    reports_router,
    social_router,
    health_router,
    performance_router,
    auth_router,
)

logger = logging.getLogger(__name__)

# Create main router
router = APIRouter(prefix="/dermatology", tags=["dermatology"])


def initialize_services() -> ServiceLocator:
    """Initialize all service instances and register them in the service locator"""
    from .service_definitions import get_service_definitions
    from .service_registry import get_service_registry
    
    service_definitions = get_service_definitions()
    registry = get_service_registry()
    
    for service_name, config in service_definitions.items():
        factory = config["factory"]
        group = config.get("group")
        registry.register(service_name, factory, group=group)
    
    services = registry.initialize_all()
    
    filtered_services = {k: v for k, v in services.items() if v is not None}
    
    service_locator = get_service_locator()
    service_locator.initialize_services(filtered_services)
    
    logger.info(f"Initialized {len(filtered_services)} services in service locator")
    return service_locator


def register_routers(router_manager: RouterManager):
    """Register all modular routers"""
    routers_config = [
        (auth_router, "auth", "User authentication, registration, and login endpoints"),
        (analysis_router, "analysis", "Image, video, texture, and advanced analysis endpoints"),
        (recommendations_router, "recommendations", "All recommendation endpoints"),
        (tracking_router, "tracking", "Progress, habits, side effects tracking endpoints"),
        (products_router, "products", "Product-related endpoints"),
        (ml_router, "ml-ai", "ML/AI advanced analysis endpoints"),
        (integrations_router, "integrations", "IoT, wearable, pharmacy integration endpoints"),
        (reports_router, "reports", "Report generation and visualization endpoints"),
        (social_router, "social", "Gamification and social features endpoints"),
        (health_router, "health", "Health monitoring and system status endpoints"),
        (performance_router, "performance", "Performance monitoring and optimization endpoints"),
    ]
    
    for router_instance, name, description in routers_config:
        router_manager.register_router(
            router_instance,
            name=name,
            prefix="/dermatology",
            tags=[name],
            description=description
        )
    
    logger.info(f"Registered {len(routers_config)} modular routers successfully")


def create_modular_router() -> APIRouter:
    """Create and configure the main modular router"""
    # Initialize services
    initialize_services()
    
    # Get router manager
    router_manager = get_router_manager()
    
    # Register all routers
    register_routers(router_manager)
    
    # Include all registered routers in main router
    for registered_router in router_manager.get_all_routers():
        router.include_router(registered_router)
    
    # Add router info endpoint
    @router.get("/router-info")
    async def get_router_info():
        """Get information about all registered routers"""
        router_manager = get_router_manager()
        return JSONResponse(content={
            "success": True,
            "routers": router_manager.get_router_info(),
            "endpoints_summary": router_manager.get_endpoints_summary()
        })
    
    logger.info("Modular router created successfully")
    return router


# Create the modular router instance
modular_router = create_modular_router()

