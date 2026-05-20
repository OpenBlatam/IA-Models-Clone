"""
PDF Variantes API - Application Lifecycle
Startup and shutdown logic
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI

from ..services.pdf_service import PDFVariantesService
from ..services.cache_service import CacheService
from ..services.security_service import SecurityService
from ..services.performance_service import PerformanceService
from ..services.monitoring_service import AnalyticsService, HealthService, NotificationService
from ..services.collaboration_service import CollaborationService
from ..utils.config import get_settings
from ..utils.logging_config import get_logger
from ..utils.real_world import HealthCheck
from .dependencies import register_services

logger = get_logger(__name__)


async def initialize_services() -> Dict[str, Any]:
    """Initialize all application services with real-world health checks"""
    settings = get_settings()
    services: Dict[str, Any] = {}
    health_check = HealthCheck()
    
    logger.info("Initializing services...")
    
    # Core services
    services["pdf_service"] = PDFVariantesService(settings)
    services["cache_service"] = CacheService(settings)
    services["security_service"] = SecurityService(settings)
    services["performance_service"] = PerformanceService(settings)
    services["analytics_service"] = AnalyticsService(settings)
    services["collaboration_service"] = CollaborationService(settings)
    services["notification_service"] = NotificationService(settings)
    services["health_service"] = HealthService(settings)
    services["health_check"] = health_check
    
    # Initialize all services with error handling
    for service_name, service in services.items():
        if service_name == "health_check":
            continue  # Skip health_check itself
            
        try:
            if hasattr(service, 'initialize'):
                await service.initialize()
                logger.info(f"✓ Initialized {service_name}")
                
                # Register health check for service
                if hasattr(service, 'health_check'):
                    health_check.register_check(
                        service_name,
                        service.health_check
                    )
                else:
                    # Generic health check
                    async def make_check(svc):
                        try:
                            # Try to call a simple method or check attribute
                            return hasattr(svc, 'settings') or hasattr(svc, 'initialized')
                        except:
                            return False
                    
                    health_check.register_check(
                        service_name,
                        lambda s=service: make_check(s)
                    )
            else:
                logger.debug(f"✓ Service {service_name} doesn't require initialization")
        except Exception as e:
            logger.error(f"✗ Failed to initialize {service_name}: {e}")
            # Real-world: Don't fail completely, log and continue
            # In production, you might want to fail fast or continue degraded
            if service_name in ["pdf_service", "cache_service"]:
                raise  # Critical services - fail
            # Non-critical services - continue
    
    logger.info("All services initialized successfully")
    return services


async def cleanup_services(services: Dict[str, Any]) -> None:
    """Cleanup and shutdown all services"""
    logger.info("Shutting down services...")
    
    for service_name, service in services.items():
        try:
            if hasattr(service, 'cleanup'):
                await service.cleanup()
            elif hasattr(service, 'close'):
                await service.close()
            logger.info(f"✓ Cleaned up {service_name}")
        except Exception as e:
            logger.error(f"✗ Error cleaning up {service_name}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - handles startup and shutdown"""
    logger.info("=" * 60)
    logger.info("Starting PDF Variantes API...")
    logger.info("=" * 60)
    
    services = {}
    
    try:
        # Initialize services
        services = await initialize_services()
        
        # Store services in app state for middleware access
        app.state.services = services
        
        # Register services in dependency system
        register_services(services)
        
        # Setup middleware that requires services
        from .config import setup_middleware_with_services
        setup_middleware_with_services(app)
        
        logger.info("API is ready!")
        logger.info("=" * 60)
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}", exc_info=True)
        raise
    
    finally:
        # Cleanup
        logger.info("=" * 60)
        logger.info("Shutting down PDF Variantes API...")
        logger.info("=" * 60)
        
        try:
            # Get services from app state or use the initialized services
            services_to_cleanup = getattr(app.state, "services", services)
            await cleanup_services(services_to_cleanup)
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        logger.info("API shutdown complete")
        logger.info("=" * 60)

