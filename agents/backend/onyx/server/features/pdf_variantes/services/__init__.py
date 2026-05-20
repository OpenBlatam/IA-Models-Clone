"""
PDF Variantes Services
Service imports and initialization
"""

from .pdf_service import PDFVariantesService
from .collaboration_service import CollaborationService
from .cache_service import CacheService
from .security_service import SecurityService
from .performance_service import PerformanceService
from .monitoring_service import (
    MonitoringSystem, AnalyticsService, HealthService, NotificationService,
    Metric, Alert
)

# Service factory functions
async def create_pdf_service(settings):
    """Create PDF service instance"""
    service = PDFVariantesService(settings)
    await service.initialize()
    return service

async def create_collaboration_service(settings):
    """Create collaboration service instance"""
    service = CollaborationService(settings)
    await service.initialize()
    return service

async def create_monitoring_system(settings):
    """Create monitoring system instance"""
    system = MonitoringSystem(settings)
    await system.initialize()
    return system

async def create_analytics_service(settings):
    """Create analytics service instance"""
    service = AnalyticsService(settings)
    await service.initialize()
    return service

async def create_health_service(settings):
    """Create health service instance"""
    service = HealthService(settings)
    await service.initialize()
    return service

async def create_notification_service(settings):
    """Create notification service instance"""
    service = NotificationService(settings)
    await service.initialize()
    return service

# Service registry
SERVICE_REGISTRY = {
    "pdf_service": create_pdf_service,
    "collaboration_service": create_collaboration_service,
    "monitoring_system": create_monitoring_system,
    "analytics_service": create_analytics_service,
    "health_service": create_health_service,
    "notification_service": create_notification_service
}

async def initialize_all_services(settings):
    """Initialize all services"""
    services = {}
    
    for service_name, factory_func in SERVICE_REGISTRY.items():
        try:
            service = await factory_func(settings)
            services[service_name] = service
            logger.info(f"Initialized {service_name}")
        except Exception as e:
            logger.error(f"Failed to initialize {service_name}: {e}")
            raise
    
    return services

async def cleanup_all_services(services):
    """Cleanup all services"""
    for service_name, service in services.items():
        try:
            if hasattr(service, 'cleanup'):
                await service.cleanup()
            logger.info(f"Cleaned up {service_name}")
        except Exception as e:
            logger.error(f"Error cleaning up {service_name}: {e}")

__all__ = [
    "PDFVariantesService",
    "CollaborationService",
    "CacheService",
    "SecurityService",
    "PerformanceService",
    "MonitoringSystem",
    "AnalyticsService",
    "HealthService",
    "NotificationService",
    "Metric",
    "Alert",
    "create_pdf_service",
    "create_collaboration_service",
    "create_monitoring_system",
    "create_analytics_service",
    "create_health_service",
    "create_notification_service",
    "initialize_all_services",
    "cleanup_all_services"
]
