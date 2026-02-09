"""
PDF Variantes API - Dependencies
FastAPI dependency injection for services
"""

from typing import Dict, Any
from fastapi import Depends

from ..services.pdf_service import PDFVariantesService
from ..services.cache_service import CacheService
from ..services.security_service import SecurityService
from ..services.performance_service import PerformanceService
from ..services.monitoring_service import AnalyticsService, HealthService, NotificationService
from ..services.collaboration_service import CollaborationService

# Global services registry (populated during lifespan)
_services_registry: Dict[str, Any] = {}


def register_services(services: Dict[str, Any]) -> None:
    """Register services in the global registry"""
    _services_registry.clear()
    _services_registry.update(services)


def get_services() -> Dict[str, Any]:
    """Get all services - FastAPI dependency"""
    return _services_registry


# Individual service dependencies
def get_pdf_service(services: Dict[str, Any] = Depends(get_services)) -> PDFVariantesService:
    """Get PDF service dependency"""
    service = services.get("pdf_service")
    if not service:
        raise RuntimeError("PDF service not available")
    return service


def get_cache_service(services: Dict[str, Any] = Depends(get_services)) -> CacheService:
    """Get cache service dependency"""
    service = services.get("cache_service")
    if not service:
        raise RuntimeError("Cache service not available")
    return service


def get_security_service(services: Dict[str, Any] = Depends(get_services)) -> SecurityService:
    """Get security service dependency"""
    service = services.get("security_service")
    if not service:
        raise RuntimeError("Security service not available")
    return service


def get_performance_service(services: Dict[str, Any] = Depends(get_services)) -> PerformanceService:
    """Get performance service dependency"""
    service = services.get("performance_service")
    if not service:
        raise RuntimeError("Performance service not available")
    return service


def get_analytics_service(services: Dict[str, Any] = Depends(get_services)) -> AnalyticsService:
    """Get analytics service dependency"""
    service = services.get("analytics_service")
    if not service:
        raise RuntimeError("Analytics service not available")
    return service


def get_collaboration_service(services: Dict[str, Any] = Depends(get_services)) -> CollaborationService:
    """Get collaboration service dependency"""
    service = services.get("collaboration_service")
    if not service:
        raise RuntimeError("Collaboration service not available")
    return service


def get_notification_service(services: Dict[str, Any] = Depends(get_services)) -> NotificationService:
    """Get notification service dependency"""
    service = services.get("notification_service")
    if not service:
        raise RuntimeError("Notification service not available")
    return service


def get_health_service(services: Dict[str, Any] = Depends(get_services)) -> HealthService:
    """Get health service dependency"""
    service = services.get("health_service")
    if not service:
        raise RuntimeError("Health service not available")
    return service

