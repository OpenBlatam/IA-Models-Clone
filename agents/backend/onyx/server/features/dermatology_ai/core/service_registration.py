"""
Service registration utilities for composition root
Extracted from composition_root.py to reduce duplication
"""

from typing import Dict, Any, Type, Optional
import logging

from .service_factory import ServiceFactory
from .domain.interfaces import (
    IAnalysisRepository,
    IUserRepository,
    IProductRepository,
    IImageProcessor,
    IAnalysisService,
    IRecommendationService,
    ICacheService,
    IEventPublisher,
)

logger = logging.getLogger(__name__)


class ServiceRegistration:
    """Utility class for registering services in the service factory"""
    
    @staticmethod
    def register_repositories(
        service_factory: ServiceFactory,
        repositories: Dict[str, Any]
    ) -> None:
        """Register all repositories"""
        repository_mappings = {
            "analysis_repository": (IAnalysisRepository, repositories["analysis_repository"]),
            "user_repository": (IUserRepository, repositories["user_repository"]),
            "product_repository": (IProductRepository, repositories["product_repository"]),
        }
        
        for service_name, (interface, instance) in repository_mappings.items():
            service_factory.register_singleton(service_name, interface, instance)
    
    @staticmethod
    def register_adapters(
        service_factory: ServiceFactory,
        adapters: Dict[str, Any]
    ) -> None:
        """Register all adapters"""
        adapter_mappings = {
            "image_processor": (IImageProcessor, adapters.get("image_processor")),
            "cache": (ICacheService, adapters.get("cache")),
            "event_publisher": (IEventPublisher, adapters.get("event_publisher")),
        }
        
        for service_name, (interface, instance) in adapter_mappings.items():
            if instance:
                service_factory.register_singleton(service_name, interface, instance)
    
    @staticmethod
    def register_domain_services(
        service_factory: ServiceFactory,
        services: Dict[str, Any]
    ) -> None:
        """Register domain services"""
        service_mappings = {
            "analysis_service": (IAnalysisService, services.get("analysis_service")),
            "recommendation_service": (IRecommendationService, services.get("recommendation_service")),
        }
        
        for service_name, (interface, instance) in service_mappings.items():
            if instance and not isinstance(instance, Exception):
                service_factory.register_singleton(service_name, interface, instance)







