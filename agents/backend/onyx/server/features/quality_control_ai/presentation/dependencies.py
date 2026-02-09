"""
FastAPI Dependencies

Dependency injection for FastAPI routes.
"""

from functools import lru_cache
from ...application.factories import ApplicationServiceFactory
from ...application.services import InspectionApplicationService

# Global factory instance
_factory: ApplicationServiceFactory = None


def get_application_service_factory() -> ApplicationServiceFactory:
    """
    Get or create application service factory.
    
    Returns:
        ApplicationServiceFactory instance
    """
    global _factory
    if _factory is None:
        _factory = ApplicationServiceFactory()
    return _factory


def get_inspection_service() -> InspectionApplicationService:
    """
    Get inspection application service (FastAPI dependency).
    
    Returns:
        InspectionApplicationService instance
    """
    factory = get_application_service_factory()
    return factory.create_inspection_application_service()


@lru_cache()
def get_config():
    """Get configuration (cached)."""
    from ...config import Config, create_default_config_file
    try:
        return Config.from_yaml("config/default_config.yaml")
    except:
        # Create default if not exists
        create_default_config_file("config/default_config.yaml")
        return Config.from_yaml("config/default_config.yaml")



