"""
API Dependencies
================

Shared dependencies for FastAPI routers.
"""

from typing import Annotated
from fastapi import Depends
import logging

from ..core.clothing_changer_service import ClothingChangerService
from ..config.clothing_changer_config import ClothingChangerConfig

logger = logging.getLogger(__name__)

# Global service instance (initialized on app startup)
_service_instance: ClothingChangerService | None = None


def get_service_instance() -> ClothingChangerService:
    """
    Get or create the global service instance.
    
    Returns:
        ClothingChangerService instance
    """
    global _service_instance
    
    if _service_instance is None:
        logger.info("Creating new ClothingChangerService instance")
        config = ClothingChangerConfig.from_env()
        _service_instance = ClothingChangerService(config=config)
    
    return _service_instance


def set_service_instance(service: ClothingChangerService) -> None:
    """
    Set the global service instance (used during app startup).
    
    Args:
        service: Service instance to set
    """
    global _service_instance
    _service_instance = service
    logger.info("Service instance set globally")


def get_service() -> Annotated[ClothingChangerService, Depends(get_service_instance)]:
    """
    FastAPI dependency to get service instance.
    
    Returns:
        ClothingChangerService instance
    """
    return get_service_instance()


# Type alias for dependency injection
ServiceDep = Annotated[ClothingChangerService, Depends(get_service)]
