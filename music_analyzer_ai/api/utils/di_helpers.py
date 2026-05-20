"""
DI Helpers for Routers

Helper functions to get services from DI container in routers.
This provides a consistent way to access services across all routers.
"""

import logging
from typing import Any, Optional

from ...core.di import get_container
from ..factories import get_service

logger = logging.getLogger(__name__)


def get_service_from_di(service_name: str) -> Any:
    """
    Get a service from DI container.
    
    This is a wrapper around get_container().get() that provides
    better error handling and logging.
    
    Args:
        service_name: Name of the service to retrieve
    
    Returns:
        Service instance
    
    Raises:
        Exception: If service is not found and is required
    """
    try:
        container = get_container()
        return container.get(service_name)
    except Exception as e:
        logger.error(f"Failed to get service '{service_name}' from DI container: {e}")
        raise


def get_service_optional(service_name: str) -> Optional[Any]:
    """
    Get an optional service from DI container.
    
    Returns None if service is not available instead of raising an exception.
    
    Args:
        service_name: Name of the service to retrieve
    
    Returns:
        Service instance or None if not available
    """
    try:
        return get_service_from_di(service_name)
    except Exception:
        logger.debug(f"Optional service '{service_name}' not available")
        return None


def get_multiple_services(*service_names: str) -> tuple:
    """
    Get multiple services from DI container at once.
    
    Args:
        *service_names: Variable number of service names
    
    Returns:
        Tuple of services in the same order as service_names
    
    Example:
        spotify, analyzer = get_multiple_services("spotify_service", "music_analyzer")
    """
    try:
        container = get_container()
        return tuple(container.get(name) for name in service_names)
    except Exception as e:
        logger.error(f"Failed to get services {service_names}: {e}")
        raise




