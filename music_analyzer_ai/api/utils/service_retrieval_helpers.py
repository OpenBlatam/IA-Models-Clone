"""
Service retrieval helper functions.

This module provides utilities for retrieving and managing service dependencies
in a consistent way across different router patterns.
"""

from typing import Any, List, Tuple, Optional, Dict
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


def get_required_services(
    router_instance: Any,
    service_names: List[str],
    raise_on_missing: bool = True
) -> Tuple[Any, ...]:
    """
    Get multiple required services from a router instance.
    
    This is a convenience wrapper around router.get_services() that provides
    better error handling and logging.
    
    Args:
        router_instance: Router instance with get_services() method
        service_names: List of service names to retrieve
        raise_on_missing: Whether to raise exception if service not found
    
    Returns:
        Tuple of services in the same order as service_names
    
    Raises:
        AttributeError: If router doesn't have get_services method
        Exception: If service not found and raise_on_missing=True
    
    Example:
        spotify, analyzer, coach = get_required_services(
            self,
            ["spotify_service", "music_analyzer", "music_coach"]
        )
    """
    if not hasattr(router_instance, 'get_services'):
        if raise_on_missing:
            raise AttributeError(
                f"Router instance {type(router_instance)} does not have get_services method"
            )
        return tuple()
    
    try:
        services = router_instance.get_services(*service_names)
        return services
    except Exception as e:
        if raise_on_missing:
            logger.error(f"Failed to retrieve services {service_names}: {e}")
            raise
        logger.warning(f"Failed to retrieve services {service_names}: {e}")
        return tuple()


def get_optional_services(
    router_instance: Any,
    service_names: List[str]
) -> Dict[str, Optional[Any]]:
    """
    Get multiple optional services, returning None for missing ones.
    
    Args:
        router_instance: Router instance with get_service_optional() method
        service_names: List of service names to retrieve
    
    Returns:
        Dictionary mapping service names to service instances (or None)
    
    Example:
        services = get_optional_services(
            self,
            ["webhook_service", "analytics_service", "history_service"]
        )
        webhook = services.get("webhook_service")
        if webhook:
            webhook.trigger(...)
    """
    result = {}
    
    if not hasattr(router_instance, 'get_service_optional'):
        logger.warning(
            f"Router instance {type(router_instance)} does not have get_service_optional method"
        )
        return {name: None for name in service_names}
    
    for service_name in service_names:
        try:
            result[service_name] = router_instance.get_service_optional(service_name)
        except Exception as e:
            logger.warning(f"Failed to retrieve optional service {service_name}: {e}")
            result[service_name] = None
    
    return result


def validate_services_available(
    services: Dict[str, Optional[Any]],
    required: List[str]
) -> None:
    """
    Validate that required services are available.
    
    Args:
        services: Dictionary of service name -> service instance
        required: List of required service names
    
    Raises:
        ValueError: If any required service is None or missing
    
    Example:
        services = get_optional_services(self, ["spotify", "analyzer"])
        validate_services_available(services, required=["spotify", "analyzer"])
    """
    missing = []
    for service_name in required:
        if service_name not in services or services[service_name] is None:
            missing.append(service_name)
    
    if missing:
        raise ValueError(f"Required services not available: {missing}")


def get_service_or_default(
    router_instance: Any,
    service_name: str,
    default: Any = None
) -> Any:
    """
    Get a service or return default if not available.
    
    Args:
        router_instance: Router instance
        service_name: Name of service to retrieve
        default: Default value if service not available
    
    Returns:
        Service instance or default
    
    Example:
        webhook = get_service_or_default(self, "webhook_service", default=None)
        if webhook:
            await webhook.trigger(...)
    """
    if hasattr(router_instance, 'get_service_optional'):
        return router_instance.get_service_optional(service_name) or default
    
    if hasattr(router_instance, 'get_service'):
        try:
            return router_instance.get_service(service_name)
        except Exception:
            return default
    
    return default








