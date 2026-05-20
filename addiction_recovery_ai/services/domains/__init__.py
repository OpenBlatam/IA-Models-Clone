"""
Service domains - Organized by functional domain
"""

from typing import Dict, Type, Any
import importlib
import os

_service_registry: Dict[str, Type[Any]] = {}


def register_service(domain: str, service_name: str, service_class: Type[Any]) -> None:
    """Register a service in the service registry"""
    key = f"{domain}.{service_name}"
    _service_registry[key] = service_class


def get_service(domain: str, service_name: str) -> Any:
    """Get a service instance from the registry"""
    key = f"{domain}.{service_name}"
    if key in _service_registry:
        return _service_registry[key]()
    raise ValueError(f"Service {key} not found in registry")


def auto_discover_services() -> None:
    """Auto-discover and register all services"""
    domains_dir = os.path.dirname(__file__)
    
    for domain_name in os.listdir(domains_dir):
        domain_path = os.path.join(domains_dir, domain_name)
        if os.path.isdir(domain_path) and not domain_name.startswith('_'):
            try:
                module = importlib.import_module(f"services.domains.{domain_name}")
                if hasattr(module, 'register_services'):
                    module.register_services()
            except ImportError:
                pass


def get_all_services() -> Dict[str, Type[Any]]:
    """Get all registered services"""
    return _service_registry.copy()



