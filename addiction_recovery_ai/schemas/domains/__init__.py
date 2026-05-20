"""
Schema domains - Organized by functional domain
"""

from typing import Dict, Type, Any
import importlib
import os

_schema_registry: Dict[str, Type[Any]] = {}


def register_schema(domain: str, schema_name: str, schema_class: Type[Any]) -> None:
    """Register a schema in the schema registry"""
    key = f"{domain}.{schema_name}"
    _schema_registry[key] = schema_class


def get_schema(domain: str, schema_name: str) -> Type[Any]:
    """Get a schema class from the registry"""
    key = f"{domain}.{schema_name}"
    if key in _schema_registry:
        return _schema_registry[key]
    raise ValueError(f"Schema {key} not found in registry")


def auto_discover_schemas() -> None:
    """Auto-discover and register all schemas"""
    domains_dir = os.path.dirname(__file__)
    
    for domain_name in os.listdir(domains_dir):
        domain_path = os.path.join(domains_dir, domain_name)
        if os.path.isdir(domain_path) and not domain_name.startswith('_'):
            try:
                module = importlib.import_module(f"schemas.domains.{domain_name}")
                if hasattr(module, 'register_schemas'):
                    module.register_schemas()
            except ImportError:
                pass


def get_all_schemas() -> Dict[str, Type[Any]]:
    """Get all registered schemas"""
    return _schema_registry.copy()



