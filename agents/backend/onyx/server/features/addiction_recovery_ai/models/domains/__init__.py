"""
Model domains - Organized by functional domain
"""

from typing import Dict, Type, Any
import importlib
import os

_model_registry: Dict[str, Type[Any]] = {}


def register_model(domain: str, model_name: str, model_class: Type[Any]) -> None:
    """Register a model in the model registry"""
    key = f"{domain}.{model_name}"
    _model_registry[key] = model_class


def get_model(domain: str, model_name: str) -> Type[Any]:
    """Get a model class from the registry"""
    key = f"{domain}.{model_name}"
    if key in _model_registry:
        return _model_registry[key]
    raise ValueError(f"Model {key} not found in registry")


def auto_discover_models() -> None:
    """Auto-discover and register all models"""
    domains_dir = os.path.dirname(__file__)
    
    for domain_name in os.listdir(domains_dir):
        domain_path = os.path.join(domains_dir, domain_name)
        if os.path.isdir(domain_path) and not domain_name.startswith('_'):
            try:
                module = importlib.import_module(f"models.domains.{domain_name}")
                if hasattr(module, 'register_models'):
                    module.register_models()
            except ImportError:
                pass


def get_all_models() -> Dict[str, Type[Any]]:
    """Get all registered models"""
    return _model_registry.copy()



