"""
Utility categories - Organized by functional domain
"""

from typing import Dict, Type, Any
import importlib
import os

_utility_registry: Dict[str, Type[Any]] = {}


def register_utility(category: str, utility_name: str, utility_class: Type[Any]) -> None:
    """Register a utility in the utility registry"""
    key = f"{category}.{utility_name}"
    _utility_registry[key] = utility_class


def get_utility(category: str, utility_name: str) -> Any:
    """Get a utility instance from the registry"""
    key = f"{category}.{utility_name}"
    if key in _utility_registry:
        return _utility_registry[key]()
    raise ValueError(f"Utility {key} not found in registry")


def auto_discover_utilities() -> None:
    """Auto-discover and register all utilities"""
    categories_dir = os.path.dirname(__file__)
    
    for category_name in os.listdir(categories_dir):
        category_path = os.path.join(categories_dir, category_name)
        if os.path.isdir(category_path) and not category_name.startswith('_'):
            try:
                module = importlib.import_module(f"utils.categories.{category_name}")
                if hasattr(module, 'register_utilities'):
                    module.register_utilities()
            except ImportError:
                pass


def get_all_utilities() -> Dict[str, Type[Any]]:
    """Get all registered utilities"""
    return _utility_registry.copy()



