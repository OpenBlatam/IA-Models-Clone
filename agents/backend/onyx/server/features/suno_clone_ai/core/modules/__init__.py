"""
Sistema de Módulos

Organiza funcionalidades en módulos independientes y reutilizables.
"""

from .module_registry import ModuleRegistry, get_module_registry
from .base_module import BaseModule

__all__ = [
    "ModuleRegistry",
    "get_module_registry",
    "BaseModule"
]

