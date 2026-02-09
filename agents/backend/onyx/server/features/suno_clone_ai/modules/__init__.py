"""
Sistema de Módulos Modulares
Cada módulo es independiente y puede ser desplegado como microservicio
"""

from modules.registry import ModuleRegistry, get_module_registry
from modules.base import BaseModule, ModuleConfig

__all__ = [
    "ModuleRegistry",
    "get_module_registry",
    "BaseModule",
    "ModuleConfig",
]















