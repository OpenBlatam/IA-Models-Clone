"""
Advanced Module System
Dynamic module loading with dependencies, isolation, and lifecycle management
"""

from .module import *
from .module_registry import *
from .module_loader import *

__all__ = [
    "Module",
    "ModuleMetadata",
    "ModuleRegistry",
    "ModuleLoader",
    "get_module_registry",
    "get_module_loader",
]















