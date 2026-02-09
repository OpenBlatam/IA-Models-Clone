"""
Addiction Recovery AI - Sistema de IA para ayudar a dejar adicciones
=====================================================================

Enhanced with PyTorch, Transformers, and Deep Learning.
Ultra-Modular Layered Architecture V8 with Complete Component Separation.

Features:
- Advanced AI models for addiction recovery support
- PyTorch and Transformers integration
- Deep learning capabilities
- Modular architecture with component separation
"""

__version__ = "3.11.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI system for addiction recovery support, enhanced with PyTorch, Transformers, and Deep Learning"

# Import from organized submodules
from ._core_exports import *
from ._models_exports import *
from ._layers_exports import *
from ._core_modules_exports import *
from ._core_system_exports import *
from ._utils_exports import *
from ._training_exports import *

# Re-export all public symbols
__all__ = []

# Collect all exports from submodules
for module_name in [
    "_core_exports",
    "_models_exports",
    "_layers_exports",
    "_core_modules_exports",
    "_core_system_exports",
    "_utils_exports",
    "_training_exports",
]:
    try:
        module = __import__(f".{module_name}", fromlist=["__all__"], level=1)
        if hasattr(module, "__all__"):
            __all__.extend(module.__all__)
    except ImportError:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Could not import {module_name}")

# Remove duplicates while preserving order
seen = set()
__all__ = [x for x in __all__ if not (x in seen or seen.add(x))]
