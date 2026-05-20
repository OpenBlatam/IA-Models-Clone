from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .config import ModularConfig, get_config
from .service import ModularCopywritingService, get_service
from .cache import CacheManager, get_cache_manager
from .optimization import OptimizationDetector, get_optimization_level
from .api import create_api_router
from typing import Any, List, Dict, Optional
import logging
import asyncio
# -*- coding: utf-8 -*-
"""
Modular Copywriting System
=========================

Sistema de copywriting modular y escalable para producción.

Architecture:
- optimization/ - Motor de optimización
- cache/ - Sistema de cache multi-nivel  
- models/ - Modelos de datos
- services/ - Servicios principales
- config/ - Configuración
- utils/ - Utilidades
"""


__version__ = "1.0.0"
__author__ = "Production Team"
__all__ = [
    "ModularConfig",
    "ModularCopywritingService", 
    "CacheManager",
    "OptimizationDetector",
    "get_config",
    "get_service",
    "get_cache_manager",
    "get_optimization_level",
    "create_api_router"
] 