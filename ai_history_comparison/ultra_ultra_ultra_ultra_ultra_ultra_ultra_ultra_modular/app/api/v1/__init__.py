"""
API v1 package initialization.
"""

from .analysis_router import router as analysis_router
from .plugin_router import router as plugin_router
from .system_router import router as system_router
from .advanced_router import router as advanced_router
from .optimized_router import router as optimized_router
from .lightning_router import router as lightning_router
from .ultra_speed_router import router as ultra_speed_router
from .hyper_performance_router import router as hyper_performance_router
from .ultimate_optimization_router import router as ultimate_optimization_router
from .extreme_router import router as extreme_router
from .infinite_router import router as infinite_router

__all__ = [
    "analysis_router",
    "plugin_router",
    "system_router",
    "advanced_router",
    "optimized_router",
    "lightning_router",
    "ultra_speed_router",
    "hyper_performance_router",
    "ultimate_optimization_router",
    "extreme_router",
    "infinite_router"
]
