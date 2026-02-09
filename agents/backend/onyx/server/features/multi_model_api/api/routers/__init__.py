"""
Routers for Multi-Model API
Separated routers for better organization
"""

from .execution import router as execution_router
from .models import router as models_router
from .health import router as health_router
from .cache import router as cache_router
from .rate_limit import router as rate_limit_router
from .metrics import router as metrics_router
from .metrics_advanced import router as metrics_advanced_router
from .performance import router as performance_router
from .openrouter import router as openrouter_router
from .batch import router as batch_router
from .streaming import router as streaming_router

__all__ = [
    "execution_router",
    "models_router",
    "health_router",
    "cache_router",
    "rate_limit_router",
    "metrics_router",
    "metrics_advanced_router",
    "performance_router",
    "openrouter_router",
    "batch_router",
    "streaming_router"
]

