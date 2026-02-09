"""
API module for Plastic Surgery Visualization AI
"""

from .routes.visualization import router as visualization_router
from .routes.comparison import router as comparison_router
from .routes.batch import router as batch_router
from .routes.health import router as health_router
from .routes.metrics import router as metrics_router
from .routes.info import router as info_router

__all__ = [
    "visualization_router",
    "comparison_router",
    "batch_router",
    "health_router",
    "metrics_router",
    "info_router",
]