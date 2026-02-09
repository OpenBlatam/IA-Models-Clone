"""API routers module"""
from .root import router as root_router
from .health import router as health_router
from .conversion import router as conversion_router
from .formats import router as formats_router
from .cache import router as cache_router
from .metrics import router as metrics_router
from .templates import router as templates_router
from .validation import router as validation_router
from .security import router as security_router

__all__ = [
    "root_router",
    "health_router",
    "conversion_router",
    "formats_router",
    "cache_router",
    "metrics_router",
    "templates_router",
    "validation_router",
    "security_router"
]
