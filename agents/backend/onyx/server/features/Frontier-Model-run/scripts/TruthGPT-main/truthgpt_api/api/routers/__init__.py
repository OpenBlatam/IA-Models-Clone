"""
API Routers
===========

FastAPI routers for API endpoints.
"""

from .health import router as health_router
from .models import router as models_router
from .batch import router as batch_router
from .utils import router as utils_router
from .advanced import router as advanced_router
from .version import router as version_router
from .analysis import router as analysis_router
from .search import router as search_router

__all__ = [
    'health_router', 'models_router', 'batch_router', 'utils_router',
    'advanced_router', 'version_router', 'analysis_router', 'search_router'
]

