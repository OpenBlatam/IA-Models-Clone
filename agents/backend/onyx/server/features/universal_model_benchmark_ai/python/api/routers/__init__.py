"""
API Routers - Modular routers for REST API endpoints.

This package provides organized routers for different
resource types in the REST API.
"""

from .results import router as results_router
from .experiments import router as experiments_router
from .models import router as models_router
from .distributed import router as distributed_router
from .costs import router as costs_router
from .webhooks import router as webhooks_router

# Router list for easy registration
ALL_ROUTERS = [
    results_router,
    experiments_router,
    models_router,
    distributed_router,
    costs_router,
    webhooks_router,
]

__all__ = [
    "results_router",
    "experiments_router",
    "models_router",
    "distributed_router",
    "costs_router",
    "webhooks_router",
    "ALL_ROUTERS",
]

