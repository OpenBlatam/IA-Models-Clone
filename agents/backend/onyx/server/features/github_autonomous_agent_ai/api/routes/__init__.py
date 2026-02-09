"""
Módulo de rutas
"""

from .agent_routes import router as agent_router
from .github_routes import router as github_router

__all__ = ["agent_router", "github_router"]
