"""
API Layer - FastAPI routers and middleware
Modular API endpoints organized by domain
"""

from .main import app, create_app
from .routes import router

__all__ = ["app", "create_app", "router"]






