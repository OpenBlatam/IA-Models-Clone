"""
API module for Artist Manager AI
"""

from .app_factory import create_app
from .routes import router

__all__ = [
    "create_app",
    "router",
]
