"""
Core Package
============

Core functionality for the Document Workflow Chain system.
"""

from .app import create_app
from .config import settings
from .database import get_database, init_database
from .container import container

__all__ = [
    "create_app",
    "settings", 
    "get_database",
    "init_database",
    "container"
]


