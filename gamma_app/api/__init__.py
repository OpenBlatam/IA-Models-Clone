"""
Gamma App - API Module
RESTful API for AI-powered content generation
"""

from .main import app
from .routes import content_router, collaboration_router, export_router
from .models import ContentRequest, ContentResponse, User, Project

__all__ = [
    'app',
    'content_router',
    'collaboration_router', 
    'export_router',
    'ContentRequest',
    'ContentResponse',
    'User',
    'Project'
]

__version__ = "1.0.0"



























