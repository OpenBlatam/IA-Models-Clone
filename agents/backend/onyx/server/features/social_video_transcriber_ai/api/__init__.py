"""API module for Social Video Transcriber AI"""

from .routes import router
from .main import app, create_app

__all__ = ["router", "app", "create_app"]












