"""
REST API for Imagen Video Enhancer AI
=====================================

REST API wrapper for the image/video enhancement agent.
"""

import logging
from fastapi import FastAPI

from .app_factory import create_app

logger = logging.getLogger(__name__)

# Create application using factory
app, _ = create_app()
