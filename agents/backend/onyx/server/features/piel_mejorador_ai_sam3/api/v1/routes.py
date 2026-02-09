"""
API v1 Routes
============

Versioned API routes.
"""

from fastapi import APIRouter
from ..piel_mejorador_api import app

# Create v1 router
v1_router = APIRouter(prefix="/v1", tags=["v1"])

# Import and include routes from main API
# This allows versioning while maintaining backward compatibility




