"""
Relapse prevention routes module
"""

from fastapi import APIRouter

try:
    from .endpoints import router
except ImportError:
    # Fallback if endpoints.py doesn't exist yet
    router = APIRouter(prefix="/relapse", tags=["Relapse Prevention"])

__all__ = ["router"]

