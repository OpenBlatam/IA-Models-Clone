"""
Support and motivation routes module
"""

from fastapi import APIRouter

try:
    from .endpoints import router
except ImportError:
    # Fallback if endpoints.py doesn't exist yet
    router = APIRouter(prefix="/support", tags=["Support & Motivation"])

__all__ = ["router"]

