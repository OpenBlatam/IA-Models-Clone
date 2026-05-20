"""
API v1
"""

from fastapi import APIRouter
from .dermatology_v1 import router as v1_router

__all__ = ["v1_router"]






