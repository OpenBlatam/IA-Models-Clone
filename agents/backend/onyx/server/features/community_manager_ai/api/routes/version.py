"""
Version API Routes
==================

Endpoints para información de versiones.
"""

from fastapi import APIRouter
from ..versioning import get_api_info

router = APIRouter(prefix="/version", tags=["version"])


@router.get("/", response_model=dict)
async def get_version():
    """Obtener información de versiones de la API"""
    return get_api_info()




