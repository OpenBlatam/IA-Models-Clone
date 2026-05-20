"""
Resource library routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional

try:
    from services.resource_library_service import ResourceLibraryService
except ImportError:
    from ...services.resource_library_service import ResourceLibraryService

router = APIRouter()

resource_library = ResourceLibraryService()


@router.get("/resources/library")
async def get_resources(
    resource_type: Optional[str] = Query(None),
    topic: Optional[str] = Query(None)
):
    """Obtiene recursos educativos"""
    try:
        resources = resource_library.get_resources(resource_type, topic)
        return JSONResponse(content={
            "resources": resources,
            "total": len(resources),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo recursos: {str(e)}")


@router.get("/resources/search")
async def search_resources(query: str = Query(...)):
    """Busca recursos educativos"""
    try:
        results = resource_library.search_resources(query)
        return JSONResponse(content={
            "query": query,
            "results": results,
            "total": len(results),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error buscando recursos: {str(e)}")



