"""
Search Routes - Rutas de búsqueda con Elasticsearch
====================================================

Endpoints para búsqueda avanzada usando Elasticsearch.
"""

import logging
import json
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel

from ...core.elasticsearch_client import get_elasticsearch_service
from ...core.oauth2 import get_current_active_user
from ..utils import handle_route_errors

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["search"])


class SearchRequest(BaseModel):
    """Request de búsqueda."""
    query: str
    filters: Optional[Dict[str, Any]] = None
    size: int = 10
    from_: int = 0


class SearchResponse(BaseModel):
    """Response de búsqueda."""
    results: List[Dict[str, Any]]
    total: int
    took: float


@router.post("/tasks", response_model=SearchResponse)
@handle_route_errors("searching tasks")
async def search_tasks(
    request: SearchRequest,
    current_user = Depends(get_current_active_user)
):
    """
    Buscar tareas usando Elasticsearch.
    
    Args:
        request: Request de búsqueda.
        current_user: Usuario autenticado.
    
    Returns:
        Resultados de búsqueda.
    """
    es_service = get_elasticsearch_service()
    
    if not es_service.client:
        raise HTTPException(
            status_code=503,
            detail="Elasticsearch not available"
        )
    
    results = es_service.search_tasks(
        query=request.query,
        filters=request.filters
    )
    
    return SearchResponse(
        results=results[:request.size],
        total=len(results),
        took=0.0
    )


@router.get("/tasks", response_model=SearchResponse)
@handle_route_errors("searching tasks")
async def search_tasks_get(
    q: str = Query(..., description="Query de búsqueda"),
    filters: Optional[str] = Query(None, description="Filtros JSON"),
    size: int = Query(10, ge=1, le=100),
    current_user = Depends(get_current_active_user)
):
    """Buscar tareas (GET)."""
    filters_dict = None
    if filters:
        try:
            filters_dict = json.loads(filters)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid filters JSON")
    
    request = SearchRequest(query=q, filters=filters_dict, size=size)
    return await search_tasks(request, current_user)


@router.get("/suggest")
async def suggest(
    q: str = Query(..., description="Query para sugerencias"),
    size: int = Query(5, ge=1, le=20)
):
    """Obtener sugerencias de búsqueda."""
    try:
        es_service = get_elasticsearch_service()
        
        if not es_service.client:
            return {"suggestions": []}
        
        # Implementar sugerencias con Elasticsearch
        # Por ahora, retornar vacío
        return {"suggestions": []}
    
    except Exception as e:
        logger.error(f"Suggest failed: {e}")
        return {"suggestions": []}




