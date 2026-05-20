"""
Search Routes - Rutas para búsqueda y filtrado avanzado.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from api.utils import handle_api_errors
from core.services.search_service import SearchService, SearchFilter, SearchOperator
from config.logging_config import get_logger
from config.di_setup import get_service

router = APIRouter()
logger = get_logger(__name__)


class SearchRequest(BaseModel):
    """Request para búsqueda."""
    query: Optional[str] = Field(None, description="Query de texto")
    filters: Optional[List[Dict[str, Any]]] = Field(None, description="Filtros")
    sort_by: Optional[str] = Field(None, description="Campo para ordenar")
    sort_order: str = Field(default="asc", description="Orden: asc/desc")
    limit: Optional[int] = Field(None, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


def get_search_service() -> SearchService:
    """Obtener servicio de búsqueda."""
    try:
        return get_service("search_service")
    except Exception:
        raise HTTPException(status_code=503, detail="Search service no disponible")


@router.post("/tasks")
@handle_api_errors
async def search_tasks(
    request: SearchRequest,
    search_service: SearchService = Depends(get_search_service)
):
    """
    Buscar tareas.
    
    Args:
        request: Parámetros de búsqueda
        
    Returns:
        Resultados de búsqueda
    """
    # Obtener tareas del storage
    try:
        from config.di_setup import get_service
        storage = get_service("storage")
        tasks = await storage.get_all_tasks()
    except Exception as e:
        logger.error(f"Error obteniendo tareas: {e}", exc_info=True)
        tasks = []
    
    # Convertir a formato de búsqueda
    search_items = [
        {
            "id": t.get("id"),
            "repository": f"{t.get('repository_owner')}/{t.get('repository_name')}",
            "instruction": t.get("instruction"),
            "status": t.get("status"),
            "created_at": t.get("created_at"),
            "error": t.get("error")
        }
        for t in tasks
    ]
    
    # Construir filtros
    filters = None
    if request.filters:
        filters = []
        for filter_data in request.filters:
            try:
                operator = SearchOperator(filter_data.get("operator", "contains"))
                filters.append(SearchFilter(
                    field=filter_data["field"],
                    operator=operator,
                    value=filter_data["value"]
                ))
            except Exception as e:
                logger.warning(f"Error creando filtro: {e}")
    
    # Buscar
    results = search_service.search(
        items=search_items,
        query=request.query,
        filters=filters,
        sort_by=request.sort_by,
        sort_order=request.sort_order,
        limit=request.limit,
        offset=request.offset
    )
    
    return results


@router.get("/history")
@handle_api_errors
async def get_search_history(
    limit: int = Query(10, ge=1, le=100),
    search_service: SearchService = Depends(get_search_service)
):
    """
    Obtener historial de búsquedas.
    
    Args:
        limit: Número máximo de resultados
        
    Returns:
        Historial de búsquedas
    """
    history = search_service.get_search_history(limit=limit)
    return {"history": history}


@router.get("/stats")
@handle_api_errors
async def get_search_stats(
    search_service: SearchService = Depends(get_search_service)
):
    """
    Obtener estadísticas de búsqueda.
    
    Returns:
        Estadísticas
    """
    return search_service.get_stats()



