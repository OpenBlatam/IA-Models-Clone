"""
API de Búsqueda Avanzada

Endpoints para:
- Búsqueda full-text
- Búsqueda fuzzy
- Autocompletado
- Filtros avanzados
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query

from services.search_engine import get_search_engine
from middleware.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/search",
    tags=["search"]
)


@router.post("/index")
async def index_document(
    doc_id: str,
    content: Dict[str, Any],
    text_fields: Optional[List[str]] = None,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Indexa un documento para búsqueda (requiere autenticación).
    """
    try:
        search_engine = get_search_engine()
        search_engine.index_document(doc_id, content, text_fields)
        
        return {
            "message": "Document indexed successfully",
            "doc_id": doc_id
        }
    except Exception as e:
        logger.error(f"Error indexing document: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error indexing document: {str(e)}"
        )


@router.get("/query")
async def search(
    q: str = Query(..., description="Query de búsqueda"),
    filters: Optional[str] = Query(None, description="Filtros JSON"),
    limit: int = Query(20, ge=1, le=100, description="Límite de resultados"),
    offset: int = Query(0, ge=0, description="Offset"),
    sort_by: Optional[str] = Query(None, description="Campo para ordenar"),
    fuzzy: bool = Query(False, description="Usar búsqueda fuzzy")
) -> Dict[str, Any]:
    """
    Busca documentos.
    """
    try:
        import json
        
        search_engine = get_search_engine()
        
        # Parsear filtros si se proporcionan
        parsed_filters = None
        if filters:
            try:
                parsed_filters = json.loads(filters)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid filters JSON"
                )
        
        results = search_engine.search(
            query=q,
            filters=parsed_filters,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            fuzzy=fuzzy
        )
        
        return {
            "query": q,
            "results": [
                {
                    "id": result.id,
                    "score": result.score,
                    "data": result.data,
                    "highlights": result.highlights
                }
                for result in results
            ],
            "total": len(results),
            "limit": limit,
            "offset": offset
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing search: {str(e)}"
        )


@router.get("/autocomplete")
async def autocomplete(
    prefix: str = Query(..., description="Prefijo para autocompletado"),
    limit: int = Query(10, ge=1, le=50, description="Límite de sugerencias")
) -> Dict[str, Any]:
    """
    Obtiene sugerencias de autocompletado.
    """
    try:
        search_engine = get_search_engine()
        suggestions = search_engine.autocomplete(prefix, limit=limit)
        
        return {
            "prefix": prefix,
            "suggestions": suggestions,
            "count": len(suggestions)
        }
    except Exception as e:
        logger.error(f"Error getting autocomplete: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting autocomplete: {str(e)}"
        )


@router.delete("/index/{doc_id}")
async def remove_document(
    doc_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Elimina un documento del índice (requiere autenticación).
    """
    try:
        search_engine = get_search_engine()
        search_engine.remove_document(doc_id)
        
        return {
            "message": f"Document {doc_id} removed from index"
        }
    except Exception as e:
        logger.error(f"Error removing document: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error removing document: {str(e)}"
        )


@router.get("/stats")
async def get_search_stats() -> Dict[str, Any]:
    """
    Obtiene estadísticas del índice de búsqueda.
    """
    try:
        search_engine = get_search_engine()
        stats = search_engine.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting search stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stats: {str(e)}"
        )

