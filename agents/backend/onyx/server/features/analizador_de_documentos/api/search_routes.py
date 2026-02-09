"""
Rutas para Búsqueda Semántica
===============================

Endpoints para búsqueda semántica avanzada.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.semantic_search import SemanticSearchEngine
from .routes import get_analyzer

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/search",
    tags=["Semantic Search"]
)


class IndexDocumentRequest(BaseModel):
    """Request para indexar documento"""
    document_id: str = Field(..., description="ID del documento")
    content: str = Field(..., description="Contenido del documento")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata adicional")


class SearchRequest(BaseModel):
    """Request para búsqueda"""
    query: str = Field(..., description="Consulta de búsqueda")
    top_k: int = Field(10, description="Número de resultados")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filtros por metadata")
    use_hybrid: bool = Field(True, description="Usar búsqueda híbrida")


# Instancia global del motor de búsqueda
_search_engine: Optional[SemanticSearchEngine] = None


def get_search_engine() -> SemanticSearchEngine:
    """Dependency para obtener motor de búsqueda"""
    global _search_engine
    if _search_engine is None:
        analyzer = get_analyzer()
        _search_engine = SemanticSearchEngine(analyzer)
    return _search_engine


@router.post("/index")
async def index_document(
    request: IndexDocumentRequest,
    engine: SemanticSearchEngine = Depends(get_search_engine)
):
    """Indexar documento para búsqueda"""
    try:
        engine.index_document(
            request.document_id,
            request.content,
            request.metadata
        )
        
        return {
            "status": "indexed",
            "document_id": request.document_id,
            "index_stats": engine.get_index_stats()
        }
    except Exception as e:
        logger.error(f"Error indexando documento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_documents(
    request: SearchRequest,
    engine: SemanticSearchEngine = Depends(get_search_engine)
):
    """Buscar documentos"""
    try:
        results = await engine.search(
            request.query,
            request.top_k,
            request.filters,
            request.use_hybrid
        )
        
        return {
            "query": request.query,
            "results_count": len(results),
            "results": [
                {
                    "document_id": r.document_id,
                    "score": r.score,
                    "content": r.content,
                    "metadata": r.metadata,
                    "highlights": r.highlights
                }
                for r in results
            ]
        }
    except Exception as e:
        logger.error(f"Error buscando documentos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_search_stats(
    engine: SemanticSearchEngine = Depends(get_search_engine)
):
    """Obtener estadísticas del índice"""
    return engine.get_index_stats()


@router.delete("/index/{document_id}")
async def remove_document(
    document_id: str,
    engine: SemanticSearchEngine = Depends(get_search_engine)
):
    """Remover documento del índice"""
    try:
        engine.remove_document(document_id)
        return {"status": "removed", "document_id": document_id}
    except Exception as e:
        logger.error(f"Error removiendo documento: {e}")
        raise HTTPException(status_code=500, detail=str(e))
















