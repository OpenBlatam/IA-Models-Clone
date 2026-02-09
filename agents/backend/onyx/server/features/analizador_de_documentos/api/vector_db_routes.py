"""
Rutas para Base de Datos Vectorial
====================================

Endpoints para integración con bases de datos vectoriales.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
import numpy as np

from ..core.vector_database import VectorDatabase
from .routes import get_analyzer
import os

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/vector-db",
    tags=["Vector Database"]
)


class IndexDocumentRequest(BaseModel):
    """Request para indexar en base vectorial"""
    document_id: str = Field(..., description="ID del documento")
    content: str = Field(..., description="Contenido del documento")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata")


class SearchVectorRequest(BaseModel):
    """Request para búsqueda vectorial"""
    query: str = Field(..., description="Consulta")
    top_k: int = Field(10, description="Número de resultados")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filtros")


# Instancia global
_vector_db: Optional[VectorDatabase] = None


def get_vector_db() -> VectorDatabase:
    """Dependency para obtener base de datos vectorial"""
    global _vector_db
    if _vector_db is None:
        backend = os.getenv("VECTOR_DB_BACKEND", "memory")
        _vector_db = VectorDatabase(backend=backend)
    return _vector_db


@router.post("/index")
async def index_document_vector(
    request: IndexDocumentRequest,
    analyzer = Depends(get_analyzer),
    vector_db: VectorDatabase = Depends(get_vector_db)
):
    """Indexar documento en base vectorial"""
    try:
        # Generar embedding
        embedding = await analyzer.embedding_generator.generate_embeddings([request.content])
        if isinstance(embedding, list):
            embedding = embedding[0]
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        # Indexar
        await vector_db.add_document(
            request.document_id,
            embedding,
            request.content,
            request.metadata
        )
        
        return {"status": "indexed", "document_id": request.document_id}
    except Exception as e:
        logger.error(f"Error indexando en base vectorial: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_vector(
    request: SearchVectorRequest,
    analyzer = Depends(get_analyzer),
    vector_db: VectorDatabase = Depends(get_vector_db)
):
    """Buscar en base vectorial"""
    try:
        # Generar embedding de la consulta
        query_embedding = await analyzer.embedding_generator.generate_embeddings([request.query])
        if isinstance(query_embedding, list):
            query_embedding = query_embedding[0]
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        
        # Buscar
        results = await vector_db.search(
            query_embedding,
            request.top_k,
            request.filters
        )
        
        return {
            "query": request.query,
            "results_count": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Error buscando en base vectorial: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/index/{document_id}")
async def delete_document_vector(
    document_id: str,
    vector_db: VectorDatabase = Depends(get_vector_db)
):
    """Eliminar documento de base vectorial"""
    try:
        vector_db.delete_document(document_id)
        return {"status": "deleted", "document_id": document_id}
    except Exception as e:
        logger.error(f"Error eliminando de base vectorial: {e}")
        raise HTTPException(status_code=500, detail=str(e))

