"""
Rutas para RAG System
=======================

Endpoints para RAG (Retrieval-Augmented Generation).
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.rag_system import (
    get_rag_system,
    RAGSystem,
    RetrievalMethod
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/rag",
    tags=["RAG System"]
)


class AddDocumentsRequest(BaseModel):
    """Request para agregar documentos"""
    documents: List[Dict[str, Any]] = Field(..., description="Documentos")


class RetrieveAndGenerateRequest(BaseModel):
    """Request para recuperar y generar"""
    query_text: str = Field(..., description="Texto de consulta")
    retrieval_method: str = Field("hybrid", description="Método")
    top_k: int = Field(5, description="Top K")


@router.post("/documents")
async def add_documents(
    request: AddDocumentsRequest,
    system: RAGSystem = Depends(get_rag_system)
):
    """Agregar documentos al store"""
    try:
        system.add_documents(request.documents)
        
        return {"status": "added", "documents": len(request.documents)}
    except Exception as e:
        logger.error(f"Error agregando documentos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieve-and-generate")
async def retrieve_and_generate(
    request: RetrieveAndGenerateRequest,
    system: RAGSystem = Depends(get_rag_system)
):
    """Recuperar y generar respuesta"""
    try:
        method = RetrievalMethod(request.retrieval_method)
        result = system.retrieve_and_generate(
            request.query_text,
            method,
            request.top_k
        )
        
        return {
            "query_id": result.query_id,
            "retrieved_documents": result.retrieved_documents,
            "generated_answer": result.generated_answer,
            "sources": result.sources,
            "confidence": result.confidence
        }
    except Exception as e:
        logger.error(f"Error en RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


