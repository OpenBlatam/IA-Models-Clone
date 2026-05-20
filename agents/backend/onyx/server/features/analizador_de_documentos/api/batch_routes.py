"""
Rutas para Procesamiento por Lotes
====================================

Endpoints para procesar múltiples documentos en paralelo.
"""

import logging
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.document_analyzer import DocumentAnalyzer, AnalysisTask
from ..utils.batch_processor import BatchProcessor
from .routes import get_analyzer

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/batch",
    tags=["Batch Processing"]
)


class BatchAnalyzeRequest(BaseModel):
    """Request para análisis por lotes"""
    documents: List[Dict[str, Any]] = Field(
        ...,
        description="Lista de documentos a analizar. Cada documento debe tener 'content' y opcionalmente 'tasks', 'document_id'"
    )
    tasks: List[str] = Field(
        None,
        description="Tareas por defecto para todos los documentos"
    )
    max_workers: int = Field(10, description="Número máximo de workers concurrentes")
    batch_size: int = Field(100, description="Tamaño de cada lote")


@router.post("/analyze")
async def batch_analyze(
    request: BatchAnalyzeRequest,
    analyzer: DocumentAnalyzer = Depends(get_analyzer)
):
    """
    Analizar múltiples documentos en paralelo
    
    Procesa documentos en lotes para optimizar el rendimiento.
    """
    try:
        # Preparar tareas
        default_tasks = None
        if request.tasks:
            default_tasks = [AnalysisTask(task) for task in request.tasks]
        
        # Crear procesador de lotes
        batch_processor = BatchProcessor(
            max_workers=request.max_workers,
            batch_size=request.batch_size
        )
        
        # Función para procesar un documento
        async def process_document(doc_data: Dict[str, Any]):
            tasks = default_tasks
            if "tasks" in doc_data and doc_data["tasks"]:
                tasks = [AnalysisTask(task) for task in doc_data["tasks"]]
            
            result = await analyzer.analyze_document(
                document_content=doc_data.get("content"),
                document_type=doc_data.get("document_type"),
                tasks=tasks,
                document_id=doc_data.get("document_id")
            )
            
            return result
        
        # Procesar lotes
        result = await batch_processor.process_batch(
            request.documents,
            process_document
        )
        
        return {
            "total": result.total,
            "successful": result.successful,
            "failed": result.failed,
            "processing_time": result.processing_time,
            "results": [r.__dict__ if hasattr(r, "__dict__") else r for r in result.results],
            "errors": result.errors
        }
    except Exception as e:
        logger.error(f"Error en procesamiento por lotes: {e}")
        raise HTTPException(status_code=500, detail=str(e))
















