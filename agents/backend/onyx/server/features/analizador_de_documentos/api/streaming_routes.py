"""
Rutas para Streaming de Resultados
====================================

Endpoints para streaming de resultados grandes.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json
import asyncio

from ..core.document_analyzer import DocumentAnalyzer
from .routes import get_analyzer

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/stream",
    tags=["Streaming"]
)


class StreamAnalysisRequest(BaseModel):
    """Request para análisis con streaming"""
    content: str = Field(..., description="Contenido del documento")
    tasks: list[str] = Field(..., description="Tareas de análisis")
    chunk_size: int = Field(1000, description="Tamaño de chunk para streaming")


@router.post("/analyze")
async def stream_analysis(
    request: StreamAnalysisRequest,
    analyzer: DocumentAnalyzer = Depends(get_analyzer)
):
    """Analizar documento con streaming de resultados"""
    try:
        async def generate_stream():
            # Enviar inicio
            yield json.dumps({"status": "started", "message": "Análisis iniciado"}) + "\n"
            
            # Procesar tareas una por una y enviar resultados
            from ..core.document_analyzer import AnalysisTask
            tasks = [AnalysisTask(task) for task in request.tasks]
            
            for task in tasks:
                try:
                    if task == AnalysisTask.CLASSIFICATION:
                        result = await analyzer.classify_document(request.content)
                        yield json.dumps({
                            "task": "classification",
                            "result": result
                        }, ensure_ascii=False) + "\n"
                    
                    elif task == AnalysisTask.SUMMARIZATION:
                        result = await analyzer.summarize_document(request.content)
                        yield json.dumps({
                            "task": "summarization",
                            "result": result
                        }, ensure_ascii=False) + "\n"
                    
                    elif task == AnalysisTask.KEYWORD_EXTRACTION:
                        result = await analyzer.extract_keywords(request.content)
                        yield json.dumps({
                            "task": "keyword_extraction",
                            "result": result
                        }, ensure_ascii=False) + "\n"
                    
                    elif task == AnalysisTask.SENTIMENT:
                        result = await analyzer.analyze_sentiment(request.content)
                        yield json.dumps({
                            "task": "sentiment",
                            "result": result
                        }, ensure_ascii=False) + "\n"
                    
                    # Agregar delay pequeño para efecto de streaming
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    yield json.dumps({
                        "task": task.value,
                        "error": str(e)
                    }) + "\n"
            
            # Enviar fin
            yield json.dumps({"status": "completed", "message": "Análisis completado"}) + "\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="application/x-ndjson",
            headers={"X-Content-Type-Options": "nosniff"}
        )
    except Exception as e:
        logger.error(f"Error en streaming: {e}")
        raise HTTPException(status_code=500, detail=str(e))
















