"""
Rutas Avanzadas para Funcionalidades Adicionales
==================================================

Endpoints para comparación, extracción estructurada, análisis de estilo, etc.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Form
from pydantic import BaseModel, Field

from ..core.document_analyzer import DocumentAnalyzer
from ..core.document_comparator import DocumentComparator
from ..core.structured_extractor import StructuredExtractor, ExtractionSchema
from ..core.style_analyzer import StyleAnalyzer
from ..utils.exporters import ResultExporter
from .routes import get_analyzer

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/advanced",
    tags=["Advanced Features"]
)


# ============================================================================
# Modelos Pydantic
# ============================================================================

class CompareRequest(BaseModel):
    """Request para comparar documentos"""
    document1_content: str = Field(..., description="Contenido del primer documento")
    document2_content: str = Field(..., description="Contenido del segundo documento")
    document1_id: Optional[str] = Field(None, description="ID del primer documento")
    document2_id: Optional[str] = Field(None, description="ID del segundo documento")
    include_analysis: bool = Field(True, description="Incluir análisis detallado")


class FindSimilarRequest(BaseModel):
    """Request para encontrar documentos similares"""
    target_document: str = Field(..., description="Documento objetivo")
    corpus: List[Dict[str, str]] = Field(..., description="Corpus de documentos (id, content)")
    threshold: float = Field(0.7, description="Umbral de similitud")
    top_k: int = Field(5, description="Número de resultados")


class PlagiarismRequest(BaseModel):
    """Request para detección de plagio"""
    suspicious_document: str = Field(..., description="Documento sospechoso")
    reference_corpus: List[Dict[str, str]] = Field(..., description="Corpus de referencia")
    threshold: float = Field(0.85, description="Umbral para considerar plagio")


class ExtractStructuredRequest(BaseModel):
    """Request para extracción estructurada"""
    content: str = Field(..., description="Contenido del documento")
    schema: List[Dict[str, Any]] = Field(..., description="Schema de extracción")


class ExportRequest(BaseModel):
    """Request para exportar resultados"""
    data: Dict[str, Any] = Field(..., description="Datos a exportar")
    format: str = Field("json", description="Formato: json, csv, markdown, html")
    filename: Optional[str] = Field(None, description="Nombre del archivo")


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/compare")
async def compare_documents(
    request: CompareRequest,
    analyzer: DocumentAnalyzer = Depends(get_analyzer)
):
    """Comparar dos documentos"""
    try:
        comparator = DocumentComparator(analyzer)
        result = await comparator.compare_documents(
            request.document1_content,
            request.document2_content,
            request.document1_id,
            request.document2_id,
            request.include_analysis
        )
        
        return {
            "similarity_score": result.similarity_score,
            "common_keywords": result.common_keywords,
            "common_entities": result.common_entities,
            "differences": result.differences,
            "timestamp": result.timestamp
        }
    except Exception as e:
        logger.error(f"Error comparando documentos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/find-similar")
async def find_similar_documents(
    request: FindSimilarRequest,
    analyzer: DocumentAnalyzer = Depends(get_analyzer)
):
    """Encontrar documentos similares en un corpus"""
    try:
        comparator = DocumentComparator(analyzer)
        
        # Convertir corpus a formato esperado
        corpus = [(doc["id"], doc["content"]) for doc in request.corpus]
        
        results = await comparator.find_similar_documents(
            request.target_document,
            corpus,
            request.threshold,
            request.top_k
        )
        
        return [
            {
                "document_id": r.document2_id,
                "similarity_score": r.similarity_score,
                "timestamp": r.timestamp
            }
            for r in results
        ]
    except Exception as e:
        logger.error(f"Error encontrando documentos similares: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-plagiarism")
async def detect_plagiarism(
    request: PlagiarismRequest,
    analyzer: DocumentAnalyzer = Depends(get_analyzer)
):
    """Detectar posible plagio"""
    try:
        comparator = DocumentComparator(analyzer)
        
        # Convertir corpus a formato esperado
        corpus = [(doc["id"], doc["content"]) for doc in request.reference_corpus]
        
        results = await comparator.detect_plagiarism(
            request.suspicious_document,
            corpus,
            request.threshold
        )
        
        return {
            "plagiarism_detected": len(results) > 0,
            "matches": results,
            "total_matches": len(results)
        }
    except Exception as e:
        logger.error(f"Error detectando plagio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-structured")
async def extract_structured_data(
    request: ExtractStructuredRequest,
    analyzer: DocumentAnalyzer = Depends(get_analyzer)
):
    """Extraer información estructurada según schema"""
    try:
        extractor = StructuredExtractor(analyzer)
        
        # Crear schema
        schema = extractor.create_schema(request.schema)
        
        # Extraer datos
        result = await extractor.extract_structured_data(
            request.content,
            schema
        )
        
        return result
    except Exception as e:
        logger.error(f"Error extrayendo datos estructurados: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-style")
async def analyze_style(
    content: str = Form(...),
    analyzer: DocumentAnalyzer = Depends(get_analyzer)
):
    """Analizar estilo de escritura"""
    try:
        style_analyzer = StyleAnalyzer(analyzer)
        result = await style_analyzer.analyze_writing_style(content)
        
        return result
    except Exception as e:
        logger.error(f"Error analizando estilo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/assess-quality")
async def assess_quality(
    content: str = Form(...),
    criteria: Optional[List[str]] = Form(None),
    analyzer: DocumentAnalyzer = Depends(get_analyzer)
):
    """Evaluar calidad del documento"""
    try:
        style_analyzer = StyleAnalyzer(analyzer)
        result = await style_analyzer.assess_quality(content, criteria)
        
        return result
    except Exception as e:
        logger.error(f"Error evaluando calidad: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export")
async def export_results(
    request: ExportRequest,
    analyzer: DocumentAnalyzer = Depends(get_analyzer)
):
    """Exportar resultados en diferentes formatos"""
    try:
        import tempfile
        import os
        
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{request.format}",
            prefix=request.filename or "export_"
        ) as tmp_file:
            output_path = tmp_file.name
        
        # Exportar según formato
        if request.format == "json":
            ResultExporter.export_json(request.data, output_path)
        elif request.format == "csv":
            data_list = request.data if isinstance(request.data, list) else [request.data]
            ResultExporter.export_csv(data_list, output_path)
        elif request.format == "markdown":
            ResultExporter.export_markdown(request.data, output_path)
        elif request.format == "html":
            ResultExporter.export_html(request.data, output_path)
        else:
            raise ValueError(f"Formato no soportado: {request.format}")
        
        # Leer archivo y retornar
        with open(output_path, "rb") as f:
            content = f.read()
        
        # Limpiar
        os.unlink(output_path)
        
        from fastapi.responses import Response
        return Response(
            content=content,
            media_type={
                "json": "application/json",
                "csv": "text/csv",
                "markdown": "text/markdown",
                "html": "text/html"
            }.get(request.format, "application/octet-stream"),
            headers={
                "Content-Disposition": f"attachment; filename={request.filename or 'export'}.{request.format}"
            }
        )
    except Exception as e:
        logger.error(f"Error exportando resultados: {e}")
        raise HTTPException(status_code=500, detail=str(e))
















