"""
Document API - API REST para Document Analyzer
==============================================

API REST completa para acceso externo al analizador de documentos.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Pydantic Models
class DocumentAnalysisRequest(BaseModel):
    """Request para análisis de documento."""
    document_content: Optional[str] = None
    document_path: Optional[str] = None
    document_type: Optional[str] = None
    tasks: Optional[List[str]] = None
    include_raw: bool = False


class DocumentVersionRequest(BaseModel):
    """Request para agregar versión."""
    document_id: str
    content: str
    version_id: Optional[str] = None
    author: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BatchAnalysisRequest(BaseModel):
    """Request para análisis batch."""
    documents: List[Dict[str, Any]]
    tasks: Optional[List[str]] = None
    max_workers: int = 10


class RecommendationRequest(BaseModel):
    """Request para recomendaciones."""
    document_analysis: Dict[str, Any]
    quality_analysis: Optional[Dict[str, Any]] = None
    grammar_analysis: Optional[Dict[str, Any]] = None


class APIResponse(BaseModel):
    """Response estándar de API."""
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class DocumentAPIServer:
    """Servidor API REST para Document Analyzer."""
    
    def __init__(self, analyzer, host: str = "0.0.0.0", port: int = 8000):
        """Inicializar servidor API."""
        self.analyzer = analyzer
        self.host = host
        self.port = port
        self.app = FastAPI(
            title="Document Analyzer API",
            description="API REST completa para análisis de documentos",
            version="1.0.0"
        )
        self._setup_routes()
    
    def _setup_routes(self):
        """Configurar rutas de la API."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "service": "Document Analyzer API",
                "version": "1.0.0",
                "status": "running"
            }
        
        @self.app.post("/api/v1/analyze", response_model=APIResponse)
        async def analyze_document(request: DocumentAnalysisRequest):
            """Analizar documento."""
            try:
                result = await self.analyzer.analyze_document(
                    document_content=request.document_content,
                    document_path=request.document_path,
                    tasks=request.tasks
                )
                
                return APIResponse(
                    success=True,
                    data=result.__dict__ if hasattr(result, '__dict__') else result,
                    message="Análisis completado exitosamente"
                )
            except Exception as e:
                logger.error(f"Error en análisis: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/analyze/quality", response_model=APIResponse)
        async def analyze_quality(request: DocumentAnalysisRequest):
            """Analizar calidad de documento."""
            try:
                if not request.document_content:
                    raise HTTPException(status_code=400, detail="document_content requerido")
                
                result = await self.analyzer.analyze_quality(
                    content=request.document_content,
                    document_type=request.document_type
                )
                
                return APIResponse(
                    success=True,
                    data=result.__dict__ if hasattr(result, '__dict__') else result,
                    message="Análisis de calidad completado"
                )
            except Exception as e:
                logger.error(f"Error en análisis de calidad: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/analyze/grammar", response_model=APIResponse)
        async def analyze_grammar(request: DocumentAnalysisRequest):
            """Analizar gramática de documento."""
            try:
                if not request.document_content:
                    raise HTTPException(status_code=400, detail="document_content requerido")
                
                result = await self.analyzer.analyze_grammar(
                    content=request.document_content
                )
                
                return APIResponse(
                    success=True,
                    data=result.__dict__ if hasattr(result, '__dict__') else result,
                    message="Análisis gramatical completado"
                )
            except Exception as e:
                logger.error(f"Error en análisis gramatical: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/versions/add", response_model=APIResponse)
        async def add_version(request: DocumentVersionRequest):
            """Agregar versión de documento."""
            try:
                version = self.analyzer.add_document_version(
                    document_id=request.document_id,
                    content=request.content,
                    version_id=request.version_id,
                    author=request.author,
                    metadata=request.metadata
                )
                
                return APIResponse(
                    success=True,
                    data=version.__dict__ if hasattr(version, '__dict__') else version,
                    message="Versión agregada exitosamente"
                )
            except Exception as e:
                logger.error(f"Error agregando versión: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/versions/{document_id}/compare", response_model=APIResponse)
        async def compare_versions(document_id: str, v1: str, v2: str):
            """Comparar versiones."""
            try:
                result = await self.analyzer.compare_document_versions(
                    document_id=document_id,
                    version1_id=v1,
                    version2_id=v2
                )
                
                return APIResponse(
                    success=True,
                    data=result.__dict__ if hasattr(result, '__dict__') else result,
                    message="Comparación completada"
                )
            except Exception as e:
                logger.error(f"Error comparando versiones: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/batch/analyze", response_model=APIResponse)
        async def analyze_batch(request: BatchAnalysisRequest):
            """Analizar batch de documentos."""
            try:
                result = await self.analyzer.process_batch(
                    documents=request.documents,
                    tasks=request.tasks,
                    max_workers=request.max_workers
                )
                
                return APIResponse(
                    success=True,
                    data=result.__dict__ if hasattr(result, '__dict__') else result,
                    message=f"Batch de {len(request.documents)} documentos procesado"
                )
            except Exception as e:
                logger.error(f"Error en batch analysis: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/recommendations/generate", response_model=APIResponse)
        async def generate_recommendations(request: RecommendationRequest):
            """Generar recomendaciones."""
            try:
                # Convertir dicts a objetos si es necesario
                doc_analysis = request.document_analysis
                quality = request.quality_analysis
                grammar = request.grammar_analysis
                
                result = await self.analyzer.generate_recommendations(
                    document_analysis=doc_analysis,
                    quality_analysis=quality,
                    grammar_analysis=grammar
                )
                
                return APIResponse(
                    success=True,
                    data=[r.__dict__ if hasattr(r, '__dict__') else r for r in result],
                    message=f"{len(result)} recomendaciones generadas"
                )
            except Exception as e:
                logger.error(f"Error generando recomendaciones: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/collaboration/{document_id}", response_model=APIResponse)
        async def analyze_collaboration(document_id: str):
            """Analizar colaboración."""
            try:
                result = await self.analyzer.analyze_collaboration(document_id=document_id)
                
                return APIResponse(
                    success=True,
                    data=result.__dict__ if hasattr(result, '__dict__') else result,
                    message="Análisis de colaboración completado"
                )
            except Exception as e:
                logger.error(f"Error analizando colaboración: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/metrics/dashboard", response_model=APIResponse)
        async def get_dashboard(period: str = "daily", days: int = 7):
            """Obtener dashboard de métricas."""
            try:
                result = await self.analyzer.generate_metrics_dashboard(
                    period=period,
                    days=days
                )
                
                return APIResponse(
                    success=True,
                    data=result.__dict__ if hasattr(result, '__dict__') else result,
                    message="Dashboard generado"
                )
            except Exception as e:
                logger.error(f"Error generando dashboard: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/metrics/statistics", response_model=APIResponse)
        async def get_statistics():
            """Obtener estadísticas."""
            try:
                result = self.analyzer.get_metrics_statistics()
                
                return APIResponse(
                    success=True,
                    data=result,
                    message="Estadísticas obtenidas"
                )
            except Exception as e:
                logger.error(f"Error obteniendo estadísticas: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/health", response_model=APIResponse)
        async def health_check():
            """Health check."""
            return APIResponse(
                success=True,
                data={"status": "healthy"},
                message="Servicio funcionando correctamente"
            )
    
    def run(self):
        """Ejecutar servidor."""
        uvicorn.run(self.app, host=self.host, port=self.port)


def create_api_server(analyzer, host: str = "0.0.0.0", port: int = 8000) -> DocumentAPIServer:
    """Crear servidor API."""
    return DocumentAPIServer(analyzer, host, port)


__all__ = [
    "DocumentAPIServer",
    "create_api_server",
    "DocumentAnalysisRequest",
    "DocumentVersionRequest",
    "BatchAnalysisRequest",
    "RecommendationRequest",
    "APIResponse"
]
















