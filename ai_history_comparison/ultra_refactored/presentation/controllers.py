"""
API Controllers - Controladores de API
====================================

Controladores REST para los endpoints de la API.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path
from typing import List, Optional
from loguru import logger

from ..application.services import HistoryService, ComparisonService, QualityService
from ..application.dto import (
    CreateHistoryEntryRequest,
    UpdateHistoryEntryRequest,
    CompareEntriesRequest,
    QualityAssessmentRequest,
    HistoryEntryResponse,
    ComparisonResultResponse,
    QualityReportResponse,
    PaginatedResponse,
    ErrorResponse
)
from ..domain.exceptions import NotFoundException, ValidationException, BusinessRuleException
from .dependencies import get_history_service, get_comparison_service, get_quality_service


class HistoryController:
    """Controlador para operaciones de historial."""
    
    def __init__(self):
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self):
        """Configurar rutas del controlador."""
        
        @self.router.post("/entries", response_model=HistoryEntryResponse, status_code=201)
        async def create_entry(
            request: CreateHistoryEntryRequest,
            service: HistoryService = Depends(get_history_service)
        ):
            """
            Crear una nueva entrada de historial.
            
            - **model_type**: Tipo de modelo de IA
            - **content**: Contenido generado por IA
            - **metadata**: Metadatos adicionales (opcional)
            - **user_id**: ID del usuario (opcional)
            - **session_id**: ID de sesión (opcional)
            - **assess_quality**: Si evaluar calidad automáticamente
            """
            try:
                logger.info(f"Creating history entry for model: {request.model_type}")
                entry = await service.create_entry(request)
                return HistoryEntryResponse(**entry.model_dump())
            except ValidationException as e:
                raise HTTPException(status_code=400, detail=str(e))
            except BusinessRuleException as e:
                raise HTTPException(status_code=422, detail=str(e))
            except Exception as e:
                logger.error(f"Error creating entry: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.router.get("/entries", response_model=List[HistoryEntryResponse])
        async def list_entries(
            user_id: Optional[str] = Query(None, description="Filtrar por ID de usuario"),
            model_type: Optional[str] = Query(None, description="Filtrar por tipo de modelo"),
            limit: int = Query(100, ge=1, le=1000, description="Límite de resultados"),
            offset: int = Query(0, ge=0, description="Offset de resultados"),
            service: HistoryService = Depends(get_history_service)
        ):
            """
            Listar entradas de historial con filtros opcionales.
            
            - **user_id**: Filtrar por ID de usuario
            - **model_type**: Filtrar por tipo de modelo
            - **limit**: Límite de resultados (1-1000)
            - **offset**: Offset de resultados
            """
            try:
                logger.info(f"Listing entries with filters: user_id={user_id}, model_type={model_type}")
                entries = await service.list_entries(
                    user_id=user_id,
                    model_type=model_type,
                    limit=limit,
                    offset=offset
                )
                return [HistoryEntryResponse(**entry.model_dump()) for entry in entries]
            except Exception as e:
                logger.error(f"Error listing entries: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.router.get("/entries/{entry_id}", response_model=HistoryEntryResponse)
        async def get_entry(
            entry_id: str = Path(..., description="ID de la entrada"),
            service: HistoryService = Depends(get_history_service)
        ):
            """
            Obtener una entrada de historial por ID.
            
            - **entry_id**: ID único de la entrada
            """
            try:
                logger.info(f"Getting entry: {entry_id}")
                entry = await service.get_entry(entry_id)
                return HistoryEntryResponse(**entry.model_dump())
            except NotFoundException as e:
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                logger.error(f"Error getting entry {entry_id}: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.router.put("/entries/{entry_id}", response_model=HistoryEntryResponse)
        async def update_entry(
            entry_id: str = Path(..., description="ID de la entrada"),
            request: UpdateHistoryEntryRequest = None,
            service: HistoryService = Depends(get_history_service)
        ):
            """
            Actualizar una entrada de historial.
            
            - **entry_id**: ID único de la entrada
            - **content**: Nuevo contenido (opcional)
            - **metadata**: Nuevos metadatos (opcional)
            - **assess_quality**: Si reevaluar calidad
            """
            try:
                logger.info(f"Updating entry: {entry_id}")
                entry = await service.update_entry(entry_id, request)
                return HistoryEntryResponse(**entry.model_dump())
            except NotFoundException as e:
                raise HTTPException(status_code=404, detail=str(e))
            except ValidationException as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Error updating entry {entry_id}: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.router.delete("/entries/{entry_id}", status_code=204)
        async def delete_entry(
            entry_id: str = Path(..., description="ID de la entrada"),
            service: HistoryService = Depends(get_history_service)
        ):
            """
            Eliminar una entrada de historial.
            
            - **entry_id**: ID único de la entrada
            """
            try:
                logger.info(f"Deleting entry: {entry_id}")
                success = await service.delete_entry(entry_id)
                if not success:
                    raise HTTPException(status_code=404, detail="Entry not found")
            except NotFoundException as e:
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                logger.error(f"Error deleting entry {entry_id}: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")


class ComparisonController:
    """Controlador para operaciones de comparación."""
    
    def __init__(self):
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self):
        """Configurar rutas del controlador."""
        
        @self.router.post("/", response_model=ComparisonResultResponse, status_code=201)
        async def compare_entries(
            request: CompareEntriesRequest,
            service: ComparisonService = Depends(get_comparison_service)
        ):
            """
            Comparar dos entradas de historial.
            
            - **entry_1_id**: ID de la primera entrada
            - **entry_2_id**: ID de la segunda entrada
            - **include_differences**: Si incluir diferencias detalladas
            """
            try:
                logger.info(f"Comparing entries: {request.entry_1_id} vs {request.entry_2_id}")
                comparison = await service.compare_entries(request)
                return ComparisonResultResponse(**comparison.model_dump())
            except NotFoundException as e:
                raise HTTPException(status_code=404, detail=str(e))
            except ValidationException as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Error comparing entries: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.router.get("/{comparison_id}", response_model=ComparisonResultResponse)
        async def get_comparison(
            comparison_id: str = Path(..., description="ID de la comparación"),
            service: ComparisonService = Depends(get_comparison_service)
        ):
            """
            Obtener resultado de comparación por ID.
            
            - **comparison_id**: ID único de la comparación
            """
            try:
                logger.info(f"Getting comparison: {comparison_id}")
                comparison = await service.get_comparison(comparison_id)
                return ComparisonResultResponse(**comparison.model_dump())
            except NotFoundException as e:
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                logger.error(f"Error getting comparison {comparison_id}: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.router.get("/", response_model=List[ComparisonResultResponse])
        async def list_comparisons(
            entry_id: Optional[str] = Query(None, description="Filtrar por ID de entrada"),
            limit: int = Query(100, ge=1, le=1000, description="Límite de resultados"),
            offset: int = Query(0, ge=0, description="Offset de resultados"),
            service: ComparisonService = Depends(get_comparison_service)
        ):
            """
            Listar comparaciones con filtros opcionales.
            
            - **entry_id**: Filtrar por ID de entrada
            - **limit**: Límite de resultados (1-1000)
            - **offset**: Offset de resultados
            """
            try:
                logger.info(f"Listing comparisons with filters: entry_id={entry_id}")
                comparisons = await service.list_comparisons(
                    entry_id=entry_id,
                    limit=limit,
                    offset=offset
                )
                return [ComparisonResultResponse(**comp.model_dump()) for comp in comparisons]
            except Exception as e:
                logger.error(f"Error listing comparisons: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")


class QualityController:
    """Controlador para operaciones de calidad."""
    
    def __init__(self):
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self):
        """Configurar rutas del controlador."""
        
        @self.router.post("/reports", response_model=QualityReportResponse, status_code=201)
        async def assess_quality(
            request: QualityAssessmentRequest,
            service: QualityService = Depends(get_quality_service)
        ):
            """
            Evaluar la calidad de una entrada.
            
            - **entry_id**: ID de la entrada a evaluar
            - **include_recommendations**: Si incluir recomendaciones
            - **detailed_analysis**: Si incluir análisis detallado
            """
            try:
                logger.info(f"Assessing quality for entry: {request.entry_id}")
                report = await service.assess_quality(request)
                return QualityReportResponse(**report.model_dump())
            except NotFoundException as e:
                raise HTTPException(status_code=404, detail=str(e))
            except ValidationException as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Error assessing quality: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.router.get("/reports/{entry_id}", response_model=QualityReportResponse)
        async def get_quality_report(
            entry_id: str = Path(..., description="ID de la entrada"),
            service: QualityService = Depends(get_quality_service)
        ):
            """
            Obtener reporte de calidad por ID de entrada.
            
            - **entry_id**: ID único de la entrada
            """
            try:
                logger.info(f"Getting quality report for entry: {entry_id}")
                # TODO: Implementar obtención de reporte existente
                raise HTTPException(status_code=501, detail="Not implemented yet")
            except Exception as e:
                logger.error(f"Error getting quality report: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")




