"""
Application Services - Servicios de Aplicación
=============================================

Servicios de aplicación que contienen la lógica de negocio
y orquestan las operaciones del sistema.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
from loguru import logger

from ..domain.models import HistoryEntry, ComparisonResult, QualityReport, AnalysisJob, ModelType, QualityLevel
from ..domain.value_objects import ContentMetrics, QualityScore, SimilarityScore, SentimentAnalysis, TextComplexity
from ..domain.exceptions import (
    NotFoundException, 
    ValidationException, 
    BusinessRuleException,
    AnalysisException,
    ComparisonException
)
from .interfaces import (
    IHistoryRepository,
    IComparisonRepository,
    IContentAnalyzer,
    IQualityAssessor,
    ISimilarityCalculator
)
from .dto import (
    CreateHistoryEntryRequest,
    UpdateHistoryEntryRequest,
    CompareEntriesRequest,
    QualityAssessmentRequest,
    AnalysisRequest
)


class HistoryService:
    """
    Servicio de aplicación para gestión de historial.
    
    Maneja las operaciones CRUD y lógica de negocio relacionada
    con las entradas de historial.
    """
    
    def __init__(
        self,
        history_repository: IHistoryRepository,
        content_analyzer: IContentAnalyzer,
        quality_assessor: IQualityAssessor
    ):
        self.history_repository = history_repository
        self.content_analyzer = content_analyzer
        self.quality_assessor = quality_assessor
    
    async def create_entry(self, request: CreateHistoryEntryRequest) -> HistoryEntry:
        """
        Crear una nueva entrada de historial.
        
        Args:
            request: Datos para crear la entrada
            
        Returns:
            HistoryEntry: Entrada creada
            
        Raises:
            ValidationException: Si los datos no son válidos
            BusinessRuleException: Si se viola una regla de negocio
        """
        try:
            logger.info(f"Creating history entry for model: {request.model_type}")
            
            # Validar modelo
            if not self._is_valid_model(request.model_type):
                raise ValidationException(f"Invalid model type: {request.model_type}")
            
            # Analizar contenido
            content_metrics = await self.content_analyzer.analyze_content(request.content)
            
            # Crear entrada
            entry = HistoryEntry(
                model_type=request.model_type,
                content=request.content,
                metadata=request.metadata or {},
                user_id=request.user_id,
                session_id=request.session_id
            )
            
            # Agregar métricas al metadata
            entry.metadata.update(content_metrics.model_dump())
            
            # Evaluar calidad si se solicita
            if request.assess_quality:
                quality_score = await self.quality_assessor.assess_quality(entry)
                entry.quality_score = quality_score.overall_score
            
            # Guardar en repositorio
            saved_entry = await self.history_repository.save(entry)
            
            logger.info(f"History entry created with ID: {saved_entry.id}")
            return saved_entry
            
        except Exception as e:
            logger.error(f"Error creating history entry: {e}")
            raise
    
    async def get_entry(self, entry_id: str) -> HistoryEntry:
        """
        Obtener una entrada de historial por ID.
        
        Args:
            entry_id: ID de la entrada
            
        Returns:
            HistoryEntry: Entrada encontrada
            
        Raises:
            NotFoundException: Si la entrada no existe
        """
        try:
            logger.info(f"Getting history entry: {entry_id}")
            
            entry = await self.history_repository.get_by_id(entry_id)
            if not entry:
                raise NotFoundException("HistoryEntry", entry_id)
            
            return entry
            
        except NotFoundException:
            raise
        except Exception as e:
            logger.error(f"Error getting history entry {entry_id}: {e}")
            raise
    
    async def update_entry(self, entry_id: str, request: UpdateHistoryEntryRequest) -> HistoryEntry:
        """
        Actualizar una entrada de historial.
        
        Args:
            entry_id: ID de la entrada
            request: Datos de actualización
            
        Returns:
            HistoryEntry: Entrada actualizada
            
        Raises:
            NotFoundException: Si la entrada no existe
            ValidationException: Si los datos no son válidos
        """
        try:
            logger.info(f"Updating history entry: {entry_id}")
            
            # Obtener entrada existente
            entry = await self.get_entry(entry_id)
            
            # Actualizar campos
            if request.content is not None:
                entry.content = request.content
                # Re-analizar contenido si cambió
                content_metrics = await self.content_analyzer.analyze_content(request.content)
                entry.metadata.update(content_metrics.model_dump())
            
            if request.metadata is not None:
                entry.metadata.update(request.metadata)
            
            if request.assess_quality:
                quality_score = await self.quality_assessor.assess_quality(entry)
                entry.quality_score = quality_score.overall_score
            
            # Guardar cambios
            updated_entry = await self.history_repository.save(entry)
            
            logger.info(f"History entry updated: {entry_id}")
            return updated_entry
            
        except NotFoundException:
            raise
        except Exception as e:
            logger.error(f"Error updating history entry {entry_id}: {e}")
            raise
    
    async def delete_entry(self, entry_id: str) -> bool:
        """
        Eliminar una entrada de historial.
        
        Args:
            entry_id: ID de la entrada
            
        Returns:
            bool: True si se eliminó correctamente
            
        Raises:
            NotFoundException: Si la entrada no existe
        """
        try:
            logger.info(f"Deleting history entry: {entry_id}")
            
            # Verificar que existe
            await self.get_entry(entry_id)
            
            # Eliminar
            success = await self.history_repository.delete(entry_id)
            
            if success:
                logger.info(f"History entry deleted: {entry_id}")
            else:
                logger.warning(f"Failed to delete history entry: {entry_id}")
            
            return success
            
        except NotFoundException:
            raise
        except Exception as e:
            logger.error(f"Error deleting history entry {entry_id}: {e}")
            raise
    
    async def list_entries(
        self,
        user_id: Optional[str] = None,
        model_type: Optional[ModelType] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[HistoryEntry]:
        """
        Listar entradas de historial con filtros.
        
        Args:
            user_id: Filtrar por usuario
            model_type: Filtrar por tipo de modelo
            limit: Límite de resultados
            offset: Offset de resultados
            
        Returns:
            List[HistoryEntry]: Lista de entradas
        """
        try:
            logger.info(f"Listing history entries with filters: user_id={user_id}, model_type={model_type}")
            
            entries = await self.history_repository.list(
                user_id=user_id,
                model_type=model_type,
                limit=limit,
                offset=offset
            )
            
            logger.info(f"Found {len(entries)} history entries")
            return entries
            
        except Exception as e:
            logger.error(f"Error listing history entries: {e}")
            raise
    
    def _is_valid_model(self, model_type: str) -> bool:
        """Validar si el tipo de modelo es válido."""
        try:
            ModelType(model_type)
            return True
        except ValueError:
            return False


class ComparisonService:
    """
    Servicio de aplicación para comparación de entradas.
    
    Maneja la lógica de comparación entre entradas de historial.
    """
    
    def __init__(
        self,
        history_repository: IHistoryRepository,
        comparison_repository: IComparisonRepository,
        similarity_calculator: ISimilarityCalculator
    ):
        self.history_repository = history_repository
        self.comparison_repository = comparison_repository
        self.similarity_calculator = similarity_calculator
    
    async def compare_entries(self, request: CompareEntriesRequest) -> ComparisonResult:
        """
        Comparar dos entradas de historial.
        
        Args:
            request: Datos de comparación
            
        Returns:
            ComparisonResult: Resultado de la comparación
            
        Raises:
            NotFoundException: Si alguna entrada no existe
            ComparisonException: Si hay error en la comparación
        """
        try:
            logger.info(f"Comparing entries: {request.entry_1_id} vs {request.entry_2_id}")
            
            # Obtener entradas
            entry_1 = await self.history_repository.get_by_id(request.entry_1_id)
            entry_2 = await self.history_repository.get_by_id(request.entry_2_id)
            
            if not entry_1:
                raise NotFoundException("HistoryEntry", request.entry_1_id)
            if not entry_2:
                raise NotFoundException("HistoryEntry", request.entry_2_id)
            
            # Validar que no sean la misma entrada
            if entry_1.id == entry_2.id:
                raise ComparisonException("Cannot compare entry with itself", entry_1.id, entry_2.id)
            
            # Calcular similitud
            similarity_score = await self.similarity_calculator.calculate_similarity(entry_1, entry_2)
            
            # Crear resultado de comparación
            comparison = ComparisonResult(
                entry_1_id=entry_1.id,
                entry_2_id=entry_2.id,
                similarity_score=similarity_score.overall_similarity,
                content_similarity=similarity_score.content_similarity,
                quality_difference=(entry_1.quality_score or 0.0) - (entry_2.quality_score or 0.0),
                differences=request.include_differences,
                analysis_metadata={
                    "semantic_similarity": similarity_score.semantic_similarity,
                    "structural_similarity": similarity_score.structural_similarity,
                    "style_similarity": similarity_score.style_similarity,
                    "comparison_timestamp": datetime.utcnow().isoformat()
                }
            )
            
            # Guardar resultado
            saved_comparison = await self.comparison_repository.save(comparison)
            
            logger.info(f"Comparison completed: {saved_comparison.id}")
            return saved_comparison
            
        except NotFoundException:
            raise
        except Exception as e:
            logger.error(f"Error comparing entries: {e}")
            raise ComparisonException(f"Failed to compare entries: {e}")
    
    async def get_comparison(self, comparison_id: str) -> ComparisonResult:
        """
        Obtener resultado de comparación por ID.
        
        Args:
            comparison_id: ID de la comparación
            
        Returns:
            ComparisonResult: Resultado de comparación
            
        Raises:
            NotFoundException: Si la comparación no existe
        """
        try:
            logger.info(f"Getting comparison: {comparison_id}")
            
            comparison = await self.comparison_repository.get_by_id(comparison_id)
            if not comparison:
                raise NotFoundException("ComparisonResult", comparison_id)
            
            return comparison
            
        except NotFoundException:
            raise
        except Exception as e:
            logger.error(f"Error getting comparison {comparison_id}: {e}")
            raise
    
    async def list_comparisons(
        self,
        entry_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ComparisonResult]:
        """
        Listar comparaciones con filtros.
        
        Args:
            entry_id: Filtrar por ID de entrada
            limit: Límite de resultados
            offset: Offset de resultados
            
        Returns:
            List[ComparisonResult]: Lista de comparaciones
        """
        try:
            logger.info(f"Listing comparisons with filters: entry_id={entry_id}")
            
            comparisons = await self.comparison_repository.list(
                entry_id=entry_id,
                limit=limit,
                offset=offset
            )
            
            logger.info(f"Found {len(comparisons)} comparisons")
            return comparisons
            
        except Exception as e:
            logger.error(f"Error listing comparisons: {e}")
            raise


class QualityService:
    """
    Servicio de aplicación para evaluación de calidad.
    
    Maneja la evaluación y análisis de calidad del contenido.
    """
    
    def __init__(
        self,
        history_repository: IHistoryRepository,
        quality_assessor: IQualityAssessor
    ):
        self.history_repository = history_repository
        self.quality_assessor = quality_assessor
    
    async def assess_quality(self, request: QualityAssessmentRequest) -> QualityReport:
        """
        Evaluar la calidad de una entrada.
        
        Args:
            request: Datos de evaluación
            
        Returns:
            QualityReport: Reporte de calidad
            
        Raises:
            NotFoundException: Si la entrada no existe
            QualityAssessmentException: Si hay error en la evaluación
        """
        try:
            logger.info(f"Assessing quality for entry: {request.entry_id}")
            
            # Obtener entrada
            entry = await self.history_repository.get_by_id(request.entry_id)
            if not entry:
                raise NotFoundException("HistoryEntry", request.entry_id)
            
            # Evaluar calidad
            quality_score = await self.quality_assessor.assess_quality(entry)
            
            # Determinar nivel de calidad
            quality_level = self._determine_quality_level(quality_score.overall_score)
            
            # Generar recomendaciones
            recommendations = self._generate_recommendations(quality_score)
            
            # Crear reporte
            report = QualityReport(
                entry_id=entry.id,
                overall_score=quality_score.overall_score,
                quality_level=quality_level,
                readability_score=quality_score.readability_score,
                coherence_score=quality_score.coherence_score,
                relevance_score=quality_score.relevance_score,
                sentiment_score=0.0,  # TODO: Implementar análisis de sentimiento
                recommendations=recommendations,
                detailed_analysis=quality_score.model_dump()
            )
            
            logger.info(f"Quality assessment completed for entry: {request.entry_id}")
            return report
            
        except NotFoundException:
            raise
        except Exception as e:
            logger.error(f"Error assessing quality for entry {request.entry_id}: {e}")
            raise
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determinar el nivel de calidad basado en el score."""
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.7:
            return QualityLevel.GOOD
        elif score >= 0.5:
            return QualityLevel.AVERAGE
        elif score >= 0.3:
            return QualityLevel.POOR
        else:
            return QualityLevel.VERY_POOR
    
    def _generate_recommendations(self, quality_score: QualityScore) -> List[str]:
        """Generar recomendaciones basadas en el score de calidad."""
        recommendations = []
        
        if quality_score.readability_score < 0.6:
            recommendations.append("Improve readability by using shorter sentences and simpler words")
        
        if quality_score.coherence_score < 0.6:
            recommendations.append("Enhance coherence by improving logical flow and structure")
        
        if quality_score.relevance_score < 0.6:
            recommendations.append("Increase relevance by focusing on the main topic")
        
        if quality_score.completeness_score < 0.6:
            recommendations.append("Add more details and examples to improve completeness")
        
        if quality_score.accuracy_score < 0.6:
            recommendations.append("Verify facts and improve accuracy of information")
        
        if not recommendations:
            recommendations.append("Content quality is good, maintain current standards")
        
        return recommendations


class AnalysisService:
    """
    Servicio de aplicación para análisis en lote.
    
    Maneja análisis masivos y trabajos de procesamiento.
    """
    
    def __init__(
        self,
        history_repository: IHistoryRepository,
        comparison_repository: IComparisonRepository,
        content_analyzer: IContentAnalyzer,
        quality_assessor: IQualityAssessor
    ):
        self.history_repository = history_repository
        self.comparison_repository = comparison_repository
        self.content_analyzer = content_analyzer
        self.quality_assessor = quality_assessor
    
    async def start_analysis_job(self, request: AnalysisRequest) -> AnalysisJob:
        """
        Iniciar un trabajo de análisis.
        
        Args:
            request: Datos del análisis
            
        Returns:
            AnalysisJob: Trabajo de análisis iniciado
        """
        try:
            logger.info(f"Starting analysis job: {request.job_type}")
            
            # Obtener entradas para analizar
            entries = await self.history_repository.list(
                user_id=request.user_id,
                model_type=request.model_type,
                limit=request.limit or 1000
            )
            
            if not entries:
                raise BusinessRuleException("No entries found for analysis")
            
            # Crear trabajo de análisis
            job = AnalysisJob(
                job_type=request.job_type,
                total_entries=len(entries),
                status="pending"
            )
            
            # Iniciar procesamiento asíncrono
            asyncio.create_task(self._process_analysis_job(job, entries, request))
            
            logger.info(f"Analysis job started: {job.id}")
            return job
            
        except Exception as e:
            logger.error(f"Error starting analysis job: {e}")
            raise
    
    async def _process_analysis_job(
        self, 
        job: AnalysisJob, 
        entries: List[HistoryEntry], 
        request: AnalysisRequest
    ):
        """Procesar trabajo de análisis de forma asíncrona."""
        try:
            job.status = "running"
            job.started_at = datetime.utcnow()
            
            results = {}
            
            for entry in entries:
                try:
                    if request.job_type == "quality_assessment":
                        quality_score = await self.quality_assessor.assess_quality(entry)
                        results[entry.id] = quality_score.model_dump()
                    
                    elif request.job_type == "content_analysis":
                        content_metrics = await self.content_analyzer.analyze_content(entry.content)
                        results[entry.id] = content_metrics.model_dump()
                    
                    job.processed_entries += 1
                    
                except Exception as e:
                    logger.error(f"Error processing entry {entry.id}: {e}")
                    job.failed_entries += 1
            
            # Completar trabajo
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.results = results
            
            logger.info(f"Analysis job completed: {job.id}")
            
        except Exception as e:
            logger.error(f"Error processing analysis job {job.id}: {e}")
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
    
    async def get_analysis_job(self, job_id: str) -> AnalysisJob:
        """
        Obtener estado de trabajo de análisis.
        
        Args:
            job_id: ID del trabajo
            
        Returns:
            AnalysisJob: Trabajo de análisis
            
        Raises:
            NotFoundException: Si el trabajo no existe
        """
        # TODO: Implementar almacenamiento de trabajos
        raise NotImplementedError("Analysis job storage not implemented")




