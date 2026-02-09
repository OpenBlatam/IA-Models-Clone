"""
Dependencies - Dependencias
=========================

Configuración de dependencias para inyección de dependencias.
"""

from fastapi import Depends
from typing import Optional

from ..application.services import HistoryService, ComparisonService, QualityService
from ..infrastructure.repositories import InMemoryHistoryRepository, InMemoryComparisonRepository
from ..infrastructure.services import TextContentAnalyzer, BasicQualityAssessor, CosineSimilarityCalculator


# Instancias globales de servicios (singleton pattern)
_history_service: Optional[HistoryService] = None
_comparison_service: Optional[ComparisonService] = None
_quality_service: Optional[QualityService] = None


def setup_dependencies():
    """Configurar dependencias globales."""
    global _history_service, _comparison_service, _quality_service
    
    # Crear repositorios
    history_repo = InMemoryHistoryRepository()
    comparison_repo = InMemoryComparisonRepository()
    
    # Crear servicios de infraestructura
    content_analyzer = TextContentAnalyzer()
    quality_assessor = BasicQualityAssessor()
    similarity_calculator = CosineSimilarityCalculator()
    
    # Crear servicios de aplicación
    _history_service = HistoryService(history_repo, content_analyzer, quality_assessor)
    _comparison_service = ComparisonService(history_repo, comparison_repo, similarity_calculator)
    _quality_service = QualityService(history_repo, quality_assessor)


def get_history_service() -> HistoryService:
    """Obtener servicio de historial."""
    if _history_service is None:
        raise RuntimeError("Dependencies not initialized. Call setup_dependencies() first.")
    return _history_service


def get_comparison_service() -> ComparisonService:
    """Obtener servicio de comparación."""
    if _comparison_service is None:
        raise RuntimeError("Dependencies not initialized. Call setup_dependencies() first.")
    return _comparison_service


def get_quality_service() -> QualityService:
    """Obtener servicio de calidad."""
    if _quality_service is None:
        raise RuntimeError("Dependencies not initialized. Call setup_dependencies() first.")
    return _quality_service


def get_content_analyzer() -> TextContentAnalyzer:
    """Obtener analizador de contenido."""
    return TextContentAnalyzer()


def get_quality_assessor() -> BasicQualityAssessor:
    """Obtener evaluador de calidad."""
    return BasicQualityAssessor()


def get_similarity_calculator() -> CosineSimilarityCalculator:
    """Obtener calculador de similitud."""
    return CosineSimilarityCalculator()




