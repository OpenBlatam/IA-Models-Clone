"""
Infrastructure Layer - Capa de Infraestructura
=============================================

Contiene las implementaciones concretas de repositorios, servicios
externos y adaptadores para el sistema de comparación de historial de IA.
"""

from .repositories import InMemoryHistoryRepository, InMemoryComparisonRepository
from .services import (
    TextContentAnalyzer,
    BasicQualityAssessor,
    CosineSimilarityCalculator
)
from .database import DatabaseConfig, create_database_engine
from .cache import InMemoryCacheService
from .logging import StructuredLogger

__all__ = [
    "InMemoryHistoryRepository",
    "InMemoryComparisonRepository",
    "TextContentAnalyzer",
    "BasicQualityAssessor", 
    "CosineSimilarityCalculator",
    "DatabaseConfig",
    "create_database_engine",
    "InMemoryCacheService",
    "StructuredLogger"
]




