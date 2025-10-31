"""
Domain Layer - Capa de Dominio
=============================

Contiene las entidades de dominio, value objects y reglas de negocio
del sistema de comparación de historial de IA.
"""

from .models import HistoryEntry, ComparisonResult, QualityReport
from .value_objects import ContentMetrics, QualityScore, SimilarityScore
from .exceptions import DomainException, ValidationException, NotFoundException

__all__ = [
    "HistoryEntry",
    "ComparisonResult",
    "QualityReport",
    "ContentMetrics",
    "QualityScore", 
    "SimilarityScore",
    "DomainException",
    "ValidationException",
    "NotFoundException"
]




