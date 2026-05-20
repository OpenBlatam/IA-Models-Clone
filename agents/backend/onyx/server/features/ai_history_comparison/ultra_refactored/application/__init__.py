"""
Application Layer - Capa de Aplicación
=====================================

Contiene los servicios de aplicación, casos de uso y DTOs
para el sistema de comparación de historial de IA.
"""

from .services import HistoryService, ComparisonService, QualityService, AnalysisService
from .dto import (
    CreateHistoryEntryRequest,
    UpdateHistoryEntryRequest,
    CompareEntriesRequest,
    QualityAssessmentRequest,
    AnalysisRequest,
    HistoryEntryResponse,
    ComparisonResultResponse,
    QualityReportResponse,
    AnalysisJobResponse
)
from .interfaces import (
    IHistoryRepository,
    IComparisonRepository,
    IContentAnalyzer,
    IQualityAssessor,
    ISimilarityCalculator
)

__all__ = [
    "HistoryService",
    "ComparisonService", 
    "QualityService",
    "AnalysisService",
    "CreateHistoryEntryRequest",
    "UpdateHistoryEntryRequest",
    "CompareEntriesRequest",
    "QualityAssessmentRequest",
    "AnalysisRequest",
    "HistoryEntryResponse",
    "ComparisonResultResponse",
    "QualityReportResponse",
    "AnalysisJobResponse",
    "IHistoryRepository",
    "IComparisonRepository",
    "IContentAnalyzer",
    "IQualityAssessor",
    "ISimilarityCalculator"
]




