"""
Application Module - Capa de Aplicación
Módulo de aplicación con casos de uso, DTOs y validadores
"""

from .use_cases import (
    AnalyzeContentUseCase,
    CompareContentUseCase,
    GenerateReportUseCase,
    TrackTrendsUseCase,
    ManageContentUseCase
)

from .dto import (
    ContentDTO,
    AnalysisDTO,
    ComparisonDTO,
    ReportDTO,
    TrendDTO
)

from .validators import (
    ContentValidator,
    AnalysisValidator,
    ComparisonValidator,
    ReportValidator
)

from .handlers import (
    ContentEventHandler,
    AnalysisEventHandler,
    ComparisonEventHandler,
    ReportEventHandler
)

__all__ = [
    # Casos de uso
    "AnalyzeContentUseCase",
    "CompareContentUseCase",
    "GenerateReportUseCase",
    "TrackTrendsUseCase",
    "ManageContentUseCase",
    
    # DTOs
    "ContentDTO",
    "AnalysisDTO",
    "ComparisonDTO",
    "ReportDTO",
    "TrendDTO",
    
    # Validadores
    "ContentValidator",
    "AnalysisValidator",
    "ComparisonValidator",
    "ReportValidator",
    
    # Manejadores de eventos
    "ContentEventHandler",
    "AnalysisEventHandler",
    "ComparisonEventHandler",
    "ReportEventHandler"
]







