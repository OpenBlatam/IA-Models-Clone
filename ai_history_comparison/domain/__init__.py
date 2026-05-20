"""
Domain Module - Lógica de Negocio
Módulo de dominio con entidades, servicios y repositorios
"""

from .entities import (
    Content,
    Analysis,
    Comparison,
    Report,
    Trend
)

from .services import (
    ContentService,
    AnalysisService,
    ComparisonService,
    ReportService
)

from .repositories import (
    ContentRepository,
    AnalysisRepository,
    ComparisonRepository,
    ReportRepository
)

from .events import (
    ContentCreatedEvent,
    ContentUpdatedEvent,
    AnalysisCompletedEvent,
    ComparisonCompletedEvent,
    ReportGeneratedEvent
)

__all__ = [
    # Entidades
    "Content",
    "Analysis", 
    "Comparison",
    "Report",
    "Trend",
    
    # Servicios
    "ContentService",
    "AnalysisService",
    "ComparisonService", 
    "ReportService",
    
    # Repositorios
    "ContentRepository",
    "AnalysisRepository",
    "ComparisonRepository",
    "ReportRepository",
    
    # Eventos
    "ContentCreatedEvent",
    "ContentUpdatedEvent",
    "AnalysisCompletedEvent",
    "ComparisonCompletedEvent",
    "ReportGeneratedEvent"
]







