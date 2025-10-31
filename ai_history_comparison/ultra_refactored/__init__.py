"""
Ultra Refactored AI History Comparison System
============================================

Sistema ultra-refactorizado con arquitectura limpia, separación de responsabilidades
y patrones de diseño modernos para el análisis y comparación de historial de IA.

Características principales:
- Arquitectura limpia con separación de capas
- Patrón Repository para acceso a datos
- Servicios de aplicación con lógica de negocio
- API REST con FastAPI
- Validación con Pydantic
- Logging estructurado
- Manejo de errores robusto
- Documentación automática
"""

__version__ = "2.0.0"
__author__ = "AI History Team"
__description__ = "Ultra Refactored AI History Comparison System"

# Imports principales
from .domain.models import HistoryEntry, ComparisonResult, QualityReport
from .application.services import HistoryService, ComparisonService, QualityService
from .infrastructure.repositories import HistoryRepository, ComparisonRepository
from .presentation.api import create_app

__all__ = [
    "HistoryEntry",
    "ComparisonResult", 
    "QualityReport",
    "HistoryService",
    "ComparisonService",
    "QualityService",
    "HistoryRepository",
    "ComparisonRepository",
    "create_app"
]