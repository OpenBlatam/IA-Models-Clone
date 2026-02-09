"""
Presentation Layer - Capa de Presentación
========================================

Contiene la API REST, controladores y middleware
para el sistema de comparación de historial de IA.
"""

from .api import create_app
from .controllers import HistoryController, ComparisonController, QualityController
from .middleware import ErrorHandlerMiddleware, LoggingMiddleware, CORSMiddleware
from .dependencies import get_history_service, get_comparison_service, get_quality_service

__all__ = [
    "create_app",
    "HistoryController",
    "ComparisonController", 
    "QualityController",
    "ErrorHandlerMiddleware",
    "LoggingMiddleware",
    "CORSMiddleware",
    "get_history_service",
    "get_comparison_service",
    "get_quality_service"
]




