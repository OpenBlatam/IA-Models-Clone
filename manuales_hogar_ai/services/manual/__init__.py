"""
Manual Service Module
====================

Módulo especializado para gestión de manuales.
"""

from .manual_service import ManualService
from .manual_repository import ManualRepository
from .manual_search_service import ManualSearchService
from .statistics_service import StatisticsService

__all__ = [
    "ManualService",
    "ManualRepository",
    "ManualSearchService",
    "StatisticsService",
]

