"""
Ultra-Modular AI History Comparison System v2
============================================

Sistema ultra-modular con máxima separación de responsabilidades.
Cada módulo tiene una sola función específica.
"""

__version__ = "2.0.0"
__author__ = "AI History Team"

# Core modules
from .core.entities import HistoryEntry, ComparisonResult
from .core.services import ContentAnalyzer, ModelComparator, QualityAssessor
from .core.repositories import HistoryRepository, ComparisonRepository

# Application modules
from .application.commands import AnalyzeCommand, CompareCommand
from .application.queries import GetEntryQuery, SearchEntriesQuery
from .application.handlers import CommandHandler, QueryHandler

# Infrastructure modules
from .infrastructure.database import DatabaseManager
from .infrastructure.cache import CacheManager
from .infrastructure.logging import Logger

# Presentation modules
from .presentation.api import APIRouter
from .presentation.middleware import ErrorHandler, RequestLogger

# Configuration modules
from .config.settings import Settings
from .config.database import DatabaseConfig

__all__ = [
    # Core
    "HistoryEntry",
    "ComparisonResult",
    "ContentAnalyzer",
    "ModelComparator",
    "QualityAssessor",
    "HistoryRepository",
    "ComparisonRepository",
    
    # Application
    "AnalyzeCommand",
    "CompareCommand",
    "GetEntryQuery",
    "SearchEntriesQuery",
    "CommandHandler",
    "QueryHandler",
    
    # Infrastructure
    "DatabaseManager",
    "CacheManager",
    "Logger",
    
    # Presentation
    "APIRouter",
    "ErrorHandler",
    "RequestLogger",
    
    # Configuration
    "Settings",
    "DatabaseConfig"
]




