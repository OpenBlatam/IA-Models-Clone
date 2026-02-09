"""
AI History Comparison System - Refactored Architecture
=====================================================

This package contains the refactored, modular architecture for the AI History Comparison system.

Architecture Overview:
- Core: Core business logic and domain models
- Infrastructure: Database, external services, and technical concerns
- Application: Use cases and application services
- Presentation: API endpoints and web interfaces
- Shared: Common utilities and shared components

The refactored system follows clean architecture principles with clear separation of concerns.
"""

__version__ = "2.0.0"
__author__ = "AI History Comparison Team"

# Core exports
from .core.domain import (
    HistoryEntry,
    ComparisonResult,
    TrendAnalysis,
    QualityReport,
    ModelDefinition,
    PerformanceMetric
)

from .core.services import (
    HistoryAnalysisService,
    ModelComparisonService,
    TrendAnalysisService,
    QualityAssessmentService
)

from .application.use_cases import (
    AnalyzeContentUseCase,
    CompareModelsUseCase,
    GenerateReportUseCase,
    TrackTrendsUseCase
)

from .infrastructure.database import DatabaseManager
from .infrastructure.repositories import (
    HistoryRepository,
    ComparisonRepository,
    ReportRepository
)

from .presentation.api import create_app

__all__ = [
    # Core
    "HistoryEntry",
    "ComparisonResult", 
    "TrendAnalysis",
    "QualityReport",
    "ModelDefinition",
    "PerformanceMetric",
    
    # Services
    "HistoryAnalysisService",
    "ModelComparisonService",
    "TrendAnalysisService",
    "QualityAssessmentService",
    
    # Use Cases
    "AnalyzeContentUseCase",
    "CompareModelsUseCase",
    "GenerateReportUseCase",
    "TrackTrendsUseCase",
    
    # Infrastructure
    "DatabaseManager",
    "HistoryRepository",
    "ComparisonRepository",
    "ReportRepository",
    
    # Presentation
    "create_app"
]




