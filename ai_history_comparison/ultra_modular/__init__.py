"""
Ultra-Modular AI History Comparison System
==========================================

This package implements an ultra-modular architecture where every component
is broken down into the smallest possible, focused modules with single
responsibilities.

Architecture Principles:
- Single Responsibility Principle (SRP)
- Interface Segregation Principle (ISP)
- Dependency Inversion Principle (DIP)
- Composition over Inheritance
- Micro-modules with focused functionality
- Plugin-based architecture
- Event-driven communication
- Zero coupling between modules
"""

__version__ = "3.0.0"
__author__ = "AI History Comparison Team"

# Core domain modules
from .domain.entities import (
    HistoryEntry,
    ComparisonResult,
    TrendAnalysis,
    QualityReport,
    AnalysisJob,
    UserFeedback
)

from .domain.value_objects import (
    ContentMetrics,
    ModelDefinition,
    PerformanceMetric,
    TrendDirection,
    AnalysisStatus
)

from .domain.events import (
    AnalysisCompletedEvent,
    ModelComparisonEvent,
    TrendDetectedEvent,
    QualityAlertEvent
)

# Application modules
from .application.commands import (
    AnalyzeContentCommand,
    CompareModelsCommand,
    GenerateReportCommand,
    TrackTrendsCommand
)

from .application.queries import (
    GetHistoryEntryQuery,
    SearchEntriesQuery,
    GetComparisonQuery,
    GetReportQuery
)

from .application.handlers import (
    CommandHandler,
    QueryHandler,
    EventHandler
)

# Infrastructure modules
from .infrastructure.persistence import (
    HistoryRepository,
    ComparisonRepository,
    ReportRepository,
    JobRepository
)

from .infrastructure.external import (
    AIService,
    CacheService,
    NotificationService,
    FileStorageService
)

# Presentation modules
from .presentation.rest import (
    AnalysisController,
    ComparisonController,
    ReportController,
    TrendController
)

from .presentation.websocket import (
    WebSocketManager,
    RealTimeUpdates
)

# Plugin system
from .plugins import (
    PluginManager,
    PluginInterface,
    PluginRegistry
)

__all__ = [
    # Domain
    "HistoryEntry",
    "ComparisonResult", 
    "TrendAnalysis",
    "QualityReport",
    "AnalysisJob",
    "UserFeedback",
    "ContentMetrics",
    "ModelDefinition",
    "PerformanceMetric",
    "TrendDirection",
    "AnalysisStatus",
    "AnalysisCompletedEvent",
    "ModelComparisonEvent",
    "TrendDetectedEvent",
    "QualityAlertEvent",
    
    # Application
    "AnalyzeContentCommand",
    "CompareModelsCommand",
    "GenerateReportCommand",
    "TrackTrendsCommand",
    "GetHistoryEntryQuery",
    "SearchEntriesQuery",
    "GetComparisonQuery",
    "GetReportQuery",
    "CommandHandler",
    "QueryHandler",
    "EventHandler",
    
    # Infrastructure
    "HistoryRepository",
    "ComparisonRepository",
    "ReportRepository",
    "JobRepository",
    "AIService",
    "CacheService",
    "NotificationService",
    "FileStorageService",
    
    # Presentation
    "AnalysisController",
    "ComparisonController",
    "ReportController",
    "TrendController",
    "WebSocketManager",
    "RealTimeUpdates",
    
    # Plugins
    "PluginManager",
    "PluginInterface",
    "PluginRegistry"
]




