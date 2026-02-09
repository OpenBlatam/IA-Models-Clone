"""
Application Module - Módulo de Aplicación
========================================

Módulo de aplicación que contiene comandos, queries, handlers
y servicios de aplicación del sistema.
"""

from .commands import (
    CreateHistoryCommand,
    UpdateHistoryCommand,
    DeleteHistoryCommand,
    CompareEntriesCommand,
    AssessQualityCommand,
    StartAnalysisCommand
)
from .queries import (
    GetHistoryQuery,
    ListHistoryQuery,
    GetComparisonQuery,
    ListComparisonsQuery,
    GetQualityReportQuery,
    GetSystemMetricsQuery
)
from .handlers import (
    CommandHandler,
    QueryHandler,
    CreateHistoryHandler,
    UpdateHistoryHandler,
    DeleteHistoryHandler,
    CompareEntriesHandler,
    AssessQualityHandler,
    StartAnalysisHandler,
    GetHistoryHandler,
    ListHistoryHandler,
    GetComparisonHandler,
    ListComparisonsHandler,
    GetQualityReportHandler,
    GetSystemMetricsHandler
)
from .services import (
    HistoryApplicationService,
    ComparisonApplicationService,
    QualityApplicationService,
    AnalyticsApplicationService
)

__all__ = [
    # Commands
    "CreateHistoryCommand",
    "UpdateHistoryCommand",
    "DeleteHistoryCommand",
    "CompareEntriesCommand",
    "AssessQualityCommand",
    "StartAnalysisCommand",
    
    # Queries
    "GetHistoryQuery",
    "ListHistoryQuery",
    "GetComparisonQuery",
    "ListComparisonsQuery",
    "GetQualityReportQuery",
    "GetSystemMetricsQuery",
    
    # Handlers
    "CommandHandler",
    "QueryHandler",
    "CreateHistoryHandler",
    "UpdateHistoryHandler",
    "DeleteHistoryHandler",
    "CompareEntriesHandler",
    "AssessQualityHandler",
    "StartAnalysisHandler",
    "GetHistoryHandler",
    "ListHistoryHandler",
    "GetComparisonHandler",
    "ListComparisonsHandler",
    "GetQualityReportHandler",
    "GetSystemMetricsHandler",
    
    # Services
    "HistoryApplicationService",
    "ComparisonApplicationService",
    "QualityApplicationService",
    "AnalyticsApplicationService"
]




