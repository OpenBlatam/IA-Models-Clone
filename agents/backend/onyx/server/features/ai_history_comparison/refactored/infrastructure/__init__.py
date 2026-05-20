"""
Infrastructure Layer
===================

This module contains the infrastructure components that handle external concerns
such as database access, external service integration, and technical implementations.

The infrastructure layer implements the interfaces defined in the application layer
and provides concrete implementations for:
- Database repositories and data access
- External API integrations
- File system operations
- Caching and performance optimization
- Security and authentication
"""

from .database import DatabaseManager, DatabaseConfig
from .repositories import (
    HistoryRepository,
    ComparisonRepository,
    ReportRepository,
    AnalysisJobRepository
)
from .services import (
    ExternalAIService,
    CacheService,
    NotificationService,
    FileStorageService
)

__all__ = [
    # Database
    "DatabaseManager",
    "DatabaseConfig",
    
    # Repositories
    "HistoryRepository",
    "ComparisonRepository", 
    "ReportRepository",
    "AnalysisJobRepository",
    
    # Services
    "ExternalAIService",
    "CacheService",
    "NotificationService",
    "FileStorageService"
]




