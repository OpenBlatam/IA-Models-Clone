"""
BUL Database Module
==================

Gestión de base de datos para el sistema BUL.
"""

from .models import (
    Base,
    Document,
    DocumentRequest,
    Agent,
    AgentUsageLog,
    DocumentFeedback,
    SystemStats,
    UserSession,
    CacheEntry,
    APILog,
    Template,
    Configuration
)

from .database_manager import (
    DatabaseManager,
    get_global_db_manager
)

__all__ = [
    "Base",
    "Document",
    "DocumentRequest",
    "Agent",
    "AgentUsageLog",
    "DocumentFeedback",
    "SystemStats",
    "UserSession",
    "CacheEntry",
    "APILog",
    "Template",
    "Configuration",
    "DatabaseManager",
    "get_global_db_manager"
]
























