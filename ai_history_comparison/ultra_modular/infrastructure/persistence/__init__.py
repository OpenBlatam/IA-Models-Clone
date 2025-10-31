"""
Infrastructure Persistence
=========================

This module contains persistence implementations, each in its own focused module
following the Single Responsibility Principle.
"""

from .history_repository import HistoryRepository
from .comparison_repository import ComparisonRepository
from .report_repository import ReportRepository
from .job_repository import JobRepository
from .feedback_repository import FeedbackRepository
from .database_manager import DatabaseManager
from .connection_pool import ConnectionPool
from .migration_manager import MigrationManager

__all__ = [
    "HistoryRepository",
    "ComparisonRepository",
    "ReportRepository",
    "JobRepository",
    "FeedbackRepository",
    "DatabaseManager",
    "ConnectionPool",
    "MigrationManager"
]




