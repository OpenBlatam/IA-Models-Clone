"""
Sistema de migraciones de base de datos para Robot Movement AI v2.0
"""

from .migration_manager import MigrationManager, Migration
from .migrations import create_initial_schema, create_indexes

__all__ = [
    "MigrationManager",
    "Migration",
    "create_initial_schema",
    "create_indexes",
]




