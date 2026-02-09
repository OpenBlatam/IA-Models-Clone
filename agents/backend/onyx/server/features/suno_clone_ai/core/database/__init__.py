"""
Database Module

Provides:
- Database utilities
- Connection management
- Query builders
"""

from .db_manager import (
    DatabaseManager,
    create_connection,
    execute_query,
    execute_transaction
)

from .query_builder import (
    QueryBuilder,
    create_query,
    build_select,
    build_insert,
    build_update
)

__all__ = [
    # Database management
    "DatabaseManager",
    "create_connection",
    "execute_query",
    "execute_transaction",
    # Query builder
    "QueryBuilder",
    "create_query",
    "build_select",
    "build_insert",
    "build_update"
]



