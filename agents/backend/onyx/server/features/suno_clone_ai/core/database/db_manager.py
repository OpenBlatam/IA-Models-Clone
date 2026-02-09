"""
Database Manager

Utilities for database operations.
"""

import logging
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Try to import database libraries
try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

try:
    import psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    logger.warning("psycopg2 not available for PostgreSQL")


class DatabaseManager:
    """Manage database connections and operations."""
    
    def __init__(
        self,
        db_type: str = "sqlite",
        connection_string: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize database manager.
        
        Args:
            db_type: Database type ('sqlite', 'postgres')
            connection_string: Connection string
            **kwargs: Additional connection parameters
        """
        self.db_type = db_type
        self.connection_string = connection_string
        self.connection_params = kwargs
        self.connection = None
    
    def connect(self) -> Any:
        """
        Connect to database.
        
        Returns:
            Database connection
        """
        if self.db_type == "sqlite":
            if not SQLITE_AVAILABLE:
                raise ImportError("sqlite3 not available")
            
            db_path = self.connection_string or ":memory:"
            self.connection = sqlite3.connect(db_path)
            logger.info(f"Connected to SQLite: {db_path}")
        
        elif self.db_type == "postgres":
            if not POSTGRES_AVAILABLE:
                raise ImportError("psycopg2 not available")
            
            self.connection = psycopg2.connect(
                self.connection_string or "",
                **self.connection_params
            )
            logger.info("Connected to PostgreSQL")
        
        else:
            raise ValueError(f"Unknown database type: {self.db_type}")
        
        return self.connection
    
    def disconnect(self) -> None:
        """Disconnect from database."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Disconnected from database")
    
    def execute(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch: bool = False
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Execute query.
        
        Args:
            query: SQL query
            params: Query parameters
            fetch: Whether to fetch results
            
        Returns:
            Query results or None
        """
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        
        try:
            cursor.execute(query, params or ())
            
            if fetch:
                columns = [desc[0] for desc in cursor.description]
                results = [dict(zip(columns, row)) for row in cursor.fetchall()]
                return results
            
            self.connection.commit()
            return None
        
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Query execution error: {e}")
            raise
        
        finally:
            cursor.close()
    
    @contextmanager
    def transaction(self):
        """Transaction context manager."""
        if not self.connection:
            self.connect()
        
        try:
            yield self
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Transaction error: {e}")
            raise
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


def create_connection(
    db_type: str = "sqlite",
    **kwargs
) -> DatabaseManager:
    """Create database connection."""
    return DatabaseManager(db_type, **kwargs)


def execute_query(
    db_manager: DatabaseManager,
    query: str,
    **kwargs
) -> Optional[List[Dict[str, Any]]]:
    """Execute query."""
    return db_manager.execute(query, **kwargs)


def execute_transaction(
    db_manager: DatabaseManager,
    queries: List[tuple]
) -> None:
    """
    Execute transaction.
    
    Args:
        db_manager: Database manager
        queries: List of (query, params) tuples
    """
    with db_manager.transaction():
        for query, params in queries:
            db_manager.execute(query, params)



