"""
Connection Pool Optimization

Optimized connection pooling for better performance.
"""

import logging
from typing import Optional
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool, NullPool

from ..config import settings

logger = logging.getLogger(__name__)


def create_optimized_engine(database_url: str, echo: bool = False) -> Engine:
    """
    Create optimized database engine with connection pooling.
    
    Args:
        database_url: Database URL
        echo: Whether to echo SQL queries
        
    Returns:
        Optimized SQLAlchemy engine
        
    Raises:
        ValueError: If database_url is None or empty
        Exception: If engine creation fails
    """
    if not database_url or not database_url.strip():
        raise ValueError("database_url cannot be None or empty")
    
    database_url = database_url.strip()
    
    try:
        # Determine pool class based on database type
        is_sqlite = "sqlite" in database_url.lower()
        
        if is_sqlite:
            # SQLite doesn't benefit from connection pooling
            poolclass = NullPool
            connect_args = {"check_same_thread": False}
        else:
            # Use QueuePool for PostgreSQL/MySQL
            poolclass = QueuePool
            connect_args = {}
        
        engine = create_engine(
            database_url,
            poolclass=poolclass,
            pool_size=20,  # Number of connections to maintain
            max_overflow=10,  # Additional connections beyond pool_size
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,  # Recycle connections after 1 hour
            echo=echo,
            connect_args=connect_args,
            # Performance optimizations
            execution_options={
                "autocommit": False,
                "isolation_level": "READ COMMITTED"
            }
        )
        
        # Add event listeners for connection optimization
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            """Optimize SQLite connections."""
            if is_sqlite:
                try:
                    cursor = dbapi_conn.cursor()
                    # Enable WAL mode for better concurrency
                    cursor.execute("PRAGMA journal_mode=WAL")
                    # Increase cache size
                    cursor.execute("PRAGMA cache_size=-64000")  # 64MB
                    # Enable foreign keys
                    cursor.execute("PRAGMA foreign_keys=ON")
                    # Optimize for speed over safety
                    cursor.execute("PRAGMA synchronous=NORMAL")
                    cursor.close()
                except Exception as e:
                    logger.warning(f"Failed to set SQLite pragmas: {e}")
        
        logger.info(
            f"Optimized engine created",
            pool_size=20,
            max_overflow=10,
            database_type="sqlite" if is_sqlite else "postgresql/mysql"
        )
        
        return engine
    except Exception as e:
        logger.error(
            f"Failed to create database engine: {e}",
            exc_info=True,
            database_url=database_url.split("@")[-1] if "@" in database_url else "local"
        )
        raise











