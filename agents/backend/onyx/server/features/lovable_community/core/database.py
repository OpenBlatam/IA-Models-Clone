"""
Database initialization and management module

Handles database engine creation, session management, and connection verification.
"""

import logging
from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine

from ..config import settings
from ..models import Base
from .connection_pool import create_optimized_engine

logger = logging.getLogger(__name__)

# Global variables for database engine and sessionmaker
_db_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def get_db_engine() -> Engine:
    """
    Get or create the optimized database engine (singleton pattern).
    
    Uses connection pooling and optimizations for better performance.
    
    Returns:
        Engine: Optimized SQLAlchemy database engine
        
    Raises:
        Exception: If engine creation fails
    """
    global _db_engine
    if _db_engine is None:
        try:
            _db_engine = create_optimized_engine(
                settings.database_url,
                echo=settings.database_echo
            )
            logger.info("Optimized database engine created successfully")
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}", exc_info=True)
            raise
    return _db_engine


def get_session_local() -> sessionmaker:
    """
    Get or create the sessionmaker (singleton pattern).
    
    Returns:
        sessionmaker: SQLAlchemy sessionmaker
    """
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_db_engine()
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        )
        logger.info("Session maker created successfully")
    return _SessionLocal


def init_database() -> None:
    """
    Initialize database by creating all tables.
    
    Raises:
        Exception: If table creation fails
    """
    try:
        engine = get_db_engine()
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
        raise


def verify_database_connection() -> bool:
    """
    Verify database connection by executing a simple query.
    
    Returns:
        bool: True if connection is successful, False otherwise
        
    Raises:
        ValueError: If database URL is invalid
    """
    try:
        if not settings.database_url or not settings.database_url.strip():
            raise ValueError("Database URL cannot be None or empty")
        
        engine = get_db_engine()
        if engine is None:
            logger.error("Database engine is None")
            return False
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            # Consume the result to ensure query executed
            result.scalar()
        logger.info("Database connection verified successfully")
        return True
    except ValueError:
        raise
    except Exception as e:
        logger.error(
            f"Database connection verification failed: {e}",
            exc_info=True,
            database_url=settings.database_url.split("@")[-1] if "@" in settings.database_url else "local"
        )
        return False


class DatabaseManager:
    """
    Database manager class for centralized database operations.
    
    Provides methods for:
    - Database initialization
    - Connection management
    - Session creation
    - Health checks
    """
    
    def __init__(self):
        self._engine: Optional[Engine] = None
        self._session_local: Optional[sessionmaker] = None
    
    @property
    def engine(self) -> Engine:
        """Get database engine"""
        if self._engine is None:
            self._engine = get_db_engine()
        return self._engine
    
    @property
    def session_local(self) -> sessionmaker:
        """Get sessionmaker"""
        if self._session_local is None:
            self._session_local = get_session_local()
        return self._session_local
    
    def initialize(self) -> None:
        """Initialize database tables"""
        init_database()
    
    def verify_connection(self) -> bool:
        """Verify database connection"""
        return verify_database_connection()
    
    def get_session(self) -> Session:
        """
        Create a new database session.
        
        Returns:
            Session: New database session
            
        Note:
            Caller is responsible for closing the session
        """
        return self.session_local()
    
    def health_check(self) -> dict:
        """
        Perform a comprehensive database health check.
        
        Returns:
            dict: Health check results with status and details
        """
        try:
            is_connected = self.verify_connection()
            return {
                "status": "healthy" if is_connected else "unhealthy",
                "connected": is_connected,
                "database_url": settings.database_url.split("@")[-1] if "@" in settings.database_url else "local",
                "echo": settings.database_echo
            }
        except Exception as e:
            logger.error(f"Database health check failed: {e}", exc_info=True)
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e)
            }

