"""
Database Session Management
Centralized database session and connection management
"""

import logging
from typing import Generator, Optional
from contextlib import contextmanager
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import QueuePool

from ...utils.config import get_settings

logger = logging.getLogger(__name__)

class DatabaseSessionManager:
    """Manages database connections and sessions"""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize database session manager"""
        settings = get_settings()
        self.database_url = database_url or settings.database_url
        self.engine: Optional[Engine] = None
        self.session_factory: Optional[sessionmaker] = None
        self._scoped_session: Optional[scoped_session] = None
        
    def initialize(self):
        """Initialize database engine and session factory"""
        if self.engine is None:
            self.engine = create_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=False
            )
            self.session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
            self._scoped_session = scoped_session(self.session_factory)
            logger.info("Database session manager initialized")
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session context manager"""
        if self.session_factory is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_scoped_session(self) -> Session:
        """Get a scoped session (for use in dependency injection)"""
        if self._scoped_session is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._scoped_session()
    
    def close(self):
        """Close all database connections"""
        if self._scoped_session:
            self._scoped_session.remove()
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")

_db_manager: Optional[DatabaseSessionManager] = None

def get_db_manager() -> DatabaseSessionManager:
    """Get the global database session manager"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseSessionManager()
        _db_manager.initialize()
    return _db_manager

def get_db_session() -> Generator[Session, None, None]:
    """FastAPI dependency for database session"""
    db_manager = get_db_manager()
    with db_manager.get_session() as session:
        yield session







