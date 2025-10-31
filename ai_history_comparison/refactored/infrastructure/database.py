"""
Database Infrastructure
======================

This module provides database connectivity, configuration, and management
for the AI History Comparison system.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
from contextlib import contextmanager
import logging
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    echo_pool: bool = False
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'DatabaseConfig':
        """Create config from dictionary"""
        return cls(
            url=config['url'],
            pool_size=config.get('pool_size', 20),
            max_overflow=config.get('max_overflow', 30),
            pool_timeout=config.get('pool_timeout', 30),
            pool_recycle=config.get('pool_recycle', 3600),
            echo=config.get('echo', False),
            echo_pool=config.get('echo_pool', False)
        )


class DatabaseManager:
    """Manages database connections and sessions"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._initialized = False
    
    def initialize(self):
        """Initialize database connection"""
        if self._initialized:
            return
        
        try:
            # Create engine with connection pooling
            self._engine = create_engine(
                self.config.url,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=self.config.echo,
                echo_pool=self.config.echo_pool
            )
            
            # Create session factory
            self._session_factory = sessionmaker(
                bind=self._engine,
                expire_on_commit=False
            )
            
            # Test connection
            with self._engine.connect() as conn:
                conn.execute("SELECT 1")
            
            self._initialized = True
            logger.info("Database connection initialized successfully")
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def create_tables(self):
        """Create database tables"""
        if not self._initialized:
            raise RuntimeError("Database not initialized")
        
        try:
            # Import models here to avoid circular imports
            from ..core.domain import Base
            Base.metadata.create_all(self._engine)
            logger.info("Database tables created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables"""
        if not self._initialized:
            raise RuntimeError("Database not initialized")
        
        try:
            from ..core.domain import Base
            Base.metadata.drop_all(self._engine)
            logger.info("Database tables dropped successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to drop tables: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Session:
        """Get database session with automatic cleanup"""
        if not self._initialized:
            raise RuntimeError("Database not initialized")
        
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_session_factory(self) -> sessionmaker:
        """Get session factory"""
        if not self._initialized:
            raise RuntimeError("Database not initialized")
        return self._session_factory
    
    def get_engine(self) -> Engine:
        """Get database engine"""
        if not self._initialized:
            raise RuntimeError("Database not initialized")
        return self._engine
    
    def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        if not self._initialized:
            return {"status": "not_initialized", "healthy": False}
        
        try:
            with self._engine.connect() as conn:
                result = conn.execute("SELECT 1")
                result.fetchone()
            
            # Get pool status
            pool = self._engine.pool
            pool_status = {
                "size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "invalid": pool.invalid()
            }
            
            return {
                "status": "healthy",
                "healthy": True,
                "pool": pool_status
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "healthy": False,
                "error": str(e)
            }
    
    def close(self):
        """Close database connections"""
        if self._engine:
            self._engine.dispose()
            self._initialized = False
            logger.info("Database connections closed")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global _db_manager
    if _db_manager is None:
        raise RuntimeError("Database manager not initialized")
    return _db_manager


def initialize_database(config: DatabaseConfig) -> DatabaseManager:
    """Initialize global database manager"""
    global _db_manager
    _db_manager = DatabaseManager(config)
    _db_manager.initialize()
    return _db_manager


def close_database():
    """Close global database manager"""
    global _db_manager
    if _db_manager:
        _db_manager.close()
        _db_manager = None




