"""
Database connection and management.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager
import os

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or self._get_default_database_url()
        self.engine = None
        self.async_engine = None
        self.session_factory = None
        self.async_session_factory = None
        self._initialized = False
    
    def _get_default_database_url(self) -> str:
        """Get default database URL from environment or use SQLite."""
        return os.getenv(
            "DATABASE_URL", 
            "sqlite:///./export_ia.db"
        )
    
    def _get_async_database_url(self) -> str:
        """Get async database URL."""
        if self.database_url.startswith("sqlite"):
            return self.database_url.replace("sqlite://", "sqlite+aiosqlite://")
        elif self.database_url.startswith("postgresql"):
            return self.database_url.replace("postgresql://", "postgresql+asyncpg://")
        elif self.database_url.startswith("mysql"):
            return self.database_url.replace("mysql://", "mysql+aiomysql://")
        else:
            return self.database_url
    
    async def initialize(self) -> None:
        """Initialize database connections."""
        if self._initialized:
            return
        
        try:
            # Create sync engine
            self.engine = create_engine(
                self.database_url,
                echo=False,
                pool_pre_ping=True,
                pool_recycle=300
            )
            
            # Create async engine
            async_url = self._get_async_database_url()
            self.async_engine = create_async_engine(
                async_url,
                echo=False,
                pool_pre_ping=True,
                pool_recycle=300
            )
            
            # Create session factories
            self.session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False
            )
            
            self.async_session_factory = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                autocommit=False,
                autoflush=False
            )
            
            # Create tables
            await self.create_tables()
            
            self._initialized = True
            logger.info("Database manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def create_tables(self) -> None:
        """Create database tables."""
        try:
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    async def drop_tables(self) -> None:
        """Drop all database tables."""
        try:
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            logger.info("Database tables dropped")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise
    
    @asynccontextmanager
    async def get_async_session(self):
        """Get async database session."""
        if not self._initialized:
            await self.initialize()
        
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    def get_sync_session(self):
        """Get sync database session."""
        if not self._initialized:
            raise RuntimeError("Database not initialized")
        
        return self.session_factory()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check database health."""
        try:
            async with self.get_async_session() as session:
                await session.execute("SELECT 1")
                return {
                    "status": "healthy",
                    "database_url": self.database_url,
                    "initialized": self._initialized
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "database_url": self.database_url,
                "initialized": self._initialized
            }
    
    async def get_connection_info(self) -> Dict[str, Any]:
        """Get database connection information."""
        return {
            "database_url": self.database_url,
            "async_database_url": self._get_async_database_url(),
            "initialized": self._initialized,
            "pool_size": self.engine.pool.size() if self.engine else None,
            "checked_out_connections": self.engine.pool.checkedout() if self.engine else None
        }
    
    async def close(self) -> None:
        """Close database connections."""
        if self.async_engine:
            await self.async_engine.dispose()
        
        if self.engine:
            self.engine.dispose()
        
        self._initialized = False
        logger.info("Database connections closed")


class DatabaseConfig:
    """Database configuration."""
    
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///./export_ia.db")
        self.pool_size = int(os.getenv("DB_POOL_SIZE", "10"))
        self.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "20"))
        self.pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
        self.pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "300"))
        self.echo = os.getenv("DB_ECHO", "false").lower() == "true"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "database_url": self.database_url,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle,
            "echo": self.echo
        }


# Global database manager instance
_database_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _database_manager
    if _database_manager is None:
        _database_manager = DatabaseManager()
    return _database_manager


async def init_database() -> DatabaseManager:
    """Initialize the global database manager."""
    db_manager = get_database_manager()
    await db_manager.initialize()
    return db_manager


async def close_database() -> None:
    """Close the global database manager."""
    global _database_manager
    if _database_manager:
        await _database_manager.close()
        _database_manager = None




