from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
from typing import AsyncGenerator, Dict, Any, Optional
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import (
from sqlalchemy.pool import QueuePool
from sqlalchemy import text, MetaData
import structlog
from .models import Base
from typing import Any, List, Dict, Optional
import logging
"""
Database Manager

Manages database connections, migrations, and session lifecycle.
"""

    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine
)


logger = structlog.get_logger()


class DatabaseManager:
    """
    Manages database connections and session lifecycle.
    
    Features:
    - Connection pooling with configurable settings
    - Automatic migrations in development
    - Session management with proper cleanup
    - Health monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        
    """__init__ function."""
self.config = config
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker] = None
        self._is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize database engine and session factory."""
        if self._is_initialized:
            return
        
        logger.info("Initializing database connection", url=self._sanitize_url(self.config["url"]))
        
        # Create async engine with connection pooling
        self.engine = create_async_engine(
            self.config["url"],
            pool_size=self.config.get("pool_size", 20),
            max_overflow=self.config.get("max_overflow", 10),
            pool_timeout=self.config.get("pool_timeout", 30),
            pool_recycle=3600,  # Recycle connections every hour
            pool_pre_ping=True,  # Validate connections before use
            poolclass=QueuePool,
            echo=self.config.get("echo", False),
            echo_pool=False,  # Disable pool logging for cleaner logs
        )
        
        # Create session factory
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,  # Manual control over flushing
            autocommit=False,
        )
        
        # Test connection
        await self._test_connection()
        
        self._is_initialized = True
        logger.info("Database initialized successfully")
    
    async def shutdown(self) -> None:
        """Shutdown database connections."""
        if not self._is_initialized:
            return
        
        logger.info("Shutting down database connections")
        
        if self.engine:
            await self.engine.dispose()
            self.engine = None
        
        self.session_factory = None
        self._is_initialized = False
        
        logger.info("Database shutdown completed")
    
    async def run_migrations(self) -> None:
        """Run database migrations (create tables)."""
        if not self._is_initialized:
            raise RuntimeError("Database manager not initialized")
        
        logger.info("Running database migrations")
        
        async with self.engine.begin() as conn:
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database migrations completed")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session with automatic cleanup.
        
        Usage:
            async with db_manager.get_session() as session:
                # Use session
                pass
        """
        if not self._is_initialized:
            raise RuntimeError("Database manager not initialized")
        
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    async def get_session_dependency(self) -> AsyncGenerator[AsyncSession, None]:
        """FastAPI dependency for getting database session."""
        async with self.get_session() as session:
            yield session
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        if not self._is_initialized:
            return {"status": "not_initialized", "healthy": False}
        
        try:
            async with self.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                result.fetchone()
            
            # Get connection pool stats
            pool = self.engine.pool
            pool_stats = {
                "size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "total": pool.size() + pool.overflow(),
            }
            
            return {
                "status": "healthy",
                "healthy": True,
                "pool_stats": pool_stats,
                "url": self._sanitize_url(self.config["url"])
            }
            
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "healthy": False,
                "error": str(e)
            }
    
    async def _test_connection(self) -> None:
        """Test database connection."""
        try:
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
        except Exception as e:
            logger.error("Database connection test failed", error=str(e))
            raise
    
    def _sanitize_url(self, url: str) -> str:
        """Sanitize database URL for logging (remove credentials)."""
        try:
            # Hide password in URL for logging
            if "@" in url:
                protocol_part, rest = url.split("://", 1)
                if "@" in rest:
                    creds_part, host_part = rest.split("@", 1)
                    if ":" in creds_part:
                        user, _ = creds_part.split(":", 1)
                        return f"{protocol_part}://{user}:***@{host_part}"
            return url
        except Exception:
            return "***SANITIZED***"


# Convenience functions for dependency injection
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get global database manager instance."""
    global _db_manager
    if _db_manager is None:
        raise RuntimeError("Database manager not initialized")
    return _db_manager


def set_database_manager(manager: DatabaseManager) -> None:
    """Set global database manager instance."""
    global _db_manager
    _db_manager = manager


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database session."""
    db_manager = get_database_manager()
    async with db_manager.get_session() as session:
        yield session 