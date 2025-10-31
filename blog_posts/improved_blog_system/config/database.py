"""
Database configuration and connection management
"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import QueuePool
from typing import AsyncGenerator

from .settings import get_settings


class DatabaseManager:
    """Database connection manager with async support."""
    
    def __init__(self):
        self.settings = get_settings()
        self.engine = None
        self.session_factory = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize database engine with optimized settings."""
        self.engine = create_async_engine(
            self.settings.database_url,
            poolclass=QueuePool,
            pool_size=self.settings.database_pool_size,
            max_overflow=self.settings.database_max_overflow,
            pool_pre_ping=self.settings.database_pool_pre_ping,
            echo=self.settings.debug,  # Log SQL queries in debug mode
            future=True,
        )
        
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False,
        )
    
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with proper cleanup."""
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def close(self):
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()


# Global database manager instance
db_manager = DatabaseManager()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database session."""
    async for session in db_manager.get_session():
        yield session


async def close_db_connections():
    """Close all database connections."""
    await db_manager.close()

