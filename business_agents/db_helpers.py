"""Database integration helpers and utilities."""
from typing import Optional, Type, TypeVar, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine
from sqlalchemy.orm import declarative_base, sessionmaker
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

Base = declarative_base()


class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[sessionmaker] = None
    
    async def init(self, database_url: Optional[str] = None):
        """Initialize database connection."""
        url = database_url or self.database_url
        if not url:
            logger.warning("No database URL provided. Database features disabled.")
            return
        
        try:
            self.engine = create_async_engine(
                url,
                echo=False,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20
            )
            self.session_factory = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            logger.info("Database connection initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self.engine = None
            self.session_factory = None
    
    @asynccontextmanager
    async def get_session(self):
        """Get a database session context manager."""
        if not self.session_factory:
            raise RuntimeError("Database not initialized. Call init() first.")
        
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def close(self):
        """Close database connection."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connection closed.")


class Repository:
    """Base repository pattern for database operations."""
    
    def __init__(self, session: AsyncSession, model: Type[T]):
        self.session = session
        self.model = model
    
    async def get_by_id(self, id: Any) -> Optional[T]:
        """Get entity by ID."""
        result = await self.session.get(self.model, id)
        return result
    
    async def get_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        """Get all entities with pagination."""
        from sqlalchemy import select
        stmt = select(self.model).limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def create(self, **kwargs) -> T:
        """Create new entity."""
        instance = self.model(**kwargs)
        self.session.add(instance)
        await self.session.flush()
        return instance
    
    async def update(self, id: Any, **kwargs) -> Optional[T]:
        """Update entity by ID."""
        instance = await self.get_by_id(id)
        if not instance:
            return None
        
        for key, value in kwargs.items():
            setattr(instance, key, value)
        
        await self.session.flush()
        return instance
    
    async def delete(self, id: Any) -> bool:
        """Delete entity by ID."""
        instance = await self.get_by_id(id)
        if not instance:
            return False
        
        await self.session.delete(instance)
        await self.session.flush()
        return True
    
    async def count(self) -> int:
        """Count total entities."""
        from sqlalchemy import func, select
        stmt = select(func.count()).select_from(self.model)
        result = await self.session.execute(stmt)
        return result.scalar() or 0


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> Optional[DatabaseManager]:
    """Get global database manager instance."""
    return _db_manager


def init_db_manager(database_url: Optional[str] = None) -> DatabaseManager:
    """Initialize global database manager."""
    global _db_manager
    _db_manager = DatabaseManager(database_url)
    return _db_manager


