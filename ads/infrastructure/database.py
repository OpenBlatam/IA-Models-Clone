"""
Database infrastructure for the ads feature.

This module consolidates database functionality from:
- db_service.py (basic database operations)
- optimized_db_service.py (production database with connection pooling)

Provides unified database management, connection pooling, and repository interfaces.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass, field

try:
    import aioredis  # type: ignore
except Exception:  # pragma: no cover - optional in tests
    aioredis = None  # type: ignore[assignment]
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy import and_, desc, func, select, update, delete
from sqlalchemy.orm import selectinload

try:
    from onyx.db.ads import AdsGeneration, BackgroundRemoval, AdsAnalytics  # type: ignore
except Exception:  # pragma: no cover - optionalize DB models for tests
    from typing import Any as _Any
    AdsGeneration = BackgroundRemoval = AdsAnalytics = _Any  # type: ignore
try:
    from onyx.utils.logger import setup_logger  # type: ignore
except Exception:  # pragma: no cover - fallback minimal logger for tests
    import logging as _logging

    def setup_logger(name: str | None = None):  # type: ignore[override]
        logger = _logging.getLogger(name or __name__)
        if not _logging.getLogger().handlers:
            _logging.basicConfig(level=_logging.INFO)
        return logger
from ..config import get_optimized_settings

logger = setup_logger()

@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    url: str
    pool_size: int = 20
    max_overflow: int = 30
    pool_pre_ping: bool = True
    pool_recycle: int = 3600
    pool_timeout: int = 30
    echo: bool = False
    max_retries: int = 100
    retry_delay: float = 1.0

class ConnectionPool:
    """Manages database connection pooling."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._engine = None
        self._session_factory = None
        self._pool_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "idle_connections": 0
        }
    
    async def get_engine(self):
        """Get or create the database engine."""
        if self._engine is None:
            self._engine = create_async_engine(
                self.config.url,
                echo=self.config.echo,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_pre_ping=self.config.pool_pre_ping,
                pool_recycle=self.config.pool_recycle,
                pool_timeout=self.config.pool_timeout
            )
            logger.info(f"Database engine created with pool size {self.config.pool_size}")
        return self._engine
    
    async def get_session_factory(self):
        """Get or create the session factory."""
        if self._session_factory is None:
            engine = await self.get_engine()
            self._session_factory = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            logger.info("Session factory created")
        return self._session_factory
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session with proper error handling."""
        session_factory = await self.get_session_factory()
        async with session_factory() as session:
            try:
                self._pool_stats["active_connections"] += 1
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                self._pool_stats["active_connections"] -= 1
                await session.close()
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        if self._engine:
            pool = self._engine.pool
            self._pool_stats.update({
                "total_connections": pool.size(),
                "active_connections": pool.checkedin() + pool.checkedout(),
                "idle_connections": pool.checkedin()
            })
        return self._pool_stats.copy()
    
    async def close(self):
        """Close all database connections."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database connections closed")

class DatabaseManager:
    """Manages database operations and connection pooling."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        if config is None:
            settings = get_optimized_settings()
            config = DatabaseConfig(
                url=settings.database_url,
                pool_size=settings.database_pool_size,
                max_overflow=settings.database_max_overflow,
                pool_timeout=settings.database_pool_timeout,
                echo=settings.debug
            )
        
        self.config = config
        self.connection_pool = ConnectionPool(config)
        self._redis_client = None
    
    @property
    async def redis_client(self):
        """Get Redis client for caching."""
        if self._redis_client is None:
            settings = get_optimized_settings()
            if aioredis is None or not getattr(settings, "redis_url", None):
                self._redis_client = None
            else:
                self._redis_client = await aioredis.from_url(
                    settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=50
                )
        return self._redis_client
    
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session."""
        async with self.connection_pool.get_session() as session:
            yield session
    
    async def execute_query(self, query_func, *args, **kwargs):
        """Execute a database query with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                async with self.get_session() as session:
                    return await query_func(session, *args, **kwargs)
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    logger.error(f"Database query failed after {self.config.max_retries} attempts: {e}")
                    raise
                logger.warning(f"Database query attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get database pool statistics."""
        return await self.connection_pool.get_pool_stats()
    
    async def close(self):
        """Close database connections."""
        await self.connection_pool.close()
        if self._redis_client:
            await self._redis_client.close()

# Repository interfaces (implemented in repositories.py)
class AdsRepository(ABC):
    """Abstract interface for ads repository operations."""
    
    @abstractmethod
    async def create(self, ads_data: Dict[str, Any]) -> AdsGeneration:
        """Create a new ads generation record."""
        pass
    
    @abstractmethod
    async def get_by_id(self, ads_id: int, user_id: int) -> Optional[AdsGeneration]:
        """Get an ads generation record by ID."""
        pass
    
    @abstractmethod
    async def list_by_user(self, user_id: int, limit: int = 100, offset: int = 0) -> List[AdsGeneration]:
        """List ads generation records for a user."""
        pass
    
    @abstractmethod
    async def update(self, ads_id: int, user_id: int, update_data: Dict[str, Any]) -> Optional[AdsGeneration]:
        """Update an ads generation record."""
        pass
    
    @abstractmethod
    async def delete(self, ads_id: int, user_id: int) -> bool:
        """Soft delete an ads generation record."""
        pass

class CampaignRepository(ABC):
    """Abstract interface for campaign repository operations."""
    
    @abstractmethod
    async def create(self, campaign_data: Dict[str, Any]) -> Any:
        """Create a new campaign."""
        pass
    
    @abstractmethod
    async def get_by_id(self, campaign_id: int, user_id: int) -> Optional[Any]:
        """Get a campaign by ID."""
        pass
    
    @abstractmethod
    async def list_by_user(self, user_id: int, limit: int = 100, offset: int = 0) -> List[Any]:
        """List campaigns for a user."""
        pass
    
    @abstractmethod
    async def update(self, campaign_id: int, user_id: int, update_data: Dict[str, Any]) -> Optional[Any]:
        """Update a campaign."""
        pass
    
    @abstractmethod
    async def delete(self, campaign_id: int, user_id: int) -> bool:
        """Delete a campaign."""
        pass

class GroupRepository(ABC):
    """Abstract interface for ad group repository operations."""
    
    @abstractmethod
    async def create(self, group_data: Dict[str, Any]) -> Any:
        """Create a new ad group."""
        pass
    
    @abstractmethod
    async def get_by_id(self, group_id: int, user_id: int) -> Optional[Any]:
        """Get an ad group by ID."""
        pass
    
    @abstractmethod
    async def list_by_campaign(self, campaign_id: int, user_id: int) -> List[Any]:
        """List ad groups for a campaign."""
        pass
    
    @abstractmethod
    async def update(self, group_id: int, user_id: int, update_data: Dict[str, Any]) -> Optional[Any]:
        """Update an ad group."""
        pass
    
    @abstractmethod
    async def delete(self, group_id: int, user_id: int) -> bool:
        """Delete an ad group."""
        pass

class PerformanceRepository(ABC):
    """Abstract interface for performance repository operations."""
    
    @abstractmethod
    async def create(self, performance_data: Dict[str, Any]) -> Any:
        """Create a new performance record."""
        pass
    
    @abstractmethod
    async def get_by_id(self, performance_id: int, user_id: int) -> Optional[Any]:
        """Get a performance record by ID."""
        pass
    
    @abstractmethod
    async def list_by_ads(self, ads_id: int, user_id: int) -> List[Any]:
        """List performance records for an ad."""
        pass
    
    @abstractmethod
    async def update(self, performance_id: int, user_id: int, update_data: Dict[str, Any]) -> Optional[Any]:
        """Update a performance record."""
        pass

class AnalyticsRepository(ABC):
    """Abstract interface for analytics repository operations."""
    
    @abstractmethod
    async def create(self, analytics_data: Dict[str, Any]) -> AdsAnalytics:
        """Create a new analytics record."""
        pass
    
    @abstractmethod
    async def get_by_id(self, analytics_id: int, user_id: int) -> Optional[AdsAnalytics]:
        """Get an analytics record by ID."""
        pass
    
    @abstractmethod
    async def list_by_ads(self, ads_id: int, user_id: int) -> List[AdsAnalytics]:
        """List analytics records for an ad."""
        pass
    
    @abstractmethod
    async def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Get user statistics."""
        pass

class OptimizationRepository(ABC):
    """Abstract interface for optimization repository operations."""
    
    @abstractmethod
    async def create(self, optimization_data: Dict[str, Any]) -> Any:
        """Create a new optimization record."""
        pass
    
    @abstractmethod
    async def get_by_id(self, optimization_id: int, user_id: int) -> Optional[Any]:
        """Get an optimization record by ID."""
        pass
    
    @abstractmethod
    async def list_by_ads(self, ads_id: int, user_id: int) -> List[Any]:
        """List optimization records for an ad."""
        pass
    
    @abstractmethod
    async def get_optimization_history(self, ads_id: int, user_id: int) -> List[Any]:
        """Get optimization history for an ad."""
        pass
