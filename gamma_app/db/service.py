"""
Database Service Implementation
"""

from typing import Any, Optional, Dict, List, AsyncGenerator
import logging
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from .base import DatabaseBase
from ..utils import retry_on_failure, log_execution_time

logger = logging.getLogger(__name__)


class DatabaseService(DatabaseBase):
    """Database service implementation using SQLAlchemy"""
    
    def __init__(
        self,
        connection_string: str,
        config_service=None,
        pool_size: int = 10,
        pool_recycle: int = 3600
    ):
        """Initialize database service"""
        self.connection_string = connection_string
        self.config_service = config_service
        self.engine = create_async_engine(
            connection_string,
            pool_size=pool_size,
            pool_recycle=pool_recycle,
            echo=False # Set to True for debug logging
        )
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        self._connected = True # Engine handles connection lazily
    
    async def connect(self) -> bool:
        """Test database connection"""
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            self._connected = True
            return True
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            self._connected = False
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from database"""
        try:
            await self.engine.dispose()
            self._connected = False
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from database: {e}")
            return False
    
    @log_execution_time
    @retry_on_failure(max_retries=3)
    async def execute_query(
        self,
        query: str,
        params: Optional[Dict] = None
    ) -> List[Dict]:
        """Execute raw SQL query"""
        if not self._connected: 
            # Try to reconnect or fail if relying on explicit connection state
            # For SQLAlchemy, engine handles it, but we respect the flag if manually managed
             if not await self.connect():
                 raise RuntimeError("Database not connected")

        try:
            async with self.engine.connect() as conn:
                result = await conn.execute(text(query), params or {})
                # Fetch all results if it's a SELECT statement
                try:
                    # distinct mapping to dicts
                    return [dict(row._mapping) for row in result.fetchall()]
                except SQLAlchemyError:
                    # Result might not return rows (e.g. INSERT/UPDATE)
                    return []
            
        except SQLAlchemyError as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[AsyncSession, None]:
        """Transaction context manager providing an AsyncSession"""
        async with self.session_factory() as session:
            async with session.begin():
                try:
                    yield session
                    # Commit is automatic with session.begin() context
                except Exception:
                   # Rollback is automatic with session.begin() context on error
                   raise

    async def migrate(self) -> bool:
        """Run migrations"""
        try:
            # TODO: Implement migrations using Alembic explicitly if needed via code
            # Usually handled by external CLI 'alembic upgrade head'
            logger.info("Migrations should be run via CLI")
            return True
        except Exception as e:
            logger.error(f"Error running migrations: {e}")
            return False
