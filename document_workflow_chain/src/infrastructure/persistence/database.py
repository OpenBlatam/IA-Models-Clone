"""
Database Configuration
======================

Advanced database configuration with connection pooling, migrations, and monitoring.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.pool import QueuePool
from sqlalchemy import event
from sqlalchemy.engine import Engine

from ...shared.config import get_settings, DatabaseConfig


logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Advanced database manager
    
    Provides connection pooling, session management, and monitoring.
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.settings = get_settings()
        self.config = config or self.settings.get_database_config()
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize database engine with advanced configuration"""
        try:
            # Create async engine with connection pooling
            self._engine = create_async_engine(
                self.config.url,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=True,  # Verify connections before use
                echo=self.config.echo,
                echo_pool=self.config.echo,
                future=True,
                connect_args={
                    "server_settings": {
                        "application_name": "workflow_chain_v3",
                        "jit": "off"  # Disable JIT for better performance
                    }
                } if self.config.type.value == "postgresql" else {}
            )
            
            # Create session factory
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False
            )
            
            # Add event listeners for monitoring
            self._add_event_listeners()
            
            logger.info(f"Database engine initialized: {self.config.type.value}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise
    
    def _add_event_listeners(self):
        """Add event listeners for database monitoring"""
        
        @event.listens_for(self._engine.sync_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set SQLite pragmas for better performance"""
            if self.config.type.value == "sqlite":
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.close()
        
        @event.listens_for(self._engine.sync_engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            """Log connection checkout"""
            logger.debug("Database connection checked out")
        
        @event.listens_for(self._engine.sync_engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            """Log connection checkin"""
            logger.debug("Database connection checked in")
        
        @event.listens_for(self._engine.sync_engine, "invalidate")
        def receive_invalidate(dbapi_connection, connection_record, exception):
            """Log connection invalidation"""
            logger.warning(f"Database connection invalidated: {exception}")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic cleanup"""
        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()
    
    async def get_session_dependency(self) -> AsyncGenerator[AsyncSession, None]:
        """FastAPI dependency for database session"""
        async with self.get_session() as session:
            yield session
    
    async def execute_raw_sql(self, sql: str, parameters: Optional[dict] = None):
        """Execute raw SQL query"""
        async with self.get_session() as session:
            result = await session.execute(sql, parameters or {})
            return result.fetchall()
    
    async def health_check(self) -> dict:
        """Perform database health check"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Test connection
            async with self.get_session() as session:
                result = await session.execute("SELECT 1")
                result.fetchone()
            
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Get connection pool status
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
                "response_time_ms": round(response_time, 2),
                "pool_status": pool_status,
                "database_type": self.config.type.value,
                "database_name": self.config.name
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "database_type": self.config.type.value,
                "database_name": self.config.name
            }
    
    async def get_statistics(self) -> dict:
        """Get database statistics"""
        try:
            pool = self._engine.pool
            
            # Get table statistics (PostgreSQL specific)
            table_stats = {}
            if self.config.type.value == "postgresql":
                async with self.get_session() as session:
                    result = await session.execute("""
                        SELECT 
                            schemaname,
                            tablename,
                            n_tup_ins as inserts,
                            n_tup_upd as updates,
                            n_tup_del as deletes,
                            n_live_tup as live_tuples,
                            n_dead_tup as dead_tuples
                        FROM pg_stat_user_tables
                        ORDER BY n_live_tup DESC
                        LIMIT 10
                    """)
                    table_stats = [dict(row._mapping) for row in result.fetchall()]
            
            return {
                "connection_pool": {
                    "size": pool.size(),
                    "checked_in": pool.checkedin(),
                    "checked_out": pool.checkedout(),
                    "overflow": pool.overflow(),
                    "invalid": pool.invalid()
                },
                "configuration": {
                    "type": self.config.type.value,
                    "host": self.config.host,
                    "port": self.config.port,
                    "name": self.config.name,
                    "pool_size": self.config.pool_size,
                    "max_overflow": self.config.max_overflow,
                    "pool_timeout": self.config.pool_timeout,
                    "pool_recycle": self.config.pool_recycle
                },
                "table_statistics": table_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get database statistics: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Close database connections"""
        if self._engine:
            await self._engine.dispose()
            logger.info("Database connections closed")
    
    def get_engine(self) -> AsyncEngine:
        """Get database engine"""
        return self._engine
    
    def get_session_factory(self) -> async_sessionmaker:
        """Get session factory"""
        return self._session_factory


# Global database manager instance
_database_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global _database_manager
    if _database_manager is None:
        _database_manager = DatabaseManager()
    return _database_manager


def get_database_session() -> async_sessionmaker:
    """Get database session factory"""
    return get_database_manager().get_session_factory()


# FastAPI dependency
async def get_database_session_dependency() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database session"""
    async with get_database_manager().get_session() as session:
        yield session


# Migration utilities
class MigrationManager:
    """Database migration manager"""
    
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager
    
    async def create_tables(self):
        """Create all tables"""
        try:
            from .models import Base
            async with self.database_manager.get_engine().begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    async def drop_tables(self):
        """Drop all tables"""
        try:
            from .models import Base
            async with self.database_manager.get_engine().begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise
    
    async def check_migrations(self) -> dict:
        """Check migration status"""
        try:
            # In a real implementation, you would check Alembic migration status
            return {
                "status": "up_to_date",
                "current_revision": "latest",
                "pending_migrations": []
            }
        except Exception as e:
            logger.error(f"Failed to check migrations: {e}")
            return {"error": str(e)}


# Database initialization
async def initialize_database():
    """Initialize database with tables and data"""
    try:
        database_manager = get_database_manager()
        migration_manager = MigrationManager(database_manager)
        
        # Create tables
        await migration_manager.create_tables()
        
        # Check health
        health = await database_manager.health_check()
        if health["status"] != "healthy":
            raise Exception(f"Database health check failed: {health}")
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


# Cleanup
async def cleanup_database():
    """Cleanup database connections"""
    try:
        database_manager = get_database_manager()
        await database_manager.close()
        logger.info("Database cleanup completed")
    except Exception as e:
        logger.error(f"Database cleanup failed: {e}")
        raise




