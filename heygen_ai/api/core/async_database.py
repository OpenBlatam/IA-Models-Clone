from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
from typing import Optional, Dict, Any, List, Union, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
from sqlalchemy.ext.asyncio import (
from sqlalchemy.pool import QueuePool
from sqlalchemy import text, MetaData
from sqlalchemy.exc import SQLAlchemyError, OperationalError
    import asyncpg
    import aiomysql
    import aiosqlite
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Enhanced Async Database System for HeyGen AI API
Supports multiple database backends: asyncpg, aiomysql, aiosqlite
Features: Connection pooling, health checks, migrations, monitoring
"""


# SQLAlchemy async imports
    AsyncSession, 
    create_async_engine, 
    async_sessionmaker,
    AsyncEngine
)

# Database driver imports
try:
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

try:
    AIOMYSQL_AVAILABLE = True
except ImportError:
    AIOMYSQL_AVAILABLE = False

try:
    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False

logger = logging.getLogger(__name__)


class DatabaseType(Enum):
    """Supported database types."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str
    type: DatabaseType
    pool_size: int = 20
    max_overflow: int = 30
    pool_pre_ping: bool = True
    pool_recycle: int = 3600
    echo: bool = False
    echo_pool: bool = False
    pool_timeout: int = 30
    pool_reset_on_return: str = "commit"
    connect_args: Optional[Dict[str, Any]] = None


@dataclass
class ConnectionStats:
    """Database connection statistics."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    overflow_connections: int = 0
    checked_out_connections: int = 0
    checked_in_connections: int = 0
    invalid_connections: int = 0
    last_health_check: Optional[datetime] = None
    health_check_status: bool = True


class AsyncDatabaseManager:
    """
    Enhanced async database manager with support for multiple backends.
    Features: Connection pooling, health checks, monitoring, migrations
    """
    
    def __init__(self, config: DatabaseConfig):
        
    """__init__ function."""
self.config = config
        self.engine: Optional[AsyncEngine] = None
        self.async_session: Optional[async_sessionmaker] = None
        self.stats = ConnectionStats()
        self.health_check_task: Optional[asyncio.Task] = None
        self.is_initialized = False
        
        # Validate database type and driver availability
        self._validate_database_type()
    
    def _validate_database_type(self) -> bool:
        """Validate database type and driver availability."""
        if self.config.type == DatabaseType.POSTGRESQL and not ASYNCPG_AVAILABLE:
            raise ImportError("asyncpg is required for PostgreSQL support")
        elif self.config.type == DatabaseType.MYSQL and not AIOMYSQL_AVAILABLE:
            raise ImportError("aiomysql is required for MySQL support")
        elif self.config.type == DatabaseType.SQLITE and not AIOSQLITE_AVAILABLE:
            raise ImportError("aiosqlite is required for SQLite support")
    
    async def initialize(self) -> None:
        """Initialize database connection and create engine."""
        try:
            logger.info(f"Initializing {self.config.type.value} database...")
            
            # Create async engine with optimized settings
            self.engine = create_async_engine(
                self.config.url,
                echo=self.config.echo,
                echo_pool=self.config.echo_pool,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_pre_ping=self.config.pool_pre_ping,
                pool_recycle=self.config.pool_recycle,
                pool_timeout=self.config.pool_timeout,
                pool_reset_on_return=self.config.pool_reset_on_return,
                connect_args=self.config.connect_args or {},
                future=True
            )
            
            # Create async session factory
            self.async_session = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
                autocommit=False
            )
            
            # Test connection
            await self._test_connection()
            
            # Start health monitoring
            await self._start_health_monitoring()
            
            self.is_initialized = True
            logger.info(f"✓ {self.config.type.value} database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def _test_connection(self) -> None:
        """Test database connection."""
        try:
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            logger.info("✓ Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise
    
    async def _start_health_monitoring(self) -> None:
        """Start periodic health monitoring."""
        if self.health_check_task:
            self.health_check_task.cancel()
        
        self.health_check_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("✓ Database health monitoring started")
    
    async def _health_monitor_loop(self) -> None:
        """Health monitoring loop."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _perform_health_check(self) -> None:
        """Perform database health check."""
        try:
            start_time = datetime.now()
            
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            # Update connection stats
            if self.engine:
                pool = self.engine.pool
                self.stats.total_connections = pool.size()
                self.stats.active_connections = pool.checkedin()
                self.stats.idle_connections = pool.checkedout()
                self.stats.overflow_connections = pool.overflow()
                self.stats.checked_out_connections = pool.checkedout()
                self.stats.checked_in_connections = pool.checkedin()
            
            self.stats.last_health_check = datetime.now()
            self.stats.health_check_status = True
            
            response_time = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Database health check passed in {response_time:.3f}s")
            
        except Exception as e:
            self.stats.health_check_status = False
            self.stats.last_health_check = datetime.now()
            logger.warning(f"Database health check failed: {e}")
    
    async def close(self) -> None:
        """Close database connections and cleanup."""
        try:
            # Stop health monitoring
            if self.health_check_task:
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Close engine
            if self.engine:
                await self.engine.dispose()
            
            self.is_initialized = False
            logger.info("✓ Database connections closed")
            
        except Exception as e:
            logger.error(f"Error closing database: {e}")
            raise
    
    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        """Get database session with automatic cleanup."""
        if not self.is_initialized:
            raise RuntimeError("Database not initialized")
        
        session = self.async_session()
        try:
            yield session
        except Exception as e:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a raw SQL query."""
        async with self.get_session() as session:
            result = await session.execute(text(query), params or {})
            return result
    
    async def execute_many(self, query: str, params_list: List[Dict[str, Any]]) -> None:
        """Execute multiple queries with different parameters."""
        async with self.get_session() as session:
            for params in params_list:
                await session.execute(text(query), params)
            await session.commit()
    
    async def get_connection_stats(self) -> ConnectionStats:
        """Get current connection statistics."""
        return self.stats
    
    async def is_healthy(self) -> bool:
        """Check if database is healthy."""
        return self.stats.health_check_status
    
    async def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics."""
        try:
            async with self.get_session() as session:
                # Get database version
                if self.config.type == DatabaseType.POSTGRESQL:
                    version_result = await session.execute(text("SELECT version()"))
                    version = version_result.scalar()
                elif self.config.type == DatabaseType.MYSQL:
                    version_result = await session.execute(text("SELECT VERSION()"))
                    version = version_result.scalar()
                elif self.config.type == DatabaseType.SQLITE:
                    version_result = await session.execute(text("SELECT sqlite_version()"))
                    version = version_result.scalar()
                else:
                    version = "Unknown"
                
                # Get table count
                if self.config.type == DatabaseType.POSTGRESQL:
                    tables_result = await session.execute(text(
                        "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'"
                    ))
                elif self.config.type == DatabaseType.MYSQL:
                    tables_result = await session.execute(text(
                        "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = DATABASE()"
                    ))
                elif self.config.type == DatabaseType.SQLITE:
                    tables_result = await session.execute(text(
                        "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                    ))
                
                table_count = tables_result.scalar()
                
                return {
                    "type": self.config.type.value,
                    "version": version,
                    "table_count": table_count,
                    "connection_stats": self.stats,
                    "pool_size": self.config.pool_size,
                    "max_overflow": self.config.max_overflow,
                    "is_healthy": self.stats.health_check_status,
                    "last_health_check": self.stats.last_health_check.isoformat() if self.stats.last_health_check else None
                }
                
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {
                "type": self.config.type.value,
                "error": str(e),
                "is_healthy": False
            }


class DatabaseConnectionPool:
    """
    Connection pool manager for multiple database backends.
    Supports load balancing and failover.
    """
    
    def __init__(self) -> Any:
        self.databases: Dict[str, AsyncDatabaseManager] = {}
        self.primary_db: Optional[str] = None
        self.failover_dbs: List[str] = []
    
    async def add_database(self, name: str, config: DatabaseConfig, is_primary: bool = False) -> None:
        """Add a database to the pool."""
        db_manager = AsyncDatabaseManager(config)
        await db_manager.initialize()
        
        self.databases[name] = db_manager
        
        if is_primary:
            self.primary_db = name
        else:
            self.failover_dbs.append(name)
        
        logger.info(f"Added database '{name}' to pool (primary: {is_primary})")
    
    async def get_session(self, db_name: Optional[str] = None) -> AsyncSession:
        """Get database session from specified or primary database."""
        if db_name:
            if db_name not in self.databases:
                raise ValueError(f"Database '{db_name}' not found in pool")
            return self.databases[db_name].get_session()
        
        if not self.primary_db:
            raise RuntimeError("No primary database configured")
        
        return self.databases[self.primary_db].get_session()
    
    async def get_healthy_database(self) -> Optional[str]:
        """Get the name of a healthy database."""
        # Check primary first
        if self.primary_db and await self.databases[self.primary_db].is_healthy():
            return self.primary_db
        
        # Check failover databases
        for db_name in self.failover_dbs:
            if await self.databases[db_name].is_healthy():
                return db_name
        
        return None
    
    async def close_all(self) -> None:
        """Close all database connections."""
        for db_manager in self.databases.values():
            await db_manager.close()
        self.databases.clear()
        self.primary_db = None
        self.failover_dbs.clear()


# Database configuration helpers
def create_postgresql_config(
    host: str = "localhost",
    port: int = 5432,
    database: str = "heygen_ai",
    username: str = "postgres",
    password: str = "",
    **kwargs
) -> DatabaseConfig:
    """Create PostgreSQL configuration."""
    url = f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{database}"
    return DatabaseConfig(
        url=url,
        type=DatabaseType.POSTGRESQL,
        **kwargs
    )


def create_mysql_config(
    host: str = "localhost",
    port: int = 3306,
    database: str = "heygen_ai",
    username: str = "root",
    password: str = "",
    **kwargs
) -> DatabaseConfig:
    """Create MySQL configuration."""
    url = f"mysql+aiomysql://{username}:{password}@{host}:{port}/{database}"
    return DatabaseConfig(
        url=url,
        type=DatabaseType.MYSQL,
        **kwargs
    )


def create_sqlite_config(
    database_path: str = "heygen_ai.db",
    **kwargs
) -> DatabaseConfig:
    """Create SQLite configuration."""
    url = f"sqlite+aiosqlite:///{database_path}"
    return DatabaseConfig(
        url=url,
        type=DatabaseType.SQLITE,
        **kwargs
    )


# Global database pool instance
db_pool = DatabaseConnectionPool()


# FastAPI dependency
async def get_db_session() -> AsyncSession:
    """FastAPI dependency for database session."""
    healthy_db = await db_pool.get_healthy_database()
    if not healthy_db:
        raise RuntimeError("No healthy database available")
    
    async with db_pool.get_session(healthy_db) as session:
        yield session


# Health check function
async def check_database_health() -> Dict[str, Any]:
    """Check health of all databases in the pool."""
    health_status = {}
    
    for name, db_manager in db_pool.databases.items():
        try:
            is_healthy = await db_manager.is_healthy()
            stats = await db_manager.get_connection_stats()
            info = await db_manager.get_database_info()
            
            health_status[name] = {
                "healthy": is_healthy,
                "stats": stats,
                "info": info
            }
        except Exception as e:
            health_status[name] = {
                "healthy": False,
                "error": str(e)
            }
    
    return health_status 