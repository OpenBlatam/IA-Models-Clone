from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

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
from sqlalchemy.pool import QueuePool, NullPool, StaticPool
from sqlalchemy import text, MetaData, event, inspect
from sqlalchemy.exc import SQLAlchemyError, OperationalError, DisconnectionError
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.sql import func
from sqlalchemy.schema import CreateTable, DropTable
from .models.sqlalchemy_models import Base, get_all_models
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Enhanced SQLAlchemy 2.0 Configuration for HeyGen AI API
Modern SQLAlchemy 2.0 setup with advanced features, connection pooling, and optimization.
"""


# SQLAlchemy 2.0 imports
    AsyncSession, 
    create_async_engine, 
    async_sessionmaker,
    AsyncEngine,
    async_scoped_session
)

# Import our models

logger = logging.getLogger(__name__)


class DatabaseType(Enum):
    """Supported database types."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"


@dataclass
class SQLAlchemyConfig:
    """SQLAlchemy 2.0 configuration."""
    url: str
    type: DatabaseType
    echo: bool = False
    echo_pool: bool = False
    
    # Connection pool settings
    pool_size: int = 20
    max_overflow: int = 30
    pool_pre_ping: bool = True
    pool_recycle: int = 3600
    pool_timeout: int = 30
    pool_reset_on_return: str = "commit"
    poolclass: type = QueuePool
    
    # Engine settings
    future: bool = True
    use_insertmanyvalues: bool = True
    use_insertmanyvalues_wo_returning: bool = True
    
    # Connection settings
    connect_args: Optional[Dict[str, Any]] = None
    
    # Performance settings
    enable_from_linting: bool = False
    enable_relationship_loaders: bool = True
    
    # Monitoring settings
    enable_metrics: bool = True
    enable_query_logging: bool = False
    
    # Migration settings
    create_tables: bool = True
    drop_tables: bool = False


class SQLAlchemyManager:
    """
    Enhanced SQLAlchemy 2.0 manager with modern features.
    Supports async operations, connection pooling, monitoring, and optimization.
    """
    
    def __init__(self, config: SQLAlchemyConfig):
        
    """__init__ function."""
self.config = config
        self.engine: Optional[AsyncEngine] = None
        self.async_session_factory: Optional[async_sessionmaker] = None
        self.scoped_session_factory: Optional[async_scoped_session] = None
        self.metadata = MetaData()
        self.is_initialized = False
        
        # Performance tracking
        self.query_count = 0
        self.total_query_time = 0.0
        self.slow_queries: List[Dict[str, Any]] = []
        
        # Health monitoring
        self.last_health_check = None
        self.health_status = True
        self.connection_errors = 0
        
        # Initialize metadata with all models
        self._setup_metadata()
    
    def _setup_metadata(self) -> None:
        """Setup metadata with all models."""
        # Import all models to register them with metadata
        for model in get_all_models():
            if not model.__table__.metadata:
                model.__table__.metadata = self.metadata
    
    async def initialize(self) -> None:
        """Initialize SQLAlchemy engine and session factory."""
        try:
            logger.info(f"Initializing SQLAlchemy 2.0 for {self.config.type.value}...")
            
            # Create async engine with optimized settings
            self.engine = create_async_engine(
                self.config.url,
                echo=self.config.echo,
                echo_pool=self.config.echo_pool,
                poolclass=self.config.poolclass,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_pre_ping=self.config.pool_pre_ping,
                pool_recycle=self.config.pool_recycle,
                pool_timeout=self.config.pool_timeout,
                pool_reset_on_return=self.config.pool_reset_on_return,
                future=self.config.future,
                use_insertmanyvalues=self.config.use_insertmanyvalues,
                use_insertmanyvalues_wo_returning=self.config.use_insertmanyvalues_wo_returning,
                connect_args=self.config.connect_args or {},
                metadata=self.metadata
            )
            
            # Create async session factory
            self.async_session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
                autocommit=False
            )
            
            # Create scoped session factory for thread safety
            self.scoped_session_factory = async_scoped_session(
                self.async_session_factory,
                scopefunc=asyncio.current_task
            )
            
            # Setup event listeners for monitoring
            if self.config.enable_metrics:
                self._setup_event_listeners()
            
            # Test connection
            await self._test_connection()
            
            # Create tables if requested
            if self.config.create_tables:
                await self._create_tables()
            
            self.is_initialized = True
            logger.info(f"✓ SQLAlchemy 2.0 initialized successfully for {self.config.type.value}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SQLAlchemy: {e}")
            raise
    
    def _setup_event_listeners(self) -> None:
        """Setup SQLAlchemy event listeners for monitoring."""
        if not self.engine:
            return
        
        @event.listens_for(self.engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany) -> Any:
            """Track query execution start."""
            conn.info.setdefault('query_start_time', []).append(datetime.now())
        
        @event.listens_for(self.engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany) -> Any:
            """Track query execution end and performance."""
            if not conn.info.get('query_start_time'):
                return
            
            start_time = conn.info['query_start_time'].pop()
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Update metrics
            self.query_count += 1
            self.total_query_time += execution_time
            
            # Log slow queries
            if execution_time > 1.0:  # Log queries taking more than 1 second
                self.slow_queries.append({
                    'statement': statement,
                    'execution_time': execution_time,
                    'timestamp': end_time.isoformat()
                })
                
                if self.config.enable_query_logging:
                    logger.warning(f"Slow query detected: {execution_time:.3f}s - {statement[:100]}...")
            
            # Keep only last 100 slow queries
            if len(self.slow_queries) > 100:
                self.slow_queries = self.slow_queries[-100:]
    
    async def _test_connection(self) -> None:
        """Test database connection."""
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                result.scalar()
            logger.info("✓ Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise
    
    async def _create_tables(self) -> None:
        """Create all tables."""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("✓ Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    async def _drop_tables(self) -> None:
        """Drop all tables (use with caution)."""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            logger.info("✓ Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise
    
    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        """Get database session with automatic cleanup."""
        if not self.is_initialized:
            raise RuntimeError("SQLAlchemy not initialized")
        
        session = self.async_session_factory()
        try:
            yield session
        except Exception as e:
            await session.rollback()
            self.connection_errors += 1
            raise
        finally:
            await session.close()
    
    @asynccontextmanager
    async def get_scoped_session(self) -> AsyncSession:
        """Get scoped database session."""
        if not self.is_initialized:
            raise RuntimeError("SQLAlchemy not initialized")
        
        session = self.scoped_session_factory()
        try:
            yield session
        except Exception as e:
            await session.rollback()
            self.connection_errors += 1
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
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        try:
            start_time = datetime.now()
            
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.last_health_check = datetime.now()
            self.health_status = True
            
            return {
                "status": "healthy",
                "response_time": execution_time,
                "last_check": self.last_health_check.isoformat(),
                "connection_errors": self.connection_errors
            }
            
        except Exception as e:
            self.health_status = False
            self.last_health_check = datetime.now()
            self.connection_errors += 1
            
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": self.last_health_check.isoformat(),
                "connection_errors": self.connection_errors
            }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get SQLAlchemy performance metrics."""
        if not self.engine:
            return {}
        
        pool = self.engine.pool
        
        return {
            "query_count": self.query_count,
            "total_query_time": self.total_query_time,
            "average_query_time": self.total_query_time / max(self.query_count, 1),
            "slow_queries_count": len(self.slow_queries),
            "connection_pool": {
                "size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "invalid": pool.invalid()
            },
            "health_status": self.health_status,
            "connection_errors": self.connection_errors,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None
        }
    
    async def get_table_info(self) -> Dict[str, Any]:
        """Get database table information."""
        try:
            async with self.get_session() as session:
                # Get table count
                result = await session.execute(text("""
                    SELECT COUNT(*) as table_count 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """))
                table_count = result.scalar()
                
                # Get table names and row counts
                tables_info = []
                for model in get_all_models():
                    table_name = model.__tablename__
                    try:
                        result = await session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                        row_count = result.scalar()
                        tables_info.append({
                            "name": table_name,
                            "row_count": row_count
                        })
                    except Exception:
                        tables_info.append({
                            "name": table_name,
                            "row_count": 0
                        })
                
                return {
                    "table_count": table_count,
                    "tables": tables_info
                }
                
        except Exception as e:
            logger.error(f"Failed to get table info: {e}")
            return {"error": str(e)}
    
    async def close(self) -> None:
        """Close SQLAlchemy engine and cleanup."""
        try:
            if self.engine:
                await self.engine.dispose()
            self.is_initialized = False
            logger.info("✓ SQLAlchemy engine closed")
        except Exception as e:
            logger.error(f"Error closing SQLAlchemy engine: {e}")
            raise


# Configuration helpers for different database types
def create_postgresql_config(
    host: str = "localhost",
    port: int = 5432,
    database: str = "heygen_ai",
    username: str = "postgres",
    password: str = "",
    **kwargs
) -> SQLAlchemyConfig:
    """Create PostgreSQL configuration."""
    url = f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{database}"
    return SQLAlchemyConfig(
        url=url,
        type=DatabaseType.POSTGRESQL,
        connect_args={
            "server_settings": {
                "application_name": "heygen_ai",
                "timezone": "UTC"
            },
            "command_timeout": 60,
            "statement_timeout": 30000
        },
        **kwargs
    )


def create_mysql_config(
    host: str = "localhost",
    port: int = 3306,
    database: str = "heygen_ai",
    username: str = "root",
    password: str = "",
    **kwargs
) -> SQLAlchemyConfig:
    """Create MySQL configuration."""
    url = f"mysql+aiomysql://{username}:{password}@{host}:{port}/{database}"
    return SQLAlchemyConfig(
        url=url,
        type=DatabaseType.MYSQL,
        connect_args={
            "charset": "utf8mb4",
            "autocommit": False,
            "sql_mode": "STRICT_TRANS_TABLES,NO_ZERO_DATE,NO_ZERO_IN_DATE,ERROR_FOR_DIVISION_BY_ZERO"
        },
        **kwargs
    )


def create_sqlite_config(
    database_path: str = "heygen_ai.db",
    **kwargs
) -> SQLAlchemyConfig:
    """Create SQLite configuration."""
    url = f"sqlite+aiosqlite:///{database_path}"
    return SQLAlchemyConfig(
        url=url,
        type=DatabaseType.SQLITE,
        connect_args={
            "timeout": 30,
            "check_same_thread": False,
            "isolation_level": None
        },
        **kwargs
    )


# Global SQLAlchemy manager instance
sqlalchemy_manager: Optional[SQLAlchemyManager] = None


# FastAPI dependency
async def get_sqlalchemy_session() -> AsyncSession:
    """FastAPI dependency for SQLAlchemy session."""
    if not sqlalchemy_manager or not sqlalchemy_manager.is_initialized:
        raise RuntimeError("SQLAlchemy not initialized")
    
    async with sqlalchemy_manager.get_session() as session:
        yield session


# Health check function
async def check_sqlalchemy_health() -> Dict[str, Any]:
    """Check SQLAlchemy health."""
    if not sqlalchemy_manager:
        return {"status": "unhealthy", "error": "SQLAlchemy not initialized"}
    
    return await sqlalchemy_manager.health_check()


# Metrics function
async def get_sqlalchemy_metrics() -> Dict[str, Any]:
    """Get SQLAlchemy metrics."""
    if not sqlalchemy_manager:
        return {}
    
    return await sqlalchemy_manager.get_metrics()


# Initialization function
async def initialize_sqlalchemy(config: SQLAlchemyConfig) -> SQLAlchemyManager:
    """Initialize SQLAlchemy manager."""
    global sqlalchemy_manager
    
    sqlalchemy_manager = SQLAlchemyManager(config)
    await sqlalchemy_manager.initialize()
    
    return sqlalchemy_manager


# Cleanup function
async def cleanup_sqlalchemy() -> None:
    """Cleanup SQLAlchemy manager."""
    global sqlalchemy_manager
    
    if sqlalchemy_manager:
        await sqlalchemy_manager.close()
        sqlalchemy_manager = None 