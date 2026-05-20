"""
Database Service
================

Advanced database service with connection pooling, migrations, and monitoring.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncpg
import aiomysql
import aioredis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text, MetaData, Table, Column, Integer, String, DateTime, Boolean, Text
from sqlalchemy.pool import QueuePool, StaticPool
import alembic
from alembic import command
from alembic.config import Config

from ..utils.helpers import DateTimeHelpers
from ..utils.decorators import log_execution, measure_performance, retry


logger = logging.getLogger(__name__)


class DatabaseType(str, Enum):
    """Database type enumeration"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    REDIS = "redis"


class ConnectionStatus(str, Enum):
    """Connection status enumeration"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    database_type: DatabaseType = DatabaseType.POSTGRESQL
    host: str = "localhost"
    port: int = 5432
    database: str = "workflow_chain"
    username: str = "postgres"
    password: str = "password"
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    ssl_mode: str = "prefer"
    charset: str = "utf8mb4"
    connect_timeout: int = 10
    command_timeout: int = 30


@dataclass
class DatabaseMetrics:
    """Database metrics representation"""
    active_connections: int = 0
    total_connections: int = 0
    connection_errors: int = 0
    query_count: int = 0
    slow_queries: int = 0
    avg_query_time: float = 0.0
    last_health_check: Optional[datetime] = None
    status: ConnectionStatus = ConnectionStatus.DISCONNECTED


class DatabaseService:
    """Advanced database service with connection pooling and monitoring"""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.engine = None
        self.session_factory = None
        self.metrics = DatabaseMetrics()
        self._is_running = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._connection_pool = None
        self._redis_pool = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize database engine"""
        try:
            if self.config.database_type == DatabaseType.POSTGRESQL:
                database_url = f"postgresql+asyncpg://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
            elif self.config.database_type == DatabaseType.MYSQL:
                database_url = f"mysql+aiomysql://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}?charset={self.config.charset}"
            elif self.config.database_type == DatabaseType.SQLITE:
                database_url = f"sqlite+aiosqlite:///{self.config.database}.db"
            else:
                raise ValueError(f"Unsupported database type: {self.config.database_type}")
            
            # Create engine with connection pooling
            self.engine = create_async_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=self.config.echo,
                connect_args={
                    "connect_timeout": self.config.connect_timeout,
                    "command_timeout": self.config.command_timeout
                } if self.config.database_type == DatabaseType.POSTGRESQL else {}
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            logger.info(f"Database engine initialized for {self.config.database_type.value}")
        
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise
    
    async def start(self):
        """Start the database service"""
        if self._is_running:
            return
        
        try:
            # Test connection
            await self.test_connection()
            
            # Initialize Redis if needed
            if self.config.database_type == DatabaseType.REDIS:
                await self._initialize_redis()
            
            self._is_running = True
            self.metrics.status = ConnectionStatus.CONNECTED
            
            # Start health check task
            self._health_check_task = asyncio.create_task(self._health_check_worker())
            
            logger.info("Database service started successfully")
        
        except Exception as e:
            logger.error(f"Failed to start database service: {e}")
            self.metrics.status = ConnectionStatus.ERROR
            raise
    
    async def stop(self):
        """Stop the database service"""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # Stop health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close Redis connection
        if self._redis_pool:
            await self._redis_pool.close()
        
        # Close database engine
        if self.engine:
            await self.engine.dispose()
        
        self.metrics.status = ConnectionStatus.DISCONNECTED
        logger.info("Database service stopped")
    
    async def test_connection(self) -> bool:
        """Test database connection"""
        try:
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            self.metrics.status = ConnectionStatus.CONNECTED
            return True
        
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            self.metrics.status = ConnectionStatus.ERROR
            self.metrics.connection_errors += 1
            return False
    
    async def _initialize_redis(self):
        """Initialize Redis connection pool"""
        try:
            self._redis_pool = aioredis.ConnectionPool.from_url(
                f"redis://{self.config.host}:{self.config.port}",
                max_connections=self.config.pool_size,
                retry_on_timeout=True
            )
            
            # Test Redis connection
            redis = aioredis.Redis(connection_pool=self._redis_pool)
            await redis.ping()
            await redis.close()
            
            logger.info("Redis connection pool initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise
    
    async def _health_check_worker(self):
        """Health check worker"""
        while self._is_running:
            try:
                await self.test_connection()
                self.metrics.last_health_check = DateTimeHelpers.now_utc()
                await asyncio.sleep(30)  # Check every 30 seconds
            
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                await asyncio.sleep(10)  # Retry in 10 seconds
    
    @measure_performance
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a raw SQL query"""
        try:
            async with self.session_factory() as session:
                result = await session.execute(text(query), parameters or {})
                
                if result.returns_rows:
                    rows = result.fetchall()
                    columns = result.keys()
                    return [dict(zip(columns, row)) for row in rows]
                else:
                    await session.commit()
                    return []
        
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    @measure_performance
    async def execute_transaction(self, queries: List[Tuple[str, Optional[Dict[str, Any]]]]) -> List[Any]:
        """Execute multiple queries in a transaction"""
        try:
            async with self.session_factory() as session:
                results = []
                
                for query, parameters in queries:
                    result = await session.execute(text(query), parameters or {})
                    
                    if result.returns_rows:
                        rows = result.fetchall()
                        columns = result.keys()
                        results.append([dict(zip(columns, row)) for row in rows])
                    else:
                        results.append(None)
                
                await session.commit()
                return results
        
        except Exception as e:
            logger.error(f"Transaction execution failed: {e}")
            raise
    
    async def get_session(self) -> AsyncSession:
        """Get database session"""
        return self.session_factory()
    
    async def get_redis(self) -> aioredis.Redis:
        """Get Redis connection"""
        if not self._redis_pool:
            raise RuntimeError("Redis not initialized")
        
        return aioredis.Redis(connection_pool=self._redis_pool)
    
    async def run_migrations(self, revision: str = "head") -> bool:
        """Run database migrations"""
        try:
            # Configure Alembic
            alembic_cfg = Config("alembic.ini")
            alembic_cfg.set_main_option("sqlalchemy.url", str(self.engine.url))
            
            # Run migrations
            command.upgrade(alembic_cfg, revision)
            
            logger.info(f"Database migrations completed up to {revision}")
            return True
        
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    async def create_migration(self, message: str) -> bool:
        """Create new migration"""
        try:
            # Configure Alembic
            alembic_cfg = Config("alembic.ini")
            alembic_cfg.set_main_option("sqlalchemy.url", str(self.engine.url))
            
            # Create migration
            command.revision(alembic_cfg, message=message, autogenerate=True)
            
            logger.info(f"Migration created: {message}")
            return True
        
        except Exception as e:
            logger.error(f"Migration creation failed: {e}")
            return False
    
    async def backup_database(self, backup_path: str) -> bool:
        """Backup database"""
        try:
            if self.config.database_type == DatabaseType.POSTGRESQL:
                import subprocess
                
                # Create backup using pg_dump
                cmd = [
                    "pg_dump",
                    f"--host={self.config.host}",
                    f"--port={self.config.port}",
                    f"--username={self.config.username}",
                    f"--dbname={self.config.database}",
                    f"--file={backup_path}",
                    "--verbose",
                    "--no-password"
                ]
                
                # Set password via environment variable
                import os
                os.environ["PGPASSWORD"] = self.config.password
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"Database backup created: {backup_path}")
                    return True
                else:
                    logger.error(f"Backup failed: {result.stderr}")
                    return False
            
            else:
                logger.warning(f"Backup not implemented for {self.config.database_type.value}")
                return False
        
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
    
    async def restore_database(self, backup_path: str) -> bool:
        """Restore database from backup"""
        try:
            if self.config.database_type == DatabaseType.POSTGRESQL:
                import subprocess
                
                # Restore backup using psql
                cmd = [
                    "psql",
                    f"--host={self.config.host}",
                    f"--port={self.config.port}",
                    f"--username={self.config.username}",
                    f"--dbname={self.config.database}",
                    f"--file={backup_path}",
                    "--verbose",
                    "--no-password"
                ]
                
                # Set password via environment variable
                import os
                os.environ["PGPASSWORD"] = self.config.password
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"Database restored from: {backup_path}")
                    return True
                else:
                    logger.error(f"Restore failed: {result.stderr}")
                    return False
            
            else:
                logger.warning(f"Restore not implemented for {self.config.database_type.value}")
                return False
        
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get database metrics"""
        return {
            "database_type": self.config.database_type.value,
            "host": self.config.host,
            "port": self.config.port,
            "database": self.config.database,
            "active_connections": self.metrics.active_connections,
            "total_connections": self.metrics.total_connections,
            "connection_errors": self.metrics.connection_errors,
            "query_count": self.metrics.query_count,
            "slow_queries": self.metrics.slow_queries,
            "avg_query_time": self.metrics.avg_query_time,
            "last_health_check": self.metrics.last_health_check.isoformat() if self.metrics.last_health_check else None,
            "status": self.metrics.status.value,
            "pool_size": self.config.pool_size,
            "max_overflow": self.config.max_overflow,
            "timestamp": DateTimeHelpers.now_utc().isoformat()
        }
    
    async def optimize_database(self) -> Dict[str, Any]:
        """Optimize database performance"""
        try:
            optimization_results = {}
            
            if self.config.database_type == DatabaseType.POSTGRESQL:
                # Analyze tables
                await self.execute_query("ANALYZE")
                optimization_results["analyze"] = "completed"
                
                # Vacuum if needed
                await self.execute_query("VACUUM ANALYZE")
                optimization_results["vacuum"] = "completed"
                
                # Update statistics
                await self.execute_query("UPDATE pg_stat_user_tables SET n_tup_ins = 0, n_tup_upd = 0, n_tup_del = 0")
                optimization_results["statistics"] = "updated"
            
            elif self.config.database_type == DatabaseType.MYSQL:
                # Analyze tables
                await self.execute_query("ANALYZE TABLE")
                optimization_results["analyze"] = "completed"
                
                # Optimize tables
                await self.execute_query("OPTIMIZE TABLE")
                optimization_results["optimize"] = "completed"
            
            logger.info("Database optimization completed")
            return optimization_results
        
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            return {"error": str(e)}
    
    async def get_database_info(self) -> Dict[str, Any]:
        """Get database information"""
        try:
            if self.config.database_type == DatabaseType.POSTGRESQL:
                # Get PostgreSQL version
                version_result = await self.execute_query("SELECT version()")
                version = version_result[0]["version"] if version_result else "unknown"
                
                # Get database size
                size_result = await self.execute_query(
                    "SELECT pg_size_pretty(pg_database_size(current_database())) as size"
                )
                size = size_result[0]["size"] if size_result else "unknown"
                
                # Get table count
                table_count_result = await self.execute_query(
                    "SELECT COUNT(*) as count FROM information_schema.tables WHERE table_schema = 'public'"
                )
                table_count = table_count_result[0]["count"] if table_count_result else 0
                
                return {
                    "version": version,
                    "size": size,
                    "table_count": table_count,
                    "database_type": "PostgreSQL"
                }
            
            elif self.config.database_type == DatabaseType.MYSQL:
                # Get MySQL version
                version_result = await self.execute_query("SELECT VERSION() as version")
                version = version_result[0]["version"] if version_result else "unknown"
                
                # Get database size
                size_result = await self.execute_query(
                    "SELECT ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) AS size_mb FROM information_schema.tables WHERE table_schema = DATABASE()"
                )
                size = f"{size_result[0]['size_mb']} MB" if size_result else "unknown"
                
                # Get table count
                table_count_result = await self.execute_query(
                    "SELECT COUNT(*) as count FROM information_schema.tables WHERE table_schema = DATABASE()"
                )
                table_count = table_count_result[0]["count"] if table_count_result else 0
                
                return {
                    "version": version,
                    "size": size,
                    "table_count": table_count,
                    "database_type": "MySQL"
                }
            
            else:
                return {
                    "database_type": self.config.database_type.value,
                    "version": "unknown",
                    "size": "unknown",
                    "table_count": 0
                }
        
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {"error": str(e)}


# Global database service
database_service = DatabaseService()


# Utility functions
async def start_database_service():
    """Start the database service"""
    await database_service.start()


async def stop_database_service():
    """Stop the database service"""
    await database_service.stop()


async def get_database_session() -> AsyncSession:
    """Get database session"""
    return await database_service.get_session()


async def get_redis_connection() -> aioredis.Redis:
    """Get Redis connection"""
    return await database_service.get_redis()


async def execute_query(query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Execute a raw SQL query"""
    return await database_service.execute_query(query, parameters)


async def execute_transaction(queries: List[Tuple[str, Optional[Dict[str, Any]]]]) -> List[Any]:
    """Execute multiple queries in a transaction"""
    return await database_service.execute_transaction(queries)


async def run_migrations(revision: str = "head") -> bool:
    """Run database migrations"""
    return await database_service.run_migrations(revision)


async def create_migration(message: str) -> bool:
    """Create new migration"""
    return await database_service.create_migration(message)


async def backup_database(backup_path: str) -> bool:
    """Backup database"""
    return await database_service.backup_database(backup_path)


async def restore_database(backup_path: str) -> bool:
    """Restore database from backup"""
    return await database_service.restore_database(backup_path)


def get_database_metrics() -> Dict[str, Any]:
    """Get database metrics"""
    return database_service.get_metrics()


async def optimize_database() -> Dict[str, Any]:
    """Optimize database performance"""
    return await database_service.optimize_database()


async def get_database_info() -> Dict[str, Any]:
    """Get database information"""
    return await database_service.get_database_info()


# Database decorators
def with_database_session(func):
    """Decorator to provide database session"""
    async def wrapper(*args, **kwargs):
        async with database_service.get_session() as session:
            return await func(session, *args, **kwargs)
    return wrapper


def with_redis_connection(func):
    """Decorator to provide Redis connection"""
    async def wrapper(*args, **kwargs):
        redis = await database_service.get_redis()
        try:
            return await func(redis, *args, **kwargs)
        finally:
            await redis.close()
    return wrapper




