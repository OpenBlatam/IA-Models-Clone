"""
Async Database Operations for Video-OpusClip

Dedicated async functions for database operations with connection pooling,
query optimization, and transaction management.
"""

import asyncio
import asyncpg
import aiomysql
import aiosqlite
import aioredis
import time
import json
from typing import (
    List, Dict, Any, Optional, Union, Tuple, AsyncIterator, 
    Callable, Awaitable, TypeVar, Generic
)
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import structlog
from datetime import datetime, timedelta

logger = structlog.get_logger()

# Type variables
T = TypeVar('T')
R = TypeVar('R')

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DatabaseConfig:
    """Configuration for database connections."""
    host: str = "localhost"
    port: int = 5432
    database: str = "video_opusclip"
    username: str = "postgres"
    password: str = ""
    min_connections: int = 5
    max_connections: int = 20
    timeout: float = 30.0
    command_timeout: float = 60.0
    pool_timeout: float = 30.0
    enable_ssl: bool = False
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    ssl_ca: Optional[str] = None

class DatabaseType(Enum):
    """Supported database types."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    REDIS = "redis"

class QueryType(Enum):
    """Query types for optimization."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"

# =============================================================================
# ASYNC DATABASE CONNECTION POOLS
# =============================================================================

class AsyncDatabasePool:
    """Generic async database connection pool."""
    
    def __init__(self, config: DatabaseConfig, db_type: DatabaseType):
        self.config = config
        self.db_type = db_type
        self.pool = None
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the connection pool."""
        raise NotImplementedError
    
    async def close(self):
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
    
    @asynccontextmanager
    async def get_connection(self):
        """Get connection from pool."""
        if not self.pool:
            await self.initialize()
        
        async with self.pool.acquire() as connection:
            yield connection

class PostgreSQLPool(AsyncDatabasePool):
    """PostgreSQL connection pool using asyncpg."""
    
    async def initialize(self):
        """Initialize PostgreSQL connection pool."""
        self.pool = await asyncpg.create_pool(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.username,
            password=self.config.password,
            min_size=self.config.min_connections,
            max_size=self.config.max_connections,
            command_timeout=self.config.command_timeout,
            timeout=self.config.timeout,
            ssl=self.config.enable_ssl
        )
        logger.info("PostgreSQL connection pool initialized")

class MySQLPool(AsyncDatabasePool):
    """MySQL connection pool using aiomysql."""
    
    async def initialize(self):
        """Initialize MySQL connection pool."""
        self.pool = await aiomysql.create_pool(
            host=self.config.host,
            port=self.config.port,
            db=self.config.database,
            user=self.config.username,
            password=self.config.password,
            minsize=self.config.min_connections,
            maxsize=self.config.max_connections,
            autocommit=True,
            ssl=self.config.enable_ssl
        )
        logger.info("MySQL connection pool initialized")

class SQLitePool(AsyncDatabasePool):
    """SQLite connection pool using aiosqlite."""
    
    async def initialize(self):
        """Initialize SQLite connection pool."""
        # SQLite doesn't have traditional connection pooling
        # We'll create a simple connection manager
        self.pool = self.config.database
        logger.info("SQLite connection initialized")

class RedisPool(AsyncDatabasePool):
    """Redis connection pool using aioredis."""
    
    async def initialize(self):
        """Initialize Redis connection pool."""
        self.pool = aioredis.from_url(
            f"redis://{self.config.host}:{self.config.port}",
            password=self.config.password,
            encoding="utf-8",
            decode_responses=True,
            max_connections=self.config.max_connections
        )
        logger.info("Redis connection pool initialized")

# =============================================================================
# ASYNC DATABASE OPERATIONS
# =============================================================================

class AsyncDatabaseOperations:
    """Dedicated async database operations."""
    
    def __init__(self, pool: AsyncDatabasePool):
        self.pool = pool
        self.query_cache = {}
        self.metrics = {
            "queries_executed": 0,
            "total_execution_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    async def execute_query(
        self, 
        query: str, 
        params: Optional[Union[List, Dict]] = None,
        query_type: QueryType = QueryType.SELECT,
        cache_key: Optional[str] = None,
        cache_ttl: int = 300
    ) -> Any:
        """Execute a database query with caching and metrics."""
        start_time = time.perf_counter()
        
        # Check cache for SELECT queries
        if query_type == QueryType.SELECT and cache_key:
            cached_result = await self._get_cached_result(cache_key)
            if cached_result is not None:
                self.metrics["cache_hits"] += 1
                return cached_result
        
        self.metrics["cache_misses"] += 1
        
        try:
            async with self.pool.get_connection() as connection:
                if isinstance(self.pool, PostgreSQLPool):
                    result = await self._execute_postgresql(connection, query, params, query_type)
                elif isinstance(self.pool, MySQLPool):
                    result = await self._execute_mysql(connection, query, params, query_type)
                elif isinstance(self.pool, SQLitePool):
                    result = await self._execute_sqlite(connection, query, params, query_type)
                elif isinstance(self.pool, RedisPool):
                    result = await self._execute_redis(connection, query, params, query_type)
                else:
                    raise ValueError(f"Unsupported database type: {type(self.pool)}")
                
                # Cache result for SELECT queries
                if query_type == QueryType.SELECT and cache_key:
                    await self._cache_result(cache_key, result, cache_ttl)
                
                # Update metrics
                execution_time = time.perf_counter() - start_time
                self.metrics["queries_executed"] += 1
                self.metrics["total_execution_time"] += execution_time
                
                logger.debug(
                    "Database query executed",
                    query_type=query_type.value,
                    execution_time=f"{execution_time:.3f}s",
                    cache_key=cache_key
                )
                
                return result
                
        except Exception as e:
            logger.error(
                "Database query failed",
                query=query,
                params=params,
                error=str(e)
            )
            raise
    
    async def _execute_postgresql(self, connection, query: str, params, query_type: QueryType):
        """Execute PostgreSQL query."""
        if query_type == QueryType.SELECT:
            if params:
                return await connection.fetch(query, *params)
            else:
                return await connection.fetch(query)
        elif query_type == QueryType.INSERT:
            if params:
                return await connection.fetchval(query, *params)
            else:
                return await connection.fetchval(query)
        elif query_type == QueryType.UPDATE:
            if params:
                return await connection.execute(query, *params)
            else:
                return await connection.execute(query)
        elif query_type == QueryType.DELETE:
            if params:
                return await connection.execute(query, *params)
            else:
                return await connection.execute(query)
    
    async def _execute_mysql(self, connection, query: str, params, query_type: QueryType):
        """Execute MySQL query."""
        async with connection.cursor() as cursor:
            if params:
                await cursor.execute(query, params)
            else:
                await cursor.execute(query)
            
            if query_type == QueryType.SELECT:
                return await cursor.fetchall()
            else:
                await connection.commit()
                return cursor.rowcount
    
    async def _execute_sqlite(self, connection, query: str, params, query_type: QueryType):
        """Execute SQLite query."""
        if params:
            cursor = await connection.execute(query, params)
        else:
            cursor = await connection.execute(query)
        
        if query_type == QueryType.SELECT:
            return await cursor.fetchall()
        else:
            await connection.commit()
            return cursor.rowcount
    
    async def _execute_redis(self, connection, query: str, params, query_type: QueryType):
        """Execute Redis command."""
        if query_type == QueryType.SELECT:
            return await connection.get(params)
        elif query_type == QueryType.INSERT:
            return await connection.set(params[0], params[1])
        elif query_type == QueryType.UPDATE:
            return await connection.set(params[0], params[1])
        elif query_type == QueryType.DELETE:
            return await connection.delete(params)
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result from Redis."""
        if isinstance(self.pool, RedisPool):
            try:
                cached = await self.pool.pool.get(f"cache:{cache_key}")
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Cache retrieval failed: {e}")
        return None
    
    async def _cache_result(self, cache_key: str, result: Any, ttl: int):
        """Cache result in Redis."""
        if isinstance(self.pool, RedisPool):
            try:
                await self.pool.pool.setex(
                    f"cache:{cache_key}",
                    ttl,
                    json.dumps(result, default=str)
                )
            except Exception as e:
                logger.warning(f"Cache storage failed: {e}")

# =============================================================================
# VIDEO-SPECIFIC DATABASE OPERATIONS
# =============================================================================

class AsyncVideoDatabase:
    """Dedicated async database operations for video processing."""
    
    def __init__(self, db_operations: AsyncDatabaseOperations):
        self.db = db_operations
    
    async def create_video_record(self, video_data: Dict[str, Any]) -> int:
        """Create a new video record."""
        query = """
        INSERT INTO videos (url, title, duration, status, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id
        """
        params = [
            video_data.get("url"),
            video_data.get("title", ""),
            video_data.get("duration", 0),
            video_data.get("status", "pending"),
            datetime.now(),
            datetime.now()
        ]
        
        result = await self.db.execute_query(
            query, params, QueryType.INSERT
        )
        return result
    
    async def get_video_by_id(self, video_id: int) -> Optional[Dict[str, Any]]:
        """Get video by ID with caching."""
        query = "SELECT * FROM videos WHERE id = $1"
        cache_key = f"video:{video_id}"
        
        result = await self.db.execute_query(
            query, [video_id], QueryType.SELECT, cache_key, 300
        )
        
        if result:
            return dict(result[0]) if isinstance(result, list) else dict(result)
        return None
    
    async def get_videos_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Get videos by status."""
        query = "SELECT * FROM videos WHERE status = $1 ORDER BY created_at DESC"
        cache_key = f"videos:status:{status}"
        
        result = await self.db.execute_query(
            query, [status], QueryType.SELECT, cache_key, 60
        )
        
        return [dict(row) for row in result]
    
    async def update_video_status(self, video_id: int, status: str) -> bool:
        """Update video status."""
        query = """
        UPDATE videos 
        SET status = $1, updated_at = $2 
        WHERE id = $3
        """
        params = [status, datetime.now(), video_id]
        
        result = await self.db.execute_query(
            query, params, QueryType.UPDATE
        )
        
        # Invalidate cache
        await self._invalidate_video_cache(video_id)
        
        return result > 0
    
    async def create_clip_record(self, clip_data: Dict[str, Any]) -> int:
        """Create a new clip record."""
        query = """
        INSERT INTO clips (video_id, start_time, end_time, duration, 
                          caption, effects, file_path, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING id
        """
        params = [
            clip_data.get("video_id"),
            clip_data.get("start_time", 0),
            clip_data.get("end_time", 0),
            clip_data.get("duration", 0),
            clip_data.get("caption", ""),
            json.dumps(clip_data.get("effects", [])),
            clip_data.get("file_path", ""),
            datetime.now()
        ]
        
        result = await self.db.execute_query(
            query, params, QueryType.INSERT
        )
        return result
    
    async def get_clips_by_video_id(self, video_id: int) -> List[Dict[str, Any]]:
        """Get clips by video ID."""
        query = "SELECT * FROM clips WHERE video_id = $1 ORDER BY start_time"
        cache_key = f"clips:video:{video_id}"
        
        result = await self.db.execute_query(
            query, [video_id], QueryType.SELECT, cache_key, 300
        )
        
        clips = []
        for row in result:
            clip = dict(row)
            # Parse effects JSON
            if clip.get("effects"):
                try:
                    clip["effects"] = json.loads(clip["effects"])
                except:
                    clip["effects"] = []
            clips.append(clip)
        
        return clips
    
    async def create_processing_job(self, job_data: Dict[str, Any]) -> int:
        """Create a new processing job."""
        query = """
        INSERT INTO processing_jobs (video_id, job_type, status, 
                                    priority, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id
        """
        params = [
            job_data.get("video_id"),
            job_data.get("job_type", "video_processing"),
            job_data.get("status", "pending"),
            job_data.get("priority", 1),
            datetime.now(),
            datetime.now()
        ]
        
        result = await self.db.execute_query(
            query, params, QueryType.INSERT
        )
        return result
    
    async def get_pending_jobs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get pending processing jobs."""
        query = """
        SELECT * FROM processing_jobs 
        WHERE status = 'pending' 
        ORDER BY priority DESC, created_at ASC 
        LIMIT $1
        """
        
        result = await self.db.execute_query(
            query, [limit], QueryType.SELECT
        )
        
        return [dict(row) for row in result]
    
    async def update_job_status(self, job_id: int, status: str, result: Optional[Dict] = None) -> bool:
        """Update job status."""
        query = """
        UPDATE processing_jobs 
        SET status = $1, result = $2, updated_at = $3 
        WHERE id = $4
        """
        params = [
            status,
            json.dumps(result) if result else None,
            datetime.now(),
            job_id
        ]
        
        result = await self.db.execute_query(
            query, params, QueryType.UPDATE
        )
        
        return result > 0
    
    async def _invalidate_video_cache(self, video_id: int):
        """Invalidate video-related cache entries."""
        if isinstance(self.db.pool, RedisPool):
            try:
                # Delete video cache
                await self.db.pool.pool.delete(f"cache:video:{video_id}")
                # Delete clips cache
                await self.db.pool.pool.delete(f"cache:clips:video:{video_id}")
                # Delete status-based video lists
                await self.db.pool.pool.delete("cache:videos:status:pending")
                await self.db.pool.pool.delete("cache:videos:status:processing")
                await self.db.pool.pool.delete("cache:videos:status:completed")
            except Exception as e:
                logger.warning(f"Cache invalidation failed: {e}")

# =============================================================================
# BATCH DATABASE OPERATIONS
# =============================================================================

class AsyncBatchDatabaseOperations:
    """Batch database operations for improved performance."""
    
    def __init__(self, db_operations: AsyncDatabaseOperations):
        self.db = db_operations
    
    async def batch_insert_videos(self, videos: List[Dict[str, Any]]) -> List[int]:
        """Batch insert multiple videos."""
        if not videos:
            return []
        
        query = """
        INSERT INTO videos (url, title, duration, status, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id
        """
        
        async with self.db.pool.get_connection() as connection:
            if isinstance(self.db.pool, PostgreSQLPool):
                # Use PostgreSQL's batch insert
                values = []
                for video in videos:
                    values.append((
                        video.get("url"),
                        video.get("title", ""),
                        video.get("duration", 0),
                        video.get("status", "pending"),
                        datetime.now(),
                        datetime.now()
                    ))
                
                result = await connection.fetch(query, *values)
                return [row['id'] for row in result]
            
            else:
                # Fallback to individual inserts
                ids = []
                for video in videos:
                    video_id = await self.db.execute_query(
                        query, list(video.values()), QueryType.INSERT
                    )
                    ids.append(video_id)
                return ids
    
    async def batch_insert_clips(self, clips: List[Dict[str, Any]]) -> List[int]:
        """Batch insert multiple clips."""
        if not clips:
            return []
        
        query = """
        INSERT INTO clips (video_id, start_time, end_time, duration, 
                          caption, effects, file_path, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING id
        """
        
        async with self.db.pool.get_connection() as connection:
            if isinstance(self.db.pool, PostgreSQLPool):
                values = []
                for clip in clips:
                    values.append((
                        clip.get("video_id"),
                        clip.get("start_time", 0),
                        clip.get("end_time", 0),
                        clip.get("duration", 0),
                        clip.get("caption", ""),
                        json.dumps(clip.get("effects", [])),
                        clip.get("file_path", ""),
                        datetime.now()
                    ))
                
                result = await connection.fetch(query, *values)
                return [row['id'] for row in result]
            
            else:
                ids = []
                for clip in clips:
                    clip_id = await self.db.execute_query(
                        query, list(clip.values()), QueryType.INSERT
                    )
                    ids.append(clip_id)
                return ids
    
    async def batch_update_video_status(self, updates: List[Tuple[int, str]]) -> int:
        """Batch update video statuses."""
        if not updates:
            return 0
        
        # Use a single query with CASE statement for PostgreSQL
        if isinstance(self.db.pool, PostgreSQLPool):
            cases = []
            video_ids = []
            for video_id, status in updates:
                cases.append(f"WHEN {video_id} THEN '{status}'")
                video_ids.append(video_id)
            
            query = f"""
            UPDATE videos 
            SET status = CASE id {' '.join(cases)} END,
                updated_at = $1
            WHERE id = ANY($2)
            """
            
            result = await self.db.execute_query(
                query, [datetime.now(), video_ids], QueryType.UPDATE
            )
            return result
        
        else:
            # Fallback to individual updates
            updated_count = 0
            for video_id, status in updates:
                success = await self.db.execute_query(
                    "UPDATE videos SET status = $1, updated_at = $2 WHERE id = $3",
                    [status, datetime.now(), video_id],
                    QueryType.UPDATE
                )
                if success:
                    updated_count += 1
            return updated_count

# =============================================================================
# TRANSACTION MANAGEMENT
# =============================================================================

class AsyncTransactionManager:
    """Async transaction management."""
    
    def __init__(self, db_operations: AsyncDatabaseOperations):
        self.db = db_operations
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions."""
        async with self.db.pool.get_connection() as connection:
            if isinstance(self.db.pool, PostgreSQLPool):
                async with connection.transaction():
                    yield connection
            elif isinstance(self.db.pool, MySQLPool):
                await connection.begin()
                try:
                    yield connection
                    await connection.commit()
                except:
                    await connection.rollback()
                    raise
            elif isinstance(self.db.pool, SQLitePool):
                await connection.execute("BEGIN")
                try:
                    yield connection
                    await connection.commit()
                except:
                    await connection.rollback()
                    raise
    
    async def execute_in_transaction(
        self, 
        operations: List[Callable[[Any], Awaitable[Any]]]
    ) -> List[Any]:
        """Execute multiple operations in a single transaction."""
        async with self.transaction() as connection:
            results = []
            for operation in operations:
                result = await operation(connection)
                results.append(result)
            return results

# =============================================================================
# DATABASE MIGRATION AND SETUP
# =============================================================================

class AsyncDatabaseSetup:
    """Database setup and migration utilities."""
    
    def __init__(self, db_operations: AsyncDatabaseOperations):
        self.db = db_operations
    
    async def create_tables(self):
        """Create database tables if they don't exist."""
        tables = [
            """
            CREATE TABLE IF NOT EXISTS videos (
                id SERIAL PRIMARY KEY,
                url VARCHAR(500) NOT NULL,
                title VARCHAR(255),
                duration INTEGER DEFAULT 0,
                status VARCHAR(50) DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS clips (
                id SERIAL PRIMARY KEY,
                video_id INTEGER REFERENCES videos(id),
                start_time INTEGER DEFAULT 0,
                end_time INTEGER DEFAULT 0,
                duration INTEGER DEFAULT 0,
                caption TEXT,
                effects JSONB,
                file_path VARCHAR(500),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS processing_jobs (
                id SERIAL PRIMARY KEY,
                video_id INTEGER REFERENCES videos(id),
                job_type VARCHAR(100) DEFAULT 'video_processing',
                status VARCHAR(50) DEFAULT 'pending',
                priority INTEGER DEFAULT 1,
                result JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
            CREATE INDEX IF NOT EXISTS idx_videos_created_at ON videos(created_at);
            CREATE INDEX IF NOT EXISTS idx_clips_video_id ON clips(video_id);
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON processing_jobs(status);
            CREATE INDEX IF NOT EXISTS idx_jobs_priority ON processing_jobs(priority);
            """
        ]
        
        for table_sql in tables:
            await self.db.execute_query(table_sql, query_type=QueryType.EXECUTE)
        
        logger.info("Database tables created successfully")
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {}
        
        # Count records in each table
        tables = ["videos", "clips", "processing_jobs"]
        for table in tables:
            result = await self.db.execute_query(
                f"SELECT COUNT(*) as count FROM {table}",
                query_type=QueryType.SELECT
            )
            stats[f"{table}_count"] = result[0]["count"] if result else 0
        
        # Get status distribution
        status_result = await self.db.execute_query(
            "SELECT status, COUNT(*) as count FROM videos GROUP BY status",
            query_type=QueryType.SELECT
        )
        stats["status_distribution"] = {row["status"]: row["count"] for row in status_result}
        
        return stats

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_database_config(**kwargs) -> DatabaseConfig:
    """Create database configuration."""
    return DatabaseConfig(**kwargs)

def create_database_pool(config: DatabaseConfig, db_type: DatabaseType) -> AsyncDatabasePool:
    """Create database connection pool."""
    if db_type == DatabaseType.POSTGRESQL:
        return PostgreSQLPool(config, db_type)
    elif db_type == DatabaseType.MYSQL:
        return MySQLPool(config, db_type)
    elif db_type == DatabaseType.SQLITE:
        return SQLitePool(config, db_type)
    elif db_type == DatabaseType.REDIS:
        return RedisPool(config, db_type)
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def create_async_database_operations(pool: AsyncDatabasePool) -> AsyncDatabaseOperations:
    """Create async database operations."""
    return AsyncDatabaseOperations(pool)

def create_async_video_database(db_operations: AsyncDatabaseOperations) -> AsyncVideoDatabase:
    """Create async video database operations."""
    return AsyncVideoDatabase(db_operations)

def create_async_batch_database_operations(db_operations: AsyncDatabaseOperations) -> AsyncBatchDatabaseOperations:
    """Create async batch database operations."""
    return AsyncBatchDatabaseOperations(db_operations)

def create_async_transaction_manager(db_operations: AsyncDatabaseOperations) -> AsyncTransactionManager:
    """Create async transaction manager."""
    return AsyncTransactionManager(db_operations)

def create_async_database_setup(db_operations: AsyncDatabaseOperations) -> AsyncDatabaseSetup:
    """Create async database setup utilities."""
    return AsyncDatabaseSetup(db_operations)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def setup_database_connection(
    db_type: DatabaseType,
    host: str = "localhost",
    port: int = 5432,
    database: str = "video_opusclip",
    username: str = "postgres",
    password: str = "",
    **kwargs
) -> AsyncDatabaseOperations:
    """Setup database connection with default configuration."""
    config = create_database_config(
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
        **kwargs
    )
    
    pool = create_database_pool(config, db_type)
    await pool.initialize()
    
    return create_async_database_operations(pool)

async def close_database_connection(db_operations: AsyncDatabaseOperations):
    """Close database connection."""
    await db_operations.pool.close()

def get_query_metrics(db_operations: AsyncDatabaseOperations) -> Dict[str, Any]:
    """Get database query metrics."""
    metrics = db_operations.metrics.copy()
    
    if metrics["queries_executed"] > 0:
        metrics["avg_execution_time"] = metrics["total_execution_time"] / metrics["queries_executed"]
        metrics["cache_hit_rate"] = metrics["cache_hits"] / (metrics["cache_hits"] + metrics["cache_misses"])
    
    return metrics 