from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import asyncpg
import aioredis
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable, TypeVar, Generic
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
import structlog
from dataclasses import dataclass, asdict, field
from enum import Enum
import weakref
from functools import lru_cache, wraps
import inspect
import gc
import json
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import signal
import os
import hashlib
import pickle
from fastapi import Request, Response, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import text, select, insert, update, delete
from sqlalchemy.orm import DeclarativeBase
from pydantic import BaseModel, Field, validator
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Async Database Operations for HeyGen AI FastAPI
Dedicated async functions for database operations with connection pooling and optimization.
"""



logger = structlog.get_logger()

# =============================================================================
# Database Types
# =============================================================================

class DatabaseType(Enum):
    """Database type enumeration."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    REDIS = "redis"

class QueryType(Enum):
    """Query type enumeration."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    TRANSACTION = "transaction"

class ConnectionState(Enum):
    """Connection state enumeration."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    CLOSED = "closed"

@dataclass
class DatabaseConfig:
    """Database configuration."""
    database_type: DatabaseType
    host: str = "localhost"
    port: int = 5432
    database: str = "heygen_ai"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: float = 30.0
    pool_recycle: int = 3600
    echo: bool = False
    ssl_mode: str = "prefer"
    connection_timeout: float = 30.0
    query_timeout: float = 30.0
    enable_query_cache: bool = True
    query_cache_ttl: int = 300
    enable_connection_monitoring: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0

@dataclass
class QueryMetrics:
    """Query performance metrics."""
    query_id: str
    query_type: QueryType
    query_text: str
    parameters: Optional[Dict[str, Any]] = None
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    success: bool = False
    error_message: Optional[str] = None
    rows_affected: int = 0
    result_count: int = 0
    cache_hit: bool = False
    connection_id: Optional[str] = None

# =============================================================================
# Base Database Class
# =============================================================================

class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass

# =============================================================================
# Async Database Manager
# =============================================================================

class AsyncDatabaseManager:
    """Main async database manager with connection pooling and optimization."""
    
    def __init__(self, config: DatabaseConfig):
        
    """__init__ function."""
self.config = config
        self.engine = None
        self.session_factory = None
        self.connection_pool = None
        self.query_cache: Dict[str, Any] = {}
        self.query_metrics: Dict[str, QueryMetrics] = {}
        self.connection_metrics: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._is_initialized = False
        self._monitoring_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> Any:
        """Initialize the database manager."""
        if self._is_initialized:
            return
        
        try:
            # Create async engine
            if self.config.database_type == DatabaseType.POSTGRESQL:
                database_url = f"postgresql+asyncpg://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
            elif self.config.database_type == DatabaseType.MYSQL:
                database_url = f"mysql+asyncmy://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
            elif self.config.database_type == DatabaseType.SQLITE:
                database_url = f"sqlite+aiosqlite:///{self.config.database}"
            else:
                raise ValueError(f"Unsupported database type: {self.config.database_type}")
            
            self.engine = create_async_engine(
                database_url,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=self.config.echo,
                connect_args={
                    "command_timeout": self.config.connection_timeout,
                    "server_settings": {
                        "statement_timeout": str(int(self.config.query_timeout * 1000))
                    }
                } if self.config.database_type == DatabaseType.POSTGRESQL else {}
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connection
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
            
            self._is_initialized = True
            
            # Start monitoring
            if self.config.enable_connection_monitoring:
                self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info(f"Database manager initialized for {self.config.database_type.value}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            raise
    
    async def cleanup(self) -> Any:
        """Cleanup the database manager."""
        if not self._is_initialized:
            return
        
        # Stop monitoring
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Close engine
        if self.engine:
            await self.engine.dispose()
        
        self._is_initialized = False
        logger.info("Database manager cleaned up")
    
    @asynccontextmanager
    async def get_session(self) -> Optional[Dict[str, Any]]:
        """Get database session from pool."""
        if not self._is_initialized:
            raise RuntimeError("Database manager not initialized")
        
        session = self.session_factory()
        try:
            yield session
        except Exception as e:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    async def _monitoring_loop(self) -> Any:
        """Database monitoring loop."""
        while self._is_initialized:
            try:
                # Monitor connection pool
                pool_info = self.engine.pool.status()
                self.connection_metrics["pool"] = {
                    "size": pool_info.size,
                    "checked_in": pool_info.checked_in,
                    "checked_out": pool_info.checked_out,
                    "overflow": pool_info.overflow,
                    "invalid": pool_info.invalid
                }
                
                # Monitor query performance
                if self.query_metrics:
                    avg_duration = statistics.mean([
                        m.duration_ms for m in self.query_metrics.values()
                        if m.duration_ms is not None
                    ])
                    self.connection_metrics["queries"] = {
                        "total": len(self.query_metrics),
                        "avg_duration_ms": avg_duration,
                        "success_rate": sum(1 for m in self.query_metrics.values() if m.success) / len(self.query_metrics)
                    }
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Database monitoring error: {e}")
                await asyncio.sleep(30)
    
    def get_connection_metrics(self) -> Dict[str, Any]:
        """Get database connection metrics."""
        return self.connection_metrics.copy()

# =============================================================================
# Async Database Operations
# =============================================================================

class AsyncDatabaseOperations:
    """Dedicated async functions for database operations."""
    
    def __init__(self, db_manager: AsyncDatabaseManager):
        
    """__init__ function."""
self.db_manager = db_manager
        self.query_cache: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        cache_key: Optional[str] = None,
        cache_ttl: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Execute a SELECT query asynchronously."""
        query_id = self._generate_query_id(query, parameters)
        
        # Check cache
        if cache_key and self.db_manager.config.enable_query_cache:
            cached_result = await self._get_cached_query(cache_key, cache_ttl)
            if cached_result is not None:
                await self._record_query_metrics(query_id, QueryType.SELECT, query, parameters, cache_hit=True)
                return cached_result
        
        # Execute query
        start_time = time.time()
        try:
            async with self.db_manager.get_session() as session:
                result = await asyncio.wait_for(
                    session.execute(text(query), parameters or {}),
                    timeout=timeout or self.db_manager.config.query_timeout
                )
                
                # Convert to list of dictionaries
                rows = [dict(row._mapping) for row in result.fetchall()]
                
                # Cache result
                if cache_key and self.db_manager.config.enable_query_cache:
                    await self._cache_query_result(cache_key, rows, cache_ttl)
                
                # Record metrics
                duration_ms = (time.time() - start_time) * 1000
                await self._record_query_metrics(
                    query_id, QueryType.SELECT, query, parameters,
                    duration_ms=duration_ms, success=True, result_count=len(rows)
                )
                
                return rows
                
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            await self._record_query_metrics(
                query_id, QueryType.SELECT, query, parameters,
                duration_ms=duration_ms, success=False, error_message="Query timeout"
            )
            raise HTTPException(status_code=408, detail="Database query timeout")
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            await self._record_query_metrics(
                query_id, QueryType.SELECT, query, parameters,
                duration_ms=duration_ms, success=False, error_message=str(e)
            )
            logger.error(f"Database query error: {e}")
            raise HTTPException(status_code=500, detail="Database error")
    
    async def execute_insert(
        self,
        table: str,
        data: Dict[str, Any],
        returning: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Execute an INSERT query asynchronously."""
        query_id = self._generate_query_id(f"INSERT INTO {table}", data)
        
        # Build INSERT query
        columns = list(data.keys())
        values = list(data.values())
        placeholders = [f":{col}" for col in columns]
        
        query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        if returning:
            query += f" RETURNING {returning}"
        
        start_time = time.time()
        try:
            async with self.db_manager.get_session() as session:
                result = await asyncio.wait_for(
                    session.execute(text(query), data),
                    timeout=timeout or self.db_manager.config.query_timeout
                )
                
                await session.commit()
                
                # Get returned data if specified
                returned_data = None
                if returning:
                    row = result.fetchone()
                    if row:
                        returned_data = dict(row._mapping)
                
                # Record metrics
                duration_ms = (time.time() - start_time) * 1000
                await self._record_query_metrics(
                    query_id, QueryType.INSERT, query, data,
                    duration_ms=duration_ms, success=True, rows_affected=1
                )
                
                return returned_data
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            await self._record_query_metrics(
                query_id, QueryType.INSERT, query, data,
                duration_ms=duration_ms, success=False, error_message=str(e)
            )
            logger.error(f"Database insert error: {e}")
            raise HTTPException(status_code=500, detail="Database insert error")
    
    async def execute_update(
        self,
        table: str,
        data: Dict[str, Any],
        where_conditions: Dict[str, Any],
        returning: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Execute an UPDATE query asynchronously."""
        query_id = self._generate_query_id(f"UPDATE {table}", {**data, **where_conditions})
        
        # Build UPDATE query
        set_clause = ", ".join([f"{col} = :{col}" for col in data.keys()])
        where_clause = " AND ".join([f"{col} = :where_{col}" for col in where_conditions.keys()])
        
        query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
        if returning:
            query += f" RETURNING {returning}"
        
        # Prepare parameters
        parameters = {**data, **{f"where_{k}": v for k, v in where_conditions.items()}}
        
        start_time = time.time()
        try:
            async with self.db_manager.get_session() as session:
                result = await asyncio.wait_for(
                    session.execute(text(query), parameters),
                    timeout=timeout or self.db_manager.config.query_timeout
                )
                
                await session.commit()
                
                # Get returned data if specified
                returned_data = None
                if returning:
                    row = result.fetchone()
                    if row:
                        returned_data = dict(row._mapping)
                
                # Record metrics
                duration_ms = (time.time() - start_time) * 1000
                await self._record_query_metrics(
                    query_id, QueryType.UPDATE, query, parameters,
                    duration_ms=duration_ms, success=True, rows_affected=result.rowcount
                )
                
                return returned_data
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            await self._record_query_metrics(
                query_id, QueryType.UPDATE, query, parameters,
                duration_ms=duration_ms, success=False, error_message=str(e)
            )
            logger.error(f"Database update error: {e}")
            raise HTTPException(status_code=500, detail="Database update error")
    
    async def execute_delete(
        self,
        table: str,
        where_conditions: Dict[str, Any],
        returning: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Execute a DELETE query asynchronously."""
        query_id = self._generate_query_id(f"DELETE FROM {table}", where_conditions)
        
        # Build DELETE query
        where_clause = " AND ".join([f"{col} = :{col}" for col in where_conditions.keys()])
        
        query = f"DELETE FROM {table} WHERE {where_clause}"
        if returning:
            query += f" RETURNING {returning}"
        
        start_time = time.time()
        try:
            async with self.db_manager.get_session() as session:
                result = await asyncio.wait_for(
                    session.execute(text(query), where_conditions),
                    timeout=timeout or self.db_manager.config.query_timeout
                )
                
                await session.commit()
                
                # Get returned data if specified
                returned_data = None
                if returning:
                    row = result.fetchone()
                    if row:
                        returned_data = dict(row._mapping)
                
                # Record metrics
                duration_ms = (time.time() - start_time) * 1000
                await self._record_query_metrics(
                    query_id, QueryType.DELETE, query, where_conditions,
                    duration_ms=duration_ms, success=True, rows_affected=result.rowcount
                )
                
                return returned_data
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            await self._record_query_metrics(
                query_id, QueryType.DELETE, query, where_conditions,
                duration_ms=duration_ms, success=False, error_message=str(e)
            )
            logger.error(f"Database delete error: {e}")
            raise HTTPException(status_code=500, detail="Database delete error")
    
    async def execute_transaction(
        self,
        queries: List[Dict[str, Any]],
        timeout: Optional[float] = None
    ) -> List[Any]:
        """Execute multiple queries in a transaction."""
        transaction_id = f"txn_{int(time.time() * 1000)}"
        
        start_time = time.time()
        try:
            async with self.db_manager.get_session() as session:
                async with session.begin():
                    results = []
                    
                    for i, query_data in enumerate(queries):
                        query_type = query_data["type"]
                        query = query_data["query"]
                        parameters = query_data.get("parameters", {})
                        
                        if query_type == QueryType.SELECT:
                            result = await session.execute(text(query), parameters)
                            rows = [dict(row._mapping) for row in result.fetchall()]
                            results.append(rows)
                            
                        elif query_type == QueryType.INSERT:
                            result = await session.execute(text(query), parameters)
                            results.append({"inserted": True})
                            
                        elif query_type == QueryType.UPDATE:
                            result = await session.execute(text(query), parameters)
                            results.append({"updated_rows": result.rowcount})
                            
                        elif query_type == QueryType.DELETE:
                            result = await session.execute(text(query), parameters)
                            results.append({"deleted_rows": result.rowcount})
                    
                    # Record transaction metrics
                    duration_ms = (time.time() - start_time) * 1000
                    await self._record_query_metrics(
                        transaction_id, QueryType.TRANSACTION, f"Transaction with {len(queries)} queries",
                        {"query_count": len(queries)},
                        duration_ms=duration_ms, success=True
                    )
                    
                    return results
                    
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            await self._record_query_metrics(
                transaction_id, QueryType.TRANSACTION, f"Transaction with {len(queries)} queries",
                {"query_count": len(queries)},
                duration_ms=duration_ms, success=False, error_message=str(e)
            )
            logger.error(f"Database transaction error: {e}")
            raise HTTPException(status_code=500, detail="Database transaction error")
    
    async def execute_batch_operations(
        self,
        operations: List[Dict[str, Any]],
        batch_size: int = 100,
        timeout: Optional[float] = None
    ) -> List[Any]:
        """Execute batch operations asynchronously."""
        results = []
        
        for i in range(0, len(operations), batch_size):
            batch = operations[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[self._execute_single_operation(op) for op in batch],
                return_exceptions=True
            )
            results.extend(batch_results)
        
        return results
    
    async def _execute_single_operation(self, operation: Dict[str, Any]) -> Any:
        """Execute a single database operation."""
        op_type = operation["type"]
        
        if op_type == QueryType.SELECT:
            return await self.execute_query(
                operation["query"],
                operation.get("parameters")
            )
        elif op_type == QueryType.INSERT:
            return await self.execute_insert(
                operation["table"],
                operation["data"],
                operation.get("returning")
            )
        elif op_type == QueryType.UPDATE:
            return await self.execute_update(
                operation["table"],
                operation["data"],
                operation["where_conditions"],
                operation.get("returning")
            )
        elif op_type == QueryType.DELETE:
            return await self.execute_delete(
                operation["table"],
                operation["where_conditions"],
                operation.get("returning")
            )
        else:
            raise ValueError(f"Unknown operation type: {op_type}")
    
    async def stream_query_results(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        chunk_size: int = 1000,
        timeout: Optional[float] = None
    ):
        """Stream database query results."""
        query_id = self._generate_query_id(query, parameters)
        
        start_time = time.time()
        try:
            async with self.db_manager.get_session() as session:
                result = await asyncio.wait_for(
                    session.execute(text(query), parameters or {}),
                    timeout=timeout or self.db_manager.config.query_timeout
                )
                
                chunk = []
                for row in result:
                    chunk.append(dict(row._mapping))
                    
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
                
                # Yield remaining rows
                if chunk:
                    yield chunk
                
                # Record metrics
                duration_ms = (time.time() - start_time) * 1000
                await self._record_query_metrics(
                    query_id, QueryType.SELECT, query, parameters,
                    duration_ms=duration_ms, success=True
                )
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            await self._record_query_metrics(
                query_id, QueryType.SELECT, query, parameters,
                duration_ms=duration_ms, success=False, error_message=str(e)
            )
            logger.error(f"Database stream error: {e}")
            raise HTTPException(status_code=500, detail="Database stream error")
    
    def _generate_query_id(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Generate unique query ID."""
        query_str = query + json.dumps(parameters or {}, sort_keys=True)
        return hashlib.md5(query_str.encode()).hexdigest()
    
    async def _get_cached_query(self, cache_key: str, ttl: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
        """Get cached query result."""
        async with self._lock:
            if cache_key in self.query_cache:
                cached_data = self.query_cache[cache_key]
                cache_ttl = ttl or self.db_manager.config.query_cache_ttl
                
                if time.time() - cached_data["timestamp"] < cache_ttl:
                    return cached_data["data"]
                else:
                    # Remove expired cache entry
                    del self.query_cache[cache_key]
        
        return None
    
    async def _cache_query_result(self, cache_key: str, data: List[Dict[str, Any]], ttl: Optional[int] = None):
        """Cache query result."""
        async with self._lock:
            self.query_cache[cache_key] = {
                "data": data,
                "timestamp": time.time()
            }
    
    async def _record_query_metrics(
        self,
        query_id: str,
        query_type: QueryType,
        query_text: str,
        parameters: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        success: bool = False,
        error_message: Optional[str] = None,
        rows_affected: int = 0,
        result_count: int = 0,
        cache_hit: bool = False
    ):
        """Record query performance metrics."""
        metrics = QueryMetrics(
            query_id=query_id,
            query_type=query_type,
            query_text=query_text,
            parameters=parameters,
            end_time=datetime.now(timezone.utc) if duration_ms else None,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            rows_affected=rows_affected,
            result_count=result_count,
            cache_hit=cache_hit
        )
        
        self.db_manager.query_metrics[query_id] = metrics
    
    def get_query_metrics(self) -> Dict[str, QueryMetrics]:
        """Get query performance metrics."""
        return self.db_manager.query_metrics.copy()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get query cache statistics."""
        async with self._lock:
            return {
                "cache_size": len(self.query_cache),
                "cache_ttl": self.db_manager.config.query_cache_ttl
            }

# =============================================================================
# Async Redis Operations
# =============================================================================

class AsyncRedisOperations:
    """Dedicated async functions for Redis operations."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        
    """__init__ function."""
self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        self._is_initialized = False
    
    async def initialize(self) -> Any:
        """Initialize Redis connection."""
        if self._is_initialized:
            return
        
        try:
            self.redis = aioredis.from_url(self.redis_url)
            await self.redis.ping()
            self._is_initialized = True
            logger.info("Redis operations initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise
    
    async def cleanup(self) -> Any:
        """Cleanup Redis connection."""
        if self.redis:
            await self.redis.close()
        self._is_initialized = False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        if not self._is_initialized:
            raise RuntimeError("Redis not initialized")
        
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis."""
        if not self._is_initialized:
            raise RuntimeError("Redis not initialized")
        
        try:
            serialized_value = json.dumps(value)
            if ttl:
                await self.redis.setex(key, ttl, serialized_value)
            else:
                await self.redis.set(key, serialized_value)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis."""
        if not self._is_initialized:
            raise RuntimeError("Redis not initialized")
        
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        if not self._is_initialized:
            raise RuntimeError("Redis not initialized")
        
        try:
            return await self.redis.exists(key)
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key."""
        if not self._is_initialized:
            raise RuntimeError("Redis not initialized")
        
        try:
            return await self.redis.expire(key, ttl)
        except Exception as e:
            logger.error(f"Redis expire error: {e}")
            return False
    
    async def get_or_set(self, key: str, fetch_func: Callable, ttl: int = 300) -> Optional[Dict[str, Any]]:
        """Get from Redis or fetch and set."""
        # Try to get from Redis
        cached_value = await self.get(key)
        if cached_value is not None:
            return cached_value
        
        # Fetch value
        value = await fetch_func()
        
        # Set in Redis
        await self.set(key, value, ttl)
        
        return value

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "DatabaseType",
    "QueryType",
    "ConnectionState",
    "DatabaseConfig",
    "QueryMetrics",
    "Base",
    "AsyncDatabaseManager",
    "AsyncDatabaseOperations",
    "AsyncRedisOperations",
] 