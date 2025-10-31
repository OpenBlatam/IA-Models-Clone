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
import logging
import functools
import inspect
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, TypeVar, Generic, Awaitable, Type
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
import json
import sqlite3
from pathlib import Path
import weakref
import contextlib
from concurrent.futures import ThreadPoolExecutor
import structlog
from pydantic import BaseModel, Field
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import text, select, insert, update, delete, and_, or_, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import QueuePool
import aiosqlite
import asyncpg
import redis.asyncio as redis
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Any, List, Dict, Optional
"""
🗄️ Async Database Operations
============================

Comprehensive async database operations module with:
- Dedicated async functions for all database operations
- Connection pooling and management
- Transaction handling
- CRUD operations with async patterns
- Query optimization and caching
- Error handling and retry logic
- Performance monitoring
- Migration support
- Multiple database backends
- Async ORM integration
"""



logger = structlog.get_logger(__name__)

T = TypeVar('T')
ModelT = TypeVar('ModelT', bound=DeclarativeBase)

class DatabaseType(Enum):
    """Supported database types"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    REDIS = "redis"
    MONGODB = "mongodb"

class OperationType(Enum):
    """Database operation types"""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    TRANSACTION = "transaction"
    BULK_OPERATION = "bulk_operation"
    MIGRATION = "migration"
    BACKUP = "backup"
    RESTORE = "restore"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    database_type: DatabaseType
    connection_string: str
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    echo_pool: bool = False
    enable_metrics: bool = True
    enable_caching: bool = True
    cache_ttl: int = 3600
    retry_attempts: int = 3
    retry_delay: float = 1.0

@dataclass
class QueryMetrics:
    """Query performance metrics"""
    operation_type: OperationType
    table_name: str
    execution_time: float
    rows_affected: int
    cache_hit: bool
    timestamp: datetime
    query_hash: str
    parameters: Dict[str, Any]

class AsyncDatabaseManager:
    """Main async database manager"""
    
    def __init__(self, config: DatabaseConfig):
        
    """__init__ function."""
self.config = config
        self.engine = None
        self.session_factory = None
        self.redis_client = None
        self.connection_pool = None
        
        # Performance tracking
        self.query_metrics: List[QueryMetrics] = []
        self.connection_usage = defaultdict(int)
        self.cache_stats = {"hits": 0, "misses": 0}
        
        # Connection pool management
        self.active_connections = 0
        self.max_connections = config.pool_size + config.max_overflow
        self.connection_semaphore = asyncio.Semaphore(config.pool_size)
        
        logger.info(f"Async Database Manager initialized for {config.database_type.value}")
    
    async def initialize(self) -> Any:
        """Initialize database connections and pools"""
        try:
            if self.config.database_type == DatabaseType.SQLITE:
                await self._initialize_sqlite()
            elif self.config.database_type == DatabaseType.POSTGRESQL:
                await self._initialize_postgresql()
            elif self.config.database_type == DatabaseType.REDIS:
                await self._initialize_redis()
            else:
                await self._initialize_sqlalchemy()
            
            # Initialize caching if enabled
            if self.config.enable_caching:
                await self._initialize_caching()
            
            logger.info("Database manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            raise
    
    async def _initialize_sqlite(self) -> Any:
        """Initialize SQLite with aiosqlite"""
        self.connection_pool = await aiosqlite.connect(self.config.connection_string)
        await self.connection_pool.execute("PRAGMA journal_mode=WAL")
        await self.connection_pool.execute("PRAGMA synchronous=NORMAL")
        await self.connection_pool.execute("PRAGMA cache_size=10000")
        await self.connection_pool.execute("PRAGMA temp_store=MEMORY")
    
    async def _initialize_postgresql(self) -> Any:
        """Initialize PostgreSQL with asyncpg"""
        self.connection_pool = await asyncpg.create_pool(
            self.config.connection_string,
            min_size=5,
            max_size=self.config.pool_size,
            command_timeout=self.config.pool_timeout
        )
    
    async def _initialize_redis(self) -> Any:
        """Initialize Redis connection"""
        self.redis_client = redis.from_url(self.config.connection_string)
        await self.redis_client.ping()
    
    async def _initialize_sqlalchemy(self) -> Any:
        """Initialize SQLAlchemy async engine"""
        self.engine = create_async_engine(
            self.config.connection_string,
            poolclass=QueuePool,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            echo=self.config.echo,
            echo_pool=self.config.echo_pool
        )
        
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def _initialize_caching(self) -> Any:
        """Initialize caching system"""
        if not self.redis_client:
            self.redis_client = redis.from_url("redis://localhost:6379")
            await self.redis_client.ping()
    
    async def get_session(self) -> AsyncSession:
        """Get database session"""
        if not self.session_factory:
            raise RuntimeError("SQLAlchemy not initialized")
        
        async with self.connection_semaphore:
            self.active_connections += 1
            session = self.session_factory()
            return session
    
    async def close_session(self, session: AsyncSession):
        """Close database session"""
        if session:
            await session.close()
            self.active_connections -= 1
    
    # CRUD Operations
    
    async def select_one(self, table_name: str, conditions: Dict[str, Any], 
                        cache_key: str = None) -> Optional[Dict[str, Any]]:
        """Select single record"""
        start_time = time.time()
        
        try:
            # Check cache first
            if cache_key and self.config.enable_caching:
                cached_result = await self._get_from_cache(cache_key)
                if cached_result:
                    self._record_metrics(OperationType.SELECT, table_name, 
                                       time.time() - start_time, 1, True)
                    return cached_result
            
            # Execute query
            if self.config.database_type == DatabaseType.SQLITE:
                result = await self._sqlite_select_one(table_name, conditions)
            elif self.config.database_type == DatabaseType.POSTGRESQL:
                result = await self._postgresql_select_one(table_name, conditions)
            else:
                result = await self._sqlalchemy_select_one(table_name, conditions)
            
            # Cache result
            if cache_key and self.config.enable_caching and result:
                await self._set_cache(cache_key, result)
            
            self._record_metrics(OperationType.SELECT, table_name, 
                               time.time() - start_time, 1 if result else 0, False)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in select_one: {e}")
            raise
    
    async def select_many(self, table_name: str, conditions: Dict[str, Any] = None,
                         limit: int = 100, offset: int = 0, 
                         order_by: str = None, cache_key: str = None) -> List[Dict[str, Any]]:
        """Select multiple records"""
        start_time = time.time()
        
        try:
            # Check cache first
            if cache_key and self.config.enable_caching:
                cached_result = await self._get_from_cache(cache_key)
                if cached_result:
                    self._record_metrics(OperationType.SELECT, table_name, 
                                       time.time() - start_time, len(cached_result), True)
                    return cached_result
            
            # Execute query
            if self.config.database_type == DatabaseType.SQLITE:
                result = await self._sqlite_select_many(table_name, conditions, limit, offset, order_by)
            elif self.config.database_type == DatabaseType.POSTGRESQL:
                result = await self._postgresql_select_many(table_name, conditions, limit, offset, order_by)
            else:
                result = await self._sqlalchemy_select_many(table_name, conditions, limit, offset, order_by)
            
            # Cache result
            if cache_key and self.config.enable_caching and result:
                await self._set_cache(cache_key, result)
            
            self._record_metrics(OperationType.SELECT, table_name, 
                               time.time() - start_time, len(result), False)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in select_many: {e}")
            raise
    
    async def insert_one(self, table_name: str, data: Dict[str, Any]) -> int:
        """Insert single record"""
        start_time = time.time()
        
        try:
            if self.config.database_type == DatabaseType.SQLITE:
                result = await self._sqlite_insert_one(table_name, data)
            elif self.config.database_type == DatabaseType.POSTGRESQL:
                result = await self._postgresql_insert_one(table_name, data)
            else:
                result = await self._sqlalchemy_insert_one(table_name, data)
            
            self._record_metrics(OperationType.INSERT, table_name, 
                               time.time() - start_time, 1, False)
            
            # Invalidate related cache
            await self._invalidate_table_cache(table_name)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in insert_one: {e}")
            raise
    
    async def insert_many(self, table_name: str, data_list: List[Dict[str, Any]]) -> int:
        """Insert multiple records"""
        start_time = time.time()
        
        try:
            if self.config.database_type == DatabaseType.SQLITE:
                result = await self._sqlite_insert_many(table_name, data_list)
            elif self.config.database_type == DatabaseType.POSTGRESQL:
                result = await self._postgresql_insert_many(table_name, data_list)
            else:
                result = await self._sqlalchemy_insert_many(table_name, data_list)
            
            self._record_metrics(OperationType.INSERT, table_name, 
                               time.time() - start_time, len(data_list), False)
            
            # Invalidate related cache
            await self._invalidate_table_cache(table_name)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in insert_many: {e}")
            raise
    
    async def update_one(self, table_name: str, conditions: Dict[str, Any], 
                        data: Dict[str, Any]) -> int:
        """Update single record"""
        start_time = time.time()
        
        try:
            if self.config.database_type == DatabaseType.SQLITE:
                result = await self._sqlite_update_one(table_name, conditions, data)
            elif self.config.database_type == DatabaseType.POSTGRESQL:
                result = await self._postgresql_update_one(table_name, conditions, data)
            else:
                result = await self._sqlalchemy_update_one(table_name, conditions, data)
            
            self._record_metrics(OperationType.UPDATE, table_name, 
                               time.time() - start_time, result, False)
            
            # Invalidate related cache
            await self._invalidate_table_cache(table_name)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in update_one: {e}")
            raise
    
    async def update_many(self, table_name: str, conditions: Dict[str, Any], 
                         data: Dict[str, Any]) -> int:
        """Update multiple records"""
        start_time = time.time()
        
        try:
            if self.config.database_type == DatabaseType.SQLITE:
                result = await self._sqlite_update_many(table_name, conditions, data)
            elif self.config.database_type == DatabaseType.POSTGRESQL:
                result = await self._postgresql_update_many(table_name, conditions, data)
            else:
                result = await self._sqlalchemy_update_many(table_name, conditions, data)
            
            self._record_metrics(OperationType.UPDATE, table_name, 
                               time.time() - start_time, result, False)
            
            # Invalidate related cache
            await self._invalidate_table_cache(table_name)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in update_many: {e}")
            raise
    
    async def delete_one(self, table_name: str, conditions: Dict[str, Any]) -> int:
        """Delete single record"""
        start_time = time.time()
        
        try:
            if self.config.database_type == DatabaseType.SQLITE:
                result = await self._sqlite_delete_one(table_name, conditions)
            elif self.config.database_type == DatabaseType.POSTGRESQL:
                result = await self._postgresql_delete_one(table_name, conditions)
            else:
                result = await self._sqlalchemy_delete_one(table_name, conditions)
            
            self._record_metrics(OperationType.DELETE, table_name, 
                               time.time() - start_time, result, False)
            
            # Invalidate related cache
            await self._invalidate_table_cache(table_name)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in delete_one: {e}")
            raise
    
    async def delete_many(self, table_name: str, conditions: Dict[str, Any]) -> int:
        """Delete multiple records"""
        start_time = time.time()
        
        try:
            if self.config.database_type == DatabaseType.SQLITE:
                result = await self._sqlite_delete_many(table_name, conditions)
            elif self.config.database_type == DatabaseType.POSTGRESQL:
                result = await self._postgresql_delete_many(table_name, conditions)
            else:
                result = await self._sqlalchemy_delete_many(table_name, conditions)
            
            self._record_metrics(OperationType.DELETE, table_name, 
                               time.time() - start_time, result, False)
            
            # Invalidate related cache
            await self._invalidate_table_cache(table_name)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in delete_many: {e}")
            raise
    
    # Transaction Operations
    
    async def transaction(self, operations: List[Callable[[AsyncSession], Awaitable[Any]]]) -> List[Any]:
        """Execute operations in a transaction"""
        start_time = time.time()
        
        try:
            if self.config.database_type == DatabaseType.SQLITE:
                result = await self._sqlite_transaction(operations)
            elif self.config.database_type == DatabaseType.POSTGRESQL:
                result = await self._postgresql_transaction(operations)
            else:
                result = await self._sqlalchemy_transaction(operations)
            
            self._record_metrics(OperationType.TRANSACTION, "transaction", 
                               time.time() - start_time, len(operations), False)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in transaction: {e}")
            raise
    
    # SQLite-specific implementations
    
    async def _sqlite_select_one(self, table_name: str, conditions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """SQLite select one implementation"""
        where_clause = " AND ".join([f"{k} = ?" for k in conditions.keys()])
        query = f"SELECT * FROM {table_name} WHERE {where_clause} LIMIT 1"
        
        async with self.connection_pool.execute(query, list(conditions.values())) as cursor:
            row = await cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, row))
            return None
    
    async def _sqlite_select_many(self, table_name: str, conditions: Dict[str, Any] = None,
                                 limit: int = 100, offset: int = 0, order_by: str = None) -> List[Dict[str, Any]]:
        """SQLite select many implementation"""
        query = f"SELECT * FROM {table_name}"
        params = []
        
        if conditions:
            where_clause = " AND ".join([f"{k} = ?" for k in conditions.keys()])
            query += f" WHERE {where_clause}"
            params.extend(conditions.values())
        
        if order_by:
            query += f" ORDER BY {order_by}"
        
        query += f" LIMIT {limit} OFFSET {offset}"
        
        async with self.connection_pool.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
    
    async def _sqlite_insert_one(self, table_name: str, data: Dict[str, Any]) -> int:
        """SQLite insert one implementation"""
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        
        async with self.connection_pool.execute(query, list(data.values())) as cursor:
            await self.connection_pool.commit()
            return cursor.lastrowid
    
    async def _sqlite_insert_many(self, table_name: str, data_list: List[Dict[str, Any]]) -> int:
        """SQLite insert many implementation"""
        if not data_list:
            return 0
        
        columns = ", ".join(data_list[0].keys())
        placeholders = ", ".join(["?" for _ in data_list[0]])
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        
        values = [list(data.values()) for data in data_list]
        
        async with self.connection_pool.executemany(query, values) as cursor:
            await self.connection_pool.commit()
            return len(data_list)
    
    async def _sqlite_update_one(self, table_name: str, conditions: Dict[str, Any], 
                                data: Dict[str, Any]) -> int:
        """SQLite update one implementation"""
        set_clause = ", ".join([f"{k} = ?" for k in data.keys()])
        where_clause = " AND ".join([f"{k} = ?" for k in conditions.keys()])
        query = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
        
        params = list(data.values()) + list(conditions.values())
        
        async with self.connection_pool.execute(query, params) as cursor:
            await self.connection_pool.commit()
            return cursor.rowcount
    
    async def _sqlite_delete_one(self, table_name: str, conditions: Dict[str, Any]) -> int:
        """SQLite delete one implementation"""
        where_clause = " AND ".join([f"{k} = ?" for k in conditions.keys()])
        query = f"DELETE FROM {table_name} WHERE {where_clause}"
        
        async with self.connection_pool.execute(query, list(conditions.values())) as cursor:
            await self.connection_pool.commit()
            return cursor.rowcount
    
    async def _sqlite_transaction(self, operations: List[Callable]) -> List[Any]:
        """SQLite transaction implementation"""
        results = []
        
        async with self.connection_pool:
            for operation in operations:
                result = await operation(self.connection_pool)
                results.append(result)
        
        return results
    
    # PostgreSQL-specific implementations
    
    async def _postgresql_select_one(self, table_name: str, conditions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """PostgreSQL select one implementation"""
        where_clause = " AND ".join([f"{k} = ${i+1}" for i, k in enumerate(conditions.keys())])
        query = f"SELECT * FROM {table_name} WHERE {where_clause} LIMIT 1"
        
        async with self.connection_pool.acquire() as connection:
            row = await connection.fetchrow(query, *conditions.values())
            return dict(row) if row else None
    
    async def _postgresql_select_many(self, table_name: str, conditions: Dict[str, Any] = None,
                                     limit: int = 100, offset: int = 0, order_by: str = None) -> List[Dict[str, Any]]:
        """PostgreSQL select many implementation"""
        query = f"SELECT * FROM {table_name}"
        params = []
        
        if conditions:
            where_clause = " AND ".join([f"{k} = ${len(params)+1}" for k in conditions.keys()])
            query += f" WHERE {where_clause}"
            params.extend(conditions.values())
        
        if order_by:
            query += f" ORDER BY {order_by}"
        
        query += f" LIMIT {limit} OFFSET {offset}"
        
        async with self.connection_pool.acquire() as connection:
            rows = await connection.fetch(query, *params)
            return [dict(row) for row in rows]
    
    async def _postgresql_insert_one(self, table_name: str, data: Dict[str, Any]) -> int:
        """PostgreSQL insert one implementation"""
        columns = ", ".join(data.keys())
        placeholders = ", ".join([f"${i+1}" for i in range(len(data))])
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders}) RETURNING id"
        
        async with self.connection_pool.acquire() as connection:
            result = await connection.fetchval(query, *data.values())
            return result
    
    async def _postgresql_insert_many(self, table_name: str, data_list: List[Dict[str, Any]]) -> int:
        """PostgreSQL insert many implementation"""
        if not data_list:
            return 0
        
        columns = ", ".join(data_list[0].keys())
        placeholders = ", ".join([f"${i+1}" for i in range(len(data_list[0]))])
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        
        async with self.connection_pool.acquire() as connection:
            await connection.executemany(query, [list(data.values()) for data in data_list])
            return len(data_list)
    
    async def _postgresql_update_one(self, table_name: str, conditions: Dict[str, Any], 
                                    data: Dict[str, Any]) -> int:
        """PostgreSQL update one implementation"""
        set_clause = ", ".join([f"{k} = ${i+1}" for i, k in enumerate(data.keys())])
        where_clause = " AND ".join([f"{k} = ${len(data)+i+1}" for i, k in enumerate(conditions.keys())])
        query = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
        
        params = list(data.values()) + list(conditions.values())
        
        async with self.connection_pool.acquire() as connection:
            result = await connection.execute(query, *params)
            return int(result.split()[-1])
    
    async def _postgresql_delete_one(self, table_name: str, conditions: Dict[str, Any]) -> int:
        """PostgreSQL delete one implementation"""
        where_clause = " AND ".join([f"{k} = ${i+1}" for i, k in enumerate(conditions.keys())])
        query = f"DELETE FROM {table_name} WHERE {where_clause}"
        
        async with self.connection_pool.acquire() as connection:
            result = await connection.execute(query, *conditions.values())
            return int(result.split()[-1])
    
    async def _postgresql_transaction(self, operations: List[Callable]) -> List[Any]:
        """PostgreSQL transaction implementation"""
        results = []
        
        async with self.connection_pool.acquire() as connection:
            async with connection.transaction():
                for operation in operations:
                    result = await operation(connection)
                    results.append(result)
        
        return results
    
    # SQLAlchemy-specific implementations
    
    async def _sqlalchemy_select_one(self, table_name: str, conditions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """SQLAlchemy select one implementation"""
        async with self.get_session() as session:
            query = text(f"SELECT * FROM {table_name} WHERE " + 
                        " AND ".join([f"{k} = :{k}" for k in conditions.keys()]) + " LIMIT 1")
            result = await session.execute(query, conditions)
            row = result.fetchone()
            return dict(row._mapping) if row else None
    
    async def _sqlalchemy_select_many(self, table_name: str, conditions: Dict[str, Any] = None,
                                     limit: int = 100, offset: int = 0, order_by: str = None) -> List[Dict[str, Any]]:
        """SQLAlchemy select many implementation"""
        async with self.get_session() as session:
            query = text(f"SELECT * FROM {table_name}")
            params = {}
            
            if conditions:
                where_clause = " AND ".join([f"{k} = :{k}" for k in conditions.keys()])
                query = text(str(query) + f" WHERE {where_clause}")
                params.update(conditions)
            
            if order_by:
                query = text(str(query) + f" ORDER BY {order_by}")
            
            query = text(str(query) + f" LIMIT {limit} OFFSET {offset}")
            
            result = await session.execute(query, params)
            rows = result.fetchall()
            return [dict(row._mapping) for row in rows]
    
    async def _sqlalchemy_insert_one(self, table_name: str, data: Dict[str, Any]) -> int:
        """SQLAlchemy insert one implementation"""
        async with self.get_session() as session:
            query = text(f"INSERT INTO {table_name} (" + 
                        ", ".join(data.keys()) + ") VALUES (" + 
                        ", ".join([f":{k}" for k in data.keys()]) + ") RETURNING id")
            result = await session.execute(query, data)
            await session.commit()
            return result.scalar()
    
    async def _sqlalchemy_insert_many(self, table_name: str, data_list: List[Dict[str, Any]]) -> int:
        """SQLAlchemy insert many implementation"""
        if not data_list:
            return 0
        
        async with self.get_session() as session:
            query = text(f"INSERT INTO {table_name} (" + 
                        ", ".join(data_list[0].keys()) + ") VALUES (" + 
                        ", ".join([f":{k}" for k in data_list[0].keys()]) + ")")
            
            for data in data_list:
                await session.execute(query, data)
            
            await session.commit()
            return len(data_list)
    
    async def _sqlalchemy_update_one(self, table_name: str, conditions: Dict[str, Any], 
                                    data: Dict[str, Any]) -> int:
        """SQLAlchemy update one implementation"""
        async with self.get_session() as session:
            set_clause = ", ".join([f"{k} = :{k}" for k in data.keys()])
            where_clause = " AND ".join([f"{k} = :{k}" for k in conditions.keys()])
            query = text(f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}")
            
            params = {**data, **conditions}
            result = await session.execute(query, params)
            await session.commit()
            return result.rowcount
    
    async def _sqlalchemy_delete_one(self, table_name: str, conditions: Dict[str, Any]) -> int:
        """SQLAlchemy delete one implementation"""
        async with self.get_session() as session:
            where_clause = " AND ".join([f"{k} = :{k}" for k in conditions.keys()])
            query = text(f"DELETE FROM {table_name} WHERE {where_clause}")
            
            result = await session.execute(query, conditions)
            await session.commit()
            return result.rowcount
    
    async def _sqlalchemy_transaction(self, operations: List[Callable]) -> List[Any]:
        """SQLAlchemy transaction implementation"""
        results = []
        
        async with self.get_session() as session:
            async with session.begin():
                for operation in operations:
                    result = await operation(session)
                    results.append(result)
        
        return results
    
    # Caching operations
    
    async def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.redis_client:
            return None
        
        try:
            value = await self.redis_client.get(key)
            if value:
                self.cache_stats["hits"] += 1
                return json.loads(value)
            else:
                self.cache_stats["misses"] += 1
                return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def _set_cache(self, key: str, value: Any, ttl: int = None):
        """Set value in cache"""
        if not self.redis_client:
            return
        
        try:
            ttl = ttl or self.config.cache_ttl
            serialized = json.dumps(value)
            await self.redis_client.setex(key, ttl, serialized)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    async def _invalidate_table_cache(self, table_name: str):
        """Invalidate cache for table"""
        if not self.redis_client:
            return
        
        try:
            pattern = f"cache:{table_name}:*"
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
    
    # Metrics and monitoring
    
    def _record_metrics(self, operation_type: OperationType, table_name: str, 
                       execution_time: float, rows_affected: int, cache_hit: bool):
        """Record query metrics"""
        metric = QueryMetrics(
            operation_type=operation_type,
            table_name=table_name,
            execution_time=execution_time,
            rows_affected=rows_affected,
            cache_hit=cache_hit,
            timestamp=datetime.now(),
            query_hash=f"{operation_type.value}_{table_name}",
            parameters={}
        )
        
        self.query_metrics.append(metric)
        
        # Keep only last 10000 metrics
        if len(self.query_metrics) > 10000:
            self.query_metrics = self.query_metrics[-10000:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if not self.query_metrics:
            return {"message": "No metrics available"}
        
        # Calculate statistics
        execution_times = [m.execution_time for m in self.query_metrics]
        cache_hit_rate = sum(1 for m in self.query_metrics if m.cache_hit) / len(self.query_metrics)
        
        # Group by operation type
        operation_stats = defaultdict(list)
        for metric in self.query_metrics:
            operation_stats[metric.operation_type.value].append(metric.execution_time)
        
        # Calculate operation-specific stats
        operation_metrics = {}
        for op_type, times in operation_stats.items():
            operation_metrics[op_type] = {
                "count": len(times),
                "avg_time": np.mean(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
                "p95_time": np.percentile(times, 95)
            }
        
        return {
            "total_queries": len(self.query_metrics),
            "cache_hit_rate": cache_hit_rate,
            "avg_execution_time": np.mean(execution_times),
            "p95_execution_time": np.percentile(execution_times, 95),
            "operation_metrics": operation_metrics,
            "connection_usage": dict(self.connection_usage),
            "cache_stats": self.cache_stats,
            "active_connections": self.active_connections
        }
    
    async def cleanup(self) -> Any:
        """Cleanup database connections"""
        try:
            if self.connection_pool:
                if hasattr(self.connection_pool, 'close'):
                    await self.connection_pool.close()
                elif hasattr(self.connection_pool, 'aclose'):
                    await self.connection_pool.aclose()
            
            if self.engine:
                await self.engine.dispose()
            
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("Database manager cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Decorators for database operations

def async_database_operation(operation_type: OperationType, table_name: str = None, 
                           cache_key: str = None, retry_attempts: int = 3):
    """Decorator for async database operations"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(db_manager: AsyncDatabaseManager, *args, **kwargs):
            
    """wrapper function."""
for attempt in range(retry_attempts):
                try:
                    if operation_type == OperationType.SELECT:
                        if table_name and cache_key:
                            return await db_manager.select_one(table_name, kwargs, cache_key)
                        elif table_name:
                            return await db_manager.select_many(table_name, kwargs)
                    elif operation_type == OperationType.INSERT:
                        if table_name:
                            return await db_manager.insert_one(table_name, kwargs)
                    elif operation_type == OperationType.UPDATE:
                        if table_name:
                            return await db_manager.update_one(table_name, args[0], kwargs)
                    elif operation_type == OperationType.DELETE:
                        if table_name:
                            return await db_manager.delete_one(table_name, kwargs)
                    elif operation_type == OperationType.TRANSACTION:
                        return await db_manager.transaction([func])
                    
                    return await func(db_manager, *args, **kwargs)
                    
                except Exception as e:
                    if attempt == retry_attempts - 1:
                        raise
                    await asyncio.sleep(db_manager.config.retry_delay * (2 ** attempt))
            
        return wrapper
    return decorator

# Example usage

async def example_database_operations():
    """Example usage of async database operations"""
    
    # Create database config
    config = DatabaseConfig(
        database_type=DatabaseType.SQLITE,
        connection_string=":memory:",
        pool_size=10,
        enable_caching=True
    )
    
    # Create database manager
    db_manager = AsyncDatabaseManager(config)
    await db_manager.initialize()
    
    try:
        # Create table
        await db_manager.connection_pool.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert user
        user_id = await db_manager.insert_one("users", {
            "name": "John Doe",
            "email": "john@example.com"
        })
        print(f"Inserted user with ID: {user_id}")
        
        # Select user
        user = await db_manager.select_one("users", {"id": user_id})
        print(f"Retrieved user: {user}")
        
        # Update user
        updated = await db_manager.update_one("users", {"id": user_id}, {
            "name": "John Smith"
        })
        print(f"Updated {updated} rows")
        
        # Select many users
        users = await db_manager.select_many("users", limit=10)
        print(f"Retrieved {len(users)} users")
        
        # Get performance metrics
        metrics = db_manager.get_performance_metrics()
        print("Performance metrics:", metrics)
        
    finally:
        await db_manager.cleanup()

match __name__:
    case "__main__":
    asyncio.run(example_database_operations()) 