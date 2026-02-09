from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import aiohttp
import asyncpg
import aiosqlite
import aioredis
import logging
import time
import json
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic
from functools import wraps
from pathlib import Path
from datetime import datetime, timedelta
import ssl
import certifi
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import pickle
from urllib.parse import urlparse
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Async I/O Manager
Product Descriptions Feature - Comprehensive Asynchronous I/O Operations for Database and External APIs
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')
K = TypeVar('K')

class ConnectionType(Enum):
    """Database connection types"""
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"
    REDIS = "redis"
    HTTP = "http"
    WEBSOCKET = "websocket"

class OperationType(Enum):
    """I/O operation types"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    UPDATE = "update"
    QUERY = "query"
    EXECUTE = "execute"

@dataclass
class ConnectionConfig:
    """Connection configuration"""
    connection_type: ConnectionType
    host: str
    port: int
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    ssl_enabled: bool = False
    ssl_context: Optional[ssl.SSLContext] = None
    pool_size: int = 10
    max_overflow: int = 20
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    connection_string: Optional[str] = None

@dataclass
class IOMetrics:
    """I/O operation metrics"""
    operation_type: OperationType
    connection_type: ConnectionType
    duration_ms: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class AsyncConnectionPool:
    """Generic async connection pool"""
    
    def __init__(self, config: ConnectionConfig):
        
    """__init__ function."""
self.config = config
        self._pool = None
        self._lock = asyncio.Lock()
        self._metrics: List[IOMetrics] = []
        self._connection_count = 0
        self._active_connections = 0
    
    async def initialize(self) -> None:
        """Initialize connection pool"""
        async with self._lock:
            if self._pool is not None:
                return
            
            if self.config.connection_type == ConnectionType.POSTGRESQL:
                self._pool = await asyncpg.create_pool(
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.username,
                    password=self.config.password,
                    ssl=self.config.ssl_context if self.config.ssl_enabled else None,
                    min_size=1,
                    max_size=self.config.pool_size,
                    command_timeout=self.config.timeout
                )
            elif self.config.connection_type == ConnectionType.SQLITE:
                # SQLite doesn't use connection pooling, but we'll manage connections
                self._pool = self.config.database
            elif self.config.connection_type == ConnectionType.REDIS:
                self._pool = await aioredis.create_redis_pool(
                    f"redis://{self.config.host}:{self.config.port}",
                    password=self.config.password,
                    db=self.config.database or 0,
                    maxsize=self.config.pool_size,
                    timeout=self.config.timeout
                )
            elif self.config.connection_type == ConnectionType.HTTP:
                # HTTP uses session pooling
                connector = aiohttp.TCPConnector(
                    limit=self.config.pool_size,
                    limit_per_host=self.config.pool_size // 2,
                    ssl=self.config.ssl_context if self.config.ssl_enabled else None,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                )
                self._pool = aiohttp.ClientSession(connector=connector)
            
            logger.info(f"Initialized {self.config.connection_type.value} connection pool")
    
    async def get_connection(self) -> Optional[Dict[str, Any]]:
        """Get connection from pool"""
        if self._pool is None:
            await self.initialize()
        
        self._active_connections += 1
        return self._pool
    
    async def release_connection(self, connection) -> None:
        """Release connection back to pool"""
        self._active_connections -= 1
    
    async def close(self) -> None:
        """Close connection pool"""
        async with self._lock:
            if self._pool is not None:
                if self.config.connection_type == ConnectionType.HTTP:
                    await self._pool.close()
                elif self.config.connection_type == ConnectionType.REDIS:
                    self._pool.close()
                    await self._pool.wait_closed()
                elif self.config.connection_type == ConnectionType.POSTGRESQL:
                    await self._pool.close()
                
                self._pool = None
                logger.info(f"Closed {self.config.connection_type.value} connection pool")
    
    async def record_metric(self, metric: IOMetrics) -> None:
        """Record I/O metric"""
        self._metrics.append(metric)
        # Keep only last 1000 metrics
        if len(self._metrics) > 1000:
            self._metrics = self._metrics[-1000:]
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get connection pool metrics"""
        return {
            "connection_type": self.config.connection_type.value,
            "pool_size": self.config.pool_size,
            "active_connections": self._active_connections,
            "total_operations": len(self._metrics),
            "success_rate": sum(1 for m in self._metrics if m.success) / len(self._metrics) if self._metrics else 0,
            "avg_duration_ms": sum(m.duration_ms for m in self._metrics) / len(self._metrics) if self._metrics else 0
        }

class AsyncDatabaseManager:
    """Async database operations manager"""
    
    def __init__(self) -> Any:
        self.pools: Dict[str, AsyncConnectionPool] = {}
        self._lock = asyncio.Lock()
    
    async def add_connection(self, name: str, config: ConnectionConfig) -> None:
        """Add database connection"""
        async with self._lock:
            pool = AsyncConnectionPool(config)
            await pool.initialize()
            self.pools[name] = pool
            logger.info(f"Added database connection: {name}")
    
    async def get_pool(self, name: str) -> AsyncConnectionPool:
        """Get connection pool by name"""
        if name not in self.pools:
            raise ValueError(f"Database connection '{name}' not found")
        return self.pools[name]
    
    async def execute_query(
        self, 
        connection_name: str, 
        query: str, 
        params: Optional[Dict[str, Any]] = None,
        operation_type: OperationType = OperationType.QUERY
    ) -> List[Dict[str, Any]]:
        """Execute database query asynchronously"""
        pool = await self.get_pool(connection_name)
        start_time = time.time()
        
        try:
            connection = await pool.get_connection()
            
            if pool.config.connection_type == ConnectionType.POSTGRESQL:
                async with connection.acquire() as conn:
                    if operation_type == OperationType.QUERY:
                        rows = await conn.fetch(query, *(params or {}).values())
                        result = [dict(row) for row in rows]
                    else:
                        result = await conn.execute(query, *(params or {}).values())
            
            elif pool.config.connection_type == ConnectionType.SQLITE:
                async with aiosqlite.connect(connection) as db:
                    if operation_type == OperationType.QUERY:
                        async with db.execute(query, params or {}) as cursor:
                            rows = await cursor.fetchall()
                            columns = [description[0] for description in cursor.description]
                            result = [dict(zip(columns, row)) for row in rows]
                    else:
                        await db.execute(query, params or {})
                        await db.commit()
                        result = []
            
            elif pool.config.connection_type == ConnectionType.REDIS:
                if operation_type == OperationType.QUERY:
                    result = await connection.get(query)
                    result = [{"value": result}] if result else []
                else:
                    result = await connection.set(query, params.get("value", ""))
                    result = [{"result": result}]
            
            await pool.release_connection(connection)
            
            # Record success metric
            duration_ms = (time.time() - start_time) * 1000
            await pool.record_metric(IOMetrics(
                operation_type=operation_type,
                connection_type=pool.config.connection_type,
                duration_ms=duration_ms,
                success=True,
                metadata={"query": query, "params": params}
            ))
            
            return result
            
        except Exception as e:
            # Record error metric
            duration_ms = (time.time() - start_time) * 1000
            await pool.record_metric(IOMetrics(
                operation_type=operation_type,
                connection_type=pool.config.connection_type,
                duration_ms=duration_ms,
                success=False,
                error_message=str(e),
                metadata={"query": query, "params": params}
            ))
            
            logger.error(f"Database query failed: {e}")
            raise
    
    async def execute_transaction(
        self, 
        connection_name: str, 
        queries: List[Dict[str, Any]]
    ) -> List[Any]:
        """Execute database transaction asynchronously"""
        pool = await self.get_pool(connection_name)
        start_time = time.time()
        
        try:
            connection = await pool.get_connection()
            results = []
            
            if pool.config.connection_type == ConnectionType.POSTGRESQL:
                async with connection.acquire() as conn:
                    async with conn.transaction():
                        for query_info in queries:
                            query = query_info["query"]
                            params = query_info.get("params", {})
                            operation_type = OperationType(query_info.get("operation", "execute"))
                            
                            if operation_type == OperationType.QUERY:
                                rows = await conn.fetch(query, *(params or {}).values())
                                results.append([dict(row) for row in rows])
                            else:
                                result = await conn.execute(query, *(params or {}).values())
                                results.append(result)
            
            elif pool.config.connection_type == ConnectionType.SQLITE:
                async with aiosqlite.connect(connection) as db:
                    async with db.execute("BEGIN TRANSACTION"):
                        for query_info in queries:
                            query = query_info["query"]
                            params = query_info.get("params", {})
                            operation_type = OperationType(query_info.get("operation", "execute"))
                            
                            if operation_type == OperationType.QUERY:
                                async with db.execute(query, params or {}) as cursor:
                                    rows = await cursor.fetchall()
                                    columns = [description[0] for description in cursor.description]
                                    results.append([dict(zip(columns, row)) for row in rows])
                            else:
                                await db.execute(query, params or {})
                                results.append(None)
                        
                        await db.commit()
            
            await pool.release_connection(connection)
            
            # Record success metric
            duration_ms = (time.time() - start_time) * 1000
            await pool.record_metric(IOMetrics(
                operation_type=OperationType.EXECUTE,
                connection_type=pool.config.connection_type,
                duration_ms=duration_ms,
                success=True,
                metadata={"query_count": len(queries)}
            ))
            
            return results
            
        except Exception as e:
            # Record error metric
            duration_ms = (time.time() - start_time) * 1000
            await pool.record_metric(IOMetrics(
                operation_type=OperationType.EXECUTE,
                connection_type=pool.config.connection_type,
                duration_ms=duration_ms,
                success=False,
                error_message=str(e),
                metadata={"query_count": len(queries)}
            ))
            
            logger.error(f"Database transaction failed: {e}")
            raise
    
    async def close_all(self) -> None:
        """Close all database connections"""
        async with self._lock:
            for name, pool in self.pools.items():
                await pool.close()
            self.pools.clear()
            logger.info("Closed all database connections")

class AsyncAPIManager:
    """Async external API operations manager"""
    
    def __init__(self) -> Any:
        self.sessions: Dict[str, aiohttp.ClientSession] = {}
        self._lock = asyncio.Lock()
        self._metrics: List[IOMetrics] = []
    
    async def create_session(
        self, 
        name: str, 
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        ssl_verify: bool = True
    ) -> None:
        """Create HTTP session"""
        async with self._lock:
            if name in self.sessions:
                return
            
            # Create SSL context if needed
            ssl_context = None
            if ssl_verify:
                ssl_context = ssl.create_default_context(cafile=certifi.where())
            
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=20,
                ssl=ssl_context,
                timeout=aiohttp.ClientTimeout(total=timeout)
            )
            
            session = aiohttp.ClientSession(
                base_url=base_url,
                headers=headers or {},
                connector=connector
            )
            
            self.sessions[name] = session
            logger.info(f"Created API session: {name}")
    
    async async def make_request(
        self,
        session_name: str,
        method: str,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Make async HTTP request"""
        if session_name not in self.sessions:
            raise ValueError(f"API session '{session_name}' not found")
        
        session = self.sessions[session_name]
        start_time = time.time()
        
        try:
            async with session.request(
                method=method,
                url=url,
                json=data,
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=timeout or 30.0)
            ) as response:
                response_data = await response.json()
                
                # Record success metric
                duration_ms = (time.time() - start_time) * 1000
                await self._record_metric(IOMetrics(
                    operation_type=OperationType.READ if method.upper() == "GET" else OperationType.WRITE,
                    connection_type=ConnectionType.HTTP,
                    duration_ms=duration_ms,
                    success=response.status < 400,
                    metadata={
                        "method": method,
                        "url": url,
                        "status_code": response.status,
                        "response_size": len(str(response_data))
                    }
                ))
                
                return {
                    "status_code": response.status,
                    "data": response_data,
                    "headers": dict(response.headers)
                }
                
        except Exception as e:
            # Record error metric
            duration_ms = (time.time() - start_time) * 1000
            await self._record_metric(IOMetrics(
                operation_type=OperationType.READ if method.upper() == "GET" else OperationType.WRITE,
                connection_type=ConnectionType.HTTP,
                duration_ms=duration_ms,
                success=False,
                error_message=str(e),
                metadata={"method": method, "url": url}
            ))
            
            logger.error(f"API request failed: {e}")
            raise
    
    async async def make_batch_requests(
        self,
        session_name: str,
        requests: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Make multiple async HTTP requests concurrently"""
        if session_name not in self.sessions:
            raise ValueError(f"API session '{session_name}' not found")
        
        session = self.sessions[session_name]
        start_time = time.time()
        
        try:
            # Create tasks for all requests
            tasks = []
            for req in requests:
                task = session.request(
                    method=req["method"],
                    url=req["url"],
                    json=req.get("data"),
                    headers=req.get("headers"),
                    params=req.get("params"),
                    timeout=aiohttp.ClientTimeout(total=req.get("timeout", 30.0))
                )
                tasks.append(task)
            
            # Execute all requests concurrently
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process responses
            results = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    results.append({
                        "status_code": None,
                        "data": None,
                        "error": str(response)
                    })
                else:
                    try:
                        response_data = await response.json()
                        results.append({
                            "status_code": response.status,
                            "data": response_data,
                            "headers": dict(response.headers)
                        })
                    except Exception as e:
                        results.append({
                            "status_code": response.status,
                            "data": None,
                            "error": str(e)
                        })
            
            # Record batch metric
            duration_ms = (time.time() - start_time) * 1000
            success_count = sum(1 for r in results if r.get("status_code", 0) < 400)
            
            await self._record_metric(IOMetrics(
                operation_type=OperationType.EXECUTE,
                connection_type=ConnectionType.HTTP,
                duration_ms=duration_ms,
                success=success_count == len(requests),
                metadata={
                    "request_count": len(requests),
                    "success_count": success_count,
                    "failure_count": len(requests) - success_count
                }
            ))
            
            return results
            
        except Exception as e:
            # Record error metric
            duration_ms = (time.time() - start_time) * 1000
            await self._record_metric(IOMetrics(
                operation_type=OperationType.EXECUTE,
                connection_type=ConnectionType.HTTP,
                duration_ms=duration_ms,
                success=False,
                error_message=str(e),
                metadata={"request_count": len(requests)}
            ))
            
            logger.error(f"Batch API requests failed: {e}")
            raise
    
    async def _record_metric(self, metric: IOMetrics) -> None:
        """Record API metric"""
        self._metrics.append(metric)
        # Keep only last 1000 metrics
        if len(self._metrics) > 1000:
            self._metrics = self._metrics[-1000:]
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get API metrics"""
        if not self._metrics:
            return {"total_requests": 0, "success_rate": 0, "avg_duration_ms": 0}
        
        return {
            "total_requests": len(self._metrics),
            "success_rate": sum(1 for m in self._metrics if m.success) / len(self._metrics),
            "avg_duration_ms": sum(m.duration_ms for m in self._metrics) / len(self._metrics),
            "requests_by_method": {
                method: len([m for m in self._metrics if m.metadata.get("method") == method])
                for method in set(m.metadata.get("method") for m in self._metrics if m.metadata.get("method"))
            }
        }
    
    async def close_all(self) -> None:
        """Close all API sessions"""
        async with self._lock:
            for name, session in self.sessions.items():
                await session.close()
            self.sessions.clear()
            logger.info("Closed all API sessions")

class AsyncIOManager:
    """Main async I/O manager"""
    
    def __init__(self) -> Any:
        self.db_manager = AsyncDatabaseManager()
        self.api_manager = AsyncAPIManager()
        self._lock = asyncio.Lock()
    
    async def initialize_database(
        self, 
        name: str, 
        config: ConnectionConfig
    ) -> None:
        """Initialize database connection"""
        await self.db_manager.add_connection(name, config)
    
    async async def initialize_api(
        self, 
        name: str, 
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        ssl_verify: bool = True
    ) -> None:
        """Initialize API session"""
        await self.api_manager.create_session(name, base_url, headers, timeout, ssl_verify)
    
    async def execute_query(
        self, 
        connection_name: str, 
        query: str, 
        params: Optional[Dict[str, Any]] = None,
        operation_type: OperationType = OperationType.QUERY
    ) -> List[Dict[str, Any]]:
        """Execute database query"""
        return await self.db_manager.execute_query(connection_name, query, params, operation_type)
    
    async def execute_transaction(
        self, 
        connection_name: str, 
        queries: List[Dict[str, Any]]
    ) -> List[Any]:
        """Execute database transaction"""
        return await self.db_manager.execute_transaction(connection_name, queries)
    
    async async def make_api_request(
        self,
        session_name: str,
        method: str,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Make API request"""
        return await self.api_manager.make_request(
            session_name, method, url, data, headers, params, timeout
        )
    
    async async def make_batch_api_requests(
        self,
        session_name: str,
        requests: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Make batch API requests"""
        return await self.api_manager.make_batch_requests(session_name, requests)
    
    async def get_database_metrics(self, connection_name: str) -> Dict[str, Any]:
        """Get database metrics"""
        pool = await self.db_manager.get_pool(connection_name)
        return await pool.get_metrics()
    
    async async def get_api_metrics(self) -> Dict[str, Any]:
        """Get API metrics"""
        return await self.api_manager.get_metrics()
    
    async def close_all(self) -> None:
        """Close all connections"""
        await self.db_manager.close_all()
        await self.api_manager.close_all()

# Global I/O manager instance
io_manager = AsyncIOManager()

# Decorators for async I/O operations
def async_io_timed(operation_name: Optional[str] = None):
    """Decorator for timing async I/O operations"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"Async I/O operation '{op_name}' completed in {duration_ms:.2f}ms")
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(f"Async I/O operation '{op_name}' failed after {duration_ms:.2f}ms: {e}")
                raise
        
        return wrapper
    return decorator

def async_io_retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator for retrying async I/O operations"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay * (attempt + 1))
                        logger.warning(f"Async I/O operation failed (attempt {attempt + 1}/{max_attempts}): {e}")
            
            logger.error(f"Async I/O operation failed after {max_attempts} attempts: {last_exception}")
            raise last_exception
        
        return wrapper
    return decorator

# Utility functions
@async_io_timed("database.initialize")
async def initialize_database_connections() -> None:
    """Initialize common database connections"""
    # PostgreSQL
    postgres_config = ConnectionConfig(
        connection_type=ConnectionType.POSTGRESQL,
        host="localhost",
        port=5432,
        database="product_descriptions",
        username="postgres",
        password="password",
        pool_size=10,
        timeout=30.0
    )
    await io_manager.initialize_database("postgres", postgres_config)
    
    # SQLite
    sqlite_config = ConnectionConfig(
        connection_type=ConnectionType.SQLITE,
        host="",
        port=0,
        database="test_data/product_descriptions.db",
        pool_size=1,
        timeout=30.0
    )
    await io_manager.initialize_database("sqlite", sqlite_config)
    
    # Redis
    redis_config = ConnectionConfig(
        connection_type=ConnectionType.REDIS,
        host="localhost",
        port=6379,
        database=0,
        pool_size=10,
        timeout=30.0
    )
    await io_manager.initialize_database("redis", redis_config)

@async_io_timed("api.initialize")
async async def initialize_api_sessions() -> None:
    """Initialize common API sessions"""
    # External API
    await io_manager.initialize_api(
        "external_api",
        "https://api.external.com",
        headers={"Authorization": "Bearer token", "Content-Type": "application/json"},
        timeout=30.0,
        ssl_verify=True
    )
    
    # Internal API
    await io_manager.initialize_api(
        "internal_api",
        "http://localhost:8000",
        headers={"Content-Type": "application/json"},
        timeout=10.0,
        ssl_verify=False
    )

@async_io_timed("io.cleanup")
async def cleanup_io_connections() -> None:
    """Cleanup all I/O connections"""
    await io_manager.close_all()

# Example usage functions
@async_io_retry(max_attempts=3, delay=1.0)
@async_io_timed("database.user_query")
async def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Get user by ID from database"""
    query = "SELECT * FROM users WHERE id = $1"
    params = {"id": user_id}
    
    results = await io_manager.execute_query("postgres", query, params, OperationType.QUERY)
    return results[0] if results else None

@async_io_retry(max_attempts=3, delay=1.0)
@async_io_timed("api.external_data")
async async def fetch_external_data(data_id: str) -> Dict[str, Any]:
    """Fetch data from external API"""
    return await io_manager.make_api_request(
        "external_api",
        "GET",
        f"/data/{data_id}",
        timeout=15.0
    )

@async_io_timed("database.batch_operations")
async def create_user_with_profile(user_data: Dict[str, Any], profile_data: Dict[str, Any]) -> List[Any]:
    """Create user with profile in transaction"""
    queries = [
        {
            "query": "INSERT INTO users (name, email) VALUES ($1, $2) RETURNING id",
            "params": {"name": user_data["name"], "email": user_data["email"]},
            "operation": "query"
        },
        {
            "query": "INSERT INTO profiles (user_id, bio, avatar) VALUES ($1, $2, $3)",
            "params": {"user_id": "$1", "bio": profile_data["bio"], "avatar": profile_data["avatar"]},
            "operation": "execute"
        }
    ]
    
    return await io_manager.execute_transaction("postgres", queries)

@async_io_timed("api.batch_requests")
async async def fetch_multiple_external_data(data_ids: List[str]) -> List[Dict[str, Any]]:
    """Fetch multiple data items from external API concurrently"""
    requests = [
        {
            "method": "GET",
            "url": f"/data/{data_id}",
            "timeout": 15.0
        }
        for data_id in data_ids
    ]
    
    return await io_manager.make_batch_api_requests("external_api", requests)

if __name__ == "__main__":
    async def main():
        """Example usage"""
        try:
            # Initialize connections
            await initialize_database_connections()
            await initialize_api_sessions()
            
            # Example operations
            user = await get_user_by_id(1)
            print(f"User: {user}")
            
            external_data = await fetch_external_data("test123")
            print(f"External data: {external_data}")
            
            # Get metrics
            db_metrics = await io_manager.get_database_metrics("postgres")
            api_metrics = await io_manager.get_api_metrics()
            
            print(f"Database metrics: {db_metrics}")
            print(f"API metrics: {api_metrics}")
            
        finally:
            # Cleanup
            await cleanup_io_connections()
    
    asyncio.run(main()) 