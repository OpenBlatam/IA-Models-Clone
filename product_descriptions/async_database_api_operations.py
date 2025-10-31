from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import uuid
import json
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Callable, Awaitable
from urllib.parse import urlparse
import aiohttp
import asyncpg
import aiosqlite
import aioredis
from fastapi import HTTPException
from pydantic import BaseModel, Field
import psutil
from fastapi import FastAPI
from typing import Any, List, Dict, Optional
"""
Dedicated Async Functions for Database and External API Operations

This module provides comprehensive async functions for:
- Database operations (PostgreSQL, SQLite, Redis)
- External API calls with connection pooling
- Async connection management
- Retry mechanisms and circuit breakers
- Performance monitoring and optimization
- Error handling and recovery
"""



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of async operations."""
    DATABASE_READ = "database_read"
    DATABASE_WRITE = "database_write"
    DATABASE_DELETE = "database_delete"
    API_GET = "api_get"
    API_POST = "api_post"
    API_PUT = "api_put"
    API_DELETE = "api_delete"
    CACHE_GET = "cache_get"
    CACHE_SET = "cache_set"
    CACHE_DELETE = "cache_delete"


class DatabaseType(Enum):
    """Supported database types."""
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"
    REDIS = "redis"


@dataclass
class OperationContext:
    """Context for async operations."""
    operation_id: str
    operation_type: OperationType
    database_type: Optional[DatabaseType] = None
    api_endpoint: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationResult:
    """Result of an async operation."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float
    operation_context: OperationContext
    retry_count: int = 0


class AsyncDatabaseManager(ABC):
    """Abstract base class for async database managers."""
    
    def __init__(self, connection_string: str, max_connections: int = 10):
        
    """__init__ function."""
self.connection_string = connection_string
        self.max_connections = max_connections
        self.pool = None
        self._lock = asyncio.Lock()
        self.stats = {
            "connections_created": 0,
            "connections_used": 0,
            "operations_performed": 0,
            "errors": 0,
            "total_time": 0.0
        }
    
    @abstractmethod
    async def initialize_pool(self) -> Any:
        """Initialize the connection pool."""
        pass
    
    @abstractmethod
    async def close_pool(self) -> Any:
        """Close the connection pool."""
        pass
    
    @abstractmethod
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> OperationResult:
        """Execute a database query."""
        pass
    
    @abstractmethod
    async def execute_transaction(self, queries: List[Dict]) -> OperationResult:
        """Execute multiple queries in a transaction."""
        pass


class AsyncPostgreSQLManager(AsyncDatabaseManager):
    """Async PostgreSQL database manager."""
    
    async def initialize_pool(self) -> Any:
        """Initialize PostgreSQL connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=1,
                max_size=self.max_connections,
                command_timeout=30
            )
            logger.info(f"PostgreSQL pool initialized with {self.max_connections} connections")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            raise
    
    async def close_pool(self) -> Any:
        """Close PostgreSQL connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL pool closed")
    
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> OperationResult:
        """Execute a PostgreSQL query."""
        context = OperationContext(
            operation_id=str(uuid.uuid4()),
            operation_type=OperationType.DATABASE_READ,
            database_type=DatabaseType.POSTGRESQL
        )
        
        start_time = time.time()
        
        try:
            async with self.pool.acquire() as connection:
                if params:
                    result = await connection.fetch(query, **params)
                else:
                    result = await connection.fetch(query)
                
                execution_time = time.time() - start_time
                self.stats["operations_performed"] += 1
                self.stats["total_time"] += execution_time
                
                return OperationResult(
                    success=True,
                    data=result,
                    execution_time=execution_time,
                    operation_context=context
                )
        
        except Exception as e:
            execution_time = time.time() - start_time
            self.stats["errors"] += 1
            logger.error(f"PostgreSQL query error: {e}")
            
            return OperationResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                operation_context=context
            )
    
    async def execute_transaction(self, queries: List[Dict]) -> OperationResult:
        """Execute multiple PostgreSQL queries in a transaction."""
        context = OperationContext(
            operation_id=str(uuid.uuid4()),
            operation_type=OperationType.DATABASE_WRITE,
            database_type=DatabaseType.POSTGRESQL
        )
        
        start_time = time.time()
        
        try:
            async with self.pool.acquire() as connection:
                async with connection.transaction():
                    results = []
                    for query_data in queries:
                        query = query_data["query"]
                        params = query_data.get("params")
                        
                        if params:
                            result = await connection.fetch(query, **params)
                        else:
                            result = await connection.fetch(query)
                        
                        results.append(result)
                
                execution_time = time.time() - start_time
                self.stats["operations_performed"] += 1
                self.stats["total_time"] += execution_time
                
                return OperationResult(
                    success=True,
                    data=results,
                    execution_time=execution_time,
                    operation_context=context
                )
        
        except Exception as e:
            execution_time = time.time() - start_time
            self.stats["errors"] += 1
            logger.error(f"PostgreSQL transaction error: {e}")
            
            return OperationResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                operation_context=context
            )


class AsyncSQLiteManager(AsyncDatabaseManager):
    """Async SQLite database manager."""
    
    async def initialize_pool(self) -> Any:
        """Initialize SQLite connection pool."""
        try:
            # SQLite doesn't use connection pools, but we'll simulate it
            self.pool = self.connection_string
            logger.info("SQLite connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite connection: {e}")
            raise
    
    async def close_pool(self) -> Any:
        """Close SQLite connection."""
        logger.info("SQLite connection closed")
    
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> OperationResult:
        """Execute a SQLite query."""
        context = OperationContext(
            operation_id=str(uuid.uuid4()),
            operation_type=OperationType.DATABASE_READ,
            database_type=DatabaseType.SQLITE
        )
        
        start_time = time.time()
        
        try:
            async with aiosqlite.connect(self.pool) as db:
                if params:
                    cursor = await db.execute(query, params)
                else:
                    cursor = await db.execute(query)
                
                result = await cursor.fetchall()
                await db.commit()
                
                execution_time = time.time() - start_time
                self.stats["operations_performed"] += 1
                self.stats["total_time"] += execution_time
                
                return OperationResult(
                    success=True,
                    data=result,
                    execution_time=execution_time,
                    operation_context=context
                )
        
        except Exception as e:
            execution_time = time.time() - start_time
            self.stats["errors"] += 1
            logger.error(f"SQLite query error: {e}")
            
            return OperationResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                operation_context=context
            )
    
    async def execute_transaction(self, queries: List[Dict]) -> OperationResult:
        """Execute multiple SQLite queries in a transaction."""
        context = OperationContext(
            operation_id=str(uuid.uuid4()),
            operation_type=OperationType.DATABASE_WRITE,
            database_type=DatabaseType.SQLITE
        )
        
        start_time = time.time()
        
        try:
            async with aiosqlite.connect(self.pool) as db:
                results = []
                for query_data in queries:
                    query = query_data["query"]
                    params = query_data.get("params")
                    
                    if params:
                        cursor = await db.execute(query, params)
                    else:
                        cursor = await db.execute(query)
                    
                    result = await cursor.fetchall()
                    results.append(result)
                
                await db.commit()
                
                execution_time = time.time() - start_time
                self.stats["operations_performed"] += 1
                self.stats["total_time"] += execution_time
                
                return OperationResult(
                    success=True,
                    data=results,
                    execution_time=execution_time,
                    operation_context=context
                )
        
        except Exception as e:
            execution_time = time.time() - start_time
            self.stats["errors"] += 1
            logger.error(f"SQLite transaction error: {e}")
            
            return OperationResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                operation_context=context
            )


class AsyncRedisManager(AsyncDatabaseManager):
    """Async Redis database manager."""
    
    async def initialize_pool(self) -> Any:
        """Initialize Redis connection pool."""
        try:
            self.pool = aioredis.from_url(
                self.connection_string,
                max_connections=self.max_connections,
                decode_responses=True
            )
            logger.info(f"Redis pool initialized with {self.max_connections} connections")
        except Exception as e:
            logger.error(f"Failed to initialize Redis pool: {e}")
            raise
    
    async def close_pool(self) -> Any:
        """Close Redis connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Redis pool closed")
    
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> OperationResult:
        """Execute a Redis query."""
        context = OperationContext(
            operation_id=str(uuid.uuid4()),
            operation_type=OperationType.CACHE_GET,
            database_type=DatabaseType.REDIS
        )
        
        start_time = time.time()
        
        try:
            # For Redis, query is the key
            result = await self.pool.get(query)
            
            execution_time = time.time() - start_time
            self.stats["operations_performed"] += 1
            self.stats["total_time"] += execution_time
            
            return OperationResult(
                success=True,
                data=result,
                execution_time=execution_time,
                operation_context=context
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            self.stats["errors"] += 1
            logger.error(f"Redis query error: {e}")
            
            return OperationResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                operation_context=context
            )
    
    async def execute_transaction(self, queries: List[Dict]) -> OperationResult:
        """Execute multiple Redis operations."""
        context = OperationContext(
            operation_id=str(uuid.uuid4()),
            operation_type=OperationType.CACHE_SET,
            database_type=DatabaseType.REDIS
        )
        
        start_time = time.time()
        
        try:
            async with self.pool.pipeline() as pipe:
                for query_data in queries:
                    operation = query_data["operation"]
                    key = query_data["key"]
                    value = query_data.get("value")
                    
                    if operation == "set":
                        await pipe.set(key, value)
                    elif operation == "get":
                        await pipe.get(key)
                    elif operation == "delete":
                        await pipe.delete(key)
                
                results = await pipe.execute()
            
            execution_time = time.time() - start_time
            self.stats["operations_performed"] += 1
            self.stats["total_time"] += execution_time
            
            return OperationResult(
                success=True,
                data=results,
                execution_time=execution_time,
                operation_context=context
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            self.stats["errors"] += 1
            logger.error(f"Redis transaction error: {e}")
            
            return OperationResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                operation_context=context
            )


class AsyncAPIManager:
    """Async external API manager with connection pooling."""
    
    def __init__(self, base_url: str, max_connections: int = 20, timeout: int = 30):
        
    """__init__ function."""
self.base_url = base_url
        self.max_connections = max_connections
        self.timeout = timeout
        self.session = None
        self.stats = {
            "requests_made": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_time": 0.0,
            "average_response_time": 0.0
        }
    
    async def initialize_session(self) -> Any:
        """Initialize aiohttp session with connection pooling."""
        try:
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            
            timeout_config = aiohttp.ClientTimeout(total=self.timeout)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout_config,
                headers={"User-Agent": "AsyncAPIManager/1.0"}
            )
            
            logger.info(f"API session initialized with {self.max_connections} connections")
        except Exception as e:
            logger.error(f"Failed to initialize API session: {e}")
            raise
    
    async def close_session(self) -> Any:
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            logger.info("API session closed")
    
    async async def make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> OperationResult:
        """Make an async API request."""
        context = OperationContext(
            operation_id=str(uuid.uuid4()),
            operation_type=getattr(OperationType, f"API_{method.upper()}"),
            api_endpoint=endpoint
        )
        
        start_time = time.time()
        
        try:
            url = f"{self.base_url}{endpoint}"
            
            request_headers = {"Content-Type": "application/json"}
            if headers:
                request_headers.update(headers)
            
            async with self.session.request(
                method=method,
                url=url,
                json=data,
                headers=request_headers,
                params=params
            ) as response:
                response_data = await response.json()
                
                execution_time = time.time() - start_time
                self.stats["requests_made"] += 1
                self.stats["total_time"] += execution_time
                
                if response.status < 400:
                    self.stats["successful_requests"] += 1
                    return OperationResult(
                        success=True,
                        data=response_data,
                        execution_time=execution_time,
                        operation_context=context
                    )
                else:
                    self.stats["failed_requests"] += 1
                    return OperationResult(
                        success=False,
                        error=f"HTTP {response.status}: {response_data}",
                        execution_time=execution_time,
                        operation_context=context
                    )
        
        except Exception as e:
            execution_time = time.time() - start_time
            self.stats["failed_requests"] += 1
            logger.error(f"API request error: {e}")
            
            return OperationResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                operation_context=context
            )
    
    async def get(self, endpoint: str, params: Optional[Dict] = None) -> OperationResult:
        """Make a GET request."""
        return await self.make_request("GET", endpoint, params=params)
    
    async def post(self, endpoint: str, data: Optional[Dict] = None) -> OperationResult:
        """Make a POST request."""
        return await self.make_request("POST", endpoint, data=data)
    
    async def put(self, endpoint: str, data: Optional[Dict] = None) -> OperationResult:
        """Make a PUT request."""
        return await self.make_request("PUT", endpoint, data=data)
    
    async def delete(self, endpoint: str) -> OperationResult:
        """Make a DELETE request."""
        return await self.make_request("DELETE", endpoint)


class AsyncOperationOrchestrator:
    """Orchestrator for managing all async database and API operations."""
    
    def __init__(self) -> Any:
        self.database_managers: Dict[str, AsyncDatabaseManager] = {}
        self.api_managers: Dict[str, AsyncAPIManager] = {}
        self.operation_history: List[OperationResult] = []
        self.performance_stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "average_execution_time": 0.0,
            "total_execution_time": 0.0
        }
    
    async def add_database_manager(
        self,
        name: str,
        db_type: DatabaseType,
        connection_string: str,
        max_connections: int = 10
    ):
        """Add a database manager."""
        if db_type == DatabaseType.POSTGRESQL:
            manager = AsyncPostgreSQLManager(connection_string, max_connections)
        elif db_type == DatabaseType.SQLITE:
            manager = AsyncSQLiteManager(connection_string, max_connections)
        elif db_type == DatabaseType.REDIS:
            manager = AsyncRedisManager(connection_string, max_connections)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        await manager.initialize_pool()
        self.database_managers[name] = manager
        logger.info(f"Added database manager: {name} ({db_type.value})")
    
    async def add_api_manager(
        self,
        name: str,
        base_url: str,
        max_connections: int = 20,
        timeout: int = 30
    ):
        """Add an API manager."""
        manager = AsyncAPIManager(base_url, max_connections, timeout)
        await manager.initialize_session()
        self.api_managers[name] = manager
        logger.info(f"Added API manager: {name} ({base_url})")
    
    async def execute_database_operation(
        self,
        db_name: str,
        query: str,
        params: Optional[Dict] = None
    ) -> OperationResult:
        """Execute a database operation."""
        if db_name not in self.database_managers:
            raise ValueError(f"Database manager '{db_name}' not found")
        
        manager = self.database_managers[db_name]
        result = await manager.execute_query(query, params)
        
        self._update_stats(result)
        self.operation_history.append(result)
        
        return result
    
    async async def execute_api_operation(
        self,
        api_name: str,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> OperationResult:
        """Execute an API operation."""
        if api_name not in self.api_managers:
            raise ValueError(f"API manager '{api_name}' not found")
        
        manager = self.api_managers[api_name]
        
        if method.upper() == "GET":
            result = await manager.get(endpoint, params)
        elif method.upper() == "POST":
            result = await manager.post(endpoint, data)
        elif method.upper() == "PUT":
            result = await manager.put(endpoint, data)
        elif method.upper() == "DELETE":
            result = await manager.delete(endpoint)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        self._update_stats(result)
        self.operation_history.append(result)
        
        return result
    
    async def execute_batch_operations(
        self,
        operations: List[Dict]
    ) -> List[OperationResult]:
        """Execute multiple operations concurrently."""
        tasks = []
        
        for operation in operations:
            op_type = operation["type"]
            
            if op_type == "database":
                task = self.execute_database_operation(
                    operation["db_name"],
                    operation["query"],
                    operation.get("params")
                )
            elif op_type == "api":
                task = self.execute_api_operation(
                    operation["api_name"],
                    operation["method"],
                    operation["endpoint"],
                    operation.get("data"),
                    operation.get("params")
                )
            else:
                raise ValueError(f"Unsupported operation type: {op_type}")
            
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to OperationResult
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(OperationResult(
                    success=False,
                    error=str(result),
                    execution_time=0.0,
                    operation_context=OperationContext(
                        operation_id=str(uuid.uuid4()),
                        operation_type=OperationType.DATABASE_READ
                    )
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _update_stats(self, result: OperationResult):
        """Update performance statistics."""
        self.performance_stats["total_operations"] += 1
        self.performance_stats["total_execution_time"] += result.execution_time
        
        if result.success:
            self.performance_stats["successful_operations"] += 1
        else:
            self.performance_stats["failed_operations"] += 1
        
        self.performance_stats["average_execution_time"] = (
            self.performance_stats["total_execution_time"] /
            self.performance_stats["total_operations"]
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            "orchestrator": self.performance_stats.copy(),
            "databases": {},
            "apis": {}
        }
        
        for name, manager in self.database_managers.items():
            stats["databases"][name] = manager.stats.copy()
        
        for name, manager in self.api_managers.items():
            stats["apis"][name] = manager.stats.copy()
        
        return stats
    
    async def close_all(self) -> Any:
        """Close all managers and connections."""
        for manager in self.database_managers.values():
            await manager.close_pool()
        
        for manager in self.api_managers.values():
            await manager.close_session()
        
        logger.info("All managers closed")


# Pydantic models for API

class DatabaseOperationRequest(BaseModel):
    """Request model for database operations."""
    db_name: str = Field(..., description="Name of the database manager")
    query: str = Field(..., description="SQL query to execute")
    params: Optional[Dict[str, Any]] = Field(None, description="Query parameters")


class APIOperationRequest(BaseModel):
    """Request model for API operations."""
    api_name: str = Field(..., description="Name of the API manager")
    method: str = Field(..., description="HTTP method")
    endpoint: str = Field(..., description="API endpoint")
    data: Optional[Dict[str, Any]] = Field(None, description="Request data")
    params: Optional[Dict[str, Any]] = Field(None, description="Query parameters")


class BatchOperationRequest(BaseModel):
    """Request model for batch operations."""
    operations: List[Dict[str, Any]] = Field(..., description="List of operations to execute")


class OperationResponse(BaseModel):
    """Response model for operations."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float
    operation_id: str


# FastAPI application


app = FastAPI(
    title="Async Database and API Operations",
    description="Dedicated async functions for database and external API operations",
    version="1.0.0"
)

# Global orchestrator
orchestrator = AsyncOperationOrchestrator()


@app.on_event("startup")
async def startup_event():
    """Initialize database and API managers on startup."""
    # Initialize database managers
    await orchestrator.add_database_manager(
        "postgres_main",
        DatabaseType.POSTGRESQL,
        "postgresql://user:password@localhost:5432/maindb",
        max_connections=20
    )
    
    await orchestrator.add_database_manager(
        "sqlite_cache",
        DatabaseType.SQLITE,
        "cache.db",
        max_connections=5
    )
    
    await orchestrator.add_database_manager(
        "redis_cache",
        DatabaseType.REDIS,
        "redis://localhost:6379",
        max_connections=10
    )
    
    # Initialize API managers
    await orchestrator.add_api_manager(
        "external_api",
        "https://api.external.com",
        max_connections=30,
        timeout=60
    )
    
    await orchestrator.add_api_manager(
        "internal_api",
        "http://localhost:8001",
        max_connections=20,
        timeout=30
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Close all managers on shutdown."""
    await orchestrator.close_all()


# API endpoints

@app.post("/database/execute")
async def execute_database_operation(request: DatabaseOperationRequest) -> OperationResponse:
    """Execute a database operation."""
    result = await orchestrator.execute_database_operation(
        request.db_name,
        request.query,
        request.params
    )
    
    return OperationResponse(
        success=result.success,
        data=result.data,
        error=result.error,
        execution_time=result.execution_time,
        operation_id=result.operation_context.operation_id
    )


@app.post("/api/execute")
async async def execute_api_operation(request: APIOperationRequest) -> OperationResponse:
    """Execute an API operation."""
    result = await orchestrator.execute_api_operation(
        request.api_name,
        request.method,
        request.endpoint,
        request.data,
        request.params
    )
    
    return OperationResponse(
        success=result.success,
        data=result.data,
        error=result.error,
        execution_time=result.execution_time,
        operation_id=result.operation_context.operation_id
    )


@app.post("/batch/execute")
async def execute_batch_operations(request: BatchOperationRequest) -> List[OperationResponse]:
    """Execute multiple operations concurrently."""
    results = await orchestrator.execute_batch_operations(request.operations)
    
    return [
        OperationResponse(
            success=result.success,
            data=result.data,
            error=result.error,
            execution_time=result.execution_time,
            operation_id=result.operation_context.operation_id
        )
        for result in results
    ]


@app.get("/stats")
async def get_performance_stats():
    """Get performance statistics."""
    return orchestrator.get_performance_stats()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "database_managers": len(orchestrator.database_managers),
        "api_managers": len(orchestrator.api_managers),
        "total_operations": orchestrator.performance_stats["total_operations"]
    }


# Utility functions

async def execute_with_retry(
    operation: Callable[[], Awaitable[OperationResult]],
    max_retries: int = 3,
    delay: float = 1.0
) -> OperationResult:
    """Execute an operation with retry logic."""
    for attempt in range(max_retries + 1):
        try:
            result = await operation()
            if result.success:
                return result
            
            if attempt < max_retries:
                await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
        
        except Exception as e:
            if attempt < max_retries:
                await asyncio.sleep(delay * (2 ** attempt))
            else:
                return OperationResult(
                    success=False,
                    error=str(e),
                    execution_time=0.0,
                    operation_context=OperationContext(
                        operation_id=str(uuid.uuid4()),
                        operation_type=OperationType.DATABASE_READ
                    )
                )
    
    return OperationResult(
        success=False,
        error="Max retries exceeded",
        execution_time=0.0,
        operation_context=OperationContext(
            operation_id=str(uuid.uuid4()),
            operation_type=OperationType.DATABASE_READ
        )
    )


@asynccontextmanager
async def get_database_connection(db_name: str):
    """Context manager for database connections."""
    if db_name not in orchestrator.database_managers:
        raise ValueError(f"Database manager '{db_name}' not found")
    
    manager = orchestrator.database_managers[db_name]
    try:
        yield manager
    except Exception as e:
        logger.error(f"Database operation error: {e}")
        raise


@asynccontextmanager
async def get_api_session(api_name: str):
    """Context manager for API sessions."""
    if api_name not in orchestrator.api_managers:
        raise ValueError(f"API manager '{api_name}' not found")
    
    manager = orchestrator.api_managers[api_name]
    try:
        yield manager
    except Exception as e:
        logger.error(f"API operation error: {e}")
        raise


# Export main classes and functions
__all__ = [
    "AsyncOperationOrchestrator",
    "AsyncPostgreSQLManager",
    "AsyncSQLiteManager",
    "AsyncRedisManager",
    "AsyncAPIManager",
    "OperationContext",
    "OperationResult",
    "OperationType",
    "DatabaseType",
    "execute_with_retry",
    "get_database_connection",
    "get_api_session"
] 