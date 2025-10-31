# Async I/O Implementation Summary

## Overview

This document provides a comprehensive overview of the async I/O implementation for the Product Descriptions Feature, focusing on minimizing blocking I/O operations through asynchronous database calls and external API requests.

## Architecture

### Async I/O Stack

The async I/O system is built with multiple layers to ensure non-blocking operations:

1. **Connection Pool Layer** - Manages database and API connection pools
2. **Database Layer** - Async database operations (PostgreSQL, SQLite, Redis)
3. **API Layer** - Async HTTP client operations
4. **Transaction Layer** - Async database transactions
5. **Batch Layer** - Concurrent batch operations
6. **Metrics Layer** - Performance monitoring and tracking
7. **Error Handling Layer** - Comprehensive error handling and retry logic

### Async I/O Flow

```
Request → Connection Pool → Async Operation → Non-blocking I/O → Response
    ↓           ↓              ↓              ↓              ↓
Input Data   Pool Check    Database/API    I/O Completion   Async Response
```

## Components

### 1. Async Connection Pool

**Purpose**: Manages database and API connection pools with automatic lifecycle management.

**Key Features**:
- Connection pooling for multiple database types
- Automatic connection lifecycle management
- SSL/TLS support
- Connection timeout and retry logic
- Metrics collection and monitoring

**Implementation**:
```python
class AsyncConnectionPool:
    """Generic async connection pool"""
    
    def __init__(self, config: ConnectionConfig):
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
                connector = aiohttp.TCPConnector(
                    limit=self.config.pool_size,
                    limit_per_host=self.config.pool_size // 2,
                    ssl=self.config.ssl_context if self.config.ssl_enabled else None,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                )
                self._pool = aiohttp.ClientSession(connector=connector)
```

**Connection Configuration**:
```python
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
```

### 2. Async Database Manager

**Purpose**: Provides async database operations for multiple database types.

**Supported Databases**:
- PostgreSQL (asyncpg)
- SQLite (aiosqlite)
- Redis (aioredis)

**Key Features**:
- Async query execution
- Transaction support
- Connection pooling
- Error handling and retry logic
- Performance metrics

**Implementation**:
```python
class AsyncDatabaseManager:
    """Async database operations manager"""
    
    def __init__(self):
        self.pools: Dict[str, AsyncConnectionPool] = {}
        self._lock = asyncio.Lock()
    
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
```

**Transaction Support**:
```python
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
```

### 3. Async API Manager

**Purpose**: Provides async HTTP client operations for external API requests.

**Key Features**:
- Async HTTP requests
- Session pooling
- SSL/TLS support
- Batch request processing
- Error handling and retry logic
- Performance metrics

**Implementation**:
```python
class AsyncAPIManager:
    """Async external API operations manager"""
    
    def __init__(self):
        self.sessions: Dict[str, aiohttp.ClientSession] = {}
        self._lock = asyncio.Lock()
        self._metrics: List[IOMetrics] = []
    
    async def make_request(
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
```

**Batch Request Processing**:
```python
async def make_batch_requests(
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
```

### 4. Async I/O Decorators

**Purpose**: Provide decorators for timing and retrying async I/O operations.

**Timing Decorator**:
```python
def async_io_timed(operation_name: Optional[str] = None):
    """Decorator for timing async I/O operations"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
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
```

**Retry Decorator**:
```python
def async_io_retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator for retrying async I/O operations"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
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
```

### 5. Main Async I/O Manager

**Purpose**: Central manager for all async I/O operations.

**Implementation**:
```python
class AsyncIOManager:
    """Main async I/O manager"""
    
    def __init__(self):
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
    
    async def initialize_api(
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
    
    async def make_api_request(
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
    
    async def make_batch_api_requests(
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
    
    async def get_api_metrics(self) -> Dict[str, Any]:
        """Get API metrics"""
        return await self.api_manager.get_metrics()
    
    async def close_all(self) -> None:
        """Close all connections"""
        await self.db_manager.close_all()
        await self.api_manager.close_all()
```

## Integration with FastAPI

### Application Setup

```python
from async_io_manager import (
    AsyncIOManager,
    ConnectionConfig,
    ConnectionType,
    OperationType,
    async_io_timed,
    async_io_retry,
    initialize_database_connections,
    initialize_api_sessions,
    cleanup_io_connections
)

# Initialize I/O manager
io_manager = AsyncIOManager()

# Initialize database connections
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

# Initialize API sessions
await io_manager.initialize_api(
    "external_api",
    "https://api.external.com",
    headers={"Authorization": "Bearer token"},
    timeout=30.0,
    ssl_verify=True
)
```

### Route Implementation

```python
@app.get("/users/{user_id}")
@async_io_retry(max_attempts=3, delay=1.0)
@async_io_timed("database.user_query")
async def get_user(user_id: int):
    """Get user by ID with async database query"""
    try:
        query = "SELECT * FROM users WHERE id = $1"
        params = {"id": user_id}
        
        results = await io_manager.execute_query(
            "postgres", query, params, OperationType.QUERY
        )
        
        if not results:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {"success": True, "data": results[0]}
        
    except Exception as e:
        logger.error(f"Failed to get user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/users")
@async_io_timed("database.user_creation")
async def create_user(user_data: Dict[str, Any]):
    """Create user with async database transaction"""
    try:
        queries = [
            {
                "query": "INSERT INTO users (name, email) VALUES ($1, $2) RETURNING id",
                "params": {"name": user_data["name"], "email": user_data["email"]},
                "operation": "query"
            },
            {
                "query": "INSERT INTO profiles (user_id, bio) VALUES ($1, $2)",
                "params": {"user_id": "$1", "bio": user_data.get("bio", "")},
                "operation": "execute"
            }
        ]
        
        results = await io_manager.execute_transaction("postgres", queries)
        
        return {"success": True, "user_id": results[0][0]["id"]}
        
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/external-data/{data_id}")
@async_io_retry(max_attempts=3, delay=1.0)
@async_io_timed("api.external_data")
async def get_external_data(data_id: str):
    """Get external data with async API request"""
    try:
        response = await io_manager.make_api_request(
            "external_api",
            "GET",
            f"/data/{data_id}",
            timeout=15.0
        )
        
        return {"success": True, "data": response["data"]}
        
    except Exception as e:
        logger.error(f"Failed to get external data {data_id}: {e}")
        raise HTTPException(status_code=500, detail="External API error")

@app.post("/batch-external-data")
@async_io_timed("api.batch_external_data")
async def get_batch_external_data(data_ids: List[str]):
    """Get multiple external data items concurrently"""
    try:
        requests = [
            {
                "method": "GET",
                "url": f"/data/{data_id}",
                "timeout": 15.0
            }
            for data_id in data_ids
        ]
        
        results = await io_manager.make_batch_api_requests("external_api", requests)
        
        return {"success": True, "data": results}
        
    except Exception as e:
        logger.error(f"Failed to get batch external data: {e}")
        raise HTTPException(status_code=500, detail="External API error")

@app.get("/io-metrics")
async def get_io_metrics():
    """Get I/O performance metrics"""
    try:
        postgres_metrics = await io_manager.get_database_metrics("postgres")
        api_metrics = await io_manager.get_api_metrics()
        
        return {
            "success": True,
            "database": postgres_metrics,
            "api": api_metrics
        }
        
    except Exception as e:
        logger.error(f"Failed to get I/O metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

## Performance Benefits

### 1. Non-blocking Operations

- **Database Queries**: All database operations are async and non-blocking
- **API Requests**: All HTTP requests are async and non-blocking
- **Connection Pooling**: Efficient connection reuse
- **Concurrent Operations**: Multiple operations can run simultaneously

### 2. Improved Throughput

- **Connection Pooling**: Reduces connection overhead
- **Concurrent Processing**: Handles multiple requests simultaneously
- **Batch Operations**: Processes multiple operations in parallel
- **Efficient Resource Usage**: Better CPU and memory utilization

### 3. Better Error Handling

- **Automatic Retries**: Configurable retry logic with exponential backoff
- **Error Isolation**: Failures don't affect other operations
- **Comprehensive Logging**: Detailed error tracking and monitoring
- **Graceful Degradation**: System continues operating despite failures

### 4. Monitoring and Metrics

- **Performance Tracking**: Detailed timing and success metrics
- **Resource Monitoring**: Connection pool and session statistics
- **Error Tracking**: Comprehensive error logging and analysis
- **Trend Analysis**: Performance trend identification

## Best Practices

### 1. Connection Management

- Use connection pooling for database connections
- Set appropriate pool sizes based on workload
- Implement proper connection cleanup
- Monitor connection usage and performance

### 2. Error Handling

- Implement retry logic for transient failures
- Use exponential backoff for retries
- Log all errors with context
- Implement circuit breaker patterns for external APIs

### 3. Performance Optimization

- Use batch operations for multiple queries
- Implement request caching where appropriate
- Monitor and optimize query performance
- Use appropriate timeouts for all operations

### 4. Monitoring

- Track all I/O operation metrics
- Monitor connection pool performance
- Set up alerts for performance degradation
- Analyze performance trends regularly

### 5. Security

- Use SSL/TLS for all external connections
- Implement proper authentication
- Validate all input data
- Use connection timeouts to prevent hanging connections

## Configuration

### Environment Variables

```bash
# Database configuration
DB_POSTGRES_HOST=localhost
DB_POSTGRES_PORT=5432
DB_POSTGRES_DATABASE=product_descriptions
DB_POSTGRES_USERNAME=postgres
DB_POSTGRES_PASSWORD=password
DB_POSTGRES_POOL_SIZE=10
DB_POSTGRES_TIMEOUT=30.0

# API configuration
API_EXTERNAL_BASE_URL=https://api.external.com
API_EXTERNAL_TIMEOUT=30.0
API_EXTERNAL_SSL_VERIFY=true
API_INTERNAL_BASE_URL=http://localhost:8000
API_INTERNAL_TIMEOUT=10.0
API_INTERNAL_SSL_VERIFY=false

# Redis configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DATABASE=0
REDIS_POOL_SIZE=10
REDIS_TIMEOUT=30.0

# I/O configuration
IO_RETRY_ATTEMPTS=3
IO_RETRY_DELAY=1.0
IO_MAX_CONCURRENT_REQUESTS=100
IO_REQUEST_TIMEOUT=30.0
```

### Customization

Each component can be customized:

```python
# Custom database configuration
custom_postgres_config = ConnectionConfig(
    connection_type=ConnectionType.POSTGRESQL,
    host="custom-host",
    port=5432,
    database="custom_db",
    username="custom_user",
    password="custom_password",
    pool_size=20,
    timeout=60.0,
    ssl_enabled=True
)

# Custom API configuration
await io_manager.initialize_api(
    "custom_api",
    "https://custom-api.com",
    headers={"Custom-Header": "value"},
    timeout=45.0,
    ssl_verify=True
)

# Custom decorators
@async_io_retry(max_attempts=5, delay=2.0)
@async_io_timed("custom_operation")
async def custom_operation():
    # Custom async I/O operation
    pass
```

## Production Considerations

### 1. Performance

- Monitor connection pool performance
- Optimize query performance
- Implement request caching
- Use load balancing for high availability

### 2. Scalability

- Use distributed connection pools
- Implement horizontal scaling
- Monitor resource usage
- Use connection pooling effectively

### 3. Monitoring

- Set up comprehensive monitoring dashboards
- Implement performance alerts
- Track error rates and response times
- Monitor resource usage trends

### 4. Maintenance

- Regular connection pool cleanup
- Monitor and optimize query performance
- Update connection configurations
- Maintain security patches

### 5. Testing

- Load test all async operations
- Test error handling and recovery
- Validate connection pool behavior
- Test concurrent operation limits

## Conclusion

The async I/O implementation provides comprehensive non-blocking operations for database calls and external API requests. It includes connection pooling, error handling, performance monitoring, and batch processing capabilities.

Key benefits:
- **Non-blocking Operations**: All I/O operations are async and non-blocking
- **High Performance**: Connection pooling and concurrent processing
- **Fault Tolerance**: Comprehensive error handling and retry logic
- **Scalability**: Efficient resource usage and connection management
- **Monitoring**: Detailed performance metrics and error tracking
- **Flexibility**: Support for multiple database types and API endpoints
- **Production Ready**: Robust error handling and resource management

The implementation is production-ready and can be extended with additional database types, advanced caching, and distributed connection pooling as needed. 