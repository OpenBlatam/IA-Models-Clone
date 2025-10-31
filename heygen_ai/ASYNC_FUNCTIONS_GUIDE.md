# Async Functions Guide

A comprehensive guide to using dedicated async functions for database and external API operations in the HeyGen AI FastAPI application to ensure non-blocking flows.

## 🎯 Overview

This guide covers the complete async functions system designed to:
- **Database Operations**: Dedicated async functions for all database operations
- **External API Operations**: Async functions for external API calls
- **Connection Pooling**: Efficient connection management
- **Retry Logic**: Automatic retry mechanisms with exponential backoff
- **Caching**: Response caching for performance optimization
- **Circuit Breaker**: Fault tolerance patterns
- **Rate Limiting**: Request rate limiting and throttling
- **Monitoring**: Comprehensive performance monitoring
- **Error Handling**: Proper async error handling patterns

## 📋 Table of Contents

1. [System Architecture](#system-architecture)
2. [Async Database Operations](#async-database-operations)
3. [Async External API Operations](#async-external-api-operations)
4. [Connection Pooling](#connection-pooling)
5. [Retry Logic](#retry-logic)
6. [Caching Strategies](#caching-strategies)
7. [Circuit Breaker Pattern](#circuit-breaker-pattern)
8. [Rate Limiting](#rate-limiting)
9. [Performance Monitoring](#performance-monitoring)
10. [Error Handling](#error-handling)
11. [Best Practices](#best-practices)
12. [Integration Examples](#integration-examples)
13. [Troubleshooting](#troubleshooting)

## 🏗️ System Architecture

### **Async Functions Architecture**

```
FastAPI Application
├── Async Database Manager (Connection pooling & optimization)
├── Async Database Operations (CRUD operations)
├── Async External API Manager (HTTP client management)
├── Async External API Operations (API calls)
├── Connection Pooling (Resource management)
├── Retry Logic (Fault tolerance)
├── Caching Layer (Performance optimization)
├── Circuit Breaker (Fault tolerance)
├── Rate Limiting (Request throttling)
└── Monitoring & Analytics (Performance tracking)
```

### **Core Components**

1. **AsyncDatabaseManager**: Main database connection manager
2. **AsyncDatabaseOperations**: Dedicated async database functions
3. **AsyncExternalAPIManager**: External API connection manager
4. **AsyncExternalAPIOperations**: Dedicated async API functions
5. **Connection Pooling**: Efficient resource management
6. **Retry Logic**: Automatic retry mechanisms
7. **Caching**: Response and query caching
8. **Circuit Breaker**: Fault tolerance patterns

## 🗄️ Async Database Operations

### **1. Database Manager Setup**

```python
from api.async_operations.async_database import AsyncDatabaseManager, DatabaseConfig, DatabaseType

# Configure database
db_config = DatabaseConfig(
    database_type=DatabaseType.POSTGRESQL,
    host="localhost",
    port=5432,
    database="heygen_ai",
    username="postgres",
    password="password",
    pool_size=20,
    max_overflow=30,
    pool_timeout=30.0,
    connection_timeout=10.0,
    query_timeout=30.0,
    enable_query_cache=True,
    query_cache_ttl=300,
    max_retries=3,
    retry_delay=1.0
)

# Initialize database manager
db_manager = AsyncDatabaseManager(db_config)
await db_manager.initialize()
```

### **2. Database Operations Setup**

```python
from api.async_operations.async_database import AsyncDatabaseOperations

# Initialize database operations
db_ops = AsyncDatabaseOperations(db_manager)

# FastAPI integration
@app.on_event("startup")
async def startup_event():
    await db_manager.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    await db_manager.cleanup()
```

### **3. CRUD Operations**

```python
# SELECT operations
@app.get("/users")
async def get_users():
    users = await db_ops.execute_query(
        "SELECT * FROM users WHERE active = true",
        timeout=10.0
    )
    return users

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    users = await db_ops.execute_query(
        "SELECT * FROM users WHERE id = :user_id",
        parameters={"user_id": user_id},
        timeout=5.0
    )
    return users[0] if users else None

# INSERT operations
@app.post("/users")
async def create_user(user_data: UserCreate):
    result = await db_ops.execute_insert(
        table="users",
        data=user_data.dict(),
        returning="id, name, email"
    )
    return result

# UPDATE operations
@app.put("/users/{user_id}")
async def update_user(user_id: int, user_data: UserUpdate):
    result = await db_ops.execute_update(
        table="users",
        data=user_data.dict(exclude_unset=True),
        where_conditions={"id": user_id},
        returning="id, name, email"
    )
    return result

# DELETE operations
@app.delete("/users/{user_id}")
async def delete_user(user_id: int):
    result = await db_ops.execute_delete(
        table="users",
        where_conditions={"id": user_id},
        returning="id"
    )
    return {"deleted": True}
```

### **4. Advanced Database Operations**

```python
# Batch operations
@app.post("/users/batch")
async def create_users_batch(users: List[UserCreate]):
    operations = [
        {
            "type": QueryType.INSERT,
            "table": "users",
            "data": user.dict()
        }
        for user in users
    ]
    results = await db_ops.execute_batch_operations(operations, batch_size=50)
    return results

# Transactions
@app.post("/transfer")
async def transfer_money(from_account: int, to_account: int, amount: float):
    queries = [
        {
            "type": QueryType.UPDATE,
            "query": "UPDATE accounts SET balance = balance - :amount WHERE id = :account_id",
            "parameters": {"amount": amount, "account_id": from_account}
        },
        {
            "type": QueryType.UPDATE,
            "query": "UPDATE accounts SET balance = balance + :amount WHERE id = :account_id",
            "parameters": {"amount": amount, "account_id": to_account}
        }
    ]
    results = await db_ops.execute_transaction(queries)
    return {"success": True}

# Streaming results
@app.get("/users/stream")
async def stream_users():
    return StreamingResponse(
        db_ops.stream_query_results("SELECT * FROM users"),
        media_type="application/json"
    )

# Cached queries
@app.get("/cached-users")
async def get_cached_users():
    users = await db_ops.execute_query(
        "SELECT * FROM users WHERE active = true",
        cache_key="active_users",
        cache_ttl=300
    )
    return users
```

### **5. Redis Operations**

```python
from api.async_operations.async_database import AsyncRedisOperations

# Initialize Redis operations
redis_ops = AsyncRedisOperations("redis://localhost:6379")
await redis_ops.initialize()

# Redis operations
@app.get("/cache/{key}")
async def get_cached_data(key: str):
    data = await redis_ops.get(key)
    return {"data": data}

@app.post("/cache/{key}")
async def set_cached_data(key: str, data: Dict[str, Any]):
    success = await redis_ops.set(key, data, ttl=300)
    return {"success": success}

@app.delete("/cache/{key}")
async def delete_cached_data(key: str):
    success = await redis_ops.delete(key)
    return {"success": success}

# Get or set pattern
@app.get("/user-profile/{user_id}")
async def get_user_profile(user_id: int):
    return await redis_ops.get_or_set(
        f"user_profile:{user_id}",
        lambda: fetch_user_profile_from_db(user_id),
        ttl=600
    )
```

## 🌐 Async External API Operations

### **1. External API Manager Setup**

```python
from api.async_operations.async_external_api import AsyncExternalAPIManager, APIConfig, APIType

# Configure external API
api_config = APIConfig(
    base_url="https://api.external.com",
    api_type=APIType.REST,
    timeout=30.0,
    max_retries=3,
    retry_delay=1.0,
    retry_backoff=2.0,
    max_connections=100,
    max_connections_per_host=10,
    connection_timeout=10.0,
    enable_ssl_verification=True,
    enable_compression=True,
    enable_keepalive=True,
    enable_caching=True,
    cache_ttl=300,
    enable_rate_limiting=True,
    rate_limit=100,
    enable_circuit_breaker=True,
    circuit_breaker_threshold=5,
    circuit_breaker_timeout=60.0,
    auth_token="your_auth_token",
    api_key="your_api_key"
)

# Initialize API manager
api_manager = AsyncExternalAPIManager(api_config)
await api_manager.initialize()
```

### **2. External API Operations Setup**

```python
from api.async_operations.async_external_api import AsyncExternalAPIOperations, APIMethod

# Initialize API operations
api_ops = AsyncExternalAPIOperations(api_manager)

# FastAPI integration
@app.on_event("startup")
async def startup_event():
    await api_manager.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    await api_manager.cleanup()
```

### **3. HTTP Operations**

```python
# GET requests
@app.get("/external-data")
async def get_external_data():
    response = await api_ops.get(
        endpoint="/data",
        params={"limit": 100},
        cache_key="external_data",
        cache_ttl=300
    )
    return response

# POST requests
@app.post("/external-create")
async def create_external_resource(data: Dict[str, Any]):
    response = await api_ops.post(
        endpoint="/resources",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    return response

# PUT requests
@app.put("/external-update/{resource_id}")
async def update_external_resource(resource_id: str, data: Dict[str, Any]):
    response = await api_ops.put(
        endpoint=f"/resources/{resource_id}",
        data=data
    )
    return response

# PATCH requests
@app.patch("/external-partial-update/{resource_id}")
async def partial_update_external_resource(resource_id: str, data: Dict[str, Any]):
    response = await api_ops.patch(
        endpoint=f"/resources/{resource_id}",
        data=data
    )
    return response

# DELETE requests
@app.delete("/external-delete/{resource_id}")
async def delete_external_resource(resource_id: str):
    response = await api_ops.delete(
        endpoint=f"/resources/{resource_id}"
    )
    return response
```

### **4. Advanced API Operations**

```python
# Batch requests
@app.post("/external-batch")
async def make_batch_requests(requests: List[Dict[str, Any]]):
    results = await api_ops.make_batch_requests(requests, max_concurrent=10)
    return results

# Streaming responses
@app.get("/external-stream")
async def stream_external_data():
    return StreamingResponse(
        api_ops.stream_request(APIMethod.GET, "/stream-data"),
        media_type="application/octet-stream"
    )

# File upload
@app.post("/external-upload")
async def upload_file_to_external_api(file: UploadFile):
    # Save uploaded file temporarily
    temp_path = f"/tmp/{file.filename}"
    async with aiofiles.open(temp_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    try:
        response = await api_ops.upload_file(
            endpoint="/upload",
            file_path=temp_path,
            file_field="file",
            additional_data={"description": "Uploaded file"}
        )
        return response
    finally:
        # Clean up temporary file
        os.remove(temp_path)

# File download
@app.get("/external-download/{file_id}")
async def download_file_from_external_api(file_id: str):
    save_path = f"/downloads/{file_id}.pdf"
    response = await api_ops.download_file(
        endpoint=f"/files/{file_id}",
        save_path=save_path
    )
    return response
```

## 🔗 Connection Pooling

### **1. Database Connection Pooling**

```python
# Database connection pooling configuration
db_config = DatabaseConfig(
    pool_size=20,  # Maximum number of connections in pool
    max_overflow=30,  # Additional connections when pool is full
    pool_timeout=30.0,  # Timeout for getting connection from pool
    pool_recycle=3600,  # Recycle connections after 1 hour
    connection_timeout=10.0  # Connection establishment timeout
)

# Connection pool monitoring
@app.get("/database/pool-status")
async def get_database_pool_status():
    metrics = db_manager.get_connection_metrics()
    return {
        "pool_size": metrics["pool"]["size"],
        "checked_in": metrics["pool"]["checked_in"],
        "checked_out": metrics["pool"]["checked_out"],
        "overflow": metrics["pool"]["overflow"],
        "invalid": metrics["pool"]["invalid"]
    }
```

### **2. External API Connection Pooling**

```python
# External API connection pooling configuration
api_config = APIConfig(
    max_connections=100,  # Maximum total connections
    max_connections_per_host=10,  # Maximum connections per host
    connection_timeout=10.0,  # Connection establishment timeout
    keepalive_timeout=30.0  # Keep-alive timeout
)

# Connection pool monitoring
@app.get("/api/pool-status")
async def get_api_pool_status():
    metrics = api_manager.get_connection_metrics()
    return metrics
```

## 🔄 Retry Logic

### **1. Database Retry Logic**

```python
# Database retry configuration
db_config = DatabaseConfig(
    max_retries=3,
    retry_delay=1.0,
    query_timeout=30.0
)

# Retry logic is automatically applied to all database operations
@app.get("/users-with-retry")
async def get_users_with_retry():
    # This will automatically retry on failure
    users = await db_ops.execute_query(
        "SELECT * FROM users",
        timeout=10.0
    )
    return users
```

### **2. External API Retry Logic**

```python
# External API retry configuration
api_config = APIConfig(
    max_retries=3,
    retry_delay=1.0,
    retry_backoff=2.0  # Exponential backoff
)

# Retry logic is automatically applied to all API operations
@app.get("/external-data-with-retry")
async def get_external_data_with_retry():
    # This will automatically retry on failure with exponential backoff
    response = await api_ops.get(
        endpoint="/data",
        retry_on_failure=True
    )
    return response
```

## 💾 Caching Strategies

### **1. Database Query Caching**

```python
# Database query caching
@app.get("/cached-users")
async def get_cached_users():
    users = await db_ops.execute_query(
        "SELECT * FROM users WHERE active = true",
        cache_key="active_users",
        cache_ttl=300  # Cache for 5 minutes
    )
    return users

# Cache invalidation
@app.post("/users")
async def create_user(user_data: UserCreate):
    # Create user
    result = await db_ops.execute_insert("users", user_data.dict())
    
    # Invalidate cache
    await db_ops._cache_query_result("active_users", None, ttl=0)
    
    return result
```

### **2. External API Response Caching**

```python
# External API response caching
@app.get("/cached-external-data")
async def get_cached_external_data():
    response = await api_ops.get(
        endpoint="/data",
        cache_key="external_data",
        cache_ttl=600  # Cache for 10 minutes
    )
    return response

# Cache statistics
@app.get("/cache/stats")
async def get_cache_stats():
    db_cache_stats = db_ops.get_cache_stats()
    api_cache_stats = api_ops.get_cache_stats()
    
    return {
        "database_cache": db_cache_stats,
        "api_cache": api_cache_stats
    }
```

### **3. Redis Caching**

```python
# Redis caching for expensive operations
@app.get("/user-profile/{user_id}")
async def get_user_profile(user_id: int):
    return await redis_ops.get_or_set(
        f"user_profile:{user_id}",
        lambda: fetch_user_profile_from_db(user_id),
        ttl=600  # Cache for 10 minutes
    )

# Cache invalidation
@app.put("/user-profile/{user_id}")
async def update_user_profile(user_id: int, profile_data: Dict[str, Any]):
    # Update profile
    await update_user_profile_in_db(user_id, profile_data)
    
    # Invalidate cache
    await redis_ops.delete(f"user_profile:{user_id}")
    
    return {"success": True}
```

## 🛡️ Circuit Breaker Pattern

### **1. External API Circuit Breaker**

```python
# Circuit breaker configuration
api_config = APIConfig(
    enable_circuit_breaker=True,
    circuit_breaker_threshold=5,  # Open after 5 failures
    circuit_breaker_timeout=60.0  # Wait 60 seconds before half-open
)

# Circuit breaker monitoring
@app.get("/api/circuit-breaker-status")
async def get_circuit_breaker_status():
    status = api_manager.get_circuit_breaker_state()
    return {
        "state": status["state"],
        "failures": status["failures"],
        "threshold": status["threshold"],
        "last_failure": status["last_failure"]
    }

# Circuit breaker is automatically applied to all API operations
@app.get("/external-data-with-circuit-breaker")
async def get_external_data_with_circuit_breaker():
    try:
        response = await api_ops.get("/data")
        return response
    except HTTPException as e:
        if e.status_code == 503:
            return {"error": "Service temporarily unavailable due to circuit breaker"}
        raise
```

## ⏱️ Rate Limiting

### **1. External API Rate Limiting**

```python
# Rate limiting configuration
api_config = APIConfig(
    enable_rate_limiting=True,
    rate_limit=100  # 100 requests per second
)

# Rate limiting is automatically applied to all API operations
@app.get("/external-data-with-rate-limiting")
async def get_external_data_with_rate_limiting():
    # This will be automatically rate limited
    response = await api_ops.get("/data")
    return response
```

## 📊 Performance Monitoring

### **1. Database Performance Monitoring**

```python
# Database performance metrics
@app.get("/database/metrics")
async def get_database_metrics():
    metrics = db_ops.get_query_metrics()
    
    # Calculate statistics
    total_queries = len(metrics)
    successful_queries = sum(1 for m in metrics.values() if m.success)
    failed_queries = total_queries - successful_queries
    
    avg_duration = statistics.mean([
        m.duration_ms for m in metrics.values() if m.duration_ms is not None
    ]) if metrics else 0
    
    return {
        "total_queries": total_queries,
        "successful_queries": successful_queries,
        "failed_queries": failed_queries,
        "success_rate": successful_queries / total_queries if total_queries > 0 else 0,
        "avg_duration_ms": avg_duration,
        "recent_queries": [
            {
                "query_id": m.query_id,
                "query_type": m.query_type.value,
                "duration_ms": m.duration_ms,
                "success": m.success,
                "timestamp": m.start_time.isoformat()
            }
            for m in list(metrics.values())[-10:]  # Last 10 queries
        ]
    }
```

### **2. External API Performance Monitoring**

```python
# External API performance metrics
@app.get("/api/metrics")
async def get_api_metrics():
    metrics = api_ops.get_api_metrics()
    
    # Calculate statistics
    total_requests = len(metrics)
    successful_requests = sum(1 for m in metrics.values() if m.success)
    failed_requests = total_requests - successful_requests
    
    avg_duration = statistics.mean([
        m.duration_ms for m in metrics.values() if m.duration_ms is not None
    ]) if metrics else 0
    
    return {
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
        "avg_duration_ms": avg_duration,
        "recent_requests": [
            {
                "api_id": m.api_id,
                "method": m.method.value,
                "endpoint": m.endpoint,
                "duration_ms": m.duration_ms,
                "success": m.success,
                "status_code": m.status_code,
                "timestamp": m.start_time.isoformat()
            }
            for m in list(metrics.values())[-10:]  # Last 10 requests
        ]
    }
```

## ⚠️ Error Handling

### **1. Database Error Handling**

```python
# Database error handling
@app.get("/users-with-error-handling")
async def get_users_with_error_handling():
    try:
        users = await db_ops.execute_query(
            "SELECT * FROM users WHERE active = true",
            timeout=10.0
        )
        return users
    except HTTPException as e:
        if e.status_code == 408:
            logger.error("Database query timeout")
            return {"error": "Database timeout, please try again"}
        elif e.status_code == 500:
            logger.error("Database error")
            return {"error": "Database error, please try again later"}
        raise
    except Exception as e:
        logger.error(f"Unexpected database error: {e}")
        return {"error": "Unexpected error occurred"}
```

### **2. External API Error Handling**

```python
# External API error handling
@app.get("/external-data-with-error-handling")
async def get_external_data_with_error_handling():
    try:
        response = await api_ops.get("/data")
        return response
    except HTTPException as e:
        if e.status_code == 503:
            logger.error("External API circuit breaker open")
            return {"error": "External service temporarily unavailable"}
        elif e.status_code == 429:
            logger.error("External API rate limit exceeded")
            return {"error": "Too many requests, please try again later"}
        elif e.status_code == 500:
            logger.error("External API error")
            return {"error": "External service error, please try again later"}
        raise
    except Exception as e:
        logger.error(f"Unexpected external API error: {e}")
        return {"error": "Unexpected error occurred"}
```

## ✅ Best Practices

### **1. Database Best Practices**

```python
# ✅ Good: Use async database operations
@app.get("/users")
async def get_users():
    return await db_ops.execute_query("SELECT * FROM users")

# ✅ Good: Use connection pooling
db_config = DatabaseConfig(pool_size=20, max_overflow=30)

# ✅ Good: Use query caching for expensive queries
users = await db_ops.execute_query(
    "SELECT * FROM users WHERE active = true",
    cache_key="active_users",
    cache_ttl=300
)

# ✅ Good: Use transactions for related operations
await db_ops.execute_transaction([
    {"type": QueryType.UPDATE, "query": "UPDATE accounts SET balance = balance - :amount WHERE id = :from_id", "parameters": {"amount": 100, "from_id": 1}},
    {"type": QueryType.UPDATE, "query": "UPDATE accounts SET balance = balance + :amount WHERE id = :to_id", "parameters": {"amount": 100, "to_id": 2}}
])

# ❌ Bad: Synchronous database operations
@app.get("/users")
def get_users():
    # This blocks the event loop
    return db.execute_sync("SELECT * FROM users")
```

### **2. External API Best Practices**

```python
# ✅ Good: Use async API operations
@app.get("/external-data")
async def get_external_data():
    return await api_ops.get("/data")

# ✅ Good: Use connection pooling
api_config = APIConfig(max_connections=100, max_connections_per_host=10)

# ✅ Good: Use response caching
response = await api_ops.get(
    endpoint="/data",
    cache_key="external_data",
    cache_ttl=300
)

# ✅ Good: Use circuit breaker for fault tolerance
api_config = APIConfig(
    enable_circuit_breaker=True,
    circuit_breaker_threshold=5
)

# ✅ Good: Use rate limiting
api_config = APIConfig(
    enable_rate_limiting=True,
    rate_limit=100
)

# ❌ Bad: Synchronous API calls
@app.get("/external-data")
def get_external_data():
    # This blocks the event loop
    return requests.get("https://api.external.com/data").json()
```

### **3. Error Handling Best Practices**

```python
# ✅ Good: Proper async error handling
@app.get("/data")
async def get_data():
    try:
        # Try database first
        data = await db_ops.execute_query("SELECT * FROM data")
        if data:
            return data
        
        # Fallback to external API
        data = await api_ops.get("/data")
        return data
        
    except HTTPException as e:
        logger.error(f"Data retrieval error: {e}")
        return {"error": "Unable to retrieve data"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"error": "Unexpected error occurred"}

# ✅ Good: Graceful degradation
@app.get("/user-profile/{user_id}")
async def get_user_profile(user_id: int):
    try:
        # Try cache first
        profile = await redis_ops.get(f"user_profile:{user_id}")
        if profile:
            return profile
        
        # Try database
        profile = await db_ops.execute_query(
            "SELECT * FROM user_profiles WHERE user_id = :user_id",
            parameters={"user_id": user_id}
        )
        if profile:
            # Cache the result
            await redis_ops.set(f"user_profile:{user_id}", profile[0], ttl=600)
            return profile[0]
        
        # Return default profile
        return {"user_id": user_id, "name": "Unknown User"}
        
    except Exception as e:
        logger.error(f"Profile retrieval error: {e}")
        return {"user_id": user_id, "name": "Unknown User"}
```

## 🔗 Integration Examples

### **1. Complete Async Functions Setup**

```python
from fastapi import FastAPI, HTTPException
from api.async_operations.async_database import AsyncDatabaseManager, AsyncDatabaseOperations, DatabaseConfig, DatabaseType
from api.async_operations.async_external_api import AsyncExternalAPIManager, AsyncExternalAPIOperations, APIConfig, APIType
from api.async_operations.async_database import AsyncRedisOperations

# Create FastAPI app
app = FastAPI(title="HeyGen AI API")

# Initialize database
db_config = DatabaseConfig(
    database_type=DatabaseType.POSTGRESQL,
    host="localhost",
    port=5432,
    database="heygen_ai",
    username="postgres",
    password="password",
    pool_size=20,
    enable_query_cache=True
)

db_manager = AsyncDatabaseManager(db_config)
db_ops = AsyncDatabaseOperations(db_manager)

# Initialize external API
api_config = APIConfig(
    base_url="https://api.external.com",
    timeout=30.0,
    max_retries=3,
    enable_caching=True,
    enable_circuit_breaker=True,
    enable_rate_limiting=True
)

api_manager = AsyncExternalAPIManager(api_config)
api_ops = AsyncExternalAPIOperations(api_manager)

# Initialize Redis
redis_ops = AsyncRedisOperations("redis://localhost:6379")

# Startup and shutdown
@app.on_event("startup")
async def startup_event():
    await db_manager.initialize()
    await api_manager.initialize()
    await redis_ops.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    await db_manager.cleanup()
    await api_manager.cleanup()
    await redis_ops.cleanup()

# Database operations
@app.get("/users")
async def get_users():
    return await db_ops.execute_query(
        "SELECT * FROM users WHERE active = true",
        cache_key="active_users",
        cache_ttl=300
    )

@app.post("/users")
async def create_user(user_data: UserCreate):
    return await db_ops.execute_insert(
        table="users",
        data=user_data.dict(),
        returning="id, name, email"
    )

# External API operations
@app.get("/external-data")
async def get_external_data():
    return await api_ops.get(
        endpoint="/data",
        cache_key="external_data",
        cache_ttl=600
    )

@app.post("/external-create")
async def create_external_resource(data: Dict[str, Any]):
    return await api_ops.post(
        endpoint="/resources",
        data=data
    )

# Combined operations
@app.get("/user-with-external-data/{user_id}")
async def get_user_with_external_data(user_id: int):
    # Get user from database
    users = await db_ops.execute_query(
        "SELECT * FROM users WHERE id = :user_id",
        parameters={"user_id": user_id}
    )
    
    if not users:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = users[0]
    
    # Get external data
    external_data = await api_ops.get(
        endpoint=f"/user-data/{user_id}",
        cache_key=f"user_external_data:{user_id}",
        cache_ttl=300
    )
    
    return {
        "user": user,
        "external_data": external_data.get("data", {})
    }

# Monitoring endpoints
@app.get("/metrics/database")
async def get_database_metrics():
    return db_ops.get_query_metrics()

@app.get("/metrics/api")
async def get_api_metrics():
    return api_ops.get_api_metrics()

@app.get("/metrics/cache")
async def get_cache_metrics():
    return {
        "database_cache": db_ops.get_cache_stats(),
        "api_cache": api_ops.get_cache_stats()
    }
```

### **2. Advanced Integration Example**

```python
# Complex operation with multiple async functions
@app.post("/process-video")
async def process_video(video_data: VideoData):
    try:
        # 1. Save video metadata to database
        video_record = await db_ops.execute_insert(
            table="videos",
            data={
                "filename": video_data.filename,
                "size": video_data.size,
                "status": "processing"
            },
            returning="id, filename, status"
        )
        
        # 2. Upload video to external storage
        upload_response = await api_ops.upload_file(
            endpoint="/upload",
            file_path=video_data.file_path,
            file_field="video",
            additional_data={"video_id": video_record["id"]}
        )
        
        # 3. Start processing with external API
        process_response = await api_ops.post(
            endpoint="/process",
            data={
                "video_id": video_record["id"],
                "upload_url": upload_response["data"]["url"],
                "options": video_data.processing_options
            }
        )
        
        # 4. Update database with processing status
        await db_ops.execute_update(
            table="videos",
            data={"status": "processing", "external_job_id": process_response["data"]["job_id"]},
            where_conditions={"id": video_record["id"]}
        )
        
        # 5. Cache the result
        await redis_ops.set(
            f"video_status:{video_record['id']}",
            {"status": "processing", "job_id": process_response["data"]["job_id"]},
            ttl=3600
        )
        
        return {
            "video_id": video_record["id"],
            "status": "processing",
            "job_id": process_response["data"]["job_id"]
        }
        
    except Exception as e:
        logger.error(f"Video processing error: {e}")
        
        # Update database with error status
        if 'video_record' in locals():
            await db_ops.execute_update(
                table="videos",
                data={"status": "error", "error_message": str(e)},
                where_conditions={"id": video_record["id"]}
            )
        
        raise HTTPException(status_code=500, detail="Video processing failed")
```

## 🔧 Troubleshooting

### **1. Common Issues**

```python
# Issue: Database connection timeout
# Solution: Increase timeout and check connection pool
db_config = DatabaseConfig(
    connection_timeout=30.0,
    pool_timeout=60.0,
    pool_size=50
)

# Issue: External API rate limiting
# Solution: Implement proper rate limiting and caching
api_config = APIConfig(
    enable_rate_limiting=True,
    rate_limit=50,  # Reduce rate limit
    enable_caching=True,
    cache_ttl=600  # Increase cache TTL
)

# Issue: Memory leaks from connection pools
# Solution: Proper cleanup
@app.on_event("shutdown")
async def shutdown_event():
    await db_manager.cleanup()
    await api_manager.cleanup()
    await redis_ops.cleanup()

# Issue: Circuit breaker stuck open
# Solution: Monitor and reset circuit breaker
@app.post("/api/reset-circuit-breaker")
async def reset_circuit_breaker():
    api_manager.circuit_breaker_state = "closed"
    api_manager.circuit_breaker_failures = 0
    return {"message": "Circuit breaker reset"}
```

### **2. Performance Monitoring**

```python
# Monitor database performance
@app.get("/database/performance")
async def get_database_performance():
    metrics = db_ops.get_query_metrics()
    
    # Analyze slow queries
    slow_queries = [
        m for m in metrics.values()
        if m.duration_ms and m.duration_ms > 1000
    ]
    
    return {
        "total_queries": len(metrics),
        "slow_queries": len(slow_queries),
        "avg_duration_ms": statistics.mean([
            m.duration_ms for m in metrics.values() if m.duration_ms
        ]) if metrics else 0,
        "slow_queries_details": [
            {
                "query": m.query_text,
                "duration_ms": m.duration_ms,
                "timestamp": m.start_time.isoformat()
            }
            for m in slow_queries
        ]
    }

# Monitor external API performance
@app.get("/api/performance")
async def get_api_performance():
    metrics = api_ops.get_api_metrics()
    
    # Analyze failed requests
    failed_requests = [
        m for m in metrics.values()
        if not m.success
    ]
    
    return {
        "total_requests": len(metrics),
        "failed_requests": len(failed_requests),
        "success_rate": (len(metrics) - len(failed_requests)) / len(metrics) if metrics else 0,
        "avg_duration_ms": statistics.mean([
            m.duration_ms for m in metrics.values() if m.duration_ms
        ]) if metrics else 0,
        "failed_requests_details": [
            {
                "endpoint": m.endpoint,
                "method": m.method.value,
                "error": m.error_message,
                "timestamp": m.start_time.isoformat()
            }
            for m in failed_requests
        ]
    }
```

## 📊 Summary

### **Key Benefits**

1. **Non-Blocking Operations**: All database and API operations are async
2. **Connection Pooling**: Efficient resource management
3. **Automatic Retry**: Built-in retry logic with exponential backoff
4. **Response Caching**: Performance optimization through caching
5. **Circuit Breaker**: Fault tolerance for external services
6. **Rate Limiting**: Request throttling and protection
7. **Performance Monitoring**: Comprehensive metrics and monitoring
8. **Error Handling**: Proper async error handling patterns

### **Implementation Checklist**

- [ ] **Setup Database Manager**: Configure async database operations
- [ ] **Setup External API Manager**: Configure async API operations
- [ ] **Implement Connection Pooling**: Efficient resource management
- [ ] **Add Retry Logic**: Automatic retry mechanisms
- [ ] **Implement Caching**: Response and query caching
- [ ] **Add Circuit Breaker**: Fault tolerance patterns
- [ ] **Setup Rate Limiting**: Request throttling
- [ ] **Monitor Performance**: Track metrics and performance
- [ ] **Handle Errors**: Proper error handling
- [ ] **Test Async Operations**: Verify non-blocking behavior

### **Next Steps**

1. **Integration**: Integrate with existing HeyGen AI services
2. **Customization**: Customize async patterns for specific needs
3. **Scaling**: Scale async operations for production workloads
4. **Advanced Patterns**: Implement advanced async patterns
5. **Automation**: Add automated performance optimization
6. **Reporting**: Create performance reports and dashboards

This comprehensive async functions system ensures your HeyGen AI API maintains optimal performance, prevents blocking operations, and scales efficiently to meet growing demands through dedicated async functions for database and external API operations. 