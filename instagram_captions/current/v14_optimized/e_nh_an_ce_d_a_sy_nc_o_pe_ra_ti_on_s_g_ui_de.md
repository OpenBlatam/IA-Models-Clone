# Enhanced Async Operations Guide for Instagram Captions API v14.0

## Overview

This guide covers the enhanced async operations system implemented in the Instagram Captions API v14.0, providing comprehensive async functions for database and external API operations with advanced features like connection pooling, circuit breakers, retry mechanisms, and performance monitoring.

## Table of Contents

1. [Enhanced Async Operations Architecture](#enhanced-async-operations-architecture)
2. [Database Operations](#database-operations)
3. [External API Operations](#external-api-operations)
4. [Connection Pooling](#connection-pooling)
5. [Circuit Breakers](#circuit-breakers)
6. [Retry Mechanisms](#retry-mechanisms)
7. [Performance Monitoring](#performance-monitoring)
8. [Batch Operations](#batch-operations)
9. [Caching Strategies](#caching-strategies)
10. [Error Handling](#error-handling)
11. [API Endpoints](#api-endpoints)
12. [Best Practices](#best-practices)
13. [Examples](#examples)

## Enhanced Async Operations Architecture

### Key Components

1. **EnhancedDatabasePool**: Advanced database connection pooling
2. **EnhancedAPIClient**: Comprehensive API client with rate limiting
3. **AsyncDataService**: High-level data service with decorators
4. **AsyncIOMonitor**: Performance monitoring and analytics
5. **EnhancedCircuitBreaker**: Advanced circuit breaker pattern

### Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Enhanced      │    │   Enhanced      │    │   Async         │
│   Database      │    │   API Client    │    │   Data Service  │
│   Pool          │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Circuit       │    │   Rate          │    │   Performance   │
│   Breakers      │    │   Limiting      │    │   Monitoring    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Database Operations

### 1. Enhanced Database Pool

The enhanced database pool provides connection pooling for multiple database types:

```python
from core.enhanced_async_operations import (
    EnhancedDatabasePool, DatabaseConfig, DatabaseType, OperationType
)

# Create database configuration
db_config = DatabaseConfig(
    postgres_url="postgresql://user:pass@localhost/db",
    postgres_pool_size=20,
    redis_url="redis://localhost:6379",
    redis_pool_size=50,
    mongodb_url="mongodb://localhost:27017",
    mongodb_database="instagram_captions",
    enable_circuit_breaker=True,
    enable_query_cache=True,
    query_cache_ttl=300
)

# Initialize database pool
db_pool = EnhancedDatabasePool(db_config)
await db_pool.initialize()
```

### 2. Database Query Operations

```python
# Execute single query with caching
result = await db_pool.execute_query(
    query="SELECT * FROM users WHERE id = $1",
    params=("user_123",),
    cache_key="user_profile_user_123",
    cache_ttl=600,
    operation_type=OperationType.DATABASE_READ
)

# Execute batch queries with transactions
queries = [
    ("INSERT INTO users (id, name) VALUES ($1, $2)", ("user_123", "John")),
    ("UPDATE user_stats SET last_login = NOW() WHERE user_id = $1", ("user_123",)),
    ("SELECT * FROM users WHERE id = $1", ("user_123",))
]

results = await db_pool.execute_batch_queries(
    queries=queries,
    operation_type=OperationType.BATCH_OPERATION
)
```

### 3. Connection Context Managers

```python
# PostgreSQL connection
async with db_pool.get_postgres_connection() as conn:
    result = await conn.fetch("SELECT * FROM users")

# Redis connection
async with db_pool.get_redis_connection() as redis:
    await redis.set("key", "value", ex=3600)

# MongoDB connection
async with db_pool.get_mongodb_database() as db:
    result = await db.users.find_one({"id": "user_123"})
```

## External API Operations

### 1. Enhanced API Client

The enhanced API client provides comprehensive API request handling:

```python
from core.enhanced_async_operations import (
    EnhancedAPIClient, APIConfig, APIType
)

# Create API configuration
api_config = APIConfig(
    timeout=30.0,
    max_retries=3,
    max_connections=100,
    enable_circuit_breaker=True,
    enable_rate_limiting=True,
    requests_per_minute=1000,
    api_keys={
        "openai": "your-openai-key",
        "huggingface": "your-huggingface-key",
        "anthropic": "your-anthropic-key"
    }
)

# Initialize API client
api_client = EnhancedAPIClient(api_config)
await api_client.initialize()
```

### 2. API Request Operations

```python
# Single API request
response = await api_client.make_request(
    method="POST",
    url="https://api.openai.com/v1/chat/completions",
    api_type=APIType.OPENAI,
    json_data={
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}]
    }
)

# Batch API requests
requests = [
    {
        "method": "POST",
        "url": "https://api.openai.com/v1/chat/completions",
        "api_type": APIType.OPENAI,
        "json_data": {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello"}]}
    },
    {
        "method": "POST",
        "url": "https://api.huggingface.co/models/gpt2",
        "api_type": APIType.HUGGINGFACE,
        "json_data": {"inputs": "Hello"}
    }
]

results = await api_client.make_batch_requests(requests, max_concurrent=5)
```

## Connection Pooling

### 1. Database Connection Pooling

```python
# PostgreSQL connection pooling
async with db_pool.get_postgres_connection() as conn:
    # Connection is automatically managed
    result = await conn.fetch("SELECT * FROM users")
    # Connection is automatically returned to pool

# Connection pool statistics
stats = db_pool.get_stats()
print(f"Active connections: {stats['active_connections']}")
print(f"Total queries: {stats['total_queries']}")
print(f"Cache hit rate: {stats['cache_hit_rate']}")
```

### 2. API Connection Pooling

```python
# HTTP connection pooling
response = await api_client.make_request(
    method="GET",
    url="https://api.example.com/data",
    api_type=APIType.CUSTOM
)

# Connection pool statistics
stats = api_client.get_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Success rate: {stats['success_rate']}")
print(f"Average response time: {stats['avg_response_time']}")
```

## Circuit Breakers

### 1. Enhanced Circuit Breaker

The enhanced circuit breaker provides advanced fault tolerance:

```python
from core.enhanced_async_operations import EnhancedCircuitBreaker

# Create circuit breaker
circuit_breaker = EnhancedCircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
    half_open_max_calls=3
)

# Check circuit breaker state
if await circuit_breaker.is_open():
    # Circuit is open, handle gracefully
    return fallback_response

# Record operation results
try:
    result = await operation()
    await circuit_breaker.record_success()
    return result
except Exception as e:
    await circuit_breaker.record_failure()
    raise

# Get circuit breaker state
state = circuit_breaker.get_state()
print(f"State: {state['state']}")
print(f"Failure count: {state['failure_count']}")
```

### 2. Circuit Breaker Integration

```python
# Database operations with circuit breaker
async def safe_database_query(query: str, params: tuple):
    circuit_breaker = db_pool.circuit_breakers[DatabaseType.POSTGRESQL.value]
    
    if await circuit_breaker.is_open():
        raise Exception("Database circuit breaker is open")
    
    try:
        result = await db_pool.execute_query(query, params)
        await circuit_breaker.record_success()
        return result
    except Exception as e:
        await circuit_breaker.record_failure()
        raise

# API operations with circuit breaker
async def safe_api_request(url: str, data: dict):
    circuit_breaker = api_client.circuit_breakers[APIType.OPENAI.value]
    
    if await circuit_breaker.is_open():
        raise Exception("API circuit breaker is open")
    
    try:
        response = await api_client.make_request("POST", url, json_data=data)
        await circuit_breaker.record_success()
        return response
    except Exception as e:
        await circuit_breaker.record_failure()
        raise
```

## Retry Mechanisms

### 1. Automatic Retry Logic

```python
# Database operations with retry
async def robust_database_operation():
    for attempt in range(3):  # 3 retry attempts
        try:
            result = await db_pool.execute_query("SELECT * FROM users")
            return result
        except Exception as e:
            if attempt == 2:  # Last attempt
                raise
            await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff

# API operations with retry
async def robust_api_operation():
    for attempt in range(3):  # 3 retry attempts
        try:
            response = await api_client.make_request("GET", "https://api.example.com/data")
            return response
        except Exception as e:
            if attempt == 2:  # Last attempt
                raise
            await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
```

### 2. Configurable Retry Settings

```python
# Configure retry settings
db_config = DatabaseConfig(
    retry_attempts=3,
    retry_delay=1.0,
    retry_backoff=2.0  # Exponential backoff multiplier
)

api_config = APIConfig(
    max_retries=3,
    retry_delay=1.0,
    retry_backoff=2.0
)
```

## Performance Monitoring

### 1. Async I/O Monitor

```python
from core.enhanced_async_operations import AsyncIOMonitor

# Create I/O monitor
io_monitor = AsyncIOMonitor()

# Record operations
io_monitor.record_operation(
    operation_type="database_read",
    duration=0.15,
    success=True,
    metadata={"table": "users", "rows": 100}
)

# Get performance summary
summary = io_monitor.get_performance_summary()
print(f"Total operations: {summary['total_operations']}")
print(f"Operation stats: {summary['operation_stats']}")
```

### 2. Performance Metrics

```python
# Database performance metrics
db_stats = db_pool.get_stats()
print(f"Total queries: {db_stats['total_queries']}")
print(f"Cache hit rate: {db_stats['cache_hit_rate']}")
print(f"Average query time: {db_stats['avg_query_time']}")
print(f"Query errors: {db_stats['query_errors']}")

# API performance metrics
api_stats = api_client.get_stats()
print(f"Total requests: {api_stats['total_requests']}")
print(f"Success rate: {api_stats['success_rate']}")
print(f"Average response time: {api_stats['avg_response_time']}")
print(f"Failed requests: {api_stats['failed_requests']}")
```

## Batch Operations

### 1. Database Batch Operations

```python
# Batch database queries
queries = [
    ("INSERT INTO users (id, name, email) VALUES ($1, $2, $3)", ("user1", "John", "john@example.com")),
    ("INSERT INTO users (id, name, email) VALUES ($1, $2, $3)", ("user2", "Jane", "jane@example.com")),
    ("UPDATE user_stats SET total_users = total_users + 2", None),
    ("SELECT COUNT(*) FROM users", None)
]

results = await db_pool.execute_batch_queries(
    queries=queries,
    operation_type=OperationType.BATCH_OPERATION
)
```

### 2. API Batch Operations

```python
# Batch API requests
requests = [
    {
        "method": "POST",
        "url": "https://api.openai.com/v1/chat/completions",
        "api_type": APIType.OPENAI,
        "json_data": {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello 1"}]}
    },
    {
        "method": "POST",
        "url": "https://api.openai.com/v1/chat/completions",
        "api_type": APIType.OPENAI,
        "json_data": {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello 2"}]}
    }
]

results = await api_client.make_batch_requests(
    requests=requests,
    max_concurrent=5  # Process up to 5 requests concurrently
)
```

## Caching Strategies

### 1. Query Caching

```python
# Cache database query results
result = await db_pool.execute_query(
    query="SELECT * FROM users WHERE id = $1",
    params=("user_123",),
    cache_key="user_profile_user_123",
    cache_ttl=600  # Cache for 10 minutes
)

# Check if result was cached
if cache_key:
    # Result was retrieved from cache
    pass
```

### 2. Multi-level Caching

```python
# Memory cache (L1)
if key in memory_cache:
    return memory_cache[key]

# Redis cache (L2)
cached_value = await redis_cache.get(key)
if cached_value:
    memory_cache[key] = cached_value
    return cached_value

# Database (L3)
value = await fetch_from_database(key)

# Cache at all levels
memory_cache[key] = value
await redis_cache.set(key, value, ex=3600)
```

## Error Handling

### 1. Comprehensive Error Handling

```python
async def robust_operation():
    try:
        # Database operation
        result = await db_pool.execute_query("SELECT * FROM users")
        return result
    except asyncio.TimeoutError:
        logger.error("Database operation timeout")
        raise HTTPException(status_code=408, detail="Request timeout")
    except Exception as e:
        logger.error(f"Database operation failed: {e}")
        raise HTTPException(status_code=500, detail="Database error")

async def robust_api_call():
    try:
        # API operation
        response = await api_client.make_request("GET", "https://api.example.com/data")
        return response
    except asyncio.TimeoutError:
        logger.error("API request timeout")
        raise HTTPException(status_code=408, detail="API timeout")
    except Exception as e:
        logger.error(f"API request failed: {e}")
        raise HTTPException(status_code=503, detail="API error")
```

### 2. Circuit Breaker Error Handling

```python
async def safe_operation():
    circuit_breaker = db_pool.circuit_breakers[DatabaseType.POSTGRESQL.value]
    
    if await circuit_breaker.is_open():
        # Return cached data or fallback response
        return get_fallback_data()
    
    try:
        result = await operation()
        await circuit_breaker.record_success()
        return result
    except Exception as e:
        await circuit_breaker.record_failure()
        # Return fallback response
        return get_fallback_data()
```

## API Endpoints

### 1. Database Operations

```http
POST /api/v14/enhanced-async/database/initialize
Authorization: Bearer <api_key>

POST /api/v14/enhanced-async/database/query
Content-Type: application/json
{
    "query": "SELECT * FROM users WHERE id = $1",
    "params": ["user_123"],
    "cache_key": "user_profile_user_123",
    "cache_ttl": 600
}

POST /api/v14/enhanced-async/database/batch
Content-Type: application/json
{
    "queries": [
        {"query": "INSERT INTO users (id, name) VALUES ($1, $2)", "params": ["user1", "John"]},
        {"query": "SELECT * FROM users WHERE id = $1", "params": ["user1"]}
    ]
}

GET /api/v14/enhanced-async/database/stats
Authorization: Bearer <api_key>
```

### 2. API Operations

```http
POST /api/v14/enhanced-async/api/initialize
Authorization: Bearer <api_key>

POST /api/v14/enhanced-async/api/request
Content-Type: application/json
{
    "method": "POST",
    "url": "https://api.openai.com/v1/chat/completions",
    "api_type": "openai",
    "json_data": {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}]
    }
}

POST /api/v14/enhanced-async/api/batch
Content-Type: application/json
{
    "requests": [
        {
            "method": "POST",
            "url": "https://api.openai.com/v1/chat/completions",
            "api_type": "openai",
            "json_data": {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello"}]}
        }
    ],
    "max_concurrent": 5
}

GET /api/v14/enhanced-async/api/stats
Authorization: Bearer <api_key>
```

### 3. AI Service Operations

```http
POST /api/v14/enhanced-async/ai/openai
Content-Type: application/json
{
    "prompt": "Generate a caption for a sunset photo",
    "model": "gpt-3.5-turbo",
    "max_tokens": 1000,
    "temperature": 0.7
}

POST /api/v14/enhanced-async/ai/huggingface
Content-Type: application/json
{
    "prompt": "Generate a caption for a sunset photo",
    "model": "gpt2",
    "max_length": 100,
    "temperature": 0.7
}

POST /api/v14/enhanced-async/ai/anthropic
Content-Type: application/json
{
    "prompt": "Generate a caption for a sunset photo",
    "model": "claude-3-sonnet-20240229",
    "max_tokens": 1000,
    "temperature": 0.7
}
```

### 4. Data Service Operations

```http
POST /api/v14/enhanced-async/data/user-profile
Content-Type: application/json
{
    "user_id": "user_123"
}

POST /api/v14/enhanced-async/data/save-profile
Content-Type: application/json
{
    "user_id": "user_123",
    "profile_data": {
        "name": "John Doe",
        "email": "john@example.com",
        "preferences": {"theme": "dark"}
    }
}

POST /api/v14/enhanced-async/data/process-request
Content-Type: application/json
{
    "user_id": "user_123",
    "prompt": "Generate a caption for my photo"
}
```

### 5. Monitoring Operations

```http
GET /api/v14/enhanced-async/monitoring/io-stats
Authorization: Bearer <api_key>

GET /api/v14/enhanced-async/monitoring/all-stats
Authorization: Bearer <api_key>

GET /api/v14/enhanced-async/health
Authorization: Bearer <api_key>

POST /api/v14/enhanced-async/cleanup
Authorization: Bearer <api_key>
```

## Best Practices

### 1. Database Operations

```python
# ✅ Good: Use connection pooling
async with db_pool.get_postgres_connection() as conn:
    result = await conn.fetch("SELECT * FROM users")

# ✅ Good: Use query caching for read operations
result = await db_pool.execute_query(
    query="SELECT * FROM users WHERE id = $1",
    params=("user_123",),
    cache_key="user_profile_user_123",
    cache_ttl=600
)

# ✅ Good: Use batch operations for multiple queries
queries = [
    ("INSERT INTO users (id, name) VALUES ($1, $2)", ("user1", "John")),
    ("UPDATE user_stats SET count = count + 1", None)
]
results = await db_pool.execute_batch_queries(queries)

# ❌ Bad: Don't create connections manually
conn = await asyncpg.connect(url)  # Don't do this
```

### 2. API Operations

```python
# ✅ Good: Use API client with circuit breaker
response = await api_client.make_request(
    method="POST",
    url="https://api.openai.com/v1/chat/completions",
    api_type=APIType.OPENAI,
    json_data={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello"}]}
)

# ✅ Good: Use batch requests for multiple API calls
requests = [
    {"method": "POST", "url": "https://api1.example.com", "json_data": data1},
    {"method": "POST", "url": "https://api2.example.com", "json_data": data2}
]
results = await api_client.make_batch_requests(requests, max_concurrent=5)

# ❌ Bad: Don't make requests without proper error handling
response = await httpx.post("https://api.example.com", json=data)  # Don't do this
```

### 3. Error Handling

```python
# ✅ Good: Comprehensive error handling
async def robust_operation():
    try:
        result = await operation()
        return result
    except asyncio.TimeoutError:
        logger.error("Operation timeout")
        raise HTTPException(status_code=408, detail="Request timeout")
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal error")

# ✅ Good: Circuit breaker integration
async def safe_operation():
    circuit_breaker = get_circuit_breaker()
    if await circuit_breaker.is_open():
        return get_fallback_response()
    
    try:
        result = await operation()
        await circuit_breaker.record_success()
        return result
    except Exception as e:
        await circuit_breaker.record_failure()
        return get_fallback_response()
```

### 4. Performance Monitoring

```python
# ✅ Good: Monitor all operations
async def monitored_operation():
    start_time = time.time()
    try:
        result = await operation()
        duration = time.time() - start_time
        
        # Record successful operation
        io_monitor.record_operation(
            operation_type="database_read",
            duration=duration,
            success=True
        )
        
        return result
    except Exception as e:
        duration = time.time() - start_time
        
        # Record failed operation
        io_monitor.record_operation(
            operation_type="database_read",
            duration=duration,
            success=False
        )
        
        raise
```

## Examples

### 1. Complete Database Service

```python
class UserService:
    def __init__(self, db_pool: EnhancedDatabasePool):
        self.db_pool = db_pool
    
    @async_database_operation(OperationType.DATABASE_READ, cache_key="user_profile")
    async def get_user(self, user_id: str) -> Dict[str, Any]:
        """Get user with caching"""
        query = "SELECT * FROM users WHERE id = $1"
        result = await self.db_pool.execute_query(query, (user_id,))
        return dict(result[0]) if result else {}
    
    @async_database_operation(OperationType.DATABASE_WRITE)
    async def create_user(self, user_data: Dict[str, Any]) -> bool:
        """Create user"""
        query = "INSERT INTO users (id, name, email) VALUES ($1, $2, $3)"
        await self.db_pool.execute_query(
            query, 
            (user_data["id"], user_data["name"], user_data["email"])
        )
        return True
    
    async def batch_create_users(self, users: List[Dict[str, Any]]) -> List[bool]:
        """Batch create users"""
        queries = []
        for user in users:
            query = "INSERT INTO users (id, name, email) VALUES ($1, $2, $3)"
            params = (user["id"], user["name"], user["email"])
            queries.append((query, params))
        
        results = await self.db_pool.execute_batch_queries(queries)
        return [True] * len(results)
```

### 2. Complete API Service

```python
class AIService:
    def __init__(self, api_client: EnhancedAPIClient):
        self.api_client = api_client
    
    @async_api_operation(APIType.OPENAI)
    async def generate_openai_content(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        """Generate content using OpenAI"""
        url = "https://api.openai.com/v1/chat/completions"
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000
        }
        
        response = await self.api_client.make_request("POST", url, json_data=data)
        return response["data"]["choices"][0]["message"]["content"]
    
    @async_api_operation(APIType.HUGGINGFACE)
    async def generate_huggingface_content(self, prompt: str, model: str = "gpt2") -> str:
        """Generate content using HuggingFace"""
        url = f"https://api.huggingface.co/models/{model}"
        data = {"inputs": prompt}
        
        response = await self.api_client.make_request("POST", url, json_data=data)
        return response["data"][0]["generated_text"]
    
    async def batch_generate_content(self, prompts: List[str], model: str = "gpt-3.5-turbo") -> List[str]:
        """Batch generate content"""
        requests = []
        for prompt in prompts:
            request = {
                "method": "POST",
                "url": "https://api.openai.com/v1/chat/completions",
                "api_type": APIType.OPENAI,
                "json_data": {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1000
                }
            }
            requests.append(request)
        
        results = await self.api_client.make_batch_requests(requests, max_concurrent=5)
        
        contents = []
        for result in results:
            if isinstance(result, Exception):
                contents.append(f"Error: {result}")
            else:
                content = result["data"]["choices"][0]["message"]["content"]
                contents.append(content)
        
        return contents
```

### 3. Complete Data Service

```python
class EnhancedDataService:
    def __init__(self):
        self.db_pool = db_pool
        self.api_client = api_client
    
    async def process_user_request(self, user_id: str, prompt: str) -> Dict[str, Any]:
        """Process complete user request"""
        try:
            # Get user profile
            profile = await self.get_user_profile(user_id)
            
            # Generate AI content
            content = await self.generate_ai_content(prompt)
            
            # Save request history
            await self.save_user_profile(user_id, {
                **profile,
                "last_request": prompt,
                "last_response": content,
                "request_count": profile.get("request_count", 0) + 1
            })
            
            return {
                "user_id": user_id,
                "prompt": prompt,
                "content": content,
                "profile": profile
            }
            
        except Exception as e:
            logger.error(f"User request processing failed: {e}")
            raise
    
    @async_database_operation(OperationType.DATABASE_READ, cache_key="user_profile")
    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile with caching"""
        query = "SELECT * FROM users WHERE id = $1"
        result = await self.db_pool.execute_query(query, (user_id,))
        return dict(result[0]) if result else {}
    
    @async_database_operation(OperationType.DATABASE_WRITE)
    async def save_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        """Save user profile"""
        query = """
            INSERT INTO users (id, profile_data, updated_at) 
            VALUES ($1, $2, NOW()) 
            ON CONFLICT (id) 
            DO UPDATE SET profile_data = $2, updated_at = NOW()
        """
        await self.db_pool.execute_query(query, (user_id, json_dumps(profile_data)))
        return True
    
    @async_api_operation(APIType.OPENAI)
    async def generate_ai_content(self, prompt: str) -> str:
        """Generate AI content"""
        url = "https://api.openai.com/v1/chat/completions"
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000
        }
        
        response = await self.api_client.make_request("POST", url, json_data=data)
        return response["data"]["choices"][0]["message"]["content"]
```

### 4. Performance Monitoring Example

```python
class PerformanceMonitor:
    def __init__(self):
        self.io_monitor = AsyncIOMonitor()
    
    async def monitor_operation(self, operation_name: str, operation_func: Callable, *args, **kwargs):
        """Monitor operation performance"""
        start_time = time.time()
        
        try:
            result = await operation_func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Record successful operation
            self.io_monitor.record_operation(
                operation_type=operation_name,
                duration=duration,
                success=True
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Record failed operation
            self.io_monitor.record_operation(
                operation_type=operation_name,
                duration=duration,
                success=False,
                metadata={"error": str(e)}
            )
            
            raise
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        summary = self.io_monitor.get_performance_summary()
        
        return {
            "total_operations": summary["total_operations"],
            "operation_stats": summary["operation_stats"],
            "recent_operations": summary["recent_operations"],
            "performance_insights": self._analyze_performance(summary)
        }
    
    def _analyze_performance(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance data"""
        insights = {}
        
        for op_type, stats in summary["operation_stats"].items():
            if stats["avg_duration"] > 1.0:
                insights[f"{op_type}_slow"] = f"Average duration {stats['avg_duration']:.3f}s is high"
            
            if stats["success_rate"] < 0.95:
                insights[f"{op_type}_unreliable"] = f"Success rate {stats['success_rate']:.2%} is low"
        
        return insights
```

## Conclusion

The enhanced async operations system in Instagram Captions API v14.0 provides comprehensive support for database and external API operations with advanced features like connection pooling, circuit breakers, retry mechanisms, and performance monitoring.

Key benefits:
- **Performance**: Optimized connection pooling and caching
- **Reliability**: Circuit breakers and retry mechanisms
- **Scalability**: Efficient resource management
- **Monitoring**: Comprehensive performance analytics
- **Flexibility**: Support for multiple database and API types
- **Error Handling**: Robust error handling and recovery

By following the patterns and best practices outlined in this guide, you can build highly performant, reliable, and scalable systems that efficiently handle database and API operations. 