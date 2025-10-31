# 🗄️🌐 Dedicated Async Operations System

## Overview

The Dedicated Async Operations System provides comprehensive async functions for database and external API operations, ensuring proper async handling throughout the backend system. This system includes dedicated async functions, connection pooling, performance optimization, error handling, and seamless FastAPI integration.

## 🏗️ Architecture

### Core Components

1. **Async Database Operations** (`async_database_operations.py`)
   - Dedicated async functions for all database operations
   - Connection pooling and management
   - Transaction handling with async patterns
   - CRUD operations with caching
   - Performance monitoring and metrics
   - Multiple database backends support

2. **Async API Client** (`async_api_client.py`)
   - Dedicated async functions for external API operations
   - HTTP client management with connection pooling
   - Authentication and authorization handling
   - Rate limiting and circuit breaker patterns
   - Request/response caching
   - Performance monitoring and metrics

3. **Async Operations Integration** (`async_operations_integration.py`)
   - FastAPI integration with dependency injection
   - Middleware for async request handling
   - Repository and service layer patterns
   - Background task integration
   - Health checks and monitoring
   - Configuration management

## 🎯 Key Features

### 1. Dedicated Async Database Functions

#### **CRUD Operations**
```python
# Select operations
async def select_one(table_name: str, conditions: Dict[str, Any], cache_key: str = None)
async def select_many(table_name: str, conditions: Dict[str, Any] = None, limit: int = 100, offset: int = 0)

# Insert operations
async def insert_one(table_name: str, data: Dict[str, Any])
async def insert_many(table_name: str, data_list: List[Dict[str, Any]])

# Update operations
async def update_one(table_name: str, conditions: Dict[str, Any], data: Dict[str, Any])
async def update_many(table_name: str, conditions: Dict[str, Any], data: Dict[str, Any])

# Delete operations
async def delete_one(table_name: str, conditions: Dict[str, Any])
async def delete_many(table_name: str, conditions: Dict[str, Any])

# Transaction operations
async def transaction(operations: List[Callable[[AsyncSession], Awaitable[Any]]])
```

#### **Database Backends Supported**
- **SQLite** with aiosqlite
- **PostgreSQL** with asyncpg
- **MySQL** with aiomysql
- **Redis** for caching
- **SQLAlchemy** async ORM

#### **Connection Pooling**
```python
class AsyncDatabaseManager:
    def __init__(self, config: DatabaseConfig):
        self.connection_pool = None
        self.session_factory = None
        self.connection_semaphore = asyncio.Semaphore(config.pool_size)
```

### 2. Dedicated Async API Functions

#### **HTTP Operations**
```python
# Basic HTTP methods
async def get(endpoint: str, params: Dict[str, Any] = None, headers: Dict[str, str] = None, cache_key: str = None)
async def post(endpoint: str, data: Any = None, params: Dict[str, Any] = None, headers: Dict[str, str] = None)
async def put(endpoint: str, data: Any = None, params: Dict[str, Any] = None, headers: Dict[str, str] = None)
async def delete(endpoint: str, params: Dict[str, Any] = None, headers: Dict[str, str] = None)
async def patch(endpoint: str, data: Any = None, params: Dict[str, Any] = None, headers: Dict[str, str] = None)

# Batch operations
async def batch_request(requests: List[APIRequest], max_concurrent: int = 10)

# WebSocket support
async def websocket_connect(endpoint: str, headers: Dict[str, str] = None)
async def websocket_send(websocket: websockets.WebSocketServerProtocol, message: str)
async def websocket_receive(websocket: websockets.WebSocketServerProtocol)
```

#### **HTTP Clients Supported**
- **aiohttp** for high-performance async HTTP
- **httpx** for modern async HTTP client
- **WebSocket** support for real-time communication

#### **Authentication Types**
- **API Key** authentication
- **Bearer Token** authentication
- **Basic Auth** authentication
- **OAuth2** authentication
- **Custom** authentication

### 3. Performance Optimization

#### **Connection Pooling**
- **Database connection pooling** with configurable pool sizes
- **HTTP connection pooling** with keep-alive connections
- **Connection reuse** to minimize connection overhead
- **Connection health monitoring** and automatic cleanup

#### **Caching Strategies**
- **Redis caching** for database query results
- **HTTP response caching** with TTL
- **Cache invalidation** patterns
- **Cache hit/miss metrics** tracking

#### **Rate Limiting & Circuit Breakers**
- **Rate limiting** to prevent API abuse
- **Circuit breaker pattern** for fault tolerance
- **Exponential backoff** for retries
- **Failure threshold monitoring**

## 📊 Performance Benefits

### 1. Database Operations
- **Reduced latency** through async I/O
- **Better connection utilization** with pooling
- **Improved throughput** with concurrent operations
- **Lower memory usage** with async patterns

### 2. API Operations
- **Faster response times** with async HTTP clients
- **Better resource utilization** with connection pooling
- **Improved reliability** with circuit breakers
- **Enhanced caching** for frequently accessed data

### 3. System Integration
- **Seamless FastAPI integration** with dependency injection
- **Background task processing** for long-running operations
- **Health monitoring** and automatic recovery
- **Performance metrics** and alerting

## 🔧 Usage Patterns

### 1. Database Operations

#### **Basic CRUD Operations**
```python
from agents.backend.onyx.server.features.utils.async_database_operations import AsyncDatabaseManager, DatabaseConfig

# Create database manager
config = DatabaseConfig(
    database_type=DatabaseType.SQLITE,
    connection_string=":memory:",
    pool_size=20,
    enable_caching=True
)

db_manager = AsyncDatabaseManager(config)
await db_manager.initialize()

# Insert user
user_id = await db_manager.insert_one("users", {
    "name": "John Doe",
    "email": "john@example.com"
})

# Select user
user = await db_manager.select_one("users", {"id": user_id}, cache_key=f"user:{user_id}")

# Update user
await db_manager.update_one("users", {"id": user_id}, {"name": "John Smith"})

# Delete user
await db_manager.delete_one("users", {"id": user_id})
```

#### **Transaction Operations**
```python
async def transfer_money(from_account: int, to_account: int, amount: float):
    operations = [
        lambda session: update_account_balance(session, from_account, -amount),
        lambda session: update_account_balance(session, to_account, amount),
        lambda session: log_transaction(session, from_account, to_account, amount)
    ]
    
    return await db_manager.transaction(operations)
```

#### **Batch Operations**
```python
# Batch insert
users_data = [
    {"name": "User 1", "email": "user1@example.com"},
    {"name": "User 2", "email": "user2@example.com"},
    {"name": "User 3", "email": "user3@example.com"}
]

inserted_count = await db_manager.insert_many("users", users_data)
```

### 2. API Operations

#### **Basic HTTP Operations**
```python
from agents.backend.onyx.server.features.utils.async_api_client import AsyncAPIClient, APIConfig

# Create API client
config = APIConfig(
    base_url="https://api.example.com",
    client_type=ClientType.AIOHTTP,
    auth_type=AuthType.API_KEY,
    api_key="your-api-key",
    timeout=30.0,
    max_retries=3
)

api_client = AsyncAPIClient(config)
await api_client.initialize()

# GET request
response = await api_client.get("/users", {"page": 1}, cache_key="users_page_1")

# POST request
user_data = {"name": "John Doe", "email": "john@example.com"}
response = await api_client.post("/users", user_data)

# PUT request
update_data = {"name": "John Smith"}
response = await api_client.put("/users/1", update_data)

# DELETE request
response = await api_client.delete("/users/1")
```

#### **Batch API Requests**
```python
# Create multiple requests
requests = [
    APIRequest(HTTPMethod.GET, "/users/1", cache_key="user_1"),
    APIRequest(HTTPMethod.GET, "/users/2", cache_key="user_2"),
    APIRequest(HTTPMethod.GET, "/users/3", cache_key="user_3")
]

# Execute batch requests
responses = await api_client.batch_request(requests, max_concurrent=5)
```

#### **WebSocket Operations**
```python
# Connect to WebSocket
websocket = await api_client.websocket_connect("/ws/chat")

# Send message
await api_client.websocket_send(websocket, json.dumps({"message": "Hello"}))

# Receive message
message = await api_client.websocket_receive(websocket)
```

### 3. FastAPI Integration

#### **Dependency Injection**
```python
from fastapi import FastAPI, Depends
from agents.backend.onyx.server.features.utils.async_operations_integration import (
    get_database_manager, get_api_client, get_redis_client
)

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(
    user_id: int,
    db_manager: AsyncDatabaseManager = Depends(get_database_manager)
):
    user = await db_manager.select_one("users", {"id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.post("/external-data")
async def get_external_data(
    api_client: AsyncAPIClient = Depends(get_api_client)
):
    response = await api_client.get("/data", cache_key="external_data")
    return response.data
```

#### **Repository Pattern**
```python
class UserRepository(AsyncRepository):
    def __init__(self, database_manager: AsyncDatabaseManager):
        super().__init__(database_manager, "users")
    
    async def find_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        return await self.database_manager.select_one(
            self.table_name,
            {"email": email},
            cache_key=f"user_email:{email}"
        )
    
    async def find_active_users(self, limit: int = 100) -> List[Dict[str, Any]]:
        return await self.database_manager.select_many(
            self.table_name,
            {"status": "active"},
            limit=limit
        )

# Usage
user_repo = UserRepository(db_manager)
user = await user_repo.find_by_email("john@example.com")
active_users = await user_repo.find_active_users()
```

#### **Service Layer Pattern**
```python
class UserService(AsyncService):
    def __init__(self, database_manager: AsyncDatabaseManager, api_clients: Dict[str, AsyncAPIClient]):
        super().__init__(database_manager, api_clients)
        self.user_repo = UserRepository(database_manager)
    
    async def create_user_with_validation(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        # Validate with external API
        validation_response = await self.process_with_api(
            "validation",
            lambda client: client.post("/validate/user", user_data)
        )
        
        if validation_response.get("valid"):
            # Create user in database
            user_id = await self.process_with_database(
                self.user_repo.create,
                user_data
            )
            return {"user_id": user_id, "status": "created"}
        else:
            raise ValueError("User validation failed")

# Usage
user_service = UserService(db_manager, api_clients)
result = await user_service.create_user_with_validation(user_data)
```

### 4. Decorators for Async Operations

#### **Database Operation Decorators**
```python
from agents.backend.onyx.server.features.utils.async_database_operations import async_database_operation

@app.get("/users/{user_id}")
@async_database_operation("users", "select")
async def get_user(user_id: int):
    return {"user_id": user_id}

@app.post("/users")
@async_database_operation("users", "insert")
async def create_user(user_data: Dict[str, Any]):
    return user_data
```

#### **API Operation Decorators**
```python
from agents.backend.onyx.server.features.utils.async_api_client import async_api_operation

@app.get("/external-data")
@async_api_operation("default", "GET", "/data", cache_key="external_data")
async def get_external_data():
    return {"data": "external"}

@app.post("/external-user")
@async_api_operation("default", "POST", "/users")
async def create_external_user(user_data: Dict[str, Any]):
    return user_data
```

#### **Background Task Decorators**
```python
from agents.backend.onyx.server.features.utils.async_operations_integration import async_background_task

@app.post("/process-data")
@async_background_task()
async def process_data_background(data: Dict[str, Any]):
    # Long-running processing
    await asyncio.sleep(10)
    return {"status": "processed"}
```

## 🛠️ Configuration

### 1. Database Configuration
```python
database_config = DatabaseConfig(
    database_type=DatabaseType.POSTGRESQL,
    connection_string="postgresql://user:pass@localhost/db",
    pool_size=20,
    max_overflow=30,
    pool_timeout=30,
    enable_caching=True,
    cache_ttl=3600,
    retry_attempts=3,
    retry_delay=1.0
)
```

### 2. API Configuration
```python
api_config = APIConfig(
    base_url="https://api.example.com",
    client_type=ClientType.AIOHTTP,
    auth_type=AuthType.BEARER_TOKEN,
    bearer_token="your-token",
    timeout=30.0,
    max_retries=3,
    rate_limit=100,
    enable_caching=True,
    cache_ttl=3600,
    connection_pool_size=20,
    max_connections=100,
    enable_circuit_breaker=True,
    circuit_breaker_threshold=5,
    circuit_breaker_timeout=60.0
)
```

### 3. Integration Configuration
```python
integration_config = IntegrationConfig(
    database_config=database_config,
    api_configs={"default": api_config},
    enable_health_checks=True,
    enable_metrics=True,
    enable_caching=True,
    enable_background_tasks=True,
    enable_monitoring=True,
    log_level="INFO",
    cors_origins=["*"],
    gzip_enabled=True
)
```

## 📈 Monitoring & Observability

### 1. Performance Metrics
- **Database query performance** with execution times
- **API response times** and success rates
- **Cache hit/miss ratios** for optimization
- **Connection pool utilization** and health
- **Error rates** and failure patterns

### 2. Health Checks
- **Database connectivity** and query performance
- **API client health** and response times
- **Redis cache availability** and performance
- **Overall system health** status

### 3. Alerting
- **Performance degradation** alerts
- **High error rate** notifications
- **Resource usage** warnings
- **Service availability** monitoring

## 🔄 Best Practices

### 1. Database Operations
- **Use connection pooling** for better performance
- **Implement caching** for frequently accessed data
- **Use transactions** for data consistency
- **Handle errors gracefully** with retry logic
- **Monitor query performance** and optimize slow queries

### 2. API Operations
- **Use appropriate HTTP clients** for your use case
- **Implement rate limiting** to prevent abuse
- **Use circuit breakers** for fault tolerance
- **Cache responses** when appropriate
- **Handle timeouts** and retries properly

### 3. Integration Patterns
- **Use dependency injection** for clean architecture
- **Implement repository pattern** for data access
- **Use service layer** for business logic
- **Handle background tasks** for long-running operations
- **Monitor and log** all operations

### 4. Performance Optimization
- **Use async/await** consistently throughout
- **Implement proper error handling** and recovery
- **Monitor resource usage** and optimize
- **Use caching strategically** for performance
- **Implement proper cleanup** of resources

## 🎯 Conclusion

The Dedicated Async Operations System provides a comprehensive solution for handling database and external API operations asynchronously. By using dedicated async functions, proper connection pooling, caching strategies, and performance monitoring, you can achieve significant performance improvements while maintaining clean, maintainable code.

Key benefits include:
- **Improved performance** through async I/O operations
- **Better resource utilization** with connection pooling
- **Enhanced reliability** with circuit breakers and retry logic
- **Seamless integration** with FastAPI and modern async patterns
- **Comprehensive monitoring** and observability

Start by implementing the basic async operations and gradually add more advanced features like caching, circuit breakers, and monitoring based on your specific requirements. 