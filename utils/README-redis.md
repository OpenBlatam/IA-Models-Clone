# Redis Integration - Onyx

This module provides Redis integration for Onyx, including caching, data management, and performance optimizations.

## Features

- **Redis Manager**: Core Redis functionality for data caching and management
- **Redis Configuration**: Environment-specific configuration settings
- **Redis Utilities**: Helper functions for common Redis operations
- **Redis Middleware**: FastAPI middleware for response caching
- **Redis Decorators**: Decorators for function and model caching
- **Redis Tests**: Comprehensive test suite for Redis integration

## Installation

1. Install Redis server:
```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis

# Windows
# Download from https://github.com/microsoftarchive/redis/releases
```

2. Install Python dependencies:
```bash
pip install -r requirements-redis.txt
```

## Configuration

The Redis configuration is managed through the `RedisConfig` class in `redis_config.py`. You can customize settings for different environments:

```python
from .redis_config import get_config

# Get configuration for current environment
config = get_config("development")  # or "testing", "production"

# Use configuration in Redis manager
redis_manager = RedisManager(
    host=config.host,
    port=config.port,
    db=config.db,
    password=config.password
)
```

## Usage

### Redis Manager

```python
from .redis_utils import RedisUtils

# Create Redis utilities instance
redis_utils = RedisUtils()

# Cache data
redis_utils.cache_data(
    data={"user_id": "123", "name": "John"},
    prefix="user_data",
    identifier="user_123",
    expire=3600  # 1 hour
)

# Get cached data
cached_data = redis_utils.get_cached_data(
    prefix="user_data",
    identifier="user_123"
)

# Cache batch data
redis_utils.cache_batch(
    data_dict={
        "user_123": {"name": "John"},
        "user_456": {"name": "Jane"}
    },
    prefix="user_data",
    expire=3600
)

# Get cached batch data
cached_batch = redis_utils.get_cached_batch(
    prefix="user_data",
    identifiers=["user_123", "user_456"]
)
```

### Redis Middleware

```python
from fastapi import FastAPI
from .redis_middleware import RedisMiddleware

app = FastAPI()

# Add Redis middleware
app.add_middleware(
    RedisMiddleware,
    config={
        "cache_ttl": 3600,  # 1 hour
        "exclude_paths": ["/admin", "/api/v1/auth"],
        "include_paths": ["/api/v1"],
        "cache_headers": True
    }
)

@app.get("/api/v1/users")
async def get_users():
    # This response will be cached
    return {"users": [...]}

@app.post("/api/v1/users")
async def create_user():
    # This response won't be cached
    return {"user": {...}}
```

### Redis Decorators

```python
from .redis_decorators import redis_decorators

# Cache function results
@redis_decorators.cache(
    prefix="function_results",
    ttl=3600  # 1 hour
)
async def get_user_data(user_id: str) -> Dict[str, Any]:
    # Function implementation
    return {"user_id": user_id, "data": {...}}

# Cache model results
@redis_decorators.cache_model(
    prefix="user_models",
    ttl=3600  # 1 hour
)
async def get_user_model(user_id: str) -> UserModel:
    # Function implementation
    return UserModel(...)

# Cache batch results
@redis_decorators.cache_batch(
    prefix="batch_results",
    ttl=3600  # 1 hour
)
async def get_batch_data(user_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    # Function implementation
    return {user_id: {...} for user_id in user_ids}

# Invalidate cache
@redis_decorators.invalidate(
    prefix="user_data"
)
async def update_user_data(user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    # Function implementation
    return {"user_id": user_id, "data": data}
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/test_redis.py

# Run specific test class
pytest tests/test_redis.py::TestRedisUtils

# Run specific test
pytest tests/test_redis.py::TestRedisUtils::test_cache_data

# Run tests with coverage
pytest --cov=. tests/test_redis.py

# Run tests in parallel
pytest -n auto tests/test_redis.py
```

## Performance Considerations

1. **Connection Pooling**: The Redis manager uses connection pooling to efficiently manage Redis connections.

2. **Batch Operations**: Use batch operations (`cache_batch`, `get_cached_batch`) for better performance when dealing with multiple items.

3. **Key Design**: Use meaningful prefixes and identifiers for better key organization and management.

4. **Expiration**: Set appropriate expiration times for cached data to prevent memory issues.

5. **Serialization**: The Redis manager uses efficient serialization methods for different data types.

## Error Handling

The Redis integration includes comprehensive error handling:

1. **Connection Errors**: Automatic retry mechanism with exponential backoff.

2. **Serialization Errors**: Proper handling of serialization/deserialization errors.

3. **Cache Misses**: Graceful handling of cache misses with fallback to fresh data.

4. **Logging**: Detailed logging for debugging and monitoring.

## Monitoring

Monitor Redis performance using the provided utilities:

```python
# Get memory usage
memory_usage = redis_utils.get_memory_usage()

# Get Redis stats
stats = redis_utils.get_stats()

# Scan keys
keys = redis_utils.scan_keys(
    prefix="user_data",
    pattern="user_*"
)
```

## Best Practices

1. **Key Naming**: Use consistent and meaningful key names with appropriate prefixes.

2. **Data Types**: Choose appropriate data types for different use cases (strings, hashes, sets, etc.).

3. **Expiration**: Set reasonable expiration times based on data volatility.

4. **Error Handling**: Always handle potential Redis errors in your application code.

5. **Monitoring**: Regularly monitor Redis performance and memory usage.

6. **Testing**: Write comprehensive tests for your Redis integration.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 