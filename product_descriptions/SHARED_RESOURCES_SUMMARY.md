# Shared Resources Dependency Injection System

## Overview

The Shared Resources Dependency Injection System provides a comprehensive, production-ready solution for managing shared resources in FastAPI applications. It implements dependency injection patterns for network sessions, cryptographic backends, database pools, and other shared resources with proper lifecycle management, health monitoring, and error handling.

## Key Features

### 🔧 **Resource Management**
- **Singleton Pattern**: Ensures single instances of shared resources
- **Connection Pooling**: Efficient resource reuse and management
- **Lifecycle Management**: Proper initialization and cleanup
- **Configuration-Driven**: Flexible resource configuration

### 🌐 **Network Sessions**
- **HTTP Sessions**: aiohttp-based with connection pooling
- **WebSocket Sessions**: WebSocket connection management
- **gRPC Sessions**: gRPC client management (extensible)
- **Custom Headers**: Configurable request headers
- **SSL/TLS Support**: Secure connection handling

### 🔐 **Cryptographic Backends**
- **AES Encryption**: AES-256-GCM and AES-256-CBC
- **RSA Operations**: RSA-2048 and RSA-4096
- **Hashing**: SHA-256 and SHA-512
- **Digital Signatures**: RSA signing and verification
- **Key Management**: Automatic key generation and rotation

### 🗄️ **Database & Cache**
- **Database Pools**: SQLAlchemy async connection pools
- **Redis Pools**: Redis connection pooling
- **Connection Limits**: Configurable connection limits
- **Health Checks**: Database connectivity monitoring

### 📊 **Monitoring & Health**
- **Health Checks**: Automated resource health monitoring
- **Metrics Collection**: Performance and usage metrics
- **Error Tracking**: Error counting and reporting
- **Response Time Monitoring**: Latency tracking

### 🔄 **FastAPI Integration**
- **Dependency Injection**: Seamless FastAPI integration
- **Context Managers**: Async context managers for resources
- **Type Safety**: Full type hints and validation
- **Error Handling**: Graceful error handling and fallbacks

## Architecture

### Core Components

```
SharedResourcesContainer
├── Resource Managers
│   ├── HTTPSessionManager
│   ├── WebSocketSessionManager
│   ├── CryptoBackendManager
│   ├── DatabasePoolManager
│   └── RedisPoolManager
├── Configuration
│   ├── SharedResourceConfig
│   ├── ResourceConfig
│   └── CryptoConfig
├── Health Monitoring
│   ├── ResourceHealth
│   └── ResourceMetrics
└── FastAPI Dependencies
    ├── get_http_session()
    ├── get_crypto_backend()
    └── Context Managers
```

### Resource Lifecycle

1. **Initialization**: Resources are created and configured
2. **Health Monitoring**: Continuous health checks run in background
3. **Resource Usage**: FastAPI dependencies provide access to resources
4. **Cleanup**: Automatic cleanup and resource disposal
5. **Shutdown**: Graceful shutdown of all resources

## Configuration

### Basic Configuration

```python
from dependencies.shared_resources import (
    SharedResourceConfig, ResourceConfig, CryptoConfig,
    ResourceType, CryptoAlgorithm
)

config = SharedResourceConfig(
    resources={
        "http_session": ResourceConfig(
            name="http_session",
            resource_type=ResourceType.HTTP_SESSION,
            max_connections=100,
            timeout=30.0,
            custom_headers={"User-Agent": "MyApp/1.0"}
        )
    },
    crypto_configs={
        "default": CryptoConfig(
            algorithm=CryptoAlgorithm.AES_256_GCM,
            key_size=256
        )
    },
    enable_health_checks=True,
    enable_monitoring=True
)
```

### Advanced Configuration

```python
config = SharedResourceConfig(
    resources={
        "http_session": ResourceConfig(
            name="http_session",
            resource_type=ResourceType.HTTP_SESSION,
            max_connections=200,
            timeout=60.0,
            keepalive_timeout=120.0,
            enable_ssl_verification=True,
            custom_headers={
                "User-Agent": "MyApp/1.0",
                "Accept": "application/json"
            },
            health_check_interval=30.0,
            circuit_breaker_threshold=5
        ),
        "database_pool": ResourceConfig(
            name="database_pool",
            resource_type=ResourceType.DATABASE_POOL,
            max_connections=50,
            pool_timeout=10.0
        )
    },
    crypto_configs={
        "aes": CryptoConfig(
            algorithm=CryptoAlgorithm.AES_256_GCM,
            key_size=256,
            enable_hardware_acceleration=True
        ),
        "rsa": CryptoConfig(
            algorithm=CryptoAlgorithm.RSA_2048,
            key_size=2048,
            key_rotation_interval=timedelta(days=30)
        )
    },
    global_timeout=60.0,
    global_max_retries=3,
    enable_monitoring=True,
    enable_health_checks=True,
    resource_cleanup_interval=300.0
)
```

## Usage Patterns

### FastAPI Dependency Injection

```python
from fastapi import FastAPI, Depends
from dependencies.shared_resources import (
    get_http_session, get_crypto_backend
)

app = FastAPI()

@app.get("/api/data")
async def get_data(
    http_session=Depends(get_http_session),
    crypto_backend=Depends(get_crypto_backend)
):
    # Use shared HTTP session
    async with http_session.get("https://api.example.com/data") as response:
        data = await response.json()
    
    # Use crypto backend
    encrypted = await crypto_backend.encrypt(data.encode())
    
    return {"data": data, "encrypted": encrypted.hex()}
```

### Context Managers

```python
from dependencies.shared_resources import (
    http_session_context, crypto_backend_context
)

async def process_data():
    # HTTP session context manager
    async with http_session_context() as session:
        async with session.get("https://api.example.com/data") as response:
            data = await response.json()
    
    # Crypto backend context manager
    async with crypto_backend_context() as backend:
        encrypted = await backend.encrypt(data.encode())
        signature = await backend.sign(data.encode())
    
    return {"encrypted": encrypted.hex(), "signature": signature.hex()}
```

### Named Resources

```python
from dependencies.shared_resources import get_crypto_backend

@app.post("/api/encrypt")
async def encrypt_data(
    data: str,
    aes_backend=Depends(lambda: get_crypto_backend("aes")),
    rsa_backend=Depends(lambda: get_crypto_backend("rsa"))
):
    # Use specific crypto backends
    aes_encrypted = await aes_backend.encrypt(data.encode())
    rsa_encrypted = await rsa_backend.encrypt(data.encode())
    
    return {
        "aes_encrypted": aes_encrypted.hex(),
        "rsa_encrypted": rsa_encrypted.hex()
    }
```

### Health Monitoring

```python
from dependencies.shared_resources import get_all_resource_health

@app.get("/health")
async def health_check():
    health_status = get_all_resource_health()
    
    return {
        "status": "healthy" if all(h.is_healthy for h in health_status.values()) else "unhealthy",
        "resources": {
            name: {
                "is_healthy": health.is_healthy,
                "response_time": health.response_time,
                "error_count": health.error_count
            }
            for name, health in health_status.items()
        }
    }
```

## Resource Types

### HTTP Session Manager

**Features:**
- Connection pooling with configurable limits
- Keep-alive connections
- SSL/TLS support
- Custom headers
- Health checks via HTTP requests
- Automatic cleanup

**Configuration:**
```python
ResourceConfig(
    name="http_session",
    resource_type=ResourceType.HTTP_SESSION,
    max_connections=100,
    timeout=30.0,
    keepalive_timeout=60.0,
    enable_ssl_verification=True,
    custom_headers={"User-Agent": "MyApp/1.0"}
)
```

### Crypto Backend Manager

**Features:**
- Multiple encryption algorithms (AES, RSA)
- Hashing operations (SHA-256, SHA-512)
- Digital signatures
- Automatic key generation
- Hardware acceleration support
- Key rotation

**Configuration:**
```python
CryptoConfig(
    algorithm=CryptoAlgorithm.AES_256_GCM,
    key_size=256,
    salt_length=32,
    iterations=100000,
    enable_hardware_acceleration=True,
    key_rotation_interval=timedelta(days=30)
)
```

### Database Pool Manager

**Features:**
- SQLAlchemy async connection pools
- Configurable pool sizes
- Connection health checks
- Automatic connection cleanup
- Support for multiple databases

**Configuration:**
```python
ResourceConfig(
    name="database_pool",
    resource_type=ResourceType.DATABASE_POOL,
    max_connections=50,
    pool_timeout=10.0
)
```

### Redis Pool Manager

**Features:**
- Redis connection pooling
- Configurable connection limits
- Health checks via PING
- Automatic reconnection
- Support for Redis clusters

**Configuration:**
```python
ResourceConfig(
    name="redis_pool",
    resource_type=ResourceType.REDIS_POOL,
    max_connections=20,
    timeout=5.0
)
```

## Best Practices

### 1. Resource Configuration

- **Set appropriate limits**: Configure max_connections based on your application needs
- **Use timeouts**: Set reasonable timeouts to prevent hanging connections
- **Enable health checks**: Monitor resource health for early problem detection
- **Configure cleanup intervals**: Regular cleanup prevents resource leaks

### 2. Error Handling

```python
@app.get("/api/data")
async def get_data(http_session=Depends(get_http_session)):
    try:
        async with http_session.get("https://api.example.com/data") as response:
            if response.status == 200:
                return await response.json()
            else:
                raise HTTPException(status_code=response.status)
    except Exception as e:
        logger.error("Failed to fetch data", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
```

### 3. Health Monitoring

```python
@app.get("/health")
async def health_check():
    health_status = get_all_resource_health()
    
    # Check if any resources are unhealthy
    unhealthy_resources = [
        name for name, health in health_status.items() 
        if not health.is_healthy
    ]
    
    if unhealthy_resources:
        logger.warning("Unhealthy resources detected", resources=unhealthy_resources)
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "unhealthy_resources": unhealthy_resources}
        )
    
    return {"status": "healthy"}
```

### 4. Resource Cleanup

```python
@app.on_event("shutdown")
async def shutdown_event():
    await shutdown_shared_resources()
    logger.info("Shared resources cleaned up")
```

### 5. Performance Optimization

- **Use connection pooling**: Reuse connections instead of creating new ones
- **Implement circuit breakers**: Prevent cascading failures
- **Monitor metrics**: Track resource usage and performance
- **Set appropriate timeouts**: Balance between responsiveness and reliability

## Testing

### Unit Tests

```python
import pytest
from dependencies.shared_resources import HTTPSessionManager, ResourceConfig

@pytest.mark.asyncio
async def test_http_session_manager():
    config = ResourceConfig(
        name="test",
        resource_type=ResourceType.HTTP_SESSION
    )
    manager = HTTPSessionManager(config)
    
    # Test session creation
    session = await manager.get_session()
    assert session is not None
    
    # Test health check
    health = await manager.health_check()
    assert isinstance(health.is_healthy, bool)
    
    # Test cleanup
    await manager.cleanup()
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_full_workflow():
    config = create_test_config()
    await initialize_shared_resources(config)
    
    try:
        # Test HTTP session
        async with http_session_context() as session:
            assert session is not None
        
        # Test crypto backend
        async with crypto_backend_context() as backend:
            test_data = b"test"
            encrypted = await backend.encrypt(test_data)
            decrypted = await backend.decrypt(encrypted)
            assert decrypted == test_data
    
    finally:
        await shutdown_shared_resources()
```

## Monitoring and Observability

### Health Checks

The system provides comprehensive health monitoring:

- **Resource Health**: Individual resource health status
- **Response Times**: Latency tracking for each resource
- **Error Counts**: Error tracking and reporting
- **Uptime**: Resource availability tracking

### Metrics

Key metrics tracked:

- **Total Requests**: Number of requests made
- **Successful Requests**: Successful operations
- **Failed Requests**: Failed operations
- **Average Response Time**: Mean latency
- **Active Connections**: Current connection count
- **Peak Connections**: Maximum connections used

### Logging

Structured logging with context:

```python
logger.info("HTTP request completed", 
           url=url, 
           status_code=response.status,
           response_time=response_time)
```

## Security Considerations

### 1. Cryptographic Operations

- **Use strong algorithms**: AES-256-GCM, RSA-2048+
- **Key rotation**: Implement automatic key rotation
- **Secure key storage**: Store keys in secure environments
- **Input validation**: Validate all inputs before processing

### 2. Network Security

- **SSL/TLS**: Enable SSL verification for HTTPS
- **Custom headers**: Use appropriate User-Agent and headers
- **Connection limits**: Prevent resource exhaustion
- **Timeout configuration**: Set appropriate timeouts

### 3. Access Control

- **Resource isolation**: Separate resources by environment
- **Permission checks**: Implement proper access controls
- **Audit logging**: Log all resource access
- **Error handling**: Don't expose sensitive information in errors

## Deployment Considerations

### 1. Environment Configuration

```python
# Development
config = SharedResourceConfig(
    resources={...},
    enable_health_checks=True,
    resource_cleanup_interval=60.0
)

# Production
config = SharedResourceConfig(
    resources={...},
    enable_health_checks=True,
    resource_cleanup_interval=300.0,
    global_timeout=60.0,
    global_max_retries=3
)
```

### 2. Resource Scaling

- **Connection pools**: Scale based on expected load
- **Health check intervals**: Adjust based on requirements
- **Cleanup intervals**: Balance between cleanup and performance
- **Monitoring**: Implement proper monitoring and alerting

### 3. High Availability

- **Circuit breakers**: Implement circuit breaker patterns
- **Fallback mechanisms**: Provide fallback resources
- **Health monitoring**: Continuous health monitoring
- **Graceful degradation**: Handle resource failures gracefully

## Troubleshooting

### Common Issues

1. **Connection Timeouts**
   - Check network connectivity
   - Verify timeout configurations
   - Monitor resource health

2. **Resource Exhaustion**
   - Increase connection limits
   - Implement connection pooling
   - Monitor resource usage

3. **Health Check Failures**
   - Verify resource availability
   - Check configuration settings
   - Review error logs

4. **Performance Issues**
   - Monitor response times
   - Check connection pool usage
   - Optimize resource configuration

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check resource health
health_status = get_all_resource_health()
for name, health in health_status.items():
    print(f"{name}: {health.is_healthy} ({health.response_time}s)")
```

## Conclusion

The Shared Resources Dependency Injection System provides a robust, scalable solution for managing shared resources in FastAPI applications. It offers:

- **Comprehensive resource management** with proper lifecycle handling
- **Flexible configuration** for different environments and requirements
- **Built-in monitoring and health checks** for operational visibility
- **Seamless FastAPI integration** with dependency injection
- **Production-ready features** including error handling and security

By following the patterns and best practices outlined in this document, you can build reliable, performant applications that efficiently manage shared resources while maintaining high availability and security standards. 