# FastAPI Dependency Injection System Guide

A comprehensive guide to managing state and shared resources using FastAPI's dependency injection system for the HeyGen AI API.

## 🎯 Overview

This guide covers the complete FastAPI dependency injection system designed to manage:
- **Database Connections**: Async database sessions and connection pooling
- **Cache Systems**: Multi-level caching with Redis and memory
- **Service Dependencies**: Business logic services with proper lifecycle management
- **Security Components**: Authentication, authorization, and session management
- **Configuration Management**: Centralized configuration with multiple sources
- **Resource Management**: Shared resources with proper cleanup and monitoring

## 📋 Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Components](#core-components)
3. [Dependency Types](#dependency-types)
4. [Configuration Management](#configuration-management)
5. [Database Dependencies](#database-dependencies)
6. [Cache Dependencies](#cache-dependencies)
7. [Service Dependencies](#service-dependencies)
8. [Security Dependencies](#security-dependencies)
9. [Resource Management](#resource-management)
10. [FastAPI Integration](#fastapi-integration)
11. [Best Practices](#best-practices)
12. [Monitoring and Health Checks](#monitoring-and-health-checks)
13. [Error Handling](#error-handling)
14. [Performance Optimization](#performance-optimization)

## 🏗️ System Architecture

### **Layered Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                      │
├─────────────────────────────────────────────────────────────┤
│                 Dependency Injection Layer                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐ │
│  │   Services  │ │   Caches    │ │  Security   │ │ Config  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────┘ │
├─────────────────────────────────────────────────────────────┤
│                  Resource Management Layer                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐ │
│  │  Database   │ │    Redis    │ │ External    │ │ Files   │ │
│  │ Resources   │ │ Resources   │ │   APIs      │ │ System  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Infrastructure Layer                      │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐ │
│  │ PostgreSQL  │ │    Redis    │ │ HTTP APIs   │ │ Storage │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### **Dependency Scopes**

1. **Request Scope**: New instance per request
2. **Session Scope**: Instance shared within user session
3. **Application Scope**: Single instance per application
4. **Singleton Scope**: Single instance across all applications

## 🔧 Core Components

### **1. Dependency Manager**

The main orchestrator for all dependencies:

```python
from api.dependency_injection.dependency_manager import FastAPIDependencyManager

# Initialize dependency manager
dependency_manager = FastAPIDependencyManager(app)

# Register dependencies
dependency_manager.register_database("main_db", "postgresql+asyncpg://...")
dependency_manager.register_redis("cache", "redis://localhost:6379")
dependency_manager.register_service("video_service", VideoService)
dependency_manager.register_security("auth", security_manager)
dependency_manager.register_config("app_config", config_manager)
```

### **2. Resource Manager**

Manages shared resources with proper lifecycle:

```python
from api.dependency_injection.resource_manager import ResourceManager, DatabaseResource

# Create resource manager
resource_manager = ResourceManager()

# Register database resource
db_resource = DatabaseResource(
    ResourceConfig(
        name="main_database",
        resource_type=ResourceType.DATABASE,
        max_connections=20,
        connection_timeout=30.0
    ),
    "postgresql+asyncpg://user:password@localhost/heygen_ai"
)

resource_manager.register_resource("main_db", db_resource)
```

### **3. Service Container**

Manages service dependencies and their lifecycle:

```python
from api.dependency_injection.service_container import ServiceContainer

# Create service container
service_container = ServiceContainer()

# Register services
service_container.register_service(
    name="video_service",
    service_class=VideoService,
    scope=ServiceScope.REQUEST,
    dependencies=["database", "cache"]
)
```

### **4. Cache Manager**

Handles multi-level caching:

```python
from api.dependency_injection.cache_manager import CacheManager, HybridCache

# Create cache manager
cache_manager = CacheManager()

# Register hybrid cache (memory + Redis)
hybrid_cache = HybridCache(
    CacheConfig(
        cache_type=CacheType.HYBRID,
        strategy=CacheStrategy.LRU,
        max_size=1000,
        default_ttl=300
    ),
    "redis://localhost:6379"
)

cache_manager.register_cache("main_cache", hybrid_cache)
```

### **5. Configuration Manager**

Centralized configuration management:

```python
from api.dependency_injection.config_manager import ConfigManager, FileConfigSource

# Create configuration manager
config_manager = ConfigManager()

# Add configuration sources
config_manager.add_source(FileConfigSource("config/app.json", ConfigFormat.JSON))
config_manager.add_source(EnvironmentConfigSource("HEYGEN_"))

# Initialize and load configuration
await config_manager.initialize()
```

### **6. Security Manager**

Handles authentication and authorization:

```python
from api.dependency_injection.security_manager import SecurityManager, SecurityConfig

# Create security manager
security_config = SecurityConfig(
    secret_key="your-secret-key",
    algorithm="HS256",
    access_token_expire_minutes=30
)

security_manager = SecurityManager(security_config)
await security_manager.initialize()
```

## 🔄 Dependency Types

### **1. Database Dependencies**

```python
from api.dependency_injection.dependency_manager import DatabaseDependency

# Create database dependency
db_dependency = DatabaseDependency(
    DependencyConfig(
        scope=DependencyScope.APPLICATION,
        priority=DependencyPriority.CRITICAL
    ),
    "postgresql+asyncpg://user:password@localhost/heygen_ai"
)

# Register with container
container.register_dependency("database", db_dependency)

# Use in FastAPI
async def get_db_session():
    async with db_dependency.get_session() as session:
        yield session
```

### **2. Cache Dependencies**

```python
from api.dependency_injection.dependency_manager import CacheDependency

# Create cache dependency
cache_dependency = CacheDependency(
    DependencyConfig(
        scope=DependencyScope.APPLICATION,
        priority=DependencyPriority.HIGH
    ),
    cache_manager
)

# Register with container
container.register_dependency("cache", cache_dependency)

# Use in FastAPI
async def get_cache():
    return cache_dependency.get_cache_manager()
```

### **3. Service Dependencies**

```python
from api.dependency_injection.dependency_manager import ServiceDependency

# Create service dependency
video_service_dependency = ServiceDependency(
    DependencyConfig(
        scope=DependencyScope.REQUEST,
        priority=DependencyPriority.NORMAL
    ),
    VideoService,
    database=db_dependency,
    cache=cache_dependency
)

# Register with container
container.register_dependency("video_service", video_service_dependency)

# Use in FastAPI
async def get_video_service():
    return video_service_dependency.get_service()
```

### **4. Security Dependencies**

```python
from api.dependency_injection.dependency_manager import SecurityDependency

# Create security dependency
security_dependency = SecurityDependency(
    DependencyConfig(
        scope=DependencyScope.APPLICATION,
        priority=DependencyPriority.CRITICAL
    ),
    security_manager
)

# Register with container
container.register_dependency("security", security_dependency)

# Use in FastAPI
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    return await security_dependency.get_current_user(credentials)
```

## ⚙️ Configuration Management

### **Configuration Sources**

```python
# Environment variables
env_source = EnvironmentConfigSource("HEYGEN_")

# JSON file
json_source = FileConfigSource("config/app.json", ConfigFormat.JSON)

# YAML file
yaml_source = FileConfigSource("config/app.yaml", ConfigFormat.YAML)

# Add to configuration manager
config_manager.add_source(env_source)
config_manager.add_source(json_source)
config_manager.add_source(yaml_source)
```

### **Configuration Validation**

```python
# Define validation rules
validation_rules = {
    "database.url": {
        "type": "string",
        "required": True,
        "pattern": r"^postgresql\+asyncpg://.*$"
    },
    "redis.url": {
        "type": "string",
        "required": True,
        "pattern": r"^redis://.*$"
    },
    "security.secret_key": {
        "type": "string",
        "min_length": 32,
        "required": True
    },
    "app.debug": {
        "type": "boolean",
        "default": False
    }
}

# Apply validation
for key, rules in validation_rules.items():
    config_manager.set_validation_rules(key, rules)
```

### **Configuration Usage**

```python
# Get configuration values
database_url = config_manager.get_setting("database.url")
redis_url = config_manager.get_setting("redis.url")
secret_key = config_manager.get_setting("security.secret_key")
debug_mode = config_manager.get_setting("app.debug", default=False)

# Get settings by prefix
database_settings = config_manager.get_settings_by_prefix("database.")
security_settings = config_manager.get_settings_by_prefix("security.")
```

## 🗄️ Database Dependencies

### **Database Resource Configuration**

```python
# Create database resource
db_resource = DatabaseResource(
    ResourceConfig(
        name="main_database",
        resource_type=ResourceType.DATABASE,
        max_connections=20,
        connection_timeout=30.0,
        retry_attempts=3,
        health_check_interval=60.0
    ),
    "postgresql+asyncpg://user:password@localhost/heygen_ai"
)

# Register with resource manager
resource_manager.register_resource("main_db", db_resource)
```

### **Database Session Management**

```python
# Get database session
async with resource_manager.get_resource_connection("main_db") as session:
    # Use session for database operations
    result = await session.execute("SELECT * FROM users")
    users = result.fetchall()

# Use in FastAPI dependency
async def get_db_session():
    async with resource_manager.get_resource_connection("main_db") as session:
        yield session
```

### **Database Health Monitoring**

```python
# Get database health status
health_status = db_resource.get_health_status()

# Monitor connection pool
pool_stats = {
    "active_connections": health_status["active_connections"],
    "max_connections": health_status["max_connections"],
    "connection_utilization": health_status["active_connections"] / health_status["max_connections"]
}
```

## 🗂️ Cache Dependencies

### **Cache Configuration**

```python
# Memory cache
memory_cache = MemoryCache(
    CacheConfig(
        cache_type=CacheType.MEMORY,
        strategy=CacheStrategy.LRU,
        max_size=1000,
        default_ttl=300
    )
)

# Redis cache
redis_cache = RedisCache(
    CacheConfig(
        cache_type=CacheType.REDIS,
        strategy=CacheStrategy.TTL,
        default_ttl=3600
    ),
    "redis://localhost:6379"
)

# Hybrid cache
hybrid_cache = HybridCache(
    CacheConfig(
        cache_type=CacheType.HYBRID,
        strategy=CacheStrategy.LRU,
        max_size=1000,
        default_ttl=300
    ),
    "redis://localhost:6379"
)
```

### **Cache Usage**

```python
# Get cache instance
cache = cache_manager.get_cache("main_cache")

# Set value
await cache.set("user:123", user_data, ttl=3600)

# Get value
user_data = await cache.get("user:123")

# Delete value
await cache.delete("user:123")

# Clear cache
await cache.clear()
```

### **Cache Decorators**

```python
from api.dependency_injection.cache_manager import cached

@cached(ttl=300, cache_name="main_cache")
async def get_user_profile(user_id: int):
    # This function result will be cached for 5 minutes
    return await user_service.get_profile(user_id)
```

## 🔧 Service Dependencies

### **Service Registration**

```python
# Register service with dependencies
service_container.register_service(
    name="video_service",
    service_class=VideoService,
    scope=ServiceScope.REQUEST,
    dependencies=["database", "cache", "config"],
    auto_initialize=True
)

# Register service with factory function
def create_video_service(database, cache, config):
    return VideoService(
        database=database,
        cache=cache,
        config=config,
        api_key=config.get_setting("video.api_key")
    )

service_container.register_service(
    name="video_service",
    service_class=VideoService,
    scope=ServiceScope.REQUEST,
    dependencies=["database", "cache", "config"],
    factory_function=create_video_service
)
```

### **Service Usage**

```python
# Get service instance
video_service = await service_container.get_service("video_service")

# Use service
videos = await video_service.get_user_videos(user_id)
video = await video_service.create_video(video_data)
```

### **Service Lifecycle**

```python
# Service initialization
await service_container.initialize_all()

# Service cleanup
await service_container.cleanup_all()

# Get service health status
health_status = service_container.get_health_status()
```

## 🔒 Security Dependencies

### **Authentication Setup**

```python
# Create authentication manager
auth_manager = AuthenticationManager(security_config)

# Set user service
auth_manager.set_user_service(user_service)

# Authenticate user
user = await auth_manager.authenticate_user("username", "password")

# Create tokens
access_token = auth_manager.create_access_token(user)
refresh_token = auth_manager.create_refresh_token(user)
```

### **Authorization Setup**

```python
# Create authorization manager
authz_manager = AuthorizationManager(security_config)

# Check permissions
has_permission = authz_manager.has_permission(user, "read:videos")
has_role = authz_manager.has_role(user, "admin")

# Get user permissions
permissions = authz_manager.get_user_permissions(user)
```

### **Session Management**

```python
# Create session manager
session_manager = SessionManager(security_config)

# Create session
session = session_manager.create_session(
    user_id=user.id,
    ip_address="192.168.1.1",
    user_agent="Mozilla/5.0..."
)

# Get session
user_session = session_manager.get_session(session.session_id)

# Invalidate session
session_manager.invalidate_session(session.session_id)
```

### **Security Dependencies in FastAPI**

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

# Security dependencies
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    return await security_manager.authenticate_user(credentials)

async def require_admin_role(current_user: User = Depends(get_current_user)):
    if not security_manager.authz_manager.has_role(current_user, "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

# Use in endpoints
@router.get("/admin/users")
async def get_users(current_user: User = Depends(require_admin_role)):
    return await user_service.get_all_users()
```

## 📦 Resource Management

### **Resource Lifecycle**

```python
# Initialize resources
await resource_manager.initialize_all()

# Get resource connection
async with resource_manager.get_resource_connection("main_db") as connection:
    # Use connection
    pass

# Cleanup resources
await resource_manager.cleanup_all()
```

### **Resource Health Monitoring**

```python
# Get resource health status
health_status = resource_manager.get_health_status()

# Monitor specific resource
db_health = health_status["resources"]["main_db"]
redis_health = health_status["resources"]["cache"]

# Check resource status
if db_health["status"] != "ready":
    logger.error("Database resource is not healthy")
```

### **Resource Configuration**

```python
# Database resource
db_config = ResourceConfig(
    name="main_database",
    resource_type=ResourceType.DATABASE,
    max_connections=20,
    connection_timeout=30.0,
    retry_attempts=3,
    health_check_interval=60.0,
    auto_reconnect=True
)

# Redis resource
redis_config = ResourceConfig(
    name="cache",
    resource_type=ResourceType.CACHE,
    max_connections=10,
    connection_timeout=5.0,
    retry_attempts=3,
    health_check_interval=30.0
)
```

## 🚀 FastAPI Integration

### **Application Setup**

```python
from fastapi import FastAPI
from api.dependency_injection.dependency_manager import FastAPIDependencyManager

# Create FastAPI app
app = FastAPI(title="HeyGen AI API")

# Initialize dependency manager
dependency_manager = FastAPIDependencyManager(app)

# Register dependencies
dependency_manager.register_database("main_db", "postgresql+asyncpg://...")
dependency_manager.register_redis("cache", "redis://localhost:6379")
dependency_manager.register_service("video_service", VideoService)
dependency_manager.register_security("auth", security_manager)
dependency_manager.register_config("config", config_manager)
```

### **Dependency Injection in Endpoints**

```python
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/v1/videos", tags=["videos"])

@router.get("/")
async def get_videos(
    db: AsyncSession = Depends(dependency_manager.get_dependency_function("database")),
    video_service = Depends(dependency_manager.get_dependency_function("video_service")),
    current_user: User = Depends(dependency_manager.get_dependency_function("security"))
):
    """Get user videos."""
    return await video_service.get_user_videos(current_user.id, db)

@router.post("/")
async def create_video(
    video_data: VideoCreate,
    video_service = Depends(dependency_manager.get_dependency_function("video_service")),
    current_user: User = Depends(dependency_manager.get_dependency_function("security"))
):
    """Create a new video."""
    return await video_service.create_video(video_data, current_user.id)
```

### **Health Check Endpoints**

```python
# Add health check endpoints
app.add_api_route("/health", dependency_manager.get_health_endpoint())
app.add_api_route("/health/dependencies", dependency_manager.get_health_endpoint())
app.add_api_route("/health/resources", resource_manager.get_health_endpoint)
app.add_api_route("/health/services", service_container.get_health_endpoint)
app.add_api_route("/health/cache", cache_manager.get_health_endpoint)
app.add_api_route("/health/config", config_manager.get_stats)
app.add_api_route("/health/security", security_manager.get_stats)
```

## ✅ Best Practices

### **1. Dependency Organization**

```python
# Organize dependencies by type
class Dependencies:
    # Database dependencies
    DATABASE = "database"
    REDIS = "redis"
    
    # Service dependencies
    VIDEO_SERVICE = "video_service"
    USER_SERVICE = "user_service"
    ANALYTICS_SERVICE = "analytics_service"
    
    # Security dependencies
    AUTH = "auth"
    SESSION = "session"
    
    # Configuration dependencies
    CONFIG = "config"
    CACHE = "cache"

# Use constants for dependency names
dependency_manager.register_database(Dependencies.DATABASE, database_url)
dependency_manager.register_service(Dependencies.VIDEO_SERVICE, VideoService)
```

### **2. Error Handling**

```python
# Proper error handling in dependencies
async def get_database_session():
    try:
        async with db_dependency.get_session() as session:
            yield session
    except Exception as e:
        logger.error(f"Database session error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable"
        )

# Graceful degradation
async def get_cache_with_fallback():
    try:
        return await cache_dependency.get_cache_manager()
    except Exception as e:
        logger.warning(f"Cache unavailable, using fallback: {e}")
        return None
```

### **3. Configuration Management**

```python
# Use configuration for dependency setup
database_url = config_manager.get_setting("database.url")
redis_url = config_manager.get_setting("redis.url")
secret_key = config_manager.get_setting("security.secret_key")

# Validate required configuration
required_settings = ["database.url", "redis.url", "security.secret_key"]
for setting in required_settings:
    if not config_manager.get_setting(setting):
        raise ValueError(f"Required setting {setting} not found")
```

### **4. Resource Cleanup**

```python
# Proper cleanup in application shutdown
@app.on_event("shutdown")
async def shutdown_event():
    await dependency_manager.cleanup_all()
    await resource_manager.cleanup_all()
    await service_container.cleanup_all()
    await cache_manager.cleanup_all()
    await config_manager.cleanup()
    await security_manager.cleanup()
```

## 📊 Monitoring and Health Checks

### **Health Check Implementation**

```python
# Comprehensive health check
async def health_check():
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "dependencies": dependency_manager.get_health_status(),
        "resources": resource_manager.get_health_status(),
        "services": service_container.get_health_status(),
        "cache": cache_manager.get_health_status(),
        "config": config_manager.get_stats(),
        "security": security_manager.get_stats()
    }
    
    # Check overall health
    all_healthy = True
    for component, status in health_status.items():
        if component != "status" and component != "timestamp" and component != "version":
            if not status.get("initialized", True):
                all_healthy = False
                health_status["status"] = "unhealthy"
    
    return health_status
```

### **Metrics Collection**

```python
# Collect dependency metrics
def collect_metrics():
    metrics = {
        "dependencies": {
            "total": len(dependency_manager.dependencies),
            "initialized": sum(1 for d in dependency_manager.dependencies.values() if d._is_initialized)
        },
        "resources": {
            "total": len(resource_manager.resources),
            "healthy": sum(1 for r in resource_manager.resources.values() if r.stats.status == ResourceStatus.READY)
        },
        "cache": {
            "hits": sum(c.stats.hits for c in cache_manager.caches.values()),
            "misses": sum(c.stats.misses for c in cache_manager.caches.values())
        }
    }
    return metrics
```

## 🚨 Error Handling

### **Dependency Error Handling**

```python
# Retry mechanism for dependencies
async def get_dependency_with_retry(dependency_name: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return await dependency_manager.get_dependency(dependency_name)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Dependency {dependency_name} attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

### **Graceful Degradation**

```python
# Graceful degradation for cache
async def get_data_with_cache_fallback(key: str):
    try:
        # Try cache first
        cache = await cache_manager.get_cache("main_cache")
        data = await cache.get(key)
        if data:
            return data
    except Exception as e:
        logger.warning(f"Cache unavailable: {e}")
    
    # Fallback to database
    try:
        db = await dependency_manager.get_dependency("database")
        data = await db.fetch_data(key)
        return data
    except Exception as e:
        logger.error(f"Database unavailable: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporarily unavailable"
        )
```

## ⚡ Performance Optimization

### **Connection Pooling**

```python
# Optimize database connection pooling
db_config = ResourceConfig(
    name="main_database",
    resource_type=ResourceType.DATABASE,
    max_connections=20,
    connection_timeout=30.0,
    retry_attempts=3,
    health_check_interval=60.0,
    auto_reconnect=True
)
```

### **Caching Strategies**

```python
# Multi-level caching
cache_config = CacheConfig(
    cache_type=CacheType.HYBRID,  # Memory + Redis
    strategy=CacheStrategy.LRU,
    max_size=1000,
    default_ttl=300,
    enable_compression=True
)
```

### **Lazy Loading**

```python
# Lazy load dependencies
service_config = ServiceConfig(
    name="video_service",
    service_class=VideoService,
    scope=ServiceScope.REQUEST,
    lazy_loading=True,  # Only initialize when needed
    auto_initialize=False
)
```

## 📝 Example Implementation

### **Complete Setup Example**

```python
from fastapi import FastAPI, Depends
from api.dependency_injection.dependency_manager import FastAPIDependencyManager
from api.dependency_injection.resource_manager import ResourceManager
from api.dependency_injection.service_container import ServiceContainer
from api.dependency_injection.cache_manager import CacheManager
from api.dependency_injection.config_manager import ConfigManager
from api.dependency_injection.security_manager import SecurityManager

# Create FastAPI app
app = FastAPI(title="HeyGen AI API")

# Initialize managers
dependency_manager = FastAPIDependencyManager(app)
resource_manager = ResourceManager()
service_container = ServiceContainer()
cache_manager = CacheManager()
config_manager = ConfigManager()
security_manager = SecurityManager(SecurityConfig(secret_key="your-secret-key"))

# Setup configuration
config_manager.add_source(FileConfigSource("config/app.json", ConfigFormat.JSON))
config_manager.add_source(EnvironmentConfigSource("HEYGEN_"))

# Register dependencies
dependency_manager.register_database("main_db", "postgresql+asyncpg://...")
dependency_manager.register_redis("cache", "redis://localhost:6379")
dependency_manager.register_service("video_service", VideoService)
dependency_manager.register_security("auth", security_manager)
dependency_manager.register_config("config", config_manager)

# Register services
service_container.register_service(
    name="video_service",
    service_class=VideoService,
    scope=ServiceScope.REQUEST,
    dependencies=["database", "cache"]
)

# Register caches
cache_manager.register_cache("main_cache", HybridCache(...))

# API endpoints
@app.get("/videos")
async def get_videos(
    video_service = Depends(dependency_manager.get_dependency_function("video_service")),
    current_user: User = Depends(dependency_manager.get_dependency_function("security"))
):
    return await video_service.get_user_videos(current_user.id)

@app.get("/health")
async def health_check():
    return {
        "dependencies": dependency_manager.get_health_status(),
        "services": service_container.get_health_status(),
        "cache": cache_manager.get_health_status()
    }
```

## 🎯 Summary

This comprehensive FastAPI dependency injection system provides:

### **Key Benefits**

1. **Centralized Management**: All dependencies managed in one place
2. **Lifecycle Management**: Proper initialization and cleanup
3. **Resource Optimization**: Connection pooling and caching
4. **Security Integration**: Built-in authentication and authorization
5. **Configuration Management**: Multiple sources with validation
6. **Health Monitoring**: Comprehensive health checks and metrics
7. **Error Handling**: Graceful degradation and retry mechanisms
8. **Performance**: Optimized resource usage and caching

### **Best Practices Implemented**

1. **Dependency Scoping**: Proper scope management for different use cases
2. **Resource Cleanup**: Automatic cleanup on application shutdown
3. **Error Handling**: Comprehensive error handling with fallbacks
4. **Monitoring**: Health checks and metrics collection
5. **Configuration**: Centralized configuration with validation
6. **Security**: Integrated authentication and authorization
7. **Performance**: Connection pooling and caching strategies

### **Next Steps**

1. **Integration**: Integrate with existing HeyGen AI services
2. **Customization**: Customize dependencies for specific requirements
3. **Testing**: Add comprehensive tests for all components
4. **Documentation**: Create API documentation for endpoints
5. **Monitoring**: Set up monitoring and alerting
6. **Deployment**: Configure for production deployment

This system provides a robust foundation for managing state and shared resources in your FastAPI application, ensuring scalability, reliability, and maintainability. 