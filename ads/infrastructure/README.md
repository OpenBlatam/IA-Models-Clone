# Infrastructure Layer - Unified Ads System

## Overview

The Infrastructure Layer provides a unified, Clean Architecture-compliant foundation for the ads feature, consolidating scattered database, storage, cache, and external service implementations into a cohesive, maintainable system.

## 🏗️ Architecture

The infrastructure layer follows Clean Architecture principles with clear separation of concerns:

```
infrastructure/
├── __init__.py              # Package initialization and exports
├── database.py              # Database management and repository interfaces
├── storage.py               # File storage with strategy pattern
├── cache.py                 # Cache management with multiple backends
├── external_services.py     # External service integrations
├── repositories.py          # Repository implementations
├── infrastructure_demo.py   # Comprehensive demonstration
└── README.md               # This documentation
```

## 🗄️ Database Management

### Core Components

- **`DatabaseConfig`**: Configuration for database connections
- **`ConnectionPool`**: Manages database connection pooling
- **`DatabaseManager`**: Orchestrates database operations with retry logic
- **Repository Interfaces**: Abstract contracts for data access

### Features

- **Connection Pooling**: Efficient database connection management
- **Retry Logic**: Automatic retry with exponential backoff
- **Pool Statistics**: Real-time connection pool monitoring
- **Async Support**: Full async/await support throughout

### Usage

```python
from .infrastructure.database import DatabaseManager, DatabaseConfig

# Configure database
db_config = DatabaseConfig(
    url="postgresql+asyncpg://user:pass@localhost/ads_db",
    pool_size=20,
    max_overflow=30
)

# Create manager
db_manager = DatabaseManager(db_config)

# Use in repositories
async with db_manager.get_session() as session:
    # Database operations
    pass
```

## 💾 Storage Management

### Core Components

- **`StorageConfig`**: Configuration for storage backends
- **`StorageStrategy`**: Abstract interface for storage operations
- **`LocalStorageStrategy`**: Local file system implementation
- **`CloudStorageStrategy`**: Cloud storage placeholder (future)
- **`FileStorageManager`**: High-level storage management

### Features

- **Strategy Pattern**: Easy switching between storage backends
- **File Validation**: Type and size validation
- **Metadata Support**: JSON metadata storage
- **Async Operations**: Non-blocking file operations
- **Temp File Cleanup**: Automatic cleanup of temporary files

### Usage

```python
from .infrastructure.storage import FileStorageManager, StorageConfig, StorageType

# Configure storage
storage_config = StorageConfig(
    base_path="./storage",
    cache_ttl=1800,
    max_file_size=100 * 1024 * 1024,  # 100MB
    storage_type=StorageType.LOCAL
)

# Create manager
storage_manager = FileStorageManager(storage_config)

# Save file
filename = await storage_manager.save_file(file_content, "document.pdf")

# Get file info
file_info = await storage_manager.get_file_info(filename)
```

## ⚡ Cache Management

### Core Components

- **`CacheConfig`**: Configuration for cache backends
- **`CacheStrategy`**: Abstract interface for cache operations
- **`MemoryCacheStrategy`**: In-memory cache implementation
- **`RedisCacheStrategy`**: Redis cache implementation
- **`CacheManager`**: High-level cache management

### Features

- **Multiple Backends**: Memory and Redis support
- **TTL Management**: Automatic expiration handling
- **Statistics**: Hit/miss rates and performance metrics
- **Pattern Invalidation**: Bulk cache invalidation
- **Fallback Support**: Graceful degradation

### Usage

```python
from .infrastructure.cache import CacheManager, CacheConfig, CacheType

# Configure cache
cache_config = CacheConfig(
    default_ttl=300,
    max_size=1000,
    cache_type=CacheType.REDIS
)

# Create manager
cache_manager = CacheManager(cache_config)

# Cache operations
await cache_manager.set("key", value, ttl=600)
cached_value = await cache_manager.get("key")
await cache_manager.delete("key")
```

## 🌐 External Services

### Core Components

- **`ExternalServiceManager`**: Centralized service management
- **`AIProviderService`**: AI service integrations
- **`AnalyticsService`**: Analytics provider integrations
- **`NotificationService`**: Notification service integrations
- **`ExternalServiceConfig`**: Service configuration

### Features

- **Service Registration**: Dynamic service registration
- **Rate Limiting**: Built-in rate limiting support
- **Health Monitoring**: Service status tracking
- **Error Handling**: Comprehensive error management
- **HTTP Session Management**: Efficient HTTP connections

### Usage

```python
from .infrastructure.external_services import (
    ExternalServiceManager, AIProviderService, ExternalServiceConfig, ServiceType
)

# Create service manager
service_manager = ExternalServiceManager()

# Register AI provider
ai_config = ExternalServiceConfig(
    service_type=ServiceType.AI_PROVIDER,
    base_url="https://api.openai.com",
    api_key="your_key",
    rate_limit=100
)

ai_service = AIProviderService(service_manager)
await ai_service.register_provider("openai", ai_config)

# Use service
response = await ai_service.generate_content("openai", "Create an ad", "gpt-4")
```

## 📚 Repository Implementations

### Core Components

- **`AdsRepositoryImpl`**: Ads data access implementation
- **`CampaignRepositoryImpl`**: Campaign data access implementation
- **`GroupRepositoryImpl`**: Ad group data access implementation
- **`PerformanceRepositoryImpl`**: Performance data access implementation
- **`AnalyticsRepositoryImpl`**: Analytics data access implementation
- **`OptimizationRepositoryImpl`**: Optimization data access implementation
- **`RepositoryFactory`**: Factory for creating repository instances

### Features

- **Clean Interfaces**: Implements repository interfaces from domain layer
- **Database Integration**: Uses DatabaseManager for operations
- **Error Handling**: Comprehensive error handling and logging
- **Statistics**: User statistics and analytics
- **Factory Pattern**: Easy repository instantiation

### Usage

```python
from .infrastructure.repositories import RepositoryFactory
from .infrastructure.database import DatabaseManager

# Create repository factory
db_manager = DatabaseManager()
repo_factory = RepositoryFactory(db_manager)

# Create specific repositories
ads_repo = repo_factory.create_ads_repository()
analytics_repo = repo_factory.create_analytics_repository()

# Use repositories
ads_data = await ads_repo.get_by_id(123, user_id=456)
user_stats = await ads_repo.get_user_stats(user_id=456)
```

## 🔧 Configuration

### Database Configuration

```python
@dataclass
class DatabaseConfig:
    url: str
    pool_size: int = 20
    max_overflow: int = 30
    pool_pre_ping: bool = True
    pool_recycle: int = 3600
    pool_timeout: int = 30
    echo: bool = False
    max_retries: int = 100
    retry_delay: float = 1.0
```

### Storage Configuration

```python
@dataclass
class StorageConfig:
    base_path: str = "storage"
    cache_ttl: int = 3600
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: List[str] = field(default_factory=lambda: [...])
    storage_type: StorageType = StorageType.LOCAL
    compression_enabled: bool = True
    encryption_enabled: bool = False
```

### Cache Configuration

```python
@dataclass
class CacheConfig:
    default_ttl: int = 300  # 5 minutes
    max_size: int = 1000
    cache_type: CacheType = CacheType.REDIS
    redis_url: Optional[str] = None
    redis_max_connections: int = 50
    compression_enabled: bool = True
    stats_enabled: bool = True
```

## 🚀 Getting Started

### 1. Basic Setup

```python
from .infrastructure import (
    DatabaseManager, FileStorageManager, CacheManager,
    ExternalServiceManager, RepositoryFactory
)

# Initialize components
db_manager = DatabaseManager()
storage_manager = FileStorageManager()
cache_manager = CacheManager()
service_manager = ExternalServiceManager()
repo_factory = RepositoryFactory(db_manager)
```

### 2. Run Demo

```python
from .infrastructure.infrastructure_demo import InfrastructureSystemDemo

# Run comprehensive demo
demo = InfrastructureSystemDemo()
await demo.run_comprehensive_demo()
```

### 3. Integration with Domain Layer

```python
from .domain.services import AdService
from .infrastructure.repositories import RepositoryFactory

# Create domain service with infrastructure
db_manager = DatabaseManager()
repo_factory = RepositoryFactory(db_manager)
ads_repo = repo_factory.create_ads_repository()

ad_service = AdService(ads_repo)
```

## 📊 Monitoring and Statistics

### Database Statistics

```python
# Get connection pool statistics
pool_stats = await db_manager.get_pool_stats()
print(f"Active connections: {pool_stats['active_connections']}")
```

### Cache Statistics

```python
# Get cache performance statistics
cache_stats = await cache_manager.get_stats()
print(f"Hit rate: {cache_stats['hit_rate']:.2%}")
```

### Storage Statistics

```python
# Get storage usage statistics
storage_stats = await storage_manager.get_storage_stats()
print(f"Total files: {storage_stats['total_files']}")
```

## 🧪 Testing

### Unit Testing

```python
import pytest
from unittest.mock import Mock, AsyncMock

@pytest.mark.asyncio
async def test_cache_manager():
    cache_manager = CacheManager()
    
    # Test cache operations
    await cache_manager.set("test_key", "test_value")
    result = await cache_manager.get("test_key")
    assert result == "test_value"
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_database_integration():
    # Use test database configuration
    db_config = DatabaseConfig(url="sqlite+aiosqlite:///test.db")
    db_manager = DatabaseManager(db_config)
    
    # Test database operations
    # ...
```

## 🔒 Security Features

- **API Key Management**: Secure API key handling
- **Rate Limiting**: Built-in rate limiting protection
- **Input Validation**: Comprehensive input validation
- **Error Sanitization**: Safe error message handling
- **Connection Security**: Secure database connections

## 📈 Performance Features

- **Connection Pooling**: Efficient database connection reuse
- **Caching**: Multi-level caching strategies
- **Async Operations**: Non-blocking I/O operations
- **Batch Operations**: Bulk data processing
- **Lazy Loading**: On-demand resource initialization

## 🔄 Migration from Legacy Code

### Before (Scattered Implementation)

```python
# Old scattered approach
from .db_service import AdsDBService
from .storage import StorageService
from .scalable_api_patterns import CacheManager

# Multiple different implementations
db_service = AdsDBService()
storage_service = StorageService()
cache_manager = CacheManager()
```

### After (Unified Infrastructure)

```python
# New unified approach
from .infrastructure import (
    DatabaseManager, FileStorageManager, CacheManager
)

# Single, consistent interface
db_manager = DatabaseManager()
storage_manager = FileStorageManager()
cache_manager = CacheManager()
```

## 🎯 Benefits

### ✅ **Consolidation**
- Eliminates scattered implementations
- Single source of truth for infrastructure
- Consistent patterns across all components

### ✅ **Maintainability**
- Clean, organized code structure
- Easy to understand and modify
- Comprehensive error handling

### ✅ **Performance**
- Connection pooling and caching
- Async operations throughout
- Optimized resource management

### ✅ **Scalability**
- Strategy pattern for different backends
- Easy to add new storage/cache providers
- Horizontal scaling support

### ✅ **Testing**
- Easy to mock and test
- Comprehensive demo system
- Clear separation of concerns

## 🚧 Future Enhancements

### Planned Features

- **Cloud Storage**: Full cloud storage integration
- **Distributed Caching**: Redis cluster support
- **Service Discovery**: Dynamic service registration
- **Circuit Breakers**: Fault tolerance patterns
- **Metrics Collection**: Prometheus integration

### Extension Points

- **Custom Storage Strategies**: User-defined storage backends
- **Custom Cache Strategies**: User-defined cache implementations
- **Plugin System**: Third-party service integrations
- **Configuration Management**: Dynamic configuration updates

## 📚 Additional Resources

- [Clean Architecture Principles](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Strategy Pattern](https://refactoring.guru/design-patterns/strategy)
- [Repository Pattern](https://martinfowler.com/eaaCatalog/repository.html)
- [Async Python](https://docs.python.org/3/library/asyncio.html)

## 🤝 Contributing

When contributing to the infrastructure layer:

1. **Follow Clean Architecture principles**
2. **Use strategy pattern for new backends**
3. **Implement comprehensive error handling**
4. **Add proper logging and monitoring**
5. **Include unit and integration tests**
6. **Update documentation**

## 📄 License

This infrastructure layer is part of the Unified Ads System and follows the same licensing terms as the parent project.

---

**Infrastructure Layer Status: ✅ COMPLETED**

The infrastructure layer provides a solid foundation for the ads feature, consolidating all scattered implementations into a unified, maintainable system that follows Clean Architecture principles.
