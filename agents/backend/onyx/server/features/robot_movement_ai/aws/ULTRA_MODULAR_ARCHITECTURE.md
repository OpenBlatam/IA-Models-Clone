# Ultra-Modular Architecture

## 🏗️ Architecture Overview

The system has been refactored into an **ultra-modular architecture** following:

- ✅ **Hexagonal Architecture** (Ports & Adapters)
- ✅ **Clean Architecture** (Layers: Presentation, Business, Data)
- ✅ **Dependency Injection** (Advanced DI container)
- ✅ **Factory Pattern** (For creating components)
- ✅ **Builder Pattern** (For composing endpoints)
- ✅ **Repository Pattern** (Data access abstraction)
- ✅ **Service Layer** (Business logic separation)

## 📊 Architecture Layers

```
┌─────────────────────────────────────┐
│   Presentation Layer                │
│   - API Routers                     │
│   - Endpoint Builders                │
│   - Response Builders                │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   Business Layer                     │
│   - Use Cases                        │
│   - Domain Services                  │
│   - Service Factory                  │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   Data Layer                         │
│   - Repository Adapters               │
│   - Cache Adapters                   │
│   - Messaging Adapters               │
└─────────────────────────────────────┘
```

## 🔌 Ports & Adapters (Hexagonal Architecture)

### Ports (Interfaces)
- `RepositoryPort`: Data persistence interface
- `CachePort`: Caching interface
- `MessagingPort`: Messaging interface
- `ServicePort`: Business service interface

### Adapters (Implementations)
- **Repository**: DynamoDB, PostgreSQL, InMemory
- **Cache**: Redis, Memcached, InMemory
- **Messaging**: Kafka, RabbitMQ, SQS

## 📁 Module Structure

```
aws/modules/
├── ports/                    # Interfaces (Ports)
│   ├── repository_port.py
│   ├── cache_port.py
│   ├── messaging_port.py
│   └── service_port.py
├── adapters/                 # Implementations (Adapters)
│   ├── repository_adapters.py
│   ├── cache_adapters.py
│   └── messaging_adapters.py
├── presentation/             # Presentation Layer
│   ├── api_router.py
│   ├── endpoint_builder.py
│   └── response_builder.py
├── business/                 # Business Layer
│   ├── use_cases.py
│   ├── domain_services.py
│   └── service_factory.py
├── data/                     # Data Layer
│   ├── repository_factory.py
│   ├── cache_factory.py
│   └── messaging_factory.py
├── composition/              # Service Composition
│   ├── service_composer.py
│   └── movement_service_composition.py
└── dependency_injection/      # DI Container
    └── container.py
```

## 🚀 Usage Examples

### Creating a Service with DI

```python
from aws.modules.composition.service_composer import ServiceComposer
from aws.modules.dependency_injection.container import get_container

# Create DI container
container = get_container()

# Register dependencies
from aws.modules.adapters import RedisCacheAdapter, DynamoDBRepositoryAdapter

container.register_singleton(
    CachePort,
    RedisCacheAdapter(redis_url="redis://localhost:6379")
)

container.register_singleton(
    RepositoryPort,
    DynamoDBRepositoryAdapter(table_name="movement-table")
)

# Compose service
composer = ServiceComposer.from_env("movement-service")
app = composer.compose_service([router])
```

### Using Ports & Adapters

```python
from aws.modules.ports.cache_port import CachePort
from aws.modules.adapters.cache_adapters import RedisCacheAdapter

# Use interface (port)
cache: CachePort = RedisCacheAdapter(redis_url="redis://...")

# Switch implementation easily
cache: CachePort = MemcachedCacheAdapter(servers="localhost:11211")
```

### Building Endpoints

```python
from aws.modules.presentation.endpoint_builder import EndpointBuilder
from aws.modules.presentation.api_router import APIRouter

router = APIRouter(prefix="/api/v1")
builder = EndpointBuilder(router)

builder \
    .path("/move/to") \
    .method("POST") \
    .handler(move_handler) \
    .response_model(MoveResponse) \
    .summary("Move robot") \
    .build()
```

### Using Domain Services

```python
from aws.modules.business.service_factory import ServiceFactory
from aws.modules.adapters import RedisCacheAdapter, DynamoDBRepositoryAdapter

# Create service factory
factory = ServiceFactory(
    repository=DynamoDBRepositoryAdapter("movement-table"),
    cache=RedisCacheAdapter("redis://...")
)

# Get domain service
movement_service = factory.create_movement_service()

# Execute operation
result = await movement_service.execute("move_to", x=0.5, y=0.3, z=0.2)
```

## 🔧 Dependency Injection

### Registering Dependencies

```python
from aws.modules.dependency_injection.container import get_container

container = get_container()

# Singleton
container.register_singleton(CachePort, RedisCacheAdapter(...))

# Factory
container.register_factory(RepositoryPort, lambda: DynamoDBRepositoryAdapter(...))

# Transient (new instance each time)
container.register_transient(ServicePort, MovementService)
```

### Using Dependencies

```python
# Get dependency
cache = container.get(CachePort)

# With scope
with container.create_scope("request") as scope:
    service = scope.get(MovementService)
```

## 📦 Module Independence

Each module can be used **completely independently**:

### Using Only Cache Module

```python
from aws.modules.adapters.cache_adapters import RedisCacheAdapter

cache = RedisCacheAdapter("redis://localhost:6379")
await cache.set("key", "value")
value = await cache.get("key")
```

### Using Only Repository Module

```python
from aws.modules.adapters.repository_adapters import DynamoDBRepositoryAdapter

repo = DynamoDBRepositoryAdapter("table-name")
entity = await repo.create(my_entity)
```

### Using Only Messaging Module

```python
from aws.modules.adapters.messaging_adapters import KafkaMessagingAdapter

messaging = KafkaMessagingAdapter("localhost:9092")
await messaging.publish("topic", {"data": "value"})
```

## 🎯 Benefits

### 1. **Complete Independence**
- Each module can be used standalone
- No coupling between modules
- Easy to test in isolation

### 2. **Easy Swapping**
- Switch implementations via ports
- Change Redis to Memcached easily
- Change DynamoDB to PostgreSQL easily

### 3. **Testability**
- Mock any port easily
- Test each layer independently
- Use in-memory adapters for testing

### 4. **Composability**
- Compose services from modules
- Mix and match components
- Build custom services easily

### 5. **Maintainability**
- Clear separation of concerns
- Single responsibility principle
- Easy to understand and modify

## 📚 Examples

### Complete Service Composition

```python
from aws.modules.composition.service_composer import ServiceComposer
from aws.modules.composition.movement_service_composition import compose_movement_service

# Compose from environment
app = compose_movement_service()

# Or with custom config
config = {
    "repository": {"type": "dynamodb", "table_name": "movement-table"},
    "cache": {"type": "redis", "redis_url": "redis://..."},
    "messaging": {"type": "kafka", "kafka_servers": "localhost:9092"}
}
app = compose_movement_service(config)
```

### Custom Service

```python
from aws.modules.composition.service_composer import ServiceComposer
from aws.modules.presentation.api_router import APIRouter

# Create composer
composer = ServiceComposer.from_env("my-service")

# Create business layer
service_factory = composer.create_business_layer()
my_service = service_factory.create_movement_service()

# Create presentation layer
router = composer.create_presentation_layer("/api/v1")

# Add endpoints
@router.router.post("/custom")
async def custom_endpoint():
    result = await my_service.execute("move_to", x=1, y=2, z=3)
    return result

# Compose
app = composer.compose_service([router])
```

## ✅ Architecture Principles

1. ✅ **Hexagonal Architecture**: Ports & Adapters
2. ✅ **Clean Architecture**: Layer separation
3. ✅ **Dependency Inversion**: Depend on abstractions
4. ✅ **Single Responsibility**: One responsibility per module
5. ✅ **Open/Closed**: Open for extension, closed for modification
6. ✅ **Interface Segregation**: Small, focused interfaces
7. ✅ **Dependency Injection**: Loose coupling
8. ✅ **Factory Pattern**: Centralized creation
9. ✅ **Builder Pattern**: Fluent API construction
10. ✅ **Repository Pattern**: Data access abstraction

## 🎉 Result

An **ultra-modular architecture** where:

- ✅ Every module is independent
- ✅ Easy to swap implementations
- ✅ Highly testable
- ✅ Composable services
- ✅ Clear separation of concerns
- ✅ Follows SOLID principles
- ✅ Production-ready

---

**The system is now ultra-modular and follows all architectural best practices!** 🚀















