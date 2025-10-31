# Advanced Microservices & Event-Driven Improvements

## Mejoras Implementadas

### 1. Event-Driven Architecture

#### Message Brokers (`infrastructure/messaging/`)
- **Redis Streams** - Para alta performance y baja latencia
- **RabbitMQ** - Para message queuing robusto
- **Kafka** - Para high-throughput event streaming
- **In-Memory** - Para testing y desarrollo

**Uso:**
```python
from infrastructure.messaging import RedisStreamBroker, EventPublisher

broker = RedisStreamBroker("redis://localhost:6379/0")
publisher = EventPublisher(broker)

event = AnalysisCompletedEvent(
    aggregate_id="content_hash_123",
    content_hash="abc123",
    redundancy_score=0.85,
    analysis_result={...}
)

await publisher.publish(event)
```

#### Domain Events (`infrastructure/messaging/events.py`)
- `AnalysisCompletedEvent` - Emitido cuando análisis completa
- `SimilarityCompletedEvent` - Emitido cuando similitud completa
- `BatchProcessingCompletedEvent` - Emitido cuando batch completa
- `WebhookDeliveredEvent` - Emitido cuando webhook se entrega

### 2. Stateless Session Management

#### Redis-Based Sessions (`infrastructure/state/stateless_session.py`)
- **No server-side state** - Todo en Redis
- **Auto-expiration** - TTL automático
- **Connection pooling** - Alta performance
- **Scalable** - Multi-instance compatible

**Características:**
```python
from infrastructure.state import get_session_manager

session_mgr = get_session_manager()

# Crear sesión
session = await session_mgr.create_session(
    user_id="user123",
    ip_address="192.168.1.1",
    ttl=3600
)

# Obtener sesión (desde cualquier instancia)
session = await session_mgr.get_session(session.session_id)

# Actualizar sesión
await session_mgr.update_session(session.session_id, {"key": "value"})
```

### 3. High-Performance Caching

#### Redis Cache (`infrastructure/cache/redis_cache.py`)
- **Connection pooling** - Máx conexiones configurables
- **Namespacing** - Separación por prefijos
- **Pattern deletion** - Limpieza masiva
- **Statistics** - Métricas de hit rate

**Uso:**
```python
from infrastructure.cache import RedisCache

cache = RedisCache(
    redis_url="redis://localhost:6379/0",
    default_ttl=3600,
    max_connections=50
)

# Get or set pattern
value = await cache.get_or_set(
    "analysis:abc123",
    lambda: perform_analysis(),
    ttl=7200
)

# Pattern deletion
await cache.delete_pattern("analysis:*", prefix="cache")
```

### 4. Advanced Observability Middleware

#### Comprehensive Monitoring (`api/middleware/observability.py`)
- **Distributed Tracing** - OpenTelemetry spans
- **Structured Logging** - JSON logs con correlation ID
- **Prometheus Metrics** - Request metrics automáticos
- **Request Correlation** - Tracking end-to-end

**Features:**
- Correlation ID propagation
- Automatic span creation
- Error tracking
- Performance metrics

## Integración con Existing Code

### Actualizar Application Services

```python
# application/services/analysis_service.py
from infrastructure.messaging import EventPublisher
from infrastructure.messaging.events import AnalysisCompletedEvent

class AnalysisService:
    def __init__(self, ..., event_publisher: EventPublisher):
        self.event_publisher = event_publisher
    
    async def analyze_content(self, request):
        result = await analyze_content_domain(request.content)
        
        # Emit event
        event = AnalysisCompletedEvent(
            aggregate_id=result.content_hash,
            content_hash=result.content_hash,
            redundancy_score=result.redundancy_score,
            analysis_result=result.to_dict()
        )
        await self.event_publisher.publish(event)
        
        return result
```

### Actualizar App.py

```python
# app.py
from infrastructure.messaging import RedisStreamBroker, EventPublisher
from infrastructure.state import get_session_manager
from infrastructure.cache import RedisCache
from api.middleware.observability import ObservabilityMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize message broker
    broker = RedisStreamBroker()
    await broker.connect()
    event_publisher = EventPublisher(broker)
    app.state.event_publisher = event_publisher
    
    # Initialize cache
    cache = RedisCache()
    app.state.cache = cache
    
    yield
    
    # Cleanup
    await broker.disconnect()

def create_app():
    app = FastAPI(...)
    
    # Add observability middleware
    app.add_middleware(ObservabilityMiddleware, service_name="content-detector")
    
    return app
```

## Architecture Benefits

### 1. Stateless Services
- ✅ No session state en servidor
- ✅ Redis para state persistence
- ✅ Horizontal scaling sin issues

### 2. Event-Driven
- ✅ Loose coupling entre servicios
- ✅ Async processing
- ✅ Event sourcing ready

### 3. High Performance
- ✅ Connection pooling
- ✅ Efficient caching
- ✅ Async operations

### 4. Observability
- ✅ Distributed tracing
- ✅ Structured logging
- ✅ Metrics collection

## Deployment Examples

### Docker Compose con Message Brokers

```yaml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  rabbitmq:
    image: rabbitmq:3-management
    ports:
      - "5672:5672"
      - "15672:15672"
  
  kafka:
    image: confluentinc/cp-kafka:latest
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
```

### Environment Variables

```bash
# Message Broker
MESSAGE_BROKER_TYPE=redis_streams
REDIS_URL=redis://localhost:6379/0
RABBITMQ_URL=amqp://guest:guest@localhost:5672/
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Cache
REDIS_CACHE_URL=redis://localhost:6379/1
CACHE_DEFAULT_TTL=3600
CACHE_MAX_CONNECTIONS=50

# Observability
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
SERVICE_NAME=content-redundancy-detector
```

## Performance Optimizations

### 1. Connection Pooling
- Redis: Max 50 connections
- Reuse connections
- Auto-reconnect on failure

### 2. Caching Strategy
- TTL-based expiration
- Pattern-based invalidation
- Namespaced keys

### 3. Async Operations
- All I/O operations async
- Non-blocking message publishing
- Concurrent request handling

## Security Considerations

### 1. Session Security
- UUID-based session IDs
- TTL enforcement
- IP tracking
- Auto-expiration

### 2. Message Security
- Encryption in transit (TLS)
- Authentication for brokers
- Message signing (optional)

### 3. Cache Security
- Namespace isolation
- TTL limits
- Access control

## Monitoring & Alerts

### Prometheus Metrics
```
http_requests_total{method, endpoint, status_code}
http_request_duration_seconds{method, endpoint}
cache_hits_total{cache_type}
cache_misses_total{cache_type}
```

### Grafana Dashboards
- Request rate by endpoint
- Error rate trends
- Cache hit rate
- Message broker throughput

## Testing

### Unit Tests
```python
from infrastructure.messaging import InMemoryBroker

def test_event_publishing():
    broker = InMemoryBroker()
    publisher = EventPublisher(broker)
    # Test event publishing
```

### Integration Tests
```python
async def test_redis_cache():
    cache = RedisCache()
    await cache.set("test", "value")
    value = await cache.get("test")
    assert value == "value"
```

## Next Steps

1. **Event Handlers** - Implement handlers para eventos
2. **CQRS** - Command Query Responsibility Segregation
3. **Event Sourcing** - Store all domain events
4. **Saga Pattern** - Distributed transactions
5. **API Gateway Integration** - Rate limiting, auth

## References

- [Event-Driven Architecture](https://martinfowler.com/articles/201701-event-driven.html)
- [Redis Streams](https://redis.io/docs/data-types/streams/)
- [RabbitMQ Patterns](https://www.rabbitmq.com/getstarted.html)
- [Apache Kafka](https://kafka.apache.org/documentation/)
- [OpenTelemetry](https://opentelemetry.io/docs/)






