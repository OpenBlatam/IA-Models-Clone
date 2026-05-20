# 🚀 Características Avanzadas - Cursor Agent 24/7

Este documento describe las características avanzadas implementadas siguiendo las mejores prácticas de microservicios, serverless y cloud-native.

## 📊 Observabilidad

### OpenTelemetry

Distributed tracing completo con OpenTelemetry:

```python
from core.observability import setup_opentelemetry, trace_span

# Configurar
setup_opentelemetry(
    service_name="cursor-agent-24-7",
    service_version="1.0.0",
    endpoint="http://otel-collector:4317"
)

# Usar en código
async with trace_span("process_task", {"task_id": task_id}):
    # Tu código aquí
    pass
```

**Configuración:**
```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
export OTEL_LOG_EXPORTER=true  # Para desarrollo
```

### Prometheus

Métricas Prometheus automáticas:

- **HTTP Metrics**: Requests, duration, size
- **Agent Metrics**: Tasks, queue size, active tasks
- **System Metrics**: Memory, CPU

**Endpoint:** `GET /metrics`

**Ejemplo de métricas:**
```
http_requests_total{method="GET",endpoint="/api/health",status="200"} 1234
http_request_duration_seconds{method="GET",endpoint="/api/health"} 0.05
agent_tasks_total{status="completed"} 567
agent_active_tasks 5
```

**Integración con Grafana:**
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'cursor-agent'
    static_configs:
      - targets: ['localhost:8024']
```

## 🔐 Seguridad

### OAuth2 / JWT

Autenticación completa con OAuth2:

```bash
# Obtener token
curl -X POST http://localhost:8024/api/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin"

# Usar token
curl http://localhost:8024/api/auth/me \
  -H "Authorization: Bearer <token>"
```

**Protección de endpoints:**
```python
from core.oauth2 import get_current_active_user, require_role

@app.get("/api/protected")
async def protected_endpoint(user = Depends(get_current_active_user)):
    return {"message": f"Hello {user.username}"}

@app.post("/api/admin")
async def admin_endpoint(user = Depends(require_role("admin"))):
    return {"message": "Admin only"}
```

**Configuración:**
```bash
export JWT_SECRET_KEY=your-secret-key-change-in-production
export ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### Rate Limiting con Redis

Rate limiting distribuido usando Redis:

```python
from core.rate_limiter_redis import rate_limit_check

@app.get("/api/tasks")
async def get_tasks(
    request: Request,
    _ = Depends(rate_limit_check(max_requests=100, window_seconds=60))
):
    # Tu código aquí
    pass
```

**Configuración:**
```bash
export REDIS_URL=redis://localhost:6379
# O
export REDIS_ENDPOINT=localhost:6379
```

**Headers de respuesta:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1234567890
```

## 📨 Message Brokers

### RabbitMQ

```python
from core.message_broker import create_broker, SystemEvents

broker = create_broker("rabbitmq")
await broker.connect()

# Publicar evento
await broker.publish(SystemEvents.TASK_CREATED, {
    "task_id": "123",
    "command": "optimize code"
})

# Suscribirse
async def handle_task_created(message):
    print(f"Task created: {message}")

await broker.subscribe(SystemEvents.TASK_CREATED, handle_task_created)
```

### Kafka

```python
broker = create_broker("kafka")
await broker.connect()

# Publicar
await broker.publish("tasks", {"task_id": "123"})

# Consumir
await broker.consume("tasks", handle_message)
```

**Configuración:**
```bash
# RabbitMQ
export RABBITMQ_URL=amqp://user:pass@localhost:5672/
export MESSAGE_BROKER_TYPE=rabbitmq

# Kafka
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
export KAFKA_GROUP_ID=cursor-agent-group
export MESSAGE_BROKER_TYPE=kafka
```

## 🔄 Circuit Breakers

Protección contra fallos en cascada:

```python
from core.circuit_breaker import get_circuit_breaker, CircuitBreakerConfig

# Crear circuit breaker
breaker = get_circuit_breaker(
    "external-api",
    config=CircuitBreakerConfig(
        failure_threshold=5,
        timeout=60.0
    )
)

# Usar
try:
    result = await breaker.call(external_api_call, arg1, arg2)
except CircuitBreakerOpenError:
    # Circuit está abierto, usar fallback
    result = fallback_value
```

## 🔁 Retries con Backoff

Reintentos automáticos con backoff exponencial:

```python
from core.retry_handler import retry_with_backoff, retry_dynamodb, retry_redis

# Decorador simple
@retry_with_backoff(max_attempts=3, min_wait=1.0, max_wait=10.0)
async def unreliable_function():
    # Tu código aquí
    pass

# Específico para DynamoDB
@retry_dynamodb
async def dynamodb_operation():
    # Operación DynamoDB
    pass

# Específico para Redis
@retry_redis
async def redis_operation():
    # Operación Redis
    pass
```

## 📈 Performance

### Caching

Caché multi-nivel:

```python
from core.aws_adapter import AWSCacheAdapter

# ElastiCache Redis
cache = AWSCacheAdapter(
    cache_type="elasticache",
    endpoint="your-redis-endpoint"
)

# DynamoDB
cache = AWSCacheAdapter(
    cache_type="dynamodb",
    table_name="cursor-agent-cache"
)

# Usar
value = await cache.get("key")
await cache.set("key", value, ttl=3600)
```

### Connection Pooling

Conexiones reutilizables para mejor performance:

```python
# Redis connection pooling (automático en aioredis)
# HTTP connection pooling (automático en httpx)
```

## 🏗️ Arquitectura

### Stateless Design

- Estado en DynamoDB/Redis
- Sin estado local
- Escalable horizontalmente

### Microservices Ready

- Separación de concerns
- Message brokers para comunicación
- API Gateway compatible
- Health checks

### Serverless Optimized

- Cold start optimization
- Lazy loading
- Minimal dependencies
- Lambda compatible

## 📊 Monitoreo

### Métricas Disponibles

1. **HTTP Metrics**
   - `http_requests_total`
   - `http_request_duration_seconds`
   - `http_request_size_bytes`

2. **Agent Metrics**
   - `agent_tasks_total`
   - `agent_tasks_duration_seconds`
   - `agent_active_tasks`
   - `agent_queue_size`

3. **System Metrics**
   - `system_memory_bytes`
   - `system_cpu_percent`

### Logging Estructurado

```python
import structlog

logger = structlog.get_logger()
logger.info("Task completed", task_id="123", duration=1.5)
```

### Distributed Tracing

Traces completos con OpenTelemetry:
- Request tracing
- Service dependencies
- Performance bottlenecks
- Error propagation

## 🔧 Configuración Completa

```bash
# Observability
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
export OTEL_LOG_EXPORTER=true

# Prometheus (automático en /metrics)

# Security
export JWT_SECRET_KEY=your-secret-key
export ACCESS_TOKEN_EXPIRE_MINUTES=30

# Rate Limiting
export REDIS_URL=redis://localhost:6379

# Message Broker
export MESSAGE_BROKER_TYPE=rabbitmq
export RABBITMQ_URL=amqp://user:pass@localhost:5672/

# O Kafka
export MESSAGE_BROKER_TYPE=kafka
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

## 📚 Recursos

- [OpenTelemetry Docs](https://opentelemetry.io/docs/)
- [Prometheus Docs](https://prometheus.io/docs/)
- [OAuth2 Spec](https://oauth.net/2/)
- [RabbitMQ Docs](https://www.rabbitmq.com/documentation.html)
- [Kafka Docs](https://kafka.apache.org/documentation/)

## 🎯 Próximos Pasos

1. Configurar Grafana dashboards
2. Configurar alertas en Prometheus
3. Integrar con servicio de tracing (Jaeger, Zipkin)
4. Configurar message broker en producción
5. Implementar rate limiting por usuario/API key




