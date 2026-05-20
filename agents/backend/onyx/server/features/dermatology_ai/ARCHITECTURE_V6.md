# Arquitectura V6.0 - Modular, Rápida y Optimizada

## Resumen

Refactorización completa del módulo `dermatology_ai` siguiendo principios de microservicios, FastAPI avanzado y optimización para entornos serverless.

## Principios de Diseño

### 1. Stateless Services
- Servicios sin estado interno
- Estado persistido en Redis/cache externo
- Permite escalado horizontal sin problemas

### 2. Microservicios
- Separación clara de responsabilidades
- Comunicación mediante APIs bien definidas
- Circuit breakers y retries para resiliencia

### 3. Serverless-Optimized
- Cold start minimizado
- Lazy loading de servicios pesados
- Warmup opcional de servicios críticos

### 4. Observabilidad
- OpenTelemetry para distributed tracing
- Prometheus para métricas
- Structured logging (JSON)

## Componentes Principales

### 1. Main Application (`main.py`)

**Características:**
- Lifespan events para startup/shutdown
- Lazy loading de routers
- Warmup opcional (deshabilitable para cold start)
- Global exception handling

**Optimizaciones:**
- Routers cargados asíncronamente
- Servicios pesados (ML models) cargados bajo demanda
- Cache manager inicializado al inicio

### 2. Cache Manager (`utils/cache.py`)

**Características:**
- Redis como backend principal
- Fallback a in-memory cache
- TTL support
- Auto-cleanup de entradas expiradas

**Uso:**
```python
from utils.cache import get_cache_manager

cache = get_cache_manager()
await cache.initialize()
value = await cache.get("key")
await cache.set("key", value, ttl=3600)
```

### 3. Circuit Breaker (`utils/circuit_breaker.py`)

**Características:**
- Estados: CLOSED, OPEN, HALF_OPEN
- Thresholds configurables
- Auto-recovery
- Prevención de cascading failures

**Uso:**
```python
from utils.circuit_breaker import get_circuit_breaker, CircuitBreakerConfig

cb = get_circuit_breaker("external-service", CircuitBreakerConfig(
    failure_threshold=5,
    timeout=60.0
))

result = await cb.call_async(external_service.call, arg1, arg2)
```

### 4. Retry Utilities (`utils/retry.py`)

**Características:**
- Exponential backoff
- Jitter para evitar thundering herd
- Configuración flexible
- Soporte sync y async

**Uso:**
```python
from utils.retry import retry, RetryConfig

@retry(RetryConfig(max_attempts=3, initial_delay=1.0))
async def unreliable_service():
    return await call_service()
```

### 5. Async Workers (`utils/async_workers.py`)

**Características:**
- Worker pool ligero
- Priority queue
- Task tracking
- Alternativa ligera a Celery

**Uso:**
```python
from utils.async_workers import get_worker_pool

pool = get_worker_pool(max_workers=4)
await pool.start()

task_id = await pool.submit(process_image, image_data, priority=1)
result = await pool.get_task_result(task_id)
```

### 6. Middleware Stack

#### Tracing Middleware (`middleware/tracing_middleware.py`)
- OpenTelemetry integration
- Request ID tracking
- Distributed tracing
- Fallback a logging si OpenTelemetry no disponible

#### Monitoring Middleware (`middleware/monitoring_middleware.py`)
- Prometheus metrics
- Request duration, size, status codes
- Active requests gauge
- Endpoint `/metrics` para scraping

#### Security Middleware (`middleware/security_middleware.py`)
- Security headers (CSP, HSTS, etc.)
- IP filtering
- HTTPS enforcement

#### Rate Limit Middleware (`middleware/rate_limit_middleware.py`)
- Rate limiting por IP/usuario
- Headers informativos
- Configuración flexible

### 7. Structured Logging (`utils/logger.py`)

**Características:**
- JSON formatting opcional
- Structured logs para análisis
- Integración con sistemas centralizados (ELK, CloudWatch)

**Uso:**
```python
from utils.logger import setup_logging, get_logger

setup_logging(log_level="INFO", use_json=True)
logger = get_logger(__name__)
logger.info("Message", extra={"user_id": "123", "action": "analyze"})
```

## Optimizaciones de Rendimiento

### 1. Cold Start Reduction
- Lazy loading de servicios pesados
- Warmup opcional (deshabilitable)
- Minimal imports en startup
- Async initialization

### 2. Caching Strategy
- Redis para estado compartido
- In-memory fallback
- TTL inteligente
- Cache warming opcional

### 3. Async/Await
- Todas las operaciones I/O son async
- No bloqueo del event loop
- Worker pool para tareas CPU-bound

### 4. Connection Pooling
- Reutilización de conexiones
- Health checks automáticos
- Auto-scaling

## Configuración

### Variables de Entorno

```bash
# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Redis
REDIS_URL=redis://localhost:6379/0
USE_REDIS=true

# Observability
OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_EXPORTER_OTLP_INSECURE=true

# Security
JWT_SECRET=your-secret-key
REQUIRE_AUTH=false
ALLOWED_ORIGINS=https://example.com

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_MAX=100
RATE_LIMIT_WINDOW=60
```

## Deployment

### Serverless (AWS Lambda, Azure Functions)

```python
# handler.py
from main import app

def handler(event, context):
    return app(event, context)
```

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8006"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dermatology-ai
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: dermatology-ai:6.0.0
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: "http://jaeger:4317"
        livenessProbe:
          httpGet:
            path: /health
            port: 8006
        readinessProbe:
          httpGet:
            path: /health
            port: 8006
```

## Monitoreo

### Métricas Prometheus

- `http_requests_total` - Total de requests
- `http_request_duration_seconds` - Duración de requests
- `http_request_size_bytes` - Tamaño de requests
- `http_response_size_bytes` - Tamaño de responses
- `active_requests` - Requests activos

### Tracing

- Distributed tracing con OpenTelemetry
- Request ID en todos los logs
- Span context propagation

### Logging

- Structured JSON logs
- Niveles configurables
- Integración con sistemas centralizados

## Health Checks

### `/health`
- Health check básico para load balancers
- Respuesta rápida (<10ms)
- Sin dependencias externas

### `/dermatology/health/detailed`
- Health check detallado
- Verifica dependencias (Redis, DB, etc.)
- Métricas del sistema

## Mejores Prácticas

### 1. Uso de Cache
```python
# ✅ Cachear operaciones costosas
result = await cache.get("key")
if not result:
    result = expensive_operation()
    await cache.set("key", result, ttl=3600)
```

### 2. Circuit Breakers
```python
# ✅ Proteger llamadas externas
cb = get_circuit_breaker("external-api")
try:
    result = await cb.call_async(external_api.call)
except CircuitBreakerOpenError:
    # Fallback o error graceful
    return cached_result
```

### 3. Retries
```python
# ✅ Retry con exponential backoff
@retry(RetryConfig(max_attempts=3))
async def unreliable_call():
    return await external_service()
```

### 4. Async Workers
```python
# ✅ Tareas background
task_id = await pool.submit(process_image, image_data)
# No bloquear request
return {"task_id": task_id, "status": "processing"}
```

## Migración desde V5.x

1. **Actualizar imports:**
   ```python
   # Antes
   from utils.rate_limiter import RateLimiter
   
   # Después (igual, pero con nuevas features)
   from utils.cache import get_cache_manager
   from utils.circuit_breaker import get_circuit_breaker
   ```

2. **Usar cache manager:**
   ```python
   cache = get_cache_manager()
   await cache.initialize()
   ```

3. **Agregar circuit breakers:**
   ```python
   cb = get_circuit_breaker("service-name")
   result = await cb.call_async(service.call)
   ```

## Próximos Pasos

1. ✅ Arquitectura modular
2. ✅ Cache distribuido
3. ✅ Circuit breakers
4. ✅ Retries
5. ✅ Observabilidad
6. ⏳ Message broker integration (RabbitMQ/Kafka)
7. ⏳ API Gateway integration
8. ⏳ Advanced load balancing

## Conclusión

La arquitectura V6.0 proporciona:

- ✅ **Modularidad**: Separación clara de responsabilidades
- ✅ **Rapidez**: Optimizado para cold start y throughput
- ✅ **Resiliencia**: Circuit breakers y retries
- ✅ **Observabilidad**: Tracing, métricas, logging
- ✅ **Escalabilidad**: Stateless, horizontal scaling
- ✅ **Serverless-ready**: Optimizado para entornos serverless

El sistema está listo para producción con arquitectura enterprise-grade.















