# PDF Variantes - Microservices & Serverless Architecture Guide

## 🏗️ Arquitectura Microservicios

La aplicación está diseñada siguiendo principios de microservicios y está lista para deployment serverless.

## 📦 Componentes Principales

### 1. Resilience Patterns

#### Circuit Breaker
```python
from ..core.circuit_breaker import get_circuit_breaker, CircuitBreakerConfig

# Uso básico
circuit = get_circuit_breaker("external-api", CircuitBreakerConfig(
    failure_threshold=5,
    timeout_seconds=60
))

result = await circuit.call(
    external_api_call,
    fallback=default_response
)
```

#### Retry Pattern
```python
from ..core.retry import retry, RetryConfig

# Retry con exponential backoff
result = await retry(
    api_call,
    config=RetryConfig(
        max_attempts=3,
        initial_delay=1.0,
        exponential_base=2.0
    )
)
```

### 2. Distributed Tracing (OpenTelemetry)

```python
from ..monitoring.tracing import get_tracing_service

tracing = get_tracing_service()

with tracing.span("process_pdf", attributes={"document_id": doc_id}):
    # Tu código aquí
    pass
```

**Configuración:**
```bash
# Habilitar tracing
export ENABLE_TRACING=true
export OTLP_ENDPOINT=http://localhost:4317
```

### 3. Metrics (Prometheus)

Las métricas se recopilan automáticamente en `/metrics`:

- `http_requests_total` - Total de requests HTTP
- `http_request_duration_seconds` - Duración de requests
- `pdf_uploads_total` - Total de uploads PDF
- `variants_generated_total` - Variantes generadas
- `cache_hits_total` / `cache_misses_total` - Estadísticas de cache

### 4. Structured Logging

```python
from ..utils.structured_logging import get_logger, set_request_context

logger = get_logger(__name__)

# Logging con contexto
logger.info("Processing document", extra={
    "document_id": doc_id,
    "user_id": user_id
})
```

### 5. Background Workers

#### Celery (Recomendado para producción)
```python
from ..workers.base_worker import create_celery_app

celery_app = create_celery_app(broker_url="redis://localhost:6379/0")

@celery_app.task
def process_pdf_async(document_id: str):
    # Procesamiento en background
    pass
```

#### RQ (Alternativa ligera)
```python
from ..workers.base_worker import create_rq_queue

queue = create_rq_queue()

job = queue.enqueue(process_pdf, document_id)
```

## 🚀 Deployment Serverless

### AWS Lambda

```python
# lambda_handler.py
from mangum import Mangum
from api.main import app

handler = Mangum(app)
```

**package.json para Lambda:**
```json
{
  "runtime": "python3.11",
  "handler": "lambda_handler.handler",
  "timeout": 900,
  "memory": 512
}
```

### Azure Functions

```python
# function_app.py
import azure.functions as func
from api.main import app

main = func.AsgiFunctionApp(app=app, http_auth_level=func.AuthLevel.ANONYMOUS)
```

### Google Cloud Run

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

## 🔐 API Gateway Integration

### Kong API Gateway

```yaml
services:
  - name: pdf-variantes-api
    url: http://api:8000
    plugins:
      - name: rate-limiting
        config:
          minute: 100
      - name: cors
        config:
          origins: ["*"]
```

### AWS API Gateway

- Rate limiting configurado
- CORS habilitado
- Request ID tracking
- Security headers

## 📊 Monitoring & Observability

### Prometheus + Grafana

1. **Prometheus scrape config:**
```yaml
scrape_configs:
  - job_name: 'pdf-variantes'
    static_configs:
      - targets: ['localhost:8000']
```

2. **Grafana Dashboards:**
- Request rate
- Error rate
- Latency (p50, p95, p99)
- Business metrics (PDFs, variants, topics)

### ELK Stack / CloudWatch

Los logs estructurados en JSON son compatibles con:
- Elasticsearch
- AWS CloudWatch Logs
- Google Cloud Logging
- Azure Monitor

## 🔄 Event-Driven Architecture

```python
from ..core.events import get_event_bus, EventType

event_bus = get_event_bus()

# Publicar evento
await event_bus.emit(
    EventType.PDF_UPLOADED,
    {"document_id": doc_id},
    source="pdf_service"
)

# Subscribirse a eventos
async def on_pdf_uploaded(event):
    # Procesar evento
    pass

event_bus.subscribe(EventType.PDF_UPLOADED, on_pdf_uploaded)
```

## 🎯 Best Practices Implementadas

### 1. Stateless Services
- Todos los servicios son stateless
- Estado en Redis/cache externo
- Session storage externalizado

### 2. Health Checks
```
GET /health - Health check básico
GET /api/v1/health/status - Health check completo con servicios
```

### 3. Graceful Shutdown
- Cleanup de conexiones
- Finalización de tareas en curso
- Cierre ordenado de recursos

### 4. Configuration Management
```python
# Variables de entorno soportadas:
ENVIRONMENT=production
ENABLE_TRACING=true
OTLP_ENDPOINT=http://...
PROMETHEUS_ENABLED=true
CORS_ORIGINS=["https://app.example.com"]
RATE_LIMIT_REQUESTS_PER_MINUTE=100
```

## 📈 Scaling Strategies

### Horizontal Scaling
- Stateless por diseño
- Load balancing ready
- Shared cache (Redis)
- Database connection pooling

### Vertical Scaling
- Async operations
- Efficient memory usage
- Background workers para tareas pesadas

### Auto-scaling (Serverless)
- Lambda: auto-scales basado en requests
- Cloud Run: auto-scales basado en CPU/concurrency
- Azure Functions: auto-scales basado en queue depth

## 🔒 Security Hardening

1. **CORS Configuration**: Configurable por ambiente
2. **Rate Limiting**: Per-IP con configuración flexible
3. **Request Validation**: Validadores centralizados
4. **Security Headers**: Automáticos en responses
5. **Structured Logging**: Sin información sensible

## 📝 Deployment Checklist

- [ ] Configure environment variables
- [ ] Setup Redis for caching/workers
- [ ] Configure external database
- [ ] Setup monitoring (Prometheus/Grafana)
- [ ] Configure tracing (OpenTelemetry)
- [ ] Setup log aggregation (ELK/CloudWatch)
- [ ] Configure API Gateway
- [ ] Setup health check endpoints
- [ ] Test circuit breakers
- [ ] Verify metrics collection
- [ ] Load test with expected traffic






