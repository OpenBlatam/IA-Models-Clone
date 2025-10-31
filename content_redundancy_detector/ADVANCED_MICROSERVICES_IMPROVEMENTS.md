# Advanced Microservices & Serverless Improvements

Este documento describe las mejoras avanzadas implementadas siguiendo los principios de microservicios, arquitectura serverless y mejores prácticas de FastAPI.

## Mejoras Implementadas

### 1. Circuit Breaker Pattern (`circuit_breaker.py`)
- **Propósito**: Prevenir fallos en cascada detectando servicios fallidos
- **Estado**: CLOSED, OPEN, HALF_OPEN
- **Configuración**: Thresholds de fallos, timeouts, retry logic
- **Uso**: Proteger llamadas a servicios externos (AI/ML, base de datos, etc.)

```python
from circuit_breaker import get_circuit_breaker, CircuitBreakerConfig

breaker = get_circuit_breaker("ai_service", CircuitBreakerConfig(
    failure_threshold=5,
    timeout=60
))

result = await breaker.call(external_service_call)
```

### 2. Distributed Tracing con OpenTelemetry (`observability.py`)
- **Propósito**: Tracing distribuido para microservicios
- **Integración**: OpenTelemetry, Jaeger, Tempo
- **Features**:
  - Spans automáticos para requests
  - Context propagation entre servicios
  - Métricas integradas con Prometheus

**Configuración**:
```python
from observability import setup_observability

setup_observability("service-name", "1.0.0")
```

### 3. Structured Logging (`observability.py`)
- **Formato**: JSON estructurado para mejor análisis
- **Integración**: ELK Stack, CloudWatch, centralizados
- **Features**:
  - Request ID tracking
  - Correlación entre logs
  - Log levels configurables

```python
from observability import get_structured_logger

logger = get_structured_logger(__name__)
logger.info("Request processed", user_id="123", endpoint="/analyze")
```

### 4. Prometheus Metrics (`observability.py`, `prometheus_metrics.py`)
- **Métricas HTTP**: Requests totales, duración, códigos de estado
- **Métricas de negocio**: Análisis, cache hits/misses
- **Métricas del sistema**: Conexiones activas, circuit breakers
- **Endpoint**: `/metrics` para scraping de Prometheus

### 5. Async Workers con Celery (`async_workers.py`)
- **Propósito**: Procesamiento asíncrono de tareas pesadas
- **Configuración**:
  - Queues separados por tipo de tarea
  - Rate limiting por worker
  - Result expiration
  - Periodic tasks con Celery Beat

**Ejemplo**:
```python
from async_workers import create_task

# Dispatch task asynchronously
task = create_task("content_redundancy_detector.tasks.analyze_content", content, threshold)
result = task.get(timeout=30)  # Get result
```

### 6. OAuth2 Security (`security_oauth2.py`)
- **Autenticación**: OAuth2 Password Flow
- **Tokens**: JWT con refresh tokens
- **Scopes**: read, write, admin
- **API Keys**: Soporte para machine-to-machine
- **Features**:
  - Password hashing con bcrypt
  - Token validation
  - Scope-based access control

**Uso**:
```python
from security_oauth2 import get_current_user, require_scopes

@router.get("/protected")
async def protected_route(user = Depends(get_current_user)):
    return {"user": user}

@router.get("/admin")
async def admin_route(user = Depends(require_scopes("admin"))):
    return {"admin": True}
```

### 7. Serverless Optimizations (`serverless_optimizer.py`)
- **Propósito**: Reducir cold start times
- **Optimizaciones**:
  - Lazy imports
  - Connection pooling reutilizable
  - Fast JSON (orjson)
  - Warm-up function
  - Garbage collection tuning

**Soporte**:
- AWS Lambda (con Mangum)
- Azure Functions
- Google Cloud Functions
- Cloud Run / Knative

```python
from serverless_optimizer import is_serverless, warm_up, optimize_for_serverless

if is_serverless():
    optimize_for_serverless()
    warm_up()
```

### 8. API Gateway Integration (`api_gateway.py`)
- **Soporte**: Kong, AWS API Gateway, Traefik, NGINX
- **Features**:
  - Request ID tracking
  - Client IP extraction
  - Rate limiting config
  - CORS headers automáticos

**Detección automática**:
```python
from api_gateway import api_gateway

gateway_info = api_gateway.process_request(request)
# Returns: type, request_id, client_ip, protocol
```

## Configuración

### Variables de Entorno

```bash
# Observability
OTLP_ENDPOINT=localhost:4317
SERVICE_NAME=content-redundancy-detector
SERVICE_VERSION=2.0.0

# Celery
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# OpenTelemetry
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Serverless
AWS_LAMBDA_FUNCTION_NAME=  # Detección automática de Lambda
FUNCTIONS_WORKER_RUNTIME=  # Detección automática de Azure
```

### Docker Compose para Desarrollo

```yaml
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
  
  jaeger:
    image: jaegertracing/all-in-one
    ports:
      - "16686:16686"
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
  
  celery-worker:
    build: .
    command: celery -A async_workers.celery_app worker --loglevel=info
```

## Arquitectura

### Flujo de Request

1. **API Gateway** → Enruta request, agrega headers
2. **Middleware Chain**:
   - LoggingMiddleware → Structured logs
   - PerformanceMiddleware → Métricas de tiempo
   - SecurityMiddleware → Headers de seguridad
   - RateLimitMiddleware → Throttling
   - ErrorHandlingMiddleware → Manejo centralizado
3. **Circuit Breaker** → Protección de servicios externos
4. **Tracing** → Span creation y propagation
5. **Route Handler** → Business logic
6. **Response** → Métricas y headers

### Async Task Flow

```
Client Request → FastAPI → Celery Task Queue
                                    ↓
                           Celery Worker (Background)
                                    ↓
                           Redis Result Backend
                                    ↓
                           Webhook Notification (opcional)
```

## Monitoreo

### Métricas Prometheus

- `http_requests_total`: Total de requests HTTP
- `http_request_duration_seconds`: Duración de requests
- `analysis_requests_total`: Requests de análisis
- `cache_hits_total` / `cache_misses_total`: Cache stats
- `circuit_breaker_state`: Estado de circuit breakers

### Dashboards Grafana

Importar dashboards pre-configurados:
- HTTP Request Metrics
- Business Metrics
- System Health
- Circuit Breaker Status

## Testing

### Circuit Breaker Testing

```python
# Simular fallos para probar circuit breaker
breaker = get_circuit_breaker("test_service")

# Forzar fallos
for _ in range(6):
    try:
        await breaker.call(failing_function)
    except:
        pass

# Verificar que está OPEN
assert breaker.state == CircuitState.OPEN
```

### Observability Testing

```python
from observability import track_request_metrics

with track_request_metrics("POST", "/analyze"):
    # Execute request
    pass
```

## Deployment

### AWS Lambda

```bash
# Package
zip -r function.zip . -x "*.git*" -x "__pycache__/*"

# Deploy
aws lambda create-function \
  --function-name content-detector \
  --runtime python3.11 \
  --handler serverless_optimizer.lambda_handler \
  --zip-file fileb://function.zip
```

### Docker (Microservices)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: content-detector
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: app
        image: content-detector:latest
        ports:
        - containerPort: 8000
        env:
        - name: OTLP_ENDPOINT
          value: "jaeger:4317"
```

## Best Practices

1. **Stateless Services**: Usar Redis para state persistence
2. **Circuit Breakers**: Proteger todas las llamadas externas
3. **Async Tasks**: Mover procesamiento pesado a Celery
4. **Structured Logging**: Siempre usar JSON logs
5. **Tracing**: Habilitar en producción para debugging
6. **Rate Limiting**: Configurar por endpoint y usuario
7. **Security**: OAuth2 para usuarios, API keys para servicios
8. **Monitoring**: Métricas Prometheus + Alertas Grafana

## Performance Tuning

### Serverless Cold Start Reduction

1. **Lazy Loading**: Importar módulos pesados solo cuando se necesiten
2. **Connection Pooling**: Reutilizar conexiones dentro del contexto
3. **Warm-up**: Pre-cargar módulos comunes en startup
4. **Optimized JSON**: Usar orjson para serialización rápida

### Microservices Optimization

1. **Connection Pooling**: Reutilizar conexiones a Redis/DB
2. **Caching**: Usar Redis para resultados frecuentes
3. **Batch Processing**: Agrupar operaciones cuando sea posible
4. **Async Operations**: Usar async/await para I/O operations

## Troubleshooting

### Circuit Breaker Stuck Open

```python
from circuit_breaker import reset_all_circuit_breakers

reset_all_circuit_breakers()
```

### Tracing Not Working

Verificar:
1. OTLP endpoint configurado
2. OpenTelemetry dependencies instaladas
3. Service name correcto

### Celery Tasks Not Executing

Verificar:
1. Celery worker corriendo
2. Broker URL correcto
3. Task routing configurado

## Referencias

- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)
- [Celery Documentation](https://docs.celeryproject.org/)
- [AWS Lambda Best Practices](https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html)
- [Microservices Patterns](https://microservices.io/patterns/)






