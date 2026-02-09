# Guía de Utilidades

Este documento describe todas las utilidades disponibles en el proyecto.

## Health Checker

**Ubicación**: `utils/health_checker.py`

Sistema de health checks para monitorear el estado de la aplicación.

### Uso Básico

```python
from utils.health_checker import get_health_checker

health_checker = get_health_checker()

# Registrar un check
async def check_database():
    # Verificar conexión a base de datos
    return True, "Database is healthy"

health_checker.register_check("database", check_database)

# Ejecutar todos los checks
results = await health_checker.run_all_checks()
```

### Ejemplo Completo

```python
from utils.health_checker import get_health_checker

health_checker = get_health_checker()

# Check de storage
async def check_storage():
    from pathlib import Path
    from config.settings import settings
    
    storage_path = Path(settings.output_dir)
    if storage_path.exists() and storage_path.is_dir():
        return True, "Storage is accessible"
    return False, "Storage not accessible"

health_checker.register_check("storage", check_storage)

# Ejecutar checks
results = await health_checker.run_all_checks(timeout=5.0)
```

## Rate Limiter

**Ubicación**: `utils/rate_limiter.py`

Sistema de rate limiting usando token bucket algorithm.

### Uso Básico

```python
from utils.rate_limiter import get_rate_limiter

rate_limiter = get_rate_limiter()

# Crear bucket: 100 requests, 10 por segundo
bucket = rate_limiter.create_bucket(
    key="api_requests",
    capacity=100,
    refill_rate=10.0
)

# Verificar si está permitido
if await rate_limiter.is_allowed("api_requests", tokens=1):
    # Procesar request
    pass
else:
    # Rate limit excedido
    raise RateLimitExceededError()
```

### Token Bucket Directo

```python
from utils.rate_limiter import TokenBucket

bucket = TokenBucket(capacity=100, refill_rate=10.0)

# Adquirir tokens
if await bucket.acquire(tokens=1):
    # Procesar
    pass

# Esperar por tokens
if await bucket.wait_for_token(tokens=1, timeout=60.0):
    # Procesar
    pass
```

## Circuit Breaker

**Ubicación**: `utils/circuit_breaker.py`

Patrón Circuit Breaker para tolerancia a fallos.

### Uso Básico

```python
from utils.circuit_breaker import CircuitBreaker, CircuitState

breaker = CircuitBreaker(
    failure_threshold=5,      # Abrir después de 5 fallos
    recovery_timeout=60.0,     # Esperar 60s antes de reintentar
    expected_exception=ConnectionError
)

# Usar con función
try:
    result = await breaker.call(risky_function, arg1, arg2)
except CircuitBreakerOpenError:
    # Circuit está abierto, servicio no disponible
    pass
```

### Como Decorator

```python
from utils.circuit_breaker import CircuitBreaker

breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)

@breaker.call
async def risky_operation():
    # Tu código aquí
    pass
```

### Estados

- **CLOSED**: Operación normal
- **OPEN**: Fallando, rechaza requests
- **HALF_OPEN**: Probando si el servicio se recuperó

## Exponential Backoff

**Ubicación**: `utils/backoff.py`

Retry con exponential backoff.

### Uso Básico

```python
from utils.retry import exponential_backoff

result = await exponential_backoff(
    func=lambda: make_api_call(),
    max_attempts=3,
    initial_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True,
    exceptions=(ConnectionError, TimeoutError)
)
```

### Como Decorator

```python
from utils.retry import retry_with_backoff

@retry_with_backoff(
    max_attempts=3,
    initial_delay=1.0,
    max_delay=60.0,
    jitter=True
)
async def unreliable_function():
    # Tu código aquí
    pass
```

### Parámetros

- `max_attempts`: Número máximo de intentos
- `initial_delay`: Delay inicial en segundos
- `max_delay`: Delay máximo en segundos
- `exponential_base`: Base para cálculo exponencial
- `jitter`: Agregar variación aleatoria al delay
- `exceptions`: Excepciones que activan retry

## Observability (Tracing)

**Ubicación**: `utils/observability.py`

Sistema simple de tracing para observabilidad.

### Uso Básico

```python
from utils.observability import get_tracer

tracer = get_tracer()

# Usar como context manager
async with tracer.span("process_image") as span:
    span.set_tag("image_id", "123")
    span.log("Starting image processing")
    
    # Tu código aquí
    result = await process_image()
    
    span.set_tag("result", "success")
    span.log("Image processed successfully")

# Obtener trace completo
trace = tracer.get_trace()
```

### Spans Manuales

```python
from utils.observability import get_tracer

tracer = get_tracer()

span = tracer.start_span("operation")
span.set_tag("key", "value")
span.log("Event occurred")

# Tu código aquí

tracer.finish_span(span)
```

## Batch Processor

**Ubicación**: `utils/batch_processor.py`

Procesamiento de items en lotes con control de concurrencia.

### Uso Básico

```python
from utils.batch_processor import BatchProcessor

processor = BatchProcessor(
    batch_size=10,        # Items por batch
    max_concurrent=5,     # Operaciones concurrentes
    stop_on_error=False   # Continuar en error
)

items = [1, 2, 3, 4, 5, ...]

async def process_item(item):
    # Procesar item
    return item * 2

results = await processor.process(items, process_item)
```

### Con Progress Callback

```python
def progress_callback(current, total):
    print(f"Progress: {current}/{total} ({current/total*100:.1f}%)")

results = await processor.process(
    items,
    process_item,
    progress_callback=progress_callback
)
```

### Con Retry

```python
results = await processor.process_with_retry(
    items,
    process_item,
    max_retries=3
)
```

## Ejemplos de Integración

### Health Check en Endpoint

```python
from fastapi import APIRouter
from utils.health_checker import get_health_checker

router = APIRouter()

@router.get("/health/detailed")
async def detailed_health():
    health_checker = get_health_checker()
    return await health_checker.run_all_checks()
```

### Rate Limiting en Endpoint

```python
from fastapi import APIRouter, HTTPException
from utils.rate_limiter import get_rate_limiter
from core.exceptions import RateLimitExceededError

router = APIRouter()
rate_limiter = get_rate_limiter()

# Crear bucket al inicio
rate_limiter.create_bucket("api", capacity=100, refill_rate=10.0)

@router.post("/api/endpoint")
async def endpoint():
    if not await rate_limiter.is_allowed("api"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Tu código aquí
    pass
```

### Circuit Breaker en Service

```python
from utils.circuit_breaker import CircuitBreaker

class ExternalAPIService:
    def __init__(self):
        self.breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0
        )
    
    async def call_external_api(self):
        return await self.breaker.call(
            self._make_api_call
        )
    
    async def _make_api_call(self):
        # Llamada real a API externa
        pass
```

### Tracing en Use Case

```python
from utils.observability import get_tracer

class CreateVisualizationUseCase:
    async def execute(self, request):
        tracer = get_tracer()
        
        async with tracer.span("create_visualization") as span:
            span.set_tag("surgery_type", request.surgery_type.value)
            
            async with tracer.span("load_image") as img_span:
                image = await self._load_image(request)
                img_span.set_tag("image_size", len(image))
            
            async with tracer.span("process_ai") as ai_span:
                result = await self._process_ai(image)
                ai_span.set_tag("processing_time", result.time)
            
            return result
```

## Mejores Prácticas

1. **Health Checks**: Registrar checks al inicio de la aplicación
2. **Rate Limiting**: Crear buckets al inicio, no en cada request
3. **Circuit Breaker**: Usar para llamadas externas críticas
4. **Backoff**: Usar para operaciones que pueden fallar temporalmente
5. **Tracing**: Agregar spans en operaciones importantes
6. **Batch Processing**: Usar para procesar grandes volúmenes de datos

## Configuración Recomendada

```python
# Health checks
health_checker.register_check("database", check_database)
health_checker.register_check("storage", check_storage)
health_checker.register_check("cache", check_cache)

# Rate limiting
rate_limiter.create_bucket("api", capacity=100, refill_rate=10.0)
rate_limiter.create_bucket("heavy_ops", capacity=10, refill_rate=1.0)

# Circuit breakers
api_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
db_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
```

