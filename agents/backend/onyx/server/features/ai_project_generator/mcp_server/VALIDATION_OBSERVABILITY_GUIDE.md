# Guía de Validación Avanzada y Observabilidad - MCP Server

## Resumen

Sistema de validación composable y utilidades de observabilidad (métricas y trazas) para el módulo MCP Server.

## Validación Avanzada

### Validadores Básicos

```python
from mcp_server.utils.validation_advanced import (
    Validator, required, not_empty, min_length, max_length,
    min_value, max_value, email, url, combine
)

# Validar campo requerido
validator = required()
validator("value")  # OK
validator(None)  # Raises ValidationError

# Validar longitud
validator = combine(min_length(3), max_length(50))
validator("test")  # OK
validator("ab")  # Raises ValidationError

# Validar email
validator = email()
validator("user@example.com")  # OK
validator("invalid")  # Raises ValidationError
```

### Validadores Personalizados

```python
from mcp_server.utils.validation_advanced import Validator, custom

# Crear validador personalizado
def is_positive(x):
    return None if x > 0 else "Must be positive"

validator = custom(is_positive)
validator(5)  # OK
validator(-1)  # Raises ValidationError

# Combinar validadores
validator = combine(
    required(),
    custom(is_positive),
    min_value(1),
    max_value(100)
)
```

### Validación de Esquemas

```python
from mcp_server.utils.validation_advanced import SchemaValidator

# Definir esquema
schema = {
    "name": {
        "type": str,
        "required": True,
        "min_length": 3,
        "max_length": 50
    },
    "age": {
        "type": int,
        "required": True,
        "min": 0,
        "max": 120
    },
    "email": {
        "type": str,
        "required": True,
        "pattern": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    }
}

# Validar datos
validator = SchemaValidator(schema)
data = {
    "name": "John Doe",
    "age": 30,
    "email": "john@example.com"
}

is_valid, errors = validator.validate(data)
if not is_valid:
    print("Errors:", errors)
```

## Observabilidad

### Métricas

```python
from mcp_server.utils.observability_utils import (
    get_metrics_collector, measure_time
)

collector = get_metrics_collector()

# Incrementar contador
collector.increment("requests.total", tags={"endpoint": "/api/users"})

# Establecer gauge
collector.set_gauge("cache.size", 1024, tags={"cache": "users"})

# Registrar histograma
collector.record_histogram("response.time", 0.123, tags={"endpoint": "/api/users"})

# Obtener métricas
metrics = collector.get_metrics()
counters = collector.get_counters()
gauges = collector.get_gauges()

# Estadísticas de histograma
stats = collector.get_histogram_stats("response.time")
# {"count": 100, "min": 0.01, "max": 0.5, "mean": 0.123, "p50": 0.1, "p95": 0.3, "p99": 0.4}
```

### Medición de Tiempo

```python
from mcp_server.utils.observability_utils import measure_time

# Medir tiempo de operación
with measure_time("api_call", tags={"endpoint": "/users"}):
    # código a medir
    result = process_request()
```

### Trazas

```python
from mcp_server.utils.observability_utils import get_tracer

tracer = get_tracer()

# Crear traza con spans
with tracer.span("request_processing", request_id="123") as span:
    span.add_tag("method", "GET")
    span.add_tag("endpoint", "/api/users")
    
    with tracer.span("database_query") as db_span:
        db_span.add_tag("query", "SELECT * FROM users")
        # ejecutar query
        db_span.finish()
    
    with tracer.span("cache_lookup") as cache_span:
        # buscar en cache
        cache_span.finish()
    
    span.finish()

# Obtener trazas
spans = tracer.get_spans()
```

## Ejemplos de Uso

### Validación de Request

```python
from mcp_server.utils.validation_advanced import SchemaValidator

request_schema = {
    "resource_id": {
        "type": str,
        "required": True,
        "min_length": 1,
        "max_length": 255
    },
    "operation": {
        "type": str,
        "required": True,
        "choices": ["read", "write", "delete"]
    },
    "parameters": {
        "type": dict,
        "required": False
    }
}

def validate_request(data: dict) -> tuple[bool, list[str]]:
    validator = SchemaValidator(request_schema)
    return validator.validate(data)
```

### Métricas de API

```python
from mcp_server.utils.observability_utils import (
    get_metrics_collector, measure_time
)

collector = get_metrics_collector()

def handle_request(request):
    with measure_time("request.duration", tags={"method": request.method}):
        collector.increment("requests.total", tags={"endpoint": request.path})
        
        try:
            result = process_request(request)
            collector.increment("requests.success")
            return result
        except Exception as e:
            collector.increment("requests.error", tags={"error": type(e).__name__})
            raise
```

### Traza Completa de Request

```python
from mcp_server.utils.observability_utils import get_tracer

tracer = get_tracer()

def process_request(request):
    with tracer.span("request", request_id=request.id) as span:
        span.add_tag("method", request.method)
        span.add_tag("endpoint", request.path)
        
        # Autenticación
        with tracer.span("authentication") as auth_span:
            user = authenticate(request)
            auth_span.add_tag("user_id", user.id)
        
        # Procesamiento
        with tracer.span("processing") as proc_span:
            result = process(user, request)
            proc_span.add_tag("result_size", len(result))
        
        span.add_log("Request completed successfully")
        return result
```

## Próximos Pasos

1. Agregar más validadores predefinidos
2. Integrar con sistemas de métricas externos (Prometheus, StatsD)
3. Agregar exportación de trazas (Jaeger, Zipkin)
4. Mejorar visualización de métricas
5. Agregar alertas basadas en métricas

