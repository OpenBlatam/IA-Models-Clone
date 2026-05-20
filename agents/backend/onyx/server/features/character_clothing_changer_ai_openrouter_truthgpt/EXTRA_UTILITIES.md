# Utilidades Adicionales Implementadas

## 🛠️ Nuevas Utilidades

### 1. Decorators (`utils/decorators.py`) ⭐ NUEVO
- **@log_execution_time**: Log del tiempo de ejecución
- **@retry_on_failure**: Retry automático con exponential backoff
- **@cache_result**: Cache de resultados con TTL
- **@validate_inputs**: Validación de inputs
- **@rate_limit**: Rate limiting a nivel de función

#### Características:
- Soporte para funciones async y sync
- Configuración flexible
- Logging integrado
- Error handling robusto

### 2. Middleware Adicional ⭐ NUEVO

#### LoggingMiddleware (`middleware/logging_middleware.py`)
- **Request logging**: Log de todas las requests
- **Response logging**: Log de todas las responses
- **Duration tracking**: Tiempo de ejecución
- **Configurable**: Paths excluidos, niveles de log
- **Body logging**: Opcional para debugging

#### ErrorHandlerMiddleware (`middleware/error_handler_middleware.py`)
- **Error handling centralizado**: Captura todas las excepciones
- **Formato consistente**: Respuestas de error estandarizadas
- **Logging de errores**: Con contexto completo
- **Status codes apropiados**: 400, 500, etc.

### 3. Custom Exceptions (`utils/exceptions.py`) ⭐ NUEVO
- **ClothingChangeError**: Excepción base
- **WorkflowError**: Errores de workflow
- **ComfyUIError**: Errores de ComfyUI
- **OpenRouterError**: Errores de OpenRouter
- **TruthGPTError**: Errores de TruthGPT
- **ValidationError**: Errores de validación
- **BatchProcessingError**: Errores de batch
- **CacheError**: Errores de cache
- **RateLimitError**: Errores de rate limit

### 4. Performance Utilities (`utils/performance.py`) ⭐ NUEVO
- **PerformanceMonitor**: Monitor de performance
- **measure_performance**: Context manager para medir
- **track_performance**: Decorator para tracking
- **Estadísticas agregadas**: Min, max, avg, total
- **Métricas con metadata**: Información adicional

## 📝 Ejemplos de Uso

### Decorators

#### Log Execution Time
```python
from utils.decorators import log_execution_time

@log_execution_time
async def my_function():
    # Function code
    pass
```

#### Retry on Failure
```python
from utils.decorators import retry_on_failure

@retry_on_failure(max_retries=3, delay=1.0)
async def api_call():
    # API call that might fail
    pass
```

#### Cache Result
```python
from utils.decorators import cache_result

@cache_result(ttl=3600)
async def expensive_operation(param):
    # Expensive operation
    pass
```

#### Validate Inputs
```python
from utils.decorators import validate_inputs

@validate_inputs(
    image_url=lambda x: x.startswith('http'),
    num_steps=lambda x: 1 <= x <= 100
)
async def process_image(image_url, num_steps):
    # Process image
    pass
```

#### Rate Limit
```python
from utils.decorators import rate_limit

@rate_limit(calls=10, period=60)
async def api_endpoint():
    # API endpoint
    pass
```

### Performance Monitoring

#### Context Manager
```python
from utils.performance import measure_performance

with measure_performance("my_operation", {"param": "value"}):
    # Code to measure
    result = expensive_operation()
```

#### Decorator
```python
from utils.performance import track_performance

@track_performance("my_function")
async def my_function():
    # Function code
    pass
```

#### Get Stats
```python
from utils.performance import get_performance_monitor

monitor = get_performance_monitor()
stats = monitor.get_stats("my_function")
# Returns: {"count": 10, "min": 0.1, "max": 2.5, "avg": 1.2, "total": 12.0}
```

### Custom Exceptions

```python
from utils.exceptions import WorkflowError, ValidationError

# Raise custom exception
if not image_url:
    raise ValidationError(
        "image_url is required",
        error_code="MISSING_IMAGE_URL",
        details={"field": "image_url"}
    )

# Catch and handle
try:
    result = await workflow.execute()
except WorkflowError as e:
    logger.error(f"Workflow error: {e.message}, code: {e.error_code}")
```

## 🔧 Integración de Middleware

### Logging Middleware
```python
from middleware.logging_middleware import LoggingMiddleware

app.add_middleware(
    LoggingMiddleware,
    log_requests=True,
    log_responses=True,
    log_body=False,
    exclude_paths=["/health", "/docs"]
)
```

### Error Handler Middleware
```python
from middleware.error_handler_middleware import ErrorHandlerMiddleware

app.add_middleware(ErrorHandlerMiddleware)
```

### Rate Limit Middleware
```python
from middleware.rate_limit_middleware import RateLimitMiddleware

app.add_middleware(
    RateLimitMiddleware,
    default_limit=100,
    default_window=60.0
)
```

## 📊 Beneficios

1. **Decorators Reutilizables**: Funcionalidad común encapsulada
2. **Middleware Centralizado**: Logging y error handling unificados
3. **Excepciones Específicas**: Mejor manejo de errores
4. **Performance Tracking**: Monitoreo integrado
5. **Código Más Limpio**: Menos repetición
6. **Mejor Debugging**: Logging y tracking completos

## 🎯 Casos de Uso

### Decorators
- **Logging automático** de tiempos de ejecución
- **Retry automático** en operaciones que pueden fallar
- **Caching** de resultados costosos
- **Validación** de inputs antes de ejecutar
- **Rate limiting** a nivel de función

### Middleware
- **Logging centralizado** de todas las requests
- **Error handling** consistente
- **Rate limiting** automático
- **Performance tracking** de requests

### Excepciones
- **Error handling** más específico
- **Códigos de error** para clientes
- **Detalles adicionales** en errores
- **Mejor debugging** con contexto

### Performance
- **Tracking automático** de performance
- **Estadísticas agregadas**
- **Identificación de bottlenecks**
- **Optimización basada en datos**

## ✅ Estado

- ✅ **Decorators implementados** y listos para usar
- ✅ **Middleware adicional** para logging y errores
- ✅ **Excepciones personalizadas** para mejor manejo
- ✅ **Performance utilities** para monitoreo
- ✅ **Documentación completa** con ejemplos

El sistema ahora tiene utilidades avanzadas para desarrollo, debugging y optimización.

