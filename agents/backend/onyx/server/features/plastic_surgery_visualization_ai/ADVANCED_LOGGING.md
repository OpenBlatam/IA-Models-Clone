# Sistema de Logging Avanzado - Plastic Surgery Visualization AI

## Resumen

Sistema de logging mejorado con soporte para logging estructurado, middleware de logging, y utilidades de desarrollo.

## 1. Logging Estructurado

### Archivo: `utils/logger.py` (MEJORADO)

**Mejoras**:
- Soporte para structlog (opcional)
- JSON logging
- Formateo estructurado
- Context-aware logging

**Funciones**:
- `setup_logging()` - Configuración con opción JSON
- `get_logger()` - Logger con soporte structlog
- `log_request()` - Helper para logging de requests

**Uso**:
```python
from utils.logger import setup_logging, get_logger

# Setup con JSON logging
setup_logging(log_level="INFO", use_json=True)

logger = get_logger(__name__)
logger.info("message", extra={"key": "value"})
```

## 2. Middleware de Logging

### Archivo: `middleware/logging_middleware.py` (NUEVO)

**Funcionalidades**:
- Logging automático de requests/responses
- Tracking de tiempo de procesamiento
- Detección de requests lentos
- Headers de performance

**Características**:
- Skip automático de health checks
- Logging estructurado
- Métricas automáticas
- Alertas de performance

## 3. Manejo de Errores Mejorado

### Archivo: `utils/error_handler.py` (NUEVO)

**Funcionalidades**:
- `format_error_response()` - Formateo de errores
- `handle_exception()` - Manejo centralizado
- `safe_execute()` - Ejecución segura sync
- `safe_execute_async()` - Ejecución segura async

**Características**:
- Logging automático de errores
- Métricas de errores
- Status codes apropiados
- Traceback opcional en debug

## 4. Utilidades Async

### Archivo: `utils/async_utils.py` (NUEVO)

**Funciones**:
- `parallel_limit()` - Procesamiento paralelo con límite
- `sequential()` - Procesamiento secuencial
- `timeout()` - Decorador de timeout
- `retry_async_with_backoff()` - Retry con backoff

**Uso**:
```python
from utils.async_helpers import parallel_limit, timeout_decorator

# Procesar con límite de concurrencia
results = await parallel_limit(items, process_item, concurrency=3)

# Timeout decorator
@timeout(30.0)
async def fetch_data():
    # código
    pass
```

## 5. Utilidades de Desarrollo

### Archivo: `utils/development.py` (NUEVO)

**Funciones**:
- `is_development()` - Verificar modo desarrollo
- `is_production()` - Verificar modo producción
- `get_project_root()` - Obtener raíz del proyecto
- `setup_dev_environment()` - Setup automático
- `print_debug_info()` - Info de debug

**Uso**:
```python
from utils.development import is_development, setup_dev_environment

if is_development():
    setup_dev_environment()
    # código de desarrollo
```

## 6. Configuración Mejorada

### Archivo: `config/settings.py` (MEJORADO)

**Nuevas configuraciones**:
- `use_json_logging` - Habilitar JSON logging
- `log_requests` - Habilitar logging de requests
- `environment` - Entorno (development/production/testing)
- `debug` - Modo debug

## 7. Integración en Main

### Archivo: `main.py` (MEJORADO)

**Mejoras**:
- Setup automático de entorno de desarrollo
- Logging configurable (JSON opcional)
- Debug info en desarrollo
- Mejor inicialización

## Estructura

```
middleware/
└── logging_middleware.py    # NUEVO

utils/
├── logger.py                # MEJORADO
├── error_handler.py         # NUEVO
├── async_utils.py           # NUEVO
└── development.py            # NUEVO
```

## Beneficios

1. **Logging Estructurado**: Mejor para análisis y monitoreo
2. **Middleware de Logging**: Tracking automático de requests
3. **Manejo de Errores**: Centralizado y consistente
4. **Utilidades Async**: Procesamiento eficiente
5. **Desarrollo**: Utilidades para desarrollo local
6. **Configuración**: Más opciones y flexibilidad

## Ejemplos

### Logging Estructurado
```python
logger.info("processing_image", image_id="123", size="2MB")
logger.error("processing_failed", error=str(e), exc_info=True)
```

### Async Utilities
```python
# Procesar en paralelo
results = await parallel_limit(items, process, concurrency=5)

# Con timeout
@timeout(30.0)
async def operation():
    pass
```

### Error Handling
```python
from utils.error_handler import safe_execute_async

result = await safe_execute_async(
    risky_operation(),
    default={"status": "failed"}
)
```

