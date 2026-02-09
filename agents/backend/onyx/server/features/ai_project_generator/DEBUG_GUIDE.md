# Guía de Debugging

Guía completa de herramientas de debugging implementadas.

## 🐛 Características de Debugging

### 1. Debug Logger

Logger avanzado con contexto y formateo estructurado.

```python
from debug.debug_logger import get_debug_logger

logger = get_debug_logger()

# Log básico
logger.debug("Mensaje de debug")
logger.info("Información")
logger.error("Error", exc_info=True)

# Log con contexto
logger.set_context(user_id="123", request_id="abc")
logger.debug("Operación realizada")

# Log de requests/responses
logger.log_request("GET", "/api/v1/projects")
logger.log_response(200, 0.05)

# Log de servicios
logger.log_service_call("ProjectService", "create_project", 0.1, True)
```

### 2. Error Tracker

Tracking y análisis de errores con contexto.

```python
from debug.error_tracker import get_error_tracker

tracker = get_error_tracker()

# Trackear error
try:
    # código
    pass
except Exception as e:
    tracker.track_error(e, context={"key": "value"})

# Obtener errores
errors = tracker.get_recent_errors(limit=50)
stats = tracker.get_error_stats()
```

### 3. Debug Middleware

Middleware automático para debugging de requests.

```python
from debug.debug_middleware import DebugMiddleware

app.add_middleware(DebugMiddleware, enable_debug=True)
```

**Características:**
- Log automático de todos los requests/responses
- Tracking de errores
- Medición de tiempos
- Headers de debug

### 4. Profiler

Profiler de performance para analizar código.

```python
from debug.profiler import get_profiler

profiler = get_profiler()

# Context manager
with profiler.profile("my_operation"):
    # código a profilear
    pass

# Decorator
@profiler.profile_function
async def my_function():
    pass

# Estadísticas
stats = profiler.get_stats()
slowest = profiler.get_slowest_functions()
```

### 5. Debug Endpoints

Endpoints HTTP para debugging.

```
GET /debug/health          # Health check detallado
GET /debug/errors          # Errores recientes
GET /debug/errors/stats    # Estadísticas de errores
GET /debug/memory          # Información de memoria
GET /debug/cache/stats     # Estadísticas de cache
GET /debug/profiler/stats # Estadísticas del profiler
GET /debug/services        # Estado de servicios
GET /debug/config          # Configuración de debugging
```

### 6. Request Debugger

Herramientas para debugging de requests.

```python
from debug.request_debugger import RequestDebugger

debugger = RequestDebugger()

# Debug de request
info = debugger.debug_request(request)
body = await debugger.debug_request_body(request)

# Debug de response
response_info = debugger.debug_response(response, duration)
```

### 7. Service Debugger

Debugging de servicios con decorators.

```python
from debug.service_debugger import get_service_debugger

service_debugger = get_service_debugger()

@service_debugger.debug_service_call("ProjectService", "create_project")
async def create_project(...):
    pass

# Estadísticas
calls = service_debugger.get_service_calls()
stats = service_debugger.get_service_stats()
```

## ⚙️ Configuración

### Variables de Entorno

```bash
# Habilitar debugging
DEBUG=true
DEBUG_ENABLED=true

# Configuración de logging
DEBUG_LOG_LEVEL=DEBUG
DEBUG_LOG_TO_FILE=true
DEBUG_LOG_DIR=logs

# Error tracking
DEBUG_ERROR_TRACKING=true
DEBUG_MAX_ERROR_HISTORY=1000

# Profiling
DEBUG_PROFILING=true
DEBUG_PROFILE_SLOW_THRESHOLD=1.0

# Request debugging
DEBUG_LOG_REQUESTS=true
DEBUG_LOG_RESPONSES=true
DEBUG_LOG_REQUEST_BODY=false
```

### Configuración Programática

```python
from debug.debug_config import DebugConfig

config = DebugConfig(
    debug_enabled=True,
    debug_logging=True,
    error_tracking=True,
    profiling=True
)
```

## 🎯 Uso

### Desarrollo

```python
# Habilitar debugging en desarrollo
import os
os.environ["DEBUG"] = "true"

from core.easy_setup import quick_start
app = quick_start()  # Debugging se habilita automáticamente
```

### Producción

```python
# Deshabilitar debugging en producción
import os
os.environ["DEBUG"] = "false"

from core.easy_setup import create_app_production
app = create_app_production()
```

## 📊 Endpoints de Debug

Una vez habilitado, accede a:

- `GET /debug/health` - Health check detallado
- `GET /debug/errors` - Lista de errores recientes
- `GET /debug/errors/stats` - Estadísticas de errores
- `GET /debug/memory` - Uso de memoria
- `GET /debug/cache/stats` - Estadísticas de cache
- `GET /debug/profiler/stats` - Estadísticas de profiling
- `GET /debug/services` - Estado de servicios

## 🔍 Ejemplos

### Debug de Request Lento

```python
from debug.debug_middleware import PerformanceDebugMiddleware

app.add_middleware(PerformanceDebugMiddleware)
# Detecta automáticamente requests > 1 segundo
```

### Trackear Error Específico

```python
from debug.error_tracker import get_error_tracker

tracker = get_error_tracker()

try:
    result = await service.do_something()
except Exception as e:
    tracker.track_error(
        e,
        context={"service": "my_service", "operation": "do_something"},
        request_id=request_id
    )
    raise
```

### Profilear Función

```python
from debug.profiler import get_profiler

profiler = get_profiler()
profiler.enable()

@profiler.profile_function
async def expensive_operation():
    # código costoso
    pass

# Ver estadísticas
stats = profiler.get_stats()
```

## 🛡️ Seguridad

**IMPORTANTE**: Los endpoints de debug solo deben estar habilitados en desarrollo.

En producción:
- Deshabilitar `DEBUG=false`
- No exponer endpoints de debug públicamente
- Usar autenticación para endpoints de debug si es necesario

## 📝 Best Practices

1. **Usar en Desarrollo**: Habilitar debugging solo en desarrollo
2. **Contexto**: Siempre agregar contexto a los logs
3. **Error Tracking**: Trackear todos los errores importantes
4. **Profiling**: Profilear código lento
5. **Logs Estructurados**: Usar logs estructurados para mejor análisis
6. **Limpiar Logs**: Limpiar logs antiguos regularmente

## 🚀 Próximas Mejoras

- [ ] Integración con Sentry
- [ ] Dashboard de debugging
- [ ] Alertas automáticas
- [ ] Exportación de logs
- [ ] Análisis de patrones de errores















