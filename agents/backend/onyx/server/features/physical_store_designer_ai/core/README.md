# Core Module

Módulo central con componentes reutilizables del Physical Store Designer AI.

## Estructura

```
core/
├── models.py              # Modelos Pydantic
├── exceptions.py          # Excepciones personalizadas
├── interfaces.py          # Interfaces/Protocols
├── factories.py           # Factories para servicios
├── validators.py          # Validadores reutilizables
├── decorators.py          # Decoradores útiles
├── dependencies.py        # Dependencies de FastAPI
├── response_models.py     # Modelos de respuesta estándar
├── logging_config.py      # Configuración de logging
├── service_base.py        # Clases base para servicios
├── service_registry.py    # Registro de servicios
├── route_helpers.py       # Helpers para rutas
├── route_utils.py         # Utilidades para rutas
├── metrics.py             # Sistema de métricas
├── cache.py               # Sistema de caché
├── timeout.py             # Manejo de timeouts
├── circuit_breaker/       # Circuit breaker modular
├── compression.py         # Compresión
├── rate_limiting.py       # Rate limiting
├── serializers.py         # Serialización
├── background_tasks.py    # Tareas en background
├── middleware/            # Middleware personalizado
└── utils/                 # Utilidades generales
```

## Componentes Principales

### Models
- `StoreDesign`, `StoreDesignRequest`, `ChatSession`, etc.

### Exceptions
- `NotFoundError`, `ValidationError`, `StorageError`, etc.

### Services Base
- `BaseService`: Clase base para todos los servicios
- `TimestampedService`: Servicio con timestamps

### Circuit Breaker
Ver `circuit_breaker/README.md` para documentación completa.

### Middleware
- `SecurityHeadersMiddleware`
- `ErrorHandlerMiddleware`
- `RequestLoggingMiddleware`
- `RateLimitMiddleware`
- `CompressionMiddleware`
- `TimeoutMiddleware`

### Utils
Organizados en subdirectorios:
- `data_utils.py`: Manipulación de datos
- `dict_utils.py`: Utilidades de diccionarios
- `file_utils.py`: Operaciones de archivos
- `format_utils.py`: Formateo
- `json_utils.py`: JSON
- `logging_utils.py`: Logging
- `math_utils.py`: Matemáticas
- `validation_utils.py`: Validación

## Uso

```python
from core.models import StoreDesign, StoreDesignRequest
from core.exceptions import NotFoundError
from core.service_base import BaseService
from core.logging_config import get_logger
from core.circuit_breaker import CircuitBreaker
```


