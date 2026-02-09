# Circuit Breaker - Estructura del Código Refactorizado

## 📁 Estructura de Archivos

```
core/
│
├── circuit_breaker.py (1585 líneas)
│   ├── Imports desde módulos refactorizados
│   ├── Clase CircuitBreaker (principal)
│   ├── Decorator circuit_breaker()
│   ├── Registry functions
│   ├── CircuitBreakerGroup
│   ├── CircuitBreakerChain
│   ├── CircuitBreakerStateStore
│   └── OpenTelemetry integration
│
└── circuit_breaker/ (Paquete refactorizado)
    ├── __init__.py
    │   └── Exporta: CircuitState, CircuitBreakerEventType, 
    │                CircuitBreakerConfig, CircuitBreakerMetrics, 
    │                CircuitBreakerEvent
    │
    ├── circuit_types.py
    │   ├── CircuitState (Enum)
    │   └── CircuitBreakerEventType (Enum)
    │
    ├── config.py
    │   └── CircuitBreakerConfig (dataclass)
    │
    ├── metrics.py
    │   └── CircuitBreakerMetrics (dataclass)
    │
    └── events.py
        ├── CircuitBreakerEvent (dataclass)
        └── EventEmitter (class)
```

## 🔄 Flujo de Imports

### Desde el archivo principal:
```python
# circuit_breaker.py
from .circuit_breaker.circuit_types import CircuitState, CircuitBreakerEventType
from .circuit_breaker.events import CircuitBreakerEvent
from .circuit_breaker.config import CircuitBreakerConfig
from .circuit_breaker.metrics import CircuitBreakerMetrics
```

### Uso desde código externo:
```python
# Opción 1: Desde archivo principal (recomendado)
from core.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitBreakerEvent
)

# Opción 2: Desde módulos específicos
from core.circuit_breaker.circuit_types import CircuitState
from core.circuit_breaker.config import CircuitBreakerConfig
```

## 📊 Estadísticas

### Archivo Principal (`circuit_breaker.py`)
- **Líneas**: 1585 (reducido de 1725)
- **Clases principales**: 4
  - `CircuitBreaker`
  - `CircuitBreakerGroup`
  - `CircuitBreakerChain`
  - `CircuitBreakerStateStore`
- **Funciones**: 10+
- **Decorators**: 1

### Módulos Refactorizados
- **circuit_types.py**: 2 enums
- **config.py**: 1 dataclass (20+ campos)
- **metrics.py**: 1 dataclass (10+ campos, 3 métodos)
- **events.py**: 1 dataclass + 1 clase EventEmitter

## ✅ Componentes Exportados

### Desde `circuit_breaker.py`:
```python
__all__ = [
    "CircuitState",
    "CircuitBreakerEventType", 
    "CircuitBreakerEvent",
    "CircuitBreakerConfig",
    "CircuitBreakerMetrics",
    # + CircuitBreaker, circuit_breaker, etc.
]
```

### Desde `circuit_breaker/__init__.py`:
```python
__all__ = [
    "CircuitState",
    "CircuitBreakerEventType",
    "CircuitBreakerConfig",
    "CircuitBreakerMetrics",
    "CircuitBreakerEvent",
]
```

## 🎯 Separación de Responsabilidades

| Módulo | Responsabilidad |
|--------|----------------|
| `circuit_types.py` | Definiciones de tipos y enums |
| `config.py` | Configuración del circuit breaker |
| `metrics.py` | Métricas y estadísticas |
| `events.py` | Eventos de dominio y emisión |
| `circuit_breaker.py` | Lógica principal del circuit breaker |

## 🔍 Dependencias

```
circuit_breaker.py
    ├── circuit_types.py (CircuitState, CircuitBreakerEventType)
    ├── config.py (CircuitBreakerConfig)
    ├── metrics.py (CircuitBreakerMetrics)
    └── events.py (CircuitBreakerEvent)
```

## ✨ Beneficios de la Estructura

1. **Modularidad**: Cada componente en su propio archivo
2. **Mantenibilidad**: Fácil encontrar y modificar código
3. **Reutilización**: Módulos pueden importarse independientemente
4. **Escalabilidad**: Fácil agregar nuevos módulos
5. **Compatibilidad**: 100% backward compatible

## 📝 Notas

- El archivo `types.py` antiguo puede eliminarse (ya renombrado a `circuit_types.py`)
- Todos los módulos compilan sin errores
- La estructura permite futuras extensiones sin afectar código existente




