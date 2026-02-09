# Circuit Breaker - Guía de Módulos

## 📚 Guía Completa de Todos los Módulos

### 🎯 Módulos Base

#### 1. `circuit_types.py`
**Propósito**: Definiciones de tipos y enums fundamentales

**Contenido**:
- `CircuitState` - Estados del circuit breaker (CLOSED, OPEN, HALF_OPEN)
- `CircuitBreakerEventType` - Tipos de eventos (12 tipos diferentes)

**Uso**:
```python
from core.circuit_breaker.circuit_types import CircuitState, CircuitBreakerEventType
```

#### 2. `config.py`
**Propósito**: Configuración del circuit breaker

**Contenido**:
- `CircuitBreakerConfig` - Dataclass con 20+ parámetros configurables

**Parámetros principales**:
- `failure_threshold`: Umbral de fallos
- `recovery_timeout`: Tiempo de recuperación
- `retry_enabled`: Habilitar retry
- `fallback_enabled`: Habilitar fallback
- `health_success_rate_threshold`: Umbral de salud

**Uso**:
```python
from core.circuit_breaker.config import CircuitBreakerConfig

config = CircuitBreakerConfig(
    failure_threshold=10,
    retry_enabled=True,
    fallback_enabled=True
)
```

#### 3. `metrics.py`
**Propósito**: Métricas y estadísticas

**Contenido**:
- `CircuitBreakerMetrics` - Dataclass con métricas completas

**Métricas incluidas**:
- Total requests, success, failure, rejected
- Success rate, failure rate
- Response times (avg, p50, p95, p99, min, max)
- Retry count, fallback count

**Uso**:
```python
from core.circuit_breaker.metrics import CircuitBreakerMetrics

metrics = breaker.get_metrics()
print(f"Success rate: {metrics.success_rate:.2%}")
```

#### 4. `events.py`
**Propósito**: Sistema de eventos de dominio

**Contenido**:
- `CircuitBreakerEvent` - Evento de dominio
- `EventEmitter` - Emisor de eventos (opcional)

**Tipos de eventos**:
- STATE_CHANGED, CIRCUIT_OPENED, CIRCUIT_CLOSED
- FAILURE_RECORDED, SUCCESS_RECORDED
- REQUEST_REJECTED, RETRY_ATTEMPTED, FALLBACK_USED
- Y más...

**Uso**:
```python
from core.circuit_breaker.events import CircuitBreakerEvent

def handle_event(event: CircuitBreakerEvent):
    print(f"Event: {event.event_type.value}")

breaker.on_event(handle_event)
```

### ⚡ Módulos Principales

#### 5. `breaker.py` ⭐
**Propósito**: Clase principal CircuitBreaker

**Contenido**:
- Clase `CircuitBreaker` completa (~1075 líneas)
- Todos los métodos públicos y privados

**Métodos principales**:
- `call()` - Ejecutar función con protección
- `call_with_fallback()` - Con fallback explícito
- `call_bulk()` - Operaciones en lote
- `reset()`, `force_open()`, `force_close()`
- Health checks: `is_healthy()`, `is_ready()`, `is_degraded()`, `is_critical()`
- `get_health_score()`, `get_health_rating()`, `get_health_status()`
- `update_config()` - Configuración dinámica
- `export_metrics_prometheus()`, `export_metrics_statsd()`
- Context manager: `async with breaker:`

**Uso**:
```python
from core.circuit_breaker.breaker import CircuitBreaker

breaker = CircuitBreaker(name="api_service")
result = await breaker.call(api_function, arg1, arg2)
```

#### 6. `registry.py`
**Propósito**: Registry global y decorator

**Contenido**:
- `circuit_breaker()` - Decorator
- `get_circuit_breaker()` - Obtener/crear breaker (async)
- `get_circuit_breaker_sync()` - Versión síncrona
- `get_all_circuit_breakers()` - Estado de todos
- `reset_all_circuit_breakers()` - Resetear todos

**Uso**:
```python
from core.circuit_breaker.registry import circuit_breaker, get_circuit_breaker

# Como decorator
@circuit_breaker(failure_threshold=5)
async def my_function():
    return await api_call()

# Desde registry
breaker = await get_circuit_breaker("my_service")
```

### 🔗 Módulos Avanzados

#### 7. `groups.py`
**Propósito**: Gestión de grupos de circuit breakers

**Contenido**:
- `CircuitBreakerGroup` - Grupo con configuración compartida

**Características**:
- Configuración compartida para múltiples breakers
- Health status del grupo completo
- Reset de todos los breakers del grupo

**Uso**:
```python
from core.circuit_breaker.groups import CircuitBreakerGroup

group = CircuitBreakerGroup("api_services", config=shared_config)
breaker1 = group.get_or_create("service1")
breaker2 = group.get_or_create("service2")
health = await group.get_health_status()
```

#### 8. `chain.py`
**Propósito**: Cadena de circuit breakers para operaciones secuenciales

**Contenido**:
- `CircuitBreakerChain` - Cadena de breakers

**Características**:
- Ejecución secuencial de funciones
- Falla en cualquier paso detiene la cadena
- Health status de toda la cadena

**Uso**:
```python
from core.circuit_breaker.chain import CircuitBreakerChain

chain = CircuitBreakerChain(breaker1, breaker2, breaker3)
results = await chain.call([func1, func2, func3], initial_arg)
```

#### 9. `tracing.py`
**Propósito**: Integración con OpenTelemetry

**Contenido**:
- `get_trace_context()` - Obtener contexto de tracing
- `add_tracing_to_circuit_breaker()` - Agregar tracing automático

**Uso**:
```python
from core.circuit_breaker.tracing import get_trace_context, add_tracing_to_circuit_breaker

# Obtener contexto
context = get_trace_context(breaker)

# Agregar tracing automático
add_tracing_to_circuit_breaker(breaker)
```

#### 10. `store.py`
**Propósito**: Persistencia de estado

**Contenido**:
- `CircuitBreakerStateStore` - Interface (ABC)
- `InMemoryStateStore` - Implementación en memoria
- `create_circuit_breaker_with_persistence()` - Factory

**Uso**:
```python
from core.circuit_breaker.store import (
    InMemoryStateStore,
    create_circuit_breaker_with_persistence
)

store = InMemoryStateStore()
breaker = create_circuit_breaker_with_persistence(
    "my_service",
    config=config,
    state_store=store
)
```

## 🔄 Flujo de Imports

### Opción 1: Desde módulo principal (Recomendado)
```python
from core.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerConfig,
    circuit_breaker,
    CircuitBreakerGroup,
    CircuitBreakerChain,
)
```

### Opción 2: Desde módulos específicos
```python
from core.circuit_breaker.breaker import CircuitBreaker
from core.circuit_breaker.registry import circuit_breaker
from core.circuit_breaker.groups import CircuitBreakerGroup
```

## 📊 Estadísticas por Módulo

| Módulo | Líneas | Responsabilidad |
|--------|--------|----------------|
| `circuit_types.py` | ~35 | Tipos y enums |
| `config.py` | ~40 | Configuración |
| `metrics.py` | ~95 | Métricas |
| `events.py` | ~95 | Eventos |
| `breaker.py` | ~1075 | Clase principal |
| `registry.py` | ~130 | Registry y decorator |
| `groups.py` | ~90 | Grupos |
| `chain.py` | ~100 | Cadenas |
| `tracing.py` | ~60 | Tracing |
| `store.py` | ~70 | Persistencia |
| **Total** | **~1690** | **Modular** |

## 🎯 Casos de Uso

### Caso 1: Uso Básico
```python
from core.circuit_breaker import CircuitBreaker

breaker = CircuitBreaker(name="api")
result = await breaker.call(api_function, arg1, arg2)
```

### Caso 2: Con Decorator
```python
from core.circuit_breaker import circuit_breaker

@circuit_breaker(failure_threshold=5)
async def my_api_call():
    return await external_api()
```

### Caso 3: Con Configuración Avanzada
```python
from core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

config = CircuitBreakerConfig(
    failure_threshold=10,
    retry_enabled=True,
    max_retries=3,
    fallback_enabled=True,
    fallback_func=default_response
)
breaker = CircuitBreaker(config=config, name="api")
```

### Caso 4: Con Health Checks
```python
health = breaker.get_health_status()
if health["critical"]:
    # Alertar o tomar acción
    pass
```

### Caso 5: Con Grupos
```python
from core.circuit_breaker import CircuitBreakerGroup

group = CircuitBreakerGroup("microservices")
service1 = group.get_or_create("user_service")
service2 = group.get_or_create("order_service")
```

## ✨ Mejores Prácticas

1. **Usar nombres descriptivos** para circuit breakers
2. **Configurar umbrales apropiados** según el servicio
3. **Habilitar retry y fallback** para servicios críticos
4. **Monitorear métricas** regularmente
5. **Usar health checks** en endpoints de health
6. **Agrupar breakers relacionados** con CircuitBreakerGroup
7. **Usar tracing** en producción para debugging

## 🚀 Próximos Pasos

- ✅ Refactorización completa
- ⏳ Tests unitarios (recomendado)
- ⏳ Documentación de API (recomendado)
- ⏳ Ejemplos de uso (recomendado)




