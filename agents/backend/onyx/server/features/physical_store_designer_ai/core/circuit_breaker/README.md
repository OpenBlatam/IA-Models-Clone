# Circuit Breaker Module

Sistema completo de Circuit Breaker para manejo de fallos y resiliencia.

## Estructura

```
circuit_breaker/
├── __init__.py          # Exports principales
├── breaker.py           # Clase CircuitBreaker principal
├── config.py            # Configuración
├── circuit_types.py     # Tipos y enums
├── metrics.py           # Métricas y estadísticas
├── events.py            # Sistema de eventos
├── registry.py          # Registro centralizado
├── groups.py            # Grupos de circuit breakers
├── chain.py             # Cadenas de circuit breakers
├── tracing.py           # Tracing y contexto
└── store.py             # Persistencia de estado
```

## Uso Básico

```python
from core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

# Crear configuración
config = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=Exception
)

# Crear circuit breaker
breaker = CircuitBreaker("my_service", config)

# Usar como decorador
@breaker
async def my_async_function():
    # Tu código aquí
    pass

# O como context manager
async with breaker:
    # Tu código aquí
    pass
```

## Registry

```python
from core.circuit_breaker import circuit_breaker, get_circuit_breaker

# Registrar circuit breaker
@circuit_breaker("my_service", failure_threshold=5)
async def my_function():
    pass

# Obtener circuit breaker registrado
breaker = get_circuit_breaker("my_service")
```

## Grupos y Cadenas

```python
from core.circuit_breaker import CircuitBreakerGroup, CircuitBreakerChain

# Crear grupo
group = CircuitBreakerGroup("api_group")
group.add_breaker("service1", breaker1)
group.add_breaker("service2", breaker2)

# Crear cadena
chain = CircuitBreakerChain([breaker1, breaker2, breaker3])
```

## Características

- ✅ Estados: CLOSED, OPEN, HALF_OPEN
- ✅ Retry automático con backoff
- ✅ Fallback functions
- ✅ Context manager support
- ✅ Eventos y callbacks
- ✅ Métricas y estadísticas
- ✅ Registry centralizado
- ✅ Grupos y cadenas
- ✅ Tracing y contexto
- ✅ Persistencia de estado

## Documentación Completa

Ver `CIRCUIT_BREAKER_INDEX.md` para documentación detallada.


