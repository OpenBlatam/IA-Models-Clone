# Mejoras del Circuit Breaker

## 📋 Resumen de Mejoras

Se han implementado mejoras significativas al Circuit Breaker basándose en mejores prácticas y características avanzadas.

## ✨ Nuevas Características

### 1. State Callbacks

Permite registrar callbacks que se ejecutan cuando el circuit breaker cambia de estado:

```python
from core.architecture.circuit_breaker import CircuitBreaker, CircuitState

async def on_circuit_opened(circuit, new_state, old_state):
    print(f"Circuit {circuit.name} opened!")
    # Enviar alerta, actualizar métricas, etc.

circuit = CircuitBreaker(name="api_service", config=config)
circuit.register_callback(CircuitState.OPEN, on_circuit_opened)
```

### 2. Expected Exception Filtering

Filtra qué excepciones cuentan como fallos:

```python
from core.architecture.circuit_breaker import CircuitBreakerConfig

config = CircuitBreakerConfig(
    failure_threshold=5,
    expected_exception=ConnectionError  # Solo ConnectionError cuenta como fallo
)

circuit = CircuitBreaker(name="api", config=config)

# ValueError no cuenta como fallo
try:
    await circuit.call(lambda: raise ValueError("Bad input"))
except ValueError:
    pass  # Circuit breaker no cuenta esto como fallo
```

### 3. Remaining Timeout

Obtener tiempo restante antes de intentar reset:

```python
info = circuit.get_state_info()
print(f"Recovery in {info['remaining_timeout']:.1f}s")
```

### 4. Backward Compatibility

Soporte para parámetros individuales además de config:

```python
# Nueva forma (recomendada)
circuit = CircuitBreaker(
    name="api",
    config=CircuitBreakerConfig(failure_threshold=5)
)

# Forma antigua (compatible)
circuit = CircuitBreaker(
    name="api",
    failure_threshold=5,
    recovery_timeout=60.0
)
```

### 5. Mejor Manejo de Timeouts Adaptativos

El timeout adaptativo ahora se actualiza correctamente cuando el circuit se abre:

```python
config = CircuitBreakerConfig(
    enable_adaptive_timeout=True,
    min_timeout=10.0,
    max_timeout=300.0,
    timeout_multiplier=2.0
)

# Primer fallo: timeout = 60s (default)
# Segundo fallo: timeout = 120s (60 * 2)
# Tercer fallo: timeout = 240s (120 * 2)
# Máximo: timeout = 300s
```

### 6. State Info Method

Método para obtener información completa del estado:

```python
info = circuit.get_state_info()
# {
#     "name": "api_service",
#     "state": "open",
#     "metrics": {...},
#     "remaining_timeout": 45.2,
#     "current_timeout": 120.0,
#     "config": {...}
# }
```

## 🔧 Mejoras de Implementación

### Separación de Lógica

- `_should_allow_call()`: Verifica si se permite la llamada
- `_on_success()`: Maneja éxito
- `_on_failure()`: Maneja fallos
- `_get_remaining_timeout()`: Calcula tiempo restante

### Mejor Performance

- Verificación de estado fuera del lock cuando es posible
- Callbacks async ejecutados como tasks
- Limpieza eficiente de fallos antiguos

### Mejor Logging

- Logging estructurado con contexto
- Información detallada en errores
- Warnings apropiados para cambios de estado

## 📊 Comparación Antes/Después

### Antes
- ❌ No había callbacks
- ❌ Todas las excepciones contaban como fallos
- ❌ No había forma de obtener tiempo restante
- ❌ Solo soportaba config object

### Después
- ✅ Callbacks para cambios de estado
- ✅ Filtrado de excepciones esperadas
- ✅ Método para tiempo restante
- ✅ Backward compatibility completa
- ✅ Mejor separación de responsabilidades
- ✅ Mejor performance

## 🚀 Ejemplos de Uso

### Con Callbacks

```python
from core.architecture.circuit_breaker import CircuitBreaker, CircuitState

async def alert_on_open(circuit, new_state, old_state):
    if new_state == CircuitState.OPEN:
        await send_alert(f"Circuit {circuit.name} opened!")

circuit = CircuitBreaker(name="critical_service", config=config)
circuit.register_callback(CircuitState.OPEN, alert_on_open)
```

### Con Expected Exception

```python
from core.architecture.circuit_breaker import CircuitBreakerConfig

# Solo errores de red cuentan como fallos
config = CircuitBreakerConfig(
    expected_exception=(ConnectionError, TimeoutError)
)

circuit = CircuitBreaker(name="api", config=config)
```

### Monitoreo

```python
# Obtener estado completo
info = circuit.get_state_info()

if info["state"] == "open":
    print(f"Circuit will retry in {info['remaining_timeout']:.1f}s")
    print(f"Current timeout: {info['current_timeout']}s")
    print(f"Failures: {info['metrics']['current_failure_count']}")
```

## 📝 Notas

- Los callbacks async se ejecutan como tasks (no bloquean)
- Los callbacks sync se ejecutan directamente
- Si un callback falla, no afecta el funcionamiento del circuit breaker
- El expected_exception por defecto es `Exception` (todas las excepciones cuentan)

---

**Fecha**: 2025-01-27
**Versión**: 2.0.0




