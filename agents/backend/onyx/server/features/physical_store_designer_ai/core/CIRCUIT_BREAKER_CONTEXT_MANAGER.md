# Circuit Breaker - Context Manager Mejorado

## ✅ Mejora #3: Context Manager Async - Implementación Mejorada

Se ha mejorado el soporte de context manager async para el Circuit Breaker, permitiendo un uso más limpio y seguro en código async.

## 🎯 Características

### 1. Soporte Async Context Manager

El Circuit Breaker ahora implementa `__aenter__` y `__aexit__` para uso con `async with`.

### 2. Eventos Automáticos

- Emite eventos al entrar y salir del context
- Registra si ocurrieron excepciones dentro del context

### 3. Auto-Reset Opcional

- Configuración `auto_reset_on_exit` para resetear automáticamente al salir
- Útil para testing y circuit breakers temporales

## 📚 Ejemplos de Uso

### Ejemplo 1: Uso Básico

```python
from .core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

breaker = CircuitBreaker(
    config=CircuitBreakerConfig(),
    name="api_service"
)

# Usar con async with
async with breaker:
    result = await breaker.call(api_function, param1, param2)
    # Circuit breaker está activo dentro del context
    # Se emiten eventos automáticamente
```

### Ejemplo 2: Manejo de Excepciones

```python
async with breaker:
    try:
        result = await breaker.call(api_function, ...)
    except ServiceError as e:
        # El context manager registra la excepción en eventos
        logger.error(f"Service error: {e}")
        raise
    # Si ocurre una excepción, se propaga normalmente
```

### Ejemplo 3: Auto-Reset para Testing

```python
# Configurar para auto-reset en testing
config = CircuitBreakerConfig(
    auto_reset_on_exit=True  # Reset automático al salir del context
)

breaker = CircuitBreaker(config=config, name="test_service")

async with breaker:
    # Hacer pruebas
    result1 = await breaker.call(test_function, ...)
    result2 = await breaker.call(test_function, ...)
    # Al salir, el circuit breaker se resetea automáticamente
```

### Ejemplo 4: Múltiples Operaciones en Context

```python
async with breaker:
    # Múltiples operaciones protegidas
    result1 = await breaker.call(operation1, ...)
    result2 = await breaker.call(operation2, ...)
    result3 = await breaker.call(operation3, ...)
    
    # Todas las operaciones comparten el mismo context
    # Eventos se emiten para cada operación
```

### Ejemplo 5: Integración con Event Handlers

```python
async def context_event_handler(event: CircuitBreakerEvent):
    """Handle context-related events"""
    if event.metadata.get("action") == "context_entered":
        logger.info(f"Circuit breaker {event.circuit_name} context started")
    elif event.metadata.get("action") == "context_exited":
        exception_occurred = event.metadata.get("exception_occurred", False)
        if exception_occurred:
            logger.warning(
                f"Circuit breaker {event.circuit_name} context exited with exception"
            )

breaker = CircuitBreaker(name="api_service")
breaker.on_event(context_event_handler)

async with breaker:
    # Los eventos de context se capturan por el handler
    result = await breaker.call(api_function, ...)
```

### Ejemplo 6: Nested Contexts

```python
# Contexts anidados son soportados
async with outer_breaker:
    async with inner_breaker:
        # Ambos circuit breakers están activos
        result = await inner_breaker.call(inner_function, ...)
        result = await outer_breaker.call(outer_function, ...)
```

### Ejemplo 7: Resource Management

```python
# El context manager puede usarse para gestión de recursos
async def process_with_circuit_breaker():
    breaker = CircuitBreaker(name="external_api")
    
    async with breaker:
        # Garantiza que el circuit breaker esté en estado conocido
        if not breaker.is_ready():
            raise RuntimeError("Circuit breaker not ready")
        
        # Operaciones protegidas
        data = await breaker.call(fetch_data, ...)
        processed = await breaker.call(process_data, data, ...)
        
        return processed
    # Cleanup automático si es necesario
```

## 🔍 Eventos Emitidos

### Context Entered

```python
{
    "event_type": "metrics_updated",
    "action": "context_entered",
    "state": "closed",
    "circuit_name": "api_service"
}
```

### Context Exited

```python
{
    "event_type": "metrics_updated",
    "action": "context_exited",
    "state": "closed",
    "exception_occurred": false,
    "exception_type": null,
    "circuit_name": "api_service"
}
```

### Context Exited with Exception

```python
{
    "event_type": "metrics_updated",
    "action": "context_exited",
    "state": "open",
    "exception_occurred": true,
    "exception_type": "ServiceError",
    "circuit_name": "api_service"
}
```

## ⚙️ Configuración

### Auto-Reset on Exit

```python
config = CircuitBreakerConfig(
    auto_reset_on_exit=True  # Reset automático al salir
)

breaker = CircuitBreaker(config=config, name="service")
```

**Cuándo usar:**
- Testing: Reset automático entre tests
- Circuit breakers temporales: Limpieza automática
- Operaciones aisladas: Estado limpio después de operación

**Cuándo NO usar:**
- Producción normal: Mantener estado entre operaciones
- Circuit breakers compartidos: No resetear estado compartido

## 🎯 Beneficios

1. **Código más limpio**: Sintaxis `async with` más legible
2. **Gestión automática**: Eventos y cleanup automáticos
3. **Debugging mejorado**: Eventos de context para análisis
4. **Testing facilitado**: Auto-reset para tests aislados
5. **Resource safety**: Garantiza estado conocido al entrar

## 📊 Comparación: Con vs Sin Context Manager

### Sin Context Manager

```python
breaker = CircuitBreaker(name="api_service")
result = await breaker.call(api_function, ...)
# No hay tracking de context
# No hay eventos de lifecycle
```

### Con Context Manager

```python
breaker = CircuitBreaker(name="api_service")
async with breaker:
    result = await breaker.call(api_function, ...)
    # Eventos automáticos
    # Tracking de lifecycle
    # Posible auto-reset
```

## ✅ Estado

- ✅ Context manager implementado
- ✅ Eventos automáticos
- ✅ Auto-reset opcional
- ✅ Documentación completa
- ✅ Listo para usar




