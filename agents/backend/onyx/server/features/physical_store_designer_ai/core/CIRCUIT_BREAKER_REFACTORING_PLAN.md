# Circuit Breaker - Plan de Refactorización Completa

## 📋 Resumen

Plan completo para refactorizar el Circuit Breaker (1725 líneas) en una estructura modular mantenible.

## 🎯 Objetivos

1. **Modularidad**: Dividir en módulos lógicos y cohesivos
2. **Mantenibilidad**: Código más fácil de entender y modificar
3. **Testabilidad**: Módulos más pequeños son más fáciles de testear
4. **Compatibilidad**: 100% compatible con código existente

## 🏗️ Estructura Propuesta

```
circuit_breaker/
├── __init__.py              # Exportaciones principales (compatibilidad)
├── circuit_types.py          # ✅ Enums (CircuitState, CircuitBreakerEventType)
├── config.py                 # ✅ CircuitBreakerConfig
├── metrics.py                # ✅ CircuitBreakerMetrics
├── events.py                 # ✅ CircuitBreakerEvent, EventEmitter
├── breaker.py                # CircuitBreaker class principal
├── state_manager.py          # Gestión de estados y transiciones
├── call_executor.py          # Ejecución de llamadas (retry, timeout, etc.)
├── health_checker.py         # Health check methods
├── decorator.py              # Decorador @circuit_breaker
├── registry.py               # Registry global y funciones
├── groups.py                 # CircuitBreakerGroup
├── chain.py                  # CircuitBreakerChain
├── persistence.py            # State persistence framework
└── observability.py          # OpenTelemetry, exporters
```

## 📊 División de Responsabilidades

### Módulos Base (✅ Creados)

1. **circuit_types.py** (✅)
   - `CircuitState` enum
   - `CircuitBreakerEventType` enum

2. **config.py** (✅)
   - `CircuitBreakerConfig` dataclass

3. **metrics.py** (✅)
   - `CircuitBreakerMetrics` dataclass
   - Cálculo de estadísticas

4. **events.py** (✅)
   - `CircuitBreakerEvent` dataclass
   - `EventEmitter` class

### Módulos Principales (Pendientes)

5. **breaker.py**
   - Clase `CircuitBreaker` principal
   - Métodos públicos: `call()`, `call_with_fallback()`, `call_bulk()`
   - Métodos de estado: `get_state()`, `reset()`, `force_open()`, `force_close()`
   - Context manager: `__aenter__`, `__aexit__`

6. **state_manager.py**
   - `_should_allow_call()`
   - `_transition_to_open()`, `_transition_to_closed()`, `_transition_to_half_open()`
   - `_update_adaptive_timeout()`
   - `_get_remaining_timeout()`
   - `_cleanup_old_failures()`

7. **call_executor.py**
   - `_call_with_retry()`
   - `_execute_call()`
   - `_do_call()`
   - Manejo de timeouts y excepciones

8. **health_checker.py**
   - `is_healthy()`, `is_ready()`, `is_degraded()`, `is_critical()`
   - `get_health_score()`, `get_health_rating()`
   - `get_health_status()`
   - `_get_health_recommendations()`

9. **decorator.py**
   - Función `circuit_breaker()` decorator
   - Wrappers async y sync

10. **registry.py**
    - `_circuit_breakers` dict
    - `_registry_lock`
    - `get_circuit_breaker()`, `get_circuit_breaker_sync()`
    - `get_all_circuit_breakers()`, `reset_all_circuit_breakers()`

11. **groups.py**
    - `CircuitBreakerGroup` class

12. **chain.py**
    - `CircuitBreakerChain` class

13. **persistence.py**
    - `CircuitBreakerStateStore` interface
    - `InMemoryStateStore` implementation
    - `create_circuit_breaker_with_persistence()`

14. **observability.py**
    - `get_trace_context()`
    - `add_tracing_to_circuit_breaker()`
    - `export_metrics_prometheus()`
    - `export_metrics_statsd()`

## 🔄 Estrategia de Refactorización

### Fase 1: Módulos Base ✅
- [x] circuit_types.py
- [x] config.py
- [x] metrics.py
- [x] events.py

### Fase 2: Extracción de Funcionalidades
- [ ] state_manager.py - Extraer gestión de estados
- [ ] call_executor.py - Extraer ejecución de llamadas
- [ ] health_checker.py - Extraer health checks

### Fase 3: Módulos Auxiliares
- [ ] decorator.py - Extraer decorador
- [ ] registry.py - Extraer registry
- [ ] groups.py - Mover CircuitBreakerGroup
- [ ] chain.py - Mover CircuitBreakerChain
- [ ] persistence.py - Mover persistence
- [ ] observability.py - Mover observability

### Fase 4: Clase Principal
- [ ] breaker.py - Refactorizar CircuitBreaker para usar módulos

### Fase 5: Integración
- [ ] Actualizar __init__.py
- [ ] Actualizar imports en core/__init__.py
- [ ] Verificar compatibilidad
- [ ] Tests

## 📝 Ejemplo de Refactorización

### Antes (Monolítico)
```python
# circuit_breaker.py (1725 líneas)
class CircuitBreaker:
    def __init__(self, ...):
        # 50+ líneas de inicialización
    
    async def call(self, ...):
        # 100+ líneas
    
    async def _on_success(self):
        # 30 líneas
    
    # ... muchos más métodos
```

### Después (Modular)
```python
# breaker.py
from .state_manager import StateManager
from .call_executor import CallExecutor
from .health_checker import HealthChecker
from .events import EventEmitter

class CircuitBreaker:
    def __init__(self, ...):
        self.state_manager = StateManager(...)
        self.call_executor = CallExecutor(...)
        self.health_checker = HealthChecker(...)
        self.event_emitter = EventEmitter(...)
    
    async def call(self, ...):
        return await self.call_executor.execute(...)
```

## ✅ Estado Actual

- ✅ Estructura de directorios creada
- ✅ Módulos base implementados (types, config, metrics, events)
- ✅ Compatibilidad mantenida
- ⏳ Refactorización completa pendiente

## 🎯 Beneficios Esperados

1. **Reducción de Complejidad**: Archivos más pequeños y enfocados
2. **Mejor Testabilidad**: Módulos independientes más fáciles de testear
3. **Reutilización**: Componentes pueden usarse independientemente
4. **Mantenibilidad**: Más fácil encontrar y modificar código específico
5. **Escalabilidad**: Fácil agregar nuevas funcionalidades

## 📚 Notas

- La refactorización se puede hacer gradualmente
- El código existente sigue funcionando durante la transición
- Se puede mantener el archivo original como fallback
- Los tests deben actualizarse para usar los nuevos módulos




