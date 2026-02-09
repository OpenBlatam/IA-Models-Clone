# Guía de Circuit Breaker - Arquitectura Mejorada

## 📋 Visión General

Implementación mejorada del patrón Circuit Breaker siguiendo la nueva arquitectura con Domain-Driven Design. Protege servicios contra fallos en cascada y proporciona recuperación automática.

## 🏗️ Arquitectura

El Circuit Breaker está implementado como una **entidad de dominio** con:

- **Domain Layer**: `CircuitBreaker` como entidad, `CircuitBreakerConfig` como value object
- **Domain Events**: Eventos para cambios de estado
- **Application Layer**: `CircuitBreakerManager` como servicio de aplicación
- **Infrastructure**: Integración con sistema de errores y DI

## 🚀 Uso Básico

### Usando el Manager

```python
from core.architecture.circuit_breaker import (
    CircuitBreakerManager,
    CircuitBreakerConfig,
    get_circuit_breaker_manager
)

# Obtener manager
manager = get_circuit_breaker_manager()

# Crear o obtener circuit breaker
config = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60.0,
    success_threshold=2
)

circuit = await manager.get_or_create("robot_service", config)

# Usar circuit breaker
try:
    result = await circuit.call(my_service_function, arg1, arg2)
except InfrastructureError as e:
    print(f"Circuit breaker rechazó la llamada: {e}")
```

### Usando el Decorator

```python
from core.architecture.circuit_breaker import circuit_breaker, CircuitBreakerConfig

@circuit_breaker(
    name="robot_movement",
    config=CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=60.0
    )
)
async def move_robot(robot_id: str, target: Position):
    # Tu código aquí
    pass
```

## ⚙️ Configuración

### CircuitBreakerConfig

```python
config = CircuitBreakerConfig(
    failure_threshold=5,          # Fallos antes de abrir
    recovery_timeout=60.0,         # Segundos antes de intentar reset
    success_threshold=2,           # Éxitos para cerrar desde half-open
    monitoring_window=300.0,       # Ventana de tiempo para contar fallos
    call_timeout=30.0,            # Timeout por llamada (opcional)
    enable_adaptive_timeout=True, # Timeout adaptativo
    min_timeout=10.0,             # Timeout mínimo
    max_timeout=300.0,            # Timeout máximo
    timeout_multiplier=2.0        # Multiplicador para timeout adaptativo
)
```

## 📊 Estados del Circuit Breaker

### CLOSED (Cerrado)
- Estado normal de operación
- Todas las llamadas pasan
- Se monitorean fallos

### OPEN (Abierto)
- Servicio está fallando
- Todas las llamadas se rechazan inmediatamente
- Espera `recovery_timeout` antes de intentar reset

### HALF_OPEN (Semi-abierto)
- Estado de prueba
- Permite algunas llamadas para verificar recuperación
- Si `success_threshold` éxitos → CLOSED
- Si cualquier fallo → OPEN

## 🔄 Flujo de Estados

```
CLOSED → (fallos >= threshold) → OPEN
OPEN → (timeout transcurrido) → HALF_OPEN
HALF_OPEN → (éxitos >= threshold) → CLOSED
HALF_OPEN → (cualquier fallo) → OPEN
```

## 📈 Métricas

```python
circuit = await manager.get("robot_service")
metrics = circuit.metrics

print(f"Total requests: {metrics.total_requests}")
print(f"Success rate: {metrics.success_rate}")
print(f"Failure rate: {metrics.failure_rate}")
print(f"Rejected requests: {metrics.rejected_requests}")
print(f"State changes: {metrics.state_changes}")
```

## 🎯 Integración con Use Cases

```python
from core.architecture.circuit_breaker import get_circuit_breaker_manager
from core.architecture.application_layer import MoveRobotUseCase

class MoveRobotUseCaseWithCircuitBreaker:
    def __init__(self, base_use_case: MoveRobotUseCase):
        self.base_use_case = base_use_case
        self.circuit_manager = get_circuit_breaker_manager()
    
    async def execute(self, command):
        circuit = await self.circuit_manager.get_or_create("move_robot")
        return await circuit.call(self.base_use_case.execute, command)
```

## 🔌 Integración con Dependency Injection

El Circuit Breaker Manager está integrado con el sistema de DI:

```python
from core.architecture.di_setup import get_di_setup
from core.architecture.circuit_breaker import CircuitBreakerManager

di_setup = get_di_setup()
manager = await di_setup.resolve(CircuitBreakerManager)
```

## 📡 Domain Events

El Circuit Breaker emite eventos cuando cambia de estado:

```python
from core.architecture.circuit_breaker import (
    CircuitBreakerOpenedEvent,
    CircuitBreakerClosedEvent,
    CircuitBreakerHalfOpenEvent
)

# Escuchar eventos
circuit = await manager.get("robot_service")
events = circuit.get_domain_events()

for event in events:
    if isinstance(event, CircuitBreakerOpenedEvent):
        logger.warning(f"Circuit abierto: {event.circuit_name}")
    elif isinstance(event, CircuitBreakerClosedEvent):
        logger.info(f"Circuit cerrado: {event.circuit_name}")
```

## 🧪 Testing

```python
import pytest
from core.architecture.circuit_breaker import (
    CircuitBreakerManager,
    CircuitBreakerConfig
)

@pytest.fixture
async def circuit_manager():
    return CircuitBreakerManager()

@pytest.mark.asyncio
async def test_circuit_breaker_opens_on_failures(circuit_manager):
    config = CircuitBreakerConfig(failure_threshold=2)
    circuit = await circuit_manager.get_or_create("test", config)
    
    # Simular fallos
    async def failing_function():
        raise Exception("Error")
    
    # Primer fallo
    try:
        await circuit.call(failing_function)
    except:
        pass
    
    # Segundo fallo - debería abrir
    try:
        await circuit.call(failing_function)
    except:
        pass
    
    assert circuit.state == CircuitState.OPEN
```

## 🎨 Ejemplos de Uso

### Proteger Llamada a Repositorio

```python
from core.architecture.circuit_breaker import get_circuit_breaker_manager
from core.architecture.application_layer import IRobotRepository

class ProtectedRobotRepository:
    def __init__(self, base_repo: IRobotRepository):
        self.base_repo = base_repo
        self.circuit_manager = get_circuit_breaker_manager()
    
    async def find_by_id(self, robot_id: str):
        circuit = await self.circuit_manager.get_or_create("robot_repository")
        return await circuit.call(self.base_repo.find_by_id, robot_id)
```

### Proteger Llamada HTTP

```python
import httpx
from core.architecture.circuit_breaker import circuit_breaker, CircuitBreakerConfig

@circuit_breaker(
    name="external_api",
    config=CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=30.0,
        call_timeout=10.0
    )
)
async def call_external_api(url: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()
```

## 🔧 Configuración Avanzada

### Timeout Adaptativo

El timeout adaptativo aumenta exponencialmente con cada fallo:

```python
config = CircuitBreakerConfig(
    enable_adaptive_timeout=True,
    min_timeout=10.0,
    max_timeout=300.0,
    timeout_multiplier=2.0
)

# Primer fallo: timeout = 10s
# Segundo fallo: timeout = 20s
# Tercer fallo: timeout = 40s
# ...
# Máximo: timeout = 300s
```

### Sliding Window

El sliding window solo cuenta fallos dentro de la ventana de tiempo:

```python
config = CircuitBreakerConfig(
    failure_threshold=5,
    monitoring_window=300.0  # Solo cuenta fallos de últimos 5 minutos
)
```

## 📝 Mejores Prácticas

1. **Usar nombres descriptivos**: `"robot_movement"` en lugar de `"cb1"`
2. **Configurar thresholds apropiados**: No muy bajos (falsos positivos) ni muy altos (fallos en cascada)
3. **Monitorear métricas**: Revisar regularmente el estado de los circuit breakers
4. **Usar timeouts**: Configurar `call_timeout` para evitar llamadas colgadas
5. **Reset manual cuando sea necesario**: Después de deploy o cambios importantes

## 🚀 Próximos Pasos

1. Integrar con sistema de métricas
2. Dashboard para monitoreo de circuit breakers
3. Alertas cuando circuit breakers se abren
4. Integración con tracing distribuido

---

**Fecha**: 2025-01-27
**Versión**: 1.0.0




