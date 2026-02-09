# Circuit Breaker - Groups, Chain y Características Avanzadas

## ✅ Mejoras Adicionales Implementadas

Se han implementado características avanzadas para gestión de múltiples circuit breakers y integración con sistemas de observabilidad.

## 🎯 Nuevas Características

### 1. ✅ Circuit Breaker Groups

Agrupa múltiples circuit breakers con configuración compartida.

**Clase:** `CircuitBreakerGroup`

**Ejemplo de uso:**
```python
from .core.circuit_breaker import CircuitBreakerGroup, CircuitBreakerConfig

# Crear grupo con configuración compartida
config = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60.0,
    retry_enabled=True
)

group = CircuitBreakerGroup("api_services", config)

# Obtener o crear breakers en el grupo
user_service = group.get_or_create("user_service")
order_service = group.get_or_create("order_service")
payment_service = group.get_or_create("payment_service")

# Cada breaker puede tener overrides
critical_service = group.get_or_create(
    "critical_service",
    failure_threshold=3,  # Override: más sensible
    recovery_timeout=120.0  # Override: más tiempo
)

# Health check del grupo
group_health = await group.get_health_status()
# {
#     "group_name": "api_services",
#     "total_breakers": 4,
#     "healthy_count": 3,
#     "degraded_count": 1,
#     "critical_count": 0,
#     "open_count": 0,
#     "breakers": {...}
# }
```

### 2. ✅ Circuit Breaker Chain

Encadena múltiples circuit breakers para operaciones secuenciales.

**Clase:** `CircuitBreakerChain`

**Ejemplo de uso:**
```python
from .core.circuit_breaker import CircuitBreakerChain, CircuitBreaker

# Crear breakers individuales
auth_breaker = CircuitBreaker(name="auth_service")
db_breaker = CircuitBreaker(name="database")
cache_breaker = CircuitBreaker(name="cache")

# Crear chain
chain = CircuitBreakerChain(auth_breaker, db_breaker, cache_breaker)

# Ejecutar funciones en secuencia
funcs = [
    authenticate_user,
    fetch_user_data,
    cache_user_data
]

try:
    results = await chain.call(funcs, user_id="123")
    # results = [auth_result, db_result, cache_result]
except ServiceError as e:
    # Si cualquier breaker falla, toda la chain falla
    logger.error(f"Chain failed at {e.service}")
```

### 3. ✅ OpenTelemetry Integration

Integración con OpenTelemetry para distributed tracing.

**Funciones:**
- `get_trace_context()`: Obtiene contexto para tracing
- `add_tracing_to_circuit_breaker()`: Agrega tracing automático

**Ejemplo de uso:**
```python
from .core.circuit_breaker import (
    CircuitBreaker,
    get_trace_context,
    add_tracing_to_circuit_breaker
)

breaker = CircuitBreaker(name="api_service")

# Agregar tracing automático
add_tracing_to_circuit_breaker(breaker)

# Obtener contexto manualmente
trace_context = get_trace_context(breaker)
# {
#     "circuit_breaker.name": "api_service",
#     "circuit_breaker.state": "closed",
#     "circuit_breaker.failure_count": 0,
#     ...
# }
```

### 4. ✅ State Persistence

Sistema para persistir estado de circuit breakers.

**Clases:**
- `CircuitBreakerStateStore`: Interface para persistencia
- `InMemoryStateStore`: Implementación en memoria (testing)

**Ejemplo de uso:**
```python
from .core.circuit_breaker import (
    create_circuit_breaker_with_persistence,
    InMemoryStateStore,
    CircuitBreakerConfig
)

# Crear state store
state_store = InMemoryStateStore()

# Crear breaker con persistencia
breaker = create_circuit_breaker_with_persistence(
    name="api_service",
    config=CircuitBreakerConfig(),
    state_store=state_store
)

# El estado se guarda automáticamente en cambios
# Y se carga al inicializar
```

## 📚 Ejemplos Completos

### Ejemplo 1: Grupo de Servicios API

```python
# Configurar grupo para servicios relacionados
api_group = CircuitBreakerGroup("external_apis", CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60.0,
    retry_enabled=True,
    max_retries=3
))

# Crear breakers para diferentes APIs
stripe_breaker = api_group.get_or_create("stripe")
twilio_breaker = api_group.get_or_create("twilio")
sendgrid_breaker = api_group.get_or_create("sendgrid")

# Health check del grupo completo
async def check_api_group_health():
    health = await api_group.get_health_status()
    
    if health["open_count"] > 0:
        logger.warning(f"{health['open_count']} APIs are down")
    
    if health["critical_count"] > 0:
        logger.error(f"{health['critical_count']} APIs are critical")
    
    return health
```

### Ejemplo 2: Chain para Operación Compleja

```python
# Operación que requiere múltiples servicios en secuencia
async def process_order(user_id: str, order_data: dict):
    # Crear chain
    chain = CircuitBreakerChain(
        get_circuit_breaker("auth"),
        get_circuit_breaker("inventory"),
        get_circuit_breaker("payment"),
        get_circuit_breaker("notification")
    )
    
    # Funciones en secuencia
    funcs = [
        lambda uid, od: validate_user(uid),
        lambda uid, od: check_inventory(od["items"]),
        lambda uid, od: process_payment(od["payment"]),
        lambda uid, od: send_confirmation(uid, od)
    ]
    
    try:
        results = await chain.call(funcs, user_id, order_data)
        return {"status": "success", "results": results}
    except ServiceError as e:
        # Rollback si es necesario
        await rollback_order(user_id, order_data)
        raise
```

### Ejemplo 3: Integración Completa con OpenTelemetry

```python
from opentelemetry import trace
from .core.circuit_breaker import (
    CircuitBreaker,
    add_tracing_to_circuit_breaker,
    get_trace_context
)

tracer = trace.get_tracer(__name__)

# Crear breaker con tracing
breaker = CircuitBreaker(name="external_api")
add_tracing_to_circuit_breaker(breaker)

# Usar en operación con span
async def call_external_api():
    with tracer.start_as_current_span("external_api_call") as span:
        # Agregar contexto del circuit breaker
        trace_context = get_trace_context(breaker)
        for key, value in trace_context.items():
            span.set_attribute(key, value)
        
        # Llamar a través del breaker
        result = await breaker.call(external_api_function, ...)
        
        # Agregar resultado al span
        span.set_attribute("result.success", True)
        
        return result
```

### Ejemplo 4: Persistencia con Redis

```python
import json
import redis
from .core.circuit_breaker import CircuitBreakerStateStore, CircuitBreaker

class RedisStateStore(CircuitBreakerStateStore):
    """Redis implementation of state store"""
    
    def __init__(self, redis_client: redis.Redis, key_prefix: str = "cb:"):
        self.redis = redis_client
        self.prefix = key_prefix
    
    async def save_state(self, breaker: CircuitBreaker):
        """Save state to Redis"""
        key = f"{self.prefix}{breaker.name}"
        state = breaker.get_state()
        # Convert to JSON-serializable
        state_json = json.dumps(state, default=str)
        self.redis.setex(key, 3600, state_json)  # 1 hour TTL
    
    async def load_state(self, breaker_name: str):
        """Load state from Redis"""
        key = f"{self.prefix}{breaker_name}"
        state_json = self.redis.get(key)
        if state_json:
            return json.loads(state_json)
        return None

# Usar
redis_client = redis.Redis()
state_store = RedisStateStore(redis_client)

breaker = create_circuit_breaker_with_persistence(
    name="api_service",
    state_store=state_store
)
```

### Ejemplo 5: Dashboard de Health del Grupo

```python
@router.get("/health/group/{group_name}")
async def get_group_health(group_name: str):
    """Get health status of circuit breaker group"""
    # En producción, esto vendría de un registry
    group = get_group_by_name(group_name)
    
    health = await group.get_health_status()
    
    # Calcular score general
    total = health["total_breakers"]
    healthy_ratio = health["healthy_count"] / total if total > 0 else 0
    
    return {
        "group": group_name,
        "overall_health": "healthy" if healthy_ratio > 0.8 else "degraded",
        "healthy_ratio": healthy_ratio,
        "summary": {
            "healthy": health["healthy_count"],
            "degraded": health["degraded_count"],
            "critical": health["critical_count"],
            "open": health["open_count"]
        },
        "breakers": health["breakers"]
    }
```

## 🎯 Beneficios

1. **Gestión Centralizada**: Groups permiten gestionar múltiples breakers juntos
2. **Operaciones Secuenciales**: Chain facilita operaciones multi-paso
3. **Observabilidad**: OpenTelemetry integration para distributed tracing
4. **Persistencia**: Estado puede persistirse entre reinicios
5. **Escalabilidad**: Fácil agregar nuevos breakers a grupos

## ✅ Estado

- ✅ Circuit Breaker Groups implementados
- ✅ Circuit Breaker Chain implementado
- ✅ OpenTelemetry integration implementada
- ✅ State persistence framework implementado
- ✅ Documentación completa
- ✅ Listo para usar

Todas las características avanzadas están implementadas y listas para usar.




