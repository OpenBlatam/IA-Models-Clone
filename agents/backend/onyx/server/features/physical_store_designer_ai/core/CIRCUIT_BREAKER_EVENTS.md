# Circuit Breaker - Sistema de Eventos de Dominio

## ✅ Implementación Completada

Se ha implementado un sistema completo de eventos de dominio para el Circuit Breaker, permitiendo integración con sistemas de monitoreo, logging, y event buses.

## 🎯 Características

### 1. Tipos de Eventos

El sistema emite los siguientes tipos de eventos:

- **STATE_CHANGED**: Cuando el estado del circuit breaker cambia
- **CIRCUIT_OPENED**: Cuando el circuit se abre
- **CIRCUIT_CLOSED**: Cuando el circuit se cierra
- **CIRCUIT_HALF_OPENED**: Cuando el circuit entra en estado half-open
- **FAILURE_RECORDED**: Cuando se registra un fallo
- **SUCCESS_RECORDED**: Cuando se registra un éxito
- **REQUEST_REJECTED**: Cuando un request es rechazado
- **RETRY_ATTEMPTED**: Cuando se intenta un retry
- **FALLBACK_USED**: Cuando se usa la función de fallback
- **TIMEOUT_OCCURRED**: Cuando ocurre un timeout
- **THRESHOLD_EXCEEDED**: Cuando se excede el threshold de fallos
- **METRICS_UPDATED**: Cuando se actualizan las métricas (cada 10 requests)

### 2. Estructura de Eventos

```python
@dataclass
class CircuitBreakerEvent:
    event_type: CircuitBreakerEventType
    circuit_name: str
    timestamp: datetime
    old_state: Optional[CircuitState] = None
    new_state: Optional[CircuitState] = None
    metadata: Dict[str, Any] = {}
```

## 📚 Ejemplos de Uso

### Ejemplo 1: Registrar Event Handler

```python
from .core.circuit_breaker import CircuitBreaker, CircuitBreakerEvent

async def handle_event(event: CircuitBreakerEvent):
    """Handle circuit breaker events"""
    print(f"Event: {event.event_type.value} for {event.circuit_name}")
    print(f"Metadata: {event.metadata}")
    
    # Integrar con sistema de monitoreo
    if event.event_type == CircuitBreakerEventType.CIRCUIT_OPENED:
        send_alert(f"Circuit {event.circuit_name} opened!")

# Registrar handler
breaker = CircuitBreaker(name="api_service")
breaker.on_event(handle_event)
```

### Ejemplo 2: Integración con Event Bus

```python
from .core.circuit_breaker import CircuitBreaker, CircuitBreakerEvent

def publish_to_event_bus(event: CircuitBreakerEvent):
    """Publish event to event bus"""
    event_bus.publish("circuit_breaker.events", event.to_dict())

breaker = CircuitBreaker(name="api_service")
breaker.on_event(publish_to_event_bus)
```

### Ejemplo 3: Logging Estructurado

```python
import json
from .core.circuit_breaker import CircuitBreaker, CircuitBreakerEvent

def log_event(event: CircuitBreakerEvent):
    """Log event in structured format"""
    logger.info(
        "circuit_breaker_event",
        extra={
            "event_type": event.event_type.value,
            "circuit_name": event.circuit_name,
            "timestamp": event.timestamp.isoformat(),
            "metadata": event.metadata
        }
    )

breaker = CircuitBreaker(name="api_service")
breaker.on_event(log_event)
```

### Ejemplo 4: Métricas en Tiempo Real

```python
from .core.circuit_breaker import CircuitBreaker, CircuitBreakerEvent, CircuitBreakerEventType

def update_metrics(event: CircuitBreakerEvent):
    """Update real-time metrics"""
    if event.event_type == CircuitBreakerEventType.CIRCUIT_OPENED:
        metrics.increment("circuit_breaker.opened", tags={"name": event.circuit_name})
    elif event.event_type == CircuitBreakerEventType.FAILURE_RECORDED:
        metrics.increment("circuit_breaker.failures", tags={"name": event.circuit_name})
    elif event.event_type == CircuitBreakerEventType.SUCCESS_RECORDED:
        metrics.increment("circuit_breaker.successes", tags={"name": event.circuit_name})

breaker = CircuitBreaker(name="api_service")
breaker.on_event(update_metrics)
```

### Ejemplo 5: Obtener Historial de Eventos

```python
# Obtener todos los eventos recientes
events = breaker.get_event_history(limit=50)

# Obtener solo eventos de un tipo específico
opened_events = breaker.get_events_by_type(
    CircuitBreakerEventType.CIRCUIT_OPENED,
    limit=10
)

# Procesar eventos
for event in events:
    print(f"{event.timestamp}: {event.event_type.value}")
    print(f"  Metadata: {event.metadata}")
```

### Ejemplo 6: Análisis de Eventos

```python
# Analizar patrones de eventos
def analyze_events(breaker: CircuitBreaker):
    """Analyze circuit breaker events"""
    events = breaker.get_event_history()
    
    # Contar eventos por tipo
    event_counts = {}
    for event in events:
        event_type = event.event_type.value
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    
    # Encontrar eventos recientes de apertura
    recent_openings = breaker.get_events_by_type(
        CircuitBreakerEventType.CIRCUIT_OPENED,
        limit=5
    )
    
    return {
        "event_counts": event_counts,
        "recent_openings": [e.to_dict() for e in recent_openings]
    }
```

## 🔄 Integración con Sistemas Externos

### Prometheus

```python
from prometheus_client import Counter, Histogram

circuit_opened_counter = Counter(
    'circuit_breaker_opened_total',
    'Total circuit breaker openings',
    ['circuit_name']
)

def prometheus_handler(event: CircuitBreakerEvent):
    if event.event_type == CircuitBreakerEventType.CIRCUIT_OPENED:
        circuit_opened_counter.labels(
            circuit_name=event.circuit_name
        ).inc()

breaker.on_event(prometheus_handler)
```

### StatsD

```python
from statsd import StatsClient

statsd = StatsClient()

def statsd_handler(event: CircuitBreakerEvent):
    """Send events to StatsD"""
    metric_name = f"circuit_breaker.{event.event_type.value}"
    statsd.increment(metric_name, tags={
        "circuit_name": event.circuit_name
    })

breaker.on_event(statsd_handler)
```

### OpenTelemetry

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

def otel_handler(event: CircuitBreakerEvent):
    """Send events to OpenTelemetry"""
    with tracer.start_as_current_span("circuit_breaker.event") as span:
        span.set_attribute("event.type", event.event_type.value)
        span.set_attribute("circuit.name", event.circuit_name)
        span.set_attribute("event.timestamp", event.timestamp.isoformat())
        for key, value in event.metadata.items():
            span.set_attribute(f"event.{key}", str(value))

breaker.on_event(otel_handler)
```

## 📊 Metadata de Eventos

Cada tipo de evento incluye metadata relevante:

### STATE_CHANGED
```python
{
    "old_state": "closed",
    "new_state": "open"
}
```

### CIRCUIT_OPENED
```python
{
    "old_state": "closed",
    "new_state": "open",
    "failure_count": 5,
    "failure_threshold": 5
}
```

### FAILURE_RECORDED
```python
{
    "error_type": "ConnectionError",
    "error_message": "Connection refused",
    "duration": 0.125,
    "failure_count": 3
}
```

### RETRY_ATTEMPTED
```python
{
    "attempt": 2,
    "max_retries": 3,
    "delay": 2.0,
    "error_type": "TimeoutError"
}
```

## 🎯 Beneficios

1. **Observabilidad**: Eventos estructurados para monitoreo
2. **Integración**: Fácil integración con sistemas externos
3. **Debugging**: Historial de eventos para análisis
4. **Alertas**: Eventos pueden disparar alertas automáticas
5. **Analytics**: Análisis de patrones de comportamiento

## ✅ Estado

Sistema de eventos completamente implementado y listo para usar.




