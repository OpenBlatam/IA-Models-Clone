# Sistemas de Observabilidad y Optimización

## Nuevas Clases Implementadas

### 1. **BulkEventSourcing** - Event Sourcing

Sistema de event sourcing para rastrear todos los eventos de operaciones bulk.

#### Características:
- Registro inmutable de eventos
- Event streams por aggregate
- Replay de eventos para reconstruir estado
- Búsqueda por tipo de evento o aggregate

#### Uso:

```python
from core.bulk_operations import BulkEventSourcing

event_sourcing = BulkEventSourcing()

# Registrar evento
event_sourcing.record_event(
    event_type="session_created",
    aggregate_id="bulk_operation_123",
    payload={"session_ids": ["s1", "s2", "s3"]},
    metadata={"user_id": "user1", "count": 3}
)

# Obtener eventos
events = event_sourcing.get_events(
    aggregate_id="bulk_operation_123",
    limit=50
)

# Replay eventos
def event_handler(event):
    print(f"Processing {event['event_type']}")
    return event["payload"]

results = event_sourcing.replay_events(
    aggregate_id="bulk_operation_123",
    handler=event_handler
)
```

#### Endpoints API:
- `POST /api/v1/bulk/events/record`
- `GET /api/v1/bulk/events?aggregate_id=xxx&event_type=xxx&limit=100`

---

### 2. **BulkObservability** - Observabilidad Avanzada

Sistema completo de observabilidad con distributed tracing, logging y métricas.

#### Características:
- Distributed tracing con spans
- Logging estructurado
- Correlación de logs con traces
- Resumen de observabilidad

#### Uso:

```python
from core.bulk_operations import BulkObservability
import uuid

observability = BulkObservability()

# Iniciar trace
trace_id = str(uuid.uuid4())
observability.start_trace(
    trace_id=trace_id,
    operation_name="bulk_create_sessions",
    metadata={"count": 100}
)

# Añadir spans
start = time.time()
session_ids = await create_sessions(100)
duration = time.time() - start

observability.add_span(
    trace_id=trace_id,
    span_name="create_sessions",
    duration=duration,
    metadata={"sessions_created": len(session_ids)}
)

# Logging
observability.log_event(
    level="info",
    message="Sessions created successfully",
    context={"count": len(session_ids)},
    trace_id=trace_id
)

# Completar trace
observability.complete_trace(trace_id, status="success")

# Obtener trace
trace = observability.get_trace(trace_id)

# Obtener logs relacionados
logs = observability.get_logs(trace_id=trace_id)
```

#### Endpoints API:
- `POST /api/v1/bulk/traces/start`
- `POST /api/v1/bulk/traces/add-span`
- `POST /api/v1/bulk/traces/complete`
- `GET /api/v1/bulk/traces/{trace_id}`
- `GET /api/v1/bulk/observability/summary`
- `POST /api/v1/bulk/observability/log`
- `GET /api/v1/bulk/observability/logs`

---

### 3. **BulkCostOptimizer** - Optimización de Costos

Sistema para trackear y optimizar costos de operaciones bulk.

#### Características:
- Tracking de costos por operación
- Análisis de costos por item y por segundo
- Sugerencias automáticas de optimización
- Resumen de costos por tipo de operación

#### Uso:

```python
from core.bulk_operations import BulkCostOptimizer

cost_optimizer = BulkCostOptimizer()

# Registrar costo de operación
cost_optimizer.record_operation_cost(
    operation_type="create_sessions",
    cost=5.50,  # en dólares o unidades
    items_processed=100,
    duration=30.5,
    metadata={"provider": "openai", "model": "gpt-4"}
)

# Obtener resumen
summary = cost_optimizer.get_cost_summary()
# Returns: {
#   "total_cost": 55.50,
#   "avg_cost_per_item": 0.055,
#   "by_operation": {...}
# }

# Obtener sugerencias
suggestions = cost_optimizer.suggest_optimizations("create_sessions")
# Returns: [
#   {
#     "type": "reduce_batch_size",
#     "reason": "High cost per item detected",
#     "recommendation": "...",
#     "potential_savings": "20-30%"
#   }
# ]
```

#### Endpoints API:
- `POST /api/v1/bulk/costs/record`
- `GET /api/v1/bulk/costs/summary`
- `GET /api/v1/bulk/costs/optimizations?operation_type=xxx`

---

### 4. **BulkAnomalyDetector** - Detección de Anomalías

Sistema de detección de anomalías basado en análisis estadístico.

#### Características:
- Detección automática usando z-score
- Threshold configurable (default 2.0 desviaciones estándar)
- Tracking de anomalías por métrica
- Resumen de anomalías

#### Uso:

```python
from core.bulk_operations import BulkAnomalyDetector

anomaly_detector = BulkAnomalyDetector(threshold_std=2.0)

# Registrar métricas (la detección es automática)
anomaly_detector.record_metric(
    metric_name="response_time",
    value=0.5,
    metadata={"operation": "create_session"}
)

anomaly_detector.record_metric(
    metric_name="response_time",
    value=15.0,  # Anomalía detectada automáticamente
    metadata={"operation": "create_session"}
)

# Obtener anomalías
anomalies = anomaly_detector.get_anomalies(
    metric_name="response_time",
    limit=10
)

# Resumen
summary = anomaly_detector.get_anomaly_summary()
# Returns: {
#   "total_anomalies": 5,
#   "by_metric": {
#     "response_time": {
#       "count": 3,
#       "latest": {...},
#       "avg_score": 2.5
#     }
#   },
#   "recent_anomalies": [...]
# }
```

#### Endpoints API:
- `POST /api/v1/bulk/anomalies/record`
- `GET /api/v1/bulk/anomalies?metric_name=xxx&limit=100`
- `GET /api/v1/bulk/anomalies/summary`

---

## Integración Completa

### Ejemplo: Sistema Completo de Observabilidad

```python
from core.bulk_operations import (
    BulkEventSourcing,
    BulkObservability,
    BulkCostOptimizer,
    BulkAnomalyDetector
)
import uuid
import time

# Inicializar
event_sourcing = BulkEventSourcing()
observability = BulkObservability()
cost_optimizer = BulkCostOptimizer()
anomaly_detector = BulkAnomalyDetector()

# Operación con observabilidad completa
async def monitored_bulk_operation(count: int):
    operation_id = f"op_{uuid.uuid4()}"
    trace_id = str(uuid.uuid4())
    
    # Iniciar trace
    observability.start_trace(
        trace_id=trace_id,
        operation_name="bulk_create_sessions",
        metadata={"operation_id": operation_id, "count": count}
    )
    
    # Registrar evento
    event_sourcing.record_event(
        event_type="operation_started",
        aggregate_id=operation_id,
        payload={"count": count},
        metadata={"trace_id": trace_id}
    )
    
    start_time = time.time()
    start_cost = calculate_cost_estimate(count)
    
    try:
        # Ejecutar operación
        session_ids = await create_sessions(count)
        duration = time.time() - start_time
        
        # Registrar métricas
        anomaly_detector.record_metric(
            "operation_duration",
            duration,
            metadata={"operation_id": operation_id}
        )
        
        # Registrar costo
        actual_cost = calculate_actual_cost(count)
        cost_optimizer.record_operation_cost(
            "create_sessions",
            actual_cost,
            count,
            duration,
            metadata={"operation_id": operation_id}
        )
        
        # Añadir span
        observability.add_span(
            trace_id,
            "create_sessions",
            duration,
            metadata={"sessions_created": len(session_ids)}
        )
        
        # Registrar evento de éxito
        event_sourcing.record_event(
            event_type="operation_completed",
            aggregate_id=operation_id,
            payload={"session_ids": session_ids},
            metadata={"trace_id": trace_id, "duration": duration}
        )
        
        # Logging
        observability.log_event(
            level="info",
            message="Operation completed successfully",
            context={"count": len(session_ids), "duration": duration},
            trace_id=trace_id
        )
        
        # Completar trace
        observability.complete_trace(trace_id, status="success")
        
        return session_ids
        
    except Exception as e:
        duration = time.time() - start_time
        
        # Registrar error
        observability.log_event(
            level="error",
            message=f"Operation failed: {str(e)}",
            context={"duration": duration},
            trace_id=trace_id
        )
        
        event_sourcing.record_event(
            event_type="operation_failed",
            aggregate_id=operation_id,
            payload={"error": str(e)},
            metadata={"trace_id": trace_id}
        )
        
        observability.complete_trace(trace_id, status="error", error=str(e))
        raise
```

---

## Casos de Uso

### 1. **Auditoría Completa**

```python
# Event sourcing para auditoría
events = event_sourcing.get_events(
    aggregate_id="bulk_operation_123"
)
# Historial completo de la operación
```

### 2. **Debugging con Traces**

```python
# Encontrar trace por operación
trace = observability.get_trace(trace_id)
logs = observability.get_logs(trace_id=trace_id)
# Ver todos los logs y spans relacionados
```

### 3. **Optimización de Costos**

```python
# Analizar costos
summary = cost_optimizer.get_cost_summary()
suggestions = cost_optimizer.suggest_optimizations("create_sessions")
# Implementar optimizaciones sugeridas
```

### 4. **Detección Proactiva de Problemas**

```python
# Revisar anomalías
anomalies = anomaly_detector.get_anomalies(metric_name="response_time")
# Alertar si hay anomalías recientes
if anomalies:
    send_alert(anomalies[-1])
```

---

## Beneficios

1. **Visibilidad Completa**: Event sourcing proporciona historial completo
2. **Debugging Eficiente**: Distributed tracing facilita encontrar problemas
3. **Optimización de Costos**: Tracking y sugerencias automáticas
4. **Detección Temprana**: Anomalías detectadas automáticamente
5. **Auditoría**: Historial completo de todas las operaciones
6. **Correlación**: Logs, traces y eventos correlacionados

---

## Mejores Prácticas

1. **Usar traces en operaciones críticas**: Facilita debugging
2. **Registrar todos los eventos importantes**: Para auditoría completa
3. **Trackear costos regularmente**: Para optimización continua
4. **Monitorear anomalías**: Detectar problemas antes de que afecten usuarios
5. **Correlacionar logs con traces**: Usar trace_id en logs
6. **Revisar sugerencias de optimización**: Implementar mejoras sugeridas

---

## Dashboard de Observabilidad

```python
# Dashboard completo
dashboard = {
    "observability": observability.get_observability_summary(),
    "costs": cost_optimizer.get_cost_summary(),
    "anomalies": anomaly_detector.get_anomaly_summary(),
    "recent_events": event_sourcing.get_events(limit=10)
}
```

Este dashboard proporciona una vista completa del estado del sistema.
















