# Funcionalidades Avanzadas de Bulk Operations

## Nuevas Clases Implementadas

### 1. **BulkRealTimeMetrics** - Métricas en Tiempo Real

Sistema completo de métricas en tiempo real para monitorear operaciones bulk.

#### Características:
- Ventana deslizante de métricas (configurable, default 60 segundos)
- Agregación automática de métricas por tipo de operación
- Cálculo de health status basado en métricas
- Tracking de: operaciones por segundo, items por segundo, success rate, duraciones

#### Métodos:
- `record_operation()`: Registrar una operación
- `get_metrics()`: Obtener métricas (todas o por tipo)
- `get_health_status()`: Obtener estado de salud del sistema

#### Endpoints API:
- `GET /api/v1/bulk/metrics/realtime?operation_type=opcional`
- `GET /api/v1/bulk/metrics/health`
- `POST /api/v1/bulk/metrics/record`

#### Ejemplo de uso:
```python
metrics = BulkRealTimeMetrics(window_size_seconds=60)

# Registrar operación
metrics.record_operation(
    operation_type="create_sessions",
    duration=2.5,
    success=True,
    items_processed=100,
    metadata={"user_id": "user123"}
)

# Obtener métricas
health = metrics.get_health_status()
# Returns: {"status": "healthy", "overall_success_rate": 95.5, ...}
```

---

### 2. **BulkAdvancedCache** - Caché Avanzado con TTL y LRU

Sistema de caché inteligente con expiración automática (TTL) y algoritmo LRU.

#### Características:
- TTL configurable por entrada o global
- Algoritmo LRU (Least Recently Used) para evicción
- Estadísticas de hit/miss rate
- Invalidación manual de entradas

#### Métodos:
- `get(key)`: Obtener valor del caché
- `set(key, value, ttl)`: Guardar valor con TTL
- `invalidate(key)`: Invalidar entrada específica
- `clear()`: Limpiar todo el caché
- `get_stats()`: Obtener estadísticas

#### Endpoints API:
- `GET /api/v1/bulk/cache/stats`
- `POST /api/v1/bulk/cache/get` - Body: `{"key": "my_key"}`
- `POST /api/v1/bulk/cache/set` - Body: `{"key": "my_key", "value": {...}, "ttl": 3600}`
- `POST /api/v1/bulk/cache/invalidate` - Body: `{"key": "my_key"}`

#### Ejemplo de uso:
```python
cache = BulkAdvancedCache(max_size=1000, default_ttl=3600)

# Guardar
cache.set("session_data_123", {"data": "value"}, ttl=7200)

# Obtener
value = cache.get("session_data_123")

# Estadísticas
stats = cache.get_stats()
# Returns: {"size": 500, "hits": 1000, "misses": 200, "hit_rate": 83.33, ...}
```

---

### 3. **BulkPriorityQueue** - Cola de Prioridades

Sistema de cola con prioridades para gestionar operaciones según su importancia.

#### Características:
- 5 niveles de prioridad predefinidos: critical, high, medium, low, background
- Prioridades numéricas personalizables
- Ordenamiento automático por prioridad y timestamp
- Estadísticas de la cola

#### Métodos:
- `enqueue(operation, priority)`: Agregar operación
- `dequeue()`: Obtener siguiente operación (mayor prioridad)
- `peek()`: Ver siguiente sin remover
- `get_by_priority(priority)`: Obtener todas de una prioridad
- `get_stats()`: Estadísticas de la cola

#### Endpoints API:
- `GET /api/v1/bulk/queue/stats`
- `POST /api/v1/bulk/queue/enqueue` - Body: `{"operation": {...}, "priority": "high"}`
- `POST /api/v1/bulk/queue/dequeue`
- `GET /api/v1/bulk/queue/peek`

#### Ejemplo de uso:
```python
queue = BulkPriorityQueue()

# Encolar con prioridad
queue.enqueue({"type": "create_session", "data": {...}}, priority="critical")
queue.enqueue({"type": "send_message", "data": {...}}, priority="low")

# Desencolar (obtiene la de mayor prioridad)
next_op = queue.dequeue()

# Estadísticas
stats = queue.get_stats()
# Returns: {"total": 50, "by_priority": {10: 5, 7: 10, 5: 20, ...}, ...}
```

---

### 4. **BulkEnhancedValidator** - Validador Mejorado

Sistema de validación avanzado con caché de resultados y validación por lotes.

#### Características:
- Validación por tipo de operación
- Caché de resultados de validación
- Validación de lotes (batch validation)
- Reglas de validación personalizables

#### Métodos:
- `add_rule(operation_type, validator)`: Añadir regla de validación
- `validate(operation_type, data, use_cache)`: Validar datos
- `validate_batch(operation_type, items)`: Validar lote
- `clear_cache()`: Limpiar caché de validación

#### Endpoints API:
- `POST /api/v1/bulk/validator/validate` - Body: `{"operation_type": "create", "data": {...}, "use_cache": true}`
- `POST /api/v1/bulk/validator/validate-batch` - Body: `{"operation_type": "create", "items": [...]}`
- `POST /api/v1/bulk/validator/add-rule`

#### Ejemplo de uso:
```python
validator = BulkEnhancedValidator()

# Añadir regla
def validate_session_data(data):
    return "session_id" in data and "user_id" in data

validator.add_rule("create_session", validate_session_data)

# Validar
is_valid, error = validator.validate("create_session", {"session_id": "123", "user_id": "user1"})

# Validar lote
results = validator.validate_batch("create_session", [
    {"session_id": "1", "user_id": "user1"},
    {"session_id": "2"},  # Inválido
    {"session_id": "3", "user_id": "user3"}
])
# Returns: {"valid": [...], "invalid": [...], "validity_rate": 66.67, ...}
```

---

### 5. **BulkDashboard** - Dashboard Integrado

Dashboard completo que integra todas las funcionalidades avanzadas.

#### Características:
- Vista unificada de métricas, caché, cola y alertas
- Resumen ejecutivo del sistema
- Sistema de alertas multi-nivel
- Health status integrado

#### Métodos:
- `get_dashboard_data()`: Obtener datos completos
- `add_alert(level, message, details)`: Añadir alerta
- `get_alerts(level)`: Obtener alertas (filtradas por nivel)

#### Endpoints API:
- `GET /api/v1/bulk/dashboard` - Datos completos
- `GET /api/v1/bulk/dashboard/summary` - Resumen ejecutivo
- `POST /api/v1/bulk/dashboard/alert` - Body: `{"level": "warning", "message": "...", "details": {...}}`
- `GET /api/v1/bulk/dashboard/alerts?level=opcional`

#### Ejemplo de uso:
```python
dashboard = BulkDashboard(
    metrics=bulk_realtime_metrics,
    cache=bulk_advanced_cache,
    priority_queue=bulk_priority_queue
)

# Obtener datos completos
data = dashboard.get_dashboard_data()
# Returns: {
#   "timestamp": "...",
#   "metrics": {...},
#   "cache": {...},
#   "queue": {...},
#   "alerts": [...],
#   "summary": {...}
# }

# Añadir alerta
dashboard.add_alert(
    level="warning",
    message="High error rate detected",
    details={"error_rate": 15.5}
)

# Obtener resumen
summary = dashboard._generate_summary()
# Returns: {
#   "system_status": "healthy",
#   "success_rate": 95.5,
#   "cache_hit_rate": 83.33,
#   "pending_operations": 10,
#   "performance": "excellent"
# }
```

---

## Integración con Operaciones Existentes

Todas las nuevas clases están diseñadas para integrarse con las operaciones bulk existentes:

```python
# Ejemplo: Usar métricas en operaciones bulk
from core.bulk_operations import BulkSessionOperations, BulkRealTimeMetrics

metrics = BulkRealTimeMetrics()
bulk_sessions = BulkSessionOperations(...)

# Envolver operaciones para tracking
async def tracked_create_sessions(count):
    start = datetime.now()
    try:
        session_ids = await bulk_sessions.create_sessions(count)
        duration = (datetime.now() - start).total_seconds()
        metrics.record_operation(
            "create_sessions",
            duration,
            success=True,
            items_processed=len(session_ids)
        )
        return session_ids
    except Exception as e:
        duration = (datetime.now() - start).total_seconds()
        metrics.record_operation(
            "create_sessions",
            duration,
            success=False,
            metadata={"error": str(e)}
        )
        raise
```

---

## Beneficios de las Nuevas Funcionalidades

1. **Visibilidad**: Métricas en tiempo real para monitorear el sistema
2. **Performance**: Caché inteligente reduce operaciones redundantes
3. **Priorización**: Cola de prioridades permite gestionar carga crítica primero
4. **Validación**: Sistema robusto de validación con caché
5. **Monitoreo**: Dashboard integrado para visión completa del sistema

---

## Uso Recomendado

1. **Métricas**: Registrar todas las operaciones bulk importantes
2. **Caché**: Usar para resultados de consultas frecuentes y validaciones
3. **Cola**: Priorizar operaciones críticas sobre background tasks
4. **Validación**: Validar datos antes de procesar en bulk
5. **Dashboard**: Monitorear salud del sistema regularmente

---

## Próximas Mejoras

- [ ] Dashboard web UI integrado
- [ ] Exportación de métricas a sistemas externos (Prometheus, etc.)
- [ ] Alertas automáticas basadas en umbrales
- [ ] Integración con sistemas de logging avanzados
- [ ] Métricas históricas con persistencia
