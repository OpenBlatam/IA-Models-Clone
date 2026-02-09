# System Architecture - Arquitectura del Sistema
## Componentes de Arquitectura Avanzados para Operaciones Bulk

Este documento describe componentes de arquitectura avanzados para sistemas distribuidos, búsqueda, logging, y workflows.

## 🏗️ Nuevos Componentes de Arquitectura

### 1. BulkDistributedCache - Cache Distribuido

Cache distribuido con TTL y eviction.

```python
from bulk_chat.core.bulk_operations_performance import BulkDistributedCache

cache = BulkDistributedCache(ttl=3600.0, max_size=10000)

# Almacenar en cache
await cache.set("key1", "value1", ttl=1800.0)

# Obtener del cache
value = await cache.get("key1")
# "value1"

# Eliminar
await cache.delete("key1")

# Estadísticas
stats = await cache.get_stats()
# {
#   "hits": 950,
#   "misses": 50,
#   "evictions": 10,
#   "size": 9000,
#   "hit_rate": 0.95
# }
```

**Características:**
- TTL configurable
- LRU eviction
- Estadísticas de cache
- Thread-safe
- **Mejora:** Cache eficiente con alta hit rate

### 2. BulkSearchIndex - Índice de Búsqueda

Índice de búsqueda optimizado.

```python
from bulk_chat.core.bulk_operations_performance import BulkSearchIndex

index = BulkSearchIndex()

# Indexar documentos
await index.index_document("doc1", "Hello world", {"author": "John"})
await index.index_document("doc2", "Hello Python", {"author": "Jane"})

# Buscar
results = await index.search("Hello")
# ["doc1", "doc2"]

results = await index.search("Python")
# ["doc2"]

# Obtener documento
doc = await index.get_document("doc1")
# {"text": "Hello world", "metadata": {"author": "John"}}

# Remover documento
await index.remove_document("doc1")
```

**Características:**
- Búsqueda por palabras
- Intersección de términos
- Metadata asociada
- **Mejora:** Búsqueda rápida y eficiente

### 3. BulkLogAggregator - Agregador de Logs

Agregador de logs optimizado.

```python
from bulk_chat.core.bulk_operations_performance import BulkLogAggregator

logger = BulkLogAggregator(max_logs=10000)

# Agregar logs
await logger.log("INFO", "Operation started", {"operation_id": "123"})
await logger.log("ERROR", "Operation failed", {"error": "Timeout"})

# Obtener logs
logs = await logger.get_logs(level="ERROR", limit=10)

# Estadísticas
stats = await logger.get_stats()
# {
#   "total_logs": 1000,
#   "by_level": {"INFO": 800, "ERROR": 150, "WARNING": 50}
# }

# Limpiar
await logger.clear()
```

**Características:**
- Filtrado por nivel
- Límite de logs
- Estadísticas por nivel
- **Mejora:** Logging eficiente y estructurado

### 4. BulkDataSerializer - Serializador Avanzado

Serializador con múltiples formatos.

```python
from bulk_chat.core.bulk_operations_performance import BulkDataSerializer

serializer = BulkDataSerializer()

# Serializar
data = {"name": "John", "age": 30}
json_bytes = serializer.serialize(data, format_name="json")
msgpack_bytes = serializer.serialize(data, format_name="msgpack")
pickle_bytes = serializer.serialize(data, format_name="pickle")

# Deserializar
data = serializer.deserialize(json_bytes, format_name="json")
data = serializer.deserialize(msgpack_bytes, format_name="msgpack")

# Registrar formato personalizado
def custom_serialize(data):
    return str(data).encode()

def custom_deserialize(data):
    return data.decode()

serializer.register("custom", custom_serialize, custom_deserialize)
```

**Formatos soportados:**
- JSON (orjson si disponible)
- Msgpack (si disponible)
- Pickle

**Mejora:** Serialización eficiente y flexible

### 5. BulkTaskScheduler - Scheduler Avanzado

Scheduler de tareas con delays y recurrencia.

```python
from bulk_chat.core.bulk_operations_performance import BulkTaskScheduler

scheduler = BulkTaskScheduler()

# Programar tarea con delay
result = await scheduler.schedule(
    "task1",
    process_data,
    delay=5.0,
    arg1,
    arg2
)

# Programar tarea recurrente
await scheduler.schedule_recurring(
    "cleanup",
    cleanup_function,
    interval=3600.0
)

# Cancelar tarea
await scheduler.cancel("cleanup")
```

**Características:**
- Tareas con delay
- Tareas recurrentes
- Cancelación de tareas
- **Mejora:** Scheduling flexible y eficiente

### 6. BulkLoadBalancer - Load Balancer

Load balancer interno con múltiples estrategias.

```python
from bulk_chat.core.bulk_operations_performance import BulkLoadBalancer

lb = BulkLoadBalancer(strategy="round_robin")

# Agregar backends
lb.add_backend(backend1, weight=1.0)
lb.add_backend(backend2, weight=2.0)
lb.add_backend(backend3, weight=1.0)

# Obtener backend
backend = await lb.get_backend()
# Selecciona según estrategia

# Liberar backend
await lb.release_backend(backend)
```

**Estrategias:**
- **round_robin**: Rotación circular
- **least_connections**: Menor número de conexiones
- **weighted**: Según peso

**Mejora:** Distribución eficiente de carga

### 7. BulkCircuitBreakerAdvanced - Circuit Breaker Avanzado

Circuit breaker para manejo de fallos.

```python
from bulk_chat.core.bulk_operations_performance import BulkCircuitBreakerAdvanced

breaker = BulkCircuitBreakerAdvanced(
    failure_threshold=5,
    timeout=60.0
)

# Ejecutar con circuit breaker
try:
    result = await breaker.call(api_call, arg1, arg2)
except Exception as e:
    # Circuit breaker abierto después de 5 fallos
    print("Circuit breaker is open")
```

**Estados:**
- **closed**: Normal, permite llamadas
- **open**: Bloquea llamadas después de fallos
- **half_open**: Prueba después de timeout

**Mejora:** Resiliencia ante fallos

### 8. BulkHealthChecker - Health Checker

Health checker avanzado.

```python
from bulk_chat.core.bulk_operations_performance import BulkHealthChecker

health = BulkHealthChecker()

# Registrar checks
async def check_database():
    # Verificar conexión a DB
    return True

health.register_check("database", check_database)

# Ejecutar check
status = await health.check("database")
# {"status": "healthy", "timestamp": 1234567890}

# Todos los checks
all_status = await health.check()
# {"database": {"status": "healthy", ...}, ...}

# Obtener estado
current_status = await health.get_status("database")
```

**Características:**
- Checks personalizados
- Estado persistente
- Múltiples checks
- **Mejora:** Monitoreo de salud del sistema

### 9. BulkMetricsCollector - Collector de Métricas

Collector de métricas con ventanas de tiempo.

```python
from bulk_chat.core.bulk_operations_performance import BulkMetricsCollector

metrics = BulkMetricsCollector()

# Registrar métricas
await metrics.record("requests_per_second", 10.5)
await metrics.record("response_time", 0.250)
await metrics.record("error_rate", 0.01)

# Obtener métrica
stats = await metrics.get_metric("response_time", window=3600.0)
# {
#   "count": 1000,
#   "sum": 250.0,
#   "avg": 0.25,
#   "min": 0.1,
#   "max": 0.5,
#   "latest": 0.25
# }

# Todas las métricas
all_metrics = await metrics.get_all_metrics(window=3600.0)
```

**Características:**
- Ventanas de tiempo
- Estadísticas (count, sum, avg, min, max)
- Historial limitado (últimos 1000)
- **Mejora:** Métricas detalladas y eficientes

### 10. BulkEventBus - Event Bus

Event bus para comunicación desacoplada.

```python
from bulk_chat.core.bulk_operations_performance import BulkEventBus

event_bus = BulkEventBus()

# Suscribirse a eventos
async def handle_user_created(data):
    print(f"User created: {data}")

await event_bus.subscribe("user.created", handle_user_created)

# Publicar evento
await event_bus.publish("user.created", {"user_id": "123", "name": "John"})

# Desuscribirse
await event_bus.unsubscribe("user.created", handle_user_created)
```

**Características:**
- Comunicación desacoplada
- Múltiples subscribers
- Ejecución paralela
- **Mejora:** Arquitectura basada en eventos

### 11. BulkStateMachine - State Machine

State machine para gestión de estados.

```python
from bulk_chat.core.bulk_operations_performance import BulkStateMachine

sm = BulkStateMachine(initial_state="idle")

# Agregar transiciones
sm.add_transition("idle", "running")
sm.add_transition("running", "completed")
sm.add_transition("running", "failed")

# Transicionar
success = await sm.transition("running")
# True

# Verificar estado
current_state = sm.get_state()
# "running"

# Historial
history = sm.get_history()
# [(1234567890, "idle", "running"), ...]
```

**Características:**
- Transiciones con condiciones
- Historial de transiciones
- Thread-safe
- **Mejora:** Gestión estructurada de estados

### 12. BulkWorkflowEngine - Motor de Workflows

Motor de workflows para procesos complejos.

```python
from bulk_chat.core.bulk_operations_performance import BulkWorkflowEngine

engine = BulkWorkflowEngine()

# Definir workflow
def step1(data, **kwargs):
    return data + 1

async def step2(data, **kwargs):
    await asyncio.sleep(0.1)
    return data * 2

engine.register_workflow("process", [
    {"func": step1, "kwargs": {}},
    {"func": step2, "kwargs": {}, "stop_on_error": True}
])

# Ejecutar workflow
result = await engine.execute("process", initial_data=5)
# 12 (5 + 1 = 6, 6 * 2 = 12)

# Estado de ejecución
status = await engine.get_execution_status("process_1234567890")
# {"status": "completed", "started": 1234567890, ...}
```

**Características:**
- Workflows multi-paso
- Parada en errores
- Seguimiento de ejecución
- **Mejora:** Procesos complejos estructurados

## 📊 Resumen de Componentes de Arquitectura

| Componente | Tipo | Mejora |
|------------|------|--------|
| **Distributed Cache** | Cache | Alta hit rate |
| **Search Index** | Búsqueda | Búsqueda rápida |
| **Log Aggregator** | Logging | Logging estructurado |
| **Data Serializer** | Serialización | Múltiples formatos |
| **Task Scheduler** | Scheduling | Tareas flexibles |
| **Load Balancer** | Balanceo | Distribución eficiente |
| **Circuit Breaker** | Resiliencia | Manejo de fallos |
| **Health Checker** | Monitoreo | Estado del sistema |
| **Metrics Collector** | Métricas | Estadísticas detalladas |
| **Event Bus** | Eventos | Comunicación desacoplada |
| **State Machine** | Estados | Gestión estructurada |
| **Workflow Engine** | Workflows | Procesos complejos |

## 🎯 Casos de Uso Arquitectónicos

### Sistema Completo con Cache y Búsqueda
```python
cache = BulkDistributedCache()
index = BulkSearchIndex()

# Indexar y cachear
async def process_document(doc_id, text):
    await index.index_document(doc_id, text)
    await cache.set(f"doc:{doc_id}", text)

# Buscar y cachear resultados
async def search_with_cache(query):
    # Verificar cache primero
    cached = await cache.get(f"search:{query}")
    if cached:
        return cached
    
    # Buscar
    results = await index.search(query)
    
    # Cachear resultados
    await cache.set(f"search:{query}", results, ttl=300.0)
    return results
```

### Sistema con Event Bus y Workflows
```python
event_bus = BulkEventBus()
workflow_engine = BulkWorkflowEngine()

# Workflow de procesamiento
workflow_engine.register_workflow("process_order", [
    {"func": validate_order},
    {"func": process_payment},
    {"func": send_notification}
])

# Handler de eventos
async def handle_order_created(data):
    await workflow_engine.execute("process_order", initial_data=data)

event_bus.subscribe("order.created", handle_order_created)

# Publicar evento
await event_bus.publish("order.created", {"order_id": "123"})
```

### Sistema con Health Checks y Métricas
```python
health = BulkHealthChecker()
metrics = BulkMetricsCollector()

# Health checks
health.register_check("database", check_db)
health.register_check("cache", check_cache)

# Métricas
await metrics.record("requests", 1)
await metrics.record("response_time", 0.25)

# Monitoreo
async def monitor_system():
    health_status = await health.check()
    metrics_stats = await metrics.get_all_metrics(window=60.0)
    
    if health_status["database"]["status"] != "healthy":
        # Alertar
        pass
```

## 📈 Beneficios Totales

1. **Distributed Cache**: Alta hit rate y rendimiento
2. **Search Index**: Búsqueda rápida y eficiente
3. **Log Aggregator**: Logging estructurado
4. **Data Serializer**: Serialización flexible
5. **Task Scheduler**: Scheduling avanzado
6. **Load Balancer**: Distribución eficiente
7. **Circuit Breaker**: Resiliencia ante fallos
8. **Health Checker**: Monitoreo del sistema
9. **Metrics Collector**: Métricas detalladas
10. **Event Bus**: Arquitectura desacoplada
11. **State Machine**: Gestión estructurada de estados
12. **Workflow Engine**: Procesos complejos

## 🚀 Resultados Esperados

Con todos los componentes de arquitectura:

- **Cache distribuido** con alta hit rate
- **Búsqueda rápida** y eficiente
- **Logging estructurado** y agregado
- **Serialización flexible** multi-formato
- **Scheduling avanzado** de tareas
- **Balanceo de carga** eficiente
- **Resiliencia** ante fallos
- **Monitoreo** de salud del sistema
- **Métricas detalladas** con ventanas de tiempo
- **Arquitectura basada en eventos**
- **Gestión estructurada** de estados
- **Workflows complejos** estructurados

El sistema ahora tiene **77+ optimizaciones, utilidades y componentes de arquitectura** que cubren todos los aspectos posibles de sistemas distribuidos y procesamiento masivo.















