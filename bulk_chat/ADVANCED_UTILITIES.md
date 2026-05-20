# Advanced Utilities - Utilidades Avanzadas
## Sistema de Utilidades Especializadas para Operaciones Bulk

Este documento describe utilidades avanzadas para procesamiento de datos, profiling, validación, y sincronización.

## 🚀 Nuevas Utilidades Avanzadas

### 1. BulkProfilerAdvanced - Profiler Avanzado

Profiler con estadísticas detalladas de tiempo y memoria.

```python
from bulk_chat.core.bulk_operations_performance import BulkProfilerAdvanced

profiler = BulkProfilerAdvanced()

# Profilear función
result = await profiler.profile("operation", expensive_function, arg1, arg2)

# Obtener estadísticas
stats = await profiler.get_stats("operation")
# {
#   "count": 100,
#   "total_time": 50.5,
#   "min_time": 0.1,
#   "max_time": 2.0,
#   "avg_time": 0.505,
#   "total_memory": 500.0,
#   "avg_memory": 5.0,
#   "errors": 0
# }

# Resetear estadísticas
await profiler.reset_stats("operation")
```

**Características:**
- Tracking de tiempo (min, max, avg, total)
- Tracking de memoria (delta, avg)
- Conteo de errores
- Estadísticas por nombre de operación
- **Mejora:** Profiling detallado sin overhead significativo

### 2. BulkDataTransformer - Transformador de Datos

Transformador de datos con pipelines personalizables.

```python
from bulk_chat.core.bulk_operations_performance import BulkDataTransformer

transformer = BulkDataTransformer()

# Registrar transformador
def normalize_data(data):
    return data.lower().strip()

transformer.register("normalize", normalize_data)

# Transformar datos
result = await transformer.transform("  HELLO  ", "normalize")
# "hello"

# Transformar batch en paralelo
items = ["  ITEM1  ", "  ITEM2  ", "  ITEM3  "]
results = await transformer.transform_batch(items, "normalize")
# ["item1", "item2", "item3"]
```

**Características:**
- Pipelines personalizables
- Procesamiento en paralelo
- Soporte para funciones async y sync
- **Mejora:** Transformación eficiente de datos

### 3. BulkDataValidator - Validador de Datos

Validador con reglas personalizadas.

```python
from bulk_chat.core.bulk_operations_performance import BulkDataValidator

validator = BulkDataValidator()

# Registrar regla de validación
def validate_email(data):
    return "@" in data and "." in data.split("@")[1]

validator.register_rule("email", validate_email)

# Validar datos
is_valid, error = await validator.validate("user@example.com", "email")
# (True, None)

is_valid, error = await validator.validate("invalid", "email")
# (False, "Validation failed")

# Validar batch en paralelo
items = ["user1@example.com", "invalid", "user2@example.com"]
results = await validator.validate_batch(items, "email")
# [(True, None), (False, "Validation failed"), (True, None)]
```

**Características:**
- Reglas personalizadas
- Validación en batch paralela
- Mensajes de error descriptivos
- **Mejora:** Validación eficiente y flexible

### 4. BulkDataAggregator - Agregador de Datos

Agregador con múltiples estrategias.

```python
from bulk_chat.core.bulk_operations_performance import BulkDataAggregator

aggregator = BulkDataAggregator()

# Registrar agregador
def sum_values(items):
    return sum(item.get("value", 0) for item in items)

aggregator.register_aggregator("sum", sum_values)

# Agregar datos
items = [{"value": 1}, {"value": 2}, {"value": 3}]
result = await aggregator.aggregate(items, "sum")
# 6

# Agregar en chunks (para grandes volúmenes)
result = await aggregator.aggregate_chunked(large_items, "sum", chunk_size=1000)
```

**Características:**
- Estrategias personalizables
- Agregación en chunks
- Soporte async
- **Mejora:** Agregación eficiente de grandes volúmenes

### 5. BulkRetryManager - Gestor de Reintentos

Gestor avanzado de reintentos con backoff exponencial.

```python
from bulk_chat.core.bulk_operations_performance import BulkRetryManager

retry_manager = BulkRetryManager(
    max_retries=3,
    initial_delay=1.0,
    backoff_factor=2.0
)

# Ejecutar con reintentos
result = await retry_manager.execute_with_retry(
    api_call,
    arg1,
    arg2,
    retry_key="api_call"
)

# Obtener estadísticas
stats = await retry_manager.get_stats("api_call")
# {
#   "success": 95,
#   "retries": 5
# }
```

**Características:**
- Backoff exponencial
- Estadísticas de reintentos
- Por key independiente
- **Mejora:** Manejo robusto de errores transitorios

### 6. BulkBatchSplitter - Divisor de Batches

Divisor inteligente de batches.

```python
from bulk_chat.core.bulk_operations_performance import BulkBatchSplitter

splitter = BulkBatchSplitter(max_batch_size=100)

# Dividir en batches iguales
items = list(range(250))
batches = splitter.split(items, strategy="equal")
# [[0-99], [100-199], [200-249]]

# Dividir según peso
class Item:
    def __init__(self, weight):
        self.weight = weight

items = [Item(30), Item(40), Item(50), Item(60)]
batches = splitter.split(items, strategy="weighted")
# [[Item(30), Item(40)], [Item(50)], [Item(60)]]
```

**Características:**
- Estrategias de división
- División por peso
- División igual
- **Mejora:** División eficiente de grandes lotes

### 7. BulkDataDeduplicator - Deduplicador

Deduplicador eficiente de datos.

```python
from bulk_chat.core.bulk_operations_performance import BulkDataDeduplicator

deduplicator = BulkDataDeduplicator()

# Verificar duplicado
is_dup = await deduplicator.is_duplicate("item1")
# False

is_dup = await deduplicator.is_duplicate("item1")
# True

# Deduplicar lista
items = ["item1", "item2", "item1", "item3"]
unique = await deduplicator.deduplicate(items)
# ["item1", "item2", "item3"]

# Con función de key personalizada
def get_id(item):
    return item["id"]

items = [{"id": 1}, {"id": 2}, {"id": 1}]
unique = await deduplicator.deduplicate(items, key_func=get_id)
# [{"id": 1}, {"id": 2}]

# Limpiar caché
await deduplicator.clear()
```

**Características:**
- Key function personalizable
- Thread-safe
- Caché persistente
- **Mejora:** Deduplicación eficiente

### 8. BulkDataFormatter - Formateador de Datos

Formateador con múltiples formatos.

```python
from bulk_chat.core.bulk_operations_performance import BulkDataFormatter

formatter = BulkDataFormatter()

# Formatear como JSON
data = {"name": "John", "age": 30}
json_str = formatter.format(data, format_name="json")
# '{"name":"John","age":30}'

# Formatear como CSV
data = [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]
csv_str = formatter.format(data, format_name="csv")
# name,age
# John,30
# Jane,25

# Formatear como XML
xml_str = formatter.format(data[0], format_name="xml")
# <root><name>John</name><age>30</age></root>

# Registrar formateador personalizado
def custom_format(data):
    return f"Custom: {data}"

formatter.register_formatter("custom", custom_format)
```

**Formatos soportados:**
- JSON (usando orjson si disponible)
- CSV
- XML (básico)
- YAML (si pyyaml disponible)

**Mejora:** Formateo eficiente y flexible

### 9. BulkDataParser - Parser de Datos

Parser con múltiples formatos.

```python
from bulk_chat.core.bulk_operations_performance import BulkDataParser

parser = BulkDataParser()

# Parsear JSON
json_str = '{"name":"John","age":30}'
data = parser.parse(json_str, format_name="json")
# {"name": "John", "age": 30}

# Parsear CSV
csv_str = "name,age\nJohn,30\nJane,25"
data = parser.parse(csv_str, format_name="csv")
# [{"name": "John", "age": "30"}, {"name": "Jane", "age": "25"}]

# Parsear XML
xml_str = "<root><name>John</name><age>30</age></root>"
data = parser.parse(xml_str, format_name="xml")
# {"name": "John", "age": "30"}

# Registrar parser personalizado
def custom_parser(data):
    return {"parsed": data}

parser.register_parser("custom", custom_parser)
```

**Formatos soportados:**
- JSON (usando orjson si disponible)
- CSV
- XML (básico)
- YAML (si pyyaml disponible)

**Mejora:** Parsing eficiente y flexible

### 10. BulkAsyncQueue - Cola Asíncrona

Cola asíncrona optimizada con estadísticas.

```python
from bulk_chat.core.bulk_operations_performance import BulkAsyncQueue

queue = BulkAsyncQueue(maxsize=100)

# Agregar items
await queue.put("item1")
await queue.put("item2")

# Obtener items
item = await queue.get()
# "item1"

# Obtener sin esperar
try:
    item = await queue.get_nowait()
except asyncio.QueueEmpty:
    pass

# Obtener tamaño
size = queue.qsize()

# Obtener estadísticas
stats = await queue.get_stats()
# {"put": 100, "get": 95}
```

**Características:**
- Estadísticas de uso
- Thread-safe
- Tamaño máximo configurable
- **Mejora:** Cola eficiente con tracking

### 11. BulkAsyncBarrier - Barrera Asíncrona

Barrera para sincronización de múltiples tareas.

```python
from bulk_chat.core.bulk_operations_performance import BulkAsyncBarrier

barrier = BulkAsyncBarrier(parties=3)

# Tareas esperan en la barrera
async def task1():
    is_last = await barrier.wait()
    print("Task 1 completed")

async def task2():
    is_last = await barrier.wait()
    print("Task 2 completed")

async def task3():
    is_last = await barrier.wait()
    print("Task 3 completed")  # is_last = True

# Todas las tareas esperan hasta que todas lleguen
await asyncio.gather(task1(), task2(), task3())

# Resetear barrera
await barrier.reset()
```

**Características:**
- Sincronización de múltiples tareas
- Identifica última tarea
- Resetable
- **Mejora:** Sincronización eficiente

### 12. BulkAsyncCondition - Condición Asíncrona

Condición asíncrona optimizada.

```python
from bulk_chat.core.bulk_operations_performance import BulkAsyncCondition

condition = BulkAsyncCondition()

# Tarea espera condición
async def waiter():
    await condition.wait()
    print("Condition met!")

# Tarea notifica condición
async def notifier():
    await asyncio.sleep(1)
    await condition.notify()  # Notificar a 1
    # o
    await condition.notify_all()  # Notificar a todos

await asyncio.gather(waiter(), notifier())
```

**Características:**
- Sincronización basada en condiciones
- Notificar uno o todos
- Lock compartido opcional
- **Mejora:** Sincronización condicional

## 📊 Resumen de Utilidades Avanzadas

| Utilidad | Tipo | Mejora |
|----------|------|--------|
| **Profiler Advanced** | Profiling | Estadísticas detalladas |
| **Data Transformer** | Transformación | Pipelines personalizables |
| **Data Validator** | Validación | Reglas flexibles |
| **Data Aggregator** | Agregación | Múltiples estrategias |
| **Retry Manager** | Reintentos | Backoff exponencial |
| **Batch Splitter** | División | Estrategias inteligentes |
| **Data Deduplicator** | Deduplicación | Eficiente |
| **Data Formatter** | Formateo | Múltiples formatos |
| **Data Parser** | Parsing | Múltiples formatos |
| **Async Queue** | Cola | Con estadísticas |
| **Async Barrier** | Sincronización | Barrera |
| **Async Condition** | Sincronización | Condición |

## 🎯 Casos de Uso Avanzados

### Pipeline de Procesamiento
```python
transformer = BulkDataTransformer()
validator = BulkDataValidator()
aggregator = BulkDataAggregator()

# Registrar componentes
transformer.register("normalize", normalize)
validator.register_rule("email", validate_email)
aggregator.register_aggregator("sum", sum_values)

# Pipeline completo
items = ["  USER@EXAMPLE.COM  ", "  USER2@TEST.COM  "]
normalized = await transformer.transform_batch(items, "normalize")
validated = await validator.validate_batch(normalized, "email")
results = await aggregator.aggregate(validated, "sum")
```

### Profiling y Optimización
```python
profiler = BulkProfilerAdvanced()

# Profilear operaciones
for i in range(100):
    await profiler.profile("operation", expensive_func, arg)

# Analizar estadísticas
stats = await profiler.get_stats("operation")
if stats["avg_time"] > 1.0:
    # Optimizar si es lento
    optimize_operation()
```

### Procesamiento con Reintentos
```python
retry_manager = BulkRetryManager(max_retries=5)

# Procesar con reintentos
results = []
for item in items:
    result = await retry_manager.execute_with_retry(
        process_item,
        item,
        retry_key="process"
    )
    results.append(result)

# Ver estadísticas
stats = await retry_manager.get_stats("process")
success_rate = stats["success"] / (stats["success"] + stats["retries"])
```

## 📈 Beneficios Totales

1. **Profiler Advanced**: Estadísticas detalladas de rendimiento
2. **Data Transformer**: Pipelines flexibles y eficientes
3. **Data Validator**: Validación robusta y paralela
4. **Data Aggregator**: Agregación eficiente de grandes volúmenes
5. **Retry Manager**: Manejo robusto de errores
6. **Batch Splitter**: División inteligente de batches
7. **Data Deduplicator**: Deduplicación eficiente
8. **Data Formatter/Parser**: Formateo y parsing flexible
9. **Async Queue**: Cola con tracking
10. **Async Barrier/Condition**: Sincronización avanzada

## 🚀 Resultados Esperados

Con todas las utilidades avanzadas:

- **Profiling detallado** de operaciones
- **Pipelines flexibles** de transformación
- **Validación robusta** y paralela
- **Agregación eficiente** de grandes volúmenes
- **Manejo robusto** de errores con reintentos
- **División inteligente** de batches
- **Deduplicación eficiente**
- **Formateo y parsing** flexible
- **Sincronización avanzada** de tareas

El sistema ahora tiene **65+ optimizaciones y utilidades** que cubren todos los aspectos posibles de procesamiento masivo.
















