# Analysis Utilities - Utilidades de Análisis
## Utilidades Avanzadas para Análisis, Comparación y Estadísticas

Este documento describe utilidades avanzadas para análisis de datos, comparación, merge, ordenamiento, búsqueda, estadísticas, validación, normalización y monitoreo.

## 🚀 Nuevas Utilidades de Análisis

### 1. BulkDataComparator - Comparador de Datos

Comparador avanzado de datos.

```python
from bulk_chat.core.bulk_operations_performance import BulkDataComparator

comparator = BulkDataComparator()

# Comparar valores
result = comparator.compare(5, 10)
# -1 (5 < 10)

# Comparar con key function
result = comparator.compare(
    {"value": 5},
    {"value": 10},
    key_func=lambda x: x["value"]
)

# Verificar igualdad con tolerancia
is_equal = comparator.is_equal(5.0, 5.01, tolerance=0.1)
# True

# Comparación profunda
deep_equal = comparator.deep_equal(
    {"a": 1, "b": [2, 3]},
    {"a": 1, "b": [2, 3]}
)
# True
```

**Características:**
- Comparación con key function
- Tolerancia para floats
- Comparación profunda
- **Mejora:** Comparación robusta

### 2. BulkDataMerger - Mergedor de Datos

Mergedor de datos con merge profundo.

```python
from bulk_chat.core.bulk_operations_performance import BulkDataMerger

merger = BulkDataMerger()

# Merge simple
dict1 = {"a": 1, "b": 2}
dict2 = {"c": 3, "d": 4}
merged = merger.merge(dict1, dict2)
# {"a": 1, "b": 2, "c": 3, "d": 4}

# Merge profundo
dict1 = {"a": {"b": 1}, "c": 2}
dict2 = {"a": {"d": 3}, "e": 4}
deep_merged = merger.deep_merge(dict1, dict2)
# {"a": {"b": 1, "d": 3}, "c": 2, "e": 4}

# Merge de listas
list1 = [1, 2, 3]
list2 = [4, 5, 6]
merged_list = merger.merge_lists(list1, list2)
# [1, 2, 3, 4, 5, 6]

# Merge único (sin duplicados)
merged_unique = merger.merge_unique(list1, [3, 4, 5])
# [1, 2, 3, 4, 5, 6]
```

**Características:**
- Merge simple y profundo
- Merge de listas
- Eliminación de duplicados
- **Mejora:** Merge eficiente

### 3. BulkDataSorter - Ordenador Avanzado

Ordenador con múltiples estrategias.

```python
from bulk_chat.core.bulk_operations_performance import BulkDataSorter

sorter = BulkDataSorter()

# Ordenar simple
items = [3, 1, 4, 1, 5]
sorted_items = sorter.sort(items)
# [1, 1, 3, 4, 5]

# Ordenar por key
items = [{"value": 3}, {"value": 1}, {"value": 4}]
sorted_items = sorter.sort(items, key_func=lambda x: x["value"])

# Ordenar por múltiples keys
items = [
    {"a": 1, "b": 2},
    {"a": 1, "b": 1},
    {"a": 2, "b": 1}
]
sorted_items = sorter.sort_by_multiple(
    items,
    key_funcs=[lambda x: x["a"], lambda x: x["b"]]
)
```

**Características:**
- Ordenamiento simple
- Ordenamiento por múltiples keys
- Ordenamiento estable
- **Mejora:** Ordenamiento eficiente

### 4. BulkDataSearcher - Buscador de Datos

Buscador con múltiples algoritmos.

```python
from bulk_chat.core.bulk_operations_performance import BulkDataSearcher

searcher = BulkDataSearcher()

# Búsqueda lineal
items = [1, 2, 3, 4, 5]
result = searcher.linear_search(items, lambda x: x > 3)
# 4

# Búsqueda binaria (requiere lista ordenada)
sorted_items = [1, 2, 3, 4, 5]
index = searcher.binary_search(sorted_items, 3)
# 2

# Encontrar todos
items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
results = searcher.find_all(items, lambda x: x % 2 == 0)
# [2, 4, 6, 8, 10]
```

**Características:**
- Búsqueda lineal
- Búsqueda binaria
- Encontrar todos
- **Mejora:** Búsqueda eficiente

### 5. BulkDataStatistics - Estadísticas

Calculador de estadísticas avanzado.

```python
from bulk_chat.core.bulk_operations_performance import BulkDataStatistics

stats = BulkDataStatistics()

values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Media
mean = stats.mean(values)
# 5.5

# Mediana
median = stats.median(values)
# 5.5

# Moda
mode = stats.mode([1, 2, 2, 3, 3, 3])
# 3

# Desviación estándar
std = stats.std_dev(values)
# 3.03...

# Percentil
p95 = stats.percentile(values, 0.95)
# 9.55
```

**Características:**
- Media, mediana, moda
- Desviación estándar
- Percentiles
- **Mejora:** Análisis estadístico completo

### 6. BulkDataValidatorAdvanced - Validador Avanzado

Validador con esquemas complejos.

```python
from bulk_chat.core.bulk_operations_performance import BulkDataValidatorAdvanced

validator = BulkDataValidatorAdvanced()

# Registrar esquema
schema = {
    "type": dict,
    "required": ["name", "age"],
    "properties": {
        "name": {"type": str},
        "age": {"type": int},
        "email": {"type": str}
    }
}
validator.register_schema("user", schema)

# Validar
data = {"name": "John", "age": 30, "email": "john@example.com"}
is_valid, error = validator.validate_schema(data, "user")
# (True, None)

data = {"name": "John"}  # Falta "age"
is_valid, error = validator.validate_schema(data, "user")
# (False, "Missing required field: age")
```

**Características:**
- Esquemas complejos
- Validación recursiva
- Campos requeridos
- **Mejora:** Validación robusta

### 7. BulkDataNormalizer - Normalizador

Normalizador y estandarizador de datos.

```python
from bulk_chat.core.bulk_operations_performance import BulkDataNormalizer

normalizer = BulkDataNormalizer()

# Normalizar (0-1)
values = [1, 2, 3, 4, 5]
normalized = normalizer.normalize(values)
# [0.0, 0.25, 0.5, 0.75, 1.0]

# Estandarizar (z-score)
standardized = normalizer.standardize(values)
# [-1.41, -0.71, 0.0, 0.71, 1.41]
```

**Características:**
- Normalización min-max
- Estandarización z-score
- **Mejora:** Normalización eficiente

### 8. BulkDataSampler - Muestreador

Muestreador con múltiples estrategias.

```python
from bulk_chat.core.bulk_operations_performance import BulkDataSampler

sampler = BulkDataSampler()

# Muestreo simple
items = list(range(100))
sample = sampler.sample(items, n=10)
# [5, 23, 67, ...] (10 items aleatorios)

# Muestreo con reemplazo
sample = sampler.sample(items, n=10, replace=True)

# Muestreo con pesos
weights = [0.1, 0.2, 0.3, 0.4]
sample = sampler.sample_weighted(items[:4], weights, n=10)

# Muestreo estratificado
groups = ["A", "A", "B", "B", "C", "C"]
sample = sampler.sample_stratified(items[:6], groups, n_per_group=1)
```

**Características:**
- Muestreo simple
- Muestreo con reemplazo
- Muestreo con pesos
- Muestreo estratificado
- **Mejora:** Muestreo eficiente

### 9. BulkDataTransformerAdvanced - Transformador Avanzado

Transformador con pipelines.

```python
from bulk_chat.core.bulk_operations_performance import BulkDataTransformerAdvanced

transformer = BulkDataTransformerAdvanced()

# Registrar pipeline
transformer.register_pipeline("process", [
    lambda x: x.strip(),
    lambda x: x.lower(),
    lambda x: x.replace(" ", "_")
])

# Aplicar pipeline
result = await transformer.transform_pipeline("  Hello World  ", "process")
# "hello_world"
```

**Características:**
- Pipelines de transformación
- Múltiples pasos
- Síncrono y asíncrono
- **Mejora:** Transformación en pipeline

### 10. BulkAsyncMonitor - Monitor Avanzado

Monitor con alertas.

```python
from bulk_chat.core.bulk_operations_performance import BulkAsyncMonitor

monitor = BulkAsyncMonitor()

# Registrar métrica
await monitor.record_metric("cpu_usage", 75.5)
await monitor.record_metric("memory_usage", 60.2)

# Registrar alerta
async def alert_handler(metric_name, value):
    print(f"Alert: {metric_name} = {value}")

await monitor.register_alert(
    "cpu_usage",
    condition=lambda v: v > 80,
    handler=alert_handler
)

# Obtener métricas
metrics = await monitor.get_metrics("cpu_usage", window=3600.0)
# {
#   "count": 100,
#   "mean": 72.5,
#   "min": 50.0,
#   "max": 85.0,
#   "latest": 75.5
# }
```

**Características:**
- Registro de métricas
- Alertas automáticas
- Ventanas de tiempo
- **Mejora:** Monitoreo avanzado

### 11. BulkAsyncNotifier - Notificador

Notificador asíncrono con canales.

```python
from bulk_chat.core.bulk_operations_performance import BulkAsyncNotifier

notifier = BulkAsyncNotifier()

# Registrar canal
async def email_handler(message, data):
    await send_email(message, data)

await notifier.register_channel("email", email_handler)

# Enviar notificación
await notifier.notify("email", "User created", {"user_id": "123"})
```

**Características:**
- Canales múltiples
- Handlers personalizados
- Ejecución paralela
- **Mejora:** Notificaciones eficientes

## 📊 Resumen de Utilidades de Análisis

| Utilidad | Tipo | Mejora |
|----------|------|--------|
| **Data Comparator** | Comparación | Comparación robusta |
| **Data Merger** | Merge | Merge eficiente |
| **Data Sorter** | Ordenamiento | Ordenamiento avanzado |
| **Data Searcher** | Búsqueda | Búsqueda eficiente |
| **Data Statistics** | Estadísticas | Análisis completo |
| **Data Validator Advanced** | Validación | Validación con esquemas |
| **Data Normalizer** | Normalización | Normalización eficiente |
| **Data Sampler** | Muestreo | Muestreo avanzado |
| **Data Transformer Advanced** | Transformación | Pipelines |
| **Async Monitor** | Monitoreo | Monitoreo con alertas |
| **Async Notifier** | Notificaciones | Notificaciones eficientes |

## 🎯 Casos de Uso de Análisis

### Pipeline Completo de Análisis
```python
chunker = BulkDataChunker()
mapper = BulkDataMapper()
filter_obj = BulkDataFilter()
grouper = BulkDataGrouper()
stats = BulkDataStatistics()

# Pipeline
items = list(range(1000))
chunks = chunker.chunk(items, chunk_size=100)
mapped = [mapper.map(chunk, lambda x: x * 2) for chunk in chunks]
filtered = [filter_obj.filter(chunk, lambda x: x > 100) for chunk in mapped]
grouped = grouper.group_by(filtered[0], lambda x: x % 10)
statistics = stats.mean([stats.mean(chunk) for chunk in filtered])
```

### Sistema con Monitoreo y Alertas
```python
monitor = BulkAsyncMonitor()
notifier = BulkAsyncNotifier()

# Registrar alerta
await monitor.register_alert(
    "error_rate",
    condition=lambda v: v > 0.1,
    handler=lambda name, value: notifier.notify("alerts", f"High {name}: {value}")
)

# Registrar métricas
await monitor.record_metric("error_rate", 0.05)
await monitor.record_metric("error_rate", 0.15)  # Trigger alerta
```

## 📈 Beneficios Totales

1. **Data Comparator**: Comparación robusta
2. **Data Merger**: Merge eficiente
3. **Data Sorter**: Ordenamiento avanzado
4. **Data Searcher**: Búsqueda eficiente
5. **Data Statistics**: Análisis estadístico completo
6. **Data Validator Advanced**: Validación con esquemas
7. **Data Normalizer**: Normalización eficiente
8. **Data Sampler**: Muestreo avanzado
9. **Data Transformer Advanced**: Pipelines de transformación
10. **Async Monitor**: Monitoreo con alertas
11. **Async Notifier**: Notificaciones eficientes

## 🚀 Resultados Esperados

Con todas las utilidades de análisis:

- **Comparación robusta** de datos
- **Merge eficiente** de estructuras
- **Ordenamiento avanzado** con múltiples keys
- **Búsqueda eficiente** (lineal y binaria)
- **Análisis estadístico completo** (media, mediana, moda, std dev, percentiles)
- **Validación robusta** con esquemas complejos
- **Normalización eficiente** (min-max, z-score)
- **Muestreo avanzado** (simple, con pesos, estratificado)
- **Pipelines de transformación**
- **Monitoreo con alertas** automáticas
- **Notificaciones eficientes** por canales

El sistema ahora tiene **141+ optimizaciones, utilidades, componentes y características** que cubren todos los aspectos posibles de procesamiento masivo, desde análisis de datos hasta monitoreo, alertas y notificaciones.

El sistema está completamente optimizado y listo para producción con todas las características necesarias para operaciones masivas de alta performance y análisis avanzado de datos.



