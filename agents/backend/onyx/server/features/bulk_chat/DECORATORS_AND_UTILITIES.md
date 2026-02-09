# Decoradores y Utilidades Avanzadas

## Decoradores para Operaciones Bulk

### 1. **@track_metrics** - Tracking Automático de Métricas

Decorador para registrar automáticamente métricas de operaciones bulk.

#### Uso:

```python
from core.bulk_operations import track_metrics, BulkRealTimeMetrics

metrics = BulkRealTimeMetrics()

@track_metrics(metrics=metrics, operation_type="create_sessions")
async def create_sessions_bulk(count: int):
    # Tu código aquí
    return session_ids

# O sin especificar operation_type (usa el nombre de la función)
@track_metrics(metrics=metrics)
async def my_operation():
    pass
```

#### Características:
- Tracking automático de duración
- Detección de éxito/fallo
- Extracción automática de items procesados
- Soporte para funciones async y síncronas
- Metadata automática (nombre de función, argumentos)

#### Ejemplo con resultado BulkOperationResult:

```python
@track_metrics(metrics=metrics)
async def delete_sessions(session_ids: List[str]):
    result = await bulk_sessions.delete_sessions(session_ids)
    return result  # Automáticamente extrae result.processed
```

---

### 2. **@cache_result** - Caché Automático de Resultados

Decorador para cachear resultados de operaciones automáticamente.

#### Uso:

```python
from core.bulk_operations import cache_result, BulkAdvancedCache

cache = BulkAdvancedCache(max_size=1000, default_ttl=3600)

@cache_result(cache=cache, ttl=7200)
async def get_session_analytics(session_id: str):
    # Operación costosa
    return analytics_data

# Con función personalizada para generar clave
def custom_key(session_id: str, include_metadata: bool = False):
    return f"analytics:{session_id}:{include_metadata}"

@cache_result(cache=cache, key_func=custom_key, ttl=3600)
async def get_analytics(session_id: str, include_metadata: bool = False):
    return analytics_data
```

#### Características:
- TTL configurable por función
- Clave de caché automática basada en argumentos
- Soporte para clave personalizada
- Funciona con funciones async y síncronas

---

### 3. **@validate_input** - Validación Automática de Inputs

Decorador para validar inputs antes de ejecutar operaciones.

#### Uso:

```python
from core.bulk_operations import validate_input, BulkEnhancedValidator

validator = BulkEnhancedValidator()

# Añadir regla de validación
def validate_session_count(count: int):
    return 1 <= count <= 10000

validator.add_rule("create_sessions", lambda data: validate_session_count(data.get("count", 0)))

@validate_input(validator=validator, operation_type="create_sessions")
async def create_sessions(count: int, initial_message: str = None):
    # Solo se ejecuta si la validación pasa
    return await bulk_sessions.create_sessions(count)
```

#### Características:
- Validación antes de ejecución
- Caché de resultados de validación
- Lanza excepción si la validación falla
- Soporte para validadores async y síncronos

---

## Sistema de Benchmarking

### **BulkBenchmark**

Sistema completo para hacer benchmarks de operaciones y compararlas.

#### Uso Básico:

```python
from core.bulk_operations import BulkBenchmark

benchmark = BulkBenchmark()

# Benchmark de una operación
result = await benchmark.benchmark_operation(
    operation=my_async_function,
    operation_name="create_sessions",
    test_data=[100, 500, 1000],  # Diferentes inputs
    iterations=5,
    warmup=1
)

# Comparar dos operaciones
comparison = benchmark.compare_operations(
    "create_sessions_v1",
    "create_sessions_v2"
)
# Returns: {"faster": "...", "speedup": 1.5, ...}

# Resumen de todos los benchmarks
summary = benchmark.get_summary()
```

#### Métodos:
- `benchmark_operation()`: Ejecutar benchmark
- `compare_operations()`: Comparar dos operaciones
- `get_summary()`: Resumen de todos los benchmarks

---

## Auto-Tuning de Parámetros

### **BulkAutoTuner**

Sistema inteligente para auto-ajustar parámetros de rendimiento.

#### Auto-Tuning de Batch Size:

```python
from core.bulk_operations import BulkAutoTuner, BulkRealTimeMetrics

tuner = BulkAutoTuner(
    metrics=BulkRealTimeMetrics(),
    benchmark=BulkBenchmark()
)

# Auto-ajustar batch size
result = await tuner.tune_batch_size(
    operation=my_operation,
    test_data=test_dataset,
    min_batch=10,
    max_batch=1000,
    step=50,
    iterations=3
)

print(f"Mejor batch size: {result['best_batch_size']}")
print(f"Tiempo: {result['best_time']}s")
```

#### Auto-Tuning de Workers:

```python
# Auto-ajustar número de workers
result = await tuner.tune_workers(
    operation=my_operation,
    test_data=test_dataset,
    min_workers=1,
    max_workers=50,
    step=5
)

print(f"Mejor número de workers: {result['best_workers']}")
print(f"Throughput: {result['best_throughput']} ops/sec")
```

#### Obtener Recomendaciones:

```python
recommendations = tuner.get_tuning_recommendations()
# Returns: {
#   "recommendations": {"batch_size": 200, "workers": 15},
#   "based_on": 3,
#   "last_tuning": "2024-01-01T12:00:00"
# }
```

---

## Ejemplos de Integración

### Ejemplo Completo: Operación con Todos los Decoradores

```python
from core.bulk_operations import (
    track_metrics, cache_result, validate_input,
    BulkRealTimeMetrics, BulkAdvancedCache, BulkEnhancedValidator
)

# Inicializar componentes
metrics = BulkRealTimeMetrics()
cache = BulkAdvancedCache()
validator = BulkEnhancedValidator()

# Configurar validación
validator.add_rule("export_sessions", lambda data: "session_ids" in data and len(data["session_ids"]) > 0)

# Función con todos los decoradores
@track_metrics(metrics=metrics, operation_type="export_sessions")
@cache_result(cache=cache, ttl=3600)
@validate_input(validator=validator, operation_type="export_sessions")
async def export_sessions_optimized(session_ids: List[str], format: str = "json"):
    """
    Operación optimizada con:
    - Tracking automático de métricas
    - Caché de resultados
    - Validación de inputs
    """
    return await bulk_exporter.export_sessions(session_ids, format)
```

### Ejemplo: Benchmarking y Auto-Tuning

```python
from core.bulk_operations import BulkBenchmark, BulkAutoTuner

benchmark = BulkBenchmark()
tuner = BulkAutoTuner(benchmark=benchmark)

# 1. Hacer benchmark de operación actual
current_result = await benchmark.benchmark_operation(
    operation=current_implementation,
    operation_name="current",
    test_data=test_data,
    iterations=5
)

# 2. Auto-ajustar parámetros
tuning_result = await tuner.tune_batch_size(
    operation=current_implementation,
    test_data=test_data
)

# 3. Usar parámetros optimizados
optimized_operation = create_optimized_version(
    batch_size=tuning_result["best_batch_size"]
)

# 4. Benchmark de versión optimizada
optimized_result = await benchmark.benchmark_operation(
    operation=optimized_operation,
    operation_name="optimized",
    test_data=test_data,
    iterations=5
)

# 5. Comparar
comparison = benchmark.compare_operations("current", "optimized")
print(f"Mejora: {comparison['speedup']}x más rápido")
```

---

## Endpoints API

### Benchmarking:
- `POST /api/v1/bulk/benchmark/run` - Ejecutar benchmark
- `GET /api/v1/bulk/benchmark/summary` - Resumen de benchmarks
- `POST /api/v1/bulk/benchmark/compare` - Comparar operaciones

### Auto-Tuning:
- `POST /api/v1/bulk/autotune/batch-size` - Auto-ajustar batch size
- `POST /api/v1/bulk/autotune/workers` - Auto-ajustar workers
- `GET /api/v1/bulk/autotune/recommendations` - Obtener recomendaciones

---

## Beneficios

1. **Automatización**: Decoradores eliminan código repetitivo
2. **Performance**: Caché automático mejora tiempos de respuesta
3. **Observabilidad**: Tracking automático de métricas
4. **Validación**: Validación automática previene errores
5. **Optimización**: Auto-tuning encuentra parámetros óptimos
6. **Comparación**: Benchmarking permite comparar implementaciones

---

## Mejores Prácticas

1. **Usar decoradores en orden**: validate → cache → track
2. **Configurar TTL apropiado**: Más largo para datos estables
3. **Validar datos críticos**: Usar validación en operaciones importantes
4. **Benchmark regularmente**: Medir cambios de rendimiento
5. **Auto-tuning periódico**: Ajustar parámetros según carga
6. **Monitorear métricas**: Usar dashboard para ver métricas en tiempo real
















