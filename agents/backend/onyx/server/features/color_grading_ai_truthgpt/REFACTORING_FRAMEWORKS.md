# Refactorización de Frameworks - Color Grading AI TruthGPT

## Resumen

Refactorización para crear frameworks unificados: validación y performance tracking.

## Nuevos Frameworks

### 1. Validation Framework

**Archivo**: `core/validation_framework.py`

**Características**:
- ✅ Rule-based validation
- ✅ Custom validators
- ✅ Error aggregation
- ✅ Type checking
- ✅ Range validation
- ✅ Pattern matching
- ✅ Enum validation
- ✅ Length validation

**Tipos de Reglas**:
- REQUIRED: Campo requerido
- TYPE: Validación de tipo
- RANGE: Validación de rango
- PATTERN: Validación de patrón (regex)
- ENUM: Validación de valores permitidos
- LENGTH: Validación de longitud
- CUSTOM: Validador personalizado

**Uso**:
```python
from core import ValidationFramework, ValidationRuleType

# Crear framework
framework = ValidationFramework()

# Crear schema
framework.create_schema("color_params", [
    {
        "field_name": "brightness",
        "rule_type": ValidationRuleType.RANGE,
        "params": {"min": -1.0, "max": 1.0},
        "error_message": "Brightness must be between -1.0 and 1.0"
    },
    {
        "field_name": "contrast",
        "rule_type": ValidationRuleType.RANGE,
        "params": {"min": 0.0, "max": 3.0},
    },
])

# Validar
try:
    validated = framework.validate("color_params", {
        "brightness": 0.5,
        "contrast": 1.2
    })
except InvalidParametersError as e:
    # Manejar error
    pass

# O sin excepciones
validated = framework.validate("color_params", data, raise_on_error=False)
```

### 2. Performance Tracker

**Archivo**: `core/performance_tracker.py`

**Características**:
- ✅ Metric collection
- ✅ Time-series tracking
- ✅ Aggregations (mean, min, max, sum)
- ✅ Percentiles (p50, p95, p99)
- ✅ Snapshots
- ✅ Timing context manager

**Uso**:
```python
from core import PerformanceTracker

# Crear tracker
tracker = PerformanceTracker()

# Registrar métricas
tracker.record_metric("processing_time", 2.5, unit="s")
tracker.record_metric("memory_usage", 512, unit="MB")

# Timing con context manager
with tracker.time_operation("video_processing"):
    await process_video()

# O manual
tracker.record_timing("image_processing", 1.2)

# Estadísticas
stats = tracker.get_metric_stats("processing_time", window_seconds=3600)
# {
#     "count": 100,
#     "mean": 2.3,
#     "min": 1.0,
#     "max": 5.0,
#     "p50": 2.2,
#     "p95": 4.0,
#     "p99": 4.8
# }

# Snapshots
snapshot = tracker.create_snapshot(tags={"environment": "production"})
snapshots = tracker.get_snapshots(
    start_date=datetime.now() - timedelta(days=1),
    limit=100
)

# Summary
summary = tracker.get_summary()
```

## Integración

### Validation + Error Handler

```python
# Integrar validación con error handling
framework = ValidationFramework()
error_handler = ErrorHandler()

def validate_with_error_handling(schema_name, data):
    try:
        return framework.validate(schema_name, data)
    except InvalidParametersError as e:
        error_context = ErrorContext(operation="validation")
        error_handler.handle_error(e, error_context)
        raise
```

### Performance + Metrics

```python
# Integrar performance tracker con metrics collector
tracker = PerformanceTracker()
metrics_collector = MetricsCollector()

# Registrar en ambos
def record_metric(name, value):
    tracker.record_metric(name, value)
    metrics_collector.record_metric(name, value)
```

## Beneficios

### Consistencia
- ✅ Validación unificada
- ✅ Performance tracking estandarizado
- ✅ Reglas reutilizables
- ✅ Métricas consistentes

### Observabilidad
- ✅ Métricas detalladas
- ✅ Percentiles
- ✅ Snapshots
- ✅ Time-series

### Mantenibilidad
- ✅ Frameworks reutilizables
- ✅ Fácil agregar reglas
- ✅ Validación declarativa
- ✅ Performance tracking automático

## Estadísticas

- **Nuevos frameworks**: 2 (Validation, Performance)
- **Tipos de reglas**: 7
- **Consistencia**: Mejorada significativamente
- **Observabilidad**: Mejorada

## Conclusión

La refactorización de frameworks proporciona:
- ✅ Validation framework unificado y declarativo
- ✅ Performance tracker con percentiles y snapshots
- ✅ Reglas reutilizables
- ✅ Métricas consistentes
- ✅ Observabilidad mejorada

**Los frameworks están ahora completamente unificados y listos para uso enterprise.**




