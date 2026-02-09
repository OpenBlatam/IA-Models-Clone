# Guía de Migración de Pipelines

## 📋 Resumen

Esta guía ayuda a migrar código que usa el wrapper de compatibilidad legacy (`core.architecture.pipelines`) al nuevo sistema modular (`core.architecture.pipelines.*`).

## 🎯 ¿Por qué migrar?

### Beneficios de la migración:
1. **Mejor rendimiento**: Sin overhead del wrapper
2. **Acceso a funcionalidades avanzadas**: Nuevas características solo disponibles en el sistema modular
3. **Mejor soporte de tipos**: Type hints más precisos
4. **Compatibilidad futura**: El wrapper será deprecado

## 🔍 Identificar código legacy

### Patrón de import legacy:
```python
# ❌ Legacy (deprecated)
from core.architecture.pipelines import Pipeline, PipelineStage
```

### Patrón de import nuevo:
```python
# ✅ Nuevo (recomendado)
from core.architecture.pipelines.pipeline import Pipeline
from core.architecture.pipelines.stages import PipelineStage
```

## 🛠️ Herramientas de migración

### 1. Verificar estado del módulo

```python
from core.architecture.pipelines import check_compatibility

report = check_compatibility()
print(report)
```

### 2. Obtener guía de migración

```python
from core.architecture.pipelines import get_migration_guide

guide = get_migration_guide()
print(guide["migration_steps"])
```

### 3. Analizar proyecto completo

```python
from pathlib import Path
from core.architecture.pipelines_migration_helper import analyze_project_for_migration

project_root = Path(".")
report = analyze_project_for_migration(project_root)
print(f"Files with legacy code: {report['files_with_legacy']}")
```

### 4. Generar reporte de migración

```python
from pathlib import Path
from core.architecture.pipelines_migration_helper import generate_migration_report

project_root = Path(".")
report = generate_migration_report(project_root, output_file=Path("migration_report.txt"))
print(report)
```

## 📝 Mapeo de imports

### Clases principales

| Legacy | Nuevo |
|--------|-------|
| `Pipeline` | `pipelines.pipeline.Pipeline` |
| `PipelineStage` | `pipelines.stages.PipelineStage` |
| `FunctionStage` | `pipelines.stages.FunctionStage` |

### Ejecutores

| Legacy | Nuevo |
|--------|-------|
| `SequentialExecutor` | `pipelines.executors.SequentialExecutor` |
| `ParallelExecutor` | `pipelines.executors.ParallelExecutor` |
| `ConditionalExecutor` | `pipelines.executors.ConditionalExecutor` |
| `ParallelPipeline` (alias) | `pipelines.executors.ParallelExecutor` |
| `ConditionalPipeline` (alias) | `pipelines.executors.ConditionalExecutor` |

### Middleware

| Legacy | Nuevo |
|--------|-------|
| `LoggingMiddleware` | `pipelines.middleware.LoggingMiddleware` |
| `MetricsMiddleware` | `pipelines.middleware.MetricsMiddleware` |
| `CachingMiddleware` | `pipelines.middleware.CachingMiddleware` |
| `RetryMiddleware` | `pipelines.middleware.RetryMiddleware` |
| `ValidationMiddleware` | `pipelines.middleware.ValidationMiddleware` |

### Builders

| Legacy | Nuevo |
|--------|-------|
| `PipelineBuilder` | `pipelines.builders.PipelineBuilder` |
| `PipelineFactory` | `pipelines.builders.PipelineFactory` |
| `PipelineRegistry` | `pipelines.builders.PipelineRegistry` |

## 🔄 Ejemplos de migración

### Ejemplo 1: Import simple

**Antes:**
```python
from core.architecture.pipelines import Pipeline, PipelineStage

def create_pipeline():
    stage = PipelineStage(name="test")
    pipeline = Pipeline(stages=[stage])
    return pipeline
```

**Después:**
```python
from core.architecture.pipelines.pipeline import Pipeline
from core.architecture.pipelines.stages import PipelineStage

def create_pipeline():
    stage = PipelineStage(name="test")
    pipeline = Pipeline(stages=[stage])
    return pipeline
```

### Ejemplo 2: Uso de alias

**Antes:**
```python
from core.architecture.pipelines import ParallelPipeline

executor = ParallelPipeline()
```

**Después:**
```python
from core.architecture.pipelines.executors import ParallelExecutor

executor = ParallelExecutor()
```

### Ejemplo 3: Múltiples imports

**Antes:**
```python
from core.architecture.pipelines import (
    Pipeline,
    PipelineBuilder,
    LoggingMiddleware,
    MetricsMiddleware
)
```

**Después:**
```python
from core.architecture.pipelines.pipeline import Pipeline
from core.architecture.pipelines.builders import PipelineBuilder
from core.architecture.pipelines.middleware import (
    LoggingMiddleware,
    MetricsMiddleware
)
```

## ✅ Checklist de migración

- [ ] Identificar todos los archivos con imports legacy
- [ ] Reemplazar imports según el mapeo
- [ ] Actualizar usos de alias (ParallelPipeline → ParallelExecutor)
- [ ] Ejecutar tests para verificar compatibilidad
- [ ] Revisar advertencias de deprecación
- [ ] Actualizar documentación
- [ ] Remover imports del wrapper legacy

## 🚀 Comandos útiles

### Analizar proyecto
```bash
python -m core.architecture.pipelines_migration_helper /path/to/project
```

### Verificar compatibilidad
```python
python -c "from core.architecture.pipelines import check_compatibility; print(check_compatibility())"
```

## 📚 Recursos adicionales

- Documentación del sistema modular: `pipelines/README.md`
- Tests de compatibilidad: `tests/test_architecture/test_pipelines_compatibility.py`
- Helper de migración: `pipelines_migration_helper.py`

## ⚠️ Notas importantes

1. El wrapper legacy seguirá funcionando durante el período de transición
2. Se emitirán advertencias de deprecación al usar el wrapper
3. Se recomienda migrar gradualmente, archivo por archivo
4. Siempre ejecutar tests después de migrar cada archivo

---

**Última actualización**: 2024

