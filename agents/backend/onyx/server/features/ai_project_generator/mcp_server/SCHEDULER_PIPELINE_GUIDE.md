# Guía de Scheduler y Pipelines - MCP Server

## Resumen

Sistema de scheduling avanzado para tareas programadas y pipelines de transformación de datos composables.

## Scheduler

### Programar Tareas

```python
from mcp_server.utils.scheduler_utils import get_scheduler, schedule_task

scheduler = get_scheduler()

# Programar tarea con intervalo
scheduler.schedule(
    "cleanup",
    cleanup_task,
    "30m",  # Cada 30 minutos
    arg1, arg2
)

# Programar tarea con cron (simplificado)
scheduler.schedule(
    "backup",
    backup_task,
    "0 2 * * *",  # Diario a las 2 AM
)

# Usar decorador
@schedule_task("health_check", "5m")
def health_check():
    check_system_health()
```

### Gestionar Tareas

```python
# Habilitar/deshabilitar tarea
scheduler.enable("cleanup")
scheduler.disable("backup")

# Obtener información de tarea
task = scheduler.get_task("cleanup")
print(f"Run count: {task.run_count}")
print(f"Last run: {task.last_run}")
print(f"Next run: {task.next_run}")

# Listar todas las tareas
tasks = scheduler.list_tasks()

# Desprogramar tarea
scheduler.unschedule("cleanup")
```

### Iniciar/Detener Scheduler

```python
# Iniciar scheduler
scheduler.start()

# Detener scheduler
scheduler.stop()
```

### Formatos de Schedule

- **Intervalos simples**: `"30s"`, `"5m"`, `"1h"`, `"2d"`
- **Cron expressions**: `"0 2 * * *"` (diario a las 2 AM)

## Pipelines

### Pipeline Básico

```python
from mcp_server.utils.pipeline_utils import Pipeline

# Crear pipeline
pipeline = Pipeline("data_processing")

# Agregar etapas
pipeline.add_stage("validate", validate_data)
pipeline.add_stage("transform", transform_data)
pipeline.add_stage("normalize", normalize_data)

# Ejecutar pipeline
result = pipeline.execute(input_data)
```

### Pipeline con Manejo de Errores

```python
def handle_error(error: Exception, data: Any) -> Any:
    logger.error(f"Error in pipeline: {error}")
    return {"error": str(error), "data": data}

pipeline = Pipeline()
pipeline.add_stage(
    "risky_stage",
    risky_transformation,
    error_handler=handle_error,
    skip_on_error=True
)
```

### Pipeline Paralelo

```python
from mcp_server.utils.pipeline_utils import Pipeline, ParallelPipeline

# Crear pipeline base
pipeline = Pipeline()
pipeline.add_stage("validate", validate_data)
pipeline.add_stage("transform", transform_data)

# Ejecutar en paralelo
parallel = ParallelPipeline(pipeline, max_workers=4)
results = parallel.execute([item1, item2, item3, item4])
```

### Pipeline Condicional

```python
from mcp_server.utils.pipeline_utils import Pipeline, ConditionalPipeline

# Crear pipelines para diferentes casos
user_pipeline = Pipeline("user")
user_pipeline.add_stage("validate_user", validate_user_data)

admin_pipeline = Pipeline("admin")
admin_pipeline.add_stage("validate_admin", validate_admin_data)

# Crear pipeline condicional
conditional = ConditionalPipeline()
conditional.add_branch(
    lambda data: data.get("role") == "admin",
    admin_pipeline
)
conditional.set_default(user_pipeline)

# Ejecutar
result = conditional.execute(data)
```

### Etapas Predefinidas

```python
from mcp_server.utils.pipeline_utils import (
    map_stage, filter_stage, validate_stage
)

pipeline = Pipeline()

# Etapa de mapeo
pipeline.add_stage("map", map_stage("map", lambda x: x * 2))

# Etapa de filtrado
pipeline.add_stage("filter", filter_stage("filter", lambda x: x > 0))

# Etapa de validación
pipeline.add_stage("validate", validate_stage("validate", lambda x: x is not None))
```

## Ejemplos de Uso

### Scheduler con Tareas Periódicas

```python
from mcp_server.utils.scheduler_utils import get_scheduler

scheduler = get_scheduler()

# Limpieza diaria
@scheduler.schedule("daily_cleanup", "0 3 * * *")
def daily_cleanup():
    cleanup_old_data()
    cleanup_cache()

# Health check cada 5 minutos
@scheduler.schedule("health_check", "5m")
def health_check():
    check_database_health()
    check_api_health()

# Iniciar scheduler
scheduler.start()
```

### Pipeline de Procesamiento de Datos

```python
from mcp_server.utils.pipeline_utils import Pipeline

def process_user_data():
    pipeline = Pipeline("user_processing")
    
    # Validar
    pipeline.add_stage("validate", validate_user_input)
    
    # Transformar
    pipeline.add_stage("normalize", normalize_user_data)
    pipeline.add_stage("enrich", enrich_user_data)
    
    # Guardar
    pipeline.add_stage("save", save_user_data)
    
    return pipeline

# Usar pipeline
pipeline = process_user_data()
result = pipeline.execute(user_input)
```

### Pipeline con Transformaciones Complejas

```python
from mcp_server.utils.pipeline_utils import Pipeline, ParallelPipeline

# Pipeline de transformación
transform_pipeline = Pipeline("transform")
transform_pipeline.add_stage("parse", parse_data)
transform_pipeline.add_stage("validate", validate_data)
transform_pipeline.add_stage("enrich", enrich_data)
transform_pipeline.add_stage("format", format_data)

# Procesar múltiples items en paralelo
parallel = ParallelPipeline(transform_pipeline, max_workers=8)
results = parallel.execute(large_dataset)
```

## Próximos Pasos

1. Agregar soporte completo para cron expressions
2. Agregar persistencia de tareas programadas
3. Mejorar manejo de errores en pipelines
4. Agregar métricas de pipelines
5. Agregar visualización de pipelines

