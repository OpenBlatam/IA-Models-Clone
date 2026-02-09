# Pipelines Module - Sistema Modular (optimizado)

Sistema modular de pipelines para procesamiento de datos y tareas en robots.

## Estructura Modular

```
pipelines/
├── __init__.py          # Exports principales
├── base.py              # Clases base y interfaces
├── sequential.py        # Pipeline secuencial
├── parallel.py          # Pipeline paralelo
├── conditional.py       # Pipeline condicional
├── stages.py            # Etapas comunes reutilizables
├── middleware.py       # Middleware para pipelines
├── factory.py           # Factory para crear pipelines
├── monitoring.py        # Sistema de monitoreo
└── README.md           # Esta documentación
```

## Características

- ✅ **Modular**: Cada tipo de pipeline en su propio módulo
- ✅ **Extensible**: Fácil agregar nuevos tipos de pipelines
- ✅ **Reutilizable**: Etapas y middleware comunes
- ✅ **Observable**: Sistema de monitoreo integrado
- ✅ **Type-safe**: Type hints completos
- ✅ **Documentado**: Documentación exhaustiva

## Uso Básico

### Pipeline Secuencial

```python
from pipelines import SequentialPipeline, TransformStage

# Crear pipeline
pipeline = SequentialPipeline("data_processing")

# Agregar etapas
pipeline.add_stage(TransformStage("normalize", normalize_data))
pipeline.add_stage(TransformStage("validate", validate_data))

# Ejecutar
result = await pipeline.execute(input_data)
```

### Pipeline Paralelo

```python
from pipelines import ParallelPipeline, TransformStage

# Crear pipeline paralelo
pipeline = ParallelPipeline("parallel_processing")

# Agregar etapas que se ejecutarán en paralelo
pipeline.add_stage(TransformStage("process_a", process_a))
pipeline.add_stage(TransformStage("process_b", process_b))

# Ejecutar
result = await pipeline.execute(input_data)
```

### Pipeline Condicional

```python
from pipelines import ConditionalPipeline, ConditionalStage, TransformStage

# Crear pipeline condicional
pipeline = ConditionalPipeline("conditional_processing")

# Agregar etapa condicional
pipeline.add_stage(
    ConditionalStage(
        "conditional_transform",
        TransformStage("transform", transform_data),
        condition=lambda data, ctx: data.get("type") == "special"
    )
)

# Ejecutar
result = await pipeline.execute(input_data)
```

### Usando Factory

```python
from pipelines import PipelineFactory, TransformStage

# Crear con factory
pipeline = PipelineFactory.create_sequential(
    name="my_pipeline",
    stages=[
        TransformStage("stage1", func1),
        TransformStage("stage2", func2)
    ]
)

result = await pipeline.execute(input_data)
```

### Con Middleware

```python
from pipelines import SequentialPipeline, middleware

pipeline = SequentialPipeline("with_middleware")

# Agregar middleware de logging
pipeline.add_middleware(middleware.logging_middleware())

# Agregar middleware de timing
pipeline.add_middleware(middleware.timing_middleware)

# Ejecutar
result = await pipeline.execute(input_data)
```

### Monitoreo

```python
from pipelines import get_monitor

monitor = get_monitor()

# Obtener métricas
metrics = monitor.get_metrics("my_pipeline")
print(f"Success rate: {metrics['success_rate']}")
print(f"Average time: {metrics['average_time']}s")

# Obtener historial
history = monitor.get_recent_executions(limit=10)
```

## Etapas Comunes

### TransformStage
Aplica una transformación a los datos.

### FilterStage
Filtra datos basado en una condición.

### ValidationStage
Valida datos y lanza excepción si no son válidos.

### LoggingStage
Registra información sin modificar datos.

### ErrorHandlingStage
Maneja errores con estrategias configurables.

## Middleware Disponible

- `timing_middleware`: Mide tiempo de ejecución
- `logging_middleware()`: Factory para logging
- `caching_middleware()`: Factory para caching
- `error_handling_middleware()`: Factory para manejo de errores
- `validation_middleware()`: Factory para validación
- `transformation_middleware()`: Factory para transformación

## Extensión

Para crear etapas personalizadas:

```python
from pipelines import PipelineStage, PipelineContext

class MyCustomStage(PipelineStage):
    async def execute(self, input_data, context):
        # Tu lógica aquí
        return processed_data
```

Para crear pipelines personalizados:

```python
from pipelines import BasePipeline, PipelineResult

class MyCustomPipeline(BasePipeline):
    async def execute(self, input_data):
        # Tu lógica aquí
        return PipelineResult(success=True, data=result)
```
