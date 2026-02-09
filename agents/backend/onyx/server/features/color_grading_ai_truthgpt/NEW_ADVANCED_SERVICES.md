# Nuevos Servicios Avanzados - Color Grading AI TruthGPT

## Resumen

Agregados 4 servicios avanzados para mejorar la arquitectura y capacidades del sistema.

## Nuevos Servicios

### 1. Service Orchestrator (`service_orchestrator.py`)

**Propósito:** Orquestar múltiples servicios para ejecutar workflows complejos.

**Características:**
- Orquestación multi-servicio
- Gestión de dependencias
- Ejecución paralela y secuencial
- Ejecución condicional
- Manejo de errores
- Agregación de resultados
- Reintentos automáticos

**Estrategias:**
- `SEQUENTIAL`: Ejecución en orden
- `PARALLEL`: Ejecución en paralelo
- `CONDITIONAL`: Ejecución basada en condiciones
- `PIPELINE`: Ejecución tipo pipeline

**Uso:**
```python
from services.service_orchestrator import ServiceOrchestrator, ServiceTask, OrchestrationStrategy

orchestrator = ServiceOrchestrator(services=all_services)

tasks = [
    ServiceTask(
        service_name="color_analyzer",
        method_name="analyze_image",
        parameters={"image_path": "input.jpg"}
    ),
    ServiceTask(
        service_name="color_matcher",
        method_name="match_colors",
        dependencies=["color_analyzer"],
        parameters={"reference": "cinematic"}
    )
]

result = await orchestrator.orchestrate(tasks, OrchestrationStrategy.SEQUENTIAL)
```

**Ubicación:** `services/service_orchestrator.py`

### 2. Event Scheduler (`event_scheduler.py`)

**Propósito:** Programar eventos y tareas para ejecución automática.

**Características:**
- Programación one-time
- Programación por intervalos
- Programación tipo cron
- Programación diaria/semanal
- Gestión de eventos
- Seguimiento de ejecución
- Límite de ejecuciones

**Tipos de Programación:**
- `ONCE`: Ejecutar una vez
- `INTERVAL`: Ejecutar a intervalos
- `CRON`: Programación tipo cron
- `DAILY`: Diario a hora específica
- `WEEKLY`: Semanal en día específico

**Uso:**
```python
from services.event_scheduler import EventScheduler, ScheduleType
from datetime import timedelta

scheduler = EventScheduler()
await scheduler.start()

# Programar evento único
event_id = await scheduler.schedule_once(
    name="cleanup_cache",
    handler=cleanup_function,
    run_at=datetime.now() + timedelta(hours=1)
)

# Programar evento periódico
event_id = await scheduler.schedule_interval(
    name="daily_backup",
    handler=backup_function,
    interval=timedelta(days=1)
)

# Programar evento diario
event_id = await scheduler.schedule_daily(
    name="morning_report",
    handler=report_function,
    time="09:00"
)
```

**Ubicación:** `services/event_scheduler.py`

### 3. API Gateway (`api_gateway.py`)

**Propósito:** Gateway unificado para routing, autenticación y gestión de requests.

**Características:**
- Gestión de rutas
- Routing de requests
- Soporte para middleware
- Autenticación
- Rate limiting
- Transformación de requests/responses
- Manejo de errores
- Timeouts

**Uso:**
```python
from services.api_gateway import APIGateway, RouteMethod, RequestContext

gateway = APIGateway()

# Registrar ruta
gateway.register_route(
    path="/api/v1/color-grade",
    method=RouteMethod.POST,
    handler=color_grade_handler,
    auth_required=True,
    rate_limit=10
)

# Manejar request
context = RequestContext(
    path="/api/v1/color-grade",
    method=RouteMethod.POST,
    body={"image": "path/to/image.jpg"}
)

response = await gateway.handle_request(
    path="/api/v1/color-grade",
    method=RouteMethod.POST,
    context=context
)
```

**Ubicación:** `services/api_gateway.py`

### 4. Data Pipeline (`data_pipeline.py`)

**Propósito:** Sistema avanzado de pipelines de procesamiento de datos.

**Características:**
- Procesamiento multi-etapa
- Gestión de dependencias
- Manejo de errores
- Lógica de reintentos
- Procesamiento paralelo
- Agregación de resultados

**Etapas:**
- `INPUT`: Entrada de datos
- `TRANSFORM`: Transformación
- `PROCESS`: Procesamiento
- `VALIDATE`: Validación
- `OUTPUT`: Salida

**Uso:**
```python
from services.data_pipeline import DataPipeline, PipelineStage

pipeline = DataPipeline(name="color_grading_pipeline")

# Agregar pasos
pipeline.add_step(
    name="load_image",
    stage=PipelineStage.INPUT,
    processor=load_image_function
)

pipeline.add_step(
    name="analyze_colors",
    stage=PipelineStage.PROCESS,
    processor=analyze_colors_function,
    dependencies=["load_image"]
)

pipeline.add_step(
    name="apply_grading",
    stage=PipelineStage.PROCESS,
    processor=apply_grading_function,
    dependencies=["analyze_colors"]
)

# Ejecutar pipeline
result = await pipeline.execute(input_data="image.jpg", parallel=True)
```

**Ubicación:** `services/data_pipeline.py`

## Integración

Todos los servicios están integrados en:
- `RefactoredServiceFactory` - Inicialización automática en `_init_advanced()`
- `services/__init__.py` - Exports disponibles
- Categoría "advanced" - Nueva categoría en el factory

## Estadísticas Actualizadas

- **Servicios totales**: 70+
- **Nuevos servicios**: 4
- **Categorías**: 13
- **Sin errores de linter**

## Beneficios

1. **Orquestación Compleja**: Ejecutar workflows multi-servicio fácilmente
2. **Automatización**: Programar tareas y eventos automáticamente
3. **API Unificada**: Gateway centralizado para todas las APIs
4. **Procesamiento de Datos**: Pipelines flexibles y potentes
5. **Arquitectura Mejorada**: Servicios más organizados y escalables

## Próximos Pasos

1. Integrar Event Scheduler con otros servicios
2. Configurar rutas en API Gateway
3. Crear pipelines personalizados para casos de uso específicos
4. Usar Service Orchestrator para workflows complejos


