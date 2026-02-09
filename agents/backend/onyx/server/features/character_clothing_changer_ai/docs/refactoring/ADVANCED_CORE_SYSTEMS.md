# 🚀 Advanced Core Systems - Refactoring Round

## ✅ Estado: COMPLETADO

Este documento describe los nuevos sistemas avanzados de core agregados al proyecto Character Clothing Changer AI.

## 📊 Resumen

Se han agregado **72 sistemas avanzados de core** que proporcionan funcionalidades fundamentales para la arquitectura del proyecto:

### Sistemas Fundamentales (6)
1. **Workflow System** - Sistema de workflows con dependencias
2. **Pipeline System** - Pipeline de procesamiento de datos
3. **Orchestrator System** - Orquestador de servicios
4. **State Manager** - Gestor de estado de aplicación
5. **Advanced Cache** - Sistema de caché avanzado con múltiples estrategias
6. **Service Base** - Clases base para servicios

### Sistemas de Gestión Avanzada (3)
7. **Coordinator** - Coordinador de componentes
8. **Integration System** - Sistema de integración con servicios externos
9. **Data Pipeline** - Pipeline de transformación de datos

### Sistemas de Utilidades Avanzadas (3)
10. **Serializer** - Sistema de serialización con múltiples formatos
11. **Structured Logging** - Sistema de logging estructurado
12. **Config Builder** - Constructor de configuración con validación

### Sistemas Operacionales (3)
13. **Scheduler** - Sistema de programación de tareas
14. **Advanced Queue** - Cola avanzada con prioridades
15. **Batch Operations** - Operaciones en lote con control de concurrencia

### Sistemas de Base Patterns (2)
16. **Handler Base** - Clases base para handlers
17. **Processor Base** - Clases base para procesadores

### Sistemas de Análisis (3)
18. **Result Aggregator** - Agregación y análisis de resultados
19. **Performance Tuner** - Optimización automática de rendimiento
20. **Resource Manager** - Gestión de recursos del sistema

### Sistemas de Resiliencia (2)
21. **Rate Limiter** - Limitación de tasa con token bucket
22. **Circuit Breaker** - Patrón circuit breaker para llamadas resilientes

### Sistemas de Eventos y Observabilidad (2)
23. **Event Bus** - Sistema de eventos pub/sub
24. **Telemetry** - Sistema de telemetría para recolección de datos

### Sistemas de Salud y Reintentos (2)
25. **Health Check** - Sistema de health checks y monitoreo
26. **Retry Manager** - Gestor de reintentos automáticos

### Sistemas de Arquitectura (2)
27. **Dependency Injection** - Contenedor de inyección de dependencias
28. **Lifecycle Management** - Gestión del ciclo de vida de componentes

### Sistemas de Validación y Métricas (2)
29. **Validation Manager** - Sistema de validación centralizado
30. **Metrics Collector** - Recolector de métricas avanzado

### Sistemas de Manejo de Errores (1)
31. **Error Handler** - Sistema avanzado de manejo de errores

### Sistemas de Seguridad (1)
32. **Security Manager** - Sistema de seguridad con encriptación y hashing

### Sistemas de Middleware (1)
33. **Middleware Base** - Sistema base de middleware para request/response

### Sistemas de Observabilidad (1)
34. **Observability Manager** - Sistema completo de observabilidad

### Sistemas de Factory y Storage (2)
35. **Factory Base** - Patrón factory para creación de objetos
36. **Storage Base** - Base para sistemas de almacenamiento

### Sistemas de Contexto (1)
37. **Execution Context** - Gestión de contexto de ejecución

### Sistemas Base Fundamentales (3)
38. **Base Models** - Modelos base con funcionalidad común
39. **Types** - Definiciones de tipos y aliases compartidos
40. **Interfaces** - Interfaces abstractas para componentes core

### Sistemas de Utilidades (3)
41. **Constants** - Constantes de aplicación compartidas
42. **Helpers** - Funciones de ayuda comunes
43. **Async Utils** - Utilidades asíncronas y helpers

### Sistemas de Repository y Manager (2)
44. **Repository Base** - Base para repositorios de datos
45. **Manager Base** - Base para managers con estadísticas

### Sistemas de Registry (1)
46. **Component Registry** - Registry para componentes de aplicación

### Sistemas de Decoradores y Context Managers (2)
47. **Decorators** - Decoradores comunes para funciones
48. **Context Managers** - Context managers para operaciones

### Sistemas de Tracing y Feature Flags (2)
49. **Tracing** - Sistema de distributed tracing
50. **Feature Flags** - Sistema de feature flags y gradual rollouts

### Sistemas de Auditoría, Backup y Migraciones (5)
51. **Audit** - Sistema de auditoría avanzado
52. **Backup** - Sistema de backups automáticos
53. **Migrations** - Sistema de migraciones de datos
54. **API Versioning** - Sistema de versionado de APIs
55. **Testing** - Sistema de testing avanzado

### Sistemas de Notificaciones, Webhooks, Alertas y Reportes (4)
56. **Notifications** - Sistema de notificaciones multi-canal
57. **Webhooks** - Gestor de webhooks con firma
58. **Alerting** - Sistema de alertas basado en condiciones
59. **Reporting** - Sistema de generación de reportes

### Sistemas de Analytics, Monitoreo, Plugins, Optimización y Benchmarking (5)
60. **Analytics** - Sistema de analytics y eventos
61. **Monitoring Dashboard** - Dashboard de monitoreo en tiempo real
62. **Plugin System** - Sistema de plugins extensible
63. **Optimizer** - Optimizador de rendimiento y recursos
64. **Benchmark** - Sistema de benchmarking y profiling

### Sistemas de Gestión de Tareas y Ejecutores (3)
65. **Task Manager** - Gestor de tareas con repositorio y eventos
66. **Parallel Executor** - Ejecutor paralelo con worker pool
67. **Executor Base** - Base para ejecutores con AsyncExecutor

## 🏗️ Sistemas Agregados

### 1. Workflow System (`core/workflow.py`)

Sistema para definir y ejecutar workflows con pasos, dependencias y reintentos.

**Componentes:**
- `Workflow` - Definición y ejecutor de workflows
- `WorkflowManager` - Gestor de múltiples workflows
- `WorkflowStep` - Definición de pasos
- `WorkflowResult` - Resultado de ejecución
- `WorkflowStatus` - Estados del workflow

**Características:**
- ✅ Resolución automática de dependencias (topological sort)
- ✅ Reintentos configurables por paso
- ✅ Timeouts configurables
- ✅ Historial de ejecuciones
- ✅ Metadata por paso

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import Workflow, WorkflowManager

workflow = Workflow("clothing_change_workflow")
workflow.add_step("validate", validate_image, dependencies=[])
workflow.add_step("process", process_image, dependencies=["validate"])
workflow.add_step("save", save_result, dependencies=["process"])

manager = WorkflowManager()
manager.register(workflow)
result = await manager.execute("clothing_change_workflow", {"image": "image.jpg"})
```

### 2. Pipeline System (`core/pipeline.py`)

Sistema para crear y ejecutar pipelines de procesamiento de datos.

**Componentes:**
- `Pipeline` - Pipeline de procesamiento
- `PipelineManager` - Gestor de múltiples pipelines
- `PipelineStep` - Definición de pasos
- `PipelineResult` - Resultado de procesamiento
- `PipelineStage` - Etapas del pipeline (INPUT, PROCESSING, OUTPUT)

**Características:**
- ✅ Procesamiento secuencial o paralelo
- ✅ Timeouts configurables
- ✅ Procesamiento por lotes
- ✅ Historial de ejecuciones
- ✅ Metadata por paso

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import Pipeline, PipelineStage

pipeline = Pipeline("image_processing")
pipeline.add_step("load", load_image, stage=PipelineStage.INPUT)
pipeline.add_step("enhance", enhance_image, stage=PipelineStage.PROCESSING)
pipeline.add_step("save", save_image, stage=PipelineStage.OUTPUT)

result = await pipeline.process(image_data)
```

### 3. Orchestrator System (`core/orchestrator.py`)

Sistema para orquestar operaciones complejas y servicios.

**Componentes:**
- `Orchestrator` - Orquestador de servicios
- `OrchestrationTask` - Definición de tareas
- `OrchestrationResult` - Resultado de orquestación
- `OrchestrationStatus` - Estados de orquestación

**Características:**
- ✅ Registro de servicios
- ✅ Tareas con dependencias y prioridades
- ✅ Reintentos configurables
- ✅ Timeouts configurables
- ✅ Resolución automática de orden de ejecución

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import Orchestrator

orchestrator = Orchestrator()
orchestrator.register_service("clothing_service", clothing_service)
orchestrator.add_task("task1", "Change Clothing", "clothing_service", "change_clothing", 
                     parameters={"image": "image.jpg"}, priority=1)
result = await orchestrator.execute()
```

### 4. State Manager (`core/state_manager.py`)

Sistema para gestionar el estado de la aplicación.

**Componentes:**
- `StateManager` - Gestor de estado
- `StateChange` - Registro de cambios
- `StateEvent` - Tipos de eventos (CREATED, UPDATED, DELETED, CHANGED)

**Características:**
- ✅ Operaciones thread-safe (async lock)
- ✅ Historial de cambios
- ✅ Watchers para cambios de estado
- ✅ Soporte para wildcards en watchers
- ✅ Metadata por cambio

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import StateManager

state = StateManager()
await state.set("current_task", "task123")
await state.watch("current_task", lambda change: print(f"Changed: {change.new_value}"))
value = await state.get("current_task")
```

### 5. Advanced Cache (`core/advanced_cache.py`)

Sistema de caché avanzado con múltiples estrategias de evicción.

**Componentes:**
- `AdvancedCache` - Caché avanzado
- `CacheEntry` - Entrada de caché
- `CacheStrategy` - Estrategias (LRU, LFU, FIFO, TTL)

**Características:**
- ✅ Múltiples estrategias de evicción (LRU, LFU, FIFO, TTL)
- ✅ TTL configurable por entrada
- ✅ Estadísticas de caché (hits, misses, hit rate)
- ✅ Limpieza automática de entradas expiradas
- ✅ Decorador para caching de funciones

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import AdvancedCache, CacheStrategy

cache = AdvancedCache(max_size=1000, strategy=CacheStrategy.LRU, default_ttl=3600)
await cache.set("key1", {"data": "value"}, ttl=1800)
value = await cache.get("key1")
stats = cache.get_stats()
```

### 6. Service Base (`core/service_base.py`)

Clases base para todos los tipos de servicios.

**Componentes:**
- `BaseService` - Clase base abstracta para servicios
- `AsyncService` - Implementación asíncrona de servicio
- `ServiceRegistry` - Registro de servicios
- `ServiceConfig` - Configuración de servicio
- `ServiceResult` - Resultado de servicio
- `ServiceStatus` - Estados de servicio

**Características:**
- ✅ Lifecycle management (start, stop, pause, resume)
- ✅ Estadísticas de servicio
- ✅ Timeouts configurables
- ✅ Reintentos configurables
- ✅ Registry para gestión centralizada

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import AsyncService, ServiceConfig, ServiceRegistry

config = ServiceConfig(name="clothing_service", timeout=30.0)
service = AsyncService(config, change_clothing_handler)
registry = ServiceRegistry()
registry.register(service)
result = await service.execute(image_path="image.jpg")
```

### 7. Coordinator (`core/coordinator.py`)

Sistema para coordinar múltiples componentes y operaciones.

**Componentes:**
- `Coordinator` - Coordinador de componentes
- `CoordinationTask` - Definición de tareas
- `CoordinationResult` - Resultado de coordinación
- `CoordinationStatus` - Estados de coordinación

**Características:**
- ✅ Registro de componentes
- ✅ Tareas con dependencias y prioridades
- ✅ Timeouts configurables
- ✅ Manejo de errores por tarea

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import Coordinator

coordinator = Coordinator()
coordinator.register_component("clothing_service", clothing_service)
coordinator.add_task("task1", "Change Clothing", "clothing_service", "change_clothing", 
                     parameters={"image": "image.jpg"}, priority=1)
result = await coordinator.coordinate()
```

### 8. Integration System (`core/integration.py`)

Sistema para integrar servicios externos y APIs.

**Componentes:**
- `IntegrationAdapter` - Adaptador base para integraciones
- `IntegrationManager` - Gestor de múltiples integraciones
- `IntegrationConfig` - Configuración de integración
- `IntegrationResult` - Resultado de integración
- `IntegrationStatus` - Estados de integración

**Características:**
- ✅ Conexión/desconexión automática
- ✅ Reintentos configurables
- ✅ Estadísticas de llamadas
- ✅ Timeouts configurables

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import IntegrationAdapter, IntegrationConfig, IntegrationManager

config = IntegrationConfig(name="external_api", endpoint="https://api.example.com")
adapter = IntegrationAdapter(config)
manager = IntegrationManager()
manager.register(adapter)
result = await adapter.call("GET", "/data")
```

### 9. Data Pipeline (`core/data_pipeline.py`)

Sistema avanzado de pipeline de transformación de datos.

**Componentes:**
- `DataPipeline` - Pipeline de transformación
- `DataPipelineManager` - Gestor de múltiples pipelines
- `TransformStep` - Definición de pasos de transformación
- `TransformResult` - Resultado de transformación

**Características:**
- ✅ Transformación secuencial o paralela
- ✅ Timeouts configurables
- ✅ Procesamiento por lotes
- ✅ Historial de ejecuciones

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import DataPipeline, DataPipelineManager

pipeline = DataPipeline("image_processing")
pipeline.add_step("normalize", normalize_image)
pipeline.add_step("enhance", enhance_image)
result = await pipeline.transform(image_data)
```

### 10. Serializer (`core/serializer.py`)

Sistema avanzado de serialización con múltiples formatos.

**Componentes:**
- `Serializer` - Serializador avanzado
- `SerializationFormat` - Formatos (JSON, PICKLE, BASE64)
- `SerializationResult` - Resultado de serialización

**Características:**
- ✅ Múltiples formatos (JSON, Pickle, Base64)
- ✅ Serialización a archivos
- ✅ Metadata de serialización
- ✅ Deserialización segura

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import Serializer, SerializationFormat

serializer = Serializer()
result = serializer.serialize(data, format=SerializationFormat.JSON)
deserialized = serializer.deserialize(result.data, format=SerializationFormat.JSON)
```

### 11. Structured Logging (`core/structured_logging.py`)

Sistema de logging estructurado con formato JSON.

**Componentes:**
- `StructuredLogger` - Logger estructurado
- `LogEntry` - Entrada de log
- `LogLevel` - Niveles de log

**Características:**
- ✅ Formato JSON estructurado
- ✅ Contexto y metadata
- ✅ Filtrado por nivel y tiempo
- ✅ Escritura a archivos
- ✅ Estadísticas de logging

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import StructuredLogger, LogLevel

logger = StructuredLogger("app", log_file=Path("logs/app.log"))
logger.info("Task completed", context={"task_id": "123"}, metadata={"duration": 1.5})
```

### 12. Config Builder (`core/config_builder.py`)

Constructor de configuración con validación y merging.

**Componentes:**
- `ConfigBuilder` - Constructor de configuración
- `ConfigSection` - Sección de configuración

**Características:**
- ✅ Secciones de configuración
- ✅ Validación personalizada
- ✅ Valores por defecto
- ✅ Carga desde archivos (JSON, YAML)
- ✅ Merging de configuraciones

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import ConfigBuilder

builder = ConfigBuilder()
builder.add_section("api", {"endpoint": "https://api.example.com"})
builder.set_default("api.timeout", 30.0)
config = builder.build()
```

### 13. Scheduler (`core/scheduler.py`)

Sistema para programar tareas y operaciones.

**Componentes:**
- `Scheduler` - Programador de tareas
- `ScheduledTask` - Definición de tareas programadas
- `ScheduleType` - Tipos de programación (ONCE, INTERVAL, CRON)

**Características:**
- ✅ Programación única (ONCE)
- ✅ Programación por intervalo (INTERVAL)
- ✅ Programación tipo cron (CRON)
- ✅ Habilitar/deshabilitar tareas
- ✅ Estadísticas de ejecución

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import Scheduler, ScheduleType

scheduler = Scheduler()
scheduler.schedule(
    task_id="cleanup",
    name="Daily Cleanup",
    task=cleanup_task,
    schedule_type=ScheduleType.CRON,
    schedule_value="0 2 * * *"  # 2 AM daily
)
await scheduler.start()
```

### 14. Advanced Queue (`core/advanced_queue.py`)

Sistema de cola avanzado con prioridades y programación.

**Componentes:**
- `AdvancedQueue` - Cola avanzada
- `QueueItem` - Item de cola
- `QueuePriority` - Prioridades (LOW, NORMAL, HIGH, URGENT, CRITICAL)
- `QueueStatus` - Estados de items

**Características:**
- ✅ Prioridades configurables
- ✅ Programación de items
- ✅ Reintentos automáticos
- ✅ Seguimiento de estados
- ✅ Estadísticas de cola

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import AdvancedQueue, QueuePriority

queue = AdvancedQueue("task_queue")
await queue.enqueue(
    item_id="task_1",
    data={"file": "image.jpg"},
    priority=QueuePriority.HIGH
)
item = await queue.dequeue()
```

### 15. Batch Operations (`core/batch_operations.py`)

Sistema de operaciones en lote con control de concurrencia.

**Componentes:**
- `BatchOperationManager` - Gestor de operaciones en lote
- `BatchOperation` - Definición de operación en lote
- `BatchItem` - Item de lote
- `BatchResult` - Resultado de operación en lote
- `BatchStatus` - Estados de lote

**Características:**
- ✅ Control de concurrencia
- ✅ Seguimiento de progreso
- ✅ Manejo de errores por item
- ✅ Resultados parciales
- ✅ Historial de operaciones

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import BatchOperationManager

manager = BatchOperationManager()
result = await manager.execute(
    operation_id="batch_1",
    items=[item1, item2, item3],
    processor=process_item,
    max_concurrent=5
)
```

### 16. Handler Base (`core/handler_base.py`)

Clases base para todos los tipos de handlers.

**Componentes:**
- `BaseHandler` - Interfaz base para handlers
- `AsyncHandler` - Implementación async de handler
- `HandlerChain` - Cadena de handlers para procesamiento secuencial
- `HandlerConfig` - Configuración de handler
- `HandlerResult` - Resultado de handler

**Características:**
- ✅ Timeouts configurables
- ✅ Reintentos automáticos
- ✅ Estadísticas de ejecución
- ✅ Cadenas de handlers

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import AsyncHandler, HandlerConfig, HandlerChain

config = HandlerConfig(name="image_handler", timeout=30.0)
handler = AsyncHandler(config, process_image)
chain = HandlerChain("image_processing")
chain.add(handler)
result = await chain.execute(image_data)
```

### 17. Processor Base (`core/processor_base.py`)

Clases base para todos los tipos de procesadores.

**Componentes:**
- `BaseProcessor` - Interfaz base para procesadores
- `AsyncProcessor` - Implementación async de procesador
- `ProcessingConfig` - Configuración de procesamiento
- `ProcessingResult` - Resultado de procesamiento
- `ProcessingStatus` - Estados de procesamiento

**Características:**
- ✅ Procesamiento secuencial o paralelo
- ✅ Control de concurrencia
- ✅ Timeouts configurables
- ✅ Procesamiento por lotes

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import AsyncProcessor, ProcessingConfig

config = ProcessingConfig(name="image_processor", parallel=True, max_concurrent=5)
processor = AsyncProcessor(config, process_image)
results = await processor.process_batch([img1, img2, img3])
```

### 18. Result Aggregator (`core/result_aggregator.py`)

Sistema para agregar y analizar resultados.

**Componentes:**
- `ResultAggregator` - Agregador de resultados
- `AggregationResult` - Resultado de agregación

**Características:**
- ✅ Estadísticas completas (promedio, mediana, percentiles)
- ✅ Filtrado por categoría y tiempo
- ✅ Análisis de errores
- ✅ Historial de resultados

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import ResultAggregator

aggregator = ResultAggregator()
aggregator.add_result(success=True, duration=1.5, category="image_processing")
result = aggregator.aggregate("daily_stats")
```

### 19. Performance Tuner (`core/performance_tuner.py`)

Sistema para optimización automática de rendimiento.

**Componentes:**
- `PerformanceTuner` - Optimizador de rendimiento
- `TuningRecommendation` - Recomendación de optimización
- `TuningAction` - Acciones de optimización

**Características:**
- ✅ Análisis automático de rendimiento
- ✅ Recomendaciones inteligentes
- ✅ Ajuste de workers según CPU/memoria
- ✅ Historial de métricas

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import PerformanceTuner

tuner = PerformanceTuner()
recommendations = tuner.analyze(
    current_workers=5,
    avg_duration=2.5,
    success_rate=0.95,
    memory_usage=75.0,
    cpu_usage=45.0
)
```

### 20. Resource Manager (`core/resource_manager.py`)

Sistema para gestionar recursos del sistema.

**Componentes:**
- `ResourceManager` - Gestor de recursos
- `ResourceType` - Tipos de recursos
- `ResourceLimit` - Límite de recurso
- `ResourceUsage` - Uso de recurso

**Características:**
- ✅ Monitoreo de memoria, CPU, disco
- ✅ Límites configurables
- ✅ Acciones automáticas (warn, block, throttle)
- ✅ Historial de uso

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import ResourceManager, ResourceType

manager = ResourceManager()
manager.set_limit(ResourceType.MEMORY, limit=80.0, action="warn")
usage = manager.get_usage(ResourceType.MEMORY)
violations = manager.check_limits()
```

### 21. Rate Limiter (`core/rate_limiter.py`)

Sistema de limitación de tasa con algoritmo token bucket.

**Componentes:**
- `RateLimiter` - Limitador de tasa
- `RateLimitConfig` - Configuración de límites

**Características:**
- ✅ Token bucket algorithm
- ✅ Límites por cliente
- ✅ Soporte de burst
- ✅ Limpieza automática de buckets antiguos

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import RateLimiter, RateLimitConfig

config = RateLimitConfig(requests_per_second=10.0, burst_size=20)
limiter = RateLimiter(config)
allowed = await limiter.is_allowed("client_1")
```

### 22. Circuit Breaker (`core/circuit_breaker.py`)

Patrón circuit breaker para llamadas resilientes a servicios.

**Componentes:**
- `CircuitBreaker` - Circuit breaker
- `CircuitBreakerRegistry` - Registry de circuit breakers
- `CircuitBreakerConfig` - Configuración
- `CircuitState` - Estados (CLOSED, OPEN, HALF_OPEN)

**Características:**
- ✅ Estados automáticos (CLOSED, OPEN, HALF_OPEN)
- ✅ Thresholds configurables
- ✅ Timeout automático
- ✅ Registry centralizado

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import CircuitBreaker, CircuitBreakerConfig

config = CircuitBreakerConfig(failure_threshold=5, timeout_seconds=60.0)
breaker = CircuitBreaker("api_service", config)
result = await breaker.call(api_call_function)
```

### 23. Event Bus (`core/event_bus.py`)

Sistema de eventos pub/sub para arquitectura dirigida por eventos.

**Componentes:**
- `EventBus` - Bus de eventos
- `Event` - Definición de evento
- `EventType` - Tipos de eventos

**Características:**
- ✅ Publicación y suscripción
- ✅ Handlers async
- ✅ Historial de eventos
- ✅ Suscripciones wildcard

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import EventBus, Event, EventType

bus = EventBus()
bus.subscribe(EventType.TASK_COMPLETED, handle_task_completed)
await bus.publish(Event(EventType.TASK_COMPLETED, {"task_id": "123"}))
```

### 24. Telemetry (`core/telemetry.py`)

Sistema de telemetría para recolección y análisis de datos del sistema.

**Componentes:**
- `TelemetryCollector` - Recolector de telemetría
- `TelemetryManager` - Gestor de múltiples colectores
- `TelemetryData` - Punto de datos
- `TelemetryType` - Tipos (METRIC, EVENT, LOG, TRACE, SPAN)

**Características:**
- ✅ Múltiples tipos de telemetría
- ✅ Handlers personalizados
- ✅ Filtrado por tipo y tiempo
- ✅ Estadísticas

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import TelemetryManager, TelemetryType

manager = TelemetryManager()
manager.collect_metric("request_duration", 1.5, tags={"endpoint": "/api"})
```

### 25. Health Check (`core/health_check.py`)

Sistema de health checks y monitoreo de estado de servicios.

**Componentes:**
- `HealthChecker` - Gestor de health checks
- `HealthCheckResult` - Resultado de health check
- `HealthStatus` - Estados (HEALTHY, DEGRADED, UNHEALTHY, UNKNOWN)

**Características:**
- ✅ Checks personalizables
- ✅ Caché de resultados
- ✅ Ejecución paralela
- ✅ Estado general del sistema

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import HealthChecker, HealthCheckResult, HealthStatus

checker = HealthChecker()
checker.register("database", check_database_health)
result = await checker.check("database")
overall = await checker.get_overall_status()
```

### 26. Retry Manager (`core/retry_manager.py`)

Gestor de reintentos automáticos con múltiples estrategias.

**Componentes:**
- `RetryManager` - Gestor de reintentos
- `RetryConfig` - Configuración
- `RetryStrategy` - Estrategias (IMMEDIATE, EXPONENTIAL_BACKOFF, FIXED_DELAY, LINEAR_BACKOFF)
- `RetryAttempt` - Información de intento

**Características:**
- ✅ Múltiples estrategias de backoff
- ✅ Clasificación de errores
- ✅ Historial de reintentos
- ✅ Estadísticas

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import RetryManager, RetryConfig, RetryStrategy

config = RetryConfig(max_retries=3, strategy=RetryStrategy.EXPONENTIAL_BACKOFF)
manager = RetryManager(config)
result = await manager.retry_task("task_1", task_func, error, attempt_number)
```

### 27. Dependency Injection (`core/dependency_injection.py`)

Contenedor de inyección de dependencias.

**Componentes:**
- `DependencyContainer` - Contenedor DI
- Funciones globales: `get_container`, `register`, `get`, `has`

**Características:**
- ✅ Registro de servicios
- ✅ Factories
- ✅ Singletons
- ✅ Aliases

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import register, get, DependencyContainer

container = DependencyContainer()
container.register("MyService", instance=my_service, singleton=True)
service = container.get("MyService")

# O usar funciones globales
register("MyService", instance=my_service)
service = get("MyService")
```

### 28. Lifecycle Management (`core/lifecycle.py`)

Sistema de gestión del ciclo de vida de componentes.

**Componentes:**
- `LifecycleManager` - Gestor de ciclo de vida
- `LifecycleComponent` - Clase base para componentes
- `LifecycleState` - Estados del ciclo de vida
- `LifecycleHook` - Hooks de ciclo de vida

**Características:**
- ✅ Estados del ciclo de vida
- ✅ Hooks configurables
- ✅ Prioridades de hooks
- ✅ Manejo de errores

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import LifecycleManager

lifecycle = LifecycleManager("MyComponent")
lifecycle.register_hook("before_start", setup_hook, priority=1)
await lifecycle.initialize()
await lifecycle.start()
```

### 29. Validation Manager (`core/validation_manager.py`)

Sistema de validación centralizado con múltiples niveles.

**Componentes:**
- `ValidationManager` - Gestor de validación
- `ValidationRule` - Regla de validación
- `ValidationResult` - Resultado de validación
- `ValidationLevel` - Niveles (STRICT, MODERATE, LENIENT)

**Características:**
- ✅ Reglas por categoría
- ✅ Validadores personalizados
- ✅ Múltiples niveles de validación
- ✅ Errores y warnings

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import ValidationManager, ValidationRule, ValidationLevel

manager = ValidationManager()
rule = ValidationRule(name="email", validator=validate_email, level=ValidationLevel.STRICT)
manager.register_rule("user", rule)
result = manager.validate("user", user_data)
```

### 30. Metrics Collector (`core/metrics_collector.py`)

Recolector de métricas avanzado con análisis estadístico.

**Componentes:**
- `MetricsCollector` - Recolector de métricas
- `MetricPoint` - Punto de métrica

**Características:**
- ✅ Time-series metrics
- ✅ Counters, gauges, histograms
- ✅ Estadísticas (min, max, avg, percentiles)
- ✅ Cálculo de rates
- ✅ Filtrado por tags

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import MetricsCollector

collector = MetricsCollector()
collector.record("request_duration", 1.5, tags={"endpoint": "/api"})
collector.increment("request_count")
stats = collector.get_statistics("request_duration")
```

### 31. Error Handler (`core/error_handler.py`)

Sistema avanzado de manejo de errores con severidad y contexto.

**Componentes:**
- `ErrorHandler` - Gestor de errores
- `ErrorHandlerDecorator` - Decorador para manejo de errores
- `ErrorInfo` - Información de error
- `ErrorSeverity` - Severidad (LOW, MEDIUM, HIGH, CRITICAL)

**Características:**
- ✅ Handlers por tipo de excepción
- ✅ Historial de errores
- ✅ Severidad configurable
- ✅ Contexto y stack traces
- ✅ Decorador para funciones

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import ErrorHandler, ErrorSeverity

handler = ErrorHandler()
handler.register_handler(ValueError, handle_value_error)
error_info = handler.handle_error(error, severity=ErrorSeverity.HIGH, context={"user": "123"})
```

### 32. Security Manager (`core/security.py`)

Sistema de seguridad con encriptación, hashing y gestión de tokens.

**Componentes:**
- `SecurityManager` - Gestor de seguridad
- `TokenManager` - Gestor de tokens
- `Token` - Estructura de token

**Características:**
- ✅ Hash de contraseñas (PBKDF2)
- ✅ Generación de tokens seguros
- ✅ Firmas HMAC
- ✅ Encriptación simple
- ✅ Gestión de tokens con expiración

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import SecurityManager, TokenManager

security = SecurityManager()
hashed, salt = security.hash_password("password123")
is_valid = security.verify_password("password123", hashed, salt)

token_manager = TokenManager(security)
token = token_manager.generate_token(expires_in_seconds=3600)
is_valid, token_obj = token_manager.validate_token(token.value)
```

### 33. Middleware Base (`core/middleware_base.py`)

Sistema base de middleware para procesamiento de requests/responses.

**Componentes:**
- `BaseMiddleware` - Clase base para middlewares
- `MiddlewarePipeline` - Pipeline de middlewares
- `Request` - Objeto de request
- `Response` - Objeto de response

**Características:**
- ✅ Procesamiento de requests
- ✅ Procesamiento de responses
- ✅ Manejo de errores
- ✅ Pipeline configurable

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import BaseMiddleware, MiddlewarePipeline, Request, Response

class AuthMiddleware(BaseMiddleware):
    async def process_request(self, request: Request) -> Optional[Response]:
        if not request.headers.get("Authorization"):
            return Response(status_code=401, body={"error": "Unauthorized"})
        return None
    
    async def process_response(self, request: Request, response: Response) -> Response:
        response.headers["X-Processed-By"] = "AuthMiddleware"
        return response

pipeline = MiddlewarePipeline()
pipeline.add(AuthMiddleware("auth"))
response = await pipeline.process(request, handler)
```

### 34. Observability Manager (`core/observability.py`)

Sistema completo de observabilidad para monitoreo y debugging.

**Componentes:**
- `ObservabilityManager` - Gestor de observabilidad
- `LogEntry` - Entrada de log
- `Metric` - Métrica
- `Span` - Span de traza

**Características:**
- ✅ Logging estructurado
- ✅ Métricas time-series
- ✅ Spans de traza
- ✅ Context managers
- ✅ Resúmenes y estadísticas

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import ObservabilityManager, LogLevel

obs = ObservabilityManager("clothing_service")
obs.log(LogLevel.INFO, "Processing request", context={"user": "123"})
obs.record_metric("request_duration", 1.5, tags={"endpoint": "/api"})

with obs.span("process_image", tags={"image_id": "123"}):
    # Process image
    pass
```

### 35. Factory Base (`core/factory_base.py`)

Patrón factory para creación de objetos con registro de tipos y creadores.

**Componentes:**
- `BaseFactory` - Factory base
- `BuilderFactory` - Factory con soporte de builder pattern

**Características:**
- ✅ Registro de tipos
- ✅ Registro de creadores
- ✅ Soporte de builder pattern
- ✅ Listado de tipos y creadores

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import BaseFactory

factory = BaseFactory("ProcessorFactory")
factory.register_type("image", ImageProcessor)
factory.register_creator("custom", lambda: CustomProcessor())

processor = factory.create("image", config)
```

### 36. Storage Base (`core/storage_base.py`)

Base para sistemas de almacenamiento con implementación de archivos.

**Componentes:**
- `BaseStorage` - Interfaz base de almacenamiento
- `FileStorage` - Implementación basada en archivos

**Características:**
- ✅ Operaciones CRUD (save, load, delete)
- ✅ Verificación de existencia
- ✅ Listado de keys
- ✅ Limpieza con prefijos
- ✅ Estadísticas

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import FileStorage
from pathlib import Path

storage = FileStorage(Path("./data"))
await storage.save("key1", {"data": "value"})
value = await storage.load("key1")
exists = await storage.exists("key1")
keys = await storage.list_keys("prefix_")
```

### 37. Execution Context (`core/execution_context.py`)

Gestión de contexto de ejecución para requests con ContextVars.

**Componentes:**
- `ExecutionContext` - Contexto de ejecución
- `ContextManager` - Gestor de contextos

**Características:**
- ✅ Context variables thread-safe
- ✅ Request ID tracking
- ✅ User ID tracking
- ✅ Metadata y duración
- ✅ Decorador para funciones

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import ExecutionContext, ContextManager
import uuid

context = ExecutionContext(request_id=str(uuid.uuid4()), user_id="user123")
ContextManager.set_context(context)

# Access context
request_id = ContextManager.get_request_id()
user_id = ContextManager.get_user_id()

# Decorator
@ContextManager.with_context(context)
async def process_request():
    pass
```

### 38. Base Models (`core/base_models.py`)

Modelos base con funcionalidad común para todos los modelos de datos.

**Componentes:**
- `BaseModel` - Modelo base con to_dict, from_dict, update, get
- `TimestampedModel` - Modelo con timestamps (created_at, updated_at)
- `IdentifiedModel` - Modelo con ID generado
- `StatusModel` - Modelo con estado y métodos de verificación

**Características:**
- ✅ Conversión a diccionario
- ✅ Creación desde diccionario
- ✅ Actualización de campos
- ✅ Acceso seguro a campos
- ✅ Manejo de timestamps
- ✅ Generación de IDs
- ✅ Verificación de estados

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import BaseModel, TimestampedModel, IdentifiedModel

@dataclass
class User(TimestampedModel, IdentifiedModel):
    name: str
    email: str

user = User(id=IdentifiedModel.generate_id(), name="John", email="john@example.com")
user_dict = user.to_dict(exclude_none=True)
user.touch()  # Update updated_at
```

### 39. Types (`core/types.py`)

Definiciones de tipos y aliases compartidos para todo el proyecto.

**Componentes:**
- Type aliases: `FilePath`, `ConfigDict`, `ResultDict`, `OptionsDict`, `MetadataDict`
- Callback types: `TaskCallback`, `ErrorCallback`, `ProgressCallback`
- Enums: `TaskPriority`
- Data classes: `FileInfo`, `ProcessingOptions`, `TaskContext`, `ProcessingResult`

**Características:**
- ✅ Type aliases para consistencia
- ✅ Callbacks tipados
- ✅ Enums para valores constantes
- ✅ Data classes estructuradas
- ✅ Propiedades calculadas

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import (
    FilePath, TaskPriority, FileInfo, ProcessingResult
)

file_info = FileInfo(
    path="/path/to/file.jpg",
    size_bytes=1024000,
    mime_type="image/jpeg",
    extension=".jpg"
)
print(f"Size: {file_info.size_mb} MB")
print(f"Is image: {file_info.is_image}")
```

### 40. Interfaces (`core/interfaces.py`)

Interfaces abstractas para componentes core del sistema.

**Componentes:**
- `IRepository` - Interface para repositorios de datos
- `IProcessor` - Interface para procesadores
- `IService` - Interface para servicios
- `ICache` - Interface para sistemas de caché
- `INotifier` - Interface para sistemas de notificación
- `IValidator` - Interface para validadores

**Características:**
- ✅ Contratos claros para implementaciones
- ✅ Separación de interfaces e implementaciones
- ✅ Facilita testing y mocking
- ✅ Documentación implícita de APIs

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import IProcessor, ProcessingResult

class ImageProcessor(IProcessor):
    async def process(self, input_path, options):
        # Implementation
        return ProcessingResult(success=True)
    
    def validate_input(self, input_path):
        return input_path.exists()
```

### 41. Constants (`core/constants.py`)

Constantes de aplicación compartidas para todo el proyecto.

**Componentes:**
- Valores por defecto (paralelismo, caché, reintentos, rate limiting)
- Nombres de directorios
- Límites de tamaño de archivos
- Formatos soportados
- Prioridades de tareas
- Estrategias de reintento
- Configuración de logging, métricas, eventos

**Características:**
- ✅ Centralización de constantes
- ✅ Fácil configuración
- ✅ Consistencia en toda la aplicación
- ✅ Valores por defecto documentados

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import (
    DEFAULT_MAX_PARALLEL_TASKS,
    DEFAULT_CACHE_TTL_HOURS,
    SUPPORTED_IMAGE_FORMATS,
    PRIORITY_HIGH
)

max_tasks = DEFAULT_MAX_PARALLEL_TASKS
cache_ttl = DEFAULT_CACHE_TTL_HOURS * 3600
if file.extension in SUPPORTED_IMAGE_FORMATS:
    process_with_priority(PRIORITY_HIGH)
```

### 42. Helpers (`core/helpers.py`)

Funciones de ayuda comunes para operaciones frecuentes.

**Componentes:**
- `normalize_path` - Normalizar paths
- `ensure_path_exists` - Asegurar que paths existan
- `ensure_directory_exists` - Crear directorios
- `create_output_directories` - Crear múltiples directorios
- `JSONFileHandler` - Manejo de archivos JSON
- `load_json_file` / `save_json_file` - Funciones de conveniencia
- `create_message` - Crear mensajes estandarizados

**Características:**
- ✅ Operaciones de archivos comunes
- ✅ Manejo de JSON simplificado
- ✅ Creación de directorios
- ✅ Mensajes estandarizados

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import (
    ensure_directory_exists,
    load_json_file,
    save_json_file,
    create_message
)

dir_path = ensure_directory_exists("./output")
config = load_json_file("config.json")
save_json_file({"key": "value"}, "data.json")
message = create_message("info", "Processing complete", {"task_id": "123"})
```

### 43. Async Utils (`core/async_utils.py`)

Utilidades asíncronas y helpers para programación asíncrona.

**Componentes:**
- `gather_with_exceptions` - Gather con manejo de excepciones
- `timeout_after` - Ejecutar con timeout
- `retry_async` - Reintentos asíncronos
- `async_to_sync` - Convertir async a sync
- `ensure_async` - Asegurar función async
- `AsyncLock` - Lock asíncrono con timeout
- `AsyncSemaphore` - Semáforo asíncrono con timeout

**Características:**
- ✅ Gather seguro con excepciones
- ✅ Timeouts configurables
- ✅ Reintentos con backoff exponencial
- ✅ Conversión async/sync
- ✅ Locks y semáforos con timeout

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import (
    gather_with_exceptions,
    timeout_after,
    retry_async,
    AsyncLock
)

# Gather con manejo de excepciones
results = await gather_with_exceptions(coro1(), coro2(), coro3())

# Timeout
result = await timeout_after(5.0, long_running_coro())

# Retry
result = await retry_async(
    fetch_data,
    max_attempts=3,
    delay=1.0,
    exponential=True
)

# Async lock
async with AsyncLock(timeout=10.0):
    # Critical section
    pass
```

### 44. Repository Base (`core/repository_base.py`)

Base para repositorios de datos con operaciones CRUD comunes.

**Componentes:**
- `BaseRepository` - Repository base con operaciones comunes
- `RepositoryMixin` - Mixin con utilidades de repositorio

**Características:**
- ✅ Operaciones CRUD (save, get, get_all, delete)
- ✅ Verificación de existencia
- ✅ Conteo de entidades
- ✅ Búsqueda por criterios
- ✅ Actualización de entidades
- ✅ Serialización/deserialización

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import BaseRepository

class UserRepository(BaseRepository[User]):
    async def save(self, entity: User) -> None:
        # Implementation
        pass
    
    async def get(self, entity_id: str) -> Optional[User]:
        # Implementation
        pass
    
    async def get_all(self) -> List[User]:
        # Implementation
        pass
    
    async def delete(self, entity_id: str) -> bool:
        # Implementation
        pass

repo = UserRepository()
await repo.save(user)
user = await repo.get("user123")
exists = await repo.exists("user123")
count = await repo.count()
users = await repo.find_by(status="active")
```

### 45. Manager Base (`core/manager_base.py`)

Base para managers con inicialización, shutdown y estadísticas.

**Componentes:**
- `BaseManager` - Manager base con funcionalidad común
- `ManagerRegistry` - Registry para múltiples managers

**Características:**
- ✅ Inicialización y shutdown
- ✅ Estadísticas integradas
- ✅ Registry de managers
- ✅ Inicialización/shutdown masivo

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import BaseManager, ManagerRegistry

class TaskManager(BaseManager):
    async def _do_initialize(self):
        # Custom initialization
        pass
    
    async def _do_shutdown(self):
        # Custom shutdown
        pass

manager = TaskManager("TaskManager")
await manager.initialize()
manager.increment_stat("tasks_processed")
stats = manager.get_stats()

registry = ManagerRegistry()
registry.register(manager)
await registry.initialize_all()
```

### 46. Component Registry (`core/component_registry.py`)

Registry para componentes de aplicación con lifecycle y DI.

**Componentes:**
- `ComponentRegistry` - Registry de componentes

**Características:**
- ✅ Registro de componentes
- ✅ Integración con Dependency Injection
- ✅ Gestión de lifecycle
- ✅ Inicialización/start/stop/shutdown masivo
- ✅ Acceso centralizado a componentes

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import ComponentRegistry

registry = ComponentRegistry()
registry.register("cache", cache_service, lifecycle=True)
registry.register("queue", queue_service, lifecycle=True)

await registry.initialize_all()
await registry.start_all()

cache = registry.get("cache")
all_components = registry.get_all()

await registry.stop_all()
await registry.shutdown_all()
```

### 47. Decorators (`core/decorators.py`)

Decoradores comunes para funciones async y sync.

**Componentes:**
- `measure_time` - Medir y registrar tiempo de ejecución
- `log_calls` - Registrar llamadas a funciones
- `handle_errors` - Manejar errores gracefully
- `validate_input` - Validar entrada de funciones
- `async_or_sync` - Detectar tipo de función

**Características:**
- ✅ Soporte para async y sync
- ✅ Medición de tiempo
- ✅ Logging de llamadas
- ✅ Manejo de errores
- ✅ Validación de entrada

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import measure_time, log_calls, handle_errors

@measure_time(log_level="info")
@log_calls(include_args=True)
async def process_image(image_path: str):
    # Process image
    pass

@handle_errors(default_return=None)
async def risky_operation():
    # May fail
    pass
```

### 48. Context Managers (`core/context_managers.py`)

Context managers para operaciones comunes.

**Componentes:**
- `timed_operation` - Operación con timing
- `retry_operation` - Operación con reintentos
- `rate_limited_operation` - Operación con rate limiting
- `cached_operation` - Operación con caché
- `monitored_operation` - Operación con monitoreo
- `OperationContext` - Contexto de operación

**Características:**
- ✅ Timing automático
- ✅ Reintentos con backoff
- ✅ Rate limiting integrado
- ✅ Caché transparente
- ✅ Monitoreo de métricas

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import (
    timed_operation,
    retry_operation,
    cached_operation,
    monitored_operation
)

async with timed_operation("process_image"):
    # Process image
    pass

async with retry_operation(max_attempts=3, delay=1.0):
    # Operation that may fail
    pass

async with cached_operation(cache, "key") as cached:
    if cached.exists():
        return cached.get()
    value = compute_value()
    cached.set(value)

async with monitored_operation(metrics, "process_image", tags={"type": "jpg"}):
    # Monitored operation
    pass
```

### 49. Tracing (`core/tracing.py`)

Sistema de distributed tracing para tracking de requests y operaciones.

**Componentes:**
- `Tracer` - Tracer distribuido
- `Trace` - Trace con spans
- `TracingSpan` - Span individual
- `get_current_trace` / `get_current_trace_id` - Helpers de contexto

**Características:**
- ✅ Traces distribuidos
- ✅ Spans anidados
- ✅ Context variables thread-safe
- ✅ Tags y logs por span
- ✅ Context managers
- ✅ Historial de traces

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import Tracer, get_current_trace_id

tracer = Tracer("clothing_service")

with tracer.trace(tags={"user_id": "123"}) as trace:
    with tracer.span("load_model", tags={"model": "clothing"}) as span:
        # Load model
        pass
    
    with tracer.span("process_image") as span:
        # Process image
        pass

trace_id = get_current_trace_id()
recent_traces = tracer.get_recent_traces(limit=10)
```

### 50. Feature Flags (`core/feature_flags.py`)

Sistema de feature flags para gradual rollouts y control de features.

**Componentes:**
- `FeatureFlagManager` - Gestor de feature flags
- `FeatureFlag` - Definición de flag
- `FeatureFlagType` - Tipos (BOOLEAN, PERCENTAGE, USER_LIST, CUSTOM)
- `feature_flag` - Decorador para funciones

**Características:**
- ✅ Flags booleanos
- ✅ Rollout por porcentaje
- ✅ Lista de usuarios
- ✅ Checks personalizados
- ✅ Persistencia en archivo
- ✅ Decorador para funciones

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import (
    FeatureFlagManager,
    FeatureFlag,
    FeatureFlagType,
    feature_flag
)

manager = FeatureFlagManager()
flag = FeatureFlag(
    name="new_algorithm",
    enabled=True,
    flag_type=FeatureFlagType.PERCENTAGE,
    percentage=25.0  # 25% of users
)
manager.register(flag)

if manager.is_enabled("new_algorithm", context={"user_id": "123"}):
    use_new_algorithm()

@feature_flag("new_algorithm", lambda: {"user_id": get_current_user()})
async def process_with_new_algorithm():
    # New algorithm code
    pass
```

### 51. Audit (`core/audit.py`)

Sistema de auditoría avanzado con tracking detallado y características de compliance.

**Componentes:**
- `AuditLogger` - Logger de auditoría
- `AuditEntry` - Entrada de auditoría
- `AuditAction` - Acciones (CREATE, READ, UPDATE, DELETE, EXECUTE, LOGIN, LOGOUT, etc.)
- `AuditLevel` - Niveles (INFO, WARNING, ERROR, CRITICAL)

**Características:**
- ✅ Tracking detallado de acciones
- ✅ Filtrado por usuario, acción, recurso, fecha
- ✅ Handlers personalizados
- ✅ Estadísticas agregadas
- ✅ Compliance y seguridad

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import AuditLogger, AuditAction, AuditLevel

audit = AuditLogger()
audit.log(
    action=AuditAction.UPDATE,
    resource="user_profile",
    user_id="user123",
    level=AuditLevel.INFO,
    success=True,
    ip_address="192.168.1.1",
    details={"field": "email", "old_value": "old@example.com", "new_value": "new@example.com"}
)

entries = audit.get_entries(user_id="user123", limit=10)
stats = audit.get_statistics()
```

### 52. Backup (`core/backup.py`)

Sistema de backups automáticos con scheduling y monitoreo.

**Componentes:**
- `BackupManager` - Gestor de backups
- `BackupConfig` - Configuración de backup
- `BackupResult` - Resultado de backup
- `BackupType` - Tipos (FULL, INCREMENTAL, DIFFERENTIAL)
- `BackupStatus` - Estados (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)

**Características:**
- ✅ Backups automáticos
- ✅ Compresión opcional
- ✅ Retención configurable
- ✅ Historial de backups
- ✅ Limpieza automática de backups antiguos
- ✅ Estadísticas de backups

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import BackupManager, BackupConfig, BackupType
from pathlib import Path

backup_manager = BackupManager(backup_dir=Path("./backups"))

config = BackupConfig(
    name="daily_backup",
    source_paths=["./data", "./config"],
    backup_type=BackupType.FULL,
    retention_days=30,
    compress=True,
    max_backups=10
)
backup_manager.register_config(config)

result = await backup_manager.run_backup("daily_backup")
stats = backup_manager.get_backup_stats("daily_backup")
```

### 53. Migrations (`core/migrations.py`)

Sistema de migraciones de datos y esquemas.

**Componentes:**
- `MigrationRunner` - Ejecutor de migraciones
- `Migration` - Definición de migración
- `MigrationStatus` - Estados (PENDING, RUNNING, COMPLETED, FAILED, ROLLED_BACK)

**Características:**
- ✅ Migraciones con dependencias
- ✅ Rollback de migraciones
- ✅ Historial de migraciones
- ✅ Ordenamiento por dependencias
- ✅ Soporte async y sync

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import MigrationRunner, Migration
from pathlib import Path

runner = MigrationRunner(migrations_dir=Path("./migrations"))

def migrate_up():
    # Migration logic
    pass

def migrate_down():
    # Rollback logic
    pass

migration = Migration(
    id="001_add_users_table",
    name="Add users table",
    version="1.0.0",
    up=migrate_up,
    down=migrate_down,
    dependencies=[]
)
runner.register(migration)

await runner.run_migration("001_add_users_table")
pending = runner.get_pending_migrations()
await runner.run_all_pending()
```

### 54. API Versioning (`core/api_versioning.py`)

Sistema de versionado de APIs y compatibilidad hacia atrás.

**Componentes:**
- `APIVersionManager` - Gestor de versiones
- `APIVersion` - Definición de versión
- `VersionedEndpoint` - Endpoint versionado
- `VersionStrategy` - Estrategias (URL_PATH, QUERY_PARAM, HEADER, ACCEPT_HEADER)

**Características:**
- ✅ Múltiples estrategias de versionado
- ✅ Deprecación de versiones
- ✅ Sunset dates
- ✅ Changelog y breaking changes
- ✅ Decorador para endpoints

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import (
    APIVersionManager,
    APIVersion,
    VersionStrategy
)
from datetime import datetime, timedelta

manager = APIVersionManager(
    default_version="1.0",
    strategy=VersionStrategy.URL_PATH
)

v1 = APIVersion(
    version="1.0",
    deprecated=False
)

v2 = APIVersion(
    version="2.0",
    deprecated=False,
    changelog=["Added new fields", "Improved performance"],
    breaking_changes=["Removed deprecated field"]
)

manager.register_version("1.0", v1)
manager.register_version("2.0", v2)

@manager.version_decorator(versions=["1.0", "2.0"])
async def endpoint_handler(request):
    version = request.get("api_version", "1.0")
    # Handle based on version
    pass

version_info = manager.get_version_info()
```

### 55. Testing (`core/testing.py`)

Sistema de testing avanzado con fixtures, mocks y utilidades.

**Componentes:**
- `TestRunner` - Ejecutor de tests
- `AsyncTestCase` - Clase base para tests async
- `TestFixture` - Gestor de fixtures
- `MockBuilder` - Constructor de mocks
- `TestConfig` - Configuración de tests
- `TestResult` - Resultado de test
- `TestDecorator` - Decoradores para tests
- `temp_directory` / `temp_file` - Context managers para archivos temporales

**Características:**
- ✅ Tests async y sync
- ✅ Fixtures con cleanup
- ✅ Mocks (Mock, MagicMock, AsyncMock)
- ✅ Timeouts y reintentos
- ✅ Reportes de tests
- ✅ Context managers para archivos temporales

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import (
    AsyncTestCase,
    TestConfig,
    MockBuilder,
    temp_directory
)

class MyTestCase(AsyncTestCase):
    async def setup(self):
        # Setup
        pass
    
    async def test_something(self):
        mock_service = MockBuilder.create_async_mock(return_value={"result": "ok"})
        result = await mock_service()
        assert result["result"] == "ok"

test_case = MyTestCase(TestConfig(timeout=5.0, verbose=True))
result = await test_case.run_test(test_case.test_something)

# With temp directory
with temp_directory() as temp_dir:
    # Use temp_dir
    pass
```

### 56. Notifications (`core/notifications.py`)

Sistema de notificaciones multi-canal con handlers personalizables.

**Componentes:**
- `NotificationManager` - Gestor de notificaciones
- `Notification` - Datos de notificación
- `NotificationChannel` - Canales (EMAIL, SMS, PUSH, WEBHOOK, SLACK, DISCORD)
- `NotificationPriority` - Prioridades (LOW, NORMAL, HIGH, URGENT)
- `NotificationHandler` - Handler base
- `EmailNotificationHandler` / `WebhookNotificationHandler` - Handlers específicos

**Características:**
- ✅ Múltiples canales
- ✅ Handlers personalizables
- ✅ Prioridades
- ✅ Historial de notificaciones
- ✅ Estadísticas de envío

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import (
    NotificationManager,
    Notification,
    NotificationChannel,
    NotificationPriority,
    WebhookNotificationHandler
)

manager = NotificationManager()
webhook_handler = WebhookNotificationHandler("https://example.com/webhook")
manager.register_handler(NotificationChannel.WEBHOOK, webhook_handler)

notification = Notification(
    title="Task Completed",
    message="Your clothing change task has been completed",
    channel=NotificationChannel.WEBHOOK,
    priority=NotificationPriority.HIGH
)

result = await manager.send(notification)
history = manager.get_history(NotificationChannel.WEBHOOK, limit=10)
stats = manager.get_stats()
```

### 57. Webhooks (`core/webhooks.py`)

Gestor de webhooks con firma y reintentos.

**Componentes:**
- `WebhookManager` - Gestor de webhooks
- `Webhook` - Configuración de webhook
- `WebhookPayload` - Payload de webhook
- `WebhookEvent` - Eventos (TASK_CREATED, TASK_STARTED, TASK_COMPLETED, TASK_FAILED, BATCH_COMPLETED)

**Características:**
- ✅ Múltiples endpoints
- ✅ Filtrado por eventos
- ✅ Firma HMAC para seguridad
- ✅ Reintentos con backoff exponencial
- ✅ Entrega asíncrona
- ✅ Estadísticas de envío

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import (
    WebhookManager,
    Webhook,
    WebhookEvent
)

manager = WebhookManager()

webhook = Webhook(
    url="https://example.com/webhook",
    events=[WebhookEvent.TASK_COMPLETED, WebhookEvent.TASK_FAILED],
    secret="webhook_secret",
    retries=3,
    timeout=10.0
)
manager.register(webhook)

await manager.send(
    event=WebhookEvent.TASK_COMPLETED,
    payload={"task_id": "123", "result": "success"},
    task_id="123"
)

stats = manager.get_stats()
await manager.close()
```

### 58. Alerting (`core/alerting.py`)

Sistema de alertas basado en condiciones con monitoreo continuo.

**Componentes:**
- `AlertManager` - Gestor de alertas
- `Alert` - Definición de alerta
- `AlertRule` - Regla de alerta
- `AlertSeverity` - Severidades (LOW, MEDIUM, HIGH, CRITICAL)
- `AlertStatus` - Estados (ACTIVE, ACKNOWLEDGED, RESOLVED, SUPPRESSED)

**Características:**
- ✅ Reglas basadas en condiciones
- ✅ Monitoreo continuo
- ✅ Cooldown para evitar spam
- ✅ Handlers personalizables
- ✅ Gestión de estados (acknowledge, resolve)
- ✅ Filtrado por severidad

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import (
    AlertManager,
    AlertRule,
    AlertSeverity
)

manager = AlertManager()

async def check_high_error_rate():
    # Check condition
    error_rate = get_error_rate()
    return error_rate > 0.1

rule = AlertRule(
    name="high_error_rate",
    condition=check_high_error_rate,
    severity=AlertSeverity.HIGH,
    message_template="Error rate is above 10%",
    cooldown_seconds=300.0
)
manager.register_rule(rule)

async def handle_alert(alert):
    # Send notification, log, etc.
    pass

manager.register_handler(handle_alert)
await manager.start_monitoring(interval_seconds=60.0)

active_alerts = manager.get_active_alerts()
critical_alerts = manager.get_alerts_by_severity(AlertSeverity.CRITICAL)
manager.acknowledge_alert("alert_id")
manager.resolve_alert("alert_id")
```

### 59. Reporting (`core/reporting.py`)

Sistema de generación de reportes con múltiples tipos y exportación.

**Componentes:**
- `ReportGenerator` - Generador de reportes
- `Report` - Definición de reporte
- `ReportType` - Tipos (SUMMARY, DETAILED, PERFORMANCE, USAGE, ERROR, CUSTOM)

**Características:**
- ✅ Múltiples tipos de reportes
- ✅ Reportes de resumen, performance, uso
- ✅ Reportes personalizados
- ✅ Exportación a JSON y Markdown
- ✅ Historial de reportes
- ✅ Períodos de tiempo

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import (
    ReportGenerator,
    ReportType
)
from pathlib import Path
from datetime import timedelta

generator = ReportGenerator()

# Summary report
metrics = {
    "total_tasks": 100,
    "successful_tasks": 95,
    "failed_tasks": 5,
    "success_rate": 0.95,
    "avg_processing_time": 2.5
}
summary_report = generator.generate_summary_report(
    metrics,
    period=timedelta(days=1)
)

# Performance report
performance_data = {
    "avg_response_time": 1.2,
    "p95_response_time": 2.5,
    "p99_response_time": 3.8,
    "throughput": 50.0,
    "error_rate": 0.05
}
perf_report = generator.generate_performance_report(performance_data)

# Custom report
custom_report = generator.generate_custom_report(
    title="Custom Analysis",
    content={"custom_data": "value"}
)

# Export
generator.export_report(summary_report, Path("./reports/summary.json"), format="json")
generator.export_report(perf_report, Path("./reports/performance.md"), format="markdown")

recent = generator.get_recent_reports(limit=10)
```

### 60. Analytics (`core/analytics.py`)

Sistema de analytics y eventos para tracking de uso.

**Componentes:**
- `AnalyticsCollector` - Recolector de eventos
- `AnalyticsReporter` - Generador de reportes de analytics
- `AnalyticsEvent` - Evento de analytics
- `EventType` - Tipos de eventos (TASK_CREATED, TASK_COMPLETED, API_CALL, CACHE_HIT, etc.)

**Características:**
- ✅ Tracking de eventos
- ✅ Filtrado por tipo, usuario, tiempo
- ✅ Estadísticas agregadas
- ✅ Reportes diarios
- ✅ Top eventos

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import (
    AnalyticsCollector,
    AnalyticsReporter,
    EventType
)

collector = AnalyticsCollector()
reporter = AnalyticsReporter(collector)

collector.track(
    event_type=EventType.TASK_COMPLETED,
    user_id="user123",
    session_id="session456",
    properties={"task_id": "task789", "duration": 2.5}
)

events = collector.get_events(
    event_type=EventType.TASK_COMPLETED,
    limit=100
)
stats = collector.get_stats(period=timedelta(days=1))
daily_report = reporter.generate_daily_report()
```

### 61. Monitoring Dashboard (`core/monitoring_dashboard.py`)

Dashboard de monitoreo en tiempo real con métricas y tendencias.

**Componentes:**
- `MonitoringDashboard` - Dashboard de monitoreo
- `DashboardMetrics` - Métricas del dashboard

**Características:**
- ✅ Métricas en tiempo real
- ✅ Datos históricos
- ✅ Tendencias de rendimiento
- ✅ Health score del sistema
- ✅ Registro automático de métricas

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import MonitoringDashboard

def get_stats():
    # Return stats from your service
    return {
        "executor_stats": {"total_tasks": 100, "completed_tasks": 95},
        "cache_stats": {"hit_rate": 0.85},
        # ...
    }

dashboard = MonitoringDashboard(stats_provider=get_stats)
metrics = dashboard.get_current_metrics()
dashboard.record_metrics()

historical = dashboard.get_historical_metrics(hours=24)
trends = dashboard.get_performance_trends()
health = dashboard.get_system_health()
```

### 62. Plugin System (`core/plugin_system.py`)

Sistema de plugins extensible para funcionalidades personalizadas.

**Componentes:**
- `PluginManager` - Gestor de plugins
- `EnhancementPlugin` - Clase base para plugins

**Características:**
- ✅ Registro de plugins
- ✅ Descubrimiento automático
- ✅ Ejecución de plugins
- ✅ Validación de parámetros
- ✅ Carga desde directorio
- ✅ Habilitar/deshabilitar plugins

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import (
    PluginManager,
    EnhancementPlugin
)

class MyPlugin(EnhancementPlugin):
    def get_name(self) -> str:
        return "my_plugin"
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def process(self, file_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        # Process logic
        return {"result": "success"}

manager = PluginManager()
plugin = MyPlugin()
manager.register(plugin)
manager.enable("my_plugin")

result = await manager.execute_plugin(
    "my_plugin",
    "/path/to/file",
    {"param": "value"}
)

plugins = manager.list_plugins()
manager.load_from_directory("./plugins")
```

### 63. Optimizer (`core/optimizer.py`)

Optimizador de rendimiento y recursos del sistema.

**Componentes:**
- `PerformanceOptimizer` - Optimizador de rendimiento
- `ResourceMonitor` - Monitor de recursos
- `OptimizationResult` - Resultado de optimización

**Características:**
- ✅ Optimización de memoria
- ✅ Optimización de caché
- ✅ Monitoreo de recursos
- ✅ Auto-optimización
- ✅ Estadísticas del sistema
- ✅ Historial de optimizaciones

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import (
    PerformanceOptimizer,
    ResourceMonitor
)

optimizer = PerformanceOptimizer()
memory_result = optimizer.optimize_memory()
cache_result = optimizer.optimize_cache(cache_manager)

system_stats = optimizer.get_system_stats()
history = optimizer.get_optimization_history()

monitor = ResourceMonitor(interval=60.0)
await monitor.start()

recent_stats = monitor.get_recent_stats(limit=100)
avg_stats = monitor.get_average_stats()

await monitor.stop()
```

### 64. Benchmark (`core/benchmark.py`)

Sistema de benchmarking y profiling para pruebas de rendimiento.

**Componentes:**
- `BenchmarkRunner` - Ejecutor de benchmarks
- `BenchmarkResult` - Resultado de benchmark
- `PerformanceProfiler` - Profiler de rendimiento

**Características:**
- ✅ Benchmarking de funciones async
- ✅ Estadísticas completas (avg, min, max, p50, p95, p99)
- ✅ Comparación de benchmarks
- ✅ Warmup iterations
- ✅ Profiling detallado
- ✅ Manejo de errores

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import (
    BenchmarkRunner,
    PerformanceProfiler
)

runner = BenchmarkRunner()

async def my_function():
    # Function to benchmark
    await asyncio.sleep(0.1)
    return "result"

result = await runner.benchmark(
    name="my_function",
    func=my_function,
    iterations=100,
    warmup=5
)

comparison = runner.compare("benchmark1", "benchmark2")
all_results = runner.get_all_results()

profiler = PerformanceProfiler()
start_time = profiler.start("operation")
# ... do operation ...
profiler.end("operation", start_time)

stats = profiler.get_stats("operation")
all_stats = profiler.get_all_stats()
```

### 65. Task Manager (`core/task_manager.py`)

Gestor de tareas con repositorio, eventos y tracking de estados.

**Componentes:**
- `TaskManager` - Gestor de tareas
- `Task` - Estructura de datos de tarea
- `TaskStatus` - Estados (PENDING, PROCESSING, COMPLETED, FAILED, CANCELLED)
- `TaskEvent` - Eventos del ciclo de vida
- `TaskRepository` - Repositorio abstracto
- `FileTaskRepository` - Repositorio basado en archivos
- `EventRegistry` - Registry de eventos

**Características:**
- ✅ Gestión completa de tareas
- ✅ Repositorio persistente (archivos)
- ✅ Sistema de eventos
- ✅ Cola de prioridades
- ✅ Tracking de estados
- ✅ Resultados y errores

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import (
    TaskManager,
    TaskStatus,
    TaskEvent
)

manager = TaskManager(storage_dir="./tasks")
await manager.initialize()

# Subscribe to events
async def on_task_completed(task):
    print(f"Task {task.id} completed!")

manager.events.subscribe(TaskEvent.COMPLETED, on_task_completed)

# Create task
task_id = await manager.create_task(
    service_type="clothing_change",
    parameters={"image": "path/to/image.jpg"},
    priority=10,
    metadata={"user_id": "123"}
)

# Get pending tasks
pending = await manager.get_pending_tasks(limit=10)

# Update status
await manager.update_task_status(task_id, "processing")

# Complete task
await manager.complete_task(task_id, {"result": "success"})

# Get task status
status = await manager.get_task_status(task_id)
result = await manager.get_task_result(task_id)
```

### 66. Parallel Executor (`core/parallel_executor.py`)

Ejecutor paralelo con worker pool configurable.

**Componentes:**
- `ParallelExecutor` - Ejecutor paralelo
- `WorkerPool` - Pool de workers

**Características:**
- ✅ Worker pool configurable
- ✅ Cola de tareas
- ✅ Soporte async y sync
- ✅ Estadísticas de ejecución
- ✅ Manejo de errores

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import ParallelExecutor

executor = ParallelExecutor(max_workers=5)
await executor.start()

async def process_image(image_path: str):
    # Process image
    return {"result": "processed"}

# Submit tasks
future1 = await executor.submit_task(process_image, "image1.jpg")
future2 = await executor.submit_task(process_image, "image2.jpg")

# Get results
result1 = await future1
result2 = await future2

# Get stats
stats = executor.get_stats()

await executor.stop()
```

### 67. Executor Base (`core/executor_base.py`)

Base para ejecutores con AsyncExecutor.

**Componentes:**
- `BaseExecutor` - Interfaz base para ejecutores
- `AsyncExecutor` - Ejecutor async con concurrencia limitada
- `ExecutionResult` - Resultado de ejecución
- `ExecutionStatus` - Estados de ejecución

**Características:**
- ✅ Interfaz base abstracta
- ✅ Ejecutor async con semáforo
- ✅ Timeouts configurables
- ✅ Ejecución en batch
- ✅ Tracking de resultados
- ✅ Estadísticas

**Ejemplo de uso:**
```python
from character_clothing_changer_ai.core import (
    AsyncExecutor,
    ExecutionStatus
)

executor = AsyncExecutor(name="MyExecutor", max_concurrent=10)

async def my_async_function(param: str):
    # Do work
    return {"result": param}

# Execute single
result = await executor.execute(
    my_async_function,
    "param_value",
    timeout=30.0
)

if result.status == ExecutionStatus.COMPLETED:
    print(f"Result: {result.result}")
else:
    print(f"Error: {result.error}")

# Execute batch
items = ["item1", "item2", "item3"]
results = await executor.execute_batch(
    my_async_function,
    items
)

# Get stats
stats = executor.get_stats()
recent = executor.get_recent_results(limit=10)
```

## 📁 Estructura de Archivos

```
core/
├── workflow.py              # Sistema de workflows
├── pipeline.py              # Sistema de pipelines
├── orchestrator.py          # Sistema de orquestación
├── state_manager.py         # Gestor de estado
├── advanced_cache.py        # Sistema de caché avanzado
├── service_base.py          # Clases base para servicios
├── coordinator.py           # Coordinador de componentes
├── integration.py           # Sistema de integración
├── data_pipeline.py         # Pipeline de transformación de datos
├── serializer.py            # Sistema de serialización
├── structured_logging.py    # Sistema de logging estructurado
├── config_builder.py        # Constructor de configuración
├── scheduler.py             # Sistema de programación de tareas
├── advanced_queue.py        # Cola avanzada con prioridades
├── batch_operations.py       # Operaciones en lote
├── handler_base.py          # Clases base para handlers
├── processor_base.py        # Clases base para procesadores
├── result_aggregator.py      # Agregación de resultados
├── performance_tuner.py     # Optimización de rendimiento
├── resource_manager.py      # Gestión de recursos
├── rate_limiter.py          # Limitación de tasa
├── circuit_breaker.py        # Circuit breaker
├── event_bus.py             # Sistema de eventos
├── telemetry.py             # Sistema de telemetría
├── health_check.py          # Health checks
├── retry_manager.py         # Gestor de reintentos
├── dependency_injection.py  # Inyección de dependencias
├── lifecycle.py             # Gestión de ciclo de vida
├── validation_manager.py     # Sistema de validación
├── metrics_collector.py      # Recolector de métricas
├── error_handler.py          # Manejo de errores
├── security.py               # Sistema de seguridad
├── middleware_base.py        # Sistema base de middleware
├── observability.py          # Sistema de observabilidad
├── factory_base.py           # Sistema factory
├── storage_base.py           # Sistema de almacenamiento
├── execution_context.py      # Contexto de ejecución
├── base_models.py            # Modelos base
├── types.py                  # Definiciones de tipos
├── interfaces.py             # Interfaces abstractas
├── constants.py              # Constantes de aplicación
├── helpers.py                # Funciones de ayuda comunes
├── async_utils.py            # Utilidades asíncronas
├── repository_base.py        # Base para repositorios
├── manager_base.py           # Base para managers
├── component_registry.py     # Registry de componentes
├── decorators.py             # Decoradores comunes
├── context_managers.py       # Context managers
├── tracing.py                # Distributed tracing
├── feature_flags.py          # Feature flags
├── audit.py                  # Audit system
├── backup.py                 # Backup system
├── migrations.py             # Migrations system
├── api_versioning.py         # API versioning
├── testing.py                # Testing utilities
├── notifications.py          # Notification system
├── webhooks.py               # Webhook manager
├── alerting.py               # Alerting system
├── reporting.py              # Reporting system
├── analytics.py              # Analytics system
├── monitoring_dashboard.py   # Monitoring dashboard
├── plugin_system.py          # Plugin system
├── optimizer.py              # Performance optimizer
├── benchmark.py              # Benchmark system
├── task_manager.py           # Task manager
├── parallel_executor.py       # Parallel executor
├── executor_base.py          # Executor base
└── __init__.py              # Exports actualizados
```

## 🔄 Integración

Todos los sistemas están integrados en el módulo `core` y exportados a través de `__init__.py`:

```python
from character_clothing_changer_ai.core import (
    Workflow, WorkflowManager,
    Pipeline, PipelineManager,
    Orchestrator,
    StateManager,
    AdvancedCache,
    BaseService, AsyncService, ServiceRegistry,
    Coordinator,
    IntegrationAdapter, IntegrationManager,
    DataPipeline, DataPipelineManager,
    Serializer,
    StructuredLogger,
    ConfigBuilder,
    Scheduler,
    AdvancedQueue,
    BatchOperationManager,
    BaseHandler, AsyncHandler, HandlerChain,
    BaseProcessor, AsyncProcessor,
    ResultAggregator,
    PerformanceTuner,
    ResourceManager,
    RateLimiter,
    CircuitBreaker,
    EventBus,
    TelemetryManager,
    HealthChecker,
    RetryManager,
    DependencyContainer,
    LifecycleManager,
    ValidationManager,
    MetricsCollector,
    ErrorHandler,
    SecurityManager,
    TokenManager,
    BaseMiddleware,
    MiddlewarePipeline,
    ObservabilityManager,
    BaseFactory,
    BuilderFactory,
    BaseStorage,
    FileStorage,
    ExecutionContext,
    ContextManager,
    BaseModel,
    TimestampedModel,
    IdentifiedModel,
    StatusModel,
    FilePath,
    TaskPriority,
    FileInfo,
    ProcessingResult,
    IRepository,
    IProcessor,
    IService,
    DEFAULT_MAX_PARALLEL_TASKS,
    SUPPORTED_IMAGE_FORMATS,
    normalize_path,
    load_json_file,
    gather_with_exceptions,
    retry_async,
    BaseRepository,
    BaseManager,
    ComponentRegistry,
    measure_time,
    log_calls,
    timed_operation,
    retry_operation,
    Tracer,
    FeatureFlagManager,
    feature_flag,
    AuditLogger,
    BackupManager,
    MigrationRunner,
    APIVersionManager,
    TestRunner,
    NotificationManager,
    WebhookManager,
    AlertManager,
    ReportGenerator,
    AnalyticsCollector,
    AnalyticsReporter,
    MonitoringDashboard,
    PluginManager,
    PerformanceOptimizer,
    ResourceMonitor,
    BenchmarkRunner,
    PerformanceProfiler,
    TaskManager,
    Task,
    TaskStatus,
    TaskEvent,
    ParallelExecutor,
    BaseExecutor,
    AsyncExecutor
)
```

## ✨ Beneficios

1. **Arquitectura Modular**: Sistemas independientes y reutilizables
2. **Gestión de Estado**: Estado centralizado y observable
3. **Caché Inteligente**: Múltiples estrategias de evicción
4. **Workflows y Pipelines**: Procesamiento estructurado y configurable
5. **Orquestación**: Coordinación de servicios complejos
6. **Base Patterns**: Patrones base para servicios consistentes
7. **Coordinación**: Gestión de componentes múltiples
8. **Integración**: Conexión con servicios externos
9. **Transformación de Datos**: Pipelines de transformación flexibles
10. **Serialización**: Múltiples formatos de serialización
11. **Logging Estructurado**: Logs con contexto y metadata
12. **Configuración Avanzada**: Construcción y validación de config
13. **Programación de Tareas**: Scheduling con múltiples tipos
14. **Cola Avanzada**: Prioridades y programación de items
15. **Operaciones en Lote**: Procesamiento concurrente controlado
16. **Handlers Base**: Patrones base para handlers consistentes
17. **Procesadores Base**: Patrones base para procesadores consistentes
18. **Agregación de Resultados**: Análisis estadístico completo
19. **Optimización Automática**: Tuning inteligente de rendimiento
20. **Gestión de Recursos**: Monitoreo y límites de sistema
21. **Limitación de Tasa**: Control de throughput con token bucket
22. **Circuit Breaker**: Protección contra fallos en cascada
23. **Event Bus**: Arquitectura dirigida por eventos
24. **Telemetría**: Observabilidad completa del sistema
25. **Health Checks**: Monitoreo de salud de servicios
26. **Reintentos Inteligentes**: Múltiples estrategias de backoff
27. **Inyección de Dependencias**: Gestión de dependencias
28. **Ciclo de Vida**: Gestión estructurada de componentes
29. **Validación Centralizada**: Sistema de validación con múltiples niveles
30. **Recolección de Métricas**: Métricas time-series con análisis estadístico
31. **Manejo de Errores**: Sistema avanzado con severidad y contexto
32. **Seguridad**: Encriptación, hashing y gestión de tokens
33. **Middleware**: Pipeline de procesamiento de requests/responses
34. **Observabilidad**: Sistema completo de logs, métricas y traces
35. **Factory Pattern**: Creación de objetos con registro de tipos
36. **Storage Base**: Sistema de almacenamiento con implementación de archivos
37. **Execution Context**: Gestión de contexto de ejecución thread-safe
38. **Base Models**: Modelos base con funcionalidad común
39. **Types**: Definiciones de tipos y aliases compartidos
40. **Interfaces**: Interfaces abstractas para componentes core
41. **Constants**: Constantes de aplicación compartidas
42. **Helpers**: Funciones de ayuda comunes
43. **Async Utils**: Utilidades asíncronas y helpers
44. **Repository Base**: Base para repositorios de datos
45. **Manager Base**: Base para managers con estadísticas
46. **Component Registry**: Registry para componentes de aplicación
47. **Decorators**: Decoradores comunes para funciones
48. **Context Managers**: Context managers para operaciones
49. **Tracing**: Sistema de distributed tracing para tracking
50. **Feature Flags**: Sistema de feature flags y gradual rollouts
51. **Audit**: Sistema de auditoría avanzado
52. **Backup**: Sistema de backups automáticos
53. **Migrations**: Sistema de migraciones de datos
54. **API Versioning**: Sistema de versionado de APIs
55. **Testing**: Sistema de testing avanzado
56. **Notifications**: Sistema de notificaciones multi-canal
57. **Webhooks**: Gestor de webhooks con firma
58. **Alerting**: Sistema de alertas basado en condiciones
59. **Reporting**: Sistema de generación de reportes
60. **Analytics**: Sistema de analytics y eventos
61. **Monitoring Dashboard**: Dashboard de monitoreo en tiempo real
62. **Plugin System**: Sistema de plugins extensible
63. **Optimizer**: Optimizador de rendimiento y recursos
64. **Benchmark**: Sistema de benchmarking y profiling
65. **Task Manager**: Gestor de tareas con repositorio y eventos
66. **Parallel Executor**: Ejecutor paralelo con worker pool
67. **Executor Base**: Base para ejecutores con AsyncExecutor

## 🎯 Próximos Pasos

Estos sistemas proporcionan la base para futuras mejoras:
- Integración con el servicio principal de cambio de ropa
- Workflows predefinidos para operaciones comunes
- Pipelines optimizados para procesamiento de imágenes
- Caché de resultados de procesamiento
- Gestión de estado para sesiones de usuario
- Coordinación de múltiples servicios de procesamiento
- Integración con APIs externas (OpenRouter, DeepSeek, etc.)
- Transformación de datos de imágenes
- Serialización de resultados y configuraciones
- Logging estructurado para debugging y monitoreo
- Configuración dinámica y validación
- Tareas programadas (limpieza, backups, etc.)
- Cola de procesamiento con prioridades
- Procesamiento en lote de múltiples imágenes
- Handlers y procesadores consistentes y reutilizables
- Análisis estadístico de rendimiento y resultados
- Optimización automática basada en métricas
- Gestión proactiva de recursos del sistema
- Limitación de tasa para protección de servicios
- Circuit breakers para resiliencia
- Arquitectura dirigida por eventos
- Telemetría completa para observabilidad
- Health checks para monitoreo continuo
- Reintentos inteligentes con múltiples estrategias
- Inyección de dependencias para desacoplamiento
- Gestión estructurada del ciclo de vida
- Validación centralizada con reglas configurables
- Recolección de métricas time-series con análisis
- Manejo avanzado de errores con contexto y severidad
- Seguridad con encriptación, hashing y tokens
- Middleware pipeline para procesamiento de requests
- Observabilidad completa con logs, métricas y spans
- Factory pattern para creación flexible de objetos
- Storage base con implementación de archivos
- Execution context thread-safe para tracking de requests
- Modelos base con funcionalidad común (to_dict, from_dict, update)
- Tipos y aliases compartidos para consistencia
- Interfaces abstractas para contratos claros
- Constantes centralizadas para configuración
- Helpers comunes para operaciones frecuentes
- Utilidades asíncronas con timeouts y reintentos
- Repository base para acceso a datos
- Manager base con lifecycle y estadísticas
- Component registry con DI y lifecycle integrado
- Decoradores comunes para timing, logging y manejo de errores
- Context managers para operaciones con timing, retry, cache y monitoreo
- Distributed tracing para tracking completo de requests
- Feature flags para gradual rollouts y control de features
- Sistema de auditoría para compliance y seguridad
- Backups automáticos con scheduling y retención
- Migraciones de datos con dependencias y rollback
- Versionado de APIs con deprecación y compatibilidad
- Testing avanzado con fixtures, mocks y reportes
- Notificaciones multi-canal con handlers personalizables
- Webhooks con firma HMAC y reintentos
- Alertas basadas en condiciones con monitoreo continuo
- Generación de reportes con múltiples tipos y exportación
- Analytics y tracking de eventos con reportes diarios
- Dashboard de monitoreo en tiempo real con health score
- Sistema de plugins extensible con descubrimiento automático
- Optimización automática de memoria y caché
- Benchmarking y profiling para pruebas de rendimiento
- Gestión completa de tareas con repositorio persistente y eventos
- Ejecución paralela con worker pool configurable
- Base para ejecutores con control de concurrencia y timeouts

## 📝 Notas

- Todos los sistemas son completamente asíncronos
- Thread-safe donde es necesario (locks async)
- Compatibles con la arquitectura existente
- 100% backward compatible
- Dependencias opcionales: `croniter` (para CRON scheduling), `pyyaml` (para YAML config)

