# Mejoras V56: Registro y Monitoreo Avanzado de Pipelines

## Resumen

Se ha agregado un sistema completo de registro y monitoreo para pipelines con metadatos, versionado, salud, alertas y estadísticas.

## Mejoras Implementadas

### 1. Pipeline Registry (`core/pipeline_registry.py`)

Sistema de registro centralizado con metadatos completos y persistencia.

#### Características Principales:

✅ **Metadatos Completos**
- Versión, descripción, autor
- Tags y dependencias
- Estado (registered, active, inactive, deprecated, error)
- Configuración y métricas de rendimiento
- Contador de uso y última vez usado

✅ **Persistencia**
- Guardado automático en JSON
- Carga automática al iniciar
- Sincronización con disco

✅ **Búsqueda y Filtrado**
- Búsqueda por query
- Filtrado por tipo, estado, tags
- Estadísticas del registro

✅ **Gestión de Estado**
- Cambiar estado de pipelines
- Deprecar pipelines con razón
- Remover pipelines

#### Uso:

```python
from core.pipeline_registry import get_registry, PipelineStatus

registry = get_registry(persist_path="./pipelines_registry.json")

# Registrar con metadatos
registry.register(
    "mi_pipeline",
    pipeline,
    pipeline_type="modular",
    metadata={
        "version": "1.2.0",
        "description": "Pipeline de procesamiento",
        "author": "Equipo AI",
        "tags": ["processing", "ml"],
        "dependencies": ["numpy", "torch"]
    }
)

# Buscar pipelines
results = registry.search("processing")

# Filtrar por tipo
modular_pipelines = registry.list_pipelines(pipeline_type="modular")

# Obtener estadísticas
stats = registry.get_statistics()
```

### 2. Pipeline Monitor (`core/pipeline_monitor.py`)

Sistema de monitoreo avanzado con salud, alertas y dashboards.

#### Características:

✅ **Monitoreo de Salud**
- Tasa de éxito/error
- Tiempo promedio de ejecución
- Última ejecución exitosa
- Detección automática de problemas

✅ **Sistema de Alertas**
- 4 niveles: INFO, WARNING, ERROR, CRITICAL
- Umbrales configurables
- Handlers personalizados
- Historial de alertas

✅ **Métricas en Tiempo Real**
- Historial de ejecuciones
- Estadísticas por pipeline
- Estadísticas globales
- Monitoreo periódico opcional

✅ **Integración Automática**
- Se integra automáticamente con el orquestador
- Registro automático de ejecuciones
- Alertas automáticas

#### Uso:

```python
from core.pipeline_monitor import get_monitor, AlertLevel

monitor = get_monitor(
    alert_thresholds={
        'error_rate': 0.1,
        'avg_time': 60.0,
        'max_time': 300.0
    }
)

# Registrar handler de alertas
def alert_handler(alert):
    print(f"ALERTA [{alert.level.value}]: {alert.message}")

monitor.register_alert_handler(alert_handler)

# Registrar ejecución (automático con orquestador)
monitor.record_execution("mi_pipeline", success=True, execution_time=45.0)

# Obtener salud
health = monitor.get_health("mi_pipeline")
print(f"Salud: {health.is_healthy}, Tasa de éxito: {health.success_rate:.2%}")

# Obtener alertas recientes
alerts = monitor.get_recent_alerts(level=AlertLevel.ERROR, limit=10)

# Iniciar monitoreo periódico
monitor.start_monitoring(interval=60.0)
```

### 3. Integración con Orchestrator

✅ **Registro Automático**
- Los pipelines se registran automáticamente en el registry
- Metadatos básicos generados automáticamente

✅ **Monitoreo Automático**
- Las ejecuciones se registran automáticamente en el monitor
- Alertas generadas automáticamente

✅ **Sin Cambios Necesarios**
- Compatible con código existente
- Integración transparente

## Arquitectura Completa

```
┌─────────────────────────────────────────┐
│     Unified Orchestrator                │
│  - Gestión centralizada                 │
│  - Ejecución asíncrona                  │
└─────────────────────────────────────────┘
           │           │
           ▼           ▼
    ┌──────────┐ ┌──────────┐
    │ Registry │ │ Monitor  │
    │ - Metadatos│ │ - Salud  │
    │ - Versión │ │ - Alertas│
    │ - Estado  │ │ - Métricas│
    └──────────┘ └──────────┘
```

## Ejemplo Completo

```python
from core.orchestrator import get_orchestrator, PipelineType
from core.pipeline_registry import get_registry
from core.pipeline_monitor import get_monitor

# 1. Configurar componentes
orchestrator = get_orchestrator()
registry = get_registry("./pipelines_registry.json")
monitor = get_monitor()

# 2. Registrar pipeline con metadatos
pipeline = create_my_pipeline()
orchestrator.register_pipeline("mi_pipeline", pipeline, PipelineType.MODULAR)

# El pipeline se registra automáticamente en el registry

# 3. Configurar alertas
def handle_alert(alert):
    if alert.level == AlertLevel.CRITICAL:
        send_notification(alert)

monitor.register_alert_handler(handle_alert)
monitor.start_monitoring()

# 4. Ejecutar pipeline
result = await orchestrator.execute_pipeline("mi_pipeline", data)

# La ejecución se registra automáticamente en el monitor

# 5. Consultar información
# Salud
health = monitor.get_health("mi_pipeline")

# Metadatos
metadata = registry.get_metadata("mi_pipeline")

# Estadísticas
stats = registry.get_statistics()
monitor_stats = monitor.get_statistics("mi_pipeline")
```

## Beneficios

1. **Visibilidad**: Conocimiento completo del estado de todos los pipelines
2. **Trazabilidad**: Historial completo de ejecuciones y cambios
3. **Alertas Proactivas**: Detección temprana de problemas
4. **Gestión**: Control centralizado de versiones y estados
5. **Análisis**: Estadísticas y métricas para optimización

## Estado

✅ **Completado**

Sistema de registro y monitoreo completamente funcional e integrado.

