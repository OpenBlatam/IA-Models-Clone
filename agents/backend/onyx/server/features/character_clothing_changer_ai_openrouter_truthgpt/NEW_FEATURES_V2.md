# ✨ Nuevas Funcionalidades V2 - Character Clothing Changer AI

## 🎉 Funcionalidades Agregadas

### 1. 📋 Sistema de Job Queue

**Archivo:** `services/job_queue_service.py`

**Características:**
- ✅ Cola de trabajos con prioridades
- ✅ Múltiples workers
- ✅ Retry automático
- ✅ Tracking de estado de jobs
- ✅ Estadísticas completas

**Uso:**
```python
from services.job_queue_service import get_job_queue_service, JobPriority

queue = get_job_queue_service()

# Registrar handler
async def process_image(payload: Dict) -> Dict:
    # Procesar imagen
    return {"result": "processed"}

queue.register_handler("process_image", process_image)

# Iniciar workers
await queue.start()

# Encolar job
job = queue.enqueue(
    job_type="process_image",
    payload={"image_url": "https://..."},
    priority=JobPriority.HIGH,
    max_retries=3
)

# Obtener estado
job = queue.get_job(job.id)
print(f"Status: {job.status}")

# Estadísticas
stats = queue.get_statistics()
```

### 2. ⏰ Sistema de Scheduler

**Archivo:** `services/scheduler_service.py`

**Características:**
- ✅ Tareas one-time
- ✅ Tareas con intervalos
- ✅ Tareas diarias/semanales
- ✅ Tracking de ejecuciones
- ✅ Manejo de errores

**Uso:**
```python
from services.scheduler_service import get_scheduler_service, ScheduleType
from datetime import datetime, timedelta

scheduler = get_scheduler_service()

# Tarea one-time
async def cleanup_task():
    # Limpiar datos antiguos
    pass

scheduler.schedule_once(
    name="Cleanup",
    task=cleanup_task,
    run_at=datetime.now() + timedelta(hours=1)
)

# Tarea con intervalo
scheduler.schedule_interval(
    name="Health Check",
    task=health_check_task,
    interval_seconds=60,
    start_immediately=True
)

# Tarea diaria
scheduler.schedule_daily(
    name="Daily Report",
    task=generate_report_task,
    time="09:00"
)

# Iniciar scheduler
await scheduler.start()
```

### 3. 📡 Sistema de Event Bus

**Archivo:** `services/event_bus_service.py`

**Características:**
- ✅ Patrón pub/sub
- ✅ Múltiples suscriptores por evento
- ✅ Manejo asíncrono de eventos
- ✅ Historial de eventos
- ✅ Estadísticas

**Uso:**
```python
from services.event_bus_service import get_event_bus_service

event_bus = get_event_bus_service()

# Suscribirse a eventos
async def handle_workflow_completed(event):
    print(f"Workflow {event.payload['prompt_id']} completed")

event_bus.subscribe("workflow.completed", handle_workflow_completed)

# Publicar evento
await event_bus.publish(
    event_type="workflow.completed",
    payload={"prompt_id": "prompt_123", "image_url": "https://..."},
    source="clothing_service"
)

# Historial
history = event_bus.get_event_history(event_type="workflow.completed", limit=10)
```

### 4. 🔌 Sistema de Circuit Breaker

**Archivo:** `services/circuit_breaker_service.py`

**Características:**
- ✅ Patrón circuit breaker
- ✅ Estados: CLOSED, OPEN, HALF_OPEN
- ✅ Thresholds configurables
- ✅ Timeout automático
- ✅ Estadísticas por breaker

**Uso:**
```python
from services.circuit_breaker_service import (
    get_circuit_breaker_service,
    CircuitBreakerConfig
)

breaker_service = get_circuit_breaker_service()

# Crear breaker
config = CircuitBreakerConfig(
    failure_threshold=5,
    success_threshold=2,
    timeout_seconds=60.0
)

breaker = breaker_service.get_breaker("comfyui_api", config)

# Usar breaker
try:
    result = await breaker.call(lambda: comfyui_client.execute_workflow(...))
except Exception as e:
    # Circuit breaker rechazó o falló
    print(f"Error: {e}")

# Estadísticas
stats = breaker.get_stats()
print(f"State: {stats['state']}, Failures: {stats['failures']}")
```

### 5. 🚩 Sistema de Feature Flags

**Archivo:** `services/feature_flags_service.py`

**Características:**
- ✅ Flags booleanos (on/off)
- ✅ Rollout por porcentaje
- ✅ Flags dirigidos (usuarios/condiciones)
- ✅ Metadata en flags
- ✅ Estadísticas de uso

**Uso:**
```python
from services.feature_flags_service import (
    get_feature_flags_service,
    FeatureFlagType
)

flags = get_feature_flags_service()

# Crear flag booleano
flags.create_flag(
    name="new_ui",
    flag_type=FeatureFlagType.BOOLEAN,
    enabled=True
)

# Crear flag con porcentaje
flags.create_flag(
    name="beta_feature",
    flag_type=FeatureFlagType.PERCENTAGE,
    percentage=25.0  # 25% de usuarios
)

# Crear flag dirigido
flags.create_flag(
    name="vip_feature",
    flag_type=FeatureFlagType.TARGETED,
    target_users=["user_123", "user_456"]
)

# Verificar flag
if flags.is_enabled("new_ui", user_id="user_123"):
    # Usar nueva UI
    pass

# Actualizar flag
flags.update_flag("beta_feature", percentage=50.0)
```

## 📊 Resumen de Servicios

### Nuevos Servicios Creados:

1. **`services/job_queue_service.py`** - Job queue con prioridades
2. **`services/scheduler_service.py`** - Scheduler de tareas
3. **`services/event_bus_service.py`** - Event bus pub/sub
4. **`services/circuit_breaker_service.py`** - Circuit breaker
5. **`services/feature_flags_service.py`** - Feature flags

## 🎯 Beneficios

### 1. Escalabilidad
- ✅ Job queue para procesamiento asíncrono
- ✅ Scheduler para tareas periódicas
- ✅ Event bus para desacoplamiento

### 2. Resiliencia
- ✅ Circuit breaker protege servicios
- ✅ Retry automático en job queue
- ✅ Manejo de errores robusto

### 3. Flexibilidad
- ✅ Feature flags para control de features
- ✅ Event bus para integraciones
- ✅ Scheduler para automatización

### 4. Observabilidad
- ✅ Estadísticas en todos los servicios
- ✅ Tracking de estados
- ✅ Historial de eventos

## 🚀 Integración con Servicios Existentes

### Job Queue en BatchService
```python
from services.job_queue_service import get_job_queue_service, JobPriority

queue = get_job_queue_service()
queue.register_handler("clothing_change", process_clothing_change)

# En lugar de procesar directamente, encolar
job = queue.enqueue(
    job_type="clothing_change",
    payload={"image_url": "...", "clothing": "..."},
    priority=JobPriority.NORMAL
)
```

### Event Bus en ClothingService
```python
from services.event_bus_service import get_event_bus_service

event_bus = get_event_bus_service()

# Publicar cuando workflow completa
await event_bus.publish(
    event_type="workflow.completed",
    payload={"prompt_id": prompt_id, "result": result}
)
```

### Circuit Breaker en ComfyUIService
```python
from services.circuit_breaker_service import get_circuit_breaker_service

breaker_service = get_circuit_breaker_service()
breaker = breaker_service.get_breaker("comfyui")

# Proteger llamadas a API
result = await breaker.call(lambda: self._execute_workflow(...))
```

### Feature Flags en API
```python
from services.feature_flags_service import get_feature_flags_service

flags = get_feature_flags_service()

@app.post("/api/v1/clothing/change")
async def change_clothing(request: Request):
    # Verificar feature flag
    if not flags.is_enabled("clothing_change_v2", user_id=request.user_id):
        return {"error": "Feature not available"}
    
    # ... procesar request
```

## ✅ Estado

**COMPLETADO** - 5 nuevos servicios avanzados agregados y documentados.

