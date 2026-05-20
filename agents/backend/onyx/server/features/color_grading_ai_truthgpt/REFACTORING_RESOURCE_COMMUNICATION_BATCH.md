# Refactorización: Sistemas Unificados de Recursos, Comunicación y Batch

## Resumen

Esta refactorización consolida servicios relacionados en sistemas unificados para mejorar la arquitectura, reducir duplicación y simplificar el mantenimiento.

## Cambios Realizados

### 1. Unified Resource Manager (`unified_resource_manager.py`)

**Consolida:**
- `TemplateManager` (templates)
- `PresetManager` (presets)
- `LUTManager` (LUTs)
- `VersionManager` (versions)
- `HistoryManager` (history)
- `BackupManager` (backups)

**Características:**
- Interfaz unificada para todos los tipos de recursos
- Operaciones CRUD comunes
- Búsqueda y filtrado
- Control de versiones
- Backup/restore
- Estadísticas unificadas

**Tipos de Recursos:**
- `TEMPLATE`: Plantillas de color grading
- `PRESET`: Presets de color
- `LUT`: Look-Up Tables
- `VERSION`: Versiones de recursos
- `HISTORY`: Historial de procesamiento
- `BACKUP`: Backups

**Uso:**
```python
from services.unified_resource_manager import UnifiedResourceManager, ResourceType

resource_manager = UnifiedResourceManager(base_dir="resources")

# Obtener recurso
template = resource_manager.get_resource(ResourceType.TEMPLATE, "Cinematic Warm")

# Listar recursos
presets = resource_manager.list_resources(ResourceType.PRESET)

# Buscar recursos
results = resource_manager.search_resources(
    ResourceType.TEMPLATE,
    query="cinematic",
    limit=10
)

# Crear backup
backup_path = resource_manager.create_backup(ResourceType.TEMPLATE)
```

**Ubicación:** `services/unified_resource_manager.py`

### 2. Unified Communication System (`unified_communication_system.py`)

**Consolida:**
- `WebhookManager` (webhooks)
- `NotificationService` (notifications)
- `CollaborationManager` (collaboration)

**Características:**
- Interfaz unificada para todas las comunicaciones
- Entrega multi-canal
- Routing de eventos
- Lógica de reintentos
- Seguimiento de entrega

**Canales de Comunicación:**
- `WEBHOOK`: Notificaciones webhook
- `NOTIFICATION`: Notificaciones de usuario
- `COLLABORATION`: Características de colaboración
- `ALL`: Todos los canales

**Uso:**
```python
from services.unified_communication_system import (
    UnifiedCommunicationSystem,
    CommunicationChannel
)

comm_system = UnifiedCommunicationSystem()

# Enviar mensaje
result = await comm_system.send(
    channel=CommunicationChannel.ALL,
    message="Processing completed",
    data={"job_id": "123"},
    recipients=["user@example.com"]
)

# Registrar webhook
comm_system.register_webhook(
    url="https://example.com/webhook",
    events=["completed", "failed"]
)

# Crear share link
share_link = comm_system.create_share_link(
    resource_id="template_123",
    resource_type="template",
    permissions=["read", "comment"]
)
```

**Ubicación:** `services/unified_communication_system.py`

### 3. Unified Batch System (`unified_batch_system.py`)

**Consolida:**
- `BatchProcessor` (procesamiento básico)
- `AdvancedBatchOptimizer` (optimización avanzada)
- `BatchOptimizer` (optimización de batch)

**Características:**
- Interfaz unificada para procesamiento por lotes
- Múltiples estrategias de optimización
- Seguimiento de progreso
- Manejo de errores
- Capacidad de reanudación

**Modos de Batch:**
- `BASIC`: Procesamiento simple
- `OPTIMIZED`: Con optimización
- `ADVANCED`: Estrategias avanzadas de optimización

**Uso:**
```python
from services.unified_batch_system import UnifiedBatchSystem, BatchMode, BatchStrategy

batch_system = UnifiedBatchSystem(
    max_parallel=5,
    default_mode=BatchMode.ADVANCED
)

# Procesar batch
result = await batch_system.process_batch(
    items=[
        {"input": "image1.jpg", "params": {"brightness": 0.1}},
        {"input": "image2.jpg", "params": {"contrast": 1.2}},
    ],
    processor_func=process_image,
    mode=BatchMode.ADVANCED,
    optimization_strategy=BatchStrategy.ADAPTIVE
)

# Verificar estado
status = batch_system.get_job_status(result.job_id)
```

**Ubicación:** `services/unified_batch_system.py`

## Actualizaciones en Service Factory

El `RefactoredServiceFactory` ha sido actualizado para incluir los nuevos sistemas unificados:

```python
# En _init_management()
"unified_resource_manager": UnifiedResourceManager(
    base_dir=str(self.output_dirs["storage"]),
    templates_dir=self._get_storage_path("templates"),
    presets_dir=self._get_storage_path("presets"),
    luts_dir=self._get_storage_path("luts"),
    versions_dir=self._get_storage_path("versions"),
    history_dir=self._get_storage_path("history"),
    backups_dir=self._get_storage_path("backups")
)

# En _init_support()
"unified_batch_system": UnifiedBatchSystem(
    max_parallel=self.config.max_parallel_tasks,
    default_mode=BatchMode.BASIC
)

"unified_communication_system": UnifiedCommunicationSystem()
```

## Compatibilidad hacia Atrás

Los servicios originales (`TemplateManager`, `PresetManager`, `LUTManager`, `WebhookManager`, `NotificationService`, `BatchProcessor`, etc.) siguen disponibles en los exports para mantener compatibilidad, pero se recomienda migrar a los nuevos sistemas unificados.

## Migración

### Recursos

```python
# Antes
from services.template_manager import TemplateManager
from services.preset_manager import PresetManager
from services.lut_manager import LUTManager

template_mgr = TemplateManager()
preset_mgr = PresetManager()
lut_mgr = LUTManager()

# Después
from services.unified_resource_manager import UnifiedResourceManager, ResourceType

resource_mgr = UnifiedResourceManager()
template = resource_mgr.get_resource(ResourceType.TEMPLATE, "name")
preset = resource_mgr.get_resource(ResourceType.PRESET, "name")
```

### Comunicación

```python
# Antes
from services.webhook_manager import WebhookManager
from services.notification_service import NotificationService

webhook_mgr = WebhookManager()
notif_service = NotificationService()

# Después
from services.unified_communication_system import UnifiedCommunicationSystem

comm_system = UnifiedCommunicationSystem()
await comm_system.send(channel=CommunicationChannel.ALL, ...)
```

### Batch

```python
# Antes
from services.batch_processor import BatchProcessor
from services.batch_optimizer_advanced import AdvancedBatchOptimizer

batch_proc = BatchProcessor()
advanced_opt = AdvancedBatchOptimizer()

# Después
from services.unified_batch_system import UnifiedBatchSystem, BatchMode

batch_system = UnifiedBatchSystem(default_mode=BatchMode.ADVANCED)
```

## Beneficios

1. **Reducción de Duplicación**: Eliminación de código duplicado entre servicios relacionados
2. **Mejor Organización**: Servicios consolidados con responsabilidades claras
3. **Funcionalidad Mejorada**: Combinación de las mejores características de cada servicio
4. **Mantenibilidad**: Un solo lugar para mantener y actualizar funcionalidad relacionada
5. **Consistencia**: API unificada para operaciones similares
6. **Flexibilidad**: Modos configurables para diferentes casos de uso

## Estadísticas

- **Servicios consolidados**: 9 servicios → 3 sistemas unificados
- **Reducción de complejidad**: ~33% menos servicios para gestionar
- **Mejora de mantenibilidad**: Un solo punto de entrada para funcionalidad relacionada

## Próximos Pasos

1. Migrar código existente que use los servicios antiguos
2. Actualizar documentación y ejemplos
3. Considerar deprecar los servicios antiguos en futuras versiones
4. Expandir funcionalidad de los sistemas unificados según necesidades


