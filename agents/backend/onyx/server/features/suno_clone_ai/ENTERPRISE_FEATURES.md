# Funcionalidades Enterprise - Suno Clone AI

Este documento describe las funcionalidades enterprise implementadas en el sistema.

## 📋 Tabla de Contenidos

1. [Sistema de Colas de Tareas](#sistema-de-colas-de-tareas)
2. [Sistema de Notificaciones Avanzado](#sistema-de-notificaciones-avanzado)
3. [Caché Distribuido](#caché-distribuido)
4. [Logging Estructurado](#logging-estructurado)
5. [API de Administración](#api-de-administración)
6. [Sistema de Backup y Recovery](#sistema-de-backup-y-recovery)

---

## 🎯 Sistema de Colas de Tareas

### Descripción

Sistema completo de colas de tareas con soporte para Celery y Redis, ideal para procesamiento asíncrono de generaciones de música.

**Ubicación**: `services/task_queue.py`

### Características

- ✅ Soporte para Celery (opcional)
- ✅ Cola en memoria como fallback
- ✅ Prioridades de tareas (LOW, NORMAL, HIGH, CRITICAL)
- ✅ Estados de tareas (PENDING, QUEUED, PROCESSING, COMPLETED, FAILED, RETRYING, CANCELLED)
- ✅ Retry automático configurable
- ✅ Monitoreo de tareas

### Uso

```python
from services.task_queue import get_task_queue, TaskPriority

# Obtener cola
task_queue = get_task_queue(use_celery=True, redis_url="redis://localhost:6379/0")

# Encolar tarea
task_id = task_queue.enqueue(
    task_type="generate_music",
    payload={"prompt": "A happy song", "duration": 30},
    priority=TaskPriority.HIGH,
    max_retries=3
)

# Obtener estado
task = task_queue.get_task(task_id)
print(task.status)

# Estadísticas
stats = task_queue.get_queue_stats()
```

### Configuración

```bash
# Redis URL para Celery
REDIS_URL=redis://localhost:6379/0

# Usar Celery (True/False)
USE_CELERY=true
```

---

## 📢 Sistema de Notificaciones Avanzado

### Descripción

Sistema de notificaciones multi-canal que soporta WebSocket, Email, Push, SMS y Webhooks.

**Ubicación**: `services/notification_service_advanced.py`

### Canales Soportados

- **WebSocket**: Notificaciones en tiempo real
- **Email**: Notificaciones por correo electrónico
- **Push**: Notificaciones push (FCM, APNS)
- **SMS**: Notificaciones SMS (futuro)
- **Webhook**: Notificaciones vía webhooks

### Uso

```python
from services.notification_service_advanced import (
    get_notification_service,
    NotificationChannel,
    NotificationPriority
)

service = get_notification_service()

# Notificación simple
notif_id = await service.send_notification(
    user_id="user123",
    title="Song Generated",
    message="Your song is ready!",
    channel=NotificationChannel.WEBSOCKET,
    priority=NotificationPriority.HIGH
)

# Notificación multi-canal
notif_ids = await service.send_multichannel(
    user_id="user123",
    title="Song Generated",
    message="Your song is ready!",
    channels=[
        NotificationChannel.WEBSOCKET,
        NotificationChannel.EMAIL,
        NotificationChannel.PUSH
    ],
    priority=NotificationPriority.URGENT
)

# Obtener notificaciones de usuario
notifications = service.get_user_notifications(
    user_id="user123",
    unread_only=True,
    limit=50
)
```

---

## 💾 Caché Distribuido

### Descripción

Sistema de caché distribuido usando Redis, con fallback a caché en memoria.

**Ubicación**: `utils/distributed_cache.py`

### Características

- ✅ Caché distribuido con Redis
- ✅ Fallback a caché en memoria
- ✅ TTL configurable
- ✅ Invalidación por patrón
- ✅ Operaciones atómicas (increment)
- ✅ Estadísticas del caché

### Uso

```python
from utils.distributed_cache import get_distributed_cache

cache = get_distributed_cache(redis_url="redis://localhost:6379/0")

# Almacenar
cache.set("user:123:profile", {"name": "John"}, ttl=3600)

# Obtener
profile = cache.get("user:123:profile")

# Eliminar
cache.delete("user:123:profile")

# Limpiar por patrón
cache.clear_pattern("user:*")

# Incrementar
views = cache.increment("song:456:views", amount=1)

# Estadísticas
stats = cache.get_stats()
```

### Configuración

```bash
REDIS_URL=redis://localhost:6379/0
```

---

## 📝 Logging Estructurado

### Descripción

Sistema de logging estructurado que produce logs en formato JSON, ideal para sistemas de logging centralizados.

**Ubicación**: `utils/structured_logging.py`

### Características

- ✅ Logs en formato JSON
- ✅ Contexto automático
- ✅ Integración con sistemas centralizados
- ✅ Decorators para logging automático

### Uso

```python
from utils.structured_logging import (
    setup_structured_logging,
    get_logger,
    log_function_call
)

# Configurar logging
setup_structured_logging(
    level="INFO",
    output_format="json",
    output_file="logs/app.log"
)

# Logger con contexto
logger = get_logger(__name__)
logger.set_context(user_id="user123", request_id="req456")
logger.info("Processing request")

# Decorator para logging automático
@log_function_call
async def generate_song(prompt: str):
    # Tu código aquí
    pass
```

### Formato de Log

```json
{
  "timestamp": "2024-01-15T10:30:00.123456",
  "level": "INFO",
  "logger": "suno_clone_ai.api",
  "message": "Processing request",
  "module": "generation",
  "function": "generate_song",
  "line": 42,
  "context": {
    "user_id": "user123",
    "request_id": "req456"
  }
}
```

---

## 🛠️ API de Administración

### Descripción

API completa para administración del sistema, incluyendo gestión de tareas, notificaciones, caché y más.

**Ubicación**: `api/routes/admin.py`

### Endpoints

#### Estadísticas del Sistema
```
GET /suno/admin/stats
```

#### Gestión de Tareas
```
GET /suno/admin/tasks?status_filter=processing&limit=100
GET /suno/admin/tasks/{task_id}
POST /suno/admin/tasks/{task_id}/cancel
```

#### Gestión de Caché
```
POST /suno/admin/cache/clear?pattern=user:*
```

#### Gestión de Notificaciones
```
GET /suno/admin/notifications?user_id=user123&unread_only=true
```

#### Mantenimiento
```
POST /suno/admin/maintenance/cleanup?days=30
GET /suno/admin/alerts?hours=24
```

### Autenticación

Todos los endpoints requieren rol `admin`:

```python
from middleware.auth_middleware import require_role

@router.get("/admin/stats")
async def get_stats(user: dict = Depends(require_role("admin"))):
    # ...
```

---

## 💾 Sistema de Backup y Recovery

### Descripción

Sistema completo de backup y recovery con verificación de integridad.

**Ubicación**: `utils/backup_recovery.py` y `api/routes/backup.py`

### Características

- ✅ Backups completos e incrementales
- ✅ Compresión (gzip)
- ✅ Verificación de integridad (checksums)
- ✅ Restauración de backups
- ✅ Limpieza automática de backups antiguos

### Uso

```python
from utils.backup_recovery import get_backup_manager

backup_manager = get_backup_manager(backup_dir="./backups")

# Crear backup
backup_id = backup_manager.create_backup(
    data={"tasks": {...}, "notifications": {...}},
    backup_type="full",
    description="Daily backup"
)

# Listar backups
backups = backup_manager.list_backups(backup_type="full")

# Verificar integridad
verification = backup_manager.verify_backup(backup_id)

# Restaurar
restored_data = backup_manager.restore_backup(backup_id)

# Limpiar backups antiguos
deleted = backup_manager.cleanup_old_backups(days=30)
```

### API Endpoints

```
POST /suno/admin/backup/create
GET /suno/admin/backup/list
GET /suno/admin/backup/{backup_id}/verify
POST /suno/admin/backup/{backup_id}/restore
DELETE /suno/admin/backup/{backup_id}
POST /suno/admin/backup/cleanup?days=30
```

---

## 🔧 Configuración Completa

### Variables de Entorno

```bash
# Redis
REDIS_URL=redis://localhost:6379/0

# Celery
USE_CELERY=true
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=logs/app.log

# Backup
BACKUP_DIR=./backups
BACKUP_RETENTION_DAYS=30

# Notificaciones
ENABLE_EMAIL_NOTIFICATIONS=true
ENABLE_PUSH_NOTIFICATIONS=true
WEBHOOK_URL=https://example.com/webhook
```

---

## 📊 Monitoreo y Observabilidad

### Métricas Disponibles

- Tareas en cola por estado
- Notificaciones enviadas/entregadas
- Caché hits/misses
- Backups creados/restaurados
- Errores por tipo

### Integración con Prometheus

Todas las métricas están disponibles en `/metrics` para scraping de Prometheus.

---

## 🚀 Próximos Pasos

- [ ] Integración con sistemas de mensajería (RabbitMQ, Kafka)
- [ ] Dashboard de administración web
- [ ] Sistema de permisos granular
- [ ] Auditoría completa de acciones
- [ ] Replicación de backups a S3/GCS
- [ ] Notificaciones SMS con Twilio
- [ ] Integración con sistemas de CI/CD

---

## 📚 Referencias

- [Celery Documentation](https://docs.celeryproject.org/)
- [Redis Documentation](https://redis.io/docs/)
- [Structured Logging Best Practices](https://www.structlog.org/en/stable/)
- [Backup Strategies](https://en.wikipedia.org/wiki/Backup)

