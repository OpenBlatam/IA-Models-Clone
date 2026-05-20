# Mejoras V18 - Autenticación, Webhooks y Scheduling

## Resumen Ejecutivo

Esta versión introduce mejoras significativas en autenticación y autorización con API keys, sistema de webhooks para integraciones externas, y sistema de scheduling de tareas para automatización avanzada.

## 🎯 Mejoras Implementadas

### 1. Sistema de Autenticación y Autorización

**Archivo**: `core/services/auth_service.py`

- **API Keys**: Generación y gestión de API keys
- **Roles y Permisos**: Sistema de roles (admin, user, readonly, service) y permisos granulares
- **Validación**: Validación de API keys con expiración
- **Revocación**: Revocación de API keys
- **Estadísticas**: Tracking de uso de API keys

**Roles Disponibles**:
- `ADMIN`: Todos los permisos
- `USER`: Permisos de usuario estándar
- `READONLY`: Solo lectura
- `SERVICE`: Permisos para servicios internos

**Permisos Granulares**:
- `task:create`, `task:read`, `task:update`, `task:delete`
- `agent:start`, `agent:stop`, `agent:pause`, `agent:resume`
- `github:connect`, `github:read`
- `llm:generate`, `llm:analyze`
- `admin:config`, `admin:users`, `admin:audit`

**Ejemplo de Uso**:
```python
from core.services import AuthService, UserRole, Permission
from config.di_setup import get_service

auth_service: AuthService = get_service("auth_service")

# Generar API key
key_id, plain_key = auth_service.generate_api_key(
    user_id="user-123",
    roles=[UserRole.USER],
    expires_in_days=30
)

# Validar API key
api_key = auth_service.validate_api_key(plain_key)
if api_key and auth_service.has_permission(api_key, Permission.TASK_CREATE):
    # Permitir crear tarea
    pass
```

### 2. Sistema de Webhooks

**Archivo**: `core/services/webhook_service.py`

- **Registro de Webhooks**: Múltiples webhooks con diferentes eventos
- **Firma HMAC**: Firma de payloads con secret para seguridad
- **Retry Logic**: Reintentos automáticos con backoff
- **Estadísticas**: Tracking de envíos exitosos/fallidos
- **Integración con Eventos**: Disparo automático desde sistema de eventos

**Eventos Disponibles**:
- `TASK_CREATED`, `TASK_UPDATED`, `TASK_COMPLETED`, `TASK_FAILED`
- `AGENT_STARTED`, `AGENT_STOPPED`, `AGENT_PAUSED`, `AGENT_RESUMED`

**Ejemplo de Uso**:
```python
from core.services import WebhookService, WebhookEvent
from config.di_setup import get_service

webhook_service: WebhookService = get_service("webhook_service")

# Registrar webhook
webhook = webhook_service.register_webhook(
    url="https://example.com/webhook",
    events=[WebhookEvent.TASK_CREATED, WebhookEvent.TASK_COMPLETED],
    secret="my-secret-key"
)

# Disparar evento manualmente
result = await webhook_service.trigger_event(
    WebhookEvent.TASK_CREATED,
    {"task_id": "123", "status": "completed"}
)
```

### 3. Sistema de Scheduling de Tareas

**Archivo**: `core/services/scheduler_service.py`

- **Múltiples Tipos**: Once, Interval, Daily, Weekly
- **Configuración Flexible**: Configuración personalizada por tipo
- **Max Runs**: Límite de ejecuciones
- **Auto-Calculation**: Cálculo automático de próxima ejecución
- **Integración**: Handler personalizable para ejecutar tareas

**Tipos de Schedule**:
- `ONCE`: Ejecutar una vez con delay
- `INTERVAL`: Ejecutar cada X segundos
- `DAILY`: Ejecutar diariamente a hora específica
- `WEEKLY`: Ejecutar semanalmente en día específico

**Ejemplo de Uso**:
```python
from core.services import SchedulerService, ScheduleType
from config.di_setup import get_service

scheduler_service: SchedulerService = get_service("scheduler_service")

# Programar tarea diaria
scheduler_service.schedule_task(
    task_id="daily-backup",
    schedule_type=ScheduleType.DAILY,
    schedule_config={"hour": 2, "minute": 0},  # 2:00 AM
    task_data={
        "repository_owner": "owner",
        "repository_name": "repo",
        "instruction": "Create backup"
    }
)

# Programar tarea con intervalo
scheduler_service.schedule_task(
    task_id="health-check",
    schedule_type=ScheduleType.INTERVAL,
    schedule_config={"interval_seconds": 3600},  # Cada hora
    task_data={"type": "health_check"},
    max_runs=100
)
```

### 4. Rutas de API para Autenticación

**Archivo**: `api/routes/auth_routes.py`

- `POST /api/v1/auth/keys` - Crear API key
- `GET /api/v1/auth/keys` - Listar API keys
- `DELETE /api/v1/auth/keys/{key_id}` - Revocar API key
- `GET /api/v1/auth/me` - Obtener usuario actual
- `GET /api/v1/auth/stats` - Estadísticas de autenticación

**Dependency para Requerir Permisos**:
```python
from api.routes.auth_routes import require_permission
from core.services import Permission

@router.post("/tasks")
async def create_task(
    current_key: dict = Depends(require_permission(Permission.TASK_CREATE))
):
    # Solo usuarios con permiso TASK_CREATE pueden acceder
    pass
```

### 5. Rutas de API para Webhooks

**Archivo**: `api/routes/webhook_routes.py`

- `POST /api/v1/webhooks` - Crear webhook
- `GET /api/v1/webhooks` - Listar webhooks
- `GET /api/v1/webhooks/{webhook_id}` - Obtener webhook
- `POST /api/v1/webhooks/{webhook_id}/enable` - Habilitar webhook
- `POST /api/v1/webhooks/{webhook_id}/disable` - Deshabilitar webhook
- `DELETE /api/v1/webhooks/{webhook_id}` - Eliminar webhook
- `POST /api/v1/webhooks/trigger/{event}` - Disparar evento manualmente
- `GET /api/v1/webhooks/stats` - Estadísticas

### 6. Rutas de API para Scheduler

**Archivo**: `api/routes/scheduler_routes.py`

- `POST /api/v1/scheduler/tasks` - Programar tarea
- `GET /api/v1/scheduler/tasks` - Listar tareas programadas
- `GET /api/v1/scheduler/tasks/{task_id}` - Obtener tarea programada
- `POST /api/v1/scheduler/tasks/{task_id}/enable` - Habilitar tarea
- `POST /api/v1/scheduler/tasks/{task_id}/disable` - Deshabilitar tarea
- `DELETE /api/v1/scheduler/tasks/{task_id}` - Eliminar tarea
- `GET /api/v1/scheduler/stats` - Estadísticas

## 📊 Impacto y Beneficios

### Seguridad
- **Autenticación**: Sistema robusto de API keys
- **Autorización**: Control granular de permisos
- **Webhooks Seguros**: Firma HMAC para verificación

### Integración
- **Webhooks**: Integración fácil con sistemas externos
- **Scheduling**: Automatización de tareas recurrentes
- **Eventos**: Disparo automático de webhooks

### Automatización
- **Tareas Programadas**: Ejecución automática de tareas
- **Múltiples Tipos**: Flexibilidad en scheduling
- **Max Runs**: Control de ejecuciones

## 🔄 Integración

### Dependency Injection

Los nuevos servicios están registrados en el DI container:

```python
from config.di_setup import get_service

# Obtener servicios
auth_service = get_service("auth_service")
webhook_service = get_service("webhook_service")
scheduler_service = get_service("scheduler_service")
```

### Inicialización Automática

- **Scheduler**: Se inicia automáticamente en startup
- **Webhooks**: Se integran automáticamente con sistema de eventos

### Integración con Eventos

Los webhooks se disparan automáticamente cuando ocurren eventos:

```python
# Al crear tarea, se dispara automáticamente a webhooks suscritos
await publish_task_event(EventType.TASK_CREATED, task_data)
# → Webhooks suscritos reciben el evento
```

## 📝 Ejemplos de Uso

### Autenticación

```python
# Crear API key
response = await client.post("/api/v1/auth/keys", json={
    "user_id": "user-123",
    "roles": ["user"],
    "expires_in_days": 30
})
api_key = response.json()["api_key"]

# Usar API key
headers = {"Authorization": f"Bearer {api_key}"}
response = await client.post("/api/v1/tasks", json=task_data, headers=headers)
```

### Webhooks

```python
# Registrar webhook
webhook = await client.post("/api/v1/webhooks", json={
    "url": "https://example.com/webhook",
    "events": ["task.created", "task.completed"],
    "secret": "my-secret"
})

# Verificar firma en el webhook receptor
import hmac
signature = request.headers.get("X-Webhook-Signature")
expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
assert signature == f"sha256={expected}"
```

### Scheduling

```python
# Programar tarea diaria
await client.post("/api/v1/scheduler/tasks", json={
    "task_id": "daily-report",
    "schedule_type": "daily",
    "schedule_config": {"hour": 9, "minute": 0},
    "task_data": {
        "repository_owner": "owner",
        "repository_name": "repo",
        "instruction": "Generate daily report"
    }
})

# Programar tarea con intervalo
await client.post("/api/v1/scheduler/tasks", json={
    "task_id": "health-check",
    "schedule_type": "interval",
    "schedule_config": {"interval_seconds": 3600},
    "task_data": {"type": "health_check"},
    "max_runs": 1000
})
```

## 🧪 Testing

### Tests Recomendados

1. **Auth Service**:
   - Generación de API keys
   - Validación de API keys
   - Verificación de permisos
   - Revocación de keys

2. **Webhook Service**:
   - Registro de webhooks
   - Envío de webhooks
   - Firma HMAC
   - Retry logic

3. **Scheduler Service**:
   - Programación de tareas
   - Cálculo de próxima ejecución
   - Ejecución de tareas
   - Max runs

## 📚 Documentación Relacionada

- `IMPROVEMENTS_V17.md` - Documentación OpenAPI y Testing
- `IMPROVEMENTS_V16.md` - Base de Datos Avanzada
- `FRONTEND_INTEGRATION.md` - Integración frontend

## 🚀 Próximos Pasos

Posibles mejoras futuras:
- [ ] OAuth2 / JWT tokens
- [ ] Refresh tokens
- [ ] Rate limiting por API key
- [ ] Webhook retry con exponential backoff mejorado
- [ ] Cron expressions para scheduling
- [ ] Timezone support en scheduling
- [ ] Webhook testing endpoint

## ✅ Checklist de Implementación

- [x] Servicio de autenticación
- [x] Servicio de webhooks
- [x] Servicio de scheduler
- [x] Rutas de API para autenticación
- [x] Rutas de API para webhooks
- [x] Rutas de API para scheduler
- [x] Integración con DI container
- [x] Inicialización automática
- [x] Integración con eventos
- [x] Documentación

---

**Versión**: 18.0  
**Fecha**: 2024-01-01  
**Autor**: GitHub Autonomous Agent Team
