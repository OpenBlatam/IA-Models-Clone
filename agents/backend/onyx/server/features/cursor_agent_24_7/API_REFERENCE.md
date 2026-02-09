# 📡 API Reference - Cursor Agent 24/7

## Base URL

```
http://localhost:8024
```

## Endpoints

### Control del Agente

#### `POST /api/start`
Iniciar el agente.

**Response:**
```json
{
  "status": "started",
  "message": "Agent started successfully"
}
```

#### `POST /api/stop`
Detener el agente.

**Response:**
```json
{
  "status": "stopped",
  "message": "Agent stopped successfully"
}
```

#### `POST /api/pause`
Pausar el agente.

**Response:**
```json
{
  "status": "paused",
  "message": "Agent paused successfully"
}
```

#### `POST /api/resume`
Reanudar el agente.

**Response:**
```json
{
  "status": "resumed",
  "message": "Agent resumed successfully"
}
```

#### `GET /api/status`
Obtener estado completo del agente.

**Response:**
```json
{
  "status": "running",
  "running": true,
  "tasks_total": 10,
  "tasks_pending": 2,
  "tasks_running": 1,
  "tasks_completed": 6,
  "tasks_failed": 1,
  "metrics": { ... },
  "notifications": { ... }
}
```

### Tareas

#### `POST /api/tasks`
Agregar una nueva tarea.

**Request:**
```json
{
  "command": "print('Hello World')"
}
```

**Response:**
```json
{
  "task_id": "task_1234567890_0",
  "status": "pending",
  "message": "Task added successfully"
}
```

#### `GET /api/tasks`
Obtener lista de tareas.

**Query Parameters:**
- `limit` (int, default: 50): Número máximo de tareas

**Response:**
```json
{
  "tasks": [
    {
      "id": "task_1234567890_0",
      "command": "print('Hello')",
      "status": "completed",
      "timestamp": "2024-01-01T12:00:00",
      "result": "Hello",
      "error": null
    }
  ]
}
```

### Programación

#### `GET /api/scheduler/tasks`
Obtener tareas programadas.

**Response:**
```json
{
  "tasks": [
    {
      "id": "scheduled_1234567890_0",
      "name": "Daily Report",
      "command": "print('Report')",
      "schedule_type": "daily",
      "enabled": true,
      "next_run": "2024-01-02T09:00:00",
      "run_count": 5
    }
  ]
}
```

#### `POST /api/scheduler/tasks`
Programar una tarea.

**Query Parameters:**
- `name` (string): Nombre de la tarea
- `command` (string): Comando a ejecutar
- `schedule_type` (string): Tipo (once, interval, daily, weekly, monthly)
- `schedule_value` (string): Valor de programación
- `max_runs` (int, optional): Máximo de ejecuciones

**Response:**
```json
{
  "task_id": "scheduled_1234567890_0",
  "message": "Task scheduled successfully"
}
```

### Backups

#### `GET /api/backups`
Listar backups.

**Response:**
```json
{
  "backups": [
    {
      "name": "backup_20240101_120000",
      "created_at": "2024-01-01T12:00:00",
      "size_mb": 1.5
    }
  ]
}
```

#### `POST /api/backups/create`
Crear backup.

**Query Parameters:**
- `name` (string, optional): Nombre del backup

**Response:**
```json
{
  "success": true,
  "backup_path": "./data/backups/backup_20240101_120000"
}
```

#### `POST /api/backups/{backup_name}/restore`
Restaurar backup.

**Response:**
```json
{
  "success": true
}
```

### Exportación

#### `POST /api/export/tasks`
Exportar tareas.

**Query Parameters:**
- `format` (string): Formato (json, csv, txt)
- `limit` (int, default: 1000): Número máximo de tareas

**Response:**
```json
{
  "success": true,
  "file_path": "./data/exports/tasks_export_20240101_120000.json",
  "format": "json"
}
```

#### `POST /api/export/status`
Exportar estado del agente.

**Response:**
```json
{
  "success": true,
  "file_path": "./data/exports/status_export_20240101_120000.json"
}
```

### Plantillas

#### `GET /api/templates`
Obtener plantillas.

**Query Parameters:**
- `category` (string, optional): Filtrar por categoría

**Response:**
```json
{
  "templates": [
    {
      "id": "hello_world",
      "name": "Hello World",
      "description": "Comando simple",
      "category": "basic",
      "variables": [],
      "usage_count": 5
    }
  ]
}
```

#### `POST /api/templates`
Crear plantilla.

**Request:**
```json
{
  "name": "My Template",
  "description": "Description",
  "template": "print('Hello {name}!')",
  "variables": ["name"],
  "category": "custom"
}
```

#### `POST /api/templates/{template_id}/render`
Renderizar plantilla.

**Request:**
```json
{
  "variables": {
    "name": "World"
  }
}
```

**Response:**
```json
{
  "command": "print('Hello World!')"
}
```

### Monitoreo

#### `GET /api/health`
Health check completo.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "checks": [
    {
      "name": "agent_status",
      "status": "healthy",
      "message": "Agent is running"
    }
  ]
}
```

#### `GET /api/metrics`
Obtener métricas.

**Response:**
```json
{
  "uptime_seconds": 3600,
  "counters": {
    "tasks_added": 100,
    "tasks_completed": 95
  },
  "gauges": {
    "agent_running": 1.0
  }
}
```

#### `GET /api/notifications`
Obtener notificaciones.

**Query Parameters:**
- `limit` (int, default: 50)
- `unread_only` (bool, default: false)

**Response:**
```json
{
  "notifications": [
    {
      "id": "notif_123",
      "title": "Task Completed",
      "message": "Task completed successfully",
      "level": "success",
      "timestamp": "2024-01-01T12:00:00",
      "read": false
    }
  ]
}
```

#### `POST /api/notifications/{notification_id}/read`
Marcar notificación como leída.

**Response:**
```json
{
  "success": true
}
```

### Alertas

#### `GET /api/alerts`
Obtener alertas.

**Query Parameters:**
- `severity` (string, optional): Filtrar por severidad
- `active_only` (bool, default: true): Solo alertas activas

**Response:**
```json
{
  "alerts": [
    {
      "id": "alert_123",
      "rule_name": "High Task Failure Rate",
      "severity": "high",
      "message": "Task failure rate is high",
      "timestamp": "2024-01-01T12:00:00",
      "acknowledged": false,
      "resolved": false
    }
  ]
}
```

#### `POST /api/alerts/{alert_id}/acknowledge`
Reconocer alerta.

**Response:**
```json
{
  "success": true
}
```

#### `POST /api/alerts/{alert_id}/resolve`
Resolver alerta.

**Response:**
```json
{
  "success": true
}
```

### Configuración

#### `GET /api/config`
Obtener configuración.

**Query Parameters:**
- `key` (string, optional): Clave específica (dot notation)

**Response:**
```json
{
  "key": "agent.check_interval",
  "value": 1.0
}
```

#### `POST /api/config`
Establecer configuración.

**Query Parameters:**
- `key` (string): Clave (dot notation)
- `value` (any): Valor

**Response:**
```json
{
  "success": true,
  "message": "Config agent.check_interval updated"
}
```

### Caché

#### `GET /api/cache/stats`
Estadísticas del caché.

**Response:**
```json
{
  "size": 50,
  "max_size": 500,
  "expired_entries": 0,
  "eviction_policy": "lru"
}
```

#### `POST /api/cache/clear`
Limpiar caché.

**Response:**
```json
{
  "success": true,
  "message": "Cache cleared"
}
```

### Rate Limiting

#### `GET /api/rate-limit/stats`
Estadísticas de rate limiting.

**Response:**
```json
{
  "active_tasks": 2,
  "max_concurrent": 10,
  "remaining_this_minute": 58
}
```

### Eventos

#### `GET /api/events`
Obtener eventos.

**Query Parameters:**
- `event_type` (string, optional): Filtrar por tipo
- `limit` (int, default: 100)

**Response:**
```json
{
  "events": [
    {
      "type": "task_completed",
      "data": {"task_id": "task_123"},
      "timestamp": "2024-01-01T12:00:00",
      "source": "agent"
    }
  ]
}
```

#### `GET /api/events/stats`
Estadísticas de eventos.

**Response:**
```json
{
  "total_events": 1000,
  "subscribers": 5,
  "events_by_type": {
    "task_completed": 500,
    "task_failed": 10
  }
}
```

### Cluster

#### `GET /api/cluster/info`
Información del cluster.

**Response:**
```json
{
  "node_id": "abc123",
  "is_leader": true,
  "leader_id": "abc123",
  "total_nodes": 3,
  "active_nodes": 2,
  "nodes": [...]
}
```

### Plugins

#### `GET /api/plugins`
Listar plugins.

**Response:**
```json
{
  "plugins": [
    {
      "name": "LoggingPlugin",
      "enabled": true,
      "class": "LoggingPlugin"
    }
  ]
}
```

### WebSocket

#### `WS /ws`
Conexión WebSocket para comunicación en tiempo real.

**Mensajes enviados:**
```json
{
  "type": "command",
  "command": "print('Hello')"
}
```

**Mensajes recibidos:**
```json
{
  "type": "task_added",
  "task_id": "task_123",
  "command": "print('Hello')"
}
```

## Códigos de Estado HTTP

- `200 OK` - Operación exitosa
- `400 Bad Request` - Solicitud inválida
- `404 Not Found` - Recurso no encontrado
- `500 Internal Server Error` - Error del servidor

## Autenticación

Actualmente no requiere autenticación. Para producción, se recomienda implementar autenticación JWT o API keys.

## Rate Limiting

El agente implementa rate limiting automático:
- Máximo 60 tareas por minuto (configurable)
- Máximo 10 tareas concurrentes (configurable)

## Ejemplos con curl

```bash
# Iniciar agente
curl -X POST http://localhost:8024/api/start

# Agregar tarea
curl -X POST http://localhost:8024/api/tasks \
  -H "Content-Type: application/json" \
  -d '{"command": "print(\"Hello\")"}'

# Ver estado
curl http://localhost:8024/api/status

# Ver health
curl http://localhost:8024/api/health

# Exportar tareas
curl -X POST "http://localhost:8024/api/export/tasks?format=json&limit=100"
```

## Más Información

- Ver [README.md](README.md) para documentación general
- Ver [USAGE.md](USAGE.md) para guía de uso
- Ver [EXAMPLES.md](EXAMPLES.md) para ejemplos



