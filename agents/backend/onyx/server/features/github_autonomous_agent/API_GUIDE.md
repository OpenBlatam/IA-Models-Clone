# 📡 API Guide - GitHub Autonomous Agent

> Guía completa de la API REST del GitHub Autonomous Agent

## 🌐 Endpoints Base

- **Base URL**: `http://localhost:8030`
- **API Version**: `v1`
- **API Prefix**: `/api/v1`

## 📚 Documentación Interactiva

Una vez iniciado el servidor, visita:
- **Swagger UI**: http://localhost:8030/docs
- **ReDoc**: http://localhost:8030/redoc
- **OpenAPI JSON**: http://localhost:8030/openapi.json

## 🔐 Autenticación

La API usa JWT (JSON Web Tokens) para autenticación.

### Obtener Token

```bash
curl -X POST http://localhost:8030/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "usuario",
    "password": "contraseña"
  }'
```

### Usar Token

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8030/api/v1/agent/status
```

## 📋 Endpoints Principales

### 🤖 Agent Endpoints

#### GET `/api/v1/agent/status`
Obtener estado del agente.

**Response:**
```json
{
  "status": "running",
  "active_tasks": 3,
  "queued_tasks": 5,
  "uptime": 3600
}
```

#### POST `/api/v1/agent/start`
Iniciar el agente.

**Request:**
```json
{
  "auto_process": true
}
```

#### POST `/api/v1/agent/stop`
Detener el agente.

#### POST `/api/v1/agent/pause`
Pausar el agente temporalmente.

#### POST `/api/v1/agent/resume`
Reanudar el agente.

---

### 🐙 GitHub Endpoints

#### POST `/api/v1/github/connect`
Conectar a un repositorio de GitHub.

**Request:**
```json
{
  "repo_owner": "usuario",
  "repo_name": "repositorio",
  "branch": "main",
  "auto_sync": true
}
```

**Response:**
```json
{
  "connected": true,
  "repo": "usuario/repositorio",
  "branch": "main",
  "last_sync": "2024-12-01T10:00:00Z"
}
```

#### GET `/api/v1/github/repos`
Listar repositorios disponibles.

**Query Parameters:**
- `type`: `all`, `owner`, `member` (default: `all`)
- `sort`: `created`, `updated`, `pushed`, `full_name` (default: `updated`)
- `direction`: `asc`, `desc` (default: `desc`)

#### GET `/api/v1/github/repo/{owner}/{repo}`
Obtener información de un repositorio específico.

#### GET `/api/v1/github/repo/{owner}/{repo}/branches`
Listar ramas del repositorio.

#### POST `/api/v1/github/repo/{owner}/{repo}/webhook`
Crear webhook para el repositorio.

---

### 📝 Task Endpoints

#### GET `/api/v1/tasks`
Listar todas las tareas.

**Query Parameters:**
- `status`: `pending`, `running`, `completed`, `failed`, `cancelled`
- `priority`: `low`, `medium`, `high`
- `limit`: número (default: 50)
- `offset`: número (default: 0)

**Response:**
```json
{
  "tasks": [
    {
      "id": "task-123",
      "instruction": "Analizar código",
      "status": "running",
      "priority": "high",
      "created_at": "2024-12-01T10:00:00Z",
      "updated_at": "2024-12-01T10:05:00Z"
    }
  ],
  "total": 100,
  "limit": 50,
  "offset": 0
}
```

#### POST `/api/v1/tasks`
Crear una nueva tarea.

**Request:**
```json
{
  "instruction": "Analizar código y generar documentación",
  "priority": "high",
  "metadata": {
    "repo": "usuario/repositorio",
    "branch": "main"
  }
}
```

**Response:**
```json
{
  "id": "task-123",
  "instruction": "Analizar código y generar documentación",
  "status": "pending",
  "priority": "high",
  "created_at": "2024-12-01T10:00:00Z"
}
```

#### GET `/api/v1/tasks/{task_id}`
Obtener detalles de una tarea específica.

#### PUT `/api/v1/tasks/{task_id}`
Actualizar una tarea.

**Request:**
```json
{
  "priority": "low",
  "status": "paused"
}
```

#### DELETE `/api/v1/tasks/{task_id}`
Eliminar una tarea.

#### POST `/api/v1/tasks/{task_id}/cancel`
Cancelar una tarea en ejecución.

#### GET `/api/v1/tasks/{task_id}/logs`
Obtener logs de una tarea.

**Query Parameters:**
- `limit`: número de líneas (default: 100)
- `level`: `DEBUG`, `INFO`, `WARNING`, `ERROR`

---

## 🔄 Webhooks

### GitHub Webhooks

El agente puede recibir webhooks de GitHub para eventos como:
- `push`: Nuevos commits
- `pull_request`: Pull requests
- `issues`: Issues creados/cerrados
- `workflow_run`: Ejecuciones de workflows

**Endpoint**: `POST /api/v1/webhooks/github`

---

## 📊 Monitoring Endpoints

### GET `/api/v1/metrics`
Obtener métricas del sistema.

**Response:**
```json
{
  "tasks": {
    "total": 100,
    "pending": 10,
    "running": 5,
    "completed": 80,
    "failed": 5
  },
  "system": {
    "cpu_percent": 45.2,
    "memory_percent": 60.5,
    "disk_percent": 30.0
  },
  "queue": {
    "size": 15,
    "workers": 4
  }
}
```

### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-12-01T10:00:00Z",
  "services": {
    "database": "ok",
    "redis": "ok",
    "github_api": "ok"
  }
}
```

---

## 🛡️ Rate Limiting

La API implementa rate limiting para prevenir abuso:

- **Default**: 60 requests por minuto por IP
- **Authenticated**: 120 requests por minuto por usuario

**Headers de respuesta:**
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1701432000
```

---

## 📝 Ejemplos de Uso

### Flujo Completo: Conectar y Crear Tarea

```bash
# 1. Conectar repositorio
REPO_RESPONSE=$(curl -X POST http://localhost:8030/api/v1/github/connect \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "repo_owner": "usuario",
    "repo_name": "repositorio",
    "branch": "main"
  }')

# 2. Crear tarea
TASK_RESPONSE=$(curl -X POST http://localhost:8030/api/v1/tasks \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "instruction": "Analizar código y generar documentación",
    "priority": "high"
  }')

TASK_ID=$(echo $TASK_RESPONSE | jq -r '.id')

# 3. Monitorear progreso
while true; do
  STATUS=$(curl -s http://localhost:8030/api/v1/tasks/$TASK_ID \
    -H "Authorization: Bearer $TOKEN" | jq -r '.status')
  
  echo "Task status: $STATUS"
  
  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
    break
  fi
  
  sleep 5
done

# 4. Ver logs
curl http://localhost:8030/api/v1/tasks/$TASK_ID/logs \
  -H "Authorization: Bearer $TOKEN"
```

### Listar Tareas con Filtros

```bash
# Tareas pendientes de alta prioridad
curl "http://localhost:8030/api/v1/tasks?status=pending&priority=high" \
  -H "Authorization: Bearer $TOKEN"

# Tareas completadas (últimas 10)
curl "http://localhost:8030/api/v1/tasks?status=completed&limit=10" \
  -H "Authorization: Bearer $TOKEN"
```

### Monitorear Estado del Agente

```bash
# Estado actual
curl http://localhost:8030/api/v1/agent/status \
  -H "Authorization: Bearer $TOKEN"

# Métricas del sistema
curl http://localhost:8030/api/v1/metrics \
  -H "Authorization: Bearer $TOKEN"
```

---

## 🔍 Búsqueda y Filtrado

### Filtros Disponibles

- **Status**: `pending`, `running`, `completed`, `failed`, `cancelled`
- **Priority**: `low`, `medium`, `high`
- **Date Range**: `created_after`, `created_before`
- **Repository**: `repo_owner`, `repo_name`

### Ejemplo de Búsqueda Avanzada

```bash
curl "http://localhost:8030/api/v1/tasks?status=running&priority=high&limit=20&offset=0" \
  -H "Authorization: Bearer $TOKEN"
```

---

## ⚠️ Códigos de Error

| Código | Significado | Solución |
|--------|-------------|----------|
| `400` | Bad Request | Verificar formato de request |
| `401` | Unauthorized | Verificar token de autenticación |
| `403` | Forbidden | Verificar permisos |
| `404` | Not Found | Recurso no existe |
| `429` | Too Many Requests | Rate limit excedido, esperar |
| `500` | Internal Server Error | Error del servidor, revisar logs |

---

## 📚 Más Información

- **Swagger UI**: http://localhost:8030/docs (cuando el servidor está corriendo)
- **ReDoc**: http://localhost:8030/redoc
- **OpenAPI Schema**: http://localhost:8030/openapi.json

---

**¿Necesitas ayuda?** Consulta la [documentación completa](README.md) o abre un issue.



