# 📚 Documentación Completa de API - Social Media Identity Clone AI

## Base URL

```
http://localhost:8000/api/v1
```

## Autenticación

Algunos endpoints requieren API key en el header:

```
X-API-Key: your_api_key_here
```

## Endpoints Principales

### 1. Extraer Perfil

**POST** `/extract-profile`

Extrae perfil de una red social.

**Request:**
```json
{
    "platform": "instagram",
    "username": "username",
    "use_cache": true
}
```

**Response:**
```json
{
    "success": true,
    "platform": "instagram",
    "username": "username",
    "profile": {
        "username": "username",
        "display_name": "Display Name",
        "bio": "Bio text",
        "videos": [...],
        "posts": [...]
    },
    "stats": {
        "videos": 10,
        "posts": 50,
        "comments": 200
    }
}
```

### 2. Construir Identidad

**POST** `/build-identity`

Construye identidad a partir de perfiles.

**Request:**
```json
{
    "tiktok_username": "user",
    "instagram_username": "user",
    "youtube_channel_id": "channel"
}
```

**Response:**
```json
{
    "success": true,
    "identity_id": "uuid",
    "identity": {...},
    "stats": {
        "total_videos": 100,
        "total_posts": 200,
        "topics_count": 15
    }
}
```

### 3. Generar Contenido

**POST** `/generate-content`

Genera contenido basado en identidad.

**Request:**
```json
{
    "identity_profile_id": "uuid",
    "platform": "instagram",
    "content_type": "post",
    "topic": "fitness",
    "style": "motivational"
}
```

**Response:**
```json
{
    "success": true,
    "content_id": "uuid",
    "content": {
        "content": "Generated content...",
        "hashtags": ["#fitness", "#motivation"],
        "platform": "instagram"
    },
    "validation": {
        "is_valid": true,
        "score": 0.85,
        "issues": [],
        "warnings": [],
        "suggestions": []
    }
}
```

## Endpoints de Tareas Asíncronas

### Crear Tarea de Extracción

**POST** `/tasks/extract-profile`

Crea tarea asíncrona para extraer perfil.

**Response:**
```json
{
    "success": true,
    "task_id": "uuid",
    "status": "pending"
}
```

### Obtener Estado de Tarea

**GET** `/tasks/{task_id}`

Obtiene estado de una tarea.

**Response:**
```json
{
    "success": true,
    "task": {
        "task_id": "uuid",
        "status": "completed",
        "result": {...},
        "error": null
    }
}
```

## Endpoints de Analytics

### Métricas del Sistema

**GET** `/metrics`

Obtiene métricas en tiempo real.

**Response:**
```json
{
    "success": true,
    "metrics": {
        "counters": {
            "profile_extraction_requests": 100,
            "content_generation_requests": 50
        },
        "timers": {
            "profile_extraction_duration": 2.5
        }
    }
}
```

### Estadísticas

**GET** `/analytics/stats`

Obtiene estadísticas agregadas.

## Endpoints de Machine Learning

### Predecir Rendimiento

**POST** `/ml/predict-performance`

Predice rendimiento de contenido.

**Request:**
```json
{
    "content": "Contenido a predecir...",
    "platform": "instagram",
    "identity_id": "uuid"
}
```

**Response:**
```json
{
    "success": true,
    "prediction": {
        "predicted_engagement": 0.75,
        "confidence": 0.8,
        "factors": ["Longitud óptima", "Tiene hook de engagement"],
        "recommendation": "Buen contenido, listo para publicar"
    }
}
```

## Endpoints de Colaboración

### Compartir Identidad

**POST** `/collaboration/share`

Comparte identidad con otro usuario.

**Request:**
```json
{
    "identity_id": "uuid",
    "shared_with_user_id": "user123",
    "permission_level": "editor",
    "shared_by_user_id": "owner"
}
```

## Endpoints de Dashboard

### Obtener Dashboard

**GET** `/dashboard`

Obtiene datos completos del dashboard.

**Response:**
```json
{
    "success": true,
    "dashboard": {
        "overview": {
            "total_identities": 50,
            "total_content": 200,
            "content_today": 5
        },
        "content_by_platform": {
            "instagram": 100,
            "tiktok": 50,
            "youtube": 50
        },
        "recent_activity": [...],
        "top_identities": [...]
    }
}
```

## Endpoints de Alertas

### Listar Alertas

**GET** `/alerts?unacknowledged_only=true&severity=critical`

Obtiene alertas con filtros.

**Response:**
```json
{
    "success": true,
    "count": 5,
    "critical_count": 2,
    "alerts": [...]
}
```

## Códigos de Estado HTTP

- `200 OK` - Operación exitosa
- `201 Created` - Recurso creado
- `400 Bad Request` - Solicitud inválida
- `404 Not Found` - Recurso no encontrado
- `500 Internal Server Error` - Error del servidor

## Rate Limiting

- 60 requests por minuto por IP
- 1000 requests por hora por IP

## Ejemplos de Uso

### Python

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"
API_KEY = "your_api_key"

headers = {"X-API-Key": API_KEY}

# Extraer perfil
response = requests.post(
    f"{BASE_URL}/extract-profile",
    json={
        "platform": "instagram",
        "username": "username"
    },
    headers=headers
)
```

### cURL

```bash
curl -X POST http://localhost:8000/api/v1/extract-profile \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "platform": "instagram",
    "username": "username"
  }'
```

## Webhooks

Registra webhooks para recibir notificaciones de eventos:

**POST** `/webhooks/register`

```json
{
    "url": "https://your-app.com/webhook",
    "events": ["identity_created", "content_generated"]
}
```

Eventos disponibles:
- `identity_created`
- `content_generated`
- `task_completed`
- `task_failed`




