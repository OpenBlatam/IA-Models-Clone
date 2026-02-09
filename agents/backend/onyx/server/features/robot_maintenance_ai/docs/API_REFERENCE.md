# API Reference - Robot Maintenance AI

## Base URL

```
http://localhost:8000/api/robot-maintenance
```

## Autenticación

Actualmente no se requiere autenticación, pero se recomienda implementar autenticación para producción.

## Rate Limiting

- **Límite**: 100 requests por minuto por IP
- **Header de respuesta**: `Retry-After` cuando se excede el límite
- **Código de error**: `429 Too Many Requests`

## Endpoints

### POST /ask

Hacer una pregunta de mantenimiento al tutor de IA.

**Request Body:**
```json
{
  "question": "¿Cómo cambio el aceite de un robot industrial?",
  "conversation_id": "conv_123",
  "robot_type": "robots_industriales",
  "maintenance_type": "preventivo",
  "difficulty": "intermedio",
  "sensor_data": {
    "temperature": 25.5,
    "pressure": 100.0
  },
  "context": "Contexto adicional"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "answer": "Para cambiar el aceite...",
    "model": "openai/gpt-4-turbo",
    "usage": {
      "prompt_tokens": 150,
      "completion_tokens": 300,
      "total_tokens": 450
    },
    "timestamp": "2024-01-01T12:00:00",
    "nlp_analysis": {...},
    "intent": {...},
    "ml_prediction": {...}
  }
}
```

### POST /procedure

Obtener un procedimiento de mantenimiento detallado.

**Request Body:**
```json
{
  "procedure": "lubricación de ejes",
  "robot_type": "robots_industriales",
  "difficulty": "intermedio"
}
```

### POST /diagnose

Diagnosticar un problema basado en síntomas.

**Request Body:**
```json
{
  "symptoms": "El robot hace ruidos extraños",
  "robot_type": "robots_industriales",
  "sensor_data": {
    "temperature": 85.0,
    "vibration": 6.5
  }
}
```

### POST /predict

Predecir necesidades de mantenimiento usando ML.

**Request Body:**
```json
{
  "robot_type": "robots_industriales",
  "sensor_data": {
    "temperature": 28.5,
    "vibration": 0.15,
    "pressure": 4.2,
    "runtime_hours": 8500
  },
  "historical_data": [...]
}
```

### POST /checklist

Generar una lista de verificación de mantenimiento.

**Request Body:**
```json
{
  "robot_type": "robots_industriales",
  "maintenance_type": "preventivo"
}
```

### GET /conversation/{conversation_id}

Obtener el historial de una conversación.

**Response:**
```json
{
  "success": true,
  "data": {
    "conversation_id": "conv_123",
    "messages": [...],
    "summary": {...}
  }
}
```

### DELETE /conversation/{conversation_id}

Limpiar el historial de una conversación.

### GET /metrics

Obtener métricas del sistema.

**Response:**
```json
{
  "success": true,
  "data": {
    "uptime_seconds": 3600,
    "total_requests": 150,
    "total_errors": 2,
    "cache_hits": 45,
    "cache_misses": 105,
    "cache_hit_rate": 0.3,
    "endpoints": {...}
  }
}
```

### GET /cache/stats

Obtener estadísticas de caché.

### GET /rate-limit/stats

Obtener estadísticas de rate limiting.

### GET /health

Health check del servicio.

## WebSocket API

Para actualizaciones en tiempo real, usa WebSockets:

### WebSocket Endpoint

```
ws://localhost:8000/ws/conversation/{conversation_id}
```

Ver [WEBSOCKETS.md](WEBSOCKETS.md) para documentación completa de WebSockets.

## Códigos de Error

- `400 Bad Request`: Error de validación
- `429 Too Many Requests`: Rate limit excedido
- `500 Internal Server Error`: Error del servidor
- `503 Service Unavailable`: Servicio no disponible
- `504 Gateway Timeout`: Timeout en la llamada a OpenRouter

## Ejemplos de Uso

Ver `examples/basic_usage.py` para ejemplos completos.

## Persistencia de Datos

El sistema incluye una base de datos SQLite opcional para:
- Almacenar conversaciones persistentemente
- Guardar historial de predicciones ML
- Mantener registros de mantenimiento

La base de datos se crea automáticamente en `data/maintenance.db` cuando se usa.

