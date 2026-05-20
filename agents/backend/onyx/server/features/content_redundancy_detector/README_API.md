# Content Redundancy Detector API

## Estado: ✅ Listo para Usar

Este módulo está completamente configurado y listo para producción.

## Características

- ✅ **Arquitectura Modular**: Separación clara de responsabilidades
- ✅ **Manejo de Errores**: Respuestas consistentes y frontend-friendly
- ✅ **Middleware Optimizado**: Logging, CORS, seguridad, rate limiting
- ✅ **Imports Resilientes**: Fallbacks para módulos opcionales
- ✅ **Webhooks**: Sistema completo de webhooks con compatibilidad hacia atrás
- ✅ **Documentación**: OpenAPI/Swagger disponible en `/docs`

## Inicio Rápido

### Usando app.py (Aplicación tradicional)
```bash
cd agents/backend/onyx/server/features/content_redundancy_detector
python app.py
```

### Usando api/main.py (Arquitectura modular)
```python
from api.main import app

# Run with uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Endpoints Principales

- `GET /` - Información del API
- `GET /docs` - Documentación Swagger
- `GET /api/v1/health` - Health check
- `POST /api/v1/analyze` - Análisis de contenido
- `POST /api/v1/similarity` - Detección de similitud
- `POST /api/v1/quality` - Evaluación de calidad

## Configuración

Los imports están configurados con múltiples niveles de fallback:

1. Import directo desde módulo
2. Import relativo desde mismo directorio
3. Fallback implementations si el módulo no está disponible

### Variables de Entorno

- `ENVIRONMENT`: development/production
- `DEBUG`: true/false
- `CORS_ORIGINS`: Orígenes permitidos (separados por coma)
- `REDIS_URL`: URL de Redis para webhooks
- `WEBHOOK_STORAGE_TYPE`: auto/redis/memory

## Webhooks

El sistema de webhooks está completamente funcional:

```python
from webhooks import send_webhook, WebhookEvent

# Enviar webhook
await send_webhook(
    WebhookEvent.ANALYSIS_COMPLETED,
    {"result": analysis_result},
    request_id="req-123",
    user_id="user-456"
)
```

## Manejo de Errores

Todos los errores se manejan de forma consistente:

```json
{
    "success": false,
    "data": null,
    "error": {
        "message": "Error description",
        "status_code": 400,
        "type": "ValidationError"
    },
    "timestamp": 1234567890.123
}
```

## Middleware

El API incluye los siguientes middlewares (en orden de aplicación):

1. **SecurityMiddleware**: Headers de seguridad
2. **CORSMiddleware**: CORS para frontend
3. **LoggingMiddleware**: Logging estructurado
4. **RateLimitMiddleware**: Rate limiting
5. **PerformanceMiddleware**: Métricas de rendimiento

## Servicios

Los servicios principales están disponibles a través de `ServiceRegistry`:

- Analysis Service
- Cache Service
- ML/AI Service (opcional)
- Webhook Manager
- Export Manager

## Notas de Compatibilidad

- ✅ Compatible con Python 3.8+
- ✅ FastAPI 0.68+
- ✅ Todos los imports tienen fallbacks
- ✅ No requiere dependencias opcionales para funcionar básicamente






