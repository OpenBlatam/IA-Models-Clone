# Características Enterprise - Piel Mejorador AI SAM3

## Resumen

Características avanzadas de nivel enterprise implementadas para producción, incluyendo rate limiting, webhooks, optimización de memoria, y más.

## 🚦 Rate Limiting

### Descripción

Sistema de rate limiting basado en token bucket para proteger la API de abuso.

**Archivo:** `core/rate_limiter.py`

### Características

- ✅ Algoritmo token bucket
- ✅ Rate limiting por cliente (IP)
- ✅ Configuración flexible por endpoint
- ✅ Cálculo de tiempo de espera
- ✅ Estadísticas de rate limiting
- ✅ Limpieza automática de buckets inactivos

### Uso

```python
from piel_mejorador_ai_sam3.core.rate_limiter import RateLimiter, RateLimitConfig

limiter = RateLimiter(
    default_config=RateLimitConfig(
        requests_per_second=10.0,
        burst_size=20
    )
)

# Verificar si request está permitido
if await limiter.is_allowed(client_ip="192.168.1.1"):
    # Procesar request
    pass
else:
    wait_time = await limiter.get_wait_time(client_ip)
    # Retornar 429 Too Many Requests
```

### Configuración en API

La API incluye rate limiting automático por IP:

```bash
# Ver estadísticas
GET /rate-limit/stats

# Respuesta:
{
  "total_requests": 1000,
  "allowed_requests": 950,
  "rate_limited_requests": 50,
  "rate_limit_rate": 0.05,
  "active_buckets": 25
}
```

## 🔔 Sistema de Webhooks

### Descripción

Sistema completo de webhooks para notificaciones asíncronas de eventos.

**Archivo:** `core/webhook_manager.py`

### Características

- ✅ Múltiples webhooks por evento
- ✅ Filtrado de eventos
- ✅ Reintentos con backoff exponencial
- ✅ Firmas HMAC para seguridad
- ✅ Entrega asíncrona
- ✅ Estadísticas de entrega

### Eventos Disponibles

- `task.created` - Tarea creada
- `task.started` - Tarea iniciada
- `task.completed` - Tarea completada
- `task.failed` - Tarea fallida
- `batch.completed` - Procesamiento en lote completado

### Uso

```python
from piel_mejorador_ai_sam3.core.webhook_manager import Webhook, WebhookEvent

# Registrar webhook
agent.register_webhook(
    url="https://example.com/webhook",
    events=[
        WebhookEvent.TASK_COMPLETED,
        WebhookEvent.TASK_FAILED
    ],
    secret="your-secret-key"  # Opcional para HMAC
)
```

### API Endpoints

```bash
# Registrar webhook
POST /webhooks/register
{
  "url": "https://example.com/webhook",
  "events": ["task.completed", "task.failed"],
  "secret": "optional-secret"
}

# Desregistrar webhook
DELETE /webhooks/unregister?url=https://example.com/webhook

# Estadísticas
GET /webhooks/stats
```

### Payload del Webhook

```json
{
  "event": "task.completed",
  "task_id": "abc123",
  "data": {
    "result": {...},
    "tokens_used": 1500
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

### Verificación de Firma

Si se proporciona un secret, el webhook incluye un header `X-Webhook-Signature` con HMAC-SHA256:

```python
import hmac
import hashlib

signature = request.headers.get("X-Webhook-Signature")
expected = hmac.new(
    secret.encode(),
    payload.encode(),
    hashlib.sha256
).hexdigest()

if signature != expected:
    # Rechazar webhook
    pass
```

## 💾 Optimización de Memoria

### Descripción

Sistema avanzado de optimización de memoria para procesamiento de archivos grandes.

**Archivo:** `core/memory_optimizer.py`

### Características

- ✅ Monitoreo de memoria en tiempo real
- ✅ Detección de presión de memoria
- ✅ Limpieza automática
- ✅ Procesamiento en chunks
- ✅ Recomendaciones inteligentes

### Uso

```python
from piel_mejorador_ai_sam3.core.memory_optimizer import MemoryOptimizer

optimizer = MemoryOptimizer(max_memory_percent=80.0)

# Verificar uso de memoria
usage = optimizer.get_memory_usage()
print(f"Memoria del proceso: {usage['process_memory_mb']:.2f}MB")
print(f"Memoria del sistema: {usage['system_memory_percent']:.1f}%")

# Optimizar si es necesario
if optimizer.is_memory_pressure():
    await optimizer.optimize()

# Procesar en chunks
async for chunk_results in optimizer.process_in_chunks(
    items=large_list,
    process_func=process_item,
    chunk_size=10
):
    # Procesar resultados del chunk
    pass
```

### API Endpoints

```bash
# Uso de memoria
GET /memory/usage

# Recomendaciones
GET /memory/recommendations

# Optimizar memoria
POST /memory/optimize?force=false
```

### Recomendaciones Automáticas

El sistema proporciona recomendaciones basadas en el uso de memoria:

- Reducir tamaño de lotes si memoria alta
- Reducir tareas concurrentes si sistema bajo presión
- Liberar recursos si memoria disponible baja

## 📊 Métricas Mejoradas

### Estadísticas Disponibles

```bash
GET /stats

# Retorna:
{
  "executor_stats": {
    "total_tasks": 1000,
    "completed_tasks": 950,
    "failed_tasks": 50,
    "success_rate": 0.95,
    "avg_task_time": 2.5
  },
  "cache_stats": {
    "hits": 200,
    "misses": 50,
    "hit_rate": 0.8,
    "cache_size": 150
  },
  "webhook_stats": {
    "total_sent": 500,
    "successful": 480,
    "failed": 20,
    "success_rate": 0.96
  },
  "memory_usage": {
    "process_memory_mb": 512.5,
    "process_memory_percent": 45.2,
    "system_memory_percent": 65.0
  },
  "running": true,
  "max_parallel_tasks": 5
}
```

## 🔒 Seguridad

### Rate Limiting

- Protección contra abuso de API
- Límites configurables por cliente
- Respuestas HTTP 429 apropiadas

### Webhooks

- Firmas HMAC para verificación
- Reintentos con backoff exponencial
- Timeouts configurables

### Validación

- Validación estricta de parámetros
- Validación de tipos de archivo
- Límites de tamaño de archivo

## 🚀 Optimizaciones de Rendimiento

### 1. Rate Limiting Eficiente

- Algoritmo token bucket O(1)
- Limpieza automática de buckets inactivos
- Mínimo overhead

### 2. Webhooks Asíncronos

- Entrega no bloqueante
- Reintentos automáticos
- Pool de conexiones HTTP

### 3. Optimización de Memoria

- Monitoreo continuo
- Limpieza proactiva
- Procesamiento en chunks

## 📝 Ejemplos de Uso

### Configuración Completa

```python
from piel_mejorador_ai_sam3 import PielMejoradorAgent, PielMejoradorConfig
from piel_mejorador_ai_sam3.core.webhook_manager import WebhookEvent

config = PielMejoradorConfig()
agent = PielMejoradorAgent(config=config)

# Registrar webhook
agent.register_webhook(
    url="https://api.example.com/webhooks/skin-enhancement",
    events=[
        WebhookEvent.TASK_COMPLETED,
        WebhookEvent.TASK_FAILED
    ],
    secret="your-secret-key"
)

# Procesar con optimización automática
task_id = await agent.mejorar_imagen("image.jpg", enhancement_level="high")

# El sistema automáticamente:
# - Envía webhook cuando completa
# - Optimiza memoria si es necesario
# - Usa caché si disponible
```

### Monitoreo en Producción

```python
# Obtener todas las métricas
stats = agent.get_performance_stats()

# Verificar salud del sistema
if stats["memory_usage"]["system_memory_percent"] > 90:
    await agent.optimize_memory(force=True)

# Ver recomendaciones
recommendations = agent.get_memory_recommendations()
for rec in recommendations:
    logger.warning(rec)
```

## 🔧 Configuración

### Variables de Entorno

```bash
# Rate Limiting
PIEL_MEJORADOR_RATE_LIMIT_RPS=10.0
PIEL_MEJORADOR_RATE_LIMIT_BURST=20

# Webhooks
PIEL_MEJORADOR_WEBHOOK_TIMEOUT=10.0
PIEL_MEJORADOR_WEBHOOK_RETRIES=3

# Memory
PIEL_MEJORADOR_MAX_MEMORY_PERCENT=80.0
PIEL_MEJORADOR_MEMORY_CHECK_INTERVAL=60
```

## 📚 Referencias

- Rate Limiting: `core/rate_limiter.py`
- Webhooks: `core/webhook_manager.py`
- Memory Optimization: `core/memory_optimizer.py`
- API: `api/piel_mejorador_api.py`

## 🎯 Mejores Prácticas

1. **Rate Limiting**: Configura límites apropiados para tu carga esperada
2. **Webhooks**: Usa secretos para verificar autenticidad
3. **Memoria**: Monitorea regularmente y optimiza proactivamente
4. **Métricas**: Revisa estadísticas regularmente para optimizar
5. **Seguridad**: Valida siempre los webhooks recibidos




