# 🔧 LLM Middleware Guide

## Descripción

Este documento describe cómo usar y configurar los middlewares para las rutas LLM.

## Middlewares Disponibles

### 1. LLMRateLimitMiddleware

Middleware de rate limiting automático para rutas LLM.

#### Características
- ✅ Rate limiting por ruta
- ✅ Rate limiting por IP
- ✅ Headers de rate limit en respuestas
- ✅ Configuración por endpoint

#### Uso

```python
from fastapi import FastAPI
from api.middleware import LLMRateLimitMiddleware

app = FastAPI()

# Agregar middleware
app.add_middleware(
    LLMRateLimitMiddleware,
    default_limit=100,
    window_seconds=60
)
```

#### Configuración de Límites

Los límites se configuran automáticamente por ruta:

- `/api/v1/llm/generate`: 50 requests/minuto
- `/api/v1/llm/generate-parallel`: 20 requests/minuto
- `/api/v1/llm/generate-stream`: 30 requests/minuto
- `/api/v1/llm/ab-test/create`: 10 requests/5 minutos
- `/api/v1/llm/tests/create-suite`: 10 requests/5 minutos

#### Headers de Respuesta

El middleware agrega los siguientes headers:

- `X-RateLimit-Limit`: Límite total
- `X-RateLimit-Remaining`: Requests restantes
- `X-RateLimit-Reset`: Timestamp de reset
- `Retry-After`: Segundos hasta el próximo intento (si está limitado)

### 2. LLMLoggingMiddleware

Middleware de logging para requests y responses LLM.

#### Características
- ✅ Logging de requests
- ✅ Logging de responses
- ✅ Medición de latencia
- ✅ Headers de performance

#### Uso

```python
from api.middleware import LLMLoggingMiddleware

app.add_middleware(
    LLMLoggingMiddleware,
    log_requests=True,
    log_responses=False  # Puede ser verbose
)
```

#### Headers Agregados

- `X-Response-Time`: Latencia en milisegundos

### 3. LLMValidationMiddleware

Middleware de validación para requests LLM.

#### Características
- ✅ Validación de Content-Type
- ✅ Validación de tamaño de body
- ✅ Protección contra requests malformados

#### Uso

```python
from api.middleware import LLMValidationMiddleware

app.add_middleware(
    LLMValidationMiddleware,
    max_body_size=10 * 1024 * 1024  # 10MB
)
```

## Orden de Middlewares

El orden importa. Se recomienda este orden:

```python
# 1. Validación (primero)
app.add_middleware(LLMValidationMiddleware)

# 2. Rate Limiting (antes del procesamiento)
app.add_middleware(LLMRateLimitMiddleware)

# 3. Logging (para capturar todo)
app.add_middleware(LLMLoggingMiddleware)
```

## Health Checks

### Endpoints Disponibles

- `GET /api/v1/llm/health` - Health check básico
- `GET /api/v1/llm/health/detailed` - Health check detallado
- `GET /api/v1/llm/health/readiness` - Readiness check
- `GET /api/v1/llm/health/liveness` - Liveness check

### Health Check Básico

```bash
curl http://localhost:8000/api/v1/llm/health
```

Respuesta:
```json
{
  "status": "healthy",
  "service": "llm",
  "timestamp": "2024-12-20T10:00:00",
  "available": true
}
```

### Health Check Detallado

```bash
curl http://localhost:8000/api/v1/llm/health/detailed
```

Verifica:
- Disponibilidad del servicio LLM
- Estado de componentes modulares
- Conectividad con OpenRouter
- Estado de cache y rate limiting

### Readiness Check

Usado por orquestadores (Kubernetes, Docker Swarm) para verificar si el servicio está listo para recibir tráfico.

```bash
curl http://localhost:8000/api/v1/llm/health/readiness
```

### Liveness Check

Usado por orquestadores para verificar si el servicio está vivo.

```bash
curl http://localhost:8000/api/v1/llm/health/liveness
```

## Integración con Kubernetes

### Readiness Probe

```yaml
readinessProbe:
  httpGet:
    path: /api/v1/llm/health/readiness
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
```

### Liveness Probe

```yaml
livenessProbe:
  httpGet:
    path: /api/v1/llm/health/liveness
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
```

## Configuración Avanzada

### Rate Limiting Personalizado

```python
from core.services.llm import get_advanced_rate_limiter, RateLimitStrategy

rate_limiter = get_advanced_rate_limiter()

# Configurar límite personalizado
rate_limiter.configure(
    key="custom:endpoint",
    limit=200,
    window_seconds=60,
    strategy=RateLimitStrategy.TOKEN_BUCKET,
    burst_size=50,
    refill_rate=3.33  # 200 requests / 60 seconds
)
```

### Deshabilitar Logging de Responses

Para reducir verbosidad en producción:

```python
app.add_middleware(
    LLMLoggingMiddleware,
    log_requests=True,
    log_responses=False  # Solo loguear requests
)
```

### Ajustar Tamaño Máximo de Body

Para endpoints que manejan archivos grandes:

```python
app.add_middleware(
    LLMValidationMiddleware,
    max_body_size=50 * 1024 * 1024  # 50MB
)
```

## Monitoreo

### Métricas Disponibles

Los middlewares exponen las siguientes métricas:

1. **Rate Limiting**
   - Requests bloqueados por rate limit
   - Tasa de éxito/fallo
   - Distribución de rate limits por endpoint

2. **Logging**
   - Latencia promedio por endpoint
   - Tasa de errores
   - Volumen de requests

3. **Validación**
   - Requests rechazados por validación
   - Tamaño promedio de requests
   - Errores de Content-Type

### Integración con Prometheus

```python
from prometheus_client import Counter, Histogram

rate_limit_hits = Counter(
    'llm_rate_limit_hits_total',
    'Total rate limit hits',
    ['endpoint', 'ip']
)

request_latency = Histogram(
    'llm_request_latency_seconds',
    'Request latency',
    ['endpoint', 'method']
)
```

## Troubleshooting

### Rate Limit Muy Estricto

Si recibes muchos errores 429:

1. Aumentar límites en configuración
2. Verificar si hay múltiples IPs compartiendo límite
3. Revisar logs para identificar patrones

### Logging Muy Verboso

Si los logs son demasiado verbosos:

1. Deshabilitar `log_responses`
2. Ajustar nivel de logging
3. Usar filtros de logging

### Validación Rechazando Requests Válidos

Si requests válidos son rechazados:

1. Verificar Content-Type headers
2. Verificar tamaño de body
3. Revisar configuración de `max_body_size`

## Mejores Prácticas

1. **Orden de Middlewares**: Validación → Rate Limiting → Logging
2. **Límites Conservadores**: Empezar con límites conservadores y ajustar
3. **Monitoreo Continuo**: Monitorear métricas de rate limiting
4. **Health Checks**: Configurar health checks en producción
5. **Logging Selectivo**: Solo loguear lo necesario en producción

## Ejemplo Completo

```python
from fastapi import FastAPI
from api.middleware import (
    LLMValidationMiddleware,
    LLMRateLimitMiddleware,
    LLMLoggingMiddleware
)

app = FastAPI()

# Agregar middlewares en orden
app.add_middleware(
    LLMValidationMiddleware,
    max_body_size=10 * 1024 * 1024
)

app.add_middleware(
    LLMRateLimitMiddleware,
    default_limit=100,
    window_seconds=60
)

app.add_middleware(
    LLMLoggingMiddleware,
    log_requests=True,
    log_responses=False
)

# Incluir rutas
from api.routes import llm_routes, llm_health
app.include_router(llm_routes.router, prefix="/api/v1")
app.include_router(llm_health.router, prefix="/api/v1")
```



