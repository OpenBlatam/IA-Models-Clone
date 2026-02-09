# Multi-Model API

API optimizada para ejecutar múltiples modelos de IA en paralelo, secuencial o con consenso.

## Características

### 🚀 Funcionalidades Principales

- **Ejecución Multi-Modelo**: Ejecuta hasta 5 modelos de IA simultáneamente
- **Integración OpenRouter**: Acceso a 1000+ modelos de IA a través de OpenRouter
  - OpenAI (GPT-4, GPT-3.5, etc.)
  - Anthropic (Claude 3 Opus, Sonnet, Haiku)
  - Google (Gemini Pro, etc.)
  - Meta (Llama 3, Llama 2)
  - Mistral AI, Cohere, y muchos más
- **Estrategias de Ejecución**:
  - `parallel`: Ejecuta todos los modelos en paralelo
  - `sequential`: Ejecuta modelos uno por uno
  - `consensus`: Ejecuta en paralelo y aplica algoritmos de consenso
- **Algoritmos de Consenso**:
  - `majority`: Voto por mayoría simple
  - `weighted`: Voto ponderado por multiplicador
  - `similarity`: Agrupación por similitud
  - `average`: Combinación de todas las respuestas
  - `best`: Selecciona el mejor modelo por métricas

### 🛡️ Seguridad y Resiliencia

- **Rate Limiting**: Protección contra abuso con límites configurables
- **Circuit Breakers**: Protección automática contra modelos fallidos
- **Retry Logic**: Reintentos automáticos con exponential backoff
- **Timeout Configurables**: Timeouts personalizables por request

### 📊 Monitoreo y Observabilidad

- **Métricas Prometheus**: Endpoint `/metrics` para scraping
- **Sentry Integration**: Tracking de errores y excepciones
- **Structured Logging**: Logging estructurado con contexto
- **Health Checks**: Endpoints de salud del sistema y modelos

### ⚡ Performance

- **Multi-Tier Caching**: Cache L1 (memoria), L2 (Redis), L3 (disco)
- **Compresión**: Compresión automática de respuestas grandes
- **Serialización Optimizada**: orjson, msgpack para máxima velocidad
- **Async I/O**: Operaciones completamente asíncronas

### 🔌 WebSocket Support

- **Streaming**: Respuestas en tiempo real vía WebSocket
- **Connection Management**: Gestión automática de conexiones
- **Real-time Updates**: Actualizaciones en tiempo real del progreso

## Endpoints

### POST `/multi-model/execute`
Ejecuta múltiples modelos con un prompt.

**Request Body:**
```json
{
  "prompt": "Your prompt here",
  "models": [
    {
      "model_type": "gpt_5.1",
      "is_enabled": true,
      "temperature": 0.7,
      "max_tokens": 1000
    }
  ],
  "strategy": "parallel",
  "consensus_method": "majority",
  "cache_enabled": true,
  "cache_ttl": 3600,
  "timeout": 30.0,
  "allow_partial_success": true,
  "min_successful_models": 2
}
```

### POST `/multi-model/execute/batch`
Procesa múltiples requests en batch (hasta 10).

### WebSocket `/multi-model/ws/stream`
Streaming de respuestas en tiempo real.

### GET `/multi-model/models`
Lista todos los modelos disponibles con su estado.

### GET `/multi-model/models/{model_type}/health`
Métricas de salud de un modelo específico.

### GET `/multi-model/health`
Health check completo del sistema.

### GET `/multi-model/stats`
Estadísticas detalladas del sistema.

### GET `/multi-model/metrics`
Métricas en formato Prometheus.

### GET `/multi-model/rate-limit/info`
Información sobre rate limits actuales.

### GET `/multi-model/openrouter/models`
Lista todos los modelos disponibles de OpenRouter (1000+ modelos).
- `provider`: Filtrar por proveedor (opcional)
- `search`: Buscar modelos por nombre (opcional)

### POST `/multi-model/execute/stream`
Ejecuta múltiples modelos con streaming en tiempo real (SSE).
- Retorna respuestas de modelos conforme se completan
- Actualizaciones de progreso
- Respuesta agregada final
- Formato: Server-Sent Events (text/event-stream)

### DELETE `/multi-model/cache`
Limpia el cache (opcionalmente por nivel).

## Configuración

Variables de entorno:

```bash
# OpenRouter (opcional pero recomendado para acceso a 1000+ modelos)
OPENROUTER_API_KEY=sk-or-v1-...

# Sentry
MULTI_MODEL_SENTRY_DSN=your-sentry-dsn
MULTI_MODEL_SENTRY_ENVIRONMENT=production

# Cache
MULTI_MODEL_REDIS_URL=redis://localhost:6379/0
MULTI_MODEL_CACHE_L1_MAX_SIZE=1000
MULTI_MODEL_CACHE_L1_TTL=300

# Rate Limiting
MULTI_MODEL_RATE_LIMIT_DEFAULT=100
MULTI_MODEL_RATE_LIMIT_WINDOW=60
```

## Uso

```python
from multi_model_api import router, websocket_router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)
app.include_router(websocket_router)

# Agregar middleware
from multi_model_api import MetricsMiddleware, LoggingMiddleware, init_sentry

init_sentry(dsn=os.getenv("SENTRY_DSN"))
app.add_middleware(MetricsMiddleware)
app.add_middleware(LoggingMiddleware)
```

## Streaming

La API soporta streaming de respuestas en tiempo real usando Server-Sent Events (SSE):

```python
import requests

response = requests.post(
    "http://localhost:8000/multi-model/execute/stream",
    json={
        "prompt": "Explain quantum computing",
        "models": [
            {"model_type": "openrouter/gpt-4", "is_enabled": True}
        ],
        "strategy": "parallel"
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        data = json.loads(line.decode('utf-8').replace('data: ', ''))
        if data['type'] == 'model_response':
            print(f"{data['model_type']}: {data['response']}")
        elif data['type'] == 'complete':
            print(f"Final: {data['aggregated_response']}")
```

## Versión

2.3.0

## Integración OpenRouter

La API incluye soporte completo para OpenRouter, proporcionando acceso a **1000+ modelos de IA**:

- **OpenAI**: GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
- **Anthropic**: Claude 3 Opus, Sonnet, Haiku
- **Google**: Gemini Pro, Gemini Pro Vision
- **Meta**: Llama 3 70B, Llama 3 8B
- **Mistral AI**: Mistral Large, Mixtral 8x7B
- **Y muchos más...**

### Uso de Modelos OpenRouter

```json
{
  "prompt": "Explain quantum computing",
  "models": [
    {
      "model_type": "openrouter/gpt-4",
      "is_enabled": true,
      "temperature": 0.7,
      "openrouter_model": "openai/gpt-4"
    },
    {
      "model_type": "openrouter/claude-3-opus",
      "is_enabled": true,
      "openrouter_model": "anthropic/claude-3-opus"
    }
  ],
  "strategy": "consensus"
}
```

### Modelos Dinámicos

También puedes usar cualquier modelo de OpenRouter directamente:

```json
{
  "prompt": "Your prompt",
  "models": [
    {
      "model_type": "openrouter/gpt-4",
      "is_enabled": true,
      "custom_params": {
        "openrouter_model": "qwen/qwen-2.5-72b-instruct"
      }
    }
  ]
}
```

Ver [OPENROUTER_INTEGRATION.md](OPENROUTER_INTEGRATION.md) para más detalles.

