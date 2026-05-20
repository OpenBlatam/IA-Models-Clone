# Multi-Model API

> Part of the [Blatam Academy Integrated Platform](../README.md)

API optimized for executing multiple AI models in parallel, sequentially, or with consensus.

## Features

### 🚀 Core Functionalities

- **Multi-Model Execution**: Execute up to 5 AI models simultaneously
- **OpenRouter Integration**: Access to 1000+ AI models through OpenRouter
  - OpenAI (GPT-4, GPT-3.5, etc.)
  - Anthropic (Claude 3 Opus, Sonnet, Haiku)
  - Google (Gemini Pro, etc.)
  - Meta (Llama 3, Llama 2)
  - Mistral AI, Cohere, and many more
- **Execution Strategies**:
  - `parallel`: Execute all models in parallel
  - `sequential`: Execute models one by one
  - `consensus`: Execute in parallel and apply consensus algorithms
- **Consensus Algorithms**:
  - `majority`: Simple majority vote
  - `weighted`: Weighted vote by multiplier
  - `similarity`: Clustering by similarity
  - `average`: Combination of all responses
  - `best`: Selects the best model by metrics

### 🛡️ Security and Resilience

- **Rate Limiting**: Abuse protection with configurable limits
- **Circuit Breakers**: Automatic protection against failed models
- **Retry Logic**: Automatic retries with exponential backoff
- **Configurable Timeouts**: Customizable timeouts per request

### 📊 Monitoring and Observability

- **Prometheus Metrics**: `/metrics` endpoint for scraping
- **Sentry Integration**: Error and exception tracking
- **Structured Logging**: Structured logging with context
- **Health Checks**: System and model health endpoints

### ⚡ Performance

- **Multi-Tier Caching**: L1 Cache (memory), L2 (Redis), L3 (disk)
- **Compression**: Automatic compression of large responses
- **Optimized Serialization**: orjson, msgpack for maximum speed
- **Async I/O**: Completely asynchronous operations

### 🔌 WebSocket Support

- **Streaming**: Real-time responses via WebSocket
- **Connection Management**: Automatic connection management
- **Real-time Updates**: Real-time progress updates

## Endpoints

### POST `/multi-model/execute`
Executes multiple models with a prompt.

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
Processes multiple requests in batch (up to 10).

### WebSocket `/multi-model/ws/stream`
Real-time response streaming.

### GET `/multi-model/models`
Lists all available models with their status.

### GET `/multi-model/models/{model_type}/health`
Health metrics for a specific model.

### GET `/multi-model/health`
Complete system health check.

### GET `/multi-model/stats`
Detailed system statistics.

### GET `/multi-model/metrics`
Metrics in Prometheus format.

### GET `/multi-model/rate-limit/info`
Information about current rate limits.

### GET `/multi-model/openrouter/models`
Lists all available OpenRouter models (1000+ models).
- `provider`: Filter by provider (optional)
- `search`: Search models by name (optional)

### POST `/multi-model/execute/stream`
Executes multiple models with real-time streaming (SSE).
- Returns model responses as they complete
- Progress updates
- Final aggregated response
- Format: Server-Sent Events (text/event-stream)

### DELETE `/multi-model/cache`
Clears the cache (optionally by level).

## Configuration

Environment variables:

```bash
# OpenRouter (optional but recommended for access to 1000+ models)
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

## Usage

```python
from multi_model_api import router, websocket_router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)
app.include_router(websocket_router)

# Add middleware
from multi_model_api import MetricsMiddleware, LoggingMiddleware, init_sentry

init_sentry(dsn=os.getenv("SENTRY_DSN"))
app.add_middleware(MetricsMiddleware)
app.add_middleware(LoggingMiddleware)
```

## Streaming

The API supports real-time response streaming using Server-Sent Events (SSE):

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

## Version

2.3.0

## OpenRouter Integration

The API includes full support for OpenRouter, providing access to **1000+ AI models**:

- **OpenAI**: GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
- **Anthropic**: Claude 3 Opus, Sonnet, Haiku
- **Google**: Gemini Pro, Gemini Pro Vision
- **Meta**: Llama 3 70B, Llama 3 8B
- **Mistral AI**: Mistral Large, Mixtral 8x7B
- **And many more...**

### Using OpenRouter Models

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

### Dynamic Models

You can also use any OpenRouter model directly:

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

See [OPENROUTER_INTEGRATION.md](OPENROUTER_INTEGRATION.md) for more details.

---

[← Back to Main README](../README.md)
