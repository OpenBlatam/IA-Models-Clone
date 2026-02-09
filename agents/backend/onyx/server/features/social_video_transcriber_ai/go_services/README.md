# Go Services - High Performance Networking Layer

🚀 Servicios de alto rendimiento en Go para **Social Video Transcriber AI**.

## Características

### ⚡ Networking de Alto Rendimiento
- **HTTP/2** soporte nativo
- **Connection pooling** automático
- **Timeouts** configurables
- **Graceful shutdown**

### 🔄 OpenRouter Client
- **Rate limiting** inteligente
- **Retry con backoff** exponencial
- **Semáforo** para concurrencia
- **Estadísticas** en tiempo real

### 🛡️ Middleware Stack
- **CORS** configurable
- **Rate limiting** por IP
- **Request logging** con zerolog
- **Panic recovery**
- **Request timeout**

### 📊 Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/stats` | Estadísticas del cliente |
| POST | `/api/v1/ai/analyze` | Analizar contenido |
| POST | `/api/v1/ai/variants` | Generar variantes |
| POST | `/api/v1/ai/summarize` | Resumir texto |
| POST | `/api/v1/ai/keywords` | Extraer keywords |
| POST | `/api/v1/ai/translate` | Traducir texto |
| POST | `/api/v1/ai/chat` | Chat directo |
| POST | `/api/v1/batch/analyze` | Batch de análisis |
| POST | `/api/v1/batch/variants` | Batch de variantes |

## Instalación

```bash
# Requisitos: Go 1.21+

cd go_services

# Instalar dependencias
go mod download

# Compilar
go build -o bin/server ./cmd/server

# Ejecutar
./bin/server
```

## Configuración

Variables de entorno:

```bash
# Puerto del servidor (default: 8081)
GO_PORT=8081

# API Key de OpenRouter (requerido)
OPENROUTER_API_KEY=sk-or-v1-...

# Redis URL (opcional)
REDIS_URL=redis://localhost:6379

# Rate limit por minuto (default: 60)
RATE_LIMIT_RPM=60

# Concurrencia máxima (default: 10)
MAX_CONCURRENCY=10

# Timeout de requests en segundos (default: 60)
REQUEST_TIMEOUT=60

# Ambiente (default: development)
ENVIRONMENT=development
```

## Uso

### Analizar Contenido

```bash
curl -X POST http://localhost:8081/api/v1/ai/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Tu contenido aquí..."}'
```

Response:
```json
{
  "framework": "Hook-Story-Offer",
  "structure": ["hook", "problem", "solution", "cta"],
  "keywords": ["palabra1", "palabra2"],
  "summary": "Resumen breve",
  "tone": "professional",
  "audience": "entrepreneurs",
  "hashtags": ["#marketing", "#business"]
}
```

### Generar Variantes

```bash
curl -X POST http://localhost:8081/api/v1/ai/variants \
  -H "Content-Type: application/json" \
  -d '{"text": "Contenido original", "count": 3}'
```

### Resumir

```bash
curl -X POST http://localhost:8081/api/v1/ai/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Texto largo...", "style": "bullets"}'
```

### Traducir

```bash
curl -X POST http://localhost:8081/api/v1/ai/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "target_lang": "Spanish"}'
```

### Batch Processing

```bash
curl -X POST http://localhost:8081/api/v1/batch/analyze \
  -H "Content-Type: application/json" \
  -d '{"texts": ["texto 1", "texto 2", "texto 3"]}'
```

## Arquitectura

```
go_services/
├── cmd/
│   └── server/
│       └── main.go          # Entry point
├── internal/
│   ├── api/
│   │   └── router.go        # HTTP handlers
│   ├── config/
│   │   └── config.go        # Configuration
│   └── openrouter/
│       └── client.go        # OpenRouter client
├── go.mod
├── go.sum
└── README.md
```

## Dependencias

- **chi** - Router HTTP ligero
- **zerolog** - Logger estructurado
- **go-redis** - Cliente Redis (opcional)
- **golang.org/x/sync** - Semáforos y primitivas de sync
- **golang.org/x/time** - Rate limiter

## Performance

| Operación | Latencia P50 | Latencia P99 |
|-----------|--------------|--------------|
| Health check | <1ms | <5ms |
| Analyze | ~1s | ~3s |
| Variants | ~2s | ~5s |
| Translate | ~0.5s | ~2s |

## Docker

```dockerfile
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY go.* ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /server ./cmd/server

FROM alpine:3.19
COPY --from=builder /server /server
EXPOSE 8081
CMD ["/server"]
```

```bash
docker build -t go-transcriber .
docker run -p 8081:8081 -e OPENROUTER_API_KEY=sk-... go-transcriber
```

## Integración con Python

El servicio Go se puede usar como sidecar para operaciones de networking intensivas:

```python
import httpx

GO_SERVICE_URL = "http://localhost:8081"

async def analyze_via_go(text: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{GO_SERVICE_URL}/api/v1/ai/analyze",
            json={"text": text}
        )
        return response.json()
```

---

**Go Services** - Alta performance para Social Video Transcriber AI 🚀












