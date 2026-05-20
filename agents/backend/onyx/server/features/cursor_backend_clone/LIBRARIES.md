# 📚 Librerías Utilizadas - Cursor Agent 24/7

## 🎯 Resumen de Librerías

Este proyecto utiliza las mejores librerías modernas de Python (2024-2025) para máximo rendimiento y funcionalidad.

## 📋 Dependencias por Módulo MCP

### MCP Server Core
- **fastapi** (0.115.0+): Framework web para el servidor MCP
- **uvicorn** (0.32.0+): ASGI server para ejecutar FastAPI
- **pydantic** (2.9.0+): Validación de modelos de request/response
- **starlette** (0.37.0+): Framework ASGI base

### MCP Middleware
- **gzip** (built-in): Compresión de respuestas
- **uuid** (built-in): Generación de Request IDs

### MCP Client
- **httpx** (0.27.0+): Cliente HTTP async para webhooks y health checks
- **orjson** (3.10.7+): JSON ultra-rápido para serialización/deserialización ⚡
- **asyncio** (built-in): Operaciones asíncronas

### MCP Events
- **asyncio** (built-in): Sistema pub/sub async

### MCP Metrics
- **collections** (built-in): defaultdict, deque para métricas
- **datetime** (built-in): Timestamps y cálculos de tiempo

### MCP Config
- **dataclasses** (built-in): Configuración type-safe
- **pathlib** (built-in): Manejo de archivos de configuración
- **json** (built-in): Serialización de configuración

### Command Validator
- **re** (built-in): Validación de patrones regex
- **dataclasses** (built-in): Resultados de validación

### MCP Request Deduplication
- **hashlib** (built-in): Generación de claves SHA256 para deduplicación
- **collections.OrderedDict** (built-in): Cache LRU para requests duplicados
- **time** (built-in): Timestamps y ventanas de tiempo

### MCP Token Bucket Rate Limiting
- **asyncio** (built-in): Locks async para concurrencia
- **time** (built-in): Cálculo de tokens y refill rate
- **dataclasses** (built-in): Estructura de TokenBucket

### MCP Request Queue
- **asyncio.PriorityQueue** (built-in): Cola de requests con prioridades
- **enum** (built-in): Prioridades de requests (LOW, NORMAL, HIGH, URGENT)
- **dataclasses** (built-in): Estructura de QueuedRequest

### MCP Connection Pool
- **httpx** (0.27.0+): Cliente HTTP async con pool de conexiones
- **contextlib** (built-in): Context managers para gestión de recursos
- **asyncio** (built-in): Operaciones async

### MCP Prometheus Export
- **typing** (built-in): Type hints para el exportador
- **logging** (built-in): Logging de errores

### MCP Adaptive Rate Limiter
- **asyncio** (built-in): Task de adaptación automática
- **collections.deque** (built-in): Almacenamiento de métricas recientes
- **time** (built-in): Cálculo de carga y adaptación
- **dataclasses** (built-in): Configuración del limiter

### MCP Errors
- **enum** (built-in): Códigos de error estandarizados
- **fastapi** (0.115.0+): HTTPException personalizado

### Authentication & Cache
- **hashlib** (built-in): Hashing de passwords
- **secrets** (built-in): Generación de API keys
- **asyncio** (built-in): Cache async
- **collections** (built-in): OrderedDict para políticas de cache

## 📦 Categorías de Librerías

### 1. Core Framework
- **FastAPI** (0.115.0+): Framework web moderno y rápido
- **Uvicorn** (0.32.0+): ASGI server ultra-rápido con HTTP/2
- **Pydantic** (2.9.0+): Validación de datos con type hints
- **Starlette**: Framework ASGI base

### 2. Async & Performance
- **httpx** (0.27.0+): Cliente HTTP async moderno
- **aiohttp** (3.11.0+): Cliente HTTP async avanzado
- **aiofiles** (24.1.0+): Operaciones de archivo async
- **orjson** (3.10.7+): JSON ultra-rápido (C++ optimizado) ⚡
- **uvloop** (0.19.0+): Event loop ultra-rápido (Linux/macOS)

### 3. Caching & Storage
- **Redis** (5.2.0+): Cache distribuido
- **aioredis** (2.0.1+): Redis async
- **diskcache** (5.6.3+): Cache en disco rápido
- **aiosqlite** (0.20.0+): SQLite async
- **tinydb** (4.9.0+): Base de datos ligera JSON

### 4. Monitoring & Observability
- **Prometheus Client** (0.20.0+): Métricas
- **OpenTelemetry** (1.27.0+): Observabilidad estándar
- **Structlog** (24.2.0+): Logging estructurado
- **Sentry SDK** (2.16.0+): Error tracking

### 5. Security
- **Cryptography** (43.0.0+): Criptografía avanzada
- **PyJWT** (2.9.0+): JSON Web Tokens
- **bcrypt** (4.2.0+): Hashing seguro

### 6. Utilities
- **Rich** (13.7.1+): Terminal formatting avanzado
- **Click/Typer** (0.12.0+): CLI moderno
- **Tenacity** (8.2.3+): Retry avanzado
- **Dynaconf** (3.2.0+): Configuración dinámica

### 7. Testing
- **Pytest** (8.3.0+): Framework de testing
- **Pytest-asyncio** (0.23.7+): Testing async
- **Pytest-cov** (5.0.0+): Coverage

### 8. Scheduling
- **APScheduler** (3.10.4+): Scheduler avanzado
- **Arq** (0.25.1+): Task queue async

## ⚡ Librerías Ultra-Rápidas

### JSON Processing
- **orjson**: ~2-3x más rápido que json estándar
- **rapidjson**: Parser C++ muy rápido
- **msgpack**: Serialización binaria

### Event Loop
- **uvloop**: ~2-4x más rápido que asyncio estándar (Linux/macOS)

### Profiling
- **py-spy**: Sampling profiler en Rust (muy rápido)

### Linting
- **ruff**: Linter en Rust (ultra-rápido, reemplaza flake8)

## 🔧 Instalación por Categorías

### Solo Core (Mínimo)
```bash
pip install fastapi uvicorn pydantic httpx aiofiles
```

### Con Cache
```bash
pip install redis aioredis diskcache
```

### Con Monitoring
```bash
pip install prometheus-client opentelemetry-api structlog sentry-sdk
```

### Para Desarrollo
```bash
pip install -r requirements-dev.txt
```

## 📊 Comparación de Rendimiento

| Librería | Alternativa | Mejora | Uso en MCP |
|----------|------------|--------|------------|
| orjson | json | 2-3x más rápido | ✅ Usado por defecto en mcp_server.py y mcp_client.py |
| uvloop | asyncio | 2-4x más rápido | ⚠️ Solo Linux/macOS |
| httpx | requests | Async, no bloquea | ✅ Usado para webhooks y connection pooling |
| structlog | logging | Estructurado, mejor análisis | ⚠️ Opcional |
| ruff | flake8 | 10-100x más rápido | ⚠️ Solo desarrollo |

**Nota:** El servidor MCP usa `orjson` automáticamente si está disponible, con fallback transparente a `json` estándar.

## 🎯 Recomendaciones de Uso

### Para JSON rápido
```python
import orjson  # En lugar de json
data = orjson.loads(json_string)
```

### Para logging estructurado
```python
import structlog
logger = structlog.get_logger()
logger.info("event", key="value")  # JSON automático
```

### Para HTTP async
```python
import httpx
async with httpx.AsyncClient() as client:
    response = await client.get("https://api.example.com")
```

### Para configuración dinámica
```python
from dynaconf import settings
value = settings.MY_SETTING  # Carga de .env, .yaml, etc.
```

## 🔄 Actualización de Librerías

Para mantener las librerías actualizadas:

```bash
# Verificar actualizaciones
pip list --outdated

# Actualizar todas
pip install --upgrade -r requirements.txt

# Actualizar una específica
pip install --upgrade fastapi
```

## 📝 Notas Importantes

1. **orjson** requiere compilación C++ (instalación automática)
2. **uvloop** solo funciona en Linux/macOS (no Windows)
3. **aioredis** es la versión async de redis (mejor para async/await)
4. **structlog** es mejor que logging estándar para producción
5. **ruff** reemplaza flake8, black, isort (todo en uno)

## 🚀 Mejores Prácticas

1. Usar **orjson** para JSON en lugar de json estándar
2. Usar **httpx** para HTTP async en lugar de requests
3. Usar **structlog** para logging estructurado
4. Usar **uvloop** en Linux/macOS para mejor rendimiento
5. Usar **ruff** para linting (más rápido que flake8)
6. Usar **dynaconf** para configuración (más flexible que python-dotenv)

## 🔍 Dependencias Opcionales

El servidor MCP está diseñado para funcionar con dependencias opcionales:

### Dependencias Opcionales para MCP
- **httpx**: Requerido solo si se usan webhooks, health checks HTTP o connection pooling
- **orjson**: Opcional pero recomendado para mejor rendimiento JSON (fallback a json estándar)
- **auth**: Módulo de autenticación (opcional si `enable_auth=False`)
- **cache**: Módulo de caché (opcional si `enable_cache=False`)
- **mcp_metrics**: Métricas (opcional si `enable_metrics=False`)
- **mcp_events**: Sistema de eventos (siempre disponible, pero opcional en funcionalidad)
- **mcp_middleware**: Middleware avanzado (opcional, mejora rendimiento)
- **command_validator**: Validador de comandos (opcional, mejora seguridad)
- **mcp_request_deduplication**: Deduplicación de requests (opcional si `enable_request_deduplication=False`)
- **mcp_token_bucket**: Token bucket rate limiting (opcional si `use_token_bucket_rate_limiting=False`)
- **mcp_request_queue**: Cola de requests (opcional si `enable_request_queue=False`)
- **mcp_connection_pool**: Pool de conexiones HTTP (opcional, mejora rendimiento de webhooks)
- **mcp_prometheus**: Exportador de métricas Prometheus (siempre disponible si métricas están habilitadas)
- **mcp_adaptive_rate_limiter**: Rate limiting adaptativo (opcional si `enable_adaptive_rate_limiting=False`)

### Fallback Graceful
Todos los módulos MCP tienen fallbacks si las dependencias no están disponibles:
- Si `httpx` no está disponible: webhooks y connection pooling se deshabilitan automáticamente
- Si `orjson` no está disponible: se usa `json` estándar de Python (más lento pero funcional)
- Si módulos opcionales no están: el servidor funciona con funcionalidad reducida
- Logging informa cuando módulos opcionales no están disponibles
- Todos los fallbacks son transparentes y no afectan la funcionalidad core

## 📦 Instalación Mínima para MCP

Para usar solo el servidor MCP básico:

```bash
pip install fastapi uvicorn pydantic
```

Para funcionalidad completa (recomendado):

```bash
pip install fastapi uvicorn pydantic httpx orjson
```

**Desglose de dependencias:**

- **Mínimo requerido:**
  - `fastapi` - Framework web
  - `uvicorn` - ASGI server
  - `pydantic` - Validación de datos

- **Recomendado para producción:**
  - `httpx` - Para webhooks y connection pooling
  - `orjson` - Para mejor rendimiento JSON (2-3x más rápido)

- **Opcional pero útil:**
  - `redis` / `aioredis` - Para cache distribuido
  - `structlog` - Para logging estructurado
  - `prometheus-client` - Para métricas avanzadas

## 🎯 Uso de Librerías en MCP

### JSON Processing
El servidor MCP usa `orjson` por defecto con fallback a `json` estándar:

```python
# En mcp_server.py y mcp_client.py:
try:
    import orjson
    _json_dumps = lambda obj: orjson.dumps(obj).decode()
    _json_loads = orjson.loads
    _has_orjson = True
except ImportError:
    import json
    _json_dumps = json.dumps
    _json_loads = json.loads
    _has_orjson = False
```

**Beneficios de orjson:**
- 2-3x más rápido que json estándar
- Mejor manejo de tipos nativos de Python
- Serialización más eficiente de bytes
- Usado automáticamente en todos los módulos MCP

### HTTP Client
El servidor MCP usa `httpx` para webhooks (opcional):

```python
try:
    import httpx
    # Usar httpx.AsyncClient
except ImportError:
    # Webhooks deshabilitados
```

### Compression
El middleware de compresión usa `gzip` built-in de Python (no requiere dependencias externas).

### Connection Pooling
El módulo `mcp_connection_pool.py` usa `httpx` para gestionar pools de conexiones HTTP:

```python
from .mcp_connection_pool import HTTPConnectionPool

pool = HTTPConnectionPool(
    max_connections=10,
    max_keepalive_connections=5,
    timeout=10.0
)

# Uso automático en webhooks
await pool.post(url, json=payload)
```

### Request Queue
El módulo `mcp_request_queue.py` usa `asyncio.PriorityQueue` para manejar requests con prioridades:

```python
from .mcp_request_queue import RequestQueue, RequestPriority

queue = RequestQueue(max_size=1000, max_workers=10)
await queue.start()

# Encolar request con prioridad
await queue.enqueue(handler, priority=RequestPriority.HIGH)
```

### Token Bucket Rate Limiting
El módulo `mcp_token_bucket.py` implementa rate limiting avanzado:

```python
from .mcp_token_bucket import TokenBucketRateLimiter

limiter = TokenBucketRateLimiter(
    capacity=100.0,
    refill_rate=1.0
)

is_allowed = await limiter.is_allowed(client_id)
retry_after = await limiter.get_retry_after(client_id)
```

### Request Deduplication
El módulo `mcp_request_deduplication.py` usa `hashlib` para detectar requests duplicados:

```python
from .mcp_request_deduplication import RequestDeduplicator

deduplicator = RequestDeduplicator(
    window_seconds=60.0,
    max_cache_size=10000
)

is_duplicate, age = deduplicator.is_duplicate(
    method="POST",
    path="/mcp/v1/command",
    body=body_string
)
```

## 📚 Documentación

### Librerías Core
- [FastAPI Docs](https://fastapi.tiangolo.com/) - Framework web moderno
- [Pydantic Docs](https://docs.pydantic.dev/) - Validación de datos
- [Uvicorn Docs](https://www.uvicorn.org/) - ASGI server

### Librerías de Rendimiento
- [Orjson Docs](https://github.com/ijl/orjson) - JSON ultra-rápido (usado en MCP)
- [httpx Docs](https://www.python-httpx.org/) - Cliente HTTP async (usado para webhooks)

### Librerías Opcionales
- [Structlog Docs](https://www.structlog.org/) - Logging estructurado
- [Redis Docs](https://redis.io/docs/) - Cache distribuido
- [Prometheus Client](https://prometheus.github.io/client_python/) - Métricas


