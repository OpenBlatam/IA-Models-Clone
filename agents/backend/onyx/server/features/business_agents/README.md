# Ultimate Quantum AI ML NLP Benchmark API

Production-ready FastAPI application with modular architecture, comprehensive features, and enterprise-grade capabilities.

## 🚀 Features

### Core Capabilities
- ✅ **Modular Architecture**: Domain registry, feature flags, plugin system
- ✅ **Advanced DI**: Injectable interfaces for Cache/HTTP/Auth
- ✅ **Security**: JWT/OAuth2, API Keys, CORS, rate limiting, security headers
- ✅ **Resilience**: Circuit breakers, retries, timeouts, request size limits
- ✅ **Caching**: Redis + in-memory fallback with TTL
- ✅ **Observability**: Prometheus metrics, OpenTelemetry support, structured logging (JSON/text)
- ✅ **Background Jobs**: Async task system with tracking endpoints
- ✅ **Error Handling**: Custom exception classes with structured error responses
- ✅ **Configuration Validation**: Automatic config validation on startup
- ✅ **Production Ready**: Docker, docker-compose, health checks, liveness/readiness

## 📦 Quick Start

### Automated Setup

**Linux/macOS:**
```bash
chmod +x setup_dev.sh
./setup_dev.sh
source venv/bin/activate
python run.py --reload
```

**Windows:**
```cmd
setup_dev.bat
venv\Scripts\activate
python run.py --reload
```

### Local Development

**Option 1: Quick Start Script (Recommended)**
```bash
# Install dependencies
pip install "uvicorn[standard]" fastapi pydantic orjson redis httpx tenacity prometheus-client PyJWT

# Run with quick start script
python agents/backend/onyx/server/features/business_agents/run.py --port 8000 --reload

# Or with options
python agents/backend/onyx/server/features/business_agents/run.py \
  --host 0.0.0.0 --port 8000 --reload --log-level info
```

**Option 2: Direct Uvicorn**
```bash
# Install dependencies
pip install "uvicorn[standard]" fastapi pydantic orjson redis httpx tenacity prometheus-client PyJWT

# Run the API
python -m uvicorn agents.backend.onyx.server.features.business_agents.main:app \
  --host 0.0.0.0 --port 8000 --reload
```

### Docker Compose

```bash
# Start entire stack (API + Redis + Prometheus + Grafana)
docker compose up --build

# Access services
# API: http://localhost:8000/docs
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

## ⚙️ Configuration

Environment variables (see `.env.example`):

```bash
# App
APP_NAME=Ultimate Quantum AI ML NLP Benchmark
ALLOWED_ORIGINS=*
RPS_LIMIT=200
MAX_BODY_BYTES=5242880
REQUEST_TIMEOUT_SECONDS=30

# Features (comma-separated, empty = all enabled)
FEATURES=basic,advanced,quantum

# Cache
REDIS_URL=redis://localhost:6379/0
CACHE_TTL_SECONDS=10

# Security
ENFORCE_AUTH=false
API_KEY=
JWT_SECRET=
JWT_ALGORITHM=HS256

# Resilience
HTTP_TIMEOUT_SECONDS=5
HTTP_RETRIES=3
CB_FAIL_THRESHOLD=5
CB_RECOVERY_SECONDS=30
```

## 🚀 Advanced Features

### Endpoint Caching
Use `@cache_endpoint` decorator for automatic response caching:

```python
from .endpoint_cache import cache_endpoint

@router.get("/expensive")
@cache_endpoint(ttl=300, vary_by=["param"])
async def expensive_operation(param: str):
    # Result cached for 5 minutes
    return result
```

### WebSockets
Real-time communication via WebSockets:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/my-channel');
ws.onmessage = (event) => console.log(JSON.parse(event.data));
```

### Webhooks
Event-driven notifications:

```bash
# Register webhook
curl -X POST http://localhost:8000/api/v1/webhooks \
  -d '{"url": "https://your-app.com/webhook", "events": ["task.completed"]}'

# Trigger event (testing)
curl -X POST http://localhost:8000/api/v1/webhooks/trigger/task.completed \
  -d '{"task_id": "123", "status": "completed"}'
```

## 🔌 Plugins

Create custom routers without modifying core code:

1. Create file in `plugins/` directory
2. Name it `plugin_*.py` or `*_plugin.py`
3. Export `router` (APIRouter) and optionally `ROUTER_PREFIX`

Example:
```python
# plugins/plugin_myfeature.py
from fastapi import APIRouter

router = APIRouter(prefix="/myfeature", tags=["My Feature"])

@router.get("/")
async def my_endpoint():
    return {"message": "From plugin!"}
```

## 📊 Endpoints

### Core
- `GET /` - System overview
- `GET /health` - Health check
- `GET /live` - Liveness probe
- `GET /ready` - Readiness probe
- `GET /status` - Detailed system status
- `GET /capabilities` - System capabilities
- `GET /metrics` - JSON metrics
- `GET /metrics/prometheus` - Prometheus scrape endpoint

### Tasks
- `POST /api/v1/tasks` - Create background task
- `GET /api/v1/tasks/{task_id}` - Get task status
- `GET /api/v1/tasks` - List all tasks

### Batch Processing
- `POST /api/v1/batch/process` - Create batch processing job
- `GET /api/v1/batch/{batch_id}` - Get batch status
- `GET /api/v1/batch` - List all batches

### WebSockets
- `WS /ws/{channel}` - Connect to WebSocket channel for real-time updates
- `POST /ws/broadcast/{channel}` - Broadcast message to channel

### Webhooks
- `POST /api/v1/webhooks` - Register webhook
- `GET /api/v1/webhooks` - List all webhooks
- `GET /api/v1/webhooks/{webhook_id}` - Get webhook details
- `DELETE /api/v1/webhooks/{webhook_id}` - Delete webhook
- `POST /api/v1/webhooks/trigger/{event}` - Trigger webhook event (testing)

### Domain Routers
All domain routers are automatically loaded from the registry based on `FEATURES` env var.

## ✅ Configuration Validation

The API automatically validates configuration on startup and logs warnings for:
- Security misconfigurations (missing auth keys when ENFORCE_AUTH=true)
- Performance concerns (very high rate limits, large body sizes)
- Cache effectiveness (very short TTLs)
- Feature flag status

Check startup logs for any configuration warnings.

## 🏗️ Architecture

### Modular Structure
```
business_agents/
├── domains/          # Domain router registry
├── plugins/          # Auto-discovered plugins
├── tasks/            # Background job system
├── interfaces.py     # Protocol definitions (DI)
├── providers.py      # Dependency providers
├── settings.py       # Centralized configuration
├── cache.py          # Caching abstraction
├── http_client.py    # Resilient HTTP client
├── security.py       # Auth/dependencies
├── main.py           # Entry point
└── ultimate_quantum_ai_app.py  # Main app
```

### DI System
- `ICache`: Cache interface (Redis/memory implementations)
- `IHTTPClient`: HTTP client interface (with circuit breaker)
- `IAuth`: Auth interface (JWT/API Key implementations)

### Feature Flags
Control which domains are loaded via `FEATURES` environment variable:
- Empty = all enabled
- `FEATURES=basic,advanced` = only basic and advanced domains

## 🔒 Security

### Authentication
- **API Key**: Set `API_KEY` env var, send as `X-API-Key` header
- **JWT**: Set `JWT_SECRET`, send as `Authorization: Bearer <token>`
- Enable with `ENFORCE_AUTH=true`

### Rate Limiting
- **In-memory rate limiting** (token bucket) - default
- **Distributed rate limiting** with Redis (optional via `USE_DISTRIBUTED_RATE_LIMIT=true`)
- Configurable via `RPS_LIMIT` (requests per minute)
- Fail-open if Redis unavailable (distributed mode)

### Security Headers
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- Referrer-Policy: no-referrer
- Permissions-Policy headers

## 📈 Observability

### Metrics
- Prometheus-compatible `/metrics/prometheus` endpoint
- Request counts, latencies, status codes
- Custom metrics per domain

### Logging
- **Structured JSON logging** (optional via `JSON_LOGGING=true`)
- **Request IDs** for tracing requests across services
- **Process time tracking** in response headers
- **Configurable log levels** via `LOG_LEVEL` env var
- **Advanced request logging** (optional via `ENABLE_REQUEST_LOGGING=true`)
  - Logs method, URL, query params, client IP
  - Optional body and headers logging (use with caution)
- **OpenTelemetry integration** (optional via `ENABLE_OTEL=true`)

### Health Checks
- `/health` - Comprehensive system health (all subsystems)
- `/live` - Kubernetes liveness probe (basic alive check)
- `/ready` - Kubernetes readiness probe (validates cache, HTTP client, and dependencies)

## 🚀 Production Deployment

### Docker
```bash
docker build -t quantum-api -f Dockerfile .
docker run -p 8000:8000 quantum-api
```

### Kubernetes
- Use provided manifests (TODO: create k8s manifests)
- Configure liveness/readiness probes
- Set resource limits and requests
- Use ConfigMaps/Secrets for configuration

### API Gateway
- Recommended: Kong/Traefik
- Handle JWT validation at gateway level
- Rate limiting, CORS, WAF

## 🧪 Testing

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest agents/backend/onyx/server/features/business_agents/tests/
```

### Test Structure
- `tests/test_basic.py` - Basic endpoint tests
- Tests use TestClient for FastAPI testing
- Mock dependencies via DI system

### Mock Dependencies
Use DI system to swap implementations:

```python
from .providers import provide_cache

# In tests
def test_with_mock_cache():
    app.state.cache = MockCache()
    # ... test code
```

## 📖 Examples

### Usage Examples
See `examples/example_usage.py` for complete API usage examples:

```bash
# Run examples (API must be running)
python agents/backend/onyx/server/features/business_agents/examples/example_usage.py
```

Examples include:
- Health checks
- Creating and tracking tasks
- Getting capabilities
- Retrieving metrics

## 🛠️ Scripts & Utilities

### Health Check Scripts
```bash
# Linux/macOS
./scripts/check_health.sh [API_URL]

# Windows
scripts\check_health.bat [API_URL]
```

### Setup Scripts
- `setup_dev.sh` (Linux/macOS) - Automated development setup
- `setup_dev.bat` (Windows) - Automated development setup

## 📚 Additional Resources

- **EXECUTIVE_SUMMARY.md** - High-level overview and quick reference
- **CHANGELOG.md** - Complete feature history
- **examples/example_usage.py** - API usage examples
- **tests/** - Test suite foundation
- **requirements.txt** - Python dependencies
- **.github/workflows/ci.yml** - CI/CD configuration

## 📝 TODO / Future Enhancements

- [ ] Celery/RQ integration for production jobs
- [ ] Kubernetes manifests and Helm charts
- [ ] Grafana dashboards preconfigured
- [ ] API Gateway configs (Kong/Traefik)
- [ ] Distributed rate limiting with Redis
- [ ] Advanced OAuth2/JWT scopes per router
- [ ] WebSocket support
- [ ] GraphQL endpoint

## 📄 License

[Your License Here]
