# Guía Rápida V9 - GitHub Autonomous Agent

## 🚀 Inicio Rápido

### Instalación

```bash
pip install -r requirements.txt
```

### Uso de Servicios

#### CacheService

```python
from core.services import CacheService
from config.di_setup import get_service

# Obtener del contenedor DI
cache = get_service("cache_service")

# O usar directamente
cache = CacheService(max_size=1000, default_ttl=300)

# Almacenar
cache.set("key", {"data": "value"}, ttl=600)

# Obtener
data = cache.get("key")

# Generar clave consistente
key = cache.generate_key("repo", owner="owner", name="repo")

# Estadísticas
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']}%")
```

#### MetricsService

```python
from core.services import MetricsService
from config.di_setup import get_service

# Obtener del contenedor DI
metrics = get_service("metrics_service")

# Registrar tarea
metrics.record_task("create_file", "completed", duration=1.5)

# Registrar API request
metrics.record_api_request("get_repository", "success", duration=0.3)

# Usar timer
metrics.start_timer("operation")
# ... código ...
duration = metrics.stop_timer("operation")

# Registrar error
metrics.record_error("GitHubClientError")

# Obtener métricas
all_metrics = metrics.get_metrics()
```

#### RateLimitService

```python
from core.services import RateLimitService, RateLimitExceededError
from config.di_setup import get_service

# Obtener del contenedor DI
rate_limit = get_service("rate_limit_service")

# Verificar rate limit
try:
    rate_limit.check_rate_limit("token_123", cost=1)
    # Hacer request...
except RateLimitExceededError as e:
    print(f"Espera {e.retry_after} segundos")

# Obtener estadísticas
stats = rate_limit.get_stats("token_123")
print(f"Requests restantes: {stats['remaining']}")
```

## 📋 Integración Rápida

### En GitHubClient

```python
class GitHubClient:
    def __init__(self, token: str, cache_service: CacheService = None, 
                 rate_limit_service: RateLimitService = None):
        self.cache = cache_service
        self.rate_limit = rate_limit_service
        # ...
    
    def get_repository(self, owner: str, repo: str):
        # Verificar rate limit
        if self.rate_limit:
            self.rate_limit.check_rate_limit(self.token)
        
        # Verificar caché
        if self.cache:
            cache_key = self.cache.generate_key("repo", owner=owner, name=repo)
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        
        # Obtener de API
        repo_obj = self.github.get_repo(f"{owner}/{repo}")
        repo_data = self._serialize_repo(repo_obj)
        
        # Almacenar en caché
        if self.cache:
            self.cache.set(cache_key, repo_data, ttl=300)
        
        return repo_data
```

### En TaskProcessor

```python
class TaskProcessor:
    def __init__(self, github_client: GitHubClient, storage: TaskStorage,
                 metrics_service: MetricsService = None):
        self.metrics = metrics_service
        # ...
    
    async def execute_task(self, task: Dict[str, Any]):
        task_id = task["id"]
        task_type = self._get_task_type(task)
        
        # Iniciar timer
        if self.metrics:
            self.metrics.start_timer(f"task_{task_id}")
        
        try:
            result = await self._execute_instruction(...)
            
            # Registrar éxito
            if self.metrics:
                duration = self.metrics.stop_timer(f"task_{task_id}")
                self.metrics.record_task(task_type, "completed", duration)
            
            return result
        except Exception as e:
            # Registrar error
            if self.metrics:
                self.metrics.record_error(type(e).__name__)
            raise
```

## 🔧 Configuración

### Variables de Entorno

```bash
# GitHub
GITHUB_TOKEN=your_token_here

# Prometheus (opcional)
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090

# Cache
CACHE_MAX_SIZE=1000
CACHE_DEFAULT_TTL=300

# Rate Limit
RATE_LIMIT_MAX=5000
RATE_LIMIT_WINDOW=3600
```

## 📊 Métricas Disponibles

### Prometheus (si está habilitado)

- `github_agent_tasks_total` - Total de tareas
- `github_agent_task_duration_seconds` - Duración de tareas
- `github_agent_api_requests_total` - Requests a API
- `github_agent_api_duration_seconds` - Duración de requests
- `github_agent_cache_operations_total` - Operaciones de caché
- `github_agent_active_tasks` - Tareas activas
- `github_agent_errors_total` - Total de errores

### Endpoint de Métricas

```python
from fastapi import FastAPI
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
```

## 🐛 Troubleshooting

### Rate Limit Excedido

```python
try:
    rate_limit.check_rate_limit("token")
except RateLimitExceededError as e:
    # Esperar hasta reset_time
    wait_time = (e.reset_time - datetime.now()).total_seconds()
    await asyncio.sleep(wait_time)
    # Reintentar
```

### Caché No Funciona

```python
# Verificar estadísticas
stats = cache.get_stats()
if stats["hit_rate"] == 0:
    # Revisar TTL y claves
    print(f"Cache size: {stats['size']}")
    print(f"Max size: {stats['max_size']}")
```

### Métricas No Aparecen

```python
# Verificar si Prometheus está disponible
metrics = MetricsService(use_prometheus=True)
all_metrics = metrics.get_metrics()
if not all_metrics.get("prometheus_enabled"):
    # Usar métricas en memoria
    print(all_metrics["metrics"])
```

## 📚 Recursos

- [Documentación Completa](./ARCHITECTURE_IMPROVEMENTS_V9.md)
- [Service Layer Pattern](https://martinfowler.com/eaaCatalog/serviceLayer.html)
- [Prometheus Docs](https://prometheus.io/docs/)

---

**Versión**: 9.0  
**Última actualización**: Diciembre 2024



