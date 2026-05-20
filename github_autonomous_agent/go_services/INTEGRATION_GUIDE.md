# 🔌 Guía de Integración - Go Services

Esta guía explica cómo integrar los servicios Go con el proyecto Python existente.

## Opciones de Integración

### Opción 1: HTTP API (Recomendado para inicio rápido)

Los servicios Go exponen una API HTTP que puede ser llamada desde Python.

#### 1. Iniciar el servicio Go

```bash
cd go_services
go run ./cmd/agent --port 8080
```

#### 2. Usar desde Python

```python
import httpx

client = httpx.Client(base_url="http://localhost:8080")

# Git operations
response = client.post(
    "/api/v1/git/clone",
    params={"url": "https://github.com/user/repo.git", "path": "/tmp/repo"}
)

# Cache operations
client.post("/api/v1/cache", params={"key": "test", "value": "data"})
value = client.get("/api/v1/cache", params={"key": "test"}).json()

# Search
results = client.get("/api/v1/search", params={"q": "query"}).json()
```

### Opción 2: Message Queue (Para alto throughput)

Usar Redis/Kafka/NATS como intermediario entre Python y Go.

#### Python Producer

```python
import redis
import json

redis_client = redis.Redis()

# Enviar tarea a Go service
task = {
    "type": "process_repo",
    "data": {"repo": "user/repo"}
}
redis_client.lpush("go:queue", json.dumps(task))
```

#### Go Consumer

```go
// En queue/taskqueue.go ya está implementado
// Solo necesitas configurar el consumer
```

### Opción 3: gRPC (Para máxima eficiencia)

Para comunicación más eficiente, implementar gRPC.

#### Definir proto

```protobuf
syntax = "proto3";

service AgentService {
  rpc ProcessRepo(RepoRequest) returns (RepoResponse);
  rpc Search(SearchRequest) returns (SearchResponse);
}
```

#### Implementar en Go

```go
// Implementar el servicio gRPC
```

#### Cliente Python

```python
import grpc
from go_services import agent_pb2, agent_pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
stub = agent_pb2_grpc.AgentServiceStub(channel)
response = stub.ProcessRepo(agent_pb2.RepoRequest(repo="user/repo"))
```

## Integración Gradual

### Fase 1: Cache (Más fácil de integrar)

Reemplazar el cache Python con el cache Go:

```python
# Antes
from core.services.cache_service import CacheService
cache = CacheService()

# Después
from go_services_client import GoServicesClient
client = GoServicesClient()
client.cache_set("key", "value")
value = client.cache_get("key")
```

### Fase 2: Git Operations

Reemplazar operaciones Git:

```python
# Antes
import subprocess
subprocess.run(["git", "clone", url, path])

# Después
client = GoServicesClient()
client.clone_repository(url, path)  # 3-5x más rápido
```

### Fase 3: Search

Reemplazar búsqueda:

```python
# Antes
from core.services.search_service import SearchService
search = SearchService()
results = search.search(query)

# Después
client = GoServicesClient()
results = client.search(query)  # 20-100x más rápido
```

### Fase 4: Task Queue

Reemplazar Celery con Go queue:

```python
# Antes
from celery import Celery
app = Celery('tasks')
app.send_task('process_repo', args=[repo])

# Después
# Usar message queue o HTTP API
```

## Docker Compose

Agregar servicio Go al docker-compose.yml:

```yaml
services:
  go-services:
    build:
      context: ./go_services
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    environment:
      - LOG_LEVEL=info
    volumes:
      - ./go_services:/app
    networks:
      - agent-network
```

## Monitoreo

### Health Check

```python
import httpx

client = httpx.Client()
health = client.get("http://localhost:8080/health").json()
print(health)  # {"status":"healthy","service":"go-services"}
```

### Métricas Prometheus

Agregar métricas en Go:

```go
import "github.com/prometheus/client_golang/prometheus"

var (
    requestsTotal = prometheus.NewCounterVec(...)
    requestDuration = prometheus.NewHistogramVec(...)
)
```

## Troubleshooting

### El servicio no inicia

```bash
# Verificar puerto
netstat -an | grep 8080

# Ver logs
go run ./cmd/agent --log-level=debug
```

### Errores de conexión desde Python

```python
# Verificar que el servicio esté corriendo
import httpx
try:
    response = httpx.get("http://localhost:8080/health", timeout=5.0)
    print("Service is up!")
except httpx.ConnectError:
    print("Service is down!")
```

## Próximos Pasos

1. ✅ Implementar HTTP API básica
2. ⏳ Agregar autenticación
3. ⏳ Implementar gRPC
4. ⏳ Agregar métricas Prometheus
5. ⏳ Implementar distributed tracing












