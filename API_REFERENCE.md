# 📡 Referencia de API - Blatam Academy Features

## 🔗 Endpoints Principales

### Integration System (Puerto 8000)

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "integration-system": "healthy",
    "bul": "healthy",
    "content-redundancy": "healthy"
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

#### API Gateway Route
```http
POST /api/v1/gateway/route
Content-Type: application/json

{
  "target_system": "bul",
  "endpoint": "/query",
  "method": "POST",
  "data": {
    "query": "Your query here"
  }
}
```

## 🎯 BUL System (Puerto 8002)

### Submit Query
```http
POST /query
Content-Type: application/json

{
  "query": "Create a marketing strategy",
  "priority": 1,
  "business_area": "marketing"
}
```

**Response:**
```json
{
  "task_id": "task_123456",
  "status": "queued",
  "estimated_time": 30
}
```

### Get Task Status
```http
GET /task/{task_id}/status
```

**Response:**
```json
{
  "task_id": "task_123456",
  "status": "completed",
  "progress": 100,
  "documents": [
    {
      "id": "doc_1",
      "type": "markdown",
      "size": 5420
    }
  ]
}
```

### Get Generated Documents
```http
GET /task/{task_id}/documents
```

### List All Documents
```http
GET /documents?page=1&limit=20&business_area=marketing
```

### Search Documents
```http
GET /search?q=marketing+strategy&limit=10
```

### Get Statistics
```http
GET /stats
```

**Response:**
```json
{
  "total_tasks": 1250,
  "completed_tasks": 1200,
  "failed_tasks": 50,
  "average_processing_time": 25.5,
  "cache_hit_rate": 0.65,
  "business_area_distribution": {
    "marketing": 450,
    "sales": 320,
    "operations": 280
  }
}
```

## ⚡ KV Cache Engine API

### Process Request
```http
POST /api/v1/kv-cache/process
Content-Type: application/json
X-API-Key: your-api-key

{
  "text": "Your input text",
  "max_length": 100,
  "temperature": 0.7,
  "session_id": "user_123"
}
```

### Get Cache Statistics
```http
GET /api/v1/kv-cache/stats
```

**Response:**
```json
{
  "hit_rate": 0.75,
  "miss_rate": 0.25,
  "total_requests": 10000,
  "cached_requests": 7500,
  "memory_usage": 0.65,
  "average_latency_ms": 85,
  "p50_latency_ms": 75,
  "p95_latency_ms": 150,
  "p99_latency_ms": 250,
  "throughput_rps": 120
}
```

### Clear Cache
```http
POST /api/v1/kv-cache/clear
X-API-Key: your-api-key
```

### Health Check
```http
GET /api/v1/kv-cache/health
```

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "cache": "healthy",
    "gpu": "healthy",
    "persistence": "healthy"
  },
  "metrics": {
    "memory_usage": 0.65,
    "cache_size": 8192,
    "active_sessions": 150
  }
}
```

## 🔐 Autenticación

### JWT Token
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "user",
  "password": "password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Usar Token
```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```

## 📊 Métricas Prometheus

### Endpoint de Métricas
```http
GET /metrics
```

**Métricas Principales:**
```
# KV Cache Metrics
kv_cache_hit_rate 0.75
kv_cache_miss_rate 0.25
kv_cache_latency_seconds{quantile="0.5"} 0.075
kv_cache_latency_seconds{quantile="0.95"} 0.150
kv_cache_latency_seconds{quantile="0.99"} 0.250
kv_cache_throughput_rps 120.0
kv_cache_memory_usage_ratio 0.65

# System Metrics
system_requests_total 10000
system_errors_total 50
system_active_connections 25
```

## 🐍 Python Client

### Cliente Simple

```python
import httpx

class BULClient:
    def __init__(self, base_url="http://localhost:8002", api_key=None):
        self.base_url = base_url
        self.headers = {}
        if api_key:
            self.headers["X-API-Key"] = api_key
    
    async def submit_query(self, query, priority=1):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/query",
                json={"query": query, "priority": priority},
                headers=self.headers
            )
            return response.json()
    
    async def get_task_status(self, task_id):
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/task/{task_id}/status",
                headers=self.headers
            )
            return response.json()
    
    async def get_documents(self, task_id):
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/task/{task_id}/documents",
                headers=self.headers
            )
            return response.json()

# Uso
client = BULClient(api_key="your-api-key")
task = await client.submit_query("Create marketing strategy")
status = await client.get_task_status(task["task_id"])
```

### Cliente Avanzado con Retry

```python
from tenacity import retry, stop_after_attempt, wait_exponential

class AdvancedBULClient(BULClient):
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def submit_query_with_retry(self, query):
        return await self.submit_query(query)
```

## 📝 Schemas

### Request Schema

```python
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    priority: int = 1
    business_area: Optional[str] = None
    format: str = "markdown"
    max_length: Optional[int] = None
```

### Response Schema

```python
class TaskResponse(BaseModel):
    task_id: str
    status: str
    estimated_time: Optional[int] = None
    created_at: datetime
```

## 🔄 Webhooks

### Configurar Webhook

```http
POST /api/v1/webhooks
Content-Type: application/json

{
  "url": "https://your-domain.com/webhook",
  "events": ["task.completed", "task.failed"],
  "secret": "webhook-secret"
}
```

### Payload del Webhook

```json
{
  "event": "task.completed",
  "task_id": "task_123456",
  "timestamp": "2024-01-01T00:00:00Z",
  "data": {
    "documents": [...]
  }
}
```

## 🧪 Testing Endpoints

### Test Health
```http
GET /health/test
```

### Test KV Cache
```http
POST /api/v1/kv-cache/test
Content-Type: application/json

{
  "iterations": 100,
  "concurrent": 10
}
```

---

**Documentación Completa:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI Spec: http://localhost:8000/openapi.json

