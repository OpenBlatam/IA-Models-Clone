# 🔗 Guía de Integración - Blatam Academy Features

## 📋 Tabla de Contenidos

- [Integración con FastAPI](#integración-con-fastapi)
- [Integración con Celery](#integración-con-celery)
- [Integración con Django](#integración-con-django)
- [Integración con Flask](#integración-con-flask)
- [Integración con Redis](#integración-con-redis)
- [Integración con PostgreSQL](#integración-con-postgresql)
- [Integración con Prometheus](#integración-con-prometheus)
- [Integración con Grafana](#integración-con-grafana)
- [API REST](#api-rest)
- [WebSocket](#websocket)
- [gRPC](#grpc)

## 🚀 Integración con FastAPI

### Setup Básico

```python
from fastapi import FastAPI, Depends, HTTPException
from bulk.core.ultra_adaptive_kv_cache_engine import (
    UltraAdaptiveKVCacheEngine,
    KVCacheConfig,
    CacheStrategy
)

app = FastAPI(title="BUL API")

# Inicializar engine
cache_config = KVCacheConfig(
    max_tokens=8192,
    cache_strategy=CacheStrategy.ADAPTIVE
)
cache_engine = UltraAdaptiveKVCacheEngine(cache_config)

@app.on_event("startup")
async def startup():
    """Inicializar en startup."""
    # Warmup opcional
    await cache_engine.warmup()

@app.post("/api/query")
async def process_query(
    query: dict,
    engine: UltraAdaptiveKVCacheEngine = Depends(get_cache_engine)
):
    """Procesar query con cache."""
    try:
        result = await engine.process_request(query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_cache_engine() -> UltraAdaptiveKVCacheEngine:
    """Dependency para obtener engine."""
    return cache_engine
```

### Con Middleware de Cache

```python
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class CacheMiddleware(BaseHTTPMiddleware):
    """Middleware para cache automático."""
    
    async def dispatch(self, request: Request, call_next):
        # Skip cache para ciertos paths
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        # Check cache
        cache_key = f"{request.method}:{request.url.path}"
        cached = await cache_engine.get_from_cache(cache_key)
        
        if cached:
            return JSONResponse(cached)
        
        # Process request
        response = await call_next(request)
        
        # Store in cache
        if response.status_code == 200:
            await cache_engine.store_in_cache(cache_key, response.body)
        
        return response

app.add_middleware(CacheMiddleware)
```

### Con Background Tasks

```python
from fastapi import BackgroundTasks

@app.post("/api/batch")
async def process_batch(
    queries: List[dict],
    background_tasks: BackgroundTasks
):
    """Procesar batch en background."""
    
    async def process_query_bg(query):
        result = await cache_engine.process_request(query)
        # Procesar resultado
        return result
    
    results = []
    for query in queries:
        background_tasks.add_task(process_query_bg, query)
    
    return {"status": "processing", "count": len(queries)}
```

## 🔄 Integración con Celery

### Setup Celery Worker

```python
# celery_app.py
from celery import Celery
from bulk.core.ultra_adaptive_kv_cache_engine import (
    UltraAdaptiveKVCacheEngine,
    KVCacheConfig
)

celery_app = Celery(
    'bul_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# Inicializar engine una vez
cache_config = KVCacheConfig(max_tokens=8192)
cache_engine = None

@celery_app.on_after_configure.connect
def setup_engine(sender, **kwargs):
    """Inicializar engine en worker."""
    global cache_engine
    cache_engine = UltraAdaptiveKVCacheEngine(cache_config)

@celery_app.task(bind=True, max_retries=3)
def process_document_task(self, query_data):
    """Task para procesar documento."""
    try:
        result = cache_engine.process_request_sync(query_data)
        return result
    except Exception as e:
        # Retry con backoff
        raise self.retry(exc=e, countdown=60)

@celery_app.task
def batch_process_task(queries):
    """Task para procesar batch."""
    results = []
    for query in queries:
        result = cache_engine.process_request_sync(query)
        results.append(result)
    return results
```

### Uso desde FastAPI

```python
from celery_app import process_document_task, batch_process_task

@app.post("/api/process")
async def process_document(query: dict):
    """Enviar a Celery para procesar."""
    task = process_document_task.delay(query)
    return {"task_id": task.id}

@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    """Obtener estado de task."""
    task = process_document_task.AsyncResult(task_id)
    
    if task.ready():
        return {
            "status": "completed",
            "result": task.result
        }
    return {"status": "processing"}
```

## 🐍 Integración con Django

### Settings

```python
# settings.py
BUL_CACHE_CONFIG = {
    'max_tokens': 8192,
    'cache_strategy': 'adaptive',
    'enable_persistence': True,
    'persistence_path': '/data/cache'
}
```

### Django App

```python
# apps/bul_integration/apps.py
from django.apps import AppConfig
from bulk.core.ultra_adaptive_kv_cache_engine import (
    UltraAdaptiveKVCacheEngine,
    KVCacheConfig
)

class BulIntegrationConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'bul_integration'
    
    def ready(self):
        """Inicializar cache engine."""
        from django.conf import settings
        config = KVCacheConfig(**settings.BUL_CACHE_CONFIG)
        self.cache_engine = UltraAdaptiveKVCacheEngine(config)
```

### View

```python
# views.py
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
@require_http_methods(["POST"])
async def process_query(request):
    """Procesar query con cache."""
    from apps.bul_integration.apps import BulIntegrationConfig
    
    engine = BulIntegrationConfig.cache_engine
    query_data = json.loads(request.body)
    
    result = await engine.process_request(query_data)
    return JsonResponse(result)
```

## 🌶️ Integración con Flask

### Setup

```python
from flask import Flask, request, jsonify
from bulk.core.ultra_adaptive_kv_cache_engine import (
    UltraAdaptiveKVCacheEngine,
    KVCacheConfig
)

app = Flask(__name__)

# Inicializar engine
cache_config = KVCacheConfig(max_tokens=8192)
cache_engine = UltraAdaptiveKVCacheEngine(cache_config)

@app.before_first_request
def setup_cache():
    """Setup cache en primer request."""
    cache_engine.warmup()

@app.route('/api/query', methods=['POST'])
async def process_query():
    """Procesar query."""
    query_data = request.json
    result = await cache_engine.process_request(query_data)
    return jsonify(result)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Obtener estadísticas."""
    stats = cache_engine.get_stats()
    return jsonify(stats)
```

## 📦 Integración con Redis

### Cache Adapter

```python
import redis
from typing import Optional
import json
import pickle

class RedisCacheAdapter:
    """Adapter para usar Redis como backend del KV Cache."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_client = redis.from_url(redis_url)
    
    def get(self, key: str) -> Optional[dict]:
        """Obtener del cache."""
        data = self.redis_client.get(f"kv_cache:{key}")
        if data:
            return pickle.loads(data)
        return None
    
    def set(self, key: str, value: dict, ttl: int = 3600):
        """Guardar en cache."""
        self.redis_client.setex(
            f"kv_cache:{key}",
            ttl,
            pickle.dumps(value)
        )
    
    def delete(self, key: str):
        """Eliminar del cache."""
        self.redis_client.delete(f"kv_cache:{key}")
    
    def clear(self):
        """Limpiar todo el cache."""
        keys = self.redis_client.keys("kv_cache:*")
        if keys:
            self.redis_client.delete(*keys)

# Uso con KV Cache Engine
from bulk.core.ultra_adaptive_kv_cache_engine import UltraAdaptiveKVCacheEngine

redis_adapter = RedisCacheAdapter()
# Integrar con engine según necesidad
```

## 🗄️ Integración con PostgreSQL

### Database Adapter

```python
from sqlalchemy import create_engine, Column, String, LargeBinary, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import pickle

Base = declarative_base()

class CacheEntry(Base):
    """Modelo para entradas de cache."""
    __tablename__ = 'cache_entries'
    
    key = Column(String, primary_key=True)
    value = Column(LargeBinary)
    created_at = Column(DateTime, default=datetime.utcnow)
    accessed_at = Column(DateTime, default=datetime.utcnow)

class PostgresCacheAdapter:
    """Adapter para usar PostgreSQL como backend."""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def get(self, key: str):
        """Obtener del cache."""
        session = self.Session()
        try:
            entry = session.query(CacheEntry).filter_by(key=key).first()
            if entry:
                entry.accessed_at = datetime.utcnow()
                session.commit()
                return pickle.loads(entry.value)
            return None
        finally:
            session.close()
    
    def set(self, key: str, value: dict):
        """Guardar en cache."""
        session = self.Session()
        try:
            entry = CacheEntry(
                key=key,
                value=pickle.dumps(value),
                created_at=datetime.utcnow(),
                accessed_at=datetime.utcnow()
            )
            session.merge(entry)
            session.commit()
        finally:
            session.close()
```

## 📊 Integración con Prometheus

### Métricas Personalizadas

```python
from prometheus_client import Counter, Histogram, Gauge
from bulk.core.ultra_adaptive_kv_cache_prometheus import PrometheusExporter

# Métricas del KV Cache
cache_requests = Counter(
    'kv_cache_requests_total',
    'Total cache requests',
    ['status']  # 'hit' or 'miss'
)

cache_latency = Histogram(
    'kv_cache_latency_seconds',
    'Cache latency',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

cache_size = Gauge(
    'kv_cache_size_bytes',
    'Current cache size'
)

class PrometheusMetrics:
    """Wrapper para métricas Prometheus."""
    
    def record_request(self, cache_hit: bool):
        """Registrar request."""
        status = 'hit' if cache_hit else 'miss'
        cache_requests.labels(status=status).inc()
    
    def record_latency(self, duration: float):
        """Registrar latencia."""
        cache_latency.observe(duration)
    
    def update_cache_size(self, size: int):
        """Actualizar tamaño del cache."""
        cache_size.set(size)

# Integrar con engine
metrics = PrometheusMetrics()
# Conectar con eventos del engine
```

## 📈 Integración con Grafana

### Dashboard Configuration

```json
{
  "dashboard": {
    "title": "BUL KV Cache Metrics",
    "panels": [
      {
        "title": "Cache Hit Rate",
        "targets": [{
          "expr": "rate(kv_cache_requests_total{status='hit'}[5m]) / rate(kv_cache_requests_total[5m])"
        }]
      },
      {
        "title": "Latency P95",
        "targets": [{
          "expr": "histogram_quantile(0.95, kv_cache_latency_seconds_bucket)"
        }]
      },
      {
        "title": "Cache Size",
        "targets": [{
          "expr": "kv_cache_size_bytes"
        }]
      }
    ]
  }
}
```

## 🌐 API REST

### Cliente Python

```python
import requests

class BULAPIClient:
    """Cliente para API REST."""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def process_query(self, query: str, priority: int = 1):
        """Procesar query."""
        response = self.session.post(
            f"{self.base_url}/api/query",
            json={"query": query, "priority": priority}
        )
        response.raise_for_status()
        return response.json()
    
    def get_task_status(self, task_id: str):
        """Obtener estado de task."""
        response = self.session.get(
            f"{self.base_url}/api/task/{task_id}/status"
        )
        response.raise_for_status()
        return response.json()
    
    def get_stats(self):
        """Obtener estadísticas."""
        response = self.session.get(f"{self.base_url}/api/stats")
        response.raise_for_status()
        return response.json()

# Uso
client = BULAPIClient()
result = client.process_query("Create marketing strategy")
```

## 🔌 WebSocket

### Servidor WebSocket

```python
from fastapi import WebSocket
import json

@app.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    """WebSocket para queries en tiempo real."""
    await websocket.accept()
    
    try:
        while True:
            # Recibir query
            data = await websocket.receive_json()
            query = data.get("query")
            
            # Procesar con cache
            result = await cache_engine.process_request({
                "text": query,
                "priority": 1
            })
            
            # Enviar resultado
            await websocket.send_json({
                "result": result["result"],
                "cache_hit": result["cache_hit"],
                "latency": result["latency"]
            })
    except Exception as e:
        await websocket.close(code=1000)
```

### Cliente WebSocket

```python
import asyncio
import websockets
import json

async def query_websocket(query: str):
    """Cliente WebSocket."""
    uri = "ws://localhost:8002/ws/query"
    
    async with websockets.connect(uri) as websocket:
        # Enviar query
        await websocket.send(json.dumps({"query": query}))
        
        # Recibir resultado
        result = await websocket.recv()
        return json.loads(result)

# Uso
result = asyncio.run(query_websocket("Create marketing strategy"))
```

## 🔷 gRPC

### Definición Proto

```protobuf
// bul.proto
syntax = "proto3";

service BULService {
  rpc ProcessQuery(QueryRequest) returns (QueryResponse);
  rpc GetStats(StatsRequest) returns (StatsResponse);
}

message QueryRequest {
  string query = 1;
  int32 priority = 2;
}

message QueryResponse {
  string result = 1;
  bool cache_hit = 2;
  float latency_ms = 3;
}
```

### Servidor gRPC

```python
import grpc
from concurrent import futures
import bul_pb2
import bul_pb2_grpc

class BULServiceServicer(bul_pb2_grpc.BULServiceServicer):
    """Servidor gRPC."""
    
    def __init__(self):
        self.cache_engine = UltraAdaptiveKVCacheEngine(KVCacheConfig())
    
    async def ProcessQuery(self, request, context):
        """Procesar query."""
        result = await self.cache_engine.process_request({
            "text": request.query,
            "priority": request.priority
        })
        
        return bul_pb2.QueryResponse(
            result=result["result"],
            cache_hit=result["cache_hit"],
            latency_ms=result["latency"]
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    bul_pb2_grpc.add_BULServiceServicer_to_server(
        BULServiceServicer(), server
    )
    server.add_in_port('[::]:50051', grpc.insecure_server_credentials())
    server.start()
    server.wait_for_termination()
```

---

**Más información:**
- [API Reference](API_REFERENCE.md)
- [Best Practices](BEST_PRACTICES.md)
- [Examples](../bulk/EXAMPLES.md)

