# ⚡ Guía de Rendimiento - Music Analyzer AI

Esta guía te ayudará a optimizar el rendimiento de Music Analyzer AI.

## 📊 Métricas de Rendimiento

### Objetivos

- **Latencia P95**: < 200ms
- **Throughput**: > 100 req/s
- **Disponibilidad**: > 99.9%
- **Tiempo de respuesta promedio**: < 100ms

### Monitoreo

```python
import time
from prometheus_client import Counter, Histogram

request_count = Counter('requests_total', 'Total requests')
request_duration = Histogram('request_duration_seconds', 'Request duration')

@app.middleware("http")
async def track_performance(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    
    request_count.inc()
    request_duration.observe(duration)
    
    return response
```

## 🚀 Optimizaciones Principales

### 1. Caché

#### Redis Caché

```python
import redis
import json
from functools import wraps

redis_client = redis.Redis.from_url(os.getenv("REDIS_URL"))

def cache_result(ttl=3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{args}:{kwargs}"
            
            # Intentar obtener de caché
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Ejecutar función
            result = await func(*args, **kwargs)
            
            # Guardar en caché
            redis_client.setex(
                cache_key,
                ttl,
                json.dumps(result)
            )
            
            return result
        return wrapper
    return decorator

# Uso
@cache_result(ttl=3600)
async def analyze_track(track_id: str):
    # Análisis costoso
    pass
```

#### Caché en Memoria

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def expensive_calculation(input_data: str):
    # Cálculo costoso
    return result
```

### 2. Async/Await

✅ **Correcto:**
```python
async def analyze_multiple_tracks(track_ids: List[str]):
    tasks = [analyze_track(tid) for tid in track_ids]
    return await asyncio.gather(*tasks)
```

❌ **Incorrecto:**
```python
def analyze_multiple_tracks(track_ids: List[str]):
    results = []
    for tid in track_ids:  # Secuencial, lento
        results.append(analyze_track(tid))
    return results
```

### 3. Connection Pooling

```python
import httpx

# Crear cliente reutilizable
async_client = httpx.AsyncClient(
    timeout=30.0,
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
)

# Reutilizar en toda la aplicación
async def get_track_from_spotify(track_id: str):
    response = await async_client.get(f"https://api.spotify.com/v1/tracks/{track_id}")
    return response.json()
```

### 4. Procesamiento en Lotes

```python
async def analyze_tracks_batch(track_ids: List[str], batch_size=10):
    results = []
    
    for i in range(0, len(track_ids), batch_size):
        batch = track_ids[i:i + batch_size]
        batch_tasks = [analyze_track(tid) for tid in batch]
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)
    
    return results
```

### 5. Lazy Loading

```python
class TrackAnalysis:
    def __init__(self, track_id: str):
        self.track_id = track_id
        self._analysis = None
        self._coaching = None
    
    @property
    async def analysis(self):
        if self._analysis is None:
            self._analysis = await self._load_analysis()
        return self._analysis
    
    @property
    async def coaching(self):
        if self._coaching is None:
            self._coaching = await self._load_coaching()
        return self._coaching
```

## 🔧 Optimización de Base de Datos

### Índices

```sql
-- Crear índices para queries frecuentes
CREATE INDEX idx_track_id ON analyses(track_id);
CREATE INDEX idx_user_id ON analyses(user_id);
CREATE INDEX idx_created_at ON analyses(created_at);
```

### Queries Optimizadas

✅ **Correcto:**
```python
# Solo obtener campos necesarios
track = session.query(Track.id, Track.name).filter(Track.id == track_id).first()
```

❌ **Incorrecto:**
```python
# Obtener todo
track = session.query(Track).filter(Track.id == track_id).first()
```

### Connection Pooling

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True
)
```

## ⚡ Optimización de Código

### Evitar Operaciones Costosas

```python
# ❌ Mal: Operación costosa en cada request
@app.get("/tracks")
async def get_tracks():
    all_tracks = load_all_tracks_from_file()  # Costoso
    return all_tracks

# ✅ Bien: Cargar una vez
ALL_TRACKS = load_all_tracks_from_file()  # Al inicio

@app.get("/tracks")
async def get_tracks():
    return ALL_TRACKS
```

### Usar Generadores

```python
# ✅ Bien: Generador (memoria eficiente)
def process_tracks(track_ids):
    for track_id in track_ids:
        yield process_track(track_id)

# ❌ Mal: Lista completa en memoria
def process_tracks(track_ids):
    return [process_track(tid) for tid in track_ids]
```

### Compilación JIT

```python
# Para cálculos numéricos intensivos
from numba import jit

@jit(nopython=True)
def fast_calculation(data):
    # Cálculo optimizado
    return result
```

## 🚀 Optimización del Servidor

### Múltiples Workers

```bash
# Producción
uvicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
```

### Configuración de Uvicorn

```python
# uvicorn_config.py
config = {
    "workers": 4,
    "worker_class": "uvicorn.workers.UvicornWorker",
    "timeout_keep_alive": 30,
    "limit_concurrency": 1000,
    "backlog": 2048
}
```

### Load Balancing

```nginx
# Nginx load balancer
upstream music_analyzer {
    least_conn;
    server 127.0.0.1:8010;
    server 127.0.0.1:8011;
    server 127.0.0.1:8012;
    server 127.0.0.1:8013;
}

server {
    location / {
        proxy_pass http://music_analyzer;
    }
}
```

## 📈 Profiling

### Profiling con cProfile

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Tu código aquí
await analyze_track(track_id)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 funciones más lentas
```

### Profiling con py-spy

```bash
# Instalar
pip install py-spy

# Profiling en tiempo real
py-spy record -o profile.svg -- python main.py
```

### Async Profiling

```python
import asyncio
import time

async def profile_async_function(func, *args, **kwargs):
    start = time.time()
    result = await func(*args, **kwargs)
    duration = time.time() - start
    print(f"{func.__name__} took {duration:.2f}s")
    return result
```

## 🎯 Optimizaciones Específicas

### Análisis Musical

```python
# ✅ Cachear análisis de tracks populares
POPULAR_TRACKS_CACHE = {}

async def analyze_track_optimized(track_id: str):
    # Verificar si es track popular
    if track_id in POPULAR_TRACKS_CACHE:
        return POPULAR_TRACKS_CACHE[track_id]
    
    # Análisis normal
    analysis = await analyze_track(track_id)
    
    # Cachear si es popular
    if analysis.popularity > 70:
        POPULAR_TRACKS_CACHE[track_id] = analysis
    
    return analysis
```

### Búsqueda Optimizada

```python
# ✅ Usar índices de búsqueda
from elasticsearch import AsyncElasticsearch

es = AsyncElasticsearch()

async def search_tracks_optimized(query: str):
    # Búsqueda con Elasticsearch (más rápido que Spotify API)
    results = await es.search(
        index="tracks",
        body={"query": {"match": {"name": query}}}
    )
    return results
```

## 📊 Monitoreo de Rendimiento

### Métricas Clave

```python
from prometheus_client import Gauge, Summary

# Métricas
active_requests = Gauge('active_requests', 'Active requests')
request_latency = Summary('request_latency_seconds', 'Request latency')

@app.middleware("http")
async def track_metrics(request: Request, call_next):
    active_requests.inc()
    with request_latency.time():
        response = await call_next(request)
    active_requests.dec()
    return response
```

### Alertas

```python
# Alertar si latencia es alta
if request_duration > 1.0:  # > 1 segundo
    logger.warning("high_latency", duration=request_duration, endpoint=request.url.path)
```

## ✅ Checklist de Optimización

### Caché
- [ ] Redis configurado
- [ ] Caché para queries frecuentes
- [ ] TTL apropiado
- [ ] Invalidación de caché

### Async
- [ ] Operaciones I/O son async
- [ ] Uso de asyncio.gather para paralelismo
- [ ] Connection pooling

### Base de Datos
- [ ] Índices creados
- [ ] Queries optimizadas
- [ ] Connection pooling
- [ ] Sin N+1 queries

### Código
- [ ] Sin operaciones costosas repetidas
- [ ] Uso de generadores cuando apropiado
- [ ] Lazy loading implementado

### Servidor
- [ ] Múltiples workers
- [ ] Load balancing
- [ ] Monitoreo activo

---

**Última actualización**: 2025  
**Versión**: 2.21.0






