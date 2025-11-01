# 🍳 Recetas de Configuración - Blatam Academy Features

## 📋 Configuraciones Listas para Usar

### Receta 1: Desarrollo Local Rápido

```python
# development_quick.py
from bulk.core.ultra_adaptive_kv_cache_engine import (
    UltraAdaptiveKVCacheEngine,
    KVCacheConfig,
    CacheStrategy
)

config = KVCacheConfig(
    max_tokens=2048,
    cache_strategy=CacheStrategy.ADAPTIVE,
    enable_profiling=True,  # Para debugging
    enable_persistence=False
)

engine = UltraAdaptiveKVCacheEngine(config)
# Listo para usar!
```

### Receta 2: Producción Estándar

```python
# production_standard.py
config = KVCacheConfig(
    max_tokens=8192,
    cache_strategy=CacheStrategy.ADAPTIVE,
    enable_persistence=True,
    persistence_path='/data/kv_cache',
    enable_prefetch=True,
    prefetch_size=16,
    use_compression=True,
    compression_ratio=0.3,
    dtype=torch.float16,
    pin_memory=True,
    non_blocking=True
)

engine = UltraAdaptiveKVCacheEngine(config)
```

### Receta 3: Alto Rendimiento (GPU Disponible)

```python
# high_performance_gpu.py
config = KVCacheConfig(
    max_tokens=16384,
    cache_strategy=CacheStrategy.ADAPTIVE,
    enable_prefetch=True,
    prefetch_size=32,
    use_compression=False,  # Sin compresión para máxima velocidad
    enable_distributed=True,  # Si hay múltiples GPUs
    distributed_backend="nccl",
    dtype=torch.float16,
    pin_memory=True,
    non_blocking=True
)

engine = UltraAdaptiveKVCacheEngine(config)
```

### Receta 4: Memoria Limitada (2GB Disponible)

```python
# memory_constrained.py
config = KVCacheConfig(
    max_tokens=1024,
    cache_strategy=CacheStrategy.LRU,  # LRU más eficiente en memoria
    use_compression=True,
    compression_ratio=0.15,  # Muy agresiva
    use_quantization=True,
    quantization_bits=4,
    enable_gc=True,
    gc_threshold=0.5,  # GC más frecuente
    dtype=torch.float16
)

engine = UltraAdaptiveKVCacheEngine(config)
```

### Receta 5: CPU Only (Sin GPU)

```python
# cpu_only.py
config = KVCacheConfig(
    max_tokens=4096,
    device='cpu',
    use_compression=True,
    compression_ratio=0.2,
    enable_gc=True,
    gc_threshold=0.6,
    dtype=torch.float32  # CPU usa float32 mejor
)

engine = UltraAdaptiveKVCacheEngine(config)
```

### Receta 6: Batch Processing Masivo

```python
# batch_processing.py
config = KVCacheConfig(
    max_tokens=16384,  # Cache grande para muchos datos
    cache_strategy=CacheStrategy.ADAPTIVE,
    enable_persistence=True,  # Para batches largos
    enable_prefetch=True,
    prefetch_size=16,
    use_compression=True,
    compression_ratio=0.3
)

engine = UltraAdaptiveKVCacheEngine(config)

# Procesar batch
async def process_large_batch(requests, batch_size=50):
    return await engine.process_batch_optimized(
        requests,
        batch_size=batch_size
    )
```

### Receta 7: Real-time con Baja Latencia

```python
# realtime_low_latency.py
config = KVCacheConfig(
    max_tokens=8192,
    cache_strategy=CacheStrategy.ADAPTIVE,
    enable_prefetch=True,
    prefetch_size=16,
    use_compression=False,  # Sin compresión = más rápido
    pin_memory=True,
    non_blocking=True,
    dtype=torch.float16
)

engine = UltraAdaptiveKVCacheEngine(config)
```

### Receta 8: Multi-Tenant SaaS

```python
# multi_tenant.py
config = KVCacheConfig(
    max_tokens=8192,
    multi_tenant=True,
    tenant_isolation=True,  # Aislamiento entre tenants
    cache_strategy=CacheStrategy.ADAPTIVE,
    enable_persistence=True,
    persistence_path='/data/kv_cache'
)

engine = UltraAdaptiveKVCacheEngine(config)
```

## 🔧 Recetas de Integración

### Receta: FastAPI con Cache

```python
# fastapi_with_cache.py
from fastapi import FastAPI, Depends
from bulk.core.ultra_adaptive_kv_cache_engine import (
    UltraAdaptiveKVCacheEngine,
    KVCacheConfig
)

app = FastAPI()

# Inicializar una vez
cache_config = KVCacheConfig(max_tokens=8192)
cache_engine = UltraAdaptiveKVCacheEngine(cache_config)

def get_cache_engine():
    return cache_engine

@app.post("/api/query")
async def query(request: dict, engine = Depends(get_cache_engine)):
    return await engine.process_request(request)
```

### Receta: Django con Cache

```python
# django_cache.py
# settings.py
BUL_CACHE_CONFIG = {
    'max_tokens': 8192,
    'cache_strategy': 'adaptive',
    'enable_persistence': True
}

# views.py
from django.conf import settings
from bulk.core.ultra_adaptive_kv_cache_engine import (
    UltraAdaptiveKVCacheEngine,
    KVCacheConfig
)

cache_engine = UltraAdaptiveKVCacheEngine(
    KVCacheConfig(**settings.BUL_CACHE_CONFIG)
)

async def process_query(request):
    result = await cache_engine.process_request(request.json())
    return JsonResponse(result)
```

### Receta: Celery Worker con Cache

```python
# celery_cache.py
from celery import Celery
from bulk.core.ultra_adaptive_kv_cache_engine import (
    UltraAdaptiveKVCacheEngine,
    KVCacheConfig
)

celery_app = Celery('tasks')

# Cache por worker
cache_engine = None

@celery_app.on_after_configure.connect
def setup_cache(sender, **kwargs):
    global cache_engine
    config = KVCacheConfig(max_tokens=4096)
    cache_engine = UltraAdaptiveKVCacheEngine(config)

@celery_app.task
def process_with_cache(query_data):
    return cache_engine.process_request_sync(query_data)
```

## 📝 Recetas de .env

### .env para Desarrollo

```bash
# development.env
APP_ENV=development
DEBUG=true
LOG_LEVEL=DEBUG

KV_CACHE_MAX_TOKENS=2048
KV_CACHE_STRATEGY=adaptive
KV_CACHE_ENABLE_PROFILING=true
KV_CACHE_ENABLE_PERSISTENCE=false
```

### .env para Producción

```bash
# production.env
APP_ENV=production
DEBUG=false
LOG_LEVEL=INFO

KV_CACHE_MAX_TOKENS=8192
KV_CACHE_STRATEGY=adaptive
KV_CACHE_ENABLE_PERSISTENCE=true
KV_CACHE_PERSISTENCE_PATH=/data/kv_cache
KV_CACHE_ENABLE_PREFETCH=true
KV_CACHE_PREFETCH_SIZE=16
KV_CACHE_USE_COMPRESSION=true
KV_CACHE_COMPRESSION_RATIO=0.3
```

## 🎯 Recetas por Objetivo

### Objetivo: Máxima Velocidad

```python
config = KVCacheConfig(
    max_tokens=16384,
    cache_strategy=CacheStrategy.ADAPTIVE,
    enable_prefetch=True,
    prefetch_size=32,
    use_compression=False,
    pin_memory=True,
    non_blocking=True,
    dtype=torch.float16
)
```

### Objetivo: Mínima Memoria

```python
config = KVCacheConfig(
    max_tokens=1024,
    use_compression=True,
    compression_ratio=0.15,
    use_quantization=True,
    quantization_bits=4,
    enable_gc=True,
    gc_threshold=0.5
)
```

### Objetivo: Máximo Hit Rate

```python
config = KVCacheConfig(
    max_tokens=16384,  # Cache muy grande
    cache_strategy=CacheStrategy.ADAPTIVE,
    enable_prefetch=True,
    enable_persistence=True  # Evitar cold starts
)
```

### Objetivo: Balance Óptimo

```python
config = KVCacheConfig(
    max_tokens=8192,
    cache_strategy=CacheStrategy.ADAPTIVE,
    enable_prefetch=True,
    prefetch_size=16,
    use_compression=True,
    compression_ratio=0.3,
    enable_persistence=True
)
```

---

**Más información:**
- [Configuration Decision Tree](CONFIGURATION_DECISION_TREE.md)
- [Quick Setup Guides](QUICK_SETUP_GUIDES.md)
- [Examples Cookbook](EXAMPLES_COOKBOOK.md)

