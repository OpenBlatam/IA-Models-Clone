# 🍳 Cookbook de Ejemplos - Blatam Academy Features

## 📋 Tabla de Contenidos

- [Ejemplos Básicos](#ejemplos-básicos)
- [Ejemplos Intermedios](#ejemplos-intermedios)
- [Ejemplos Avanzados](#ejemplos-avanzados)
- [Patrones Comunes](#patrones-comunes)
- [Recetas de Integración](#recetas-de-integración)

## 🎯 Ejemplos Básicos

### Ejemplo 1: Setup Básico

```python
from bulk.core.ultra_adaptive_kv_cache_engine import (
    UltraAdaptiveKVCacheEngine,
    KVCacheConfig,
    CacheStrategy
)

# Configuración básica
config = KVCacheConfig(
    max_tokens=4096,
    cache_strategy=CacheStrategy.ADAPTIVE
)

# Crear engine
engine = UltraAdaptiveKVCacheEngine(config)

# Procesar request
result = await engine.process_request({
    'text': 'Create a marketing strategy',
    'priority': 1
})

print(f"Result: {result['result']}")
print(f"Cache hit: {result['cache_hit']}")
```

### Ejemplo 2: Cache Simple

```python
import asyncio

async def simple_cache_example():
    """Ejemplo simple de uso de cache."""
    config = KVCacheConfig(max_tokens=2048)
    engine = UltraAdaptiveKVCacheEngine(config)
    
    # Primera llamada - cache miss
    result1 = await engine.process_request({
        'text': 'Hello world',
        'priority': 1
    })
    print(f"First call: {result1['cache_hit']}")  # False
    
    # Segunda llamada - cache hit
    result2 = await engine.process_request({
        'text': 'Hello world',
        'priority': 1
    })
    print(f"Second call: {result2['cache_hit']}")  # True

asyncio.run(simple_cache_example())
```

### Ejemplo 3: Obtener Estadísticas

```python
async def stats_example():
    """Ejemplo de obtención de estadísticas."""
    config = KVCacheConfig(max_tokens=4096)
    engine = UltraAdaptiveKVCacheEngine(config)
    
    # Procesar algunos requests
    for i in range(10):
        await engine.process_request({
            'text': f'Query {i % 3}',  # Algunas repeticiones
            'priority': 1
        })
    
    # Obtener estadísticas
    stats = engine.get_stats()
    
    print(f"Total requests: {stats['total_requests']}")
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Cache misses: {stats['cache_misses']}")
    print(f"Hit rate: {stats['hit_rate']:.2%}")
    print(f"Avg latency: {stats['avg_latency']:.2f}ms")

asyncio.run(stats_example())
```

## 🎓 Ejemplos Intermedios

### Ejemplo 4: Batch Processing

```python
async def batch_processing_example():
    """Ejemplo de procesamiento en batch."""
    config = KVCacheConfig(max_tokens=8192)
    engine = UltraAdaptiveKVCacheEngine(config)
    
    # Crear batch de requests
    requests = [
        {'text': f'Generate document {i}', 'priority': 1}
        for i in range(100)
    ]
    
    # Procesar en batch optimizado
    results = await engine.process_batch_optimized(
        requests,
        batch_size=20
    )
    
    print(f"Processed {len(results)} requests")
    print(f"Cache hits: {sum(1 for r in results if r['cache_hit'])}")

asyncio.run(batch_processing_example())
```

### Ejemplo 5: Con Persistencia

```python
async def persistence_example():
    """Ejemplo con persistencia."""
    config = KVCacheConfig(
        max_tokens=4096,
        enable_persistence=True,
        persistence_path='/data/cache'
    )
    
    engine = UltraAdaptiveKVCacheEngine(config)
    
    # Procesar y almacenar
    await engine.process_request({
        'text': 'Important query',
        'priority': 1
    })
    
    # Persistir
    engine.persist()
    print("Cache persisted")
    
    # Crear nuevo engine y cargar
    new_engine = UltraAdaptiveKVCacheEngine(config)
    new_engine.load()
    
    # Verificar que se restauró
    result = await new_engine.process_request({
        'text': 'Important query',
        'priority': 1
    })
    print(f"Cache hit after restore: {result['cache_hit']}")  # True

asyncio.run(persistence_example())
```

### Ejemplo 6: Con Compresión

```python
async def compression_example():
    """Ejemplo con compresión."""
    # Sin compresión
    config_no_comp = KVCacheConfig(
        max_tokens=4096,
        use_compression=False
    )
    engine_no_comp = UltraAdaptiveKVCacheEngine(config_no_comp)
    
    # Con compresión
    config_comp = KVCacheConfig(
        max_tokens=4096,
        use_compression=True,
        compression_ratio=0.3
    )
    engine_comp = UltraAdaptiveKVCacheEngine(config_comp)
    
    # Comparar uso de memoria
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        await engine_no_comp.process_request({'text': 'Test', 'priority': 1})
        memory_no_comp = torch.cuda.max_memory_allocated()
        
        torch.cuda.reset_peak_memory_stats()
        await engine_comp.process_request({'text': 'Test', 'priority': 1})
        memory_comp = torch.cuda.max_memory_allocated()
        
        print(f"Memory without compression: {memory_no_comp / 1024**2:.2f} MB")
        print(f"Memory with compression: {memory_comp / 1024**2:.2f} MB")
        print(f"Savings: {(1 - memory_comp/memory_no_comp)*100:.1f}%")

asyncio.run(compression_example())
```

## 🚀 Ejemplos Avanzados

### Ejemplo 7: Multi-GPU

```python
async def multi_gpu_example():
    """Ejemplo con múltiples GPUs."""
    import torch
    
    if torch.cuda.device_count() < 2:
        print("Requires at least 2 GPUs")
        return
    
    config = KVCacheConfig(
        max_tokens=16384,
        enable_distributed=True,
        distributed_backend="nccl"
    )
    
    engine = UltraAdaptiveKVCacheEngine(config)
    
    # Procesar requests distribuidos
    requests = [
        {'text': f'Query {i}', 'priority': 1}
        for i in range(1000)
    ]
    
    results = await asyncio.gather(*[
        engine.process_request(req) for req in requests
    ])
    
    # Verificar uso de múltiples GPUs
    stats = engine.get_stats()
    print(f"GPU count: {stats.get('gpu_count', 1)}")
    print(f"Processed {len(results)} requests")

asyncio.run(multi_gpu_example())
```

### Ejemplo 8: Adaptive Strategy

```python
async def adaptive_strategy_example():
    """Ejemplo demostrando estrategia adaptativa."""
    from bulk.core.ultra_adaptive_kv_cache_engine import AdaptiveKVCache
    
    config = KVCacheConfig(
        max_tokens=8192,
        cache_strategy=CacheStrategy.ADAPTIVE
    )
    
    cache = AdaptiveKVCache(config)
    
    # Simular diferentes patrones de acceso
    # Patrón 1: Secuencial (debería usar LRU)
    for i in range(100):
        await cache.process_request({'text': f'Query {i}', 'priority': 1})
    
    print(f"Strategy after sequential: {cache.get_current_strategy()}")
    
    # Patrón 2: Repetitivo (debería usar LFU)
    queries = ['Query A', 'Query B', 'Query C']
    for _ in range(100):
        import random
        await cache.process_request({
            'text': random.choice(queries),
            'priority': 1
        })
    
    print(f"Strategy after repetitive: {cache.get_current_strategy()}")

asyncio.run(adaptive_strategy_example())
```

### Ejemplo 9: Prefetching Inteligente

```python
async def prefetching_example():
    """Ejemplo de prefetching inteligente."""
    config = KVCacheConfig(
        max_tokens=8192,
        enable_prefetch=True,
        prefetch_size=8
    )
    
    engine = UltraAdaptiveKVCacheEngine(config)
    
    # Simular patrón donde Query N+1 sigue a Query N
    current_query = "Start"
    for i in range(50):
        result = await engine.process_request({
            'text': current_query,
            'priority': 1
        })
        
        # Predecir siguiente (en este caso, secuencial)
        current_query = f"Query {i+1}"
    
    stats = engine.get_stats()
    prefetch_stats = engine.get_prefetch_stats()
    
    print(f"Prefetch hits: {prefetch_stats.get('hits', 0)}")
    print(f"Cache hit rate: {stats['hit_rate']:.2%}")

asyncio.run(prefetching_example())
```

## 🎨 Patrones Comunes

### Patrón 1: Singleton Engine

```python
class CacheEngineSingleton:
    """Singleton para el cache engine."""
    _instance = None
    _engine = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_engine(self) -> UltraAdaptiveKVCacheEngine:
        """Obtener o crear engine."""
        if self._engine is None:
            config = KVCacheConfig(max_tokens=8192)
            self._engine = UltraAdaptiveKVCacheEngine(config)
        return self._engine

# Uso
engine = CacheEngineSingleton().get_engine()
```

### Patrón 2: Factory Pattern

```python
class CacheEngineFactory:
    """Factory para crear engines con diferentes configuraciones."""
    
    @staticmethod
    def create_development() -> UltraAdaptiveKVCacheEngine:
        """Crear engine para desarrollo."""
        config = KVCacheConfig(
            max_tokens=2048,
            enable_profiling=True
        )
        return UltraAdaptiveKVCacheEngine(config)
    
    @staticmethod
    def create_production() -> UltraAdaptiveKVCacheEngine:
        """Crear engine para producción."""
        config = KVCacheConfig(
            max_tokens=16384,
            enable_persistence=True,
            enable_prefetch=True
        )
        return UltraAdaptiveKVCacheEngine(config)
    
    @staticmethod
    def create_high_performance() -> UltraAdaptiveKVCacheEngine:
        """Crear engine de alto rendimiento."""
        config = KVCacheConfig(
            max_tokens=16384,
            use_compression=False,
            enable_prefetch=True,
            prefetch_size=32
        )
        return UltraAdaptiveKVCacheEngine(config)

# Uso
engine = CacheEngineFactory.create_production()
```

### Patrón 3: Decorator para Cache

```python
from functools import wraps

def cached(engine: UltraAdaptiveKVCacheEngine):
    """Decorator para cachear resultados."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Crear key del cache
            import hashlib
            import json
            key = hashlib.md5(
                json.dumps({'args': args, 'kwargs': kwargs}).encode()
            ).hexdigest()
            
            # Intentar obtener del cache
            cached_result = await engine.get_from_cache(key)
            if cached_result:
                return cached_result
            
            # Ejecutar función
            result = await func(*args, **kwargs)
            
            # Guardar en cache
            await engine.store_in_cache(key, result)
            
            return result
        return wrapper
    return decorator

# Uso
@cached(engine)
async def expensive_function(query: str):
    # Procesamiento costoso
    return {'result': f'Processed {query}'}
```

## 🔗 Recetas de Integración

### Receta 1: FastAPI Integration

```python
from fastapi import FastAPI, Depends
from bulk.core.ultra_adaptive_kv_cache_engine import (
    UltraAdaptiveKVCacheEngine,
    KVCacheConfig
)

app = FastAPI()

# Inicializar engine una vez
cache_config = KVCacheConfig(max_tokens=8192)
cache_engine = UltraAdaptiveKVCacheEngine(cache_config)

@app.get("/cache/stats")
async def get_stats():
    """Obtener estadísticas del cache."""
    return cache_engine.get_stats()

@app.post("/cache/query")
async def process_query(query: dict):
    """Procesar query con cache."""
    return await cache_engine.process_request(query)
```

### Receta 2: Celery Integration

```python
from celery import Celery
from bulk.core.ultra_adaptive_kv_cache_engine import (
    UltraAdaptiveKVCacheEngine,
    KVCacheConfig
)

celery_app = Celery('tasks')

# Inicializar engine en worker
cache_engine = None

@celery_app.on_after_configure.connect
def setup_cache(sender, **kwargs):
    global cache_engine
    config = KVCacheConfig(max_tokens=8192)
    cache_engine = UltraAdaptiveKVCacheEngine(config)

@celery_app.task
def process_with_cache(query_data):
    """Task con cache."""
    return cache_engine.process_request_sync(query_data)
```

### Receta 3: Context Manager

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def cache_engine_context(config: KVCacheConfig):
    """Context manager para cache engine."""
    engine = UltraAdaptiveKVCacheEngine(config)
    try:
        yield engine
    finally:
        # Cleanup si es necesario
        engine.persist()
        engine.clear_cache()

# Uso
async def example():
    config = KVCacheConfig(max_tokens=4096)
    async with cache_engine_context(config) as engine:
        result = await engine.process_request({'text': 'Test', 'priority': 1})
        return result
```

---

**Más información:**
- [Ejemplos BUL](bulk/EXAMPLES.md)
- [Advanced Usage](bulk/ADVANCED_USAGE_GUIDE.md)
- [Integration Guide](INTEGRATION_GUIDE.md)

