# 🔍 Troubleshooting Avanzado - BUL KV Cache

## 📋 Tabla de Contenidos

- [Problemas de Rendimiento](#problemas-de-rendimiento)
- [Problemas de Memoria](#problemas-de-memoria)
- [Problemas de GPU](#problemas-de-gpu)
- [Problemas de Cache](#problemas-de-cache)
- [Problemas de Persistencia](#problemas-de-persistencia)
- [Problemas de Distribución](#problemas-de-distribución)
- [Debugging Avanzado](#debugging-avanzado)

## ⚡ Problemas de Rendimiento

### Latencia Alta

#### Síntomas
- Latencia P50 > 200ms
- Latencia P95 > 1s
- Timeouts frecuentes

#### Diagnóstico

```python
from bulk.core.ultra_adaptive_kv_cache_engine import UltraAdaptiveKVCacheEngine

engine = UltraAdaptiveKVCacheEngine(config)

# 1. Verificar estadísticas
stats = engine.get_stats()
print(f"Avg latency: {stats['avg_latency']}ms")
print(f"P95 latency: {stats['p95_latency']}ms")
print(f"Cache hit rate: {stats['hit_rate']:.2%}")

# 2. Verificar bottlenecks
from bulk.core.ultra_adaptive_kv_cache_analytics import PerformanceAnalyzer

analyzer = PerformanceAnalyzer(engine)
bottlenecks = analyzer.identify_bottlenecks()
print(f"Bottlenecks: {bottlenecks}")
```

#### Soluciones

1. **Optimizar configuración**
```python
config = KVCacheConfig(
    max_tokens=8192,
    cache_strategy=CacheStrategy.ADAPTIVE,
    enable_prefetch=True,
    prefetch_size=16,
    use_compression=False,  # Desactivar si afecta latencia
    pin_memory=True,
    non_blocking=True
)
```

2. **Aumentar tamaño de cache**
```python
config.max_tokens = 16384  # Aumentar cache
```

3. **Habilitar prefetching más agresivo**
```python
config.enable_prefetch = True
config.prefetch_size = 32  # Aumentar tamaño de prefetch
```

4. **Verificar GPU utilization**
```bash
nvidia-smi -l 1
# Verificar que GPU no esté saturado
```

### Throughput Bajo

#### Síntomas
- <50 req/s cuando debería ser >100 req/s
- Queue backlog creciendo

#### Diagnóstico

```python
# Medir throughput real
import time
import asyncio

async def benchmark_throughput(engine, num_requests=1000):
    start = time.time()
    tasks = [
        engine.process_request({'text': f'Query {i}'})
        for i in range(num_requests)
    ]
    await asyncio.gather(*tasks)
    duration = time.time() - start
    throughput = num_requests / duration
    print(f"Throughput: {throughput:.2f} req/s")
```

#### Soluciones

1. **Aumentar workers**
```python
config.num_workers = 32  # Más workers
```

2. **Batch processing**
```python
# Procesar en batches
results = await engine.process_batch_optimized(
    requests,
    batch_size=20
)
```

3. **Distribuir carga**
```python
# Usar múltiples instancias
# Configurar load balancer
```

## 💾 Problemas de Memoria

### Memory Leak

#### Síntomas
- Memoria crece continuamente
- OOM (Out of Memory) errors

#### Diagnóstico

```python
import tracemalloc
import gc

# Iniciar tracking
tracemalloc.start()

# Tu código
result = await engine.process_request({'text': 'test'})

# Tomar snapshot
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

# Top 10 líneas con más memoria
for stat in top_stats[:10]:
    print(stat)

# Verificar objetos no recolectados
gc.collect()
print(f"Uncollectable: {len(gc.garbage)}")
```

#### Soluciones

1. **Forzar garbage collection**
```python
config.enable_gc = True
config.gc_threshold = 0.7  # Más agresivo
```

2. **Reducir tamaño de cache**
```python
config.max_tokens = 4096  # Reducir
config.use_compression = True
config.compression_ratio = 0.2
```

3. **Limpiar cache periódicamente**
```python
import asyncio

async def periodic_cleanup(engine):
    while True:
        await asyncio.sleep(3600)  # Cada hora
        engine.clear_cache()

# Ejecutar en background
asyncio.create_task(periodic_cleanup(engine))
```

### Memory Fragmentation

#### Síntomas
- Alta memoria usada pero poca disponible
- Allocations fallan

#### Soluciones

1. **Usar memory pool**
```python
config.enable_memory_pool = True
config.memory_pool_size = 1024 * 1024 * 1024  # 1GB pool
```

2. **Compactar cache**
```python
engine.compact_cache()
```

## 🎮 Problemas de GPU

### GPU Out of Memory

#### Síntomas
- `CUDA out of memory` error
- GPU memory al 100%

#### Diagnóstico

```python
import torch

# Verificar memoria GPU
if torch.cuda.is_available():
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"GPU Memory Max: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
```

#### Soluciones

1. **Reducir max_tokens**
```python
config.max_tokens = 2048  # Reducir
```

2. **Habilitar compresión**
```python
config.use_compression = True
config.compression_ratio = 0.2  # Compresión agresiva
```

3. **Usar CPU fallback**
```python
config.device = 'cpu'  # Fallback a CPU
# O usar mixed precision
config.dtype = torch.float16
```

4. **Limpiar cache GPU**
```python
import torch

torch.cuda.empty_cache()
torch.cuda.synchronize()
```

### GPU No Detectada

#### Síntomas
- `CUDA not available`
- Usa CPU en lugar de GPU

#### Diagnóstico

```bash
# Verificar CUDA
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
```

#### Soluciones

1. **Verificar instalación CUDA**
```bash
nvcc --version
```

2. **Reinstalar PyTorch con CUDA**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

3. **Verificar drivers**
```bash
nvidia-smi
# Actualizar drivers si es necesario
```

## 🗄️ Problemas de Cache

### Cache Hit Rate Bajo

#### Síntomas
- Hit rate <50%
- Muchos cache misses

#### Diagnóstico

```python
stats = engine.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Hits: {stats['cache_hits']}")
print(f"Misses: {stats['cache_misses']}")

# Analizar patrones
from bulk.core.ultra_adaptive_kv_cache_analytics import CacheAnalyzer

analyzer = CacheAnalyzer(engine)
patterns = analyzer.analyze_access_patterns()
print(f"Patterns: {patterns}")
```

#### Soluciones

1. **Aumentar tamaño de cache**
```python
config.max_tokens = 16384  # Aumentar
```

2. **Cambiar estrategia**
```python
config.cache_strategy = CacheStrategy.ADAPTIVE  # Mejor hit rate
```

3. **Verificar que requests sean similares**
```python
# Normalizar queries antes de cache
def normalize_query(query):
    return query.lower().strip()
```

4. **Habilitar prefetching**
```python
config.enable_prefetch = True
config.prefetch_size = 16
```

### Cache Corruption

#### Síntomas
- Resultados incorrectos
- Excepciones al recuperar

#### Diagnóstico

```python
# Verificar integridad
validation = engine.validate_cache_integrity()
if not validation['is_valid']:
    print(f"Errors: {validation['errors']}")
```

#### Soluciones

1. **Limpiar cache corrupto**
```python
engine.clear_cache()
```

2. **Verificar persistencia**
```python
# Si usas persistencia, verificar archivo
import os
if os.path.exists(config.persistence_path):
    # Verificar tamaño, checksums, etc.
    pass
```

3. **Restaurar desde backup**
```python
engine.restore_from_backup('/backup/cache.pt')
```

## 💿 Problemas de Persistencia

### Persistencia Falla

#### Síntomas
- Error al guardar cache
- Cache no se restaura

#### Diagnóstico

```python
# Intentar persistir
try:
    success = engine.persist()
    if not success:
        print("Persist failed")
except Exception as e:
    print(f"Error: {e}")

# Verificar permisos
import os
path = config.persistence_path
print(f"Path exists: {os.path.exists(path)}")
print(f"Writable: {os.access(path, os.W_OK)}")
```

#### Soluciones

1. **Verificar permisos**
```bash
chmod -R 755 /data/cache
chown -R user:user /data/cache
```

2. **Verificar espacio en disco**
```bash
df -h /data/cache
```

3. **Cambiar ruta**
```python
config.persistence_path = '/tmp/cache'  # Tmp como fallback
```

### Cache No Se Restaura

#### Síntomas
- Cache vacío después de reiniciar
- Archivo de persistencia existe pero no se carga

#### Soluciones

1. **Cargar explícitamente**
```python
engine.load()  # Cargar manualmente
```

2. **Verificar formato**
```python
import torch
try:
    data = torch.load(config.persistence_path)
    print(f"Loaded: {len(data)} entries")
except Exception as e:
    print(f"Error loading: {e}")
```

## 🌐 Problemas de Distribución

### Sincronización Multi-GPU

#### Síntomas
- Resultados inconsistentes entre GPUs
- Carga desbalanceada

#### Soluciones

1. **Sincronizar explícitamente**
```python
import torch

# Sincronizar después de operaciones
torch.cuda.synchronize()

# En distributed mode
if config.enable_distributed:
    import torch.distributed as dist
    dist.barrier()  # Sincronizar todos los procesos
```

2. **Balancear carga**
```python
from bulk.core.ultra_adaptive_kv_cache_engine import MultiGPULoadBalancer

balancer = MultiGPULoadBalancer(engine)
balancer.balance_load()

# Verificar balance
stats = balancer.get_gpu_stats()
print(f"GPU stats: {stats}")
```

## 🐛 Debugging Avanzado

### Habilitar Debug Logging

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('ultra_adaptive_kv_cache_engine')
logger.setLevel(logging.DEBUG)
```

### Profiling Detallado

```python
from torch.profiler import profile, record_function, ProfilerActivity

config.enable_profiling = True

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
) as prof:
    result = await engine.process_request({'text': 'test'})

# Exportar
prof.export_chrome_trace("trace.json")
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def process_with_profile():
    result = engine.process_request_sync({'text': 'test'})
    return result
```

### Tracing

```python
import sys
import traceback

try:
    result = await engine.process_request({'text': 'test'})
except Exception as e:
    traceback.print_exc()
    # O guardar en log
    with open('error.log', 'a') as f:
        traceback.print_exc(file=f)
```

---

**Más información:**
- [Troubleshooting Guide General](../TROUBLESHOOTING_GUIDE.md)
- [Performance Tuning](../PERFORMANCE_TUNING.md)
- [Best Practices](../BEST_PRACTICES.md)

