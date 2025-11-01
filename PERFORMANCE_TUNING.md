# ⚡ Guía de Tuning de Rendimiento

## 🎯 Objetivos de Rendimiento

### Métricas Objetivo
- **Latencia P50**: <100ms
- **Latencia P95**: <500ms
- **Latencia P99**: <1s
- **Throughput**: 100+ req/s
- **Cache Hit Rate**: >70%

## 🔧 Optimizaciones por Componente

### 1. KV Cache Engine

#### Alto Rendimiento
```python
config = KVCacheConfig(
    max_tokens=16384,
    cache_strategy=CacheStrategy.ADAPTIVE,
    use_compression=False,  # Sin compresión para velocidad
    dtype=torch.float16,   # FP16 más rápido
    enable_prefetch=True,
    prefetch_size=16,       # Prefetch agresivo
    num_workers=32         # Más workers
)
```

#### Eficiencia de Memoria
```python
config = KVCacheConfig(
    max_tokens=4096,
    use_compression=True,
    compression_ratio=0.2,
    use_quantization=True,
    quantization_bits=4,
    max_memory_mb=2048
)
```

### 2. Base de Datos

#### Connection Pooling
```python
from sqlalchemy import create_engine

engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

#### Índices
```sql
CREATE INDEX idx_session_id ON cache_table(session_id);
CREATE INDEX idx_created_at ON cache_table(created_at);
CREATE INDEX idx_business_area ON documents(business_area);
```

### 3. Redis Cache

```python
import redis

redis_client = redis.Redis(
    host='localhost',
    port=6379,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_keepalive=True
)

# Configurar TTL apropiado
redis_client.setex('key', 3600, 'value')
```

### 4. Async Operations

```python
# ✅ Correcto - Paralelo
tasks = [process_request(req) for req in requests]
results = await asyncio.gather(*tasks)

# ❌ Incorrecto - Secuencial
results = []
for req in requests:
    results.append(await process_request(req))
```

## 📊 Benchmarking

### Test de Throughput
```python
import asyncio
import time

async def benchmark_throughput(engine, num_requests=1000):
    start = time.time()
    
    tasks = [
        engine.process_request({'text': f'Request {i}'})
        for i in range(num_requests)
    ]
    
    results = await asyncio.gather(*tasks)
    
    duration = time.time() - start
    throughput = num_requests / duration
    
    print(f"Throughput: {throughput:.2f} req/s")
    return throughput
```

### Test de Latencia
```python
async def benchmark_latency(engine, num_requests=100):
    latencies = []
    
    for i in range(num_requests):
        start = time.time()
        await engine.process_request({'text': f'Request {i}'})
        latency = (time.time() - start) * 1000
        latencies.append(latency)
    
    import numpy as np
    print(f"P50: {np.percentile(latencies, 50):.2f}ms")
    print(f"P95: {np.percentile(latencies, 95):.2f}ms")
    print(f"P99: {np.percentile(latencies, 99):.2f}ms")
```

## 🎛️ Ajustes por Escenario

### Alto Tráfico
1. Aumentar workers: `num_workers=32`
2. Habilitar prefetching: `prefetch_size=16`
3. Batch processing: `batch_size=20`
4. Escalar horizontalmente

### Baja Latencia
1. Desactivar compresión
2. Usar FP16
3. Cache agresivo
4. Prefetching predictivo

### Memoria Limitada
1. Reducir `max_tokens`
2. Compresión agresiva
3. Quantization
4. GC más frecuente

## 📈 Monitoring de Performance

```python
from bulk.core.ultra_adaptive_kv_cache_analytics import PerformanceAnalyzer

analyzer = PerformanceAnalyzer(engine)

# Identificar bottlenecks
bottlenecks = analyzer.identify_bottlenecks()

# Obtener recomendaciones
recommendations = analyzer.get_optimization_recommendations()

# Generar reporte
report = analyzer.generate_performance_report()
```

---

**Más información:**
- [Guía de Uso Avanzado](bulk/ADVANCED_USAGE_GUIDE.md)
- [Mejores Prácticas](BEST_PRACTICES.md)
- [Troubleshooting](TROUBLESHOOTING_GUIDE.md)

