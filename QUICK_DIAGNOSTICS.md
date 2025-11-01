# 🔍 Diagnóstico Rápido - Blatam Academy Features

## 🚀 Scripts de Diagnóstico Rápido

### Health Check Completo

```python
# quick_health_check.py
import asyncio
from bulk.core.ultra_adaptive_kv_cache_engine import (
    UltraAdaptiveKVCacheEngine,
    KVCacheConfig
)

async def quick_health_check():
    """Health check completo en 30 segundos."""
    issues = []
    warnings = []
    
    print("🔍 Quick Health Check")
    print("=" * 50)
    
    # 1. Verificar CUDA
    import torch
    if torch.cuda.is_available():
        print("✅ CUDA available")
        memory = torch.cuda.memory_allocated() / 1024**3
        if memory > 8:
            warnings.append(f"⚠️  High GPU memory: {memory:.2f} GB")
        else:
            print(f"✅ GPU memory OK: {memory:.2f} GB")
    else:
        print("⚠️  CUDA not available (using CPU)")
    
    # 2. Verificar KV Cache Engine
    try:
        config = KVCacheConfig(max_tokens=4096)
        engine = UltraAdaptiveKVCacheEngine(config)
        print("✅ KV Cache engine OK")
        
        # Validar configuración
        validation = engine.validate_configuration()
        if validation['is_valid']:
            print("✅ Configuration valid")
        else:
            issues.append(f"❌ Config issues: {validation['issues']}")
    except Exception as e:
        issues.append(f"❌ KV Cache error: {e}")
    
    # 3. Verificar persistencia (si está habilitada)
    if config.enable_persistence:
        import os
        if os.path.exists(config.persistence_path):
            print(f"✅ Persistence path OK: {config.persistence_path}")
        else:
            warnings.append(f"⚠️  Persistence path missing: {config.persistence_path}")
    
    # Resumen
    print("\n" + "=" * 50)
    if issues:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    if warnings:
        print("⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  {warning}")
    if not issues and not warnings:
        print("✅ All checks passed!")
    
    return len(issues) == 0

asyncio.run(quick_health_check())
```

### Performance Snapshot

```python
# performance_snapshot.py
import asyncio
import time
from bulk.core.ultra_adaptive_kv_cache_engine import UltraAdaptiveKVCacheEngine

async def performance_snapshot(engine):
    """Snapshot rápido de performance."""
    print("📊 Performance Snapshot")
    print("=" * 50)
    
    # Latencia rápida (10 requests)
    latencies = []
    for i in range(10):
        start = time.time()
        await engine.process_request({'text': f'Test {i}', 'priority': 1})
        latencies.append((time.time() - start) * 1000)
    
    import numpy as np
    print(f"Latency P50: {np.percentile(latencies, 50):.2f}ms")
    print(f"Latency P95: {np.percentile(latencies, 95):.2f}ms")
    
    # Stats del cache
    stats = engine.get_stats()
    print(f"Cache hit rate: {stats['hit_rate']:.2%}")
    print(f"Total requests: {stats['total_requests']}")
    
    # Memoria
    import psutil
    import os
    memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory:.2f} MB")
    
    # Evaluación rápida
    print("\n📈 Quick Assessment:")
    if stats['hit_rate'] < 0.5:
        print("⚠️  Low cache hit rate - consider increasing cache size")
    if np.percentile(latencies, 95) > 500:
        print("⚠️  High latency - check performance tuning guide")
    if memory > 4000:
        print("⚠️  High memory usage - consider compression")
    
    if (stats['hit_rate'] >= 0.5 and 
        np.percentile(latencies, 95) < 500 and 
        memory < 4000):
        print("✅ Performance looks good!")

# Uso
# engine = UltraAdaptiveKVCacheEngine(config)
# asyncio.run(performance_snapshot(engine))
```

### Configuration Validator

```python
# validate_config.py
from bulk.core.ultra_adaptive_kv_cache_engine import KVCacheConfig

def validate_config_quick(config: KVCacheConfig):
    """Validar configuración rápidamente."""
    issues = []
    
    # Validaciones básicas
    if config.max_tokens <= 0:
        issues.append("max_tokens must be > 0")
    
    if config.max_tokens > 65536:
        issues.append("max_tokens very large, may cause memory issues")
    
    if config.compression_ratio < 0 or config.compression_ratio > 1:
        issues.append("compression_ratio must be between 0 and 1")
    
    elif config.compression_ratio < 0.1:
        issues.append("compression_ratio very low, may affect quality")
    
    if config.prefetch_size > 64:
        issues.append("prefetch_size very large, may use too much memory")
    
    if config.enable_persistence and not config.persistence_path:
        issues.append("persistence enabled but no path specified")
    
    if issues:
        print("❌ Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Configuration looks good!")
        return True

# Uso
config = KVCacheConfig(max_tokens=8192)
validate_config_quick(config)
```

### Resource Usage Check

```python
# resource_check.py
import psutil
import os
import torch

def resource_check():
    """Check rápido de recursos del sistema."""
    print("💻 Resource Check")
    print("=" * 50)
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    print(f"CPU Usage: {cpu_percent}% ({cpu_count} cores)")
    
    # RAM
    memory = psutil.virtual_memory()
    print(f"RAM: {memory.percent}% used ({memory.used / 1024**3:.2f} GB / {memory.total / 1024**3:.2f} GB)")
    
    # GPU (si disponible)
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU: {gpu_count} device(s) available")
        
        for i in range(gpu_count):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    else:
        print("GPU: Not available")
    
    # Disk
    disk = psutil.disk_usage('/')
    print(f"Disk: {disk.percent}% used ({disk.used / 1024**3:.2f} GB / {disk.total / 1024**3:.2f} GB)")
    
    # Evaluación
    print("\n📊 Assessment:")
    if cpu_percent > 80:
        print("⚠️  High CPU usage")
    if memory.percent > 80:
        print("⚠️  High RAM usage")
    if disk.percent > 90:
        print("⚠️  Low disk space")
    
    if cpu_percent < 80 and memory.percent < 80 and disk.percent < 90:
        print("✅ Resources look healthy")

resource_check()
```

## 🎯 Diagnóstico por Categoría

### Diagnóstico de Cache

```bash
# Script rápido de diagnóstico de cache
python -c "
from bulk.core.ultra_adaptive_kv_cache_engine import UltraAdaptiveKVCacheEngine, KVCacheConfig
import asyncio

async def cache_diagnostic():
    config = KVCacheConfig(max_tokens=4096)
    engine = UltraAdaptiveKVCacheEngine(config)
    
    # Procesar algunos requests
    for i in range(20):
        await engine.process_request({'text': f'Query {i % 5}', 'priority': 1})
    
    stats = engine.get_stats()
    print(f'Cache Stats:')
    print(f'  Hit rate: {stats[\"hit_rate\"]:.2%}')
    print(f'  Hits: {stats[\"cache_hits\"]}')
    print(f'  Misses: {stats[\"cache_misses\"]}')
    print(f'  Avg latency: {stats[\"avg_latency\"]:.2f}ms')
    
    if stats['hit_rate'] < 0.5:
        print('⚠️  Low hit rate - consider increasing cache size')
    else:
        print('✅ Cache performance OK')

asyncio.run(cache_diagnostic())
"
```

### Diagnóstico de Servicios

```bash
# Verificar todos los servicios
echo "Checking services..."
docker-compose ps

echo "Checking health endpoints..."
curl -s http://localhost:8000/health | jq .
curl -s http://localhost:8002/api/stats | jq .

echo "Checking database..."
docker-compose exec postgres psql -U postgres -c "SELECT 1" || echo "❌ DB connection failed"

echo "Checking Redis..."
docker-compose exec redis redis-cli ping || echo "❌ Redis connection failed"
```

## 📋 Checklist de Diagnóstico Rápido

### Health Check (< 1 minuto)

- [ ] CUDA disponible (si se necesita GPU)
- [ ] KV Cache engine inicializa
- [ ] Configuración válida
- [ ] Persistencia OK (si habilitada)
- [ ] Servicios Docker corriendo

### Performance Check (< 2 minutos)

- [ ] Latencia P95 < 500ms
- [ ] Cache hit rate > 50%
- [ ] Memoria < límites
- [ ] Throughput aceptable

### Resource Check (< 30 segundos)

- [ ] CPU < 80%
- [ ] RAM < 80%
- [ ] Disk space > 10%
- [ ] GPU memory OK (si aplica)

---

**Más información:**
- [Troubleshooting by Symptom](TROUBLESHOOTING_BY_SYMPTOM.md)
- [Error Codes Reference](ERROR_CODES_REFERENCE.md)
- [Troubleshooting Quick Reference](TROUBLESHOOTING_QUICK_REFERENCE.md)

