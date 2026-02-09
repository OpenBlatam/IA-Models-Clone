# 🔧 Troubleshooting Quick Reference - Blatam Academy Features

## 🚨 Problemas Comunes - Soluciones Rápidas

### Error: "CUDA out of memory"

```bash
# Solución rápida 1: Reducir max_tokens
export KV_CACHE_MAX_TOKENS=2048

# Solución rápida 2: Habilitar compresión
export KV_CACHE_USE_COMPRESSION=true
export KV_CACHE_COMPRESSION_RATIO=0.2

# Solución rápida 3: Limpiar cache GPU
python -c "import torch; torch.cuda.empty_cache()"
```

### Error: "Connection refused"

```bash
# Verificar servicios
docker-compose ps

# Reiniciar servicios
docker-compose restart

# Ver logs
docker-compose logs -f [service-name]
```

### Error: "Cache not working"

```python
# Verificar configuración
from bulk.core.ultra_adaptive_kv_cache_engine import UltraAdaptiveKVCacheEngine, KVCacheConfig

config = KVCacheConfig(max_tokens=4096)
engine = UltraAdaptiveKVCacheEngine(config)

# Verificar stats
stats = engine.get_stats()
print(f"Hit rate: {stats['hit_rate']}")

# Limpiar y reiniciar
engine.clear_cache()
```

### Error: "High latency"

```python
# Solución 1: Habilitar prefetching
config.enable_prefetch = True
config.prefetch_size = 16

# Solución 2: Cambiar estrategia
config.cache_strategy = CacheStrategy.ADAPTIVE

# Solución 3: Aumentar cache size
config.max_tokens = 16384
```

### Error: "Memory leak"

```python
# Forzar garbage collection
import gc
gc.collect()

# Reducir cache size
config.max_tokens = 2048
config.enable_gc = True
config.gc_threshold = 0.7
```

## 📋 Comandos de Diagnóstico Rápido

### Health Check Completo

```bash
# Script rápido
python -c "
from bulk.core.ultra_adaptive_kv_cache_engine import UltraAdaptiveKVCacheEngine, KVCacheConfig
import torch

config = KVCacheConfig(max_tokens=4096)
engine = UltraAdaptiveKVCacheEngine(config)

print('✅ Engine initialized')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB')

stats = engine.get_stats()
print(f'✅ Stats retrieved: {stats}')
"
```

### Verificar Configuración

```python
# Verificar config actual
from bulk.core.ultra_adaptive_kv_cache_engine import KVCacheConfig
import os

config = KVCacheConfig()
print(f"Max tokens: {config.max_tokens}")
print(f"Strategy: {config.cache_strategy}")
print(f"Compression: {config.use_compression}")
print(f"Persistence: {config.enable_persistence}")
```

### Ver Logs en Tiempo Real

```bash
# Docker logs
docker-compose logs -f bul

# KV Cache específico
docker-compose logs -f bul | grep "kv_cache"

# Errores solamente
docker-compose logs bul | grep -i error
```

## 🔍 Verificación Rápida de Problemas

### Checklist de Diagnóstico

```python
def quick_diagnostics():
    """Diagnóstico rápido del sistema."""
    issues = []
    
    # 1. Verificar CUDA
    import torch
    if not torch.cuda.is_available():
        issues.append("❌ CUDA not available")
    else:
        print("✅ CUDA available")
    
    # 2. Verificar memoria GPU
    if torch.cuda.is_available():
        memory = torch.cuda.memory_allocated() / 1024**3
        if memory > 8:
            issues.append(f"⚠️  High GPU memory: {memory:.2f} GB")
        else:
            print(f"✅ GPU memory OK: {memory:.2f} GB")
    
    # 3. Verificar cache engine
    try:
        from bulk.core.ultra_adaptive_kv_cache_engine import UltraAdaptiveKVCacheEngine, KVCacheConfig
        config = KVCacheConfig()
        engine = UltraAdaptiveKVCacheEngine(config)
        print("✅ Cache engine OK")
    except Exception as e:
        issues.append(f"❌ Cache engine error: {e}")
    
    # 4. Verificar configuración
    try:
        validation = engine.validate_configuration()
        if not validation['is_valid']:
            issues.append(f"❌ Config issues: {validation['issues']}")
        else:
            print("✅ Configuration valid")
    except:
        pass
    
    # 5. Verificar persistencia (si está habilitada)
    if config.enable_persistence:
        import os
        if not os.path.exists(config.persistence_path):
            issues.append(f"⚠️  Persistence path missing: {config.persistence_path}")
        else:
            print(f"✅ Persistence path OK: {config.persistence_path}")
    
    return issues

# Ejecutar
issues = quick_diagnostics()
if issues:
    print("\n⚠️  Issues found:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("\n✅ All checks passed!")
```

## 🚀 Soluciones Rápidas por Escenario

### Escenario 1: Sistema Lento

```python
# Solución rápida
config = KVCacheConfig(
    max_tokens=16384,  # Aumentar cache
    enable_prefetch=True,
    prefetch_size=16,
    cache_strategy=CacheStrategy.ADAPTIVE
)
```

### Escenario 2: Memoria Insuficiente

```python
# Solución rápida
config = KVCacheConfig(
    max_tokens=2048,  # Reducir cache
    use_compression=True,
    compression_ratio=0.2,  # Compresión agresiva
    use_quantization=True,
    quantization_bits=4
)
```

### Escenario 3: Cache Hit Rate Bajo

```python
# Solución rápida
config = KVCacheConfig(
    max_tokens=16384,  # Aumentar tamaño
    cache_strategy=CacheStrategy.ADAPTIVE,  # Mejor estrategia
    enable_prefetch=True
)
```

### Escenario 4: Errores de GPU

```python
# Fallback a CPU
config = KVCacheConfig(
    max_tokens=4096,
    device='cpu'  # O usar dtype más ligero
)

# O reducir tamaño
config.max_tokens = 1024
```

## 📞 Comandos Útiles

### Limpiar Todo

```bash
# Limpiar cache
python -c "from bulk.core.ultra_adaptive_kv_cache_engine import UltraAdaptiveKVCacheEngine, KVCacheConfig; engine = UltraAdaptiveKVCacheEngine(KVCacheConfig()); engine.clear_cache(); print('Cache cleared')"

# Limpiar Docker
docker-compose down -v

# Limpiar logs
docker-compose logs --tail=0 -f
```

### Reiniciar Sistema

```bash
# Reinicio completo
docker-compose down
docker-compose up -d

# Reinicio específico
docker-compose restart bul

# Con rebuild
docker-compose up -d --build bul
```

### Ver Estadísticas

```bash
# Stats desde CLI
python bulk/core/ultra_adaptive_kv_cache_cli.py stats

# Health check
python bulk/core/ultra_adaptive_kv_cache_cli.py health

# Monitor en tiempo real
python bulk/core/ultra_adaptive_kv_cache_cli.py monitor
```

## 🎯 Presets de Configuración Rápida

### Preset: Desarrollo

```python
config = KVCacheConfig(
    max_tokens=2048,
    enable_profiling=True,
    enable_persistence=False
)
```

### Preset: Producción

```python
config = KVCacheConfig(
    max_tokens=8192,
    cache_strategy=CacheStrategy.ADAPTIVE,
    enable_persistence=True,
    enable_prefetch=True,
    prefetch_size=16,
    use_compression=True
)
```

### Preset: Alto Rendimiento

```python
config = KVCacheConfig(
    max_tokens=16384,
    cache_strategy=CacheStrategy.ADAPTIVE,
    enable_prefetch=True,
    prefetch_size=32,
    use_compression=False  # Máxima velocidad
)
```

### Preset: Eficiencia de Memoria

```python
config = KVCacheConfig(
    max_tokens=4096,
    use_compression=True,
    compression_ratio=0.2,
    use_quantization=True,
    quantization_bits=4,
    enable_gc=True,
    gc_threshold=0.6
)
```

## 🔗 Referencias Rápidas

- **Troubleshooting Completo**: [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)
- **Troubleshooting Avanzado**: [bulk/ADVANCED_TROUBLESHOOTING.md](bulk/ADVANCED_TROUBLESHOOTING.md)
- **Performance Tuning**: [PERFORMANCE_TUNING.md](PERFORMANCE_TUNING.md)
- **FAQ**: [FAQ.md](FAQ.md)

---

**Última actualización**: Documentación completa de troubleshooting rápido



