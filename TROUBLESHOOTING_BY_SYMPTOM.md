# 🔍 Troubleshooting por Síntomas - Blatam Academy Features

## 🚨 Síntomas Comunes y Soluciones

### "El sistema está muy lento"

#### Diagnóstico
```python
# Verificar latencia
stats = engine.get_stats()
print(f"P95 latency: {stats['p95_latency']}ms")
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
```

#### Soluciones Rápidas
1. **Aumentar cache size**
```python
config.max_tokens = 16384  # Aumentar de 4096
```

2. **Habilitar prefetching**
```python
config.enable_prefetch = True
config.prefetch_size = 16
```

3. **Cambiar estrategia**
```python
config.cache_strategy = CacheStrategy.ADAPTIVE
```

4. **Verificar carga del sistema**
```bash
docker stats
top
htop
```

### "El sistema consume mucha memoria"

#### Diagnóstico
```python
import psutil
import os

process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / 1024 / 1024
print(f"Memory: {memory_mb:.2f} MB")

if torch.cuda.is_available():
    gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"GPU Memory: {gpu_memory:.2f} MB")
```

#### Soluciones Rápidas
1. **Reducir cache size**
```python
config.max_tokens = 2048  # Reducir
```

2. **Habilitar compresión**
```python
config.use_compression = True
config.compression_ratio = 0.2  # Agresiva
```

3. **Habilitar cuantización**
```python
config.use_quantization = True
config.quantization_bits = 4
```

4. **Limpiar cache**
```python
engine.clear_cache()
gc.collect()
```

### "Muchos errores de cache miss"

#### Diagnóstico
```python
stats = engine.get_stats()
hit_rate = stats['hit_rate']
print(f"Hit rate: {hit_rate:.2%}")

if hit_rate < 0.5:
    print("⚠️  Low hit rate!")
```

#### Soluciones Rápidas
1. **Aumentar tamaño de cache**
```python
config.max_tokens = 16384
```

2. **Verificar que queries sean similares**
```python
# Normalizar queries antes de cachear
def normalize_query(query):
    return query.lower().strip()
```

3. **Cambiar a estrategia Adaptive**
```python
config.cache_strategy = CacheStrategy.ADAPTIVE
```

4. **Habilitar prefetching**
```python
config.enable_prefetch = True
```

### "Error: CUDA out of memory"

#### Diagnóstico
```python
import torch

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved: {reserved:.2f} GB")
```

#### Soluciones Rápidas
1. **Reducir max_tokens**
```python
config.max_tokens = 1024  # Reducir significativamente
```

2. **Habilitar compresión agresiva**
```python
config.use_compression = True
config.compression_ratio = 0.15  # Muy agresiva
```

3. **Limpiar GPU memory**
```python
torch.cuda.empty_cache()
torch.cuda.synchronize()
```

4. **Usar CPU fallback**
```python
config.device = 'cpu'  # Temporalmente
```

### "El sistema no responde / Timeouts"

#### Diagnóstico
```bash
# Verificar servicios
docker-compose ps

# Ver logs de errores
docker-compose logs bul | grep -i error

# Verificar recursos
docker stats --no-stream
```

#### Soluciones Rápidas
1. **Reiniciar servicio**
```bash
docker-compose restart bul
```

2. **Verificar base de datos**
```bash
docker-compose exec postgres psql -U postgres -c "SELECT 1"
```

3. **Verificar Redis**
```bash
docker-compose exec redis redis-cli ping
```

4. **Limpiar y reiniciar**
```bash
docker-compose down
docker-compose up -d
```

### "Los resultados del cache son incorrectos"

#### Diagnóstico
```python
# Verificar integridad del cache
validation = engine.validate_cache_integrity()
if not validation['is_valid']:
    print(f"Errors: {validation['errors']}")
```

#### Soluciones Rápidas
1. **Limpiar cache**
```python
engine.clear_cache()
```

2. **Verificar persistencia**
```python
# Si usas persistencia, verificar archivo
import os
if os.path.exists(config.persistence_path):
    # Verificar tamaño y checksum
    pass
```

3. **Restaurar desde backup**
```python
engine.restore_from_backup('/backup/cache.pt')
```

### "Cache no persiste entre reinicios"

#### Diagnóstico
```python
# Verificar configuración
print(f"Persistence enabled: {config.enable_persistence}")
print(f"Persistence path: {config.persistence_path}")

# Verificar permisos
import os
if config.persistence_path:
    print(f"Path exists: {os.path.exists(config.persistence_path)}")
    print(f"Writable: {os.access(config.persistence_path, os.W_OK)}")
```

#### Soluciones Rápidas
1. **Habilitar persistencia**
```python
config.enable_persistence = True
config.persistence_path = '/data/cache'
```

2. **Verificar permisos**
```bash
chmod 755 /data/cache
chown user:user /data/cache
```

3. **Persistir manualmente**
```python
engine.persist()
```

4. **Cargar al iniciar**
```python
engine.load()
```

### "Bajo throughput / Pocas requests por segundo"

#### Diagnóstico
```python
# Medir throughput
import time
import asyncio

async def measure_throughput():
    start = time.time()
    requests = [{'text': f'Query {i}', 'priority': 1} for i in range(100)]
    await asyncio.gather(*[engine.process_request(r) for r in requests])
    duration = time.time() - start
    throughput = 100 / duration
    print(f"Throughput: {throughput:.2f} req/s")
```

#### Soluciones Rápidas
1. **Usar batch processing**
```python
results = await engine.process_batch_optimized(requests, batch_size=20)
```

2. **Aumentar workers**
```python
config.num_workers = 32
```

3. **Habilitar prefetching agresivo**
```python
config.enable_prefetch = True
config.prefetch_size = 32
```

4. **Verificar bottlenecks**
```python
from bulk.core.ultra_adaptive_kv_cache_analytics import PerformanceAnalyzer
analyzer = PerformanceAnalyzer(engine)
bottlenecks = analyzer.identify_bottlenecks()
```

### "Errores de conexión a base de datos"

#### Diagnóstico
```bash
# Verificar conexión
docker-compose exec postgres psql -U postgres -c "SELECT 1"

# Verificar variables de entorno
echo $DATABASE_URL

# Verificar logs
docker-compose logs postgres
```

#### Soluciones Rápidas
1. **Verificar DATABASE_URL**
```bash
export DATABASE_URL=postgresql://user:password@postgres:5432/dbname
```

2. **Verificar servicio está corriendo**
```bash
docker-compose ps postgres
```

3. **Reiniciar servicio**
```bash
docker-compose restart postgres
```

4. **Verificar pool de conexiones**
```python
# Aumentar pool size si hay muchos timeouts
engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=40)
```

### "Logs muestran muchos warnings"

#### Diagnóstico
```bash
# Ver warnings en logs
docker-compose logs bul | grep -i warning | tail -20
```

#### Soluciones Rápidas
1. **Ajustar nivel de log**
```python
import logging
logging.getLogger('bulk').setLevel(logging.ERROR)
```

2. **Verificar configuración**
```python
# Muchos warnings pueden indicar config incorrecta
validation = engine.validate_configuration()
```

3. **Actualizar dependencias**
```bash
pip install --upgrade -r requirements.txt
```

## 🔄 Flujo de Diagnóstico General

```
Problema detectado
    │
    ▼
┌───────────────────┐
│ Verificar logs    │
└─────────┬─────────┘
          │
    ┌─────┴─────┐
    │           │
  Error      Warning
    │           │
    ▼           ▼
┌─────────┐ ┌──────────┐
│ Check   │ │ Check    │
│ config  │ │ metrics  │
└────┬────┘ └────┬─────┘
     │           │
     └─────┬─────┘
           │
           ▼
    ┌──────────────┐
    │ Aplicar fix  │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Verificar    │
    └──────┬───────┘
           │
      ┌────┴────┐
      │         │
    Fixed    Still broken
      │         │
      │         ▼
      │    ┌──────────────┐
      │    │ Escalar o    │
      │    │ buscar ayuda  │
      │    └───────────────┘
      │
      ▼
   Resuelto ✅
```

## 📞 Cuando Necesitas Más Ayuda

Si los problemas persisten:

1. **Revisar documentación completa**
   - [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)
   - [ADVANCED_TROUBLESHOOTING.md](bulk/ADVANCED_TROUBLESHOOTING.md)
   - [FAQ.md](FAQ.md)

2. **Recopilar información**
   - Logs completos
   - Configuración actual
   - Estadísticas del sistema
   - Pasos para reproducir

3. **Buscar en issues**
   - GitHub Issues
   - Documentación
   - FAQ

4. **Contactar soporte**
   - Con toda la información recopilada

---

**Más información:**
- [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)
- [Troubleshooting Quick Reference](TROUBLESHOOTING_QUICK_REFERENCE.md)
- [Advanced Troubleshooting](bulk/ADVANCED_TROUBLESHOOTING.md)

