# 🔧 Guía Completa de Troubleshooting

## 📋 Índice

1. [Problemas Comunes](#problemas-comunes)
2. [Problemas del KV Cache](#kv-cache)
3. [Problemas de Rendimiento](#rendimiento)
4. [Problemas de Memoria](#memoria)
5. [Problemas de Red](#red)
6. [Problemas de Seguridad](#seguridad)
7. [Diagnóstico Avanzado](#diagnóstico)

## 🚨 Problemas Comunes

### Servicio No Inicia

**Síntomas:**
- El servicio no responde
- Error en logs de Docker
- Puerto no disponible

**Diagnóstico:**
```bash
# Ver logs del servicio
docker-compose logs [service-name]

# Verificar si el puerto está en uso
netstat -an | grep 8000

# Verificar recursos del sistema
docker stats
```

**Soluciones:**

1. **Puerto ocupado:**
```bash
# Cambiar puerto en docker-compose.yml
ports:
  - "8001:8000"  # Cambiar 8000 a 8001
```

2. **Memoria insuficiente:**
```bash
# Aumentar límite de memoria en docker-compose.yml
deploy:
  resources:
    limits:
      memory: 4G
```

3. **Dependencias faltantes:**
```bash
# Reconstruir imagen
docker-compose build --no-cache [service-name]
docker-compose up -d [service-name]
```

### Error de Conexión a Base de Datos

**Síntomas:**
- `ConnectionError` o `TimeoutError`
- No se pueden realizar queries
- Errores de autenticación

**Diagnóstico:**
```bash
# Verificar conexión PostgreSQL
docker-compose exec postgres psql -U postgres -c "SELECT 1;"

# Verificar variables de entorno
docker-compose exec postgres env | grep DATABASE

# Verificar logs
docker-compose logs postgres
```

**Soluciones:**

1. **Credenciales incorrectas:**
```bash
# Verificar .env file
cat .env | grep DATABASE_URL

# Actualizar si es necesario
DATABASE_URL=postgresql://postgres:password@postgres:5432/blatam_academy
```

2. **Base de datos no existe:**
```bash
# Crear base de datos
docker-compose exec postgres psql -U postgres -c "CREATE DATABASE blatam_academy;"
```

3. **Connection pool agotado:**
```python
# Aumentar pool size en código
from sqlalchemy import create_engine

engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True
)
```

## ⚡ Problemas del KV Cache

### Alto Uso de Memoria

**Síntomas:**
- `OutOfMemoryError`
- Sistema lento
- OOM kills

**Diagnóstico:**
```python
# Verificar uso de memoria
stats = engine.get_stats()
print(f"Memory usage: {stats['memory_usage']}%")
print(f"Allocated: {stats['allocated_mb']} MB")
```

**Soluciones:**

1. **Reducir tamaño de caché:**
```python
from bulk.core.ultra_adaptive_kv_cache_config_manager import ConfigManager

config_manager = ConfigManager(engine)
await config_manager.update_config('cache_size', 4096)  # Reducir de 16384
```

2. **Usar preset memory_efficient:**
```python
from bulk.core.ultra_adaptive_kv_cache_config_manager import ConfigPreset

ConfigPreset.apply_preset(engine, 'memory_efficient')
```

3. **Limpiar caché:**
```bash
python bulk/core/ultra_adaptive_kv_cache_cli.py clear-cache
```

4. **Habilitar compresión:**
```python
await config_manager.update_config('use_compression', True)
await config_manager.update_config('compression_ratio', 0.2)
```

### Bajo Cache Hit Rate

**Síntomas:**
- Hit rate < 50%
- Alto número de cache misses
- Latencia alta

**Diagnóstico:**
```python
stats = engine.get_stats()
print(f"Hit rate: {stats['hit_rate']}")
print(f"Miss rate: {stats['miss_rate']}")
```

**Soluciones:**

1. **Aumentar tamaño de caché:**
```python
await config_manager.update_config('cache_size', 32768)  # Aumentar
```

2. **Mejorar reutilización de sesiones:**
```python
# Usar session_id consistente
session_id = f"user_{user_id}_{document_type}"  # Reutilizar
```

3. **Habilitar prefetching:**
```python
await config_manager.update_config('enable_prefetch', True)
await config_manager.update_config('prefetch_size', 8)
```

4. **Usar estrategia Adaptive:**
```python
from bulk.core.ultra_adaptive_kv_cache_engine import CacheStrategy

config.cache_strategy = CacheStrategy.ADAPTIVE
```

### Latencia Alta

**Síntomas:**
- P95 latency > 1s
- Timeouts frecuentes
- Usuarios reportan lentitud

**Diagnóstico:**
```python
from bulk.core.ultra_adaptive_kv_cache_analytics import PerformanceAnalyzer

analyzer = PerformanceAnalyzer(engine)
bottlenecks = analyzer.identify_bottlenecks()
print(bottlenecks)
```

**Soluciones:**

1. **Desactivar compresión:**
```python
await config_manager.update_config('use_compression', False)
```

2. **Aumentar workers:**
```python
await config_manager.update_config('num_workers', 16)
```

3. **Usar FP16:**
```python
import torch
config.dtype = torch.float16
```

4. **Habilitar prefetching:**
```python
await config_manager.update_config('enable_prefetch', True)
await config_manager.update_config('prefetch_size', 16)
```

### Error de GPU

**Síntomas:**
- `CUDA out of memory`
- GPU no disponible
- Fallback a CPU

**Diagnóstico:**
```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
```

**Soluciones:**

1. **Limpiar cache de CUDA:**
```python
torch.cuda.empty_cache()
engine.clear()
```

2. **Usar CPU si GPU no disponible:**
```python
config.use_cuda = False
engine = UltraAdaptiveKVCacheEngine(config)
```

3. **Reducir batch size:**
```python
await config_manager.update_config('batch_size', 4)  # Reducir
```

4. **Usar mixed precision:**
```python
from torch.cuda.amp import autocast

with autocast():
    result = await engine.process_kv(key, value)
```

## 📈 Problemas de Rendimiento

### Throughput Bajo

**Síntomas:**
- < 10 req/s
- Cola de requests creciente
- Tiempos de respuesta largos

**Soluciones:**

1. **Aumentar workers:**
```python
await config_manager.update_config('num_workers', 32)
```

2. **Habilitar batch processing:**
```python
results = await engine.process_batch_optimized(
    requests,
    batch_size=20,
    deduplicate=True
)
```

3. **Escalar horizontalmente:**
```bash
docker-compose up -d --scale bul=4
```

4. **Optimizar base de datos:**
```sql
-- Crear índices
CREATE INDEX idx_session_id ON cache_table(session_id);
CREATE INDEX idx_timestamp ON cache_table(created_at);
```

### High CPU Usage

**Síntomas:**
- CPU > 90%
- Sistema lento
- Otros procesos afectados

**Soluciones:**

1. **Reducir workers:**
```python
await config_manager.update_config('num_workers', 4)
```

2. **Usar GPU:**
```python
# Mover procesamiento a GPU
config.use_cuda = True
```

3. **Optimizar código:**
```python
# Usar async apropiadamente
# Evitar bloqueos
# Usar connection pooling
```

## 💾 Problemas de Memoria

### Memory Leaks

**Síntomas:**
- Memoria crece continuamente
- Sistema eventualmente crashea
- OOM kills

**Diagnóstico:**
```python
import tracemalloc

tracemalloc.start()
# ... código ...
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:10]:
    print(stat)
```

**Soluciones:**

1. **Habilitar GC agresivo:**
```python
config.enable_gc = True
config.gc_threshold = 0.6  # Más agresivo
```

2. **Limpiar referencias circulares:**
```python
import gc
gc.collect()
```

3. **Limitar historial:**
```python
# Limitar performance_history
if len(self._performance_history) > 500:
    self._performance_history = self._performance_history[-500:]
```

### Swap Usage Alto

**Síntomas:**
- Sistema muy lento
- Swap usage > 50%
- I/O wait alto

**Soluciones:**

1. **Reducir memoria:**
```python
config.max_memory_mb = 2048  # Reducir límite
```

2. **Desactivar swap:**
```bash
sudo swapoff -a
```

3. **Aumentar RAM:**
- Agregar más RAM física
- O usar instancia con más RAM

## 🌐 Problemas de Red

### Timeouts

**Síntomas:**
- `TimeoutError`
- Requests fallan
- Conexiones perdidas

**Soluciones:**

1. **Aumentar timeout:**
```python
import httpx

client = httpx.AsyncClient(timeout=60.0)  # Aumentar timeout
```

2. **Implementar retry:**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def process_with_retry(request):
    return await engine.process_request(request)
```

3. **Verificar red:**
```bash
# Verificar conectividad
ping database_host
telnet database_host 5432
```

### Alta Latencia de Red

**Síntomas:**
- Requests lentos
- Timeouts frecuentes
- Alta latencia de red

**Soluciones:**

1. **Usar edge caching:**
```python
# Cachear en edge nodes
await engine.sync_to_edge(key, value, target_nodes=['edge-1'])
```

2. **CDN para contenido estático:**
```nginx
# nginx.conf
location /static {
    proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=static:10m;
    proxy_cache static;
}
```

3. **Comprimir respuestas:**
```python
# Usar compresión HTTP
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

## 🔒 Problemas de Seguridad

### Rate Limiting Muy Estricto

**Síntomas:**
- Requests bloqueados
- `429 Too Many Requests`
- Usuarios legítimos bloqueados

**Soluciones:**

1. **Ajustar límites:**
```python
from bulk.core.ultra_adaptive_kv_cache_security import SecureEngineWrapper

secure_engine = SecureEngineWrapper(
    engine,
    rate_limit_per_minute=120  # Aumentar límite
)
```

2. **Whitelist IPs:**
```python
secure_engine.add_ip_to_whitelist(['192.168.1.0/24'])
```

3. **Usar API keys:**
```python
# Validar API keys en lugar de solo IP
secure_engine.enable_api_key_validation(True)
```

### Falsos Positivos en Sanitización

**Síntomas:**
- Requests válidos bloqueados
- Contenido legítimo filtrado

**Soluciones:**

1. **Ajustar reglas:**
```python
from bulk.core.ultra_adaptive_kv_cache_security import WAFRules

waf = WAFRules(
    block_sql_injection=True,
    block_xss=True,
    strict_mode=False  # Modo menos estricto
)
```

2. **Whitelist patterns:**
```python
waf.add_pattern_whitelist(['SELECT.*FROM.*WHERE'])  # Permitir patrón
```

## 🔍 Diagnóstico Avanzado

### Profiling Completo

```python
import cProfile
import pstats
from io import StringIO

profiler = cProfile.Profile()
profiler.enable()

# Tu código aquí
await process_requests()

profiler.disable()

# Analizar resultados
s = StringIO()
stats = pstats.Stats(profiler, stream=s)
stats.sort_stats('cumulative')
stats.print_stats(30)

print(s.getvalue())
```

### Monitoring Detallado

```python
from bulk.core.ultra_adaptive_kv_cache_monitor import PerformanceMonitor
from bulk.core.ultra_adaptive_kv_cache_analytics import PerformanceAnalyzer

# Monitoring completo
monitor = PerformanceMonitor(engine, check_interval=1.0)
await monitor.start_monitoring()

# Analytics detallado
analyzer = PerformanceAnalyzer(engine)
report = analyzer.generate_performance_report()
bottlenecks = analyzer.identify_bottlenecks()
recommendations = analyzer.get_optimization_recommendations()

print(f"Report: {report}")
print(f"Bottlenecks: {bottlenecks}")
print(f"Recommendations: {recommendations}")
```

### Health Check Completo

```python
from bulk.core.ultra_adaptive_kv_cache_health_checker import HealthChecker

health_checker = HealthChecker(engine)
status = await health_checker.check_health()

print(f"Overall: {status['overall']}")
print(f"Components: {status['components']}")
print(f"Metrics: {status['metrics']}")
print(f"Recommendations: {status['recommendations']}")
```

### Debugging Mode

```python
# Habilitar debug logging
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('bulk.core')
logger.setLevel(logging.DEBUG)

# Habilitar profiling
config.enable_profiling = True

# Habilitar verbose output
engine.set_verbose(True)
```

## 📞 Obtener Ayuda

Si los problemas persisten:

1. **Recopilar información:**
   ```bash
   # Logs del sistema
   docker-compose logs > system_logs.txt
   
   # Métricas
   python bulk/core/ultra_adaptive_kv_cache_cli.py stats > stats.json
   
   # Health check
   python bulk/core/ultra_adaptive_kv_cache_cli.py health > health.json
   ```

2. **Documentar el problema:**
   - Descripción del problema
   - Pasos para reproducir
   - Logs relevantes
   - Configuración actual

3. **Consultar documentación:**
   - [README Principal](README.md)
   - [Guía de Uso Avanzado](bulk/ADVANCED_USAGE_GUIDE.md)
   - [Mejores Prácticas](BEST_PRACTICES.md)

---

**Para más información:**
- [Documentación KV Cache](bulk/core/README_ULTRA_ADAPTIVE_KV_CACHE.md)
- [Guía de Arquitectura](ARCHITECTURE_GUIDE.md)
- [Índice de Documentación](DOCUMENTATION_INDEX.md)

