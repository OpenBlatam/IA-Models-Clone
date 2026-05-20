# ⚠️ Anti-Patrones - Qué NO Hacer - Blatam Academy Features

## 📋 Tabla de Contenidos

- [Anti-Patrones de Cache](#anti-patrones-de-cache)
- [Anti-Patrones de Configuración](#anti-patrones-de-configuración)
- [Anti-Patrones de Performance](#anti-patrones-de-performance)
- [Anti-Patrones de Seguridad](#anti-patrones-de-seguridad)
- [Anti-Patrones de Código](#anti-patrones-de-código)

## 🗄️ Anti-Patrones de Cache

### ❌ Anti-Patrón 1: Cache Siempre Habilitado Sin Validación

```python
# ❌ MALO
def process_request(request):
    # Siempre usar cache sin verificar si es apropiado
    result = cache_engine.process_request(request)
    return result

# ✅ BUENO
async def process_request(request):
    # Verificar si el request es cacheable
    if should_cache(request):
        result = await cache_engine.process_request(request)
    else:
        result = await process_directly(request)
    return result
```

### ❌ Anti-Patrón 2: Ignorar TTL y Expiración

```python
# ❌ MALO
# Cache nunca expira, puede crecer indefinidamente
config = KVCacheConfig(max_tokens=999999999)

# ✅ BUENO
config = KVCacheConfig(
    max_tokens=8192,
    enable_gc=True,
    gc_threshold=0.8
)
```

### ❌ Anti-Patrón 3: Cache Keys No Únicos

```python
# ❌ MALO
# Keys pueden colisionar
cache_key = query[:10]  # Muy corto, puede colisionar

# ✅ BUENO
import hashlib
cache_key = hashlib.sha256(query.encode()).hexdigest()
```

### ❌ Anti-Patrón 4: Cache Sin Manejo de Errores

```python
# ❌ MALO
result = await cache_engine.process_request(request)
# Si cache falla, todo falla

# ✅ BUENO
try:
    result = await cache_engine.process_request(request)
except CacheError:
    # Fallback a procesamiento directo
    result = await process_directly(request)
```

## ⚙️ Anti-Patrones de Configuración

### ❌ Anti-Patrón 5: Configuración Hardcodeada

```python
# ❌ MALO
config = KVCacheConfig(max_tokens=4096)  # Hardcodeado

# ✅ BUENO
import os
config = KVCacheConfig(
    max_tokens=int(os.getenv('KV_CACHE_MAX_TOKENS', '4096'))
)
```

### ❌ Anti-Patrón 6: Usar Misma Config en Dev y Prod

```python
# ❌ MALO
# Misma configuración para todo
config = KVCacheConfig(max_tokens=4096)

# ✅ BUENO
import os
env = os.getenv('ENVIRONMENT', 'development')
if env == 'production':
    config = KVCacheConfig(
        max_tokens=16384,
        enable_persistence=True,
        enable_prefetch=True
    )
else:
    config = KVCacheConfig(
        max_tokens=2048,
        enable_profiling=True
    )
```

### ❌ Anti-Patrón 7: No Validar Configuración

```python
# ❌ MALO
config = KVCacheConfig(max_tokens=-1)  # Inválido pero no validado
engine = UltraAdaptiveKVCacheEngine(config)

# ✅ BUENO
config = KVCacheConfig(max_tokens=4096)
engine = UltraAdaptiveKVCacheEngine(config)

# Validar antes de usar
validation = engine.validate_configuration()
if not validation['is_valid']:
    raise ValueError(f"Invalid config: {validation['issues']}")
```

## ⚡ Anti-Patrones de Performance

### ❌ Anti-Patrón 8: Procesar Requests Secuencialmente

```python
# ❌ MALO
results = []
for request in requests:
    result = await process_request(request)  # Secuencial
    results.append(result)

# ✅ BUENO
# Procesar en paralelo
results = await asyncio.gather(*[
    process_request(req) for req in requests
])

# O en batch
results = await engine.process_batch_optimized(requests)
```

### ❌ Anti-Patrón 9: No Usar Prefetch Cuando Apropiado

```python
# ❌ MALO
# Prefetch deshabilitado cuando hay patrones predecibles
config = KVCacheConfig(enable_prefetch=False)

# ✅ BUENO
# Habilitar prefetch para patrones conocidos
config = KVCacheConfig(
    enable_prefetch=True,
    prefetch_size=16
)
```

### ❌ Anti-Patrón 10: Cache Demasiado Pequeño

```python
# ❌ MALO
# Cache muy pequeño, muchos misses
config = KVCacheConfig(max_tokens=128)

# ✅ BUENO
# Cache apropiado para el workload
config = KVCacheConfig(max_tokens=8192)  # Ajustar según necesidad
```

### ❌ Anti-Patrón 11: Compresión Excesiva

```python
# ❌ MALO
# Compresión muy agresiva afecta calidad
config = KVCacheConfig(
    use_compression=True,
    compression_ratio=0.05  # Demasiado agresivo
)

# ✅ BUENO
# Balance entre memoria y calidad
config = KVCacheConfig(
    use_compression=True,
    compression_ratio=0.3  # Balance adecuado
)
```

## 🔒 Anti-Patrones de Seguridad

### ❌ Anti-Patrón 12: Secrets en Código

```python
# ❌ MALO
API_KEY = "sk-1234567890abcdef"  # NUNCA hacer esto

# ✅ BUENO
import os
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not set")
```

### ❌ Anti-Patrón 13: Sin Rate Limiting

```python
# ❌ MALO
@app.post("/api/query")
async def query(request: dict):
    # Sin rate limiting, vulnerable a abuse
    return await process(request)

# ✅ BUENO
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/api/query")
@limiter.limit("100/minute")
async def query(request: dict):
    return await process(request)
```

### ❌ Anti-Patrón 14: Input Sin Sanitización

```python
# ❌ MALO
async def process_query(query: str):
    # Sin sanitización, vulnerable a injection
    result = await engine.process_request({'text': query})
    return result

# ✅ BUENO
from bulk.core.ultra_adaptive_kv_cache_security import SecureEngineWrapper

secure_engine = SecureEngineWrapper(
    engine,
    enable_sanitization=True
)

async def process_query(query: str):
    result = await secure_engine.process_request_secure(
        {'text': query},
        api_key=api_key
    )
    return result
```

### ❌ Anti-Patrón 15: Logs Con Información Sensible

```python
# ❌ MALO
logger.info(f"Processing request: {request}")  # Puede contener secrets

# ✅ BUENO
# Redactar información sensible
def redact_request(request):
    redacted = request.copy()
    if 'api_key' in redacted:
        redacted['api_key'] = '***'
    return redacted

logger.info(f"Processing request: {redact_request(request)}")
```

## 💻 Anti-Patrones de Código

### ❌ Anti-Patrón 16: No Manejar Excepciones Async

```python
# ❌ MALO
async def process():
    result = await engine.process_request(request)
    return result  # Si falla, no se maneja

# ✅ BUENO
async def process():
    try:
        result = await engine.process_request(request)
        return result
    except CacheError as e:
        logger.error(f"Cache error: {e}")
        # Fallback
        return await process_directly(request)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
```

### ❌ Anti-Patrón 17: Crear Engine Múltiples Veces

```python
# ❌ MALO
def process_request(request):
    # Crear nuevo engine cada vez - muy ineficiente
    engine = UltraAdaptiveKVCacheEngine(KVCacheConfig())
    return engine.process_request(request)

# ✅ BUENO
# Crear engine una vez y reutilizar
engine = UltraAdaptiveKVCacheEngine(KVCacheConfig())

def process_request(request):
    return engine.process_request(request)
```

### ❌ Anti-Patrón 18: No Cerrar Recursos

```python
# ❌ MALO
async def process():
    engine = UltraAdaptiveKVCacheEngine(config)
    result = await engine.process_request(request)
    # Engine no se cierra, recursos no liberados
    return result

# ✅ BUENO
async def process():
    async with cache_engine_context(config) as engine:
        result = await engine.process_request(request)
        return result
    # Context manager cierra recursos automáticamente
```

### ❌ Anti-Patrón 19: Ignorar Métricas y Monitoreo

```python
# ❌ MALO
async def process():
    result = await engine.process_request(request)
    # No monitorear nada
    return result

# ✅ BUENO
async def process():
    start = time.time()
    result = await engine.process_request(request)
    duration = time.time() - start
    
    # Registrar métricas
    metrics.counter('requests_total').inc()
    metrics.histogram('request_duration').observe(duration)
    
    return result
```

### ❌ Anti-Patrón 20: Magic Numbers

```python
# ❌ MALO
config = KVCacheConfig(max_tokens=8192)  # ¿Por qué 8192?

# ✅ BUENO
# Usar constantes con significado
DEFAULT_CACHE_SIZE = 8192  # Basado en análisis de workload
config = KVCacheConfig(max_tokens=DEFAULT_CACHE_SIZE)
```

## 🎯 Resumen: Mejores Prácticas

### ✅ DO (Hacer)

- Validar configuración antes de usar
- Usar variables de entorno para configuración
- Manejar errores apropiadamente
- Monitorear métricas
- Usar rate limiting
- Sanitizar inputs
- Procesar en paralelo cuando sea posible
- Usar prefetching para patrones conocidos
- Cerrar recursos apropiadamente

### ❌ DON'T (No Hacer)

- Hardcodear configuración
- Ignorar TTL y expiración
- Procesar secuencialmente cuando se puede en paralelo
- Cachear sin validar
- Exponer secrets en código
- Ignorar rate limiting
- Usar mismas configs para dev y prod
- Crear engines múltiples veces
- Ignorar métricas y monitoreo

---

**Más información:**
- [Best Practices](BEST_PRACTICES.md)
- [Security Guide](SECURITY_GUIDE.md)
- [Performance Tuning](PERFORMANCE_TUNING.md)



