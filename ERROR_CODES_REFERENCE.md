# 🚨 Referencia de Códigos de Error - Blatam Academy Features

## 📋 Errores Comunes y Soluciones Rápidas

### KV Cache Engine Errors

#### E001: CUDA Out of Memory

```
Error: CUDA out of memory. Tried to allocate X.XX GB
```

**Causa**: Memoria GPU insuficiente

**Solución Rápida**:
```python
# Opción 1: Reducir max_tokens
config.max_tokens = 2048  # Reducir

# Opción 2: Habilitar compresión
config.use_compression = True
config.compression_ratio = 0.2

# Opción 3: Limpiar cache GPU
torch.cuda.empty_cache()
```

**Solución Completa**: Ver [TROUBLESHOOTING_BY_SYMPTOM.md](TROUBLESHOOTING_BY_SYMPTOM.md#error-cuda-out-of-memory)

---

#### E002: Cache Key Not Found

```
Error: Cache key not found: <key>
```

**Causa**: Intentando acceder a key que no existe en cache

**Solución Rápida**:
```python
# Verificar antes de acceder
if key in cache:
    value = cache[key]
else:
    value = compute_value()  # Fallback
```

---

#### E003: Invalid Configuration

```
Error: Invalid configuration: <field> must be > 0
```

**Causa**: Configuración inválida

**Solución Rápida**:
```python
# Validar configuración
validation = engine.validate_configuration()
if not validation['is_valid']:
    print(f"Issues: {validation['issues']}")
    # Corregir issues
```

---

#### E004: Cache Corruption

```
Error: Cache corruption detected
```

**Causa**: Cache corrupto o inconsistente

**Solución Rápida**:
```python
# Limpiar cache corrupto
engine.clear_cache()

# O restaurar desde backup
engine.restore_from_backup('/backup/cache.pt')
```

---

#### E005: Persistence Failed

```
Error: Failed to persist cache: <error>
```

**Causa**: Error al guardar cache en disco

**Solución Rápida**:
```bash
# Verificar permisos
chmod 755 /data/cache
chown user:user /data/cache

# Verificar espacio
df -h /data/cache
```

---

### System Errors

#### S001: Service Unavailable

```
Error: Service unavailable (503)
```

**Causa**: Servicio no está corriendo o sobrecargado

**Solución Rápida**:
```bash
# Verificar servicios
docker-compose ps

# Reiniciar servicio
docker-compose restart bul

# Verificar logs
docker-compose logs bul
```

---

#### S002: Database Connection Failed

```
Error: Database connection failed
```

**Causa**: No se puede conectar a PostgreSQL

**Solución Rápida**:
```bash
# Verificar servicio DB
docker-compose ps postgres

# Verificar conexión
docker-compose exec postgres psql -U postgres -c "SELECT 1"

# Verificar DATABASE_URL
echo $DATABASE_URL
```

---

#### S003: Redis Connection Failed

```
Error: Redis connection failed
```

**Causa**: No se puede conectar a Redis

**Solución Rápida**:
```bash
# Verificar servicio Redis
docker-compose ps redis

# Verificar conexión
docker-compose exec redis redis-cli ping

# Verificar REDIS_URL
echo $REDIS_URL
```

---

#### S004: Rate Limit Exceeded

```
Error: Rate limit exceeded (429)
```

**Causa**: Demasiadas requests muy rápido

**Solución Rápida**:
```python
# Implementar backoff
import time

def request_with_backoff(url, max_retries=3):
    for i in range(max_retries):
        try:
            return requests.get(url)
        except RateLimitError:
            wait = 2 ** i  # Exponential backoff
            time.sleep(wait)
    raise Exception("Rate limit exceeded")
```

---

### API Errors

#### A001: Authentication Failed

```
Error: Authentication failed (401)
```

**Causa**: Token inválido o expirado

**Solución Rápida**:
```python
# Verificar token
token = get_token()
if is_token_expired(token):
    token = refresh_token()

# Obtener nuevo token
```

---

#### A002: Invalid Request Format

```
Error: Invalid request format (400)
```

**Causa**: Request mal formado

**Solución Rápida**:
```python
# Validar request antes de enviar
def validate_request(request):
    required_fields = ['text', 'priority']
    for field in required_fields:
        if field not in request:
            raise ValueError(f"Missing required field: {field}")
    return True
```

---

#### A003: Resource Not Found

```
Error: Resource not found (404)
```

**Causa**: Endpoint o recurso no existe

**Solución Rápida**:
```bash
# Verificar endpoint
curl http://localhost:8002/health

# Verificar documentación API
# Ver API_REFERENCE.md
```

---

## 🔍 Búsqueda Rápida por Error

### Por Mensaje de Error

```bash
# Buscar en logs
docker-compose logs bul | grep -i "error"

# Buscar error específico
docker-compose logs bul | grep -i "E001"

# Buscar en documentación
grep -r "E001" docs/
```

### Por Código de Error HTTP

- **400**: Bad Request → Ver [API_REFERENCE.md](API_REFERENCE.md)
- **401**: Unauthorized → Verificar autenticación
- **403**: Forbidden → Verificar permisos
- **404**: Not Found → Verificar endpoint
- **429**: Too Many Requests → Rate limiting
- **500**: Internal Server Error → Ver logs
- **503**: Service Unavailable → Verificar servicios

## 🛠️ Herramientas de Diagnóstico por Error

### Script de Diagnóstico Automático

```python
# diagnose_error.py
import sys

def diagnose_error(error_code):
    """Diagnosticar error automáticamente."""
    
    error_handlers = {
        'E001': diagnose_cuda_oom,
        'E002': diagnose_cache_key,
        'E003': diagnose_invalid_config,
        'E004': diagnose_cache_corruption,
        'E005': diagnose_persistence,
        'S001': diagnose_service_unavailable,
        'S002': diagnose_db_connection,
        'S003': diagnose_redis_connection,
        'S004': diagnose_rate_limit
    }
    
    handler = error_handlers.get(error_code)
    if handler:
        return handler()
    else:
        return f"Unknown error code: {error_code}"

def diagnose_cuda_oom():
    """Diagnosticar CUDA OOM."""
    import torch
    
    checks = {
        'cuda_available': torch.cuda.is_available(),
        'memory_allocated': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
        'memory_reserved': torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
    }
    
    if checks['cuda_available']:
        print(f"GPU Memory Allocated: {checks['memory_allocated']:.2f} GB")
        print(f"GPU Memory Reserved: {checks['memory_reserved']:.2f} GB")
        
        if checks['memory_allocated'] > 6:
            return "High GPU memory usage. Solutions: Reduce max_tokens, enable compression, or use CPU fallback"
    else:
        return "CUDA not available. Use CPU fallback."
    
    return "GPU memory OK"

if __name__ == "__main__":
    error_code = sys.argv[1] if len(sys.argv) > 1 else "E001"
    result = diagnose_error(error_code)
    print(result)
```

## 📊 Tabla de Referencia Rápida

| Código | Tipo | Severidad | Solución Rápida |
|--------|------|-----------|-----------------|
| E001 | CUDA OOM | Alta | Reducir max_tokens |
| E002 | Key Not Found | Baja | Verificar key existe |
| E003 | Invalid Config | Media | Validar configuración |
| E004 | Cache Corruption | Alta | Limpiar cache |
| E005 | Persistence Failed | Media | Verificar permisos |
| S001 | Service Unavailable | Alta | Reiniciar servicio |
| S002 | DB Connection | Alta | Verificar DB |
| S003 | Redis Connection | Media | Verificar Redis |
| S004 | Rate Limit | Baja | Implementar backoff |
| A001 | Auth Failed | Alta | Refrescar token |
| A002 | Invalid Request | Baja | Validar request |
| A003 | Not Found | Baja | Verificar endpoint |

---

**Más información:**
- [Troubleshooting by Symptom](TROUBLESHOOTING_BY_SYMPTOM.md)
- [Troubleshooting Quick Reference](TROUBLESHOOTING_QUICK_REFERENCE.md)
- [FAQ](FAQ.md)

