# 🚀 Guía de Producción - BUL System

## 📋 Checklist de Producción

### ✅ Pre-Deployment

- [ ] **Configuración**
  - [ ] Variables de entorno configuradas
  - [ ] Secrets en secret manager
  - [ ] API keys rotadas y seguras
  - [ ] SSL/TLS configurado
  - [ ] CORS configurado correctamente

- [ ] **Base de Datos**
  - [ ] PostgreSQL configurado y respaldado
  - [ ] Connection pooling configurado
  - [ ] Índices creados
  - [ ] Migrations aplicadas

- [ ] **Cache**
  - [ ] Redis configurado y monitoreado
  - [ ] KV Cache configurado para producción
  - [ ] Persistencia habilitada
  - [ ] Backup automático configurado

- [ ] **Monitoreo**
  - [ ] Prometheus configurado
  - [ ] Grafana dashboards creados
  - [ ] Alertas configuradas
  - [ ] Logs centralizados

- [ ] **Seguridad**
  - [ ] Rate limiting activado
  - [ ] Authentication implementado
  - [ ] Input validation activado
  - [ ] Security headers configurados
  - [ ] Firewall configurado

- [ ] **Escalabilidad**
  - [ ] Load balancer configurado
  - [ ] Auto-scaling configurado (si aplica)
  - [ ] Health checks implementados
  - [ ] Graceful shutdown implementado

## ⚙️ Configuración Óptima para Producción

### KV Cache Production Config

```python
from bulk.core.ultra_adaptive_kv_cache_engine import (
    UltraAdaptiveKVCacheEngine,
    KVCacheConfig,
    CacheStrategy
)

production_config = KVCacheConfig(
    # Core
    num_heads=16,
    head_dim=128,
    max_tokens=16384,  # Ajustar según memoria disponible
    block_size=256,
    
    # Strategy - Adaptive para mejor rendimiento general
    cache_strategy=CacheStrategy.ADAPTIVE,
    cache_mode=CacheMode.INFERENCE,
    
    # Optimization
    use_compression=True,
    compression_ratio=0.3,  # Balance entre memoria y velocidad
    use_quantization=False,  # Solo si memoria es crítica
    
    # Memory
    max_memory_mb=8192,  # Ajustar según servidor
    enable_gc=True,
    gc_threshold=0.8,
    
    # Performance
    pin_memory=True,
    non_blocking=True,
    dtype=torch.float16,  # Mejor rendimiento
    
    # Advanced
    enable_persistence=True,
    persistence_path="/data/kv_cache",  # Ruta persistente
    enable_prefetch=True,
    prefetch_size=16,  # Agresivo para mejor rendimiento
    enable_profiling=False,  # Desactivado en producción
    enable_distributed=True,  # Si hay múltiples GPUs
    distributed_backend="nccl",
    
    # Security
    multi_tenant=False,  # O True si es SaaS
    tenant_isolation=True
)

engine = UltraAdaptiveKVCacheEngine(production_config)
```

### Environment Variables

```bash
# .env.production

# Application
APP_ENV=production
DEBUG=false
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://user:password@db:5432/blatam_academy
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40

# Redis
REDIS_URL=redis://redis:6379/0
REDIS_TTL=3600

# KV Cache
KV_CACHE_MAX_TOKENS=16384
KV_CACHE_STRATEGY=adaptive
KV_CACHE_ENABLE_PERSISTENCE=true
KV_CACHE_PERSISTENCE_PATH=/data/kv_cache
KV_CACHE_ENABLE_PREFETCH=true
KV_CACHE_PREFETCH_SIZE=16

# Security
SECRET_KEY=<strong-secret-key>
ACCESS_TOKEN_EXPIRE_MINUTES=30
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=3600

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
GRAFANA_ENABLED=true
GRAFANA_PORT=3000

# API Keys
OPENAI_API_KEY=<your-key>
ANTHROPIC_API_KEY=<your-key>
```

## 🔒 Seguridad en Producción

### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# Por endpoint
@app.post("/api/query")
@limiter.limit("100/minute")
async def query_endpoint(request: Request):
    pass

# Por usuario
@app.post("/api/batch")
@limiter.limit("10/minute", key_func=get_user_id)
async def batch_endpoint(request: Request):
    pass
```

### Input Sanitization

```python
from bulk.core.ultra_adaptive_kv_cache_security import SecureEngineWrapper

secure_engine = SecureEngineWrapper(
    engine,
    enable_sanitization=True,
    block_sql_injection=True,
    block_xss=True,
    block_path_traversal=True,
    max_input_length=10000
)
```

### HTTPS y SSL

```nginx
server {
    listen 443 ssl http2;
    ssl_certificate /etc/ssl/certs/app.crt;
    ssl_certificate_key /etc/ssl/private/app.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    location / {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📊 Monitoreo en Producción

### Health Check Endpoint

```python
@app.get("/health")
async def health_check():
    """Health check completo."""
    checks = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {}
    }
    
    # Check database
    try:
        db.execute("SELECT 1")
        checks["checks"]["database"] = "healthy"
    except:
        checks["checks"]["database"] = "unhealthy"
        checks["status"] = "degraded"
    
    # Check Redis
    try:
        redis_client.ping()
        checks["checks"]["redis"] = "healthy"
    except:
        checks["checks"]["redis"] = "unhealthy"
        checks["status"] = "degraded"
    
    # Check KV Cache
    try:
        stats = engine.get_stats()
        checks["checks"]["kv_cache"] = {
            "status": "healthy",
            "hit_rate": stats["hit_rate"],
            "memory_usage": stats["memory_usage"]
        }
    except:
        checks["checks"]["kv_cache"] = "unhealthy"
        checks["status"] = "degraded"
    
    status_code = 200 if checks["status"] == "healthy" else 503
    return JSONResponse(checks, status_code=status_code)
```

### Alertas Críticas

```yaml
# prometheus/alerts.yml
groups:
  - name: bul_alerts
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, kv_cache_latency_seconds_bucket) > 1
        for: 5m
        annotations:
          summary: "High latency detected"
      
      - alert: LowCacheHitRate
        expr: rate(kv_cache_requests_total{status="hit"}[5m]) / rate(kv_cache_requests_total[5m]) < 0.5
        for: 10m
        annotations:
          summary: "Low cache hit rate"
      
      - alert: HighMemoryUsage
        expr: kv_cache_memory_usage_bytes / kv_cache_memory_limit_bytes > 0.9
        for: 5m
        annotations:
          summary: "High memory usage"
```

## 💾 Backup y Disaster Recovery

### Backup Automático

```python
import schedule
import time

def backup_cache():
    """Backup del cache."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"/backup/cache_{timestamp}.pt"
    
    engine.persist(backup_path)
    
    # Comprimir
    import gzip
    import shutil
    
    with open(backup_path, 'rb') as f_in:
        with gzip.open(f"{backup_path}.gz", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Eliminar backup sin comprimir
    os.remove(backup_path)
    
    # Mantener solo últimos 7 días
    cleanup_old_backups()

def cleanup_old_backups():
    """Limpiar backups antiguos."""
    import glob
    backups = glob.glob("/backup/cache_*.pt.gz")
    backups.sort()
    
    # Mantener últimos 7 días
    if len(backups) > 7:
        for backup in backups[:-7]:
            os.remove(backup)

# Programar backup cada 6 horas
schedule.every(6).hours.do(backup_cache)

# Ejecutar en thread separado
import threading
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)

threading.Thread(target=run_scheduler, daemon=True).start()
```

### Restore Procedure

```python
def restore_from_backup(backup_path: str):
    """Restaurar desde backup."""
    import gzip
    import shutil
    
    # Descomprimir si es necesario
    if backup_path.endswith('.gz'):
        decompressed = backup_path[:-3]
        with gzip.open(backup_path, 'rb') as f_in:
            with open(decompressed, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        backup_path = decompressed
    
    # Cargar
    engine.load(backup_path)
    print(f"✅ Restored from {backup_path}")
```

## 📈 Performance Tuning para Producción

### Optimizaciones Recomendadas

1. **Connection Pooling**
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

2. **Async Processing**
```python
# Usar async/await para I/O
async def process_request(request):
    result = await cache_engine.process_request(request)
    return result
```

3. **Batch Processing**
```python
# Procesar en batches para mejor throughput
results = await cache_engine.process_batch_optimized(
    requests,
    batch_size=20
)
```

4. **Caching Agresivo**
```python
# Configurar TTLs apropiados
redis_client.setex('key', 3600, 'value')  # 1 hora
```

## 🚨 Incident Response

### Runbook

1. **Alta Latencia**
   - Verificar métricas en Grafana
   - Revisar logs de errores
   - Verificar carga del servidor
   - Escalar si es necesario

2. **Error de Cache**
   - Verificar integridad del cache
   - Limpiar cache si es necesario
   - Restaurar desde backup

3. **Memory Leak**
   - Identificar proceso problemático
   - Reiniciar si es necesario
   - Investigar causa raíz

4. **Database Issues**
   - Verificar conexiones
   - Revisar queries lentas
   - Verificar índices

---

**Más información:**
- [Deployment Checklist](../DEPLOYMENT_CHECKLIST.md)
- [Security Guide](../SECURITY_GUIDE.md)
- [Performance Tuning](../PERFORMANCE_TUNING.md)
- [Troubleshooting](../TROUBLESHOOTING_GUIDE.md)

