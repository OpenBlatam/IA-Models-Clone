# ⚡ Guías de Setup Rápido por Caso de Uso

## 📋 Setup Rápido por Escenario

### 🚀 Setup: Desarrollo Local

```bash
# 1. Clonar repositorio
git clone <repo-url>
cd blatam-academy/agents/backend/onyx/server/features

# 2. Crear .env
cp .env.example .env
# Editar .env con valores de desarrollo

# 3. Iniciar servicios
docker-compose up -d

# 4. Verificar
./scripts/health_check.sh

# 5. Configurar KV Cache para desarrollo
python -c "
from bulk.core.ultra_adaptive_kv_cache_engine import KVCacheConfig
config = KVCacheConfig(max_tokens=2048, enable_profiling=True)
print('✅ Development config ready')
"
```

**Tiempo estimado**: 5 minutos

### 🏭 Setup: Producción

```bash
# 1. Preparar servidor
sudo apt update
sudo apt install docker docker-compose

# 2. Configurar .env de producción
cp config/templates/production.env.template .env
# Editar con valores reales de producción

# 3. Configurar KV Cache producción
cp config/templates/kv_cache_production.yaml config/kv_cache.yaml

# 4. Setup SSL (si aplica)
# Configurar certificados SSL

# 5. Iniciar servicios
docker-compose up -d

# 6. Verificar health
./scripts/health_check.sh

# 7. Configurar monitoreo
# Prometheus y Grafana ya están configurados

# 8. Verificar backups
# Configurar backups automáticos
```

**Tiempo estimado**: 30 minutos

### 🧪 Setup: Testing/QA

```bash
# 1. Setup base (como desarrollo)
docker-compose up -d

# 2. Configurar para testing
export ENVIRONMENT=test
export KV_CACHE_MAX_TOKENS=1024

# 3. Ejecutar tests
pytest tests/ -v

# 4. Tests de integración
pytest tests/integration/ -v

# 5. Tests de performance
pytest benchmarks/ --benchmark-only
```

**Tiempo estimado**: 10 minutos

### 📊 Setup: Monitoreo y Analytics

```bash
# 1. Setup base
docker-compose up -d

# 2. Acceder a Prometheus
open http://localhost:9090

# 3. Acceder a Grafana
open http://localhost:3000
# Usuario: admin / Password: admin (cambiar)

# 4. Importar dashboards
# Dashboards ya están configurados en docker-compose

# 5. Configurar alertas
# Ver prometheus/alerts.yml
```

**Tiempo estimado**: 15 minutos

### 🔒 Setup: Alta Seguridad

```bash
# 1. Setup base de producción
# ... (ver setup producción)

# 2. Configurar firewall
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# 3. Configurar rate limiting agresivo
# En .env:
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# 4. Habilitar todas las features de seguridad
# Ver SECURITY_CHECKLIST.md

# 5. Configurar logging de seguridad
# Audit logs habilitados

# 6. Verificar SSL
openssl s_client -connect yourdomain.com:443
```

**Tiempo estimado**: 45 minutos

### ⚡ Setup: Alto Rendimiento

```bash
# 1. Setup base de producción
# ... (ver setup producción)

# 2. Configurar KV Cache para performance
config = KVCacheConfig(
    max_tokens=16384,
    cache_strategy=CacheStrategy.ADAPTIVE,
    enable_prefetch=True,
    prefetch_size=32,
    use_compression=False,  # Máxima velocidad
    enable_persistence=True
)

# 3. Configurar múltiples workers
NUM_WORKERS=32

# 4. Configurar connection pooling agresivo
DATABASE_POOL_SIZE=40
DATABASE_MAX_OVERFLOW=80

# 5. Habilitar batch processing
BATCH_SIZE=50

# 6. Configurar load balancer
# Ver nginx/ config
```

**Tiempo estimado**: 30 minutos

### 💾 Setup: Eficiencia de Memoria

```bash
# 1. Setup base
docker-compose up -d

# 2. Configurar KV Cache eficiente
config = KVCacheConfig(
    max_tokens=2048,
    use_compression=True,
    compression_ratio=0.2,  # Agresiva
    use_quantization=True,
    quantization_bits=4,
    enable_gc=True,
    gc_threshold=0.6
)

# 3. Configurar límites de memoria
MAX_MEMORY_MB=2048

# 4. Habilitar memory cleanup frecuente
GC_INTERVAL=300  # Cada 5 minutos
```

**Tiempo estimado**: 15 minutos

## 🎯 Presets de Configuración Rápida

### Preset 1: Desarrollo Rápido

```python
# development_quick.py
from bulk.core.ultra_adaptive_kv_cache_engine import (
    UltraAdaptiveKVCacheEngine,
    KVCacheConfig
)

config = KVCacheConfig(
    max_tokens=2048,
    enable_profiling=True,
    enable_persistence=False
)

engine = UltraAdaptiveKVCacheEngine(config)
print("✅ Development engine ready")
```

### Preset 2: Producción Estándar

```python
# production_standard.py
config = KVCacheConfig(
    max_tokens=8192,
    cache_strategy=CacheStrategy.ADAPTIVE,
    enable_persistence=True,
    enable_prefetch=True,
    prefetch_size=16,
    use_compression=True,
    compression_ratio=0.3
)

engine = UltraAdaptiveKVCacheEngine(config)
print("✅ Production engine ready")
```

### Preset 3: Alto Rendimiento

```python
# high_performance.py
config = KVCacheConfig(
    max_tokens=16384,
    cache_strategy=CacheStrategy.ADAPTIVE,
    enable_prefetch=True,
    prefetch_size=32,
    use_compression=False,  # Sin compresión para velocidad
    enable_persistence=True,
    pin_memory=True,
    non_blocking=True
)

engine = UltraAdaptiveKVCacheEngine(config)
print("✅ High performance engine ready")
```

### Preset 4: Memoria Limitada

```python
# memory_limited.py
config = KVCacheConfig(
    max_tokens=2048,
    use_compression=True,
    compression_ratio=0.15,  # Muy agresiva
    use_quantization=True,
    quantization_bits=4,
    enable_gc=True,
    gc_threshold=0.5
)

engine = UltraAdaptiveKVCacheEngine(config)
print("✅ Memory-efficient engine ready")
```

## 🔧 Scripts de Setup Automático

### Setup Completo Automático

```bash
#!/bin/bash
# setup_auto.sh

echo "🚀 Auto Setup Blatam Academy Features"

# Detectar entorno
if [ -f ".env.production" ]; then
    ENV="production"
elif [ -f ".env.development" ]; then
    ENV="development"
else
    ENV="development"
fi

echo "📋 Environment: $ENV"

# Setup según entorno
case $ENV in
    production)
        echo "🏭 Setting up for production..."
        cp config/templates/production.env.template .env
        cp config/templates/kv_cache_production.yaml config/kv_cache.yaml
        ;;
    development)
        echo "💻 Setting up for development..."
        cp .env.example .env
        ;;
esac

# Iniciar servicios
echo "🐳 Starting Docker services..."
docker-compose up -d

# Esperar servicios
echo "⏳ Waiting for services..."
sleep 10

# Health check
echo "🏥 Running health check..."
./scripts/health_check.sh

echo "✅ Setup complete!"
```

### Setup Solo KV Cache

```python
# setup_kv_cache_only.py
from bulk.core.ultra_adaptive_kv_cache_engine import (
    UltraAdaptiveKVCacheEngine,
    KVCacheConfig,
    CacheStrategy
)
import os

def setup_kv_cache(environment=None):
    """Setup KV Cache según entorno."""
    if not environment:
        environment = os.getenv('ENVIRONMENT', 'development')
    
    if environment == 'production':
        config = KVCacheConfig(
            max_tokens=8192,
            cache_strategy=CacheStrategy.ADAPTIVE,
            enable_persistence=True,
            enable_prefetch=True,
            prefetch_size=16
        )
    else:
        config = KVCacheConfig(
            max_tokens=2048,
            enable_profiling=True
        )
    
    engine = UltraAdaptiveKVCacheEngine(config)
    print(f"✅ KV Cache configured for {environment}")
    return engine

if __name__ == "__main__":
    engine = setup_kv_cache()
```

## ✅ Verificación Post-Setup

```python
# verify_setup.py
import asyncio
from bulk.core.ultra_adaptive_kv_cache_engine import (
    UltraAdaptiveKVCacheEngine,
    KVCacheConfig
)

async def verify_setup():
    """Verificar que el setup es correcto."""
    config = KVCacheConfig(max_tokens=4096)
    engine = UltraAdaptiveKVCacheEngine(config)
    
    checks = {
        "engine_created": engine is not None,
        "config_valid": engine.validate_configuration()['is_valid'],
    }
    
    # Test básico
    try:
        result = await engine.process_request({
            'text': 'Test query',
            'priority': 1
        })
        checks["basic_request"] = 'result' in result
    except Exception as e:
        checks["basic_request"] = False
        checks["error"] = str(e)
    
    # Stats
    try:
        stats = engine.get_stats()
        checks["stats_available"] = stats is not None
    except:
        checks["stats_available"] = False
    
    print("\n✅ Setup Verification:")
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
    
    return all(v for k, v in checks.items() if k != 'error')

if __name__ == "__main__":
    result = asyncio.run(verify_setup())
    exit(0 if result else 1)
```

---

**Más información:**
- [Quick Start Guide](QUICK_START_GUIDE.md)
- [Deployment Checklist](DEPLOYMENT_CHECKLIST.md)
- [Production Ready](bulk/PRODUCTION_READY.md)



