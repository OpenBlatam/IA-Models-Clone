# 🚀 Cheatsheet de Comandos - Blatam Academy Features

## 🐳 Docker Commands

### Gestión Básica

```bash
# Iniciar todos los servicios
docker-compose up -d

# Detener todos los servicios
docker-compose down

# Reiniciar un servicio específico
docker-compose restart bul

# Ver logs en tiempo real
docker-compose logs -f bul

# Ver logs de todos los servicios
docker-compose logs -f

# Ver estado de servicios
docker-compose ps

# Ejecutar comando en contenedor
docker-compose exec bul python --version
```

### Desarrollo

```bash
# Rebuild después de cambios
docker-compose up -d --build

# Rebuild sin cache
docker-compose build --no-cache

# Ver uso de recursos
docker stats

# Limpiar sistema
docker system prune -a
```

## 🐍 Python Commands

### KV Cache CLI

```bash
# Estadísticas
python bulk/core/ultra_adaptive_kv_cache_cli.py stats

# Health check
python bulk/core/ultra_adaptive_kv_cache_cli.py health

# Monitor en tiempo real
python bulk/core/ultra_adaptive_kv_cache_cli.py monitor

# Monitor con dashboard
python bulk/core/ultra_adaptive_kv_cache_cli.py monitor --dashboard

# Limpiar cache
python bulk/core/ultra_adaptive_kv_cache_cli.py clear-cache

# Test de cache
python bulk/core/ultra_adaptive_kv_cache_cli.py test --text "Hello world"

# Backup
python bulk/core/ultra_adaptive_kv_cache_cli.py backup --path /backup/cache.pt

# Restore
python bulk/core/ultra_adaptive_kv_cache_cli.py restore --path /backup/cache.pt
```

### Scripts de Sistema

```bash
# Setup completo
./scripts/setup_complete.sh

# Health check
./scripts/health_check.sh

# Benchmarking
./scripts/benchmark.sh
```

## 🔧 Configuración

### Variables de Entorno

```bash
# Ver variables actuales
env | grep KV_CACHE

# Exportar variable temporalmente
export KV_CACHE_MAX_TOKENS=8192

# Cargar desde archivo
source .env

# Verificar configuración
python -c "from bulk.core.ultra_adaptive_kv_cache_engine import KVCacheConfig; print(KVCacheConfig())"
```

### Configuración Rápida

```python
# Desarrollo
python -c "
from bulk.core.ultra_adaptive_kv_cache_engine import KVCacheConfig
config = KVCacheConfig(max_tokens=2048, enable_profiling=True)
print('Development config ready')
"

# Producción
python -c "
from bulk.core.ultra_adaptive_kv_cache_engine import KVCacheConfig
config = KVCacheConfig(max_tokens=8192, enable_persistence=True)
print('Production config ready')
"
```

## 📊 Monitoreo

### Prometheus

```bash
# Acceder a Prometheus
curl http://localhost:9090/api/v1/query?query=kv_cache_requests_total

# Ver métricas específicas
curl http://localhost:9090/api/v1/query?query=kv_cache_latency_seconds

# Exportar métricas
curl http://localhost:9090/api/v1/query?query=kv_cache_requests_total > metrics.json
```

### Grafana

```bash
# Acceder a Grafana
open http://localhost:3000

# Dashboard default
# Usuario: admin
# Password: admin (cambiar en primera vez)
```

### Logs

```bash
# Ver logs del sistema
docker-compose logs -f --tail=100

# Filtrar por servicio
docker-compose logs -f bul | grep "error"

# Exportar logs
docker-compose logs > logs.txt

# Limpiar logs antiguos
docker-compose logs --tail=0
```

## 🗄️ Base de Datos

### PostgreSQL

```bash
# Conectar a PostgreSQL
docker-compose exec postgres psql -U postgres -d blatam_academy

# Backup
docker-compose exec postgres pg_dump -U postgres blatam_academy > backup.sql

# Restore
docker-compose exec -T postgres psql -U postgres blatam_academy < backup.sql

# Ver tablas
docker-compose exec postgres psql -U postgres -d blatam_academy -c "\dt"
```

### Redis

```bash
# Conectar a Redis CLI
docker-compose exec redis redis-cli

# Ver todas las keys
docker-compose exec redis redis-cli KEYS "*"

# Ver tamaño del cache
docker-compose exec redis redis-cli INFO memory

# Limpiar cache
docker-compose exec redis redis-cli FLUSHALL
```

## 🔍 Debugging

### Python Debug

```bash
# Debug con pdb
python -m pdb bulk/main.py

# Debug con ipdb (si instalado)
python -m ipdb bulk/main.py

# Verificar imports
python -c "import bulk.core.ultra_adaptive_kv_cache_engine; print('OK')"

# Verificar versión
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### GPU Debug

```bash
# Ver estado GPU
nvidia-smi

# Monitoreo continuo GPU
watch -n 1 nvidia-smi

# Ver procesos usando GPU
fuser -v /dev/nvidia*

# Limpiar procesos GPU
nvidia-smi --gpu-reset
```

## 📡 API Testing

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Query endpoint
curl -X POST http://localhost:8002/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Test query", "priority": 1}'

# Stats endpoint
curl http://localhost:8002/api/stats

# Batch endpoint
curl -X POST http://localhost:8002/api/batch \
  -H "Content-Type: application/json" \
  -d '{"queries": ["Query 1", "Query 2"]}'
```

### HTTPie

```bash
# Instalar HTTPie (si no está)
pip install httpie

# GET request
http GET localhost:8002/api/stats

# POST request
http POST localhost:8002/api/query query="Test" priority:=1

# Con autenticación
http POST localhost:8002/api/query query="Test" Authorization:"Bearer token"
```

## 🧪 Testing

### Unit Tests

```bash
# Ejecutar todos los tests
pytest tests/

# Tests específicos
pytest tests/test_cache.py

# Con coverage
pytest --cov=bulk tests/

# Con verbose
pytest -v tests/

# Solo tests marcados
pytest -m "slow" tests/
```

### Integration Tests

```bash
# Tests de integración
pytest tests/integration/

# Con servicios up
docker-compose up -d && pytest tests/integration/
```

### Performance Tests

```bash
# Benchmarking
pytest benchmarks/ --benchmark-only

# Comparar benchmarks
pytest benchmarks/ --benchmark-compare
```

## 🔐 Seguridad

### SSL/TLS

```bash
# Verificar certificado
openssl x509 -in cert.pem -text -noout

# Test SSL connection
openssl s_client -connect localhost:443

# Verificar expiración
openssl x509 -in cert.pem -noout -dates
```

### Security Audit

```bash
# Verificar dependencias vulnerables
pip list --outdated

# Security check (usar safety si instalado)
safety check

# Verificar permisos
ls -la /data/cache
```

## 💾 Backup y Restore

### Backup Completo

```bash
# Backup database
docker-compose exec postgres pg_dump -U postgres blatam_academy > db_backup.sql

# Backup cache
python bulk/core/ultra_adaptive_kv_cache_cli.py backup --path /backup/cache_$(date +%Y%m%d).pt

# Backup configuración
cp .env .env.backup
cp config/kv_cache.yaml config/kv_cache.yaml.backup
```

### Restore

```bash
# Restore database
docker-compose exec -T postgres psql -U postgres blatam_academy < db_backup.sql

# Restore cache
python bulk/core/ultra_adaptive_kv_cache_cli.py restore --path /backup/cache_20240101.pt

# Restore configuración
cp .env.backup .env
```

## 🚀 Deployment

### Pre-Deployment

```bash
# Verificar configuración
python -c "from bulk.core.ultra_adaptive_kv_cache_engine import KVCacheConfig; config = KVCacheConfig(); print('Valid' if config.max_tokens > 0 else 'Invalid')"

# Health check
./scripts/health_check.sh

# Verificar recursos
docker stats --no-stream
```

### Post-Deployment

```bash
# Verificar servicios
docker-compose ps

# Verificar logs
docker-compose logs --tail=50

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8002/api/stats

# Verificar métricas
curl http://localhost:9090/api/v1/query?query=up
```

## 🔄 Maintenance

### Limpieza

```bash
# Limpiar cache
python bulk/core/ultra_adaptive_kv_cache_cli.py clear-cache

# Limpiar logs antiguos
docker-compose logs --tail=0

# Limpiar imágenes no usadas
docker image prune -a

# Limpiar volúmenes no usados
docker volume prune
```

### Actualización

```bash
# Pull últimos cambios
git pull origin main

# Rebuild servicios
docker-compose up -d --build

# Verificar versión
python -c "import bulk; print(bulk.__version__)"
```

---

**Más información:**
- [Quick Start Guide](QUICK_START_GUIDE.md)
- [Troubleshooting Quick Reference](TROUBLESHOOTING_QUICK_REFERENCE.md)
- [Production Ready](bulk/PRODUCTION_READY.md)

