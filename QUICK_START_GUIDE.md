# 🚀 Guía de Inicio Rápido - Blatam Academy Features

## ⚡ Inicio Rápido (5 minutos)

### Paso 1: Verificar Prerrequisitos

```bash
# Verificar Docker
docker --version

# Verificar Docker Compose
docker-compose --version

# Verificar Python
python --version  # Debe ser 3.8+
```

### Paso 2: Iniciar Sistema Completo

```bash
# Navegar al directorio
cd C:\blatam-academy\agents\backend\onyx\server\features

# Iniciar todo el sistema
python start_system.py start
```

### Paso 3: Verificar Estado

```bash
# Verificar estado de servicios
python start_system.py status

# O verificar manualmente
curl http://localhost:8000/health
```

### Paso 4: Acceder a la Documentación

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

## 🎯 Uso Rápido por Módulo

### BUL - Generación de Documentos

```bash
# Iniciar BUL
cd bulk
python main.py --mode full

# Ejemplo de uso
curl -X POST "http://localhost:8002/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Create a marketing strategy", "priority": 1}'
```

### Content Redundancy Detector

```bash
# Analizar contenido
curl -X POST "http://localhost:8001/analyze" \
  -H "Content-Type: application/json" \
  -d '{"content": "Your text here"}'
```

### Business Agents

```bash
# Ejecutar agente
curl -X POST "http://localhost:8004/business-agents/marketing_001/execute" \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"task": "plan campaign"}}'
```

### Export IA

```bash
# Exportar documento
curl -X POST "http://localhost:8005/api/v1/exports/generate" \
  -H "Content-Type: application/json" \
  -d '{"content": {...}, "format": "pdf"}'
```

## ⚡ Ultra Adaptive KV Cache - Inicio Rápido

### Uso Básico

```python
from bulk.core.ultra_adaptive_kv_cache_engine import TruthGPTIntegration

# Crear engine
engine = TruthGPTIntegration.create_engine_for_truthgpt()

# Procesar request
result = await engine.process_request({
    'text': 'Your query',
    'max_length': 100,
    'session_id': 'user_123'
})
```

### Con Seguridad

```python
from bulk.core.ultra_adaptive_kv_cache_security import SecureEngineWrapper

secure_engine = SecureEngineWrapper(
    engine,
    enable_sanitization=True,
    enable_rate_limiting=True
)

result = await secure_engine.process_request_secure(
    request,
    client_ip="192.168.1.100",
    api_key="your-key"
)
```

### Monitoreo

```bash
# CLI de monitoreo
python bulk/core/ultra_adaptive_kv_cache_cli.py monitor --dashboard

# Estadísticas
python bulk/core/ultra_adaptive_kv_cache_cli.py stats

# Health check
python bulk/core/ultra_adaptive_kv_cache_cli.py health
```

## 🔧 Configuración Rápida

### Variables de Entorno Esenciales

```bash
# Crear archivo .env
cat > .env << EOF
OPENROUTER_API_KEY=tu-api-key-aqui
DATABASE_URL=postgresql://postgres:password@localhost:5432/blatam_academy
REDIS_URL=redis://localhost:6379
SECRET_KEY=tu-clave-secreta
EOF
```

### Configuración de KV Cache

```python
from bulk.core.ultra_adaptive_kv_cache_config_manager import ConfigPreset

# Aplicar preset de producción
ConfigPreset.apply_preset(engine, 'production')
```

## 📊 Verificación Rápida

```bash
# 1. Verificar servicios Docker
docker ps

# 2. Verificar logs
docker-compose logs -f integration-system

# 3. Verificar métricas
curl http://localhost:9090/metrics

# 4. Verificar KV Cache
python bulk/core/ultra_adaptive_kv_cache_cli.py stats
```

## 🐛 Troubleshooting Rápido

### Servicio no inicia
```bash
docker-compose logs [service-name]
docker-compose restart [service-name]
```

### Error de memoria
```bash
# Limpiar caché Docker
docker system prune -a

# Reiniciar servicios
docker-compose down && docker-compose up -d
```

### KV Cache con problemas
```bash
# Limpiar caché
python bulk/core/ultra_adaptive_kv_cache_cli.py clear-cache

# Verificar health
python bulk/core/ultra_adaptive_kv_cache_cli.py health
```

## 📚 Siguientes Pasos

1. **Leer README principal**: [`README.md`](README.md)
2. **Explorar BUL**: [`bulk/README.md`](bulk/README.md)
3. **KV Cache Docs**: [`bulk/core/README_ULTRA_ADAPTIVE_KV_CACHE.md`](bulk/core/README_ULTRA_ADAPTIVE_KV_CACHE.md)
4. **API Documentation**: http://localhost:8000/docs

---

**¡Listo para empezar!** 🚀

