# ⚡ Referencia Rápida - BUL System

## 🚀 Comandos Esenciales

### Iniciar Sistema
```bash
# Sistema completo
python main.py --mode full

# Solo procesador
python main.py --mode processor

# Solo API
python main.py --mode api --port 8002
```

### KV Cache CLI
```bash
# Estadísticas
python core/ultra_adaptive_kv_cache_cli.py stats

# Monitoreo
python core/ultra_adaptive_kv_cache_cli.py monitor --dashboard

# Health check
python core/ultra_adaptive_kv_cache_cli.py health

# Limpiar caché
python core/ultra_adaptive_kv_cache_cli.py clear-cache

# Test
python core/ultra_adaptive_kv_cache_cli.py test --text "Hello"
```

## 📡 API Endpoints Rápidos

### Submit Query
```bash
curl -X POST http://localhost:8002/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Create marketing strategy", "priority": 1}'
```

### Get Status
```bash
curl http://localhost:8002/task/{task_id}/status
```

### Get Documents
```bash
curl http://localhost:8002/task/{task_id}/documents
```

## 🐍 Código Python Rápido

### Setup Básico
```python
from bulk.core.ultra_adaptive_kv_cache_engine import TruthGPTIntegration

engine = TruthGPTIntegration.create_engine_for_truthgpt()
result = await engine.process_request({'text': 'Your query'})
```

### Con Seguridad
```python
from bulk.core.ultra_adaptive_kv_cache_security import SecureEngineWrapper

secure_engine = SecureEngineWrapper(engine, enable_sanitization=True)
result = await secure_engine.process_request_secure(request, api_key="key")
```

### Batch Processing
```python
results = await engine.process_batch_optimized(requests, batch_size=10)
```

## ⚙️ Configuraciones Rápidas

### Preset Production
```python
from bulk.core.ultra_adaptive_kv_cache_config_manager import ConfigPreset
ConfigPreset.apply_preset(engine, 'production')
```

### Preset High Performance
```python
ConfigPreset.apply_preset(engine, 'high_performance')
```

### Preset Memory Efficient
```python
ConfigPreset.apply_preset(engine, 'memory_efficient')
```

## 📊 Métricas Clave

```python
stats = engine.get_stats()
print(f"Hit rate: {stats['hit_rate']}")
print(f"Memory: {stats['memory_usage']}%")
print(f"Latency P50: {stats['p50_latency']}ms")
```

## 🔧 Troubleshooting Rápido

### Alto uso de memoria
```python
ConfigPreset.apply_preset(engine, 'memory_efficient')
```

### Bajo rendimiento
```python
ConfigPreset.apply_preset(engine, 'high_performance')
```

### Limpiar caché
```bash
python core/ultra_adaptive_kv_cache_cli.py clear-cache
```

---

**Más información:**
- [README Completo](README.md)
- [Guía Avanzada](ADVANCED_USAGE_GUIDE.md)
- [Ejemplos](EXAMPLES.md)



