# ❓ Preguntas Frecuentes (FAQ) - Blatam Academy Features

## 📋 Tabla de Contenidos

- [General](#general)
- [KV Cache Engine](#kv-cache-engine)
- [Configuración](#configuración)
- [Rendimiento](#rendimiento)
- [Troubleshooting](#troubleshooting)
- [Deployment](#deployment)
- [Desarrollo](#desarrollo)

## 🌐 General

### ¿Qué es Blatam Academy Features?

Blatam Academy Features es un ecosistema completo de servicios de IA y automatización empresarial que integra más de 40 módulos especializados para generación de contenido, procesamiento de documentos, y automatización de negocios.

### ¿Qué servicios incluye?

- **Integration System** (8000): API Gateway principal
- **Content Redundancy Detector** (8001): Detección de contenido duplicado
- **BUL** (8002): Generación de documentos empresariales
- **Gamma App** (8003): Generación de contenido
- **Business Agents** (8004): Agentes de negocio automatizados
- **Export IA** (8005): Exportación avanzada

### ¿Qué tecnologías usa?

- **Backend**: FastAPI, Python
- **Base de Datos**: PostgreSQL
- **Cache**: Redis + Ultra Adaptive KV Cache Engine
- **Infraestructura**: Docker, Docker Compose
- **Monitoring**: Prometheus, Grafana
- **Logging**: ELK Stack

## ⚡ KV Cache Engine

### ¿Qué es el Ultra Adaptive KV Cache Engine?

Es un sistema de caché de nivel empresarial optimizado para modelos transformer (especialmente TruthGPT) que mejora significativamente el rendimiento mediante técnicas avanzadas de caching, compresión, y optimización.

### ¿Qué estrategias de cache están disponibles?

- **LRU** (Least Recently Used): Evicta entradas menos usadas recientemente
- **LFU** (Least Frequently Used): Evicta entradas menos frecuentemente accedidas
- **Adaptive**: Ajusta automáticamente la estrategia basado en patrones
- **Paged**: Asignación de memoria por páginas
- **Compressed**: Con compresión
- **Quantized**: Con cuantización

### ¿Cómo elijo la mejor estrategia?

- **LRU**: Para patrones de acceso secuencial
- **LFU**: Para acceso aleatorio con repetición
- **Adaptive**: Para patrones mixtos (recomendado)
- **Compressed/Quantized**: Para ahorrar memoria

### ¿Cuál es el rendimiento típico del KV Cache?

- **Latencia P50**: <100ms
- **Latencia P95**: <500ms
- **Latencia P99**: <1s
- **Throughput**: 50-200 req/s (cached)
- **Cache Hit Rate**: 65-75% típico

### ¿Cómo funciona el prefetching?

El prefetching predice qué entradas se necesitarán próximamente y las carga proactivamente en memoria, reduciendo la latencia percibida.

### ¿El cache soporta multi-GPU?

Sí, el engine detecta automáticamente múltiples GPUs y distribuye la carga entre ellas para máximo rendimiento.

## ⚙️ Configuración

### ¿Cómo configuro el KV Cache para producción?

Usa la plantilla de configuración:

```bash
cp config/templates/production.env.template .env
cp config/templates/kv_cache_production.yaml config/kv_cache.yaml
```

Luego ajusta los valores según tus necesidades.

### ¿Qué variables de entorno son necesarias?

Ver [production.env.template](config/templates/production.env.template) para la lista completa. Las más importantes:

- `DATABASE_URL`: URL de PostgreSQL
- `REDIS_URL`: URL de Redis
- `KV_CACHE_MAX_TOKENS`: Tamaño máximo del cache
- `KV_CACHE_ENABLE_PERSISTENCE`: Habilitar persistencia

### ¿Cómo cambio la estrategia de cache?

```python
from ultra_adaptive_kv_cache_engine import KVCacheConfig, CacheStrategy

config = KVCacheConfig(cache_strategy=CacheStrategy.ADAPTIVE)
engine = UltraAdaptiveKVCacheEngine(config)
```

O vía variables de entorno:
```env
KV_CACHE_STRATEGY=adaptive
```

## 📊 Rendimiento

### ¿Cómo mejoro el rendimiento?

1. **Habilita compresión**: Reduce uso de memoria
2. **Ajusta `max_tokens`**: Balance entre memoria y rendimiento
3. **Usa prefetching**: Reduce latencia percibida
4. **Habilita persistencia**: Evita cold starts
5. **Usa multi-GPU**: Aumenta throughput

Ver [PERFORMANCE_TUNING.md](PERFORMANCE_TUNING.md) para más detalles.

### ¿Cuánta memoria necesita el KV Cache?

Depende de `max_tokens` y `compression_ratio`. Generalmente:
- **4096 tokens**: ~2-4 GB
- **8192 tokens**: ~4-8 GB
- **16384 tokens**: ~8-16 GB

Con compresión, puede reducirse a 30-70% del tamaño original.

### ¿Cómo monitoreo el rendimiento?

```bash
# CLI tool
python bulk/core/ultra_adaptive_kv_cache_cli.py stats

# Dashboard
python bulk/core/ultra_adaptive_kv_cache_cli.py monitor --dashboard

# Prometheus
http://localhost:9090
```

## 🐛 Troubleshooting

### El cache no está funcionando

1. Verifica que esté habilitado:
```python
config = KVCacheConfig()  # Por defecto está habilitado
```

2. Revisa logs:
```bash
docker-compose logs bul | grep cache
```

3. Verifica configuración:
```python
engine.validate_configuration()
```

### Alto uso de memoria

1. Reduce `max_tokens`
2. Habilita compresión: `use_compression=True`
3. Habilita cuantización: `use_quantization=True`
4. Reduce `compression_ratio`

### Baja tasa de cache hits

1. Verifica que los requests sean similares
2. Aumenta `max_tokens` si es posible
3. Revisa la estrategia de cache (considera `ADAPTIVE`)
4. Verifica que no estés limpiando el cache frecuentemente

### Errores de GPU

1. Verifica que CUDA esté disponible:
```python
import torch
print(torch.cuda.is_available())
```

2. Verifica drivers de GPU
3. Reduce `max_tokens` si falta memoria
4. Usa CPU fallback si no hay GPU

Ver [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md) para más.

## 🚀 Deployment

### ¿Cómo despliego en producción?

1. Usa el checklist: [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)
2. Configura variables de entorno
3. Ejecuta setup:
```bash
./scripts/setup_complete.sh
```
4. Verifica health:
```bash
./scripts/health_check.sh
```

### ¿Cómo escalo horizontalmente?

1. Aumenta réplicas en `docker-compose.yml`
2. Configura load balancer (Nginx)
3. Usa base de datos y Redis compartidos
4. Monitorea recursos

### ¿Cómo hago backup del cache?

```python
# Manual
engine.persist('/backup/cache.pt')

# Automático (configurar)
config = KVCacheConfig(
    enable_persistence=True,
    persistence_path='/backup/cache'
)
```

## 🛠️ Desarrollo

### ¿Cómo extiendo el KV Cache?

Ver [bulk/core/DEVELOPMENT_GUIDE.md](bulk/core/DEVELOPMENT_GUIDE.md) para guía completa.

Ejemplo rápido:
```python
from ultra_adaptive_kv_cache_engine import BaseKVCache

class CustomCache(BaseKVCache):
    def _evict(self):
        # Tu lógica personalizada
        pass
```

### ¿Cómo escribo tests?

Ver [bulk/core/TESTING_GUIDE.md](bulk/core/TESTING_GUIDE.md) para guía completa.

Ejemplo:
```python
import pytest
from ultra_adaptive_kv_cache_engine import UltraAdaptiveKVCacheEngine

@pytest.mark.asyncio
async def test_cache():
    engine = UltraAdaptiveKVCacheEngine(config)
    result = await engine.process_request({'text': 'test'})
    assert 'result' in result
```

### ¿Cómo contribuyo?

1. Fork el repositorio
2. Crea branch: `git checkout -b feature/mi-feature`
3. Escribe código y tests
4. Commit: `git commit -m "feat: mi feature"`
5. Push y crea Pull Request

Ver [CONTRIBUTING.md](CONTRIBUTING.md) para más detalles.

## 🔐 Seguridad

### ¿Cómo configuro autenticación?

Ver [SECURITY_GUIDE.md](SECURITY_GUIDE.md) para guía completa.

Básico:
```python
from fastapi import Depends
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/api/endpoint")
async def endpoint(token = Depends(security)):
    # Verificar token
    pass
```

### ¿Cómo habilito rate limiting?

```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/api/endpoint")
@limiter.limit("100/minute")
async def endpoint():
    pass
```

## 🔄 Migración

### ¿Cómo migro a una nueva versión?

Ver [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) para guía completa.

Pasos básicos:
1. Backup completo
2. Actualizar código
3. Migrar configuración
4. Migrar datos si es necesario
5. Verificar y testear

### ¿Es backward compatible?

Sí, el sistema mantiene compatibilidad hacia atrás y migra automáticamente cuando es posible.

---

**Más información:**
- [README Principal](README.md)
- [Índice de Documentación](DOCUMENTATION_INDEX.md)
- [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)

