# 🚀 Características Avanzadas - KV Cache

## Nuevas Funcionalidades Agregadas

### 1. Integración con Transformers

#### `TransformersKVCache`
Integración seamless con HuggingFace Transformers:

```python
from kv_cache import TransformersKVCache, KVCacheConfig
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

config = KVCacheConfig(max_tokens=4096)
cache = TransformersKVCache(config, model=model, tokenizer=tokenizer)

# Auto-configuración desde el modelo
# Detecta automáticamente:
# - Número de heads de atención
# - Dimensión de head
# - Max sequence length

# Warmup del cache
cache.warmup(num_samples=100, seq_len=128)

# Estadísticas
stats = cache.get_cache_stats()
```

#### `ModelCacheWrapper`
Wrapper para agregar caching a cualquier modelo:

```python
from kv_cache import ModelCacheWrapper, KVCacheConfig

wrapper = ModelCacheWrapper(model, cache_config)
wrapper.enable_cache()

# Usar modelo con cache automático
outputs = model(input_ids)

# Ver estadísticas
stats = wrapper.get_cache_stats()
```

### 2. Monitoring y Observabilidad

#### `CacheMonitor`
Monitoreo en tiempo real con alertas:

```python
from kv_cache import CacheMonitor

monitor = CacheMonitor(
    window_size=1000,
    alert_thresholds={
        "hit_rate": 0.3,  # Alerta si hit rate < 30%
        "memory_usage_mb": 8000,  # Alerta si memoria > 8GB
    }
)

# Registrar operaciones
monitor.record_operation(operation_time)

# Actualizar métricas
stats = cache.get_stats()
metrics = monitor.update_metrics(stats)

# Obtener alertas
alerts = monitor.get_alerts()

# Resumen
summary = monitor.get_summary()
```

#### `MetricsExporter`
Exportación de métricas a diferentes backends:

```python
from kv_cache.monitoring import MetricsExporter

# Exportar a dict
metrics_dict = MetricsExporter.export_to_dict(monitor)

# Exportar a formato Prometheus
prometheus_metrics = MetricsExporter.export_to_prometheus_format(monitor)
```

### 3. Persistencia de Cache

#### `CachePersistence`
Guardar y cargar estado del cache:

```python
from kv_cache import CachePersistence

persistence = CachePersistence("./cache_checkpoints")

# Guardar cache
persistence.save_cache(cache, "cache_state.pkl")

# Cargar cache
new_cache = BaseKVCache(config)
persistence.load_cache(new_cache, "cache_state.pkl")
```

#### Funciones de checkpoint:

```python
from kv_cache.persistence import save_cache_checkpoint, load_cache_checkpoint

# Guardar checkpoint
save_cache_checkpoint(cache, "./checkpoints", step=1000)

# Cargar checkpoint
load_cache_checkpoint(cache, "./checkpoints", step=1000)

# Cargar último checkpoint
load_cache_checkpoint(cache, "./checkpoints", step=None)
```

### 4. Ejemplos de Uso

El archivo `examples.py` contiene ejemplos completos:

- `example_basic_usage()`: Uso básico
- `example_adaptive_cache()`: Cache adaptativo
- `example_paged_cache()`: Cache paginado
- `example_with_profiling()`: Con profiling
- `example_monitoring()`: Con monitoring
- `example_persistence()`: Con persistencia

## Casos de Uso Completos

### Caso 1: Inference con Transformers

```python
from transformers import AutoModelForCausalLM
from kv_cache import TransformersKVCache, KVCacheConfig

model = AutoModelForCausalLM.from_pretrained("gpt2")
config = KVCacheConfig(
    max_tokens=4096,
    use_quantization=True,
    enable_profiling=True,
)

cache = TransformersKVCache(config, model=model)

# Generar con cache
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs, use_cache=True)

# Ver estadísticas
stats = cache.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

### Caso 2: Training con Monitoring

```python
from kv_cache import BaseKVCache, CacheMonitor, KVCacheConfig

config = KVCacheConfig(max_tokens=2048)
cache = BaseKVCache(config)
monitor = CacheMonitor()

for epoch in range(num_epochs):
    for batch in dataloader:
        # Training step
        ...
        
        # Monitorear cache cada 100 steps
        if step % 100 == 0:
            stats = cache.get_stats()
            metrics = monitor.update_metrics(stats)
            
            if metrics.hit_rate < 0.3:
                print("Warning: Low cache hit rate!")
```

### Caso 3: Production con Persistencia

```python
from kv_cache import BaseKVCache, CachePersistence, save_cache_checkpoint

cache = BaseKVCache(config)

# Durante inference
for request in requests:
    process_with_cache(cache, request)
    
    # Guardar checkpoint cada 1000 requests
    if request_id % 1000 == 0:
        save_cache_checkpoint(cache, "./checkpoints", step=request_id)
```

## Beneficios de las Nuevas Características

1. **Integración Fácil**: Integración directa con Transformers
2. **Observabilidad**: Monitoring completo con alertas
3. **Recuperación**: Persistencia para checkpoints y recovery
4. **Producción-Ready**: Herramientas para deployment

## Próximos Pasos

- Tests unitarios
- Documentación de API completa
- Benchmarks de rendimiento
- Integración con wandb/tensorboard

---

**Versión**: 2.5.0 (Advanced Features)  
**Fecha**: 2024



