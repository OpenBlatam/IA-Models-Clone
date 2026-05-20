# 🚀 Guía de Uso Avanzado - BUL System

## 📚 Índice

1. [Optimización Avanzada del KV Cache](#optimización-avanzada)
2. [Integración con Otros Sistemas](#integración)
3. [Patrones de Uso Avanzados](#patrones)
4. [Tuning de Rendimiento](#tuning)
5. [Escalabilidad y Producción](#escalabilidad)

## ⚡ Optimización Avanzada del KV Cache

### Configuración de Alto Rendimiento

```python
from bulk.core.ultra_adaptive_kv_cache_engine import UltraAdaptiveKVCacheEngine, KVCacheConfig, CacheStrategy
import torch

# Configuración para máximo rendimiento
high_perf_config = KVCacheConfig(
    num_heads=16,
    head_dim=128,
    max_tokens=8192,
    block_size=256,
    cache_strategy=CacheStrategy.ADAPTIVE,
    use_compression=False,  # Desactivar compresión para velocidad
    use_quantization=False,
    pin_memory=True,
    non_blocking=True,
    dtype=torch.float16,  # Usar FP16 para velocidad
    enable_prefetch=True,
    prefetch_size=8,  # Prefetch más agresivo
    adaptive_compression=False,
    enable_profiling=False  # Desactivar profiling en producción
)

engine = UltraAdaptiveKVCacheEngine(high_perf_config)
```

### Configuración para Memoria Limitada

```python
# Configuración optimizada para memoria
memory_efficient_config = KVCacheConfig(
    max_tokens=2048,  # Reducir tamaño
    block_size=64,    # Bloques más pequeños
    cache_strategy=CacheStrategy.PAGED,
    use_compression=True,
    compression_ratio=0.2,  # Compresión agresiva
    use_quantization=True,
    quantization_bits=4,  # Cuantización agresiva
    max_memory_mb=4096,  # Límite de memoria
    enable_gc=True,
    gc_threshold=0.7  # GC más agresivo
)

engine = UltraAdaptiveKVCacheEngine(memory_efficient_config)
```

### Auto-Tuning Continuo

```python
# Habilitar auto-tuning continuo
async def setup_auto_tuning(engine):
    # El engine ajustará parámetros automáticamente
    await engine.auto_tune_continuous()
    
    # Obtener recomendaciones
    recommendations = await engine.get_tuning_recommendations()
    print(f"Recomendaciones: {recommendations}")
```

### Prefetching Inteligente

```python
from bulk.core.ultra_adaptive_kv_cache_advanced_features import RequestPrefetcher

# Crear prefetcher
prefetcher = RequestPrefetcher(engine)

# Iniciar prefetching basado en patrones
prefetcher.start()

# Prefetch basado en sesión
await prefetcher.prefetch_by_session('user_123')

# Prefetch basado en patrón
pattern = {'business_area': 'marketing', 'doc_type': 'strategy'}
await prefetcher.prefetch_by_pattern(pattern)
```

## 🔗 Integración

### Integración con FastAPI

```python
from fastapi import FastAPI, BackgroundTasks
from bulk.core.ultra_adaptive_kv_cache_engine import TruthGPTIntegration
from bulk.core.ultra_adaptive_kv_cache_integration import FastAPIMiddleware

app = FastAPI()

# Crear engine
engine = TruthGPTIntegration.create_engine_for_truthgpt()

# Agregar middleware
app.add_middleware(FastAPIMiddleware, engine=engine)

@app.post("/process")
async def process_request(request: dict, background_tasks: BackgroundTasks):
    # Procesar con caché automático
    result = await engine.process_request(request)
    
    # Tareas en background para optimización
    background_tasks.add_task(engine.optimize_cache_async)
    
    return result
```

### Integración con Celery

```python
from celery import Celery
from bulk.core.ultra_adaptive_kv_cache_engine import TruthGPTIntegration

celery_app = Celery('bul')

engine = TruthGPTIntegration.create_engine_for_truthgpt()

@celery_app.task
async def process_document(query: str):
    result = await engine.process_request({
        'text': query,
        'max_length': 500,
        'session_id': celery_app.current_task.request.id
    })
    return result
```

### Circuit Breaker Pattern

```python
from bulk.core.ultra_adaptive_kv_cache_integration import CircuitBreaker

# Crear circuit breaker
circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30,
    expected_exception=Exception
)

@circuit_breaker
async def process_with_circuit_breaker(request):
    return await engine.process_request(request)
```

## 🎯 Patrones de Uso Avanzados

### Batch Processing Optimizado

```python
# Procesar múltiples requests en batch
requests = [
    {'text': f'Query {i}', 'max_length': 100}
    for i in range(100)
]

# Procesar con deduplicación y optimización
results = await engine.process_batch_optimized(
    requests,
    deduplicate=True,
    prioritize=True,
    batch_size=10
)
```

### Streaming de Respuestas

```python
# Crear stream
stream = await engine.create_stream('stream_123')

# Procesar y enviar chunks
async for chunk in engine.stream_response(request):
    await stream.put(chunk)
    # Enviar chunk al cliente inmediatamente

# Cerrar stream
await engine.close_stream('stream_123')
```

### Priority Queue

```python
from bulk.core.ultra_adaptive_kv_cache_advanced_features import PriorityQueue, Priority

# Crear priority queue
priority_queue = PriorityQueue(engine)

# Agregar requests con prioridad
await priority_queue.add(
    request={'text': 'Urgent task'},
    priority=Priority.CRITICAL,
    deadline=time.time() + 60
)

# Procesar por prioridad
results = await priority_queue.process_batch()
```

### Multi-Tenant con Aislamiento

```python
# Configurar multi-tenant
config.multi_tenant = True
config.tenant_isolation = True

engine = UltraAdaptiveKVCacheEngine(config)

# Procesar para diferentes tenants
result_tenant1 = await engine.process_kv(
    key, value, tenant_id='tenant_1'
)

result_tenant2 = await engine.process_kv(
    key, value, tenant_id='tenant_2'
)
```

## 🔧 Tuning de Rendimiento

### Análisis de Rendimiento

```python
from bulk.core.ultra_adaptive_kv_cache_analytics import PerformanceAnalyzer

analyzer = PerformanceAnalyzer(engine)

# Análisis completo
report = analyzer.generate_performance_report()

# Análisis de cuello de botella
bottlenecks = analyzer.identify_bottlenecks()

# Recomendaciones
recommendations = analyzer.get_optimization_recommendations()
```

### Benchmarking

```python
from bulk.core.ultra_adaptive_kv_cache_benchmark import BenchmarkRunner

runner = BenchmarkRunner(engine)

# Benchmark completo
results = await runner.run_comprehensive_benchmark(
    duration_seconds=300,
    concurrent_requests=50
)

# Benchmark específico
throughput = await runner.benchmark_throughput(
    requests_per_second=100,
    duration=60
)
```

### Profiling Avanzado

```python
# Habilitar profiling detallado
config.enable_profiling = True

engine = UltraAdaptiveKVCacheEngine(config)

# Perfil de operación específica
with engine.profile_operation('process_kv'):
    result = await engine.process_kv(key, value)

# Obtener estadísticas de profiling
stats = engine.get_profiling_stats()
print(f"Operación más lenta: {stats['slowest_operation']}")
print(f"Tiempo promedio: {stats['avg_time']}ms")
```

## 📈 Escalabilidad y Producción

### Configuración Distribuida

```python
# Configuración para múltiples nodos
config.enable_distributed = True
config.distributed_backend = "nccl"  # Para GPU
# o "gloo" para CPU

engine = UltraAdaptiveKVCacheEngine(config)

# Sincronizar entre nodos
await engine.sync_to_all_nodes(key, value)
```

### Edge Computing

```python
# Sincronizar con edge nodes
await engine.sync_to_edge(
    key='cache_key',
    value=cached_value,
    target_nodes=['edge-1', 'edge-2']
)

# Obtener del edge más cercano
value = await engine.get_from_nearest_edge(
    key='cache_key',
    client_location='us-east-1'
)
```

### Auto-Scaling

```python
# Configurar auto-scaling
from bulk.core.ultra_adaptive_kv_cache_monitor import AutoScaler

scaler = AutoScaler(
    engine,
    min_workers=4,
    max_workers=32,
    scale_up_threshold=0.8,
    scale_down_threshold=0.3
)

await scaler.start()
```

### Backup y Restauración Automática

```python
from bulk.core.ultra_adaptive_kv_cache_backup import ScheduledBackup

# Backup programado
backup_mgr = BackupManager(engine)
scheduler = ScheduledBackup(
    backup_mgr,
    interval_hours=6,
    keep_backups=10,
    compress=True
)

await scheduler.start()
```

## 🛡️ Seguridad Avanzada

### WAF (Web Application Firewall)

```python
from bulk.core.ultra_adaptive_kv_cache_security import WAFRules

waf = WAFRules(
    block_sql_injection=True,
    block_xss=True,
    block_path_traversal=True,
    rate_limit_per_ip=100  # requests/min
)

secure_engine = SecureEngineWrapper(
    engine,
    waf_rules=waf
)
```

### HMAC Validation

```python
secure_engine = SecureEngineWrapper(
    engine,
    enable_hmac=True,
    hmac_secret='your-secret-key'
)

# Request debe incluir signature
result = await secure_engine.process_request_secure(
    request,
    signature='hmac_signature',
    timestamp=time.time()
)
```

## 📊 Monitoring y Alertas

### Prometheus Metrics

```python
from bulk.core.ultra_adaptive_kv_cache_prometheus import PrometheusMetrics

metrics = PrometheusMetrics(engine)

# Iniciar servidor de métricas
await metrics.start_server(port=9090)

# Métricas custom
metrics.record_custom_metric('custom_operation', 1.5)
```

### Alertas Personalizadas

```python
from bulk.core.ultra_adaptive_kv_cache_monitor import AlertManager

alert_manager = AlertManager(engine)

# Configurar alertas
alert_manager.add_alert(
    name='high_memory',
    condition=lambda stats: stats['memory_usage'] > 0.9,
    action=lambda: logger.warning('High memory usage!')
)

alert_manager.add_alert(
    name='low_hit_rate',
    condition=lambda stats: stats['hit_rate'] < 0.5,
    action=lambda: logger.warning('Low cache hit rate!')
)

await alert_manager.start_monitoring()
```

## 🔄 Optimización Continua

### Machine Learning para Optimización

```python
# El engine aprende patrones automáticamente
await engine.enable_ml_optimization()

# Entrenar con datos históricos
await engine.train_optimization_model(historical_data)

# El engine ajustará parámetros automáticamente
# basándose en patrones aprendidos
```

### A/B Testing de Configuraciones

```python
from bulk.core.ultra_adaptive_kv_cache_optimizer import ABTesting

ab_test = ABTesting(engine)

# Probar dos configuraciones
config_a = KVCacheConfig(cache_size=8192)
config_b = KVCacheConfig(cache_size=16384)

results = await ab_test.compare_configs(
    config_a, config_b,
    duration_minutes=60,
    traffic_split=0.5
)

print(f"Mejor configuración: {results['winner']}")
```

## 📝 Ejemplos Completos

### Sistema Completo de Producción

```python
import asyncio
from bulk.core.ultra_adaptive_kv_cache_engine import (
    UltraAdaptiveKVCacheEngine,
    KVCacheConfig,
    CacheStrategy
)
from bulk.core.ultra_adaptive_kv_cache_security import SecureEngineWrapper
from bulk.core.ultra_adaptive_kv_cache_monitor import PerformanceMonitor
from bulk.core.ultra_adaptive_kv_cache_prometheus import PrometheusMetrics

async def setup_production_system():
    # 1. Configuración
    config = KVCacheConfig(
        max_tokens=16384,
        cache_strategy=CacheStrategy.ADAPTIVE,
        use_compression=True,
        enable_persistence=True,
        enable_prefetch=True
    )
    
    # 2. Crear engine
    engine = UltraAdaptiveKVCacheEngine(config)
    
    # 3. Seguridad
    secure_engine = SecureEngineWrapper(
        engine,
        enable_sanitization=True,
        enable_rate_limiting=True
    )
    
    # 4. Monitoreo
    monitor = PerformanceMonitor(secure_engine)
    await monitor.start_monitoring()
    
    # 5. Métricas
    metrics = PrometheusMetrics(secure_engine)
    await metrics.start_server(port=9090)
    
    # 6. Auto-tuning
    asyncio.create_task(engine.auto_tune_continuous())
    
    return secure_engine

# Uso
if __name__ == "__main__":
    engine = asyncio.run(setup_production_system())
```

---

**Para más información, consulta:**
- [README Principal](../README.md)
- [README KV Cache](core/README_ULTRA_ADAPTIVE_KV_CACHE.md)
- [Documentación Completa](core/ULTRA_ADAPTIVE_KV_CACHE_COMPLETE_FEATURES.md)



