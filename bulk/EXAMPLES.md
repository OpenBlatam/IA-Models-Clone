# 💡 Ejemplos Prácticos - BUL System

## 📋 Índice

1. [Ejemplos Básicos](#ejemplos-básicos)
2. [Ejemplos Intermedios](#ejemplos-intermedios)
3. [Ejemplos Avanzados](#ejemplos-avanzados)
4. [Ejemplos de Integración](#ejemplos-de-integración)
5. [Ejemplos de Producción](#ejemplos-de-producción)

## 🟢 Ejemplos Básicos

### Ejemplo 1: Generación Simple de Documento

```python
from bulk.core.ultra_adaptive_kv_cache_engine import TruthGPTIntegration

# Crear engine
engine = TruthGPTIntegration.create_engine_for_truthgpt()

# Generar documento simple
result = await engine.process_request({
    'text': 'Create a marketing strategy for a new product',
    'max_length': 500,
    'temperature': 0.7
})

print(result['response']['text'])
```

### Ejemplo 2: Verificar Estado del Sistema

```python
from bulk.core.ultra_adaptive_kv_cache_cli import CLI

cli = CLI()

# Ver estadísticas
stats = await cli.cmd_stats(None)
print(f"Cache hit rate: {stats['hit_rate']}")
print(f"Memory usage: {stats['memory_usage']}%")

# Health check
health = await cli.cmd_health(None)
print(f"Status: {health['status']}")
```

### Ejemplo 3: Limpiar Caché

```python
# Limpiar caché manualmente
await engine.clear()

# O usar CLI
python bulk/core/ultra_adaptive_kv_cache_cli.py clear-cache
```

## 🟡 Ejemplos Intermedios

### Ejemplo 4: Batch Processing con Deduplicación

```python
# Lista de requests
requests = [
    {'text': 'Marketing strategy', 'business_area': 'marketing'},
    {'text': 'Sales process', 'business_area': 'sales'},
    {'text': 'Marketing strategy', 'business_area': 'marketing'},  # Duplicado
]

# Procesar con deduplicación
results = await engine.process_batch_optimized(
    requests,
    batch_size=10,
    deduplicate=True,  # Elimina duplicados
    prioritize=True
)

for i, result in enumerate(results):
    print(f"Document {i+1}: {result['response']['text'][:100]}...")
```

### Ejemplo 5: Configuración Personalizada

```python
from bulk.core.ultra_adaptive_kv_cache_engine import (
    UltraAdaptiveKVCacheEngine,
    KVCacheConfig,
    CacheStrategy
)

# Crear configuración personalizada
config = KVCacheConfig(
    max_tokens=8192,
    cache_strategy=CacheStrategy.ADAPTIVE,
    use_compression=True,
    compression_ratio=0.3,
    enable_prefetch=True,
    prefetch_size=4
)

# Crear engine con configuración
engine = UltraAdaptiveKVCacheEngine(config)
```

### Ejemplo 6: Monitoreo en Tiempo Real

```python
from bulk.core.ultra_adaptive_kv_cache_monitor import PerformanceMonitor

# Crear monitor
monitor = PerformanceMonitor(engine, check_interval=5.0)

# Iniciar monitoreo
await monitor.start_monitoring()

# Obtener estado actual
status = monitor.get_current_status()
print(f"Performance: {status['performance']}")
print(f"Alerts: {status['alerts']}")
```

### Ejemplo 7: Usar Presets

```python
from bulk.core.ultra_adaptive_kv_cache_config_manager import ConfigPreset

# Aplicar preset de producción
ConfigPreset.apply_preset(engine, 'production')

# Otros presets disponibles:
# - 'development'
# - 'high_performance'
# - 'memory_efficient'
# - 'bulk_processing'
```

## 🔴 Ejemplos Avanzados

### Ejemplo 8: Sistema Completo con Seguridad

```python
from bulk.core.ultra_adaptive_kv_cache_engine import TruthGPTIntegration
from bulk.core.ultra_adaptive_kv_cache_security import SecureEngineWrapper
from bulk.core.ultra_adaptive_kv_cache_monitor import PerformanceMonitor
from bulk.core.ultra_adaptive_kv_cache_prometheus import PrometheusMetrics

# Setup completo
async def setup_complete_system():
    # 1. Crear engine base
    engine = TruthGPTIntegration.create_engine_for_truthgpt()
    
    # 2. Agregar seguridad
    secure_engine = SecureEngineWrapper(
        engine,
        enable_sanitization=True,
        enable_rate_limiting=True,
        rate_limit_per_minute=100,
        enable_access_control=True
    )
    
    # 3. Agregar monitoreo
    monitor = PerformanceMonitor(secure_engine)
    await monitor.start_monitoring()
    
    # 4. Agregar métricas Prometheus
    metrics = PrometheusMetrics(secure_engine)
    await metrics.start_server(port=9090)
    
    return secure_engine

# Uso
system = await setup_complete_system()
result = await system.process_request_secure(
    request,
    client_ip="192.168.1.100",
    api_key="valid-api-key"
)
```

### Ejemplo 9: Streaming de Respuestas

```python
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/stream")
async def stream_generation(websocket: WebSocket):
    await websocket.accept()
    
    # Crear stream
    stream_id = "stream_123"
    await engine.create_stream(stream_id)
    
    # Procesar y enviar chunks
    async for chunk in engine.stream_response({
        'text': 'Generate long document',
        'max_length': 2000
    }):
        await websocket.send_json({
            'chunk': chunk,
            'type': 'content'
        })
    
    # Cerrar stream
    await engine.close_stream(stream_id)
```

### Ejemplo 10: Priority Queue con Deadlines

```python
from bulk.core.ultra_adaptive_kv_cache_advanced_features import (
    PriorityQueue,
    Priority
)
import time

# Crear priority queue
priority_queue = PriorityQueue(engine)

# Agregar requests con diferentes prioridades
await priority_queue.add(
    request={'text': 'Urgent task'},
    priority=Priority.CRITICAL,
    deadline=time.time() + 60  # 1 minuto
)

await priority_queue.add(
    request={'text': 'Normal task'},
    priority=Priority.NORMAL,
    deadline=time.time() + 300  # 5 minutos
)

# Procesar por prioridad
results = await priority_queue.process_batch()
```

### Ejemplo 11: Auto-Tuning Continuo

```python
# Habilitar auto-tuning
await engine.auto_tune_continuous()

# El engine ajustará automáticamente:
# - Cache size
# - Compression ratio
# - Prefetch size
# - Worker count

# Obtener recomendaciones actuales
recommendations = await engine.get_tuning_recommendations()
print(f"Recommended cache size: {recommendations['cache_size']}")
print(f"Recommended workers: {recommendations['num_workers']}")
```

### Ejemplo 12: Multi-Tenant con Aislamiento

```python
from bulk.core.ultra_adaptive_kv_cache_engine import (
    UltraAdaptiveKVCacheEngine,
    KVCacheConfig
)

# Configurar multi-tenant
config = KVCacheConfig(
    multi_tenant=True,
    tenant_isolation=True,
    max_tokens=16384
)

engine = UltraAdaptiveKVCacheEngine(config)

# Procesar para diferentes tenants
tenant_a_result = await engine.process_kv(
    key=key_tensor,
    value=value_tensor,
    tenant_id='company_a'
)

tenant_b_result = await engine.process_kv(
    key=key_tensor,
    value=value_tensor,
    tenant_id='company_b'
)

# Los caches están completamente aislados
```

## 🔗 Ejemplos de Integración

### Ejemplo 13: Integración con FastAPI

```python
from fastapi import FastAPI, BackgroundTasks
from bulk.core.ultra_adaptive_kv_cache_engine import TruthGPTIntegration
from bulk.core.ultra_adaptive_kv_cache_integration import FastAPIMiddleware

app = FastAPI()

# Crear engine
engine = TruthGPTIntegration.create_engine_for_truthgpt()

# Agregar middleware
app.add_middleware(FastAPIMiddleware, engine=engine)

@app.post("/generate")
async def generate_document(request: dict, background_tasks: BackgroundTasks):
    # Procesar con caché automático
    result = await engine.process_request(request)
    
    # Optimizar en background
    background_tasks.add_task(engine.optimize_cache_async)
    
    return result

@app.get("/cache/stats")
async def get_cache_stats():
    return engine.get_stats()
```

### Ejemplo 14: Integración con Celery

```python
from celery import Celery
from bulk.core.ultra_adaptive_kv_cache_engine import TruthGPTIntegration

celery_app = Celery('bul')

# Crear engine (compartido entre workers)
engine = TruthGPTIntegration.create_engine_for_truthgpt()

@celery_app.task(bind=True, max_retries=3)
async def process_document_task(self, query: str):
    try:
        result = await engine.process_request({
            'text': query,
            'max_length': 500,
            'session_id': self.request.id
        })
        return result
    except Exception as e:
        # Retry con exponential backoff
        raise self.retry(exc=e, countdown=2 ** self.request.retries)

# Uso
task = process_document_task.delay("Generate marketing strategy")
result = task.get()
```

### Ejemplo 15: Circuit Breaker Pattern

```python
from bulk.core.ultra_adaptive_kv_cache_integration import CircuitBreaker

# Crear circuit breaker
circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30,
    expected_exception=Exception
)

@circuit_breaker
async def process_with_resilience(request):
    return await engine.process_request(request)

# Uso automático con protección
try:
    result = await process_with_resilience(request)
except CircuitBreakerOpenError:
    # Usar fallback
    result = await fallback_process(request)
```

## 🏭 Ejemplos de Producción

### Ejemplo 16: Setup de Producción Completo

```python
import asyncio
from bulk.core.ultra_adaptive_kv_cache_engine import (
    UltraAdaptiveKVCacheEngine,
    KVCacheConfig,
    CacheStrategy
)
from bulk.core.ultra_adaptive_kv_cache_security import SecureEngineWrapper
from bulk.core.ultra_adaptive_kv_cache_monitor import (
    PerformanceMonitor,
    AlertManager
)
from bulk.core.ultra_adaptive_kv_cache_prometheus import PrometheusMetrics
from bulk.core.ultra_adaptive_kv_cache_backup import ScheduledBackup, BackupManager

async def setup_production_system():
    # 1. Configuración de producción
    config = KVCacheConfig(
        max_tokens=16384,
        cache_strategy=CacheStrategy.ADAPTIVE,
        use_compression=True,
        compression_ratio=0.3,
        enable_persistence=True,
        persistence_path='/data/cache',
        enable_prefetch=True,
        prefetch_size=8,
        enable_profiling=False  # Desactivar en producción
    )
    
    # 2. Crear engine
    engine = UltraAdaptiveKVCacheEngine(config)
    
    # 3. Seguridad
    secure_engine = SecureEngineWrapper(
        engine,
        enable_sanitization=True,
        enable_rate_limiting=True,
        rate_limit_per_minute=1000,
        enable_access_control=True,
        enable_hmac=True,
        hmac_secret=os.getenv('HMAC_SECRET')
    )
    
    # 4. Monitoreo
    monitor = PerformanceMonitor(secure_engine, check_interval=5.0)
    await monitor.start_monitoring()
    
    # 5. Alertas
    alert_manager = AlertManager(secure_engine)
    alert_manager.add_alert(
        name='high_memory',
        condition=lambda s: s['memory_usage'] > 0.9,
        action=lambda: send_slack_alert('High memory usage!')
    )
    await alert_manager.start()
    
    # 6. Métricas Prometheus
    metrics = PrometheusMetrics(secure_engine)
    await metrics.start_server(port=9090)
    
    # 7. Backup automático
    backup_mgr = BackupManager(secure_engine)
    scheduler = ScheduledBackup(
        backup_mgr,
        interval_hours=6,
        keep_backups=10,
        compress=True
    )
    await scheduler.start()
    
    # 8. Auto-tuning
    asyncio.create_task(secure_engine.auto_tune_continuous())
    
    return secure_engine

# Inicializar
production_engine = asyncio.run(setup_production_system())
```

### Ejemplo 17: Load Testing

```python
from bulk.core.ultra_adaptive_kv_cache_benchmark import BenchmarkRunner
import asyncio

async def load_test():
    runner = BenchmarkRunner(engine)
    
    # Benchmark completo
    results = await runner.run_comprehensive_benchmark(
        duration_seconds=300,  # 5 minutos
        concurrent_requests=50,
        request_rate=100  # req/s
    )
    
    print(f"Throughput: {results['throughput']} req/s")
    print(f"P50 latency: {results['p50_latency']}ms")
    print(f"P95 latency: {results['p95_latency']}ms")
    print(f"P99 latency: {results['p99_latency']}ms")
    print(f"Error rate: {results['error_rate']}%")

# Ejecutar
asyncio.run(load_test())
```

### Ejemplo 18: Analytics y Reportes

```python
from bulk.core.ultra_adaptive_kv_cache_analytics import Analytics

analytics = Analytics(engine)

# Generar reporte completo
report = analytics.generate_detailed_report()

print(f"Total requests: {report['total_requests']}")
print(f"Cache hit rate: {report['cache_hit_rate']}")
print(f"Average latency: {report['avg_latency']}ms")
print(f"Cost analysis: {report['cost_analysis']}")

# Análisis de uso
usage_analysis = analytics.analyze_usage_patterns()
print(f"Peak hours: {usage_analysis['peak_hours']}")
print(f"Popular queries: {usage_analysis['popular_queries']}")

# Calcular costos
costs = analytics.calculate_cost(
    tokens_processed=report['total_tokens'],
    cost_per_1k_tokens=0.01
)
print(f"Total cost: ${costs['total_cost']}")
```

### Ejemplo 19: Configuración Dinámica

```python
from bulk.core.ultra_adaptive_kv_cache_config_manager import ConfigManager

# Crear config manager con watch file
config_manager = ConfigManager(
    engine,
    config_file='config.json',
    watch_file=True  # Auto-reload en cambios
)

# Actualizar configuración en runtime
await config_manager.update_config('cache_size', 32768)
await config_manager.update_config('num_workers', 16)

# El archivo se actualiza automáticamente
# Y otros procesos pueden recargar

# Recargar desde archivo
await config_manager.reload_from_file()
```

### Ejemplo 20: Distributed Cache

```python
# Configuración para múltiples nodos
config = KVCacheConfig(
    enable_distributed=True,
    distributed_backend="nccl",  # Para GPU
    max_tokens=16384
)

engine = UltraAdaptiveKVCacheEngine(config)

# Sincronizar entre nodos
await engine.sync_to_all_nodes(
    key='shared_cache_key',
    value=cached_value
)

# Obtener del nodo más cercano
value = await engine.get_from_nearest_node(
    key='shared_cache_key',
    current_node='node-1'
)

# Edge computing
await engine.sync_to_edge(
    key='cache_key',
    value=cached_value,
    target_nodes=['edge-us-east', 'edge-eu-west']
)
```

## 🎓 Ejemplos de Aprendizaje

### Ejemplo 21: Entrenar Modelo de Optimización

```python
# El engine aprende patrones automáticamente
await engine.enable_ml_optimization()

# Entrenar con datos históricos
historical_data = load_historical_requests()

await engine.train_optimization_model(
    data=historical_data,
    epochs=100
)

# El engine ajustará automáticamente parámetros
# basándose en patrones aprendidos
```

### Ejemplo 22: A/B Testing

```python
from bulk.core.ultra_adaptive_kv_cache_optimizer import ABTesting

ab_test = ABTesting(engine)

# Probar dos configuraciones
config_a = KVCacheConfig(
    cache_strategy=CacheStrategy.LRU,
    max_tokens=8192
)

config_b = KVCacheConfig(
    cache_strategy=CacheStrategy.ADAPTIVE,
    max_tokens=16384
)

# Comparar
results = await ab_test.compare_configs(
    config_a, config_b,
    duration_minutes=60,
    traffic_split=0.5,
    metrics=['latency', 'hit_rate', 'throughput', 'cost']
)

print(f"Winner: {results['winner']}")
print(f"Improvement: {results['improvement']}%")
print(f"Metrics: {results['metrics']}")
```

## 🔄 Ejemplos de Operaciones

### Ejemplo 23: Backup y Restauración

```python
from bulk.core.ultra_adaptive_kv_cache_backup import (
    BackupManager,
    ScheduledBackup
)

# Crear backup
backup_mgr = BackupManager(engine)
backup_path = backup_mgr.create_backup(
    compress=True,
    include_metadata=True
)

print(f"Backup created: {backup_path}")

# Restaurar backup
backup_mgr.restore_backup(backup_path)

# Backup programado
scheduler = ScheduledBackup(
    backup_mgr,
    interval_hours=6,
    keep_backups=10,
    compress=True
)

await scheduler.start()
```

### Ejemplo 24: Migración de Configuración

```python
# Exportar configuración actual
current_config = engine.config
config_json = json.dumps(asdict(current_config), indent=2)

with open('current_config.json', 'w') as f:
    f.write(config_json)

# Importar nueva configuración
with open('new_config.json', 'r') as f:
    new_config = json.load(f)

config_manager.update_config_from_dict(new_config)
```

---

**Para más ejemplos:**
- [Casos de Uso Reales](USE_CASES.md)
- [Guía de Uso Avanzado](ADVANCED_USAGE_GUIDE.md)
- [Mejores Prácticas](../BEST_PRACTICES.md)

