# Circuit Breaker - Características Avanzadas

## ✅ Mejoras Adicionales Implementadas

Se han implementado características avanzadas adicionales para el Circuit Breaker.

## 🎯 Nuevas Características

### 1. ✅ Bulk Operations Support

Permite procesar múltiples llamadas en lote con protección del circuit breaker.

**Método:** `call_bulk()`

**Ejemplo de uso:**
```python
# Procesar múltiples items
items = [
    ("user1", "data1"),
    ("user2", "data2"),
    ("user3", "data3")
]

# Procesar todos, continuar en errores
results = await breaker.call_bulk(
    api_function,
    items,
    stop_on_first_error=False
)
# results = [result1, result2, None]  # None para fallos

# Procesar hasta primer error
try:
    results = await breaker.call_bulk(
        api_function,
        items,
        stop_on_first_error=True
    )
except Exception as e:
    # Se detiene en primer error
    logger.error(f"Bulk operation failed: {e}")
```

### 2. ✅ Configuración Dinámica

Permite actualizar la configuración del circuit breaker en tiempo de ejecución.

**Método:** `update_config()`

**Ejemplo de uso:**
```python
# Actualizar configuración dinámicamente
updated = await breaker.update_config(
    failure_threshold=10,      # Aumentar threshold
    recovery_timeout=120.0,     # Aumentar timeout
    retry_enabled=True,         # Habilitar retry
    max_retries=5               # Más reintentos
)

# Resultado:
# {
#     "failure_threshold": {"old": 5, "new": 10},
#     "recovery_timeout": {"old": 60.0, "new": 120.0},
#     ...
# }
```

**Casos de uso:**
- Ajustar thresholds basado en condiciones del sistema
- Habilitar/deshabilitar features según carga
- Ajustar timeouts según latencia observada

### 3. ✅ Exportación de Métricas

Exporta métricas en formatos estándar para sistemas de monitoreo.

#### Prometheus Format

**Método:** `export_metrics_prometheus()`

```python
# Exportar métricas en formato Prometheus
prometheus_metrics = breaker.export_metrics_prometheus()

# Usar con prometheus_client
from prometheus_client import Gauge, Counter, Histogram

for metric_name, metric_data in prometheus_metrics.items():
    value = metric_data["value"]
    labels = metric_data["labels"]
    
    if "total" in metric_name or "successful" in metric_name:
        counter = Counter(metric_name, f"{metric_name} description")
        counter.labels(**labels).inc(value)
    elif "rate" in metric_name or "score" in metric_name:
        gauge = Gauge(metric_name, f"{metric_name} description")
        gauge.labels(**labels).set(value)
    elif "response_time" in metric_name:
        histogram = Histogram(metric_name, f"{metric_name} description")
        histogram.labels(**labels).observe(value)
```

#### StatsD Format

**Método:** `export_metrics_statsd()`

```python
# Exportar métricas en formato StatsD
statsd_metrics = breaker.export_metrics_statsd()

# Usar con statsd client
from statsd import StatsClient

statsd = StatsClient()

for metric in statsd_metrics:
    if metric["type"] == "counter":
        statsd.increment(metric["name"], metric["value"], tags=metric.get("tags", {}))
    elif metric["type"] == "gauge":
        statsd.gauge(metric["name"], metric["value"], tags=metric.get("tags", {}))
    elif metric["type"] == "histogram":
        statsd.histogram(metric["name"], metric["value"], tags=metric.get("tags", {}))
```

## 📚 Ejemplos Completos

### Ejemplo 1: Bulk Processing con Manejo de Errores

```python
async def process_users_bulk(breaker: CircuitBreaker, user_ids: List[str]):
    """Process multiple users with circuit breaker"""
    items = [(user_id,) for user_id in user_ids]
    
    results = await breaker.call_bulk(
        fetch_user_data,
        items,
        stop_on_first_error=False
    )
    
    # Procesar resultados
    successful = []
    failed = []
    
    for i, result in enumerate(results):
        if result is not None:
            successful.append((user_ids[i], result))
        else:
            failed.append(user_ids[i])
    
    return {
        "successful": successful,
        "failed": failed,
        "total": len(user_ids),
        "success_rate": len(successful) / len(user_ids)
    }
```

### Ejemplo 2: Configuración Dinámica Basada en Carga

```python
async def adjust_circuit_breaker_for_load(breaker: CircuitBreaker, current_load: float):
    """Adjust circuit breaker configuration based on system load"""
    if current_load > 0.9:  # High load
        # Aumentar thresholds para ser más tolerante
        await breaker.update_config(
            failure_threshold=10,  # Más tolerante
            recovery_timeout=180.0  # Más tiempo para recuperar
        )
    elif current_load < 0.3:  # Low load
        # Reducir thresholds para detectar problemas más rápido
        await breaker.update_config(
            failure_threshold=3,   # Más sensible
            recovery_timeout=30.0  # Recuperación más rápida
        )
```

### Ejemplo 3: Exportación Automática de Métricas

```python
import asyncio
from prometheus_client import start_http_server, Gauge, Counter, Histogram

# Crear métricas Prometheus
circuit_state = Gauge('circuit_breaker_state', 'Circuit breaker state', ['circuit_name', 'state'])
circuit_requests = Counter('circuit_breaker_requests_total', 'Total requests', ['circuit_name'])
circuit_health = Gauge('circuit_breaker_health_score', 'Health score', ['circuit_name'])

async def export_metrics_periodically(breaker: CircuitBreaker, interval: int = 60):
    """Export metrics to Prometheus periodically"""
    while True:
        metrics = breaker.export_metrics_prometheus()
        
        # Actualizar métricas Prometheus
        for metric_name, metric_data in metrics.items():
            value = metric_data["value"]
            labels = metric_data["labels"]
            
            if "state" in metric_name:
                circuit_state.labels(**labels).set(value)
            elif "requests_total" in metric_name:
                circuit_requests.labels(**labels).inc(value)
            elif "health_score" in metric_name:
                circuit_health.labels(**labels).set(value)
        
        await asyncio.sleep(interval)

# Iniciar servidor Prometheus
start_http_server(8000)

# Iniciar exportación
breaker = CircuitBreaker(name="api_service")
asyncio.create_task(export_metrics_periodically(breaker))
```

### Ejemplo 4: Ajuste Dinámico Basado en Métricas

```python
async def auto_tune_circuit_breaker(breaker: CircuitBreaker):
    """Automatically tune circuit breaker based on metrics"""
    health = breaker.get_health_status()
    metrics = breaker.get_metrics()
    
    # Si success rate es muy alto, podemos ser más agresivos
    if health["success_rate"] > 0.99:
        await breaker.update_config(
            failure_threshold=max(3, breaker.config.failure_threshold - 1)
        )
    
    # Si success rate es bajo pero no crítico, aumentar threshold
    elif health["success_rate"] < 0.85:
        await breaker.update_config(
            failure_threshold=min(20, breaker.config.failure_threshold + 2)
        )
    
    # Ajustar timeout basado en response times
    if metrics.avg_response_time > 1.0:  # > 1 segundo
        await breaker.update_config(
            call_timeout=min(30.0, (breaker.config.call_timeout or 10.0) * 1.5)
        )
```

### Ejemplo 5: Bulk Operations con Retry Individual

```python
async def process_with_individual_retry(breaker: CircuitBreaker, items: List[Tuple]):
    """Process items with individual retry logic"""
    results = []
    
    for item in items:
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                result = await breaker.call(process_item, *item)
                results.append(result)
                break
            except Exception as e:
                if attempt == max_attempts - 1:
                    results.append(None)
                    logger.error(f"Failed to process item after {max_attempts} attempts: {e}")
                else:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    return results
```

## 🎯 Beneficios

1. **Bulk Operations**: Procesamiento eficiente de múltiples items
2. **Configuración Dinámica**: Ajuste en tiempo real sin reiniciar
3. **Exportación de Métricas**: Integración fácil con sistemas de monitoreo
4. **Flexibilidad**: Adaptación a condiciones cambiantes
5. **Observabilidad**: Métricas en formatos estándar

## ✅ Estado

- ✅ Bulk operations implementadas
- ✅ Configuración dinámica implementada
- ✅ Exportación Prometheus implementada
- ✅ Exportación StatsD implementada
- ✅ Documentación completa
- ✅ Listo para usar

Todas las características avanzadas están implementadas y listas para usar.




