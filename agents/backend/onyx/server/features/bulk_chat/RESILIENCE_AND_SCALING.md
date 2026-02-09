# Sistemas Avanzados de Resiliencia y Escalado

## Nuevas Clases Implementadas

### 1. **BulkAdaptiveRateLimiter** - Rate Limiter Adaptativo

Sistema de rate limiting que ajusta automáticamente sus límites según el rendimiento del sistema.

#### Características:
- Ajuste automático de límites según éxito/fallo
- Incremento cuando el sistema funciona bien
- Reducción cuando hay errores
- Tracking de response times y success rate

#### Uso:

```python
from core.bulk_operations import BulkAdaptiveRateLimiter

rate_limiter = BulkAdaptiveRateLimiter(
    initial_rate=100,
    min_rate=10,
    max_rate=1000,
    adjustment_factor=1.2
)

# Verificar si se puede procesar
if rate_limiter.can_proceed():
    # Procesar petición
    result = await process_request()
    rate_limiter.record_success(response_time=0.5)
else:
    # Rate limit alcanzado
    return {"error": "Rate limit exceeded"}

# Registrar fallo
rate_limiter.record_failure()

# Obtener estado
status = rate_limiter.get_status()
# Returns: {
#   "current_rate": 150,
#   "success_rate": 95.5,
#   "utilization": 75.0,
#   ...
# }
```

#### Endpoints API:
- `GET /api/v1/bulk/rate-limiter/status`
- `POST /api/v1/bulk/rate-limiter/check`
- `POST /api/v1/bulk/rate-limiter/record`

---

### 2. **BulkLoadBalancer** - Load Balancer Inteligente

Sistema de balanceo de carga que distribuye tareas entre workers de forma inteligente.

#### Características:
- Selección de worker con menor carga
- Tracking de performance por worker
- Añadir/remover workers dinámicamente
- Estadísticas detalladas por worker

#### Uso:

```python
from core.bulk_operations import BulkLoadBalancer

lb = BulkLoadBalancer(initial_workers=10)

# Seleccionar worker para nueva tarea
worker_id = lb.select_worker()
lb.assign_task(worker_id)

# Procesar tarea
try:
    result = await process_task(data)
    lb.complete_task(worker_id, success=True, response_time=0.3)
except Exception as e:
    lb.complete_task(worker_id, success=False, response_time=0.1)

# Añadir worker cuando hay alta carga
new_worker = lb.add_worker()

# Remover worker cuando hay baja carga (solo si no tiene tareas)
lb.remove_worker(worker_id)

# Estadísticas
stats = lb.get_stats()
```

#### Endpoints API:
- `GET /api/v1/bulk/load-balancer/stats`
- `POST /api/v1/bulk/load-balancer/select-worker`
- `POST /api/v1/bulk/load-balancer/complete-task`
- `POST /api/v1/bulk/load-balancer/add-worker`

---

### 3. **BulkLoadPredictor** - Predicción de Carga

Sistema que predice carga futura basándose en patrones históricos.

#### Características:
- Análisis de patrones por hora del día
- Predicción de carga futura
- Nivel de confianza en predicciones
- Tracking de tendencias recientes

#### Uso:

```python
from core.bulk_operations import BulkLoadPredictor

predictor = BulkLoadPredictor(window_size=60)

# Registrar carga actual
current_load = calculate_current_load()  # 0.0 - 1.0
predictor.record_load(current_load)

# Predecir carga en 5 minutos
prediction = predictor.predict_load(minutes_ahead=5)
# Returns: {
#   "predicted_load": 0.75,
#   "confidence": 0.85,
#   "based_on": 50,
#   "target_time": "2024-01-01T12:05:00",
#   "historical_avg": 0.70
# }

# Obtener patrones por hora
pattern = predictor.get_load_pattern()
# Returns: {
#   "hourly_patterns": {
#     "9": {"avg_load": 0.8, "samples": 100, ...},
#     "10": {"avg_load": 0.9, "samples": 100, ...},
#     ...
#   },
#   "current_load": 0.75,
#   "total_samples": 1440
# }
```

#### Endpoints API:
- `GET /api/v1/bulk/load-predictor/predict?minutes_ahead=5`
- `GET /api/v1/bulk/load-predictor/pattern`
- `POST /api/v1/bulk/load-predictor/record`

---

### 4. **BulkAutoScaler** - Auto-Scaling Inteligente

Sistema de auto-scaling que ajusta el número de workers basándose en carga actual y predicciones.

#### Características:
- Escalado automático basado en utilización
- Considera predicciones de carga futura
- Scale-up y scale-down inteligente
- Historial de decisiones de scaling

#### Uso:

```python
from core.bulk_operations import BulkAutoScaler, BulkLoadBalancer, BulkLoadPredictor, BulkRealTimeMetrics

scaler = BulkAutoScaler(
    load_balancer=BulkLoadBalancer(initial_workers=10),
    load_predictor=BulkLoadPredictor(),
    metrics=BulkRealTimeMetrics(),
    min_workers=1,
    max_workers=100,
    scale_up_threshold=0.8,
    scale_down_threshold=0.3
)

# Evaluar si es necesario hacer scaling
evaluation = scaler.evaluate_scaling()
# Returns: {
#   "action": "scale_up",
#   "current_workers": 10,
#   "recommended_workers": 12,
#   "utilization": 0.85,
#   "predicted_load": 0.9,
#   "reason": "High utilization (85.00%) or predicted load (0.90)"
# }

# Ejecutar scaling si es necesario
result = await scaler.execute_scaling()

# Obtener historial
history = scaler.get_scaling_history()
```

#### Endpoints API:
- `GET /api/v1/bulk/autoscaler/evaluate` - Evaluar necesidad de scaling
- `POST /api/v1/bulk/autoscaler/execute` - Ejecutar scaling
- `GET /api/v1/bulk/autoscaler/history` - Historial de scaling

---

## Integración Completa

### Ejemplo: Sistema Completo con Todas las Funcionalidades

```python
from core.bulk_operations import (
    BulkAdaptiveRateLimiter,
    BulkLoadBalancer,
    BulkLoadPredictor,
    BulkAutoScaler,
    BulkRealTimeMetrics
)

# Inicializar componentes
metrics = BulkRealTimeMetrics()
rate_limiter = BulkAdaptiveRateLimiter()
load_balancer = BulkLoadBalancer(initial_workers=10)
load_predictor = BulkLoadPredictor()
auto_scaler = BulkAutoScaler(
    load_balancer=load_balancer,
    load_predictor=load_predictor,
    metrics=metrics
)

# Procesar petición con todos los sistemas
async def process_request_with_resilience(data):
    # 1. Verificar rate limit
    if not rate_limiter.can_proceed():
        return {"error": "Rate limit exceeded"}
    
    start_time = time.time()
    
    try:
        # 2. Seleccionar worker
        worker_id = load_balancer.select_worker()
        load_balancer.assign_task(worker_id)
        
        # 3. Procesar
        result = await process_task(data)
        response_time = time.time() - start_time
        
        # 4. Registrar éxito
        rate_limiter.record_success(response_time)
        load_balancer.complete_task(worker_id, success=True, response_time=response_time)
        
        # 5. Registrar carga
        current_load = calculate_load()
        load_predictor.record_load(current_load)
        
        # 6. Evaluar scaling
        await auto_scaler.execute_scaling()
        
        return result
        
    except Exception as e:
        response_time = time.time() - start_time
        rate_limiter.record_failure()
        load_balancer.complete_task(worker_id, success=False, response_time=response_time)
        raise
```

---

## Patrones de Uso

### 1. **Protección contra Picos de Carga**

```python
# Usar rate limiter adaptativo
if rate_limiter.can_proceed():
    process_request()
else:
    queue_request()  # Encolar para procesar después
```

### 2. **Distribución Inteligente de Carga**

```python
# Usar load balancer para distribuir tareas
worker_id = load_balancer.select_worker()
# Worker seleccionado tiene menor carga activa
```

### 3. **Preparación Anticipada**

```python
# Predecir carga y preparar recursos
prediction = load_predictor.predict_load(minutes_ahead=10)
if prediction["predicted_load"] > 0.8:
    # Pre-escalar antes de que llegue la carga
    await auto_scaler.execute_scaling()
```

### 4. **Auto-Scaling Proactivo**

```python
# Auto-scaling basado en predicciones
evaluation = auto_scaler.evaluate_scaling()
if evaluation["action"] != "none":
    await auto_scaler.execute_scaling()
```

---

## Beneficios

1. **Resiliencia**: Sistema se adapta automáticamente a condiciones cambiantes
2. **Eficiencia**: Distribución óptima de carga entre workers
3. **Predicción**: Anticipación de picos de carga
4. **Auto-Scaling**: Ajuste automático de recursos según necesidad
5. **Rate Limiting Inteligente**: Ajuste automático según rendimiento
6. **Costos Optimizados**: Escalar solo cuando es necesario

---

## Configuración Recomendada

### Para Alta Carga:
```python
rate_limiter = BulkAdaptiveRateLimiter(
    initial_rate=500,
    max_rate=5000,
    adjustment_factor=1.5
)

auto_scaler = BulkAutoScaler(
    min_workers=10,
    max_workers=200,
    scale_up_threshold=0.7,  # Más agresivo
    scale_down_threshold=0.2
)
```

### Para Baja Carga:
```python
rate_limiter = BulkAdaptiveRateLimiter(
    initial_rate=50,
    max_rate=500,
    adjustment_factor=1.1
)

auto_scaler = BulkAutoScaler(
    min_workers=1,
    max_workers=50,
    scale_up_threshold=0.9,  # Más conservador
    scale_down_threshold=0.4
)
```

---

## Monitoreo

Todas las clases proporcionan métodos `get_status()` o `get_stats()` para monitoreo:

```python
# Dashboard de resiliencia
resilience_dashboard = {
    "rate_limiter": rate_limiter.get_status(),
    "load_balancer": load_balancer.get_stats(),
    "load_prediction": load_predictor.predict_load(),
    "scaling_evaluation": auto_scaler.evaluate_scaling(),
    "scaling_history": auto_scaler.get_scaling_history()
}
```

---

## Próximos Pasos

1. Integrar con sistema de alertas
2. Añadir métricas a sistema de monitoreo externo
3. Implementar persistencia de patrones de carga
4. Añadir machine learning para predicciones más precisas
5. Integrar con sistemas de orquestación (Kubernetes, etc.)
















