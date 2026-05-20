# Patrones de Resiliencia - Color Grading AI TruthGPT

## Resumen

Patrones de resiliencia implementados: circuit breaker, retry manager, load balancer y feature flags.

## Nuevos Patrones

### 1. Circuit Breaker

**Archivo**: `services/circuit_breaker.py`

**Características**:
- ✅ Protección contra fallos en cascada
- ✅ Estados: CLOSED, OPEN, HALF_OPEN
- ✅ Recovery automático
- ✅ Configuración flexible

**Estados**:
- **CLOSED**: Operación normal
- **OPEN**: Falla detectada, rechaza requests
- **HALF_OPEN**: Probando recuperación

**Uso**:
```python
# Crear circuit breaker
breaker = CircuitBreaker(
    "external_api",
    config=CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=2,
        timeout=60.0
    )
)

# Usar con protección
try:
    result = await breaker.call(external_api_call, param1, param2)
except CircuitBreakerOpenError:
    # Circuit abierto, usar fallback
    result = fallback_function()
```

### 2. Retry Manager

**Archivo**: `services/retry_manager.py`

**Características**:
- ✅ Múltiples estrategias de retry
- ✅ Exponential backoff
- ✅ Jitter para sistemas distribuidos
- ✅ Estadísticas de retry
- ✅ Decorator automático

**Estrategias**:
- FIXED: Delay fijo
- EXPONENTIAL: Backoff exponencial
- LINEAR: Backoff lineal
- CUSTOM: Lógica personalizada

**Uso**:
```python
# Crear retry manager
retry_manager = RetryManager(
    config=RetryConfig(
        max_attempts=3,
        initial_delay=1.0,
        strategy=RetryStrategy.EXPONENTIAL,
        jitter=True
    )
)

# Usar con retry
result = await retry_manager.execute_with_retry(
    unreliable_function,
    operation_name="api_call"
)

# O con decorator
@retry_on_failure(config=RetryConfig(max_attempts=3))
async def api_call():
    return await unreliable_api()
```

### 3. Load Balancer

**Archivo**: `services/load_balancer.py`

**Características**:
- ✅ Múltiples estrategias de balanceo
- ✅ Health checking
- ✅ Weighted distribution
- ✅ Monitoreo de carga
- ✅ Estadísticas

**Estrategias**:
- ROUND_ROBIN: Round-robin simple
- LEAST_CONNECTIONS: Menos conexiones activas
- WEIGHTED_ROUND_ROBIN: Round-robin con pesos
- LEAST_LOAD: Menor carga
- RANDOM: Selección aleatoria

**Uso**:
```python
# Crear load balancer
balancer = LoadBalancer(strategy=LoadBalanceStrategy.LEAST_LOAD)

# Registrar workers
balancer.register_worker("worker1", weight=2)
balancer.register_worker("worker2", weight=1)

# Seleccionar worker
worker_id = balancer.select_worker()
if worker_id:
    result = await process_on_worker(worker_id, data)
```

### 4. Feature Flags

**Archivo**: `services/feature_flags.py`

**Características**:
- ✅ Boolean flags
- ✅ Percentage rollouts
- ✅ User-based flags
- ✅ Custom rules
- ✅ A/B testing

**Tipos**:
- BOOLEAN: On/off simple
- PERCENTAGE: Rollout gradual
- USER_LIST: Lista de usuarios
- CUSTOM: Regla personalizada

**Uso**:
```python
# Crear feature flag manager
flags = FeatureFlagManager()

# Registrar flag
flags.register_flag(
    "new_grading_algorithm",
    enabled=True,
    flag_type=FeatureFlagType.PERCENTAGE,
    percentage=10.0  # 10% de usuarios
)

# Verificar flag
if flags.is_enabled("new_grading_algorithm", user_id="user123"):
    # Usar nuevo algoritmo
    result = await new_algorithm()
else:
    # Usar algoritmo antiguo
    result = await old_algorithm()

# Con decorator
@feature_flag("new_feature", default=False)
async def new_feature_function(self):
    # Solo ejecuta si flag está habilitado
    pass
```

## Beneficios

### Resiliencia
- ✅ Circuit breaker previene fallos en cascada
- ✅ Retry manager maneja fallos temporales
- ✅ Load balancer distribuye carga
- ✅ Feature flags para rollouts seguros

### Observabilidad
- ✅ Estadísticas de circuit breaker
- ✅ Estadísticas de retry
- ✅ Estadísticas de load balancer
- ✅ Tracking de feature flags

### Flexibilidad
- ✅ Múltiples estrategias
- ✅ Configuración flexible
- ✅ Custom rules
- ✅ A/B testing

## Integración

### Circuit Breaker + Retry
```python
# Combinar circuit breaker y retry
breaker = CircuitBreaker("api")
retry = RetryManager()

async def resilient_call():
    return await breaker.call(
        lambda: retry.execute_with_retry(unreliable_api)
    )
```

### Load Balancer + Feature Flags
```python
# Balancear con feature flags
if flags.is_enabled("new_workers", user_id):
    worker = new_balancer.select_worker()
else:
    worker = old_balancer.select_worker()
```

## Estadísticas Finales

### Servicios Totales: 51+

**Nuevos Servicios**:
- CircuitBreaker
- RetryManager
- LoadBalancer
- FeatureFlagManager

### Patrones de Resiliencia

✅ **Circuit Breaker**
- Protección contra fallos
- Recovery automático
- Estados gestionados

✅ **Retry**
- Múltiples estrategias
- Exponential backoff
- Jitter

✅ **Load Balancing**
- Múltiples estrategias
- Health checking
- Weighted distribution

✅ **Feature Flags**
- Rollouts graduales
- A/B testing
- Custom rules

## Conclusión

El sistema ahora incluye patrones de resiliencia enterprise:
- ✅ Circuit breaker para protección
- ✅ Retry manager para recuperación
- ✅ Load balancer para distribución
- ✅ Feature flags para rollouts

**El proyecto está completamente resiliente y listo para producción a gran escala.**




