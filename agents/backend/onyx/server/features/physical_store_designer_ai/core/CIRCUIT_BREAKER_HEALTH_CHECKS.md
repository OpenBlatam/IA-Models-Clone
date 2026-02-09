# Circuit Breaker - Health Check Integration Mejorada

## ✅ Mejora #4: Health Check Integration - Implementación Avanzada

Se ha implementado un sistema completo de health checks para el Circuit Breaker, permitiendo integración con sistemas de monitoreo y health check endpoints.

## 🎯 Características

### 1. Métodos de Health Check

- **`is_healthy()`**: Verifica si está saludable
- **`is_ready()`**: Verifica si está listo para aceptar requests
- **`is_degraded()`**: Verifica si está en estado degradado
- **`is_critical()`**: Verifica si está en estado crítico
- **`get_health_score()`**: Obtiene score de salud (0.0 - 1.0)
- **`get_health_rating()`**: Obtiene rating legible ("excellent", "good", "degraded", "critical", "unavailable")
- **`get_health_status()`**: Obtiene estado completo de salud

### 2. Configuración de Umbrales

- **`health_success_rate_threshold`**: Tasa de éxito mínima para considerar saludable (default: 0.95 = 95%)
- **`health_degraded_threshold`**: Tasa de éxito por debajo de la cual se considera degradado (default: 0.80 = 80%)

### 3. Health Score

Score numérico de 0.0 a 1.0 que considera:
- Tasa de éxito
- Estado del circuit breaker
- Proximidad al threshold de fallos
- Historial reciente

### 4. Recomendaciones Automáticas

El sistema genera recomendaciones basadas en el estado actual.

## 📚 Ejemplos de Uso

### Ejemplo 1: Health Check Básico

```python
from .core.circuit_breaker import CircuitBreaker

breaker = CircuitBreaker(name="api_service")

# Verificar salud básica
if breaker.is_healthy():
    result = await breaker.call(api_function, ...)
else:
    logger.warning("Circuit breaker is not healthy")

# Verificar si está listo
if breaker.is_ready():
    # Aceptar requests
    pass
```

### Ejemplo 2: Health Check Detallado

```python
# Obtener estado completo de salud
health = breaker.get_health_status()

print(f"Healthy: {health['healthy']}")
print(f"Ready: {health['ready']}")
print(f"Degraded: {health['degraded']}")
print(f"Critical: {health['critical']}")
print(f"Health Score: {health['health_score']:.2f}")
print(f"Health Rating: {health['health_rating']}")
print(f"Success Rate: {health['success_rate']:.2%}")

# Ver recomendaciones
for recommendation in health['recommendations']:
    print(f"  - {recommendation}")
```

### Ejemplo 3: Health Check Endpoint (FastAPI)

```python
from fastapi import APIRouter, HTTPException
from .core.circuit_breaker import get_all_circuit_breakers

router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    breakers = get_all_circuit_breakers()
    overall_healthy = True
    details = {}
    
    for name, state in breakers.items():
        health = state.get('health', {})
        details[name] = {
            "healthy": health.get('healthy', False),
            "rating": health.get('health_rating', 'unknown'),
            "score": health.get('health_score', 0.0)
        }
        
        if not health.get('healthy', False):
            overall_healthy = False
    
    if overall_healthy:
        return {
            "status": "healthy",
            "circuit_breakers": details
        }
    else:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "circuit_breakers": details
            }
        )
```

### Ejemplo 4: Health Check con Umbrales Personalizados

```python
from .core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

# Configurar umbrales personalizados
config = CircuitBreakerConfig(
    health_success_rate_threshold=0.98,  # 98% para saludable
    health_degraded_threshold=0.85       # 85% para degradado
)

breaker = CircuitBreaker(config=config, name="critical_service")

# Verificar estados
if breaker.is_healthy():
    # Servicio está saludable (>98% success rate)
    pass
elif breaker.is_degraded():
    # Servicio está degradado (85-98% success rate)
    logger.warning("Service is degraded")
elif breaker.is_critical():
    # Servicio está crítico (<85% success rate)
    logger.error("Service is critical!")
```

### Ejemplo 5: Monitoreo Continuo

```python
import asyncio
from .core.circuit_breaker import CircuitBreaker

async def monitor_health(breaker: CircuitBreaker, interval: int = 60):
    """Monitor circuit breaker health continuously"""
    while True:
        health = breaker.get_health_status()
        
        # Log health status
        logger.info(
            f"Circuit breaker {breaker.name} health",
            extra={
                "health_score": health['health_score'],
                "health_rating": health['health_rating'],
                "state": health['state'],
                "success_rate": health['success_rate']
            }
        )
        
        # Alert on critical state
        if health['critical']:
            send_alert(
                f"Circuit breaker {breaker.name} is in critical state!",
                health
            )
        
        # Alert on degraded state
        elif health['degraded']:
            send_warning(
                f"Circuit breaker {breaker.name} is degraded",
                health
            )
        
        await asyncio.sleep(interval)

# Iniciar monitoreo
breaker = CircuitBreaker(name="api_service")
asyncio.create_task(monitor_health(breaker, interval=60))
```

### Ejemplo 6: Health Score para Load Balancing

```python
def select_healthiest_breaker(breakers: List[CircuitBreaker]) -> CircuitBreaker:
    """Select the healthiest circuit breaker"""
    if not breakers:
        raise ValueError("No breakers provided")
    
    # Filter ready breakers
    ready_breakers = [b for b in breakers if b.is_ready()]
    
    if not ready_breakers:
        # All are open, return the one with shortest remaining timeout
        return min(
            breakers,
            key=lambda b: b._get_remaining_timeout()
        )
    
    # Return the one with highest health score
    return max(
        ready_breakers,
        key=lambda b: b.get_health_score()
    )
```

### Ejemplo 7: Health Check con Prometheus

```python
from prometheus_client import Gauge

circuit_breaker_health = Gauge(
    'circuit_breaker_health_score',
    'Circuit breaker health score',
    ['circuit_name']
)

circuit_breaker_healthy = Gauge(
    'circuit_breaker_healthy',
    'Circuit breaker healthy status',
    ['circuit_name']
)

def update_prometheus_metrics(breaker: CircuitBreaker):
    """Update Prometheus metrics from health status"""
    health = breaker.get_health_status()
    
    circuit_breaker_health.labels(
        circuit_name=breaker.name
    ).set(health['health_score'])
    
    circuit_breaker_healthy.labels(
        circuit_name=breaker.name
    ).set(1.0 if health['healthy'] else 0.0)
```

## 📊 Health Ratings

| Rating | Score Range | Description |
|--------|-------------|-------------|
| **excellent** | 0.95 - 1.0 | Perfecto funcionamiento |
| **good** | 0.80 - 0.95 | Buen funcionamiento |
| **degraded** | 0.60 - 0.80 | Funcionamiento degradado |
| **critical** | 0.0 - 0.60 | Estado crítico |
| **unavailable** | N/A | Circuit abierto |

## 🔍 Health Status Structure

```python
{
    "healthy": bool,              # Is healthy?
    "ready": bool,                # Is ready to accept requests?
    "degraded": bool,             # Is degraded?
    "critical": bool,             # Is critical?
    "health_score": float,        # 0.0 - 1.0
    "health_rating": str,         # "excellent", "good", etc.
    "state": str,                 # "closed", "open", "half_open"
    "failure_count": int,         # Current failure count
    "success_count": int,         # Current success count
    "success_rate": float,        # Success rate (0.0 - 1.0)
    "failure_rate": float,        # Failure rate (0.0 - 1.0)
    "total_requests": int,        # Total requests
    "remaining_timeout": float,   # Remaining timeout if open
    "thresholds": {
        "failure_threshold": int,
        "success_rate_threshold": float,
        "degraded_threshold": float
    },
    "recommendations": [str]      # List of recommendations
}
```

## 🎯 Recomendaciones Automáticas

El sistema genera recomendaciones basadas en:

- **Estado del circuit**: Abierto, cerrado, half-open
- **Tasa de éxito**: Comparada con umbrales
- **Conteo de fallos**: Proximidad al threshold
- **Datos disponibles**: Si hay suficientes datos para evaluación

Ejemplos de recomendaciones:
- "Circuit is open. Wait for recovery timeout or manually reset."
- "Circuit is in critical state. Monitor closely."
- "Circuit is degraded. Success rate below optimal threshold."
- "High failure rate detected. Investigate service issues."
- "Circuit breaker is operating normally."

## 🔄 Integración con Health Check Systems

### Kubernetes Liveness/Readiness Probes

```python
@router.get("/health/live")
async def liveness_probe():
    """Kubernetes liveness probe"""
    # Check if circuit breaker process is alive
    return {"status": "alive"}

@router.get("/health/ready")
async def readiness_probe():
    """Kubernetes readiness probe"""
    breaker = await get_circuit_breaker("api_service")
    
    if breaker.is_ready():
        return {"status": "ready"}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready"}
        )
```

### Consul Health Checks

```python
def register_consul_health_check(breaker: CircuitBreaker):
    """Register health check with Consul"""
    def check():
        health = breaker.get_health_status()
        if health['healthy']:
            return "pass"
        elif health['degraded']:
            return "warning"
        else:
            return "fail"
    
    consul.agent.service.register(
        name="api_service",
        check=consul.Check.http(
            "/health",
            interval="10s",
            check_func=check
        )
    )
```

## ✅ Estado

- ✅ Health checks básicos implementados
- ✅ Health checks avanzados (degraded, critical)
- ✅ Health score y rating
- ✅ Recomendaciones automáticas
- ✅ Configuración de umbrales
- ✅ Integración con sistemas de monitoreo
- ✅ Documentación completa

El sistema de health checks está completamente implementado y listo para integración con sistemas de monitoreo y health check endpoints.




