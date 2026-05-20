# Microservices Architecture

## 🏗️ Arquitectura de Microservicios

Implementación completa de patrones de microservicios con service discovery, API Gateway, y comunicación inter-servicios.

## 📁 Componentes

### 1. Service Discovery (`microservices/service_discovery.py`)

**Características:**
- ✅ Registro de servicios
- ✅ Health checking
- ✅ Load balancing (round-robin, random, least-connections)
- ✅ Heartbeat management
- ✅ Service status tracking

**Uso:**
```python
from microservices.service_discovery import get_service_registry

registry = get_service_registry()

# Registrar servicio
registry.register(
    service_name="user-service",
    instance_id="instance-1",
    host="localhost",
    port=8001
)

# Descubrir servicios
instances = registry.discover("user-service", healthy_only=True)

# Obtener instancia (con load balancing)
instance = registry.get_instance("user-service", strategy="round_robin")
```

### 2. Service Client (`microservices/service_client.py`)

**Características:**
- ✅ HTTP client para comunicación inter-servicios
- ✅ Circuit breaker integrado
- ✅ Retry con exponential backoff
- ✅ Connection pooling
- ✅ Timeout handling
- ✅ Service discovery integration

**Uso:**
```python
from microservices.service_client import get_service_client

# Obtener cliente de servicio
client = get_service_client("user-service")

# Hacer requests
user = await client.get("/users/123")
result = await client.post("/users", json={"name": "John"})
```

### 3. API Gateway (`microservices/api_gateway.py`)

**Características:**
- ✅ Request routing a microservicios
- ✅ Request/response transformation
- ✅ Rate limiting por ruta
- ✅ Service discovery integration
- ✅ Load balancing

**Uso:**
```python
from microservices.api_gateway import get_api_gateway, RateLimitConfig

gateway = get_api_gateway()

# Registrar ruta
gateway.register_route(
    path_prefix="/api/users",
    service_name="user-service",
    rate_limit=RateLimitConfig(requests_per_minute=60)
)

# En main.py
app.add_middleware(gateway.get_middleware())
```

### 4. Event Bus (`microservices/event_bus.py`)

**Características:**
- ✅ Event publishing
- ✅ Event subscription
- ✅ SNS integration
- ✅ Local event handling
- ✅ Event routing

**Uso:**
```python
from microservices.event_bus import get_event_bus, EventType

event_bus = get_event_bus()

# Publicar evento
await event_bus.publish_event(
    event_type=EventType.USER_CREATED.value,
    source="user-service",
    data={"user_id": "123", "name": "John"}
)

# Suscribirse a eventos
class MyHandler:
    async def handle(self, data):
        print(f"Event received: {data}")

event_bus.subscribe(EventType.USER_CREATED.value, MyHandler())
```

## 🔄 Flujo de Comunicación

### Request Flow

```
Client Request
    ↓
API Gateway (Routing)
    ↓
Service Discovery (Find Instance)
    ↓
Service Client (HTTP Request)
    ↓
Circuit Breaker Check
    ↓
Retry Logic (if needed)
    ↓
Microservice
    ↓
Response
```

### Event Flow

```
Service Action
    ↓
Event Bus (Publish)
    ↓
SNS (if available)
    ↓
Local Subscribers
    ↓
Event Handlers
```

## 🎯 Patrones Implementados

### 1. Service Discovery Pattern

- **Registro**: Servicios se registran automáticamente
- **Descubrimiento**: Clientes descubren servicios dinámicamente
- **Health Checking**: Monitoreo continuo de salud
- **Load Balancing**: Distribución de carga

### 2. API Gateway Pattern

- **Routing**: Enrutamiento a microservicios
- **Transformation**: Transformación de requests/responses
- **Rate Limiting**: Limitación de tasa por ruta
- **Authentication**: Autenticación centralizada

### 3. Circuit Breaker Pattern

- **Failure Detection**: Detección de fallos
- **Automatic Recovery**: Recuperación automática
- **Fallback**: Respuestas de fallback

### 4. Event-Driven Pattern

- **Publish/Subscribe**: Patrón pub/sub
- **Event Sourcing**: Trazabilidad de eventos
- **Async Processing**: Procesamiento asíncrono

## 📊 Service Registry

### Registrar Servicio

```python
registry = get_service_registry()

registry.register(
    service_name="recovery-service",
    instance_id="recovery-1",
    host="recovery-service.internal",
    port=8000,
    metadata={"region": "us-east-1", "version": "1.0.0"}
)
```

### Health Checking

```python
# Heartbeat automático
registry.heartbeat("recovery-service", "recovery-1")

# Health check manual
status = registry.check_health("recovery-service", "recovery-1")
```

### Service Information

```python
info = registry.get_service_info("recovery-service")
# {
#     "service_name": "recovery-service",
#     "total_instances": 3,
#     "healthy_instances": 2,
#     "instances": [...]
# }
```

## 🔌 Service Client

### Configuración

```python
client = ServiceClient(
    service_name="user-service",
    timeout=5.0,
    max_retries=3
)
```

### Requests

```python
# GET
data = await client.get("/users/123")

# POST
result = await client.post("/users", json={"name": "John"})

# PUT
updated = await client.put("/users/123", json={"name": "Jane"})

# DELETE
await client.delete("/users/123")
```

### Error Handling

```python
try:
    data = await client.get("/users/123")
except CircuitBreakerError:
    # Circuit breaker abierto
    pass
except httpx.HTTPStatusError as e:
    # Error HTTP
    pass
```

## 🌐 API Gateway

### Configuración de Rutas

```python
gateway = get_api_gateway()

# Ruta simple
gateway.register_route(
    path_prefix="/api/users",
    service_name="user-service"
)

# Ruta con rate limiting
gateway.register_route(
    path_prefix="/api/recovery",
    service_name="recovery-service",
    rate_limit=RateLimitConfig(
        requests_per_minute=60,
        requests_per_hour=1000
    )
)
```

### Response Transformation

```python
def transform_response(response):
    # Transformar respuesta
    data = response.json()
    return Response(
        content=json.dumps({"transformed": data}),
        status_code=response.status_code
    )

gateway.get_middleware().register_transformer(
    "user-service",
    transform_response
)
```

## 📨 Event Bus

### Publicar Eventos

```python
event_bus = get_event_bus()

# Evento simple
await event_bus.publish_event(
    event_type="user.created",
    source="user-service",
    data={"user_id": "123"}
)

# Evento con correlation ID
await event_bus.publish_event(
    event_type="milestone.achieved",
    source="recovery-service",
    data={"user_id": "123", "milestone": "30_days"},
    correlation_id="req-456"
)
```

### Suscribirse a Eventos

```python
class NotificationHandler:
    async def handle(self, data):
        # Enviar notificación
        pass

event_bus.subscribe("milestone.achieved", NotificationHandler())
```

## 🔄 Integración con FastAPI

### En main.py

```python
from microservices.api_gateway import get_api_gateway
from microservices.service_discovery import get_service_registry

# Registrar servicios
registry = get_service_registry()
registry.register(
    service_name="recovery-service",
    instance_id="main",
    host="0.0.0.0",
    port=8000
)

# Configurar API Gateway
gateway = get_api_gateway()
gateway.register_route("/api/recovery", "recovery-service")

# Agregar middleware
app.add_middleware(gateway.get_middleware())
```

## 🧪 Testing

### Mock Service Registry

```python
from unittest.mock import Mock
from microservices.service_discovery import ServiceInstance, ServiceStatus

registry = get_service_registry()
instance = ServiceInstance(
    service_name="test-service",
    instance_id="test-1",
    host="localhost",
    port=8000,
    status=ServiceStatus.HEALTHY
)
registry._services["test-service"] = [instance]
```

### Mock Service Client

```python
from unittest.mock import AsyncMock

client = get_service_client("test-service")
client.client = AsyncMock()
client.client.request.return_value = Mock(
    status_code=200,
    json=lambda: {"id": "123"}
)
```

## 📈 Monitoreo

### Service Health

```python
# Health de todos los servicios
for service_name in registry.list_services():
    info = registry.get_service_info(service_name)
    print(f"{service_name}: {info['healthy_instances']}/{info['total_instances']}")
```

### Event Statistics

```python
# Listar suscriptores
subscribers = event_bus.list_subscribers("user.created")
print(f"Subscribers: {subscribers}")
```

## 🚀 Deployment

### Service Registration

```python
# En cada microservicio
import os
from microservices.service_discovery import get_service_registry

registry = get_service_registry()
registry.register(
    service_name=os.getenv("SERVICE_NAME"),
    instance_id=os.getenv("INSTANCE_ID"),
    host=os.getenv("SERVICE_HOST"),
    port=int(os.getenv("SERVICE_PORT"))
)
```

### Health Endpoint

```python
@app.get("/health")
async def health():
    registry = get_service_registry()
    instance = registry.get_service_info("recovery-service")
    return {"status": "healthy", "service": instance}
```

## ✅ Checklist

- [x] Service discovery
- [x] Service client con circuit breaker
- [x] API Gateway
- [x] Event bus
- [x] Load balancing
- [x] Health checking
- [x] Retry logic
- [x] Error handling
- [x] Integration con FastAPI
- [x] Documentación

---

**Arquitectura de microservicios completada** ✅

Sistema listo para comunicación inter-servicios con service discovery, API Gateway y event-driven architecture.















