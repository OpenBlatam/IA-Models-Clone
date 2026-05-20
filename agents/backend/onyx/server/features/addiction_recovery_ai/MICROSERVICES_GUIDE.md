# Microservices Guide - Addiction Recovery AI

## ✅ Microservices Components

### Microservices Structure

```
microservices/
├── api_gateway.py              # ✅ API Gateway
├── service_discovery.py        # ✅ Service discovery
├── service_mesh.py             # ✅ Service mesh
├── service_client.py           # ✅ Service client
├── load_balancer.py            # ✅ Load balancer
├── event_bus.py                # ✅ Event bus
├── health_distributed.py       # ✅ Distributed health checks
└── graceful_degradation.py     # ✅ Graceful degradation
```

## 📦 Microservices Components

### `microservices/service_discovery.py` - Service Discovery
- **Status**: ✅ Active
- **Purpose**: Service registration and discovery
- **Features**: 
  - Service registration
  - Health checking
  - Load balancing (round-robin, random, least-connections)
  - Heartbeat management

**Usage:**
```python
from microservices.service_discovery import get_service_registry

registry = get_service_registry()
registry.register("user-service", instance_id="1", host="localhost", port=8001)
instances = registry.discover("user-service", healthy_only=True)
```

### `microservices/api_gateway.py` - API Gateway
- **Status**: ✅ Active
- **Purpose**: API Gateway for microservices
- **Features**: Routing, authentication, rate limiting

### `microservices/service_mesh.py` - Service Mesh
- **Status**: ✅ Active
- **Purpose**: Service mesh implementation
- **Features**: Service-to-service communication, traffic management

### `microservices/service_client.py` - Service Client
- **Status**: ✅ Active
- **Purpose**: HTTP client for inter-service communication
- **Features**: 
  - Circuit breaker
  - Retry with exponential backoff
  - Connection pooling
  - Service discovery integration

### `microservices/load_balancer.py` - Load Balancer
- **Status**: ✅ Active
- **Purpose**: Load balancing for services
- **Features**: Multiple strategies, health checks

### `microservices/event_bus.py` - Event Bus
- **Status**: ✅ Active
- **Purpose**: Event-driven communication
- **Features**: Pub/sub, event routing

### `microservices/health_distributed.py` - Distributed Health
- **Status**: ✅ Active
- **Purpose**: Distributed health checks
- **Features**: Health aggregation, service status

### `microservices/graceful_degradation.py` - Graceful Degradation
- **Status**: ✅ Active
- **Purpose**: Graceful degradation patterns
- **Features**: Fallback strategies, circuit breakers

## 📝 Usage Examples

### Service Discovery
```python
from microservices.service_discovery import get_service_registry

registry = get_service_registry()
registry.register("assessment-service", "instance-1", "localhost", 8001)
instance = registry.get_instance("assessment-service", strategy="round_robin")
```

### Service Client
```python
from microservices.service_client import ServiceClient

client = ServiceClient(service_name="user-service")
response = await client.get("/users/123")
```

### API Gateway
```python
from microservices.api_gateway import APIGateway

gateway = APIGateway()
gateway.add_route("/api/users", "user-service")
```

## 📚 Additional Resources

- See `MICROSERVICES_ARCHITECTURE.md` for detailed architecture
- See `INFRASTRUCTURE_GUIDE.md` for infrastructure
- See `AWS_DEPLOYMENT.md` for AWS deployment






