# Microservices Architecture

## 🏗️ Architecture Overview

The system has been refactored into a **true microservices architecture** with:

- ✅ **Independent Services**: Each service runs separately
- ✅ **Stateless Services**: No local state, uses external storage
- ✅ **Service Discovery**: Automatic service registration and discovery
- ✅ **API Gateway**: Single entry point routing to services
- ✅ **Inter-Service Communication**: HTTP with retries and circuit breakers
- ✅ **Event-Driven**: Optional event-based communication
- ✅ **Serverless Ready**: Each service can be deployed independently

## 📊 Service Architecture

```
┌─────────────────┐
│   API Gateway   │  Port 8000
│  (Entry Point)  │
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────┐
    │         │          │          │
    ▼         ▼          ▼          ▼
┌────────┐ ┌──────────┐ ┌──────┐ ┌──────────┐
│Movement│ │Trajectory│ │ Chat │ │  Other   │
│Service │ │ Service  │ │Service│ │ Services │
│ :8001  │ │  :8002   │ │ :8003│ │          │
└────────┘ └──────────┘ └──────┘ └──────────┘
    │         │          │
    └─────────┴──────────┘
         │
    ┌────▼────┐
    │ Service │
    │Registry │
    └─────────┘
```

## 🔧 Services

### 1. API Gateway Service
- **Port**: 8000
- **Purpose**: Single entry point, routes requests to microservices
- **Features**:
  - Request routing
  - Load balancing
  - Service discovery integration
  - Health check aggregation

### 2. Movement Service
- **Port**: 8001
- **Purpose**: Robot movement operations
- **Dependencies**: Trajectory Service
- **Endpoints**:
  - `POST /api/v1/move/to` - Move to position
  - `POST /api/v1/move/stop` - Stop movement
  - `GET /api/v1/movement/status` - Get status

### 3. Trajectory Service
- **Port**: 8002
- **Purpose**: Trajectory optimization
- **Dependencies**: None (stateless)
- **Endpoints**:
  - `POST /api/v1/trajectory/optimize` - Optimize trajectory
  - `POST /api/v1/trajectory/validate` - Validate trajectory

### 4. Chat Service
- **Port**: 8003
- **Purpose**: Chat-based robot control
- **Dependencies**: Movement Service
- **Endpoints**:
  - `POST /api/v1/chat` - Process chat message
  - `WS /ws/chat` - WebSocket chat

## 🚀 Running Services

### Run Individual Service

```bash
# API Gateway
python -m aws.services.service_runner api-gateway 8000

# Movement Service
python -m aws.services.service_runner movement-service 8001

# Trajectory Service
python -m aws.services.service_runner trajectory-service 8002

# Chat Service
python -m aws.services.service_runner chat-service 8003
```

### Run with Docker Compose

```yaml
version: '3.8'
services:
  api-gateway:
    build: .
    command: python -m aws.services.service_runner api-gateway 8000
    ports:
      - "8000:8000"
  
  movement-service:
    build: .
    command: python -m aws.services.service_runner movement-service 8001
    ports:
      - "8001:8001"
  
  trajectory-service:
    build: .
    command: python -m aws.services.service_runner trajectory-service 8002
    ports:
      - "8002:8002"
  
  chat-service:
    build: .
    command: python -m aws.services.service_runner chat-service 8003
    ports:
      - "8003:8003"
```

## 🔌 Service Discovery

### Service Registry

Services automatically register themselves:

```python
from aws.services.service_registry import get_service_registry

registry = get_service_registry()

# Register service
instance = ServiceInstance(
    service_name="movement-service",
    instance_id="movement-1",
    host="localhost",
    port=8001,
    health_check_url="http://localhost:8001/health"
)
registry.register(instance)

# Get service instance
instance = registry.get_instance("movement-service")
```

### Health Checks

Services automatically check health:

```python
# Registry checks health every 30 seconds
registry.start_heartbeat_checker()
```

## 📡 Inter-Service Communication

### Service Client

```python
from aws.services.service_client import ServiceClientFactory

# Get client
client = ServiceClientFactory.get_client("movement-service")

# Make request
result = await client.post("/api/v1/move/to", json={"x": 0.5, "y": 0.3, "z": 0.2})
```

### Features

- **Automatic Retries**: 3 retries with exponential backoff
- **Circuit Breaker**: Prevents cascading failures
- **Service Discovery**: Automatically finds healthy instances
- **Load Balancing**: Round-robin or random selection

## 🔒 Stateless Design

All services are **stateless**:

- ✅ No local state storage
- ✅ Use Redis for caching
- ✅ Use external databases
- ✅ Use message queues for async operations
- ✅ Can scale horizontally

## 📈 Scaling

### Horizontal Scaling

Each service can be scaled independently:

```bash
# Scale movement service
docker-compose up --scale movement-service=3
```

### Load Balancing

API Gateway automatically load balances:

```python
# Round-robin (default)
instance = registry.get_instance("movement-service", strategy="round_robin")

# Random
instance = registry.get_instance("movement-service", strategy="random")
```

## 🚀 Serverless Deployment

Each service can be deployed as Lambda:

```python
# Lambda handler for movement service
from aws.services.movement_service import MovementService
from mangum import Mangum

service = MovementService()
app = service.setup()
handler = Mangum(app)
```

## 📊 Monitoring

### Service Metrics

Each service exposes metrics:

- `/health` - Health check
- `/metrics` - Prometheus metrics

### Distributed Tracing

OpenTelemetry traces requests across services:

```python
# Automatic tracing via middleware
# Traces flow: API Gateway -> Movement Service -> Trajectory Service
```

## 🔄 Event-Driven Communication

Optional event-based communication:

```python
from aws.plugins.messaging import KafkaMessagingPlugin

# Publish event
messaging = plugin_manager.registry.get_messaging_plugin()
messaging.publish("movement.started", {"position": [0.5, 0.3, 0.2]})
```

## 📁 File Structure

```
aws/services/
├── __init__.py
├── base_service.py          # Base microservice class
├── service_registry.py       # Service discovery
├── service_client.py         # Inter-service communication
├── movement_service.py       # Movement microservice
├── trajectory_service.py     # Trajectory microservice
├── chat_service.py           # Chat microservice
├── api_gateway.py            # API Gateway service
└── service_runner.py         # Service runner utility
```

## ✅ Benefits

1. **Independence**: Services can be developed/deployed independently
2. **Scalability**: Scale each service based on load
3. **Resilience**: Failure in one service doesn't affect others
4. **Technology Diversity**: Each service can use different tech
5. **Team Autonomy**: Different teams can own different services
6. **Stateless**: Easy to scale horizontally
7. **Serverless Ready**: Can deploy as Lambda functions

## 🎯 Next Steps

1. Add more microservices (analytics, reporting, etc.)
2. Implement service mesh (Istio/Linkerd)
3. Add API Gateway features (rate limiting, auth)
4. Implement distributed tracing
5. Add service monitoring dashboard
6. Implement event sourcing
7. Add database per service pattern

---

**The system is now a true microservices architecture!** 🚀















