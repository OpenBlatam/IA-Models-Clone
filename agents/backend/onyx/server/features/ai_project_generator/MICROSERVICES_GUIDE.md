# Guía de Microservicios y Serverless

Esta guía documenta todas las características avanzadas de microservicios, serverless y arquitectura cloud-native implementadas en el AI Project Generator.

## 📋 Tabla de Contenidos

1. [Configuración](#configuración)
2. [Middleware Avanzado](#middleware-avanzado)
3. [Circuit Breakers](#circuit-breakers)
4. [Retry Logic](#retry-logic)
5. [Redis Cache](#redis-cache)
6. [Prometheus Metrics](#prometheus-metrics)
7. [API Gateway](#api-gateway)
8. [Message Brokers](#message-brokers)
9. [Async Workers](#async-workers)
10. [Serverless Optimizations](#serverless-optimizations)
11. [OAuth2 Security](#oauth2-security)

## 🔧 Configuración

Todas las características se configuran mediante variables de entorno con el prefijo `MICROSERVICES_`:

```bash
# Deployment
MICROSERVICES_DEPLOYMENT_TYPE=serverless  # standard, serverless, container, kubernetes
MICROSERVICES_SERVERLESS_PROVIDER=aws_lambda  # aws_lambda, azure_functions, gcp_cloud_functions
MICROSERVICES_MINIMIZE_COLD_START=true
MICROSERVICES_SERVERLESS_TIMEOUT=30

# API Gateway
MICROSERVICES_API_GATEWAY_TYPE=kong  # kong, aws_api_gateway, azure_api_management, traefik, nginx
MICROSERVICES_API_GATEWAY_URL=http://localhost:8001
MICROSERVICES_API_GATEWAY_KEY=your-api-key

# Message Broker
MICROSERVICES_MESSAGE_BROKER_TYPE=rabbitmq  # rabbitmq, kafka, redis, sqs
MICROSERVICES_MESSAGE_BROKER_URL=amqp://localhost:5672
MICROSERVICES_MESSAGE_BROKER_USERNAME=guest
MICROSERVICES_MESSAGE_BROKER_PASSWORD=guest

# Cache
MICROSERVICES_CACHE_BACKEND=redis  # redis, memcached, in_memory
MICROSERVICES_CACHE_URL=redis://localhost:6379
MICROSERVICES_CACHE_TTL=3600

# Workers
MICROSERVICES_WORKER_BACKEND=celery  # celery, rq, arq
MICROSERVICES_WORKER_BROKER_URL=redis://localhost:6379/0
MICROSERVICES_WORKER_RESULT_BACKEND=redis://localhost:6379/0

# Circuit Breaker
MICROSERVICES_CIRCUIT_BREAKER_ENABLED=true
MICROSERVICES_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
MICROSERVICES_CIRCUIT_BREAKER_TIMEOUT=60

# Retry
MICROSERVICES_RETRY_ENABLED=true
MICROSERVICES_RETRY_MAX_ATTEMPTS=3
MICROSERVICES_RETRY_BACKOFF_FACTOR=2.0

# Monitoring
MICROSERVICES_PROMETHEUS_ENABLED=true
MICROSERVICES_PROMETHEUS_PORT=9090

# Tracing
MICROSERVICES_OPENTELEMETRY_ENABLED=true
MICROSERVICES_OPENTELEMETRY_ENDPOINT=http://localhost:4317

# Logging
MICROSERVICES_STRUCTURED_LOGGING=true
MICROSERVICES_LOG_LEVEL=INFO

# Security
MICROSERVICES_OAUTH2_ENABLED=true
MICROSERVICES_OAUTH2_PROVIDER=google  # google, github, auth0
MICROSERVICES_RATE_LIMITING_ENABLED=true
MICROSERVICES_DDOS_PROTECTION_ENABLED=true
```

## 🛡️ Middleware Avanzado

El middleware avanzado incluye:

- **Structured Logging**: Logging estructurado en formato JSON
- **Distributed Tracing**: OpenTelemetry para tracing distribuido
- **Performance Monitoring**: Monitoreo de tiempos de respuesta
- **Security Headers**: Headers de seguridad automáticos

### Uso

Se configura automáticamente al usar `setup_microservices_app()`:

```python
from core.microservices_integration import setup_microservices_app

app = FastAPI()
app = setup_microservices_app(app)
```

## 🔄 Circuit Breakers

Los circuit breakers protegen contra fallos en cascada:

```python
from core.circuit_breaker import get_circuit_breaker, circuit_breaker

# Uso como decorator
@circuit_breaker(name="external_api", failure_threshold=5, timeout=60)
async def call_external_api():
    # Tu código aquí
    pass

# Uso directo
breaker = get_circuit_breaker("external_api")
result = await breaker.call_async(my_function, arg1, arg2)
```

## 🔁 Retry Logic

Lógica de reintentos con backoff exponencial:

```python
from core.retry_logic import retry, RetryConfig

# Uso como decorator
@retry(max_attempts=3, backoff_factor=2.0, initial_delay=1.0)
async def unreliable_function():
    # Tu código aquí
    pass

# Uso directo
from core.retry_logic import retry_async

config = RetryConfig(max_attempts=3, backoff_factor=2.0)
result = await retry_async(my_function, config=config, arg1=value1)
```

## 💾 Redis Cache

Cliente Redis para cache distribuido:

```python
from core.redis_client import get_redis_client

redis_client = get_redis_client()

# Sync
value = redis_client.get("key")
redis_client.set("key", {"data": "value"}, ttl=3600)

# Async
value = await redis_client.aget("key")
await redis_client.aset("key", {"data": "value"}, ttl=3600)

# Pub/Sub
await redis_client.publish("channel", {"event": "data"})
async for message in redis_client.subscribe("channel"):
    print(message)
```

## 📊 Prometheus Metrics

Métricas Prometheus para monitoreo:

```python
from core.prometheus_metrics import (
    record_project_generation,
    record_cache_operation,
    record_worker_task,
    update_queue_size
)

# Registrar métricas
record_project_generation("success", "chat", "fastapi", duration=5.2)
record_cache_operation("redis", hit=True)
record_worker_task("generate_project", "success", duration=10.5)
update_queue_size(5)
```

Las métricas están disponibles en `/metrics`.

## 🌐 API Gateway

Integración con API Gateways:

```python
from core.api_gateway import get_api_gateway_client

gateway = get_api_gateway_client()

# Registrar servicio
await gateway.register_service(
    service_name="ai-project-generator",
    service_url="http://localhost:8020",
    routes=[
        {
            "paths": ["/api/v1"],
            "methods": ["GET", "POST"]
        }
    ]
)

# Configurar rate limiting
await gateway.configure_rate_limit("ai-project-generator", limit=100, window=60)
```

## 📨 Message Brokers

Integración con message brokers para arquitectura event-driven:

```python
from core.message_broker import get_message_broker

broker = get_message_broker()

# Publicar evento
await broker.publish("project.created", {
    "project_id": "123",
    "status": "completed"
})

# Suscribirse a eventos
async def handle_event(event):
    print(f"Received event: {event}")

await broker.subscribe("project.created", handle_event)
```

## 👷 Async Workers

Workers asíncronos para tareas en background:

```python
from core.async_workers import get_worker_manager, task

# Definir tarea
@task()
def generate_project_task(project_id: str):
    # Tu código aquí
    return result

# Encolar tarea
worker_mgr = get_worker_manager()
task_id = worker_mgr.enqueue_task(generate_project_task, "project-123")

# Obtener resultado
result = worker_mgr.get_task_result(task_id)
```

## ☁️ Serverless Optimizations

Optimizaciones para entornos serverless:

```python
from core.serverless_optimizer import get_serverless_optimizer

optimizer = get_serverless_optimizer()

# Crear handler Lambda
lambda_handler = optimizer.create_lambda_handler(app)

# Obtener recomendaciones
recommendations = optimizer.get_optimization_recommendations()
```

## 🔐 OAuth2 Security

Autenticación OAuth2:

```python
from core.oauth2_security import (
    get_oauth2_manager,
    get_current_user,
    require_scopes
)

# Dependency para obtener usuario actual
@app.get("/protected")
async def protected_route(current_user = Depends(get_current_user)):
    return {"user": current_user.username}

# Requerir scopes específicos
@app.post("/admin")
@require_scopes("admin", "write")
async def admin_route(current_user = Depends(get_current_user)):
    return {"message": "Admin access"}
```

## 🚀 Integración Completa

Para usar todas las características juntas:

```python
from core.microservices_integration import get_microservices_integration

integration = get_microservices_integration()

# Cache
value = await integration.get_cache("key")
await integration.set_cache("key", {"data": "value"})

# Events
await integration.publish_event("project.created", {"id": "123"})

# Workers
task_id = integration.enqueue_task(my_task, arg1, arg2)
```

## 📝 Notas

- Todas las características son opcionales y se activan mediante configuración
- Los componentes fallan gracefully si no están disponibles
- Las dependencias opcionales se importan con try/except
- La configuración se puede hacer mediante variables de entorno o código

## 🔗 Referencias

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenTelemetry](https://opentelemetry.io/)
- [Prometheus](https://prometheus.io/)
- [Celery](https://docs.celeryproject.org/)
- [Redis](https://redis.io/)
- [RabbitMQ](https://www.rabbitmq.com/)
- [Kafka](https://kafka.apache.org/)















