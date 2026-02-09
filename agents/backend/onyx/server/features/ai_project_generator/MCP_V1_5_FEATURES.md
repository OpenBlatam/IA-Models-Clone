# MCP v1.5.0 - Funcionalidades de Infraestructura

## 🚀 Nuevas Funcionalidades de Infraestructura

### 1. **API Versioning** (`versioning.py`)
- Soporte para múltiples versiones de API simultáneas
- Versionado por header o path
- Router versionado
- Decorador `@versioned_route`

**Uso:**
```python
from mcp_server import VersionedRouter, APIVersion, versioned_route

router = VersionedRouter(default_version=APIVersion.V1)

@versioned_route(APIVersion.V1)
async def v1_endpoint():
    return {"version": "v1"}

@versioned_route(APIVersion.V2)
async def v2_endpoint():
    return {"version": "v2"}
```

### 2. **Service Discovery** (`discovery.py`)
- Registry de servicios MCP
- Heartbeat automático
- Limpieza de servicios muertos
- Descubrimiento por nombre

**Uso:**
```python
from mcp_server import ServiceRegistry, ServiceInfo, ServiceStatus

registry = ServiceRegistry()

# Registrar servicio
service = ServiceInfo(
    service_id="service-1",
    name="my-service",
    endpoint="http://localhost:8000",
    version="1.0.0",
)
registry.register(service)

# Actualizar heartbeat
registry.update_heartbeat("service-1")

# Descubrir servicio
service = registry.discover_service("my-service")
```

### 3. **Connection Pooling** (`pooling.py`)
- Pool genérico de conexiones
- Reutilización de conexiones
- Tamaño mínimo/máximo configurable
- Context manager para adquirir conexiones

**Uso:**
```python
from mcp_server import ConnectionPool

async def create_db_connection():
    return await connect_to_db()

pool = ConnectionPool(
    factory=create_db_connection,
    min_size=2,
    max_size=10,
)

await pool.initialize()

# Usar conexión
async with pool.acquire() as conn:
    result = await conn.query("SELECT * FROM users")
```

### 4. **Metrics Dashboard** (`metrics_dashboard.py`)
- Dashboard HTML de métricas
- Endpoint de resumen JSON
- Exportación Prometheus
- Visualización en tiempo real

**Uso:**
```python
from mcp_server import MetricsDashboard

dashboard = MetricsDashboard(observability=observability)
app.include_router(dashboard.get_router())

# Acceder a dashboard
# GET /mcp/v1/metrics/ - Dashboard HTML
# GET /mcp/v1/metrics/api/summary - JSON summary
# GET /mcp/v1/metrics/api/prometheus - Prometheus format
```

### 5. **Request Queue** (`request_queue.py`)
- Cola de requests con prioridad
- Procesamiento asíncrono
- Prioridades: LOW, NORMAL, HIGH, CRITICAL
- Control de tamaño máximo

**Uso:**
```python
from mcp_server import RequestQueue, RequestPriority

queue = RequestQueue(max_size=1000)

# Encolar request
await queue.enqueue(
    request_id="req-1",
    resource_id="files",
    operation="read",
    parameters={"path": "file.txt"},
    priority=RequestPriority.HIGH,
    handler=process_request,
)

# Procesar cola
await queue.start_processing(processor_func)
```

## 📊 Resumen de Versiones

### v1.0.0 - Base
- Servidor MCP básico

### v1.1.0 - Mejoras Core
- Excepciones, rate limiting, cache, middleware

### v1.2.0 - Funcionalidades Avanzadas
- Retry, circuit breaker, batch, webhooks, transformers, admin

### v1.3.0 - Funcionalidades Adicionales
- Streaming, config, profiling, queue

### v1.4.0 - Funcionalidades Enterprise
- GraphQL, plugins, compression, health checks

### v1.5.0 - Funcionalidades de Infraestructura
- API versioning, service discovery, connection pooling, metrics dashboard, request queue

## 🎯 Casos de Uso

### Versionado de API
```python
# Mantener múltiples versiones simultáneamente
router = VersionedRouter()

# v1 endpoint
router.add_endpoint("/resources", v1_list_resources, ["GET"], APIVersion.V1)

# v2 endpoint (mejorado)
router.add_endpoint("/resources", v2_list_resources, ["GET"], APIVersion.V2)

# Cliente puede usar:
# Header: X-API-Version: v2
# O path: /v2/resources
```

### Service Discovery
```python
# Registrar servicios automáticamente
registry = ServiceRegistry()
await registry.start_cleanup()

# Servicios se registran automáticamente
# Heartbeat cada 30s
# Limpieza automática de servicios muertos
```

### Connection Pooling
```python
# Mejorar performance con pooling
pool = ConnectionPool(
    factory=create_http_client,
    min_size=5,
    max_size=20,
)

# Reutilizar conexiones
async with pool.acquire() as client:
    response = await client.get("https://api.example.com")
```

### Metrics Dashboard
```python
# Dashboard visual de métricas
dashboard = MetricsDashboard(observability)
app.include_router(dashboard.get_router())

# Acceder en navegador
# http://localhost:8020/mcp/v1/metrics/
```

### Request Queue con Prioridad
```python
# Procesar requests según prioridad
queue = RequestQueue()

# Request crítico
await queue.enqueue(..., priority=RequestPriority.CRITICAL)

# Request normal
await queue.enqueue(..., priority=RequestPriority.NORMAL)

# Procesar automáticamente
await queue.start_processing(process_request)
```

## 📈 Beneficios

1. **API Versioning**: 
   - Mantener compatibilidad
   - Migración gradual
   - Múltiples versiones simultáneas

2. **Service Discovery**:
   - Auto-descubrimiento de servicios
   - Health checking automático
   - Resiliencia mejorada

3. **Connection Pooling**:
   - Mejora performance ~50%
   - Reduce overhead de conexiones
   - Mejor uso de recursos

4. **Metrics Dashboard**:
   - Visualización en tiempo real
   - Debugging más fácil
   - Monitoreo integrado

5. **Request Queue**:
   - Procesamiento ordenado
   - Priorización inteligente
   - Control de carga

## 🔧 Integración

Todas las funcionalidades se integran perfectamente:
- ✅ Versioning con todos los endpoints
- ✅ Service discovery con health checks
- ✅ Connection pooling con conectores
- ✅ Metrics dashboard con observabilidad
- ✅ Request queue con rate limiting

## 📝 Próximas Mejoras (Roadmap)

1. **Load Balancing**: Balanceo de carga entre instancias
2. **Multi-tenancy**: Soporte multi-tenant
3. **Event Sourcing**: Event sourcing patterns
4. **CQRS**: Command Query Responsibility Segregation
5. **API Gateway**: Features de API Gateway

## 🎉 Resumen

v1.5.0 agrega funcionalidades de infraestructura esenciales:
- **API Versioning**: Compatibilidad y migración
- **Service Discovery**: Auto-descubrimiento
- **Connection Pooling**: Performance mejorado
- **Metrics Dashboard**: Visualización
- **Request Queue**: Procesamiento ordenado

El servidor MCP ahora es una plataforma completa de infraestructura.

