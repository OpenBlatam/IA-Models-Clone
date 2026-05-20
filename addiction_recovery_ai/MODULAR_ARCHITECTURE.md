# Modular Architecture Documentation

## 🏗️ Arquitectura Modular

La aplicación ha sido reorganizada siguiendo principios de arquitectura modular, separación de concerns, y dependency injection.

## 📁 Estructura de Módulos

```
addiction_recovery_ai/
├── core/                      # Core abstracciones e interfaces
│   ├── interfaces.py         # Interfaces y contratos
│   └── service_container.py  # Dependency injection container
│
├── infrastructure/            # Implementaciones de infraestructura
│   ├── storage.py            # Storage services (DynamoDB, SQLite)
│   ├── cache.py              # Cache services (Redis, Memory)
│   ├── messaging.py         # Messaging (SQS, SNS)
│   ├── observability.py     # Observability (CloudWatch, Prometheus)
│   └── security.py          # Security (JWT, OAuth2)
│
├── handlers/                  # Handlers modulares
│   ├── event_handlers.py    # Event handlers
│   └── task_handlers.py     # Background task handlers
│
├── api/                       # API endpoints
├── services/                  # Business logic services
├── middleware/               # Middleware components
└── aws/                       # AWS-specific implementations
```

## 🔌 Interfaces y Abstracciones

### Core Interfaces (`core/interfaces.py`)

Define contratos para todos los servicios:

- **IStorageService**: Contrato para servicios de almacenamiento
- **ICacheService**: Contrato para servicios de caché
- **IFileStorageService**: Contrato para almacenamiento de archivos
- **IMessageQueueService**: Contrato para colas de mensajes
- **INotificationService**: Contrato para notificaciones
- **IMetricsService**: Contrato para métricas
- **ITracingService**: Contrato para tracing
- **IAuthenticationService**: Contrato para autenticación

### Ventajas

✅ **Desacoplamiento**: Los servicios no dependen de implementaciones específicas
✅ **Testabilidad**: Fácil mockear interfaces en tests
✅ **Flexibilidad**: Cambiar implementaciones sin afectar código cliente
✅ **Principio de Inversión de Dependencias**: Depender de abstracciones, no de concreciones

## 🏭 Service Factories

### Storage Factory (`infrastructure/storage.py`)

```python
from infrastructure.storage import StorageServiceFactory

# Crear servicio de storage
storage = StorageServiceFactory.create("dynamodb")  # o "sqlite"
```

**Implementaciones disponibles:**
- `DynamoDBStorageService`: Para producción AWS
- `SQLiteStorageService`: Para desarrollo local

### Cache Factory (`infrastructure/cache.py`)

```python
from infrastructure.cache import CacheServiceFactory

# Crear servicio de cache
cache = CacheServiceFactory.create("redis")  # o "memory"
```

**Implementaciones disponibles:**
- `RedisCacheService`: Para producción
- `InMemoryCacheService`: Para desarrollo/testing

### Messaging Factory (`infrastructure/messaging.py`)

```python
from infrastructure.messaging import MessagingServiceFactory

# Crear servicios de messaging
queue = MessagingServiceFactory.create_queue_service("sqs")
notification = MessagingServiceFactory.create_notification_service("sns")
```

### Observability Factory (`infrastructure/observability.py`)

```python
from infrastructure.observability import ObservabilityServiceFactory

# Crear servicios de observabilidad
metrics = ObservabilityServiceFactory.create_metrics_service("cloudwatch")
tracing = ObservabilityServiceFactory.create_tracing_service("opentelemetry")
```

### Security Factory (`infrastructure/security.py`)

```python
from infrastructure.security import SecurityServiceFactory

# Crear servicio de autenticación
auth = SecurityServiceFactory.create_authentication_service("jwt")
```

## 🎯 Service Container

### Dependency Injection (`core/service_container.py`)

El `ServiceContainer` centraliza la gestión de servicios:

```python
from core.service_container import get_container

# Obtener servicios
container = get_container()
storage = container.get_storage_service()
cache = container.get_cache_service()
metrics = container.get_metrics_service()
```

### FastAPI Integration

```python
from fastapi import Depends
from core.service_container import get_storage_service

@router.get("/users/{user_id}")
async def get_user(
    user_id: str,
    storage: IStorageService = Depends(get_storage_service)
):
    return await storage.get(user_id)
```

## 📨 Event Handlers

### Event Handler Registry (`handlers/event_handlers.py`)

```python
from handlers.event_handlers import EventHandlerRegistry, register_event_handlers

# Crear registry
registry = EventHandlerRegistry()

# Registrar handlers
register_event_handlers(registry)

# Manejar evento
result = await registry.handle({
    "event_type": "user.milestone",
    "data": {"user_id": "123", "milestone": "30_days"}
})
```

**Handlers disponibles:**
- `UserMilestoneEventHandler`: Maneja milestones de usuarios
- `ProgressUpdateEventHandler`: Maneja actualizaciones de progreso

## 🔧 Task Handlers

### Task Handler Registry (`handlers/task_handlers.py`)

```python
from handlers.task_handlers import TaskHandlerRegistry, register_task_handlers

# Crear registry
registry = TaskHandlerRegistry()

# Registrar handlers
register_task_handlers(registry)

# Ejecutar tarea
result = await registry.execute("generate_report", {
    "user_id": "123",
    "report_type": "monthly"
})
```

**Handlers disponibles:**
- `GenerateReportTask`: Genera reportes
- `SendNotificationTask`: Envía notificaciones
- `UpdateAnalyticsTask`: Actualiza analytics

## 🔄 Flujo de Datos

### Request Flow

```
API Request
    ↓
Middleware (Auth, Observability)
    ↓
Router Handler
    ↓
Service (Business Logic)
    ↓
Infrastructure Service (Storage, Cache, etc.)
    ↓
External Service (DynamoDB, S3, etc.)
```

### Event Flow

```
Business Event
    ↓
Event Publisher
    ↓
Message Queue (SQS)
    ↓
Event Processor
    ↓
Event Handler Registry
    ↓
Specific Handler
    ↓
Services
```

## 🧪 Testing

### Mocking Interfaces

```python
from unittest.mock import Mock
from core.interfaces import IStorageService

# Mock storage service
mock_storage = Mock(spec=IStorageService)
mock_storage.get.return_value = {"id": "123", "name": "Test"}

# Use in tests
result = await mock_storage.get("123")
```

### Service Container Testing

```python
from core.service_container import ServiceContainer

# Create test container
container = ServiceContainer()

# Register mock services
container.register_service("storage", mock_storage)

# Use in tests
storage = container.get_storage_service()
```

## 📊 Ventajas de la Arquitectura Modular

### 1. Separación de Concerns

- **Core**: Abstracciones y contratos
- **Infrastructure**: Implementaciones concretas
- **Handlers**: Lógica de procesamiento
- **API**: Endpoints y routing

### 2. Testabilidad

- Interfaces fáciles de mockear
- Servicios independientes
- Tests unitarios aislados

### 3. Flexibilidad

- Cambiar implementaciones sin afectar código cliente
- Soporte para múltiples backends
- Fácil agregar nuevos servicios

### 4. Mantenibilidad

- Código organizado y claro
- Responsabilidades bien definidas
- Fácil de entender y modificar

### 5. Escalabilidad

- Servicios independientes
- Fácil agregar nuevos módulos
- Soporte para microservicios

## 🚀 Uso en Producción

### Configuración por Entorno

```python
# Development
storage = StorageServiceFactory.create("sqlite")
cache = CacheServiceFactory.create("memory")

# Production
storage = StorageServiceFactory.create("dynamodb")
cache = CacheServiceFactory.create("redis")
```

### Service Container Global

```python
# En main.py
from core.service_container import get_container

container = get_container()

# Servicios disponibles automáticamente
storage = container.get_storage_service()
cache = container.get_cache_service()
```

## 📝 Ejemplo Completo

```python
from fastapi import FastAPI, Depends
from core.service_container import (
    get_storage_service,
    get_cache_service,
    get_metrics_service
)
from core.interfaces import IStorageService, ICacheService, IMetricsService

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(
    user_id: str,
    storage: IStorageService = Depends(get_storage_service),
    cache: ICacheService = Depends(get_cache_service),
    metrics: IMetricsService = Depends(get_metrics_service)
):
    # Check cache first
    cached = await cache.get(f"user_{user_id}")
    if cached:
        await metrics.increment("cache_hits")
        return cached
    
    # Get from storage
    user = await storage.get(user_id)
    
    # Cache result
    if user:
        await cache.set(f"user_{user_id}", user, ttl=300)
        await metrics.increment("cache_misses")
    
    return user
```

## ✅ Checklist de Modularidad

- [x] Interfaces definidas para todos los servicios
- [x] Factories para creación de servicios
- [x] Service container para dependency injection
- [x] Handlers modulares y registrables
- [x] Separación clara de concerns
- [x] Soporte para múltiples backends
- [x] Fácil de testear
- [x] Documentación completa

---

**Arquitectura modular completada** ✅

Sistema organizado, mantenible y escalable con separación clara de responsabilidades.
