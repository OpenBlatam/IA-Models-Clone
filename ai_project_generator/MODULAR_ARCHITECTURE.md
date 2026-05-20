# Arquitectura Modular

Este documento describe la arquitectura modular implementada siguiendo principios de microservicios.

## 📁 Estructura de Directorios

```
ai_project_generator/
├── api/                    # Capa de API
│   ├── routes/            # Endpoints organizados por dominio
│   │   ├── projects.py    # Endpoints de proyectos
│   │   ├── generation.py  # Endpoints de generación
│   │   ├── validation.py  # Endpoints de validación
│   │   ├── export.py      # Endpoints de exportación
│   │   ├── deployment.py  # Endpoints de despliegue
│   │   └── analytics.py   # Endpoints de analytics
│   ├── app_factory.py     # Factory para crear app FastAPI
│   └── generator_api.py   # API legacy (compatibilidad)
│
├── services/               # Servicios de negocio
│   ├── project_service.py      # Gestión de proyectos
│   ├── generation_service.py   # Generación de proyectos
│   ├── validation_service.py   # Validación de proyectos
│   ├── export_service.py        # Exportación de proyectos
│   ├── deployment_service.py    # Despliegue de proyectos
│   └── analytics_service.py     # Analytics
│
├── infrastructure/         # Infraestructura compartida
│   ├── cache.py          # Servicio de cache
│   ├── events.py         # Servicio de eventos
│   ├── workers.py        # Servicio de workers
│   ├── monitoring.py     # Servicio de monitoreo
│   ├── security.py       # Servicio de seguridad
│   └── dependencies.py   # Dependencies para FastAPI
│
├── domain/                # Modelos de dominio
│   └── models.py         # Modelos Pydantic
│
└── core/                  # Core (lógica compartida)
    ├── project_generator.py
    ├── continuous_generator.py
    └── ... (otros módulos core)
```

## 🏗️ Principios de Diseño

### 1. Separación de Responsabilidades

- **API Layer** (`api/routes/`): Solo maneja HTTP, validación de requests, y respuestas
- **Service Layer** (`services/`): Contiene lógica de negocio
- **Infrastructure Layer** (`infrastructure/`): Abstrae infraestructura (cache, workers, etc.)
- **Domain Layer** (`domain/`): Modelos de dominio

### 2. Inyección de Dependencias

Todos los servicios usan inyección de dependencias a través de FastAPI:

```python
from fastapi import Depends
from infrastructure.dependencies import get_project_service

@router.get("/{project_id}")
async def get_project(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service)
):
    return await project_service.get_project(project_id)
```

### 3. Abstracción de Infraestructura

Los servicios no dependen directamente de implementaciones específicas:

```python
# En lugar de usar Redis directamente
class ProjectService:
    def __init__(self, cache_service: CacheService):
        self.cache = cache_service  # Abstracción
```

### 4. Event-Driven Architecture

Los servicios publican eventos para comunicación desacoplada:

```python
await event_publisher.publish("project.created", {
    "project_id": "123",
    "status": "completed"
})
```

## 🔄 Flujo de Datos

```
HTTP Request
    ↓
API Route (api/routes/)
    ↓
Service (services/)
    ↓
Infrastructure (infrastructure/)
    ↓
Core/Utils (core/, utils/)
```

## 📦 Servicios

### ProjectService

**Responsabilidades:**
- Crear proyectos
- Obtener información de proyectos
- Listar proyectos
- Eliminar proyectos
- Gestionar cola

**Dependencias:**
- ProjectGenerator
- ContinuousGenerator
- CacheService
- EventPublisher

### GenerationService

**Responsabilidades:**
- Generar proyectos
- Generación asíncrona
- Batch generation

**Dependencias:**
- ProjectGenerator
- WorkerService
- EventPublisher
- CacheService

### ValidationService

**Responsabilidades:**
- Validar proyectos
- Verificar estructura
- Análisis de calidad

**Dependencias:**
- ProjectValidator

### ExportService

**Responsabilidades:**
- Exportar proyectos
- Múltiples formatos (ZIP, TAR)

**Dependencias:**
- ExportGenerator

### DeploymentService

**Responsabilidades:**
- Desplegar proyectos
- Múltiples plataformas

**Dependencias:**
- DeploymentGenerator

### AnalyticsService

**Responsabilidades:**
- Analytics y métricas
- Estadísticas

**Dependencias:**
- AnalyticsEngine

## 🏭 Infraestructura

### CacheService

Abstrae el backend de cache (Redis, Memcached, in-memory).

**Interfaz:**
```python
async def get(key: str) -> Optional[Any]
async def set(key: str, value: Any, ttl: Optional[int]) -> bool
async def delete(key: str) -> bool
```

### EventPublisher

Abstrae el message broker (RabbitMQ, Kafka, Redis).

**Interfaz:**
```python
async def publish(event_type: str, event_data: Dict[str, Any]) -> bool
```

### WorkerService

Abstrae los workers asíncronos (Celery, RQ, ARQ).

**Interfaz:**
```python
def enqueue_task(task_func, *args, **kwargs) -> Optional[str]
def get_task_status(task_id: str) -> Optional[Dict[str, Any]]
```

## 🚀 Uso

### Crear Aplicación Modular

```python
from api.app_factory import create_app

app = create_app(
    base_output_dir="generated_projects",
    enable_continuous=True
)
```

### Agregar Nuevo Endpoint

1. Crear servicio en `services/`:
```python
class MyService:
    async def do_something(self):
        pass
```

2. Crear router en `api/routes/`:
```python
router = APIRouter(prefix="/api/v1/my", tags=["my"])

@router.get("")
async def endpoint(service: MyService = Depends(get_my_service)):
    return await service.do_something()
```

3. Agregar dependency en `infrastructure/dependencies.py`:
```python
def get_my_service() -> MyService:
    return MyService()
```

4. Registrar router en `api/app_factory.py`:
```python
from .routes.my import router as my_router
app.include_router(my_router)
```

## ✅ Ventajas de la Arquitectura Modular

1. **Separación de Responsabilidades**: Cada módulo tiene una responsabilidad clara
2. **Testabilidad**: Fácil de testear cada componente por separado
3. **Escalabilidad**: Cada servicio puede escalarse independientemente
4. **Mantenibilidad**: Código organizado y fácil de mantener
5. **Reutilización**: Servicios pueden reutilizarse en diferentes contextos
6. **Desacoplamiento**: Servicios no dependen de implementaciones específicas
7. **Extensibilidad**: Fácil agregar nuevos servicios y endpoints

## 🔄 Migración desde Código Legacy

El código legacy en `api/generator_api.py` sigue funcionando gracias a la función `create_generator_app()` que ahora usa el factory modular internamente.

## 📝 Próximos Pasos

1. Migrar más endpoints a la estructura modular
2. Agregar más servicios según necesidad
3. Implementar tests para cada servicio
4. Documentar APIs con OpenAPI/Swagger
5. Agregar versionado de API















