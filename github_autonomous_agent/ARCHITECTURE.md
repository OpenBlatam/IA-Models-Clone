# 🏗️ Architecture Guide - GitHub Autonomous Agent

> Guía detallada de la arquitectura del GitHub Autonomous Agent

## 📐 Visión General

El GitHub Autonomous Agent sigue una **arquitectura modular** basada en **Clean Architecture** y **Domain-Driven Design (DDD)**, con separación clara de responsabilidades y alta testabilidad.

## 🎯 Principios de Diseño

1. **Separación de Responsabilidades**: Cada capa tiene una responsabilidad única
2. **Dependency Inversion**: Las capas superiores no dependen de las inferiores
3. **Testabilidad**: Cada componente puede ser testeado de forma aislada
4. **Escalabilidad**: Arquitectura preparada para crecer
5. **Mantenibilidad**: Código organizado y fácil de entender

## 🏛️ Arquitectura por Capas

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                        │
│                  (API Routes, Middleware)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  Application Layer                           │
│              (Use Cases, Business Logic)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                     Domain Layer                             │
│            (Core Business Logic, Entities)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                Infrastructure Layer                          │
│    (GitHub Client, Storage, Queue, External Services)        │
└──────────────────────────────────────────────────────────────┘
```

## 📦 Estructura de Directorios

```
github_autonomous_agent/
├── api/                      # Presentation Layer
│   ├── routes/              # FastAPI routers
│   ├── schemas.py           # Pydantic models (DTOs)
│   ├── dependencies.py      # FastAPI dependencies
│   ├── middleware.py        # Custom middleware
│   └── validators.py        # Request validators
│
├── application/             # Application Layer
│   └── use_cases/          # Business use cases
│       ├── github_use_cases.py
│       └── task_use_cases.py
│
├── core/                    # Domain & Infrastructure
│   ├── github_client.py     # GitHub API client
│   ├── task_processor.py    # Task processing logic
│   ├── worker.py            # Background worker
│   ├── storage.py           # Data persistence
│   ├── di/                  # Dependency Injection
│   │   └── container.py
│   └── services/           # Core services
│       └── cache_service.py
│
└── config/                  # Configuration
    ├── settings.py         # App settings
    ├── di_setup.py         # DI configuration
    └── logging_config.py   # Logging setup
```

## 🔄 Flujo de Datos

### 1. Request Flow

```
Client Request
    ↓
FastAPI Middleware (CORS, Logging, Error Handling)
    ↓
API Route Handler
    ↓
Dependency Injection (Resolve Dependencies)
    ↓
Use Case (Business Logic)
    ↓
Domain Service / Repository
    ↓
Infrastructure (GitHub API, Database, Queue)
    ↓
Response
```

### 2. Task Processing Flow

```
Frontend → API → Create Task
    ↓
Task Queue (Celery)
    ↓
Worker Process
    ↓
Task Processor
    ↓
GitHub Client
    ↓
Storage (Save Results)
    ↓
Notification (Update Frontend)
```

## 🧩 Componentes Principales

### API Layer (`api/`)

**Responsabilidad**: Manejar requests HTTP, validación, serialización

**Componentes**:
- **Routes**: Endpoints REST (`agent_routes`, `github_routes`, `task_routes`)
- **Schemas**: Pydantic models para request/response
- **Middleware**: CORS, logging, error handling
- **Dependencies**: Dependency injection para FastAPI

**Ejemplo**:
```python
# api/routes/task_routes.py
@router.post("/tasks")
async def create_task(
    task_data: TaskCreateSchema,
    use_case: TaskUseCase = Depends(get_task_use_case)
):
    return await use_case.create_task(task_data)
```

### Application Layer (`application/`)

**Responsabilidad**: Orquestar casos de uso y lógica de negocio

**Componentes**:
- **Use Cases**: Lógica de negocio específica
  - `GitHubUseCase`: Operaciones con GitHub
  - `TaskUseCase`: Gestión de tareas

**Ejemplo**:
```python
# application/use_cases/task_use_cases.py
class TaskUseCase:
    def __init__(self, task_repo, github_client):
        self.task_repo = task_repo
        self.github_client = github_client
    
    async def create_task(self, task_data):
        # Business logic here
        task = Task.create(task_data)
        await self.task_repo.save(task)
        return task
```

### Core Layer (`core/`)

**Responsabilidad**: Lógica de dominio e infraestructura

**Componentes**:
- **GitHub Client**: Cliente para GitHub API
- **Task Processor**: Procesamiento de tareas
- **Worker**: Worker de Celery para tareas asíncronas
- **Storage**: Persistencia de datos
- **Services**: Servicios core (cache, etc.)

### Infrastructure

**Componentes Externos**:
- **Database**: SQLite (dev) / PostgreSQL (prod)
- **Queue**: Redis + Celery
- **GitHub API**: API externa
- **Storage**: Sistema de archivos local

## 🔌 Dependency Injection

El proyecto usa **Dependency Injection** para desacoplar componentes:

```python
# config/di_setup.py
container = Container()

# Registrar dependencias
container.register(GithubClient, GithubClientImpl)
container.register(TaskRepository, TaskRepositoryImpl)
container.register(TaskUseCase, TaskUseCase)

# Resolver en FastAPI
def get_task_use_case() -> TaskUseCase:
    return container.resolve(TaskUseCase)
```

## 🔄 Task Queue Architecture

```
┌─────────────┐
│   FastAPI   │
│   (API)     │
└──────┬──────┘
       │ Create Task
       ▼
┌─────────────┐
│   Celery    │
│   Broker    │
│  (Redis)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Worker    │
│  (Celery)   │
└──────┬──────┘
       │ Process
       ▼
┌─────────────┐
│   Storage   │
│  (Database) │
└─────────────┘
```

## 🗄️ Data Flow

### Task Lifecycle

```
CREATED → PENDING → QUEUED → RUNNING → COMPLETED
                                    ↓
                                 FAILED
                                    ↓
                                RETRY → RUNNING
```

### State Management

- **In-Memory**: Estado temporal del agente
- **Database**: Estado persistente de tareas
- **Redis**: Cache y cola de tareas
- **File System**: Logs y archivos temporales

## 🔒 Security Architecture

```
┌─────────────────────────────────────┐
│         Client Request              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      Rate Limiting Middleware       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      Authentication Middleware      │
│         (JWT Validation)            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      Authorization Check            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│         Request Handler              │
└──────────────────────────────────────┘
```

## 📊 Monitoring & Observability

### Logging

- **Structured Logging**: Logs en formato JSON
- **Log Levels**: DEBUG, INFO, WARNING, ERROR
- **Log Storage**: Archivos locales + opcionalmente centralizado

### Metrics

- **Prometheus**: Métricas del sistema (opcional)
- **Custom Metrics**: Tareas, errores, performance

### Tracing

- **Request IDs**: Tracking de requests
- **Task IDs**: Tracking de tareas
- **Correlation IDs**: Relacionar eventos

## 🚀 Escalabilidad

### Horizontal Scaling

- **API Servers**: Múltiples instancias de FastAPI
- **Workers**: Múltiples workers de Celery
- **Load Balancer**: Nginx o similar

### Vertical Scaling

- **Database**: Optimización de queries
- **Cache**: Redis para datos frecuentes
- **Connection Pooling**: Pool de conexiones a DB

## 🔄 Error Handling

### Strategy

1. **Validation Errors**: 400 Bad Request
2. **Authentication Errors**: 401 Unauthorized
3. **Authorization Errors**: 403 Forbidden
4. **Not Found**: 404 Not Found
5. **Rate Limiting**: 429 Too Many Requests
6. **Server Errors**: 500 Internal Server Error

### Retry Logic

- **Exponential Backoff**: Para errores transitorios
- **Max Retries**: Límite de reintentos
- **Dead Letter Queue**: Tareas fallidas persistentemente

## 📝 Patrones de Diseño Utilizados

1. **Repository Pattern**: Abstracción de acceso a datos
2. **Dependency Injection**: Inversión de dependencias
3. **Factory Pattern**: Creación de objetos complejos
4. **Strategy Pattern**: Algoritmos intercambiables
5. **Observer Pattern**: Notificaciones de eventos
6. **Singleton Pattern**: Configuración global

## 🔍 Decisiones de Arquitectura

### ¿Por qué FastAPI?
- Async/await nativo
- Validación automática con Pydantic
- Documentación automática
- Alto performance

### ¿Por qué Celery?
- Tareas asíncronas robustas
- Escalabilidad horizontal
- Retry automático
- Monitoreo con Flower

### ¿Por qué Clean Architecture?
- Testabilidad
- Mantenibilidad
- Escalabilidad
- Independencia de frameworks

## 📚 Referencias

- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Domain-Driven Design](https://martinfowler.com/bliki/DomainDrivenDesign.html)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)
- [Celery Documentation](https://docs.celeryq.dev/)

---

**Para más detalles, consulta la [documentación completa](README.md)**



