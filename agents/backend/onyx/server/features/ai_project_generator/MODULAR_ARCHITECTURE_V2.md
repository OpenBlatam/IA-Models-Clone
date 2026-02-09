# Arquitectura Modular V2 - Ultra Modular

Arquitectura completamente modular siguiendo principios avanzados de microservicios.

## 🏗️ Estructura Modular

```
ai_project_generator/
├── interfaces/          # Contratos y abstracciones
│   ├── repository.py    # IRepository, IProjectRepository
│   ├── service.py       # IService, IProjectService, IGenerationService
│   ├── cache.py         # ICacheService
│   ├── events.py        # IEventPublisher, IEventSubscriber
│   └── workers.py       # IWorkerService
│
├── repositories/         # Implementaciones de repositorios
│   ├── project_repository.py    # Usando ContinuousGenerator
│   └── memory_repository.py     # En memoria (testing)
│
├── services/            # Servicios de negocio
│   ├── project_service.py
│   ├── generation_service.py
│   └── ...
│
├── factories/           # Factories para crear instancias
│   ├── repository_factory.py
│   ├── service_factory.py
│   └── infrastructure_factory.py
│
├── strategies/          # Patrones Strategy
│   ├── generation_strategy.py
│   └── cache_strategy.py
│
├── config/              # Configuración modular
│   ├── app_config.py
│   ├── service_config.py
│   └── infrastructure_config.py
│
├── infrastructure/      # Servicios de infraestructura
│   ├── cache.py
│   ├── events.py
│   └── workers.py
│
├── domain/              # Modelos de dominio
│   └── models.py
│
└── api/                 # Capa de API
    ├── routes/
    └── app_factory.py
```

## 🎯 Principios Aplicados

### 1. Dependency Inversion (SOLID)

**Interfaces definen contratos:**

```python
from interfaces.repository import IProjectRepository
from interfaces.service import IProjectService

# Servicios dependen de interfaces, no de implementaciones
class ProjectService:
    def __init__(self, repository: IProjectRepository):
        self.repository = repository
```

### 2. Factory Pattern

**Factories crean instancias:**

```python
from factories.service_factory import ServiceFactory

# Factory crea servicio con todas sus dependencias
service = ServiceFactory.create_project_service()
```

### 3. Strategy Pattern

**Estrategias intercambiables:**

```python
from strategies.generation_strategy import (
    SyncGenerationStrategy,
    AsyncGenerationStrategy
)

# Puedes cambiar la estrategia sin cambiar el código
strategy = SyncGenerationStrategy(project_generator)
# o
strategy = AsyncGenerationStrategy(worker_service)
```

### 4. Repository Pattern

**Repositorios abstraen acceso a datos:**

```python
from repositories import ProjectRepository, MemoryProjectRepository

# Puedes cambiar el repositorio sin cambiar el servicio
repository = ProjectRepository(continuous_generator)
# o para testing
repository = MemoryProjectRepository()
```

### 5. Separation of Concerns

**Cada módulo tiene una responsabilidad:**

- `interfaces/` - Define contratos
- `repositories/` - Acceso a datos
- `services/` - Lógica de negocio
- `factories/` - Creación de instancias
- `strategies/` - Algoritmos intercambiables
- `config/` - Configuración
- `infrastructure/` - Infraestructura
- `domain/` - Modelos de dominio
- `api/` - Capa de presentación

## 📦 Módulos

### Interfaces

Define contratos que deben cumplir las implementaciones:

```python
from interfaces.repository import IProjectRepository
from interfaces.service import IProjectService
from interfaces.cache import ICacheService
```

### Repositories

Implementaciones de repositorios:

```python
# Repositorio usando ContinuousGenerator
repository = ProjectRepository(continuous_generator)

# Repositorio en memoria (testing)
repository = MemoryProjectRepository()
```

### Services

Servicios de negocio que dependen de interfaces:

```python
from services.project_service import ProjectService

service = ProjectService(
    repository=repository,
    cache_service=cache_service,
    event_publisher=event_publisher
)
```

### Factories

Crean instancias con dependencias resueltas:

```python
from factories.service_factory import ServiceFactory

# Factory resuelve todas las dependencias automáticamente
service = ServiceFactory.create_project_service()
```

### Strategies

Estrategias intercambiables:

```python
from strategies.generation_strategy import SyncGenerationStrategy

strategy = SyncGenerationStrategy(project_generator)
result = await strategy.generate(description="...")
```

### Config

Configuración separada por módulos:

```python
from config.app_config import get_app_config
from config.service_config import get_service_config
from config.infrastructure_config import get_infrastructure_config

app_config = get_app_config()
service_config = get_service_config()
infra_config = get_infrastructure_config()
```

## 🔄 Flujo de Datos

```
HTTP Request
    ↓
API Route (api/routes/)
    ↓
Service (services/) ← depende de → Interface (interfaces/)
    ↓                                    ↑
Repository (repositories/) ─────────────┘
    ↓
Data Source (ContinuousGenerator, Memory, etc.)
```

## 🎨 Uso

### Crear Servicio con Factory

```python
from factories.service_factory import ServiceFactory

# Factory resuelve todas las dependencias
service = ServiceFactory.create_project_service()
project = await service.create_project("description", "name")
```

### Usar Repository Directamente

```python
from factories.repository_factory import RepositoryFactory

repository = RepositoryFactory.create_project_repository_auto()
projects = await repository.list(filters={"status": "completed"})
```

### Cambiar Estrategia

```python
from strategies.generation_strategy import (
    SyncGenerationStrategy,
    AsyncGenerationStrategy
)

# Síncrono
strategy = SyncGenerationStrategy(project_generator)

# Asíncrono
strategy = AsyncGenerationStrategy(worker_service)
```

### Testing con Memory Repository

```python
from repositories.memory_repository import MemoryProjectRepository

# Usar repositorio en memoria para tests
repository = MemoryProjectRepository()
service = ProjectService(repository=repository)
```

## ✅ Ventajas

1. **Testabilidad**: Fácil mockear interfaces
2. **Flexibilidad**: Cambiar implementaciones sin cambiar código
3. **Mantenibilidad**: Cada módulo tiene responsabilidad clara
4. **Escalabilidad**: Fácil agregar nuevas implementaciones
5. **Desacoplamiento**: Módulos independientes
6. **Reutilización**: Interfaces y factories reutilizables

## 🔧 Extensibilidad

### Agregar Nuevo Repository

```python
# 1. Implementar interfaz
class DatabaseProjectRepository(IProjectRepository):
    async def get_by_id(self, id: str):
        # Implementación con base de datos
        pass

# 2. Agregar a factory
class RepositoryFactory:
    @staticmethod
    def create_project_repository(type="database"):
        if type == "database":
            return DatabaseProjectRepository()
```

### Agregar Nueva Estrategia

```python
# 1. Implementar interfaz
class DistributedGenerationStrategy(GenerationStrategy):
    async def generate(self, description, **kwargs):
        # Generación distribuida
        pass

# 2. Usar en servicio
strategy = DistributedGenerationStrategy()
```

## 📝 Testing

```python
# Test con memory repository
def test_project_service():
    repository = MemoryProjectRepository()
    service = ProjectService(repository=repository)
    
    project = await service.create_project("test", "test_project")
    assert project["project_id"] is not None
```

## 🚀 Próximos Pasos

1. Agregar más implementaciones de repositorios (Database, File, etc.)
2. Agregar más estrategias (Distributed, Cached, etc.)
3. Implementar Unit of Work pattern
4. Agregar Domain Events
5. Implementar CQRS pattern















