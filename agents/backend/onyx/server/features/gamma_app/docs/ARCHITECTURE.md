# Gamma App - Arquitectura

## Visión General

La aplicación Gamma App sigue una arquitectura en capas (Layered Architecture) con principios de Clean Architecture y Domain-Driven Design (DDD).

## Estructura de Capas

```
gamma_app/
├── domain/              # Capa de Dominio (Lógica de Negocio)
│   └── interfaces/      # Interfaces y contratos
│       ├── repositories.py  # Interfaces de repositorios
│       └── services.py      # Interfaces de servicios
│
├── application/         # Capa de Aplicación (Casos de Uso)
│   ├── di/            # Dependency Injection
│   │   └── container.py
│   └── unit_of_work.py # Patrón Unit of Work
│
├── infrastructure/     # Capa de Infraestructura
│   ├── database/       # Gestión de base de datos
│   │   └── session.py
│   └── repositories/  # Implementaciones de repositorios
│       ├── base.py
│       └── user_repository.py
│
├── api/               # Capa de Presentación (API)
│   ├── routes.py
│   ├── dependencies.py
│   └── lifespan.py
│
├── core/              # Lógica de negocio específica
├── services/          # Servicios de aplicación
└── models/            # Modelos de datos (SQLAlchemy)
```

## Principios Arquitectónicos

### 1. Separación de Responsabilidades

- **Domain Layer**: Contiene la lógica de negocio pura, sin dependencias de infraestructura
- **Application Layer**: Orquesta los casos de uso y coordina entre capas
- **Infrastructure Layer**: Implementa detalles técnicos (BD, cache, APIs externas)
- **Presentation Layer**: Maneja HTTP requests/responses

### 2. Dependency Inversion

Las capas superiores definen interfaces que las capas inferiores implementan:

```python
# Domain define la interfaz
class IUserRepository(ABC):
    @abstractmethod
    async def get_by_email(self, email: str) -> Optional[User]:
        pass

# Infrastructure implementa la interfaz
class UserRepository(IUserRepository):
    def __init__(self, session: Session):
        ...
```

### 3. Dependency Injection

El contenedor DI (`DIContainer`) gestiona el ciclo de vida de los servicios:

```python
from application.di.container import get_container

container = get_container()
content_generator = container.get(ContentGenerator)
```

### 4. Repository Pattern

Abstrae el acceso a datos mediante repositorios:

```python
from infrastructure.repositories.user_repository import UserRepository
from application.unit_of_work import unit_of_work

async with unit_of_work() as uow:
    user_repo = uow.get_repository(UserRepository)
    user = await user_repo.get_by_email("user@example.com")
    await uow.commit()
```

### 5. Unit of Work Pattern

Gestiona transacciones y coordina múltiples repositorios:

```python
async with unit_of_work() as uow:
    user_repo = uow.get_repository(UserRepository)
    project_repo = uow.get_repository(ProjectRepository)
    
    user = await user_repo.create(new_user)
    project = await project_repo.create(new_project)
    
    await uow.commit()  # Todo se commitea en una transacción
```

## Flujo de Datos

### Request Flow

1. **API Layer** (`api/routes.py`)
   - Recibe HTTP request
   - Valida con Pydantic models
   - Llama a servicios de aplicación

2. **Application Layer** (`application/`)
   - Orquesta casos de uso
   - Usa Unit of Work para transacciones
   - Llama a repositorios a través de interfaces

3. **Domain Layer** (`domain/`)
   - Contiene lógica de negocio
   - Define contratos (interfaces)

4. **Infrastructure Layer** (`infrastructure/`)
   - Implementa acceso a datos
   - Gestiona conexiones a BD, Redis, etc.

### Response Flow

1. **Infrastructure** → Retorna entidades de dominio
2. **Application** → Transforma a DTOs si es necesario
3. **API** → Serializa a JSON y retorna HTTP response

## Ventajas de esta Arquitectura

1. **Testabilidad**: Fácil mockear interfaces para tests unitarios
2. **Mantenibilidad**: Separación clara de responsabilidades
3. **Escalabilidad**: Fácil agregar nuevas funcionalidades
4. **Flexibilidad**: Cambiar implementaciones sin afectar otras capas
5. **Reutilización**: Interfaces permiten múltiples implementaciones

## Ejemplo de Uso

### Crear un nuevo repositorio

1. Definir interfaz en `domain/interfaces/repositories.py`:
```python
class IContentRepository(IRepository, ABC):
    @abstractmethod
    async def get_by_user_id(self, user_id: str) -> List[Content]:
        pass
```

2. Implementar en `infrastructure/repositories/`:
```python
class ContentRepository(BaseRepository[Content], IContentRepository):
    def __init__(self, session: Session):
        super().__init__(session, Content)
    
    async def get_by_user_id(self, user_id: str) -> List[Content]:
        return self.session.query(Content).filter(
            Content.creator_id == user_id
        ).all()
```

3. Usar en casos de uso:
```python
async with unit_of_work() as uow:
    content_repo = uow.get_repository(ContentRepository)
    contents = await content_repo.get_by_user_id(user_id)
    await uow.commit()
```

## Próximos Pasos

- [ ] Implementar más repositorios (Content, Project, Analytics)
- [ ] Agregar eventos de dominio (Domain Events)
- [ ] Implementar CQRS para queries complejas
- [ ] Agregar validación de negocio en la capa de dominio
- [ ] Implementar Value Objects para entidades complejas







