# Organización del Código - Gamma App

## Estructura de Directorios

```
gamma_app/
├── api/                    # Capa de Presentación (API)
│   ├── app_factory.py      # Factory para crear la app FastAPI
│   ├── dependencies.py     # Dependencias de FastAPI
│   ├── endpoints.py        # Endpoints base (root, health)
│   ├── error_handlers.py   # Decorador para manejo de errores
│   ├── exceptions.py       # Handlers de excepciones
│   ├── lifespan.py         # Gestión del ciclo de vida
│   ├── main.py             # Punto de entrada principal
│   ├── middleware.py       # Configuración de middleware
│   ├── models.py           # Modelos Pydantic para API
│   ├── rate_limiting.py    # Rate limiting con slowapi
│   ├── routes.py           # Rutas de la API
│   ├── router_registry.py  # Registro de routers
│   └── service_manager.py  # Utilidades para servicios
│
├── application/           # Capa de Aplicación
│   ├── di/                 # Dependency Injection
│   │   └── container.py    # Contenedor DI
│   └── unit_of_work.py     # Patrón Unit of Work
│
├── domain/                 # Capa de Dominio
│   └── interfaces/         # Interfaces y contratos
│       ├── repositories.py # Interfaces de repositorios
│       └── services.py    # Interfaces de servicios
│
├── infrastructure/         # Capa de Infraestructura
│   ├── database/           # Gestión de base de datos
│   │   └── session.py      # Session manager centralizado
│   ├── middleware/         # Middleware avanzado (opcional)
│   └── repositories/       # Implementaciones de repositorios
│       ├── base.py         # BaseRepository
│       └── user_repository.py
│
├── models/                 # Modelos SQLAlchemy
│   └── database.py         # Modelos de base de datos
│
├── core/                   # Lógica de negocio específica
│   ├── content_generator.py
│   ├── collaboration_engine.py
│   └── design_engine.py
│
├── services/               # Servicios de aplicación
│   ├── analytics_service.py
│   ├── cache_service.py
│   ├── collaboration_service.py
│   ├── security_service.py
│   └── ...
│
├── engines/                # Motores de generación
│   ├── presentation_engine.py
│   ├── document_engine.py
│   └── ...
│
├── utils/                  # Utilidades
│   ├── config.py           # Configuración
│   ├── auth.py             # Autenticación
│   ├── logging_config.py   # Logging
│   └── retry_helpers.py    # Helpers de retry
│
├── middleware/             # Middleware avanzado (legacy)
│   ├── rate_limit_middleware.py
│   └── security_middleware.py
│
├── cli/                    # CLI
│   └── commands/           # Comandos CLI
│
└── tests/                  # Tests
    ├── unit/
    ├── integration/
    └── conftest.py
```

## Principios de Organización

### 1. Separación por Capas

- **Presentation Layer** (`api/`): Maneja HTTP, validación de entrada, serialización
- **Application Layer** (`application/`): Orquesta casos de uso, coordina servicios
- **Domain Layer** (`domain/`): Lógica de negocio pura, interfaces
- **Infrastructure Layer** (`infrastructure/`): Implementaciones técnicas (BD, cache, etc.)

### 2. Gestión de Base de Datos

**Usar**: `infrastructure.database.session`
- `DatabaseSessionManager`: Gestión centralizada de conexiones
- `get_db_session()`: Dependency para FastAPI

**Deprecado**: `models.database.DatabaseManager`
- Mantenido solo para compatibilidad hacia atrás
- `init_database()` y `get_db()` redirigen a la nueva implementación

### 3. Modelos

- **Pydantic Models** (`api/models.py`): Para validación de API
- **SQLAlchemy Models** (`models/database.py`): Para persistencia

### 4. Middleware

- **Básico** (`api/middleware.py`): CORS, TrustedHost
- **Rate Limiting** (`api/rate_limiting.py`): Usa slowapi
- **Avanzado** (`middleware/`): Implementaciones legacy (opcional)

### 5. Servicios

- Ubicados en `services/`
- Acceden a datos a través de repositorios (cuando estén implementados)
- Gestionados por DI Container

## Convenciones de Nombres

- **Archivos**: `snake_case.py`
- **Clases**: `PascalCase`
- **Funciones/Métodos**: `snake_case`
- **Constantes**: `UPPER_SNAKE_CASE`
- **Interfaces**: Prefijo `I` (ej: `IRepository`)

## Imports

### Rutas Relativas
```python
from ..utils.config import get_settings
from ...domain.interfaces.repositories import IRepository
```

### Rutas Absolutas (desde raíz del proyecto)
```python
from gamma_app.domain.interfaces.repositories import IRepository
from gamma_app.infrastructure.database.session import get_db_manager
```

## Migración de Código Legacy

### Database Manager
```python
# ❌ Viejo
from models.database import DatabaseManager, get_db
db_manager = DatabaseManager(database_url)

# ✅ Nuevo
from infrastructure.database.session import get_db_manager, get_db_session
db_manager = get_db_manager()
```

### Repositorios
```python
# ✅ Usar Unit of Work
from application.unit_of_work import unit_of_work
from infrastructure.repositories.user_repository import UserRepository

async with unit_of_work() as uow:
    user_repo = uow.get_repository(UserRepository)
    user = await user_repo.get_by_email("user@example.com")
    await uow.commit()
```

## Archivos a Limpiar

- [x] Líneas en blanco excesivas en `cli/commands/__init__.py`
- [x] Líneas en blanco en `middleware/rate_limit_middleware.py`
- [x] Líneas en blanco en `middleware/security_middleware.py`
- [x] Líneas en blanco en `models/database.py`
- [x] Líneas en blanco en `__init__.py`

## Próximos Pasos de Organización

1. **Mover archivos de raíz**:
   - `start_gamma_app.py` → `scripts/`
   - `test_gamma_app.py` → `tests/`

2. **Consolidar middleware**:
   - Evaluar si `middleware/` avanzado se usa
   - Si no, mover a `infrastructure/middleware/` o eliminar

3. **Reorganizar modelos**:
   - Considerar mover `models/database.py` → `infrastructure/models/`
   - O mantener en `models/` pero limpiar

4. **Estandarizar imports**:
   - Usar rutas relativas dentro del mismo módulo
   - Rutas absolutas para imports externos







