# Guía de Repositorios - Robot Movement AI

## 📋 Visión General

Sistema de repositorios implementado siguiendo el patrón Repository de la nueva arquitectura. Soporta múltiples backends y estrategias de cache.

## 🏗️ Arquitectura de Repositorios

```
┌─────────────────────────────────────┐
│   Application Layer (Use Cases)     │
│   - Usa interfaces IRobotRepository │
│   - Usa interfaces IMovementRepo    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Repository Factory                 │
│   - Selecciona implementación       │
│   - Configura cache                 │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┐
       │               │
       ▼               ▼
┌─────────────┐  ┌─────────────┐
│ In-Memory   │  │ SQL         │
│ (Testing)   │  │ (Production)│
└─────────────┘  └──────┬──────┘
                        │
                        ▼
                 ┌─────────────┐
                 │ With Cache   │
                 │ (Decorator)  │
                 └─────────────┘
```

## 📦 Tipos de Repositorios

### 1. In-Memory Repositories

**Uso**: Desarrollo, testing, demos

**Características**:
- ✅ Rápido y simple
- ✅ No requiere base de datos
- ✅ Perfecto para testing
- ❌ Datos se pierden al reiniciar

**Ejemplo**:
```python
from core.architecture.infrastructure_repositories import InMemoryRobotRepository

repo = InMemoryRobotRepository()
await repo.initialize()

robot = Robot(robot_id="robot-1", brand="KUKA", model="KR210")
await repo.save(robot)

found_robot = await repo.find_by_id("robot-1")
```

### 2. SQL Repositories

**Uso**: Producción

**Características**:
- ✅ Persistencia real
- ✅ Soporta SQLite, PostgreSQL, MySQL
- ✅ Transacciones
- ✅ Escalable

**Ejemplo**:
```python
import sqlite3
from core.architecture.infrastructure_repositories import SQLRobotRepository

# SQLite
db = sqlite3.connect("robots.db")
repo = SQLRobotRepository(db)
await repo.initialize()

# PostgreSQL (asyncpg)
import asyncpg
db = await asyncpg.connect("postgresql://user:pass@localhost/db")
repo = SQLRobotRepository(db)
await repo.initialize()
```

### 3. Cached Repositories

**Uso**: Producción con alta carga

**Características**:
- ✅ Cache en memoria
- ✅ Reduce carga en BD
- ✅ TTL configurable
- ✅ Invalidación automática

**Ejemplo**:
```python
from core.architecture.repository_factory import CachedRobotRepository
from core.architecture.infrastructure_repositories import SQLRobotRepository

base_repo = SQLRobotRepository(db)
cached_repo = CachedRobotRepository(
    base_repo,
    cache_config={
        'ttl': 300,  # 5 minutos
        'max_size': 1000
    }
)
```

## 🏭 Repository Factory

El factory permite crear repositorios según configuración sin acoplar el código.

### Uso Básico

```python
from core.architecture.repository_factory import (
    RepositoryFactory,
    RepositoryType
)

# In-Memory (desarrollo)
factory = RepositoryFactory({
    'repository_type': RepositoryType.IN_MEMORY.value
})
robot_repo = factory.create_robot_repository()
movement_repo = factory.create_movement_repository()

# SQL (producción)
factory = RepositoryFactory({
    'repository_type': RepositoryType.SQL.value,
    'db_connection': db_connection
})
robot_repo = factory.create_robot_repository()

# SQL con Cache (producción optimizada)
factory = RepositoryFactory({
    'repository_type': RepositoryType.SQL_WITH_CACHE.value,
    'db_connection': db_connection,
    'cache_config': {
        'ttl': 300,
        'max_size': 1000
    }
})
robot_repo = factory.create_robot_repository()
```

### Helper Function

```python
from core.architecture.repository_factory import create_repository_factory

# In-Memory
factory = create_repository_factory("in_memory")

# SQL
factory = create_repository_factory(
    "sql",
    db_connection=db_conn
)

# SQL con Cache
factory = create_repository_factory(
    "sql_with_cache",
    db_connection=db_conn,
    cache_config={'ttl': 300}
)
```

## 🔌 Integración con Use Cases

Los repositorios se integran perfectamente con los use cases:

```python
from core.architecture.application_layer import MoveRobotUseCase
from core.architecture.repository_factory import create_repository_factory

# Crear factory
factory = create_repository_factory("in_memory")

# Crear repositorios
robot_repo = factory.create_robot_repository()
movement_repo = factory.create_movement_repository()

# Crear use case con repositorios
use_case = MoveRobotUseCase(
    robot_repository=robot_repo,
    movement_repository=movement_repo
)

# Usar use case
command = MoveRobotCommand(
    robot_id="robot-1",
    target_x=0.5,
    target_y=0.3,
    target_z=0.2
)
result = await use_case.execute(command)
```

## 🔧 Configuración

### Variables de Entorno

```env
# Tipo de repositorio
REPOSITORY_TYPE=in_memory  # in_memory, sql, sql_with_cache

# Base de datos (para SQL)
DATABASE_URL=sqlite:///robots.db
# o
DATABASE_URL=postgresql://user:pass@localhost/robots

# Cache (para sql_with_cache)
CACHE_TTL=300
CACHE_MAX_SIZE=1000
```

### Código de Configuración

```python
import os
from core.architecture.repository_factory import create_repository_factory

# Leer configuración
repo_type = os.getenv("REPOSITORY_TYPE", "in_memory")
db_url = os.getenv("DATABASE_URL")

# Crear conexión si es SQL
db_connection = None
if repo_type in ["sql", "sql_with_cache"]:
    if db_url.startswith("sqlite"):
        import sqlite3
        db_connection = sqlite3.connect(db_url.replace("sqlite:///", ""))
    elif db_url.startswith("postgresql"):
        import asyncpg
        db_connection = await asyncpg.connect(db_url)

# Crear factory
factory = create_repository_factory(
    repository_type=repo_type,
    db_connection=db_connection,
    cache_config={
        'ttl': int(os.getenv("CACHE_TTL", "300")),
        'max_size': int(os.getenv("CACHE_MAX_SIZE", "1000"))
    }
)
```

## 📊 Métodos Disponibles

### IRobotRepository

```python
async def find_by_id(robot_id: str) -> Optional[Robot]
async def save(robot: Robot) -> None
async def find_all() -> List[Robot]
```

### IMovementRepository

```python
async def find_by_id(movement_id: str) -> Optional[RobotMovement]
async def save(movement: RobotMovement) -> None
async def find_by_robot_id(robot_id: str, limit: int = 100) -> List[RobotMovement]
```

## 🧪 Testing

Los repositorios en memoria son perfectos para testing:

```python
import pytest
from core.architecture.infrastructure_repositories import (
    InMemoryRobotRepository,
    InMemoryMovementRepository
)

@pytest.fixture
async def robot_repo():
    repo = InMemoryRobotRepository()
    await repo.initialize()
    yield repo
    repo.clear()

@pytest.fixture
async def movement_repo():
    repo = InMemoryMovementRepository()
    await repo.initialize()
    yield repo
    repo.clear()

@pytest.mark.asyncio
async def test_move_robot(robot_repo, movement_repo):
    # Crear robot
    robot = Robot(robot_id="test-1", brand="KUKA", model="KR210")
    await robot_repo.save(robot)
    
    # Verificar
    found = await robot_repo.find_by_id("test-1")
    assert found is not None
    assert found.brand == "KUKA"
```

## 🚀 Próximos Pasos

1. **Implementar más métodos de búsqueda**:
   - `find_by_brand()`
   - `find_by_status()`
   - `find_recent_movements()`

2. **Agregar más backends**:
   - MongoDB
   - Redis
   - InfluxDB (para métricas)

3. **Optimizaciones**:
   - Batch operations
   - Connection pooling
   - Query optimization

4. **Migraciones**:
   - Sistema de migraciones de BD
   - Versionado de esquemas

## 📝 Notas

- Los repositorios implementan las interfaces definidas en `application_layer.py`
- El factory permite cambiar de backend sin modificar código de aplicación
- El cache usa el patrón Decorator para agregar funcionalidad sin modificar repositorios base
- Todos los repositorios son async para mejor rendimiento

---

**Fecha**: 2025-01-27
**Versión**: 1.0.0




