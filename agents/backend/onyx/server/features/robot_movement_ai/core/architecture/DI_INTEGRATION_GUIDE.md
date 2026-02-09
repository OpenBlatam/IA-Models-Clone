# Guía de Integración de Dependency Injection

## 📋 Visión General

El sistema de Dependency Injection ha sido completamente integrado con la arquitectura mejorada, permitiendo una gestión centralizada de dependencias y facilitando el testing y mantenimiento.

## 🏗️ Componentes de la Integración

### 1. DISetup (`di_setup.py`)

Clase principal que configura todos los servicios en el contenedor de DI.

**Características**:
- ✅ Configuración centralizada
- ✅ Setup automático de repositorios
- ✅ Setup automático de use cases
- ✅ Integración con event bus
- ✅ Soporte para configuración desde variables de entorno

### 2. Integración con Sistema de Inicialización

El DI se inicializa automáticamente durante el startup del sistema, después de la configuración y antes de otros servicios.

**Etapa**: `InitStage.DEPENDENCY_INJECTION`

## 🚀 Uso Básico

### Setup Automático (Recomendado)

El DI se configura automáticamente al iniciar la aplicación:

```python
from core.initialization import initialize_system

# Esto inicializa automáticamente el DI
result = await initialize_system()
```

### Setup Manual

Si necesitas configuración personalizada:

```python
from core.architecture.di_setup import setup_dependency_injection

# Configurar con opciones personalizadas
container = await setup_dependency_injection({
    'repository_type': 'sql',
    'db_connection': db_conn,
    'cache_config': {'ttl': 300},
    'enable_event_bus': True
})
```

## 📦 Servicios Disponibles

### Repositorios

```python
from core.architecture.di_setup import get_robot_repository, get_movement_repository

robot_repo = await get_robot_repository()
movement_repo = await get_movement_repository()
```

### Use Cases

```python
from core.architecture.di_setup import (
    get_move_robot_use_case,
    get_robot_status_use_case,
    get_movement_history_use_case
)

move_use_case = await get_move_robot_use_case()
status_use_case = await get_robot_status_use_case()
history_use_case = await get_movement_history_use_case()
```

### Resolver Cualquier Servicio

```python
from core.architecture.di_setup import get_di_setup
from core.architecture.application_layer import MoveRobotUseCase

di_setup = get_di_setup()
use_case = await di_setup.resolve(MoveRobotUseCase)
```

## 🔌 Integración con FastAPI

### Opción 1: Usando Helper Functions

```python
from fastapi import FastAPI, Depends
from core.architecture.di_setup import get_move_robot_use_case
from core.architecture.application_layer import MoveRobotCommand, MoveRobotUseCase

app = FastAPI()

@app.post("/api/v1/robots/{robot_id}/move")
async def move_robot(
    robot_id: str,
    target_x: float,
    target_y: float,
    target_z: float
):
    use_case = await get_move_robot_use_case()
    
    command = MoveRobotCommand(
        robot_id=robot_id,
        target_x=target_x,
        target_y=target_y,
        target_z=target_z
    )
    
    result = await use_case.execute(command)
    return result
```

### Opción 2: Usando Dependencies de FastAPI

```python
from fastapi import FastAPI, Depends
from core.architecture.di_setup import create_dependency
from core.architecture.application_layer import MoveRobotUseCase, MoveRobotCommand

app = FastAPI()

# Crear dependency
MoveRobotUseCaseDep = Depends(create_dependency(MoveRobotUseCase))

@app.post("/api/v1/robots/{robot_id}/move")
async def move_robot(
    robot_id: str,
    target_x: float,
    target_y: float,
    target_z: float,
    use_case: MoveRobotUseCase = MoveRobotUseCaseDep
):
    command = MoveRobotCommand(
        robot_id=robot_id,
        target_x=target_x,
        target_y=target_y,
        target_z=target_z
    )
    
    result = await use_case.execute(command)
    return result
```

## 🧪 Testing con DI

### Setup para Tests

```python
import pytest
from core.architecture.di_setup import DISetup

@pytest.fixture
async def di_setup():
    """Setup de DI para tests."""
    setup = DISetup({
        'repository_type': 'in_memory',
        'enable_event_bus': False
    })
    await setup.setup()
    yield setup

@pytest.mark.asyncio
async def test_move_robot(di_setup):
    use_case = await di_setup.resolve(MoveRobotUseCase)
    
    command = MoveRobotCommand(
        robot_id="test-1",
        target_x=0.5,
        target_y=0.3,
        target_z=0.2
    )
    
    result = await use_case.execute(command)
    assert result.movement_id is not None
```

## ⚙️ Configuración

### Variables de Entorno

```env
# Tipo de repositorio
REPOSITORY_TYPE=in_memory  # in_memory, sql, sql_with_cache

# Base de datos (para SQL)
DATABASE_URL=sqlite:///robots.db

# Cache
CACHE_TTL=300
CACHE_MAX_SIZE=1000

# Event Bus
ENABLE_EVENT_BUS=true
```

### Configuración Programática

```python
config = {
    'repository_type': 'sql',
    'db_connection': db_conn,
    'cache_config': {
        'ttl': 300,
        'max_size': 1000
    },
    'enable_event_bus': True
}

await setup_dependency_injection(config)
```

## 🔄 Ciclo de Vida de Servicios

### Singleton (Default)

Los servicios se crean una vez y se reutilizan:

```python
# Primera llamada - crea instancia
use_case1 = await get_move_robot_use_case()

# Segunda llamada - reutiliza misma instancia
use_case2 = await get_move_robot_use_case()

assert use_case1 is use_case2  # True
```

### Scoped (Por Request)

Para servicios que deben ser únicos por request:

```python
from core.architecture.di_setup import get_di_setup

di_setup = get_di_setup()
container = di_setup.get_container()

# Crear scope para request
scope_id = container.create_scope()
container.enter_scope(scope_id)

try:
    # Resolver servicios scoped
    use_case = await container.resolve_async(MoveRobotUseCase)
    # ... usar use case
finally:
    # Limpiar scope
    container.exit_scope()
```

### Transient

Para servicios que deben crearse nuevos cada vez:

```python
# Registrar como transient
container.register(
    SomeService,
    factory=lambda: SomeService(),
    lifecycle=Lifecycle.TRANSIENT
)
```

## 📊 Arquitectura de la Integración

```
┌─────────────────────────────────────┐
│   Application Startup                │
│   - initialize_system()              │
└──────────────┬───────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   InitStage.DEPENDENCY_INJECTION     │
│   - setup_dependency_injection()    │
└──────────────┬───────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   DISetup.setup()                    │
│   ├─ setup_repositories()            │
│   ├─ setup_use_cases()               │
│   └─ setup_auxiliary_services()      │
└──────────────┬───────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Container (DI)                      │
│   - Todos los servicios registrados │
└─────────────────────────────────────┘
```

## ✅ Ventajas de la Integración

1. **Configuración Centralizada**: Un solo lugar para configurar todos los servicios
2. **Testing Fácil**: Fácil mockear dependencias
3. **Bajo Acoplamiento**: Servicios no conocen implementaciones concretas
4. **Gestión de Ciclo de Vida**: Control sobre cuándo se crean/destruyen servicios
5. **Inicialización Automática**: Se configura automáticamente al iniciar la app

## 🔧 Ejemplos Avanzados

### Registrar Servicio Personalizado

```python
from core.architecture.di_setup import get_di_setup
from core.architecture.dependency_injection import Lifecycle

di_setup = get_di_setup()
container = di_setup.get_container()

# Registrar servicio personalizado
container.register(
    MyCustomService,
    factory=lambda: MyCustomService(),
    lifecycle=Lifecycle.SINGLETON
)

# Usar servicio
my_service = await container.resolve_async(MyCustomService)
```

### Auto-resolución de Dependencias

El sistema puede auto-resolver dependencias analizando constructores:

```python
class MyService:
    def __init__(self, robot_repo: IRobotRepository):
        self.robot_repo = robot_repo

# Registrar servicio
container.register(MyService, lifecycle=Lifecycle.SINGLETON)

# Auto-resuelve IRobotRepository
my_service = await container.resolve_async(MyService)
```

## 📝 Notas Importantes

1. **Siempre inicializar primero**: El DI debe configurarse antes de usar cualquier servicio
2. **Singleton por defecto**: Los servicios son singletons a menos que se especifique lo contrario
3. **Async**: Todos los métodos de resolución son async
4. **Error handling**: El sistema maneja errores apropiadamente si un servicio no está registrado

## 🚀 Próximos Pasos

1. Integrar con más servicios del sistema
2. Agregar más use cases
3. Implementar middleware de DI para FastAPI
4. Agregar métricas de DI (tiempo de resolución, etc.)

---

**Fecha**: 2025-01-27
**Versión**: 1.0.0




