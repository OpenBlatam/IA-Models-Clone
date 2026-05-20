# Guía de Migración - Nueva Arquitectura

## Resumen

Este documento describe cómo migrar del código actual a la nueva arquitectura mejorada. La nueva arquitectura introduce:

- **Service Layer**: Separación de lógica de negocio
- **Strategy Pattern**: Estrategias de ejecución extensibles
- **Repository Pattern**: Abstracción de acceso a datos
- **Dependency Injection**: Mejor testabilidad
- **Exception Handling Centralizado**: Manejo consistente de errores

## Cambios Principales

### 1. Estructura de Directorios

**Antes:**
```
api/
├── router.py (972 líneas - monolítico)
├── helpers.py
└── schemas.py
```

**Después:**
```
api/
├── routers/
│   ├── execution.py
│   ├── models.py
│   └── ...
├── dependencies.py
├── exceptions.py
├── exception_handlers.py
└── helpers.py

core/
├── services/
│   ├── execution_service.py
│   ├── cache_service.py
│   └── consensus_service.py
├── strategies/
│   ├── base.py
│   ├── parallel.py
│   ├── sequential.py
│   └── factory.py
└── repositories/
    ├── model_repository.py
    └── registry_repository.py
```

### 2. Uso de Services

**Antes:**
```python
# En router.py
@router.post("/execute")
async def execute_multi_model(request: MultiModelRequest):
    # Lógica de negocio mezclada con routing
    enabled_models = [m for m in request.models if m.is_enabled]
    # ... 100+ líneas de lógica ...
```

**Después:**
```python
# En routers/execution.py
@router.post("/execute")
async def execute_multi_model(
    request: MultiModelRequest,
    execution_service: ExecutionService = Depends(get_execution_service)
):
    return await execution_service.execute(request)
```

### 3. Estrategias de Ejecución

**Antes:**
```python
# Funciones en router.py
async def _execute_parallel(...):
    # ...

async def _execute_sequential(...):
    # ...
```

**Después:**
```python
# Clases en core/strategies/
strategy = StrategyFactory.create("parallel")
responses = await strategy.execute(models, prompt, execute_func)
```

### 4. Manejo de Excepciones

**Antes:**
```python
try:
    # ...
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```

**Después:**
```python
# Excepciones custom
from ...api.exceptions import ModelExecutionException

raise ModelExecutionException(
    message="Failed to execute",
    model_type="gpt-4"
)
```

## Pasos de Migración

### Paso 1: Registrar Exception Handlers

```python
# En main.py o donde se crea la app FastAPI
from multi_model_api.api.exception_handlers import register_exception_handlers

app = FastAPI()
register_exception_handlers(app)
```

### Paso 2: Actualizar Imports

**Antes:**
```python
from multi_model_api import router
app.include_router(router)
```

**Después:**
```python
from multi_model_api.api.routers import execution_router, models_router

app.include_router(execution_router)
app.include_router(models_router)
```

### Paso 3: Migrar Endpoints Gradualmente

Puedes migrar endpoints uno por uno:

1. **Empezar con `/execute`**: Ya migrado en `routers/execution.py`
2. **Migrar `/models`**: Ya migrado en `routers/models.py`
3. **Migrar otros endpoints**: Similar proceso

### Paso 4: Actualizar Tests

**Antes:**
```python
def test_execute():
    # Test directo del router
    response = await execute_multi_model(request)
```

**Después:**
```python
def test_execute():
    # Test del service (más fácil de mockear)
    service = ExecutionService(...)
    response = await service.execute(request)
```

## Compatibilidad

### Backward Compatibility

La nueva arquitectura es **compatible hacia atrás**:

- ✅ Mismos endpoints
- ✅ Mismos schemas
- ✅ Misma funcionalidad
- ✅ Mismo comportamiento

### Cambios No Compatibles

Ninguno. Todos los cambios son internos.

## Ejemplos de Uso

### Ejemplo 1: Usar ExecutionService Directamente

```python
from multi_model_api.core.services import ExecutionService
from multi_model_api.core.repositories import RegistryModelRepository
from multi_model_api.core.services import CacheService, ConsensusService
from multi_model_api.core.strategies import StrategyFactory
from multi_model_api.core.models import get_registry
from multi_model_api.core.cache import get_cache

# Crear servicios
repository = RegistryModelRepository(get_registry())
cache_service = CacheService(get_cache())
consensus_service = ConsensusService()
factory = StrategyFactory()

service = ExecutionService(
    model_repository=repository,
    cache_service=cache_service,
    consensus_service=consensus_service,
    strategy_factory=factory
)

# Usar
response = await service.execute(request)
```

### Ejemplo 2: Agregar Nueva Estrategia

```python
from multi_model_api.core.strategies import ExecutionStrategy, StrategyFactory

class CustomStrategy(ExecutionStrategy):
    async def execute(self, models, prompt, execute_func, **kwargs):
        # Implementación personalizada
        pass

# Registrar
StrategyFactory.register_strategy("custom", CustomStrategy)

# Usar
strategy = StrategyFactory.create("custom")
```

### Ejemplo 3: Custom Repository

```python
from multi_model_api.core.repositories import ModelRepository

class DatabaseModelRepository(ModelRepository):
    async def execute_model(self, model_type, prompt, **kwargs):
        # Implementación con base de datos
        pass
```

## Testing

### Test de Services

```python
import pytest
from unittest.mock import Mock, AsyncMock
from multi_model_api.core.services import ExecutionService

@pytest.fixture
def mock_repository():
    repo = Mock()
    repo.execute_model = AsyncMock(return_value=ModelResponse(...))
    return repo

async def test_execution_service(mock_repository):
    service = ExecutionService(
        model_repository=mock_repository,
        cache_service=Mock(),
        consensus_service=Mock(),
        strategy_factory=Mock()
    )
    
    response = await service.execute(request)
    assert response.success_count > 0
```

### Test de Strategies

```python
async def test_parallel_strategy():
    strategy = ParallelStrategy()
    responses = await strategy.execute(
        models=[...],
        prompt="test",
        execute_func=mock_execute
    )
    assert len(responses) == len(models)
```

## Beneficios de la Migración

1. **Mantenibilidad**: Código más organizado y fácil de mantener
2. **Testabilidad**: Componentes aislados son más fáciles de testear
3. **Extensibilidad**: Fácil agregar nuevas estrategias y servicios
4. **Separación de Responsabilidades**: Cada componente tiene una responsabilidad clara
5. **Reutilización**: Services pueden ser reutilizados en diferentes contextos

## Troubleshooting

### Problema: Import Errors

**Solución**: Asegúrate de que todos los `__init__.py` exporten correctamente.

### Problema: Dependency Injection Fails

**Solución**: Verifica que todas las dependencias estén registradas en `dependencies.py`.

### Problema: Exception Handlers No Funcionan

**Solución**: Asegúrate de llamar `register_exception_handlers(app)` después de crear la app.

## Próximos Pasos

1. ✅ Migrar todos los endpoints restantes
2. ✅ Escribir tests para nuevos componentes
3. ✅ Actualizar documentación
4. ✅ Optimizar performance si es necesario
5. ✅ Agregar más estrategias si se necesitan

## Recursos

- [ARCHITECTURE_IMPROVEMENTS.md](ARCHITECTURE_IMPROVEMENTS.md) - Documentación completa de mejoras
- [README.md](README.md) - Documentación general
- [REFACTORING.md](REFACTORING.md) - Refactorizaciones previas




