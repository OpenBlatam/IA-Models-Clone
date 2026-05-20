# Advanced Patterns - V7.2.0

## Resumen

Implementación de patrones avanzados para microservicios: CQRS, Sagas, Feature Flags, API Versioning, y Advanced Caching.

## Patrones Implementados

### 1. CQRS (Command Query Responsibility Segregation)

**Separación de operaciones de lectura y escritura para mejor escalabilidad.**

#### Componentes:
- **Commands** (`core/cqrs/commands.py`): Operaciones de escritura
- **Queries** (`core/cqrs/queries.py`): Operaciones de lectura
- **Handlers** (`core/cqrs/handlers.py`): Procesadores de commands/queries
- **CommandBus/QueryBus**: Buses para dispatch

#### Uso:
```python
from core.cqrs.commands import AnalyzeImageCommand
from core.cqrs.queries import GetAnalysisQuery
from core.cqrs.handlers import get_command_bus, get_query_bus

# Command (Write)
command = AnalyzeImageCommand(
    user_id="user123",
    image_data=b"image_bytes"
)
result = await get_command_bus().dispatch(command)

# Query (Read)
query = GetAnalysisQuery(analysis_id="analysis123")
analysis = await get_query_bus().dispatch(query)
```

#### Ventajas:
- Escalabilidad independiente de reads/writes
- Optimización separada
- Modelos de lectura optimizados
- Separación clara de responsabilidades

### 2. Sagas Pattern

**Manejo de transacciones distribuidas con compensación.**

#### Componentes:
- **Saga** (`core/sagas/saga.py`): Transacción distribuida
- **SagaStep**: Paso individual con compensación
- **SagaOrchestrator**: Orquestador de sagas

#### Uso:
```python
from core.sagas.saga import Saga
from core.sagas.orchestrator import get_saga_orchestrator

saga = Saga()

# Add steps with compensation
saga.add_step(
    name="create_analysis",
    execute=lambda ctx: create_analysis(ctx),
    compensate=lambda ctx: delete_analysis(ctx)
)

saga.add_step(
    name="send_notification",
    execute=lambda ctx: send_notification(ctx),
    compensate=lambda ctx: cancel_notification(ctx)
)

# Execute saga
orchestrator = get_saga_orchestrator()
success = await orchestrator.execute_saga(saga)
```

#### Ventajas:
- Transacciones distribuidas
- Compensación automática
- Retry con exponential backoff
- Timeout support

### 3. Feature Flags

**Habilitar/deshabilitar features sin deployment.**

#### Componentes:
- **FeatureFlag**: Definición de flag
- **FeatureFlagManager**: Gestor de flags
- **Decorator**: `@feature_flag` para funciones

#### Uso:
```python
from core.feature_flags import get_feature_flag_manager, feature_flag

# Register flag
manager = get_feature_flag_manager()
manager.register_flag(FeatureFlag(
    name="new_analysis_algorithm",
    enabled=True,
    flag_type=FeatureFlagType.PERCENTAGE,
    percentage=50  # 50% rollout
))

# Check flag
if manager.is_enabled("new_analysis_algorithm", user_id="user123"):
    # Use new algorithm
    pass

# Decorator
@feature_flag("new_feature", user_id="user123")
async def new_feature_function():
    # This only executes if flag is enabled
    pass
```

#### Tipos de Flags:
- **BOOLEAN**: Simple on/off
- **PERCENTAGE**: Rollout gradual
- **USER_LIST**: Usuarios específicos
- **CUSTOM**: Lógica personalizada

### 4. API Versioning

**Soporte para múltiples versiones de API.**

#### Componentes:
- **VersionRouter**: Gestor de versiones
- **APIVersion**: Enum de versiones
- **extract_api_version**: Extrae versión de request

#### Uso:
```python
from api.versioning import get_version_router, APIVersion

version_router = get_version_router()

# Register v1
v1_router = APIRouter()
version_router.register_version("v1", v1_router, deprecated=True)

# Register v2
v2_router = APIRouter()
version_router.register_version("v2", v2_router)

# Extract version from request
version = extract_api_version(request)
router = version_router.get_router(version)
```

#### Detección de Versión:
1. URL path: `/api/v1/...`
2. Header: `X-API-Version: v1`
3. Query param: `?version=v1`
4. Accept header: `application/vnd.api+json;version=1`

### 5. Advanced Caching Strategies

**Múltiples estrategias de caching.**

#### Estrategias:
- **CACHE_ASIDE**: Application maneja cache
- **WRITE_THROUGH**: Escribe a cache y DB simultáneamente
- **WRITE_BACK**: Escribe a cache, flush a DB después
- **REFRESH_AHEAD**: Prefetch antes de expiración

#### Uso:
```python
from utils.advanced_caching import AdvancedCache, CacheStrategy, CacheDecorator

# Create cache with strategy
cache = AdvancedCache(cache_service, CacheStrategy.WRITE_THROUGH)

# Get with fetch function
value = await cache.get(
    "key",
    fetch_fn=lambda: fetch_from_database(),
    ttl=3600
)

# Set with write function
await cache.set(
    "key",
    value,
    write_fn=lambda k, v: write_to_database(k, v)
)

# Decorator
@CacheDecorator(cache, ttl=3600)
async def expensive_function(param1, param2):
    # Result cached automatically
    return compute_result(param1, param2)
```

## Ejemplos de Integración

### CQRS con Use Cases

```python
# Command Handler
class AnalyzeImageCommandHandler(CommandHandler):
    def __init__(self, analyze_use_case: AnalyzeImageUseCase):
        self.analyze_use_case = analyze_use_case
    
    async def handle(self, command: AnalyzeImageCommand) -> Analysis:
        return await self.analyze_use_case.execute(
            command.user_id,
            command.image_data,
            command.metadata
        )

# Register handler
command_bus = get_command_bus()
command_bus.register_handler(
    AnalyzeImageCommand,
    AnalyzeImageCommandHandler(analyze_use_case)
)
```

### Saga para Transacción Compleja

```python
async def create_analysis_with_notification_saga():
    saga = Saga()
    
    # Step 1: Create analysis
    saga.add_step(
        name="create_analysis",
        execute=lambda ctx: create_analysis(ctx["user_id"], ctx["image_data"]),
        compensate=lambda ctx: delete_analysis(ctx.get("analysis_id"))
    )
    
    # Step 2: Send notification
    saga.add_step(
        name="send_notification",
        execute=lambda ctx: send_notification(ctx["analysis_id"]),
        compensate=lambda ctx: cancel_notification(ctx.get("notification_id"))
    )
    
    # Execute
    orchestrator = get_saga_orchestrator()
    return await orchestrator.execute_saga(saga)
```

### Feature Flag con A/B Testing

```python
# Register A/B test flag
manager = get_feature_flag_manager()
manager.register_flag(FeatureFlag(
    name="new_recommendation_algorithm",
    enabled=True,
    flag_type=FeatureFlagType.PERCENTAGE,
    percentage=25  # 25% of users
))

# Use in code
if manager.is_enabled("new_recommendation_algorithm", user_id=user_id):
    recommendations = await new_algorithm.generate(user_id)
else:
    recommendations = await old_algorithm.generate(user_id)
```

## Ventajas de los Patrones

### CQRS
- ✅ Escalabilidad independiente
- ✅ Optimización separada
- ✅ Modelos de lectura optimizados
- ✅ Separación clara

### Sagas
- ✅ Transacciones distribuidas
- ✅ Compensación automática
- ✅ Resiliencia
- ✅ Retry automático

### Feature Flags
- ✅ Deploy sin riesgo
- ✅ Rollout gradual
- ✅ A/B testing
- ✅ Kill switch

### API Versioning
- ✅ Backward compatibility
- ✅ Migración gradual
- ✅ Múltiples versiones
- ✅ Deprecation management

### Advanced Caching
- ✅ Múltiples estrategias
- ✅ Optimización por caso
- ✅ Write-through/back
- ✅ Auto-refresh

## Mejores Prácticas

### CQRS
1. Usar commands para writes
2. Usar queries para reads
3. Separar modelos de lectura
4. Optimizar queries independientemente

### Sagas
1. Cada step debe ser idempotente
2. Siempre proveer compensación
3. Usar timeouts apropiados
4. Log todos los pasos

### Feature Flags
1. Flags con nombres descriptivos
2. Documentar cada flag
3. Limpiar flags deprecated
4. Monitorear flag usage

### API Versioning
1. Versionar breaking changes
2. Mantener versiones deprecated
3. Comunicar sunset dates
4. Migrar gradualmente

### Caching
1. Elegir estrategia apropiada
2. Invalidar cuando sea necesario
3. Monitorear hit rates
4. Usar TTL apropiados

## Conclusión

Los patrones avanzados proporcionan:

- ✅ **CQRS**: Escalabilidad y optimización
- ✅ **Sagas**: Transacciones distribuidas
- ✅ **Feature Flags**: Deploy sin riesgo
- ✅ **API Versioning**: Backward compatibility
- ✅ **Advanced Caching**: Performance optimizado

El sistema está ahora equipado con patrones enterprise-grade para microservicios.















