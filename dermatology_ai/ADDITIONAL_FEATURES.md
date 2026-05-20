# Additional Features - V7.3.0

## Resumen

Características adicionales para completar el ecosistema enterprise: GraphQL, gRPC, Database Migrations, Testing Framework, y Performance Tools.

## Nuevas Características

### 1. GraphQL Support (Opcional)

**API GraphQL opcional junto a REST API.**

#### Componentes:
- **Schema** (`api/graphql/schema.py`): Definición de schema GraphQL
- **Resolvers**: Resolvers para queries
- **Strawberry GraphQL**: Framework moderno para GraphQL

#### Uso:
```python
from api.graphql.schema import create_graphql_router

# Add to FastAPI app
graphql_router = create_graphql_router()
if graphql_router:
    app.include_router(graphql_router, prefix="/graphql")
```

#### Ventajas:
- ✅ Query flexible
- ✅ Reducción de over-fetching
- ✅ Type-safe queries
- ✅ Single endpoint

### 2. gRPC Support (Opcional)

**Comunicación inter-servicios de alto rendimiento.**

#### Componentes:
- **gRPC Service** (`api/grpc/service.py`): Implementación de servicios gRPC
- **Protocol Buffers**: Definición de contratos
- **Async gRPC**: Soporte asíncrono

#### Uso:
```python
from api.grpc.service import create_grpc_server

# Create gRPC server
grpc_server = create_grpc_server(port=50051)

# Start server
await grpc_server.start()
await grpc_server.wait_for_termination()
```

#### Ventajas:
- ✅ Alto rendimiento (binario)
- ✅ Streaming support
- ✅ Type-safe contracts
- ✅ Inter-service communication

### 3. Database Migrations

**Sistema de migraciones versionado.**

#### Componentes:
- **MigrationManager** (`core/migrations/migration_manager.py`): Gestor de migraciones
- **Migration**: Definición de migración
- **Up/Down**: Migración y rollback

#### Uso:
```python
from core.migrations import Migration, MigrationManager, get_migration_manager

# Create migration
migration = Migration(
    version="001",
    name="create_analyses_table",
    up=lambda db: db.execute("CREATE TABLE analyses ..."),
    down=lambda db: db.execute("DROP TABLE analyses")
)

# Register and apply
manager = get_migration_manager(database_adapter)
manager.register_migration(migration)
await manager.migrate()
```

#### Ventajas:
- ✅ Versionado de schema
- ✅ Rollback support
- ✅ Tracking de migraciones
- ✅ Reproducible deployments

### 4. Testing Framework

**Framework completo de testing.**

#### Componentes:
- **conftest.py**: Fixtures compartidas
- **test_domain.py**: Tests de domain layer
- **test_use_cases.py**: Tests de use cases
- **Mocks**: Mocks para todas las interfaces

#### Uso:
```python
import pytest
from tests.conftest import mock_analysis_repository

@pytest.mark.asyncio
async def test_use_case(mock_analysis_repository):
    # Test implementation
    pass
```

#### Fixtures Disponibles:
- `mock_analysis_repository`
- `mock_user_repository`
- `mock_image_processor`
- `mock_cache_service`
- `mock_event_publisher`
- `service_factory`
- `plugin_registry`
- `sample_analysis_data`
- `sample_user_data`

#### Ventajas:
- ✅ Tests aislados
- ✅ Mocks listos
- ✅ Fixtures reutilizables
- ✅ Async support

### 5. Performance Profiling

**Herramientas para profiling y optimización.**

#### Componentes:
- **PerformanceMonitor**: Monitoreo de métricas
- **profile_context**: Context manager para profiling
- **profile_function**: Decorator para profiling

#### Uso:
```python
from utils.performance_profiler import (
    profile_context,
    profile_function,
    get_performance_monitor
)

# Context manager
with profile_context("profile.stats"):
    # Code to profile
    expensive_operation()

# Decorator
@profile_function
async def my_function():
    # Automatically profiled
    pass

# Monitor
monitor = get_performance_monitor()
monitor.record("operation", duration=0.5)
stats = monitor.get_stats("operation")
```

#### Ventajas:
- ✅ Identificar bottlenecks
- ✅ Métricas detalladas
- ✅ Integración con cProfile
- ✅ Estadísticas automáticas

### 6. Load Testing Script

**Script para testing de carga.**

#### Componentes:
- **load_test.py**: Script de load testing
- **Async requests**: Requests concurrentes
- **Statistics**: Estadísticas detalladas

#### Uso:
```bash
# Basic load test
python scripts/load_test.py --url http://localhost:8006/health --requests 1000 --concurrent 50

# Custom test
python scripts/load_test.py \
    --url http://localhost:8006/api/v1/analysis \
    --requests 500 \
    --concurrent 25 \
    --method POST
```

#### Métricas:
- Total requests
- Success rate
- Requests per second
- Response times (avg, min, max, median, P95, P99)

#### Ventajas:
- ✅ Testing de carga fácil
- ✅ Métricas detalladas
- ✅ Configurable
- ✅ Async/await

## Estructura de Testing

```
tests/
├── conftest.py           # Shared fixtures
├── test_domain.py        # Domain layer tests
├── test_use_cases.py    # Use case tests
├── test_repositories.py # Repository tests (to be added)
└── test_controllers.py  # Controller tests (to be added)
```

## Ejemplos de Uso

### GraphQL Query

```graphql
query {
  getAnalysis(analysisId: "123") {
    id
    status
    metrics {
      overallScore
      textureScore
    }
    conditions {
      name
      confidence
    }
  }
}
```

### gRPC Service

```python
# .proto file (example)
service DermatologyService {
  rpc AnalyzeImage(AnalyzeImageRequest) returns (AnalysisResponse);
  rpc GetAnalysis(GetAnalysisRequest) returns (AnalysisResponse);
}
```

### Migration Example

```python
# migrations/001_create_analyses.py
from core.migrations import Migration

migration = Migration(
    version="001",
    name="create_analyses_table",
    up=async lambda db: await db.execute("""
        CREATE TABLE analyses (
            id VARCHAR PRIMARY KEY,
            user_id VARCHAR NOT NULL,
            status VARCHAR NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """),
    down=async lambda db: await db.execute("DROP TABLE analyses")
)
```

### Performance Monitoring

```python
from utils.performance_profiler import get_performance_monitor

monitor = get_performance_monitor()

# Record operation
start = time.time()
await expensive_operation()
duration = time.time() - start
monitor.record("expensive_operation", duration)

# Get stats
stats = monitor.get_stats("expensive_operation")
print(f"Avg: {stats['avg_duration']:.4f}s")
print(f"Count: {stats['count']}")
```

## Ventajas de las Características

### GraphQL
- ✅ Query flexible
- ✅ Reducción de over-fetching
- ✅ Type-safe
- ✅ Single endpoint

### gRPC
- ✅ Alto rendimiento
- ✅ Streaming
- ✅ Type-safe
- ✅ Inter-service

### Migrations
- ✅ Versionado
- ✅ Rollback
- ✅ Reproducible
- ✅ Tracking

### Testing
- ✅ Tests aislados
- ✅ Mocks listos
- ✅ Fixtures
- ✅ Async support

### Profiling
- ✅ Identificar bottlenecks
- ✅ Métricas detalladas
- ✅ Optimización guiada
- ✅ Performance tracking

### Load Testing
- ✅ Testing de carga fácil
- ✅ Métricas detalladas
- ✅ Configurable
- ✅ CI/CD ready

## Mejores Prácticas

### GraphQL
1. Usar para queries complejas
2. Evitar over-fetching
3. Implementar rate limiting
4. Validar queries

### gRPC
1. Usar para inter-service communication
2. Definir contratos claros
3. Versionar servicios
4. Manejar errores apropiadamente

### Migrations
1. Versionar todas las migraciones
2. Siempre proveer rollback
3. Testear migraciones
4. Documentar cambios

### Testing
1. Testear domain layer primero
2. Usar mocks para infraestructura
3. Tests aislados
4. Cobertura alta

### Profiling
1. Profile en producción-like environment
2. Identificar hotspots
3. Optimizar iterativamente
4. Monitorear continuamente

### Load Testing
1. Empezar con carga baja
2. Incrementar gradualmente
3. Monitorear recursos
4. Documentar resultados

## Conclusión

Las características adicionales proporcionan:

- ✅ **GraphQL**: API flexible
- ✅ **gRPC**: Comunicación de alto rendimiento
- ✅ **Migrations**: Gestión de schema
- ✅ **Testing**: Framework completo
- ✅ **Profiling**: Optimización guiada
- ✅ **Load Testing**: Validación de performance

El sistema está ahora completamente equipado con todas las herramientas necesarias para desarrollo, testing, y producción enterprise.















