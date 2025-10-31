# Arquitectura Modular - Content Redundancy Detector

Esta aplicación sigue principios de **Clean Architecture** y **Hexagonal Architecture** (Ports and Adapters) para una estructura modular y mantenible.

## Estructura de Directorios

```
content_redundancy_detector/
├── domain/                   # Domain Layer (Core Business Logic)
│   ├── entities.py          # Business entities with behavior
│   ├── value_objects.py    # Immutable value objects
│   ├── interfaces.py        # Domain interfaces (Ports)
│   └── services.py          # Domain services
│
├── application/             # Application Layer (Use Cases)
│   ├── services.py         # Application services (orchestration)
│   ├── dtos.py             # Data Transfer Objects
│   └── dependencies.py     # Dependency injection
│
├── infrastructure/          # Infrastructure Layer (Adapters)
│   ├── adapters.py         # Concrete implementations
│   ├── cache.py            # Cache implementations
│   └── service_registry.py  # Service discovery
│
├── api/                     # Presentation Layer
│   ├── routes/             # Route modules by domain
│   │   ├── analysis.py    # Analysis endpoints
│   │   ├── batch.py       # Batch processing endpoints
│   │   ├── ai_ml.py       # AI/ML endpoints
│   │   └── __init__.py    # Route aggregation
│   ├── middleware.py       # API middleware
│   └── exception_handlers.py
│
├── core/                    # Core utilities
│   ├── config.py          # Configuration
│   └── logging_config.py  # Logging setup
│
└── shared/                  # Shared utilities
    └── utils.py            # Common utilities
```

## Capas de la Arquitectura

### 1. Domain Layer (Dominio)

**Responsabilidad**: Contiene la lógica de negocio pura, independiente de frameworks.

#### Entidades (`entities.py`)
```python
@dataclass
class ContentAnalysis:
    """Business entity with behavior"""
    content: str
    redundancy_score: float
    ...
    
    def is_redundant(self, threshold: float = 0.8) -> bool:
        """Business logic method"""
        return self.redundancy_score >= threshold
```

#### Value Objects (`value_objects.py`)
```python
@dataclass(frozen=True)
class AnalysisResult:
    """Immutable value object"""
    content_hash: str
    redundancy_score: float
    ...
```

#### Interfaces (Ports) (`interfaces.py`)
```python
class IAnalysisRepository(ABC):
    """Port - defines contract for persistence"""
    @abstractmethod
    async def save_analysis(self, analysis: ContentAnalysis) -> None:
        pass
```

### 2. Application Layer (Aplicación)

**Responsabilidad**: Orquesta la lógica de dominio y coordina entre capas.

#### Application Services (`services.py`)
```python
class AnalysisService:
    """Orchestrates domain logic"""
    def __init__(self, repository, cache_service, ml_service):
        self.repository = repository
        self.cache_service = cache_service
        self.ml_service = ml_service
    
    async def analyze_content(self, request: AnalysisRequest) -> AnalysisResult:
        # 1. Check cache
        # 2. Execute domain service
        # 3. Save to repository
        # 4. Cache result
```

#### DTOs (`dtos.py`)
```python
class AnalysisRequest(BaseModel):
    """Data Transfer Object for API"""
    content: str
    threshold: float
```

### 3. Infrastructure Layer (Infraestructura)

**Responsabilidad**: Implementaciones técnicas concretas.

#### Adapters (`adapters.py`)
```python
class RedisCacheAdapter(ICacheService):
    """Adapter - implements cache interface"""
    async def get(self, key: str) -> Optional[Any]:
        return self.redis_client.get(key)
```

### 4. API Layer (Presentación)

**Responsabilidad**: Endpoints HTTP y manejo de requests.

#### Routes (`api/routes/`)
```python
@router.post("/analyze")
async def analyze_content(
    request: AnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service)
):
    """API endpoint"""
    result = await service.analyze_content(request)
    return {"success": True, "data": result.to_dict()}
```

## Dependency Injection

### FastAPI Dependencies

```python
# application/dependencies.py
def get_analysis_service(
    repository: Annotated[IAnalysisRepository, Depends(get_analysis_repository)],
    cache_service: Annotated[ICacheService, Depends(get_cache_service)]
) -> AnalysisService:
    """Inject dependencies"""
    return AnalysisService(repository, cache_service)
```

### Service Registry

```python
# infrastructure/service_registry.py
class ServiceRegistry:
    """Service locator pattern (optional)"""
    _services = {}
    
    @classmethod
    def register(cls, name: str, service: Any):
        cls._services[name] = service
    
    @classmethod
    def get(cls, name: str) -> Any:
        return cls._services.get(name)
```

## Principios Aplicados

### 1. Separation of Concerns
- Cada capa tiene una responsabilidad clara
- Domain no depende de frameworks
- API no contiene lógica de negocio

### 2. Dependency Inversion
- Dependencias apuntan hacia adentro (Domain)
- Interfaces definen contratos
- Implementaciones son intercambiables

### 3. Single Responsibility
- Cada clase tiene una razón para cambiar
- Servicios pequeños y enfocados

### 4. Open/Closed Principle
- Abierto para extensión (nuevos adapters)
- Cerrado para modificación (domain invariante)

## Organización por Dominio

### Routers por Dominio

```
api/routes/
├── analysis.py      # Content analysis endpoints
├── batch.py         # Batch processing endpoints
├── ai_ml.py         # AI/ML features endpoints
├── export.py        # Export functionality
├── analytics.py     # Analytics and reporting
└── webhooks.py      # Webhook management
```

### Agregación de Routers

```python
# api/routes/__init__.py
from fastapi import APIRouter
from .analysis import router as analysis_router
from .batch import router as batch_router

api_router = APIRouter()
api_router.include_router(analysis_router)
api_router.include_router(batch_router)
```

## Ventajas de la Arquitectura Modular

### 1. Testabilidad
```python
# Test con mocks fáciles
mock_repository = Mock(spec=IAnalysisRepository)
service = AnalysisService(mock_repository, mock_cache)
```

### 2. Mantenibilidad
- Cambios localizados por capa
- Fácil de entender y navegar
- Código organizado por responsabilidad

### 3. Escalabilidad
- Servicios independientes
- Fácil de dividir en microservicios
- Reutilización de componentes

### 4. Flexibilidad
- Cambiar implementaciones sin afectar dominio
- Agregar nuevas adapters fácilmente
- Swappable dependencies

## Ejemplo de Flujo Completo

### 1. Request llega
```
HTTP POST /api/v1/analyze
{
  "content": "text...",
  "threshold": 0.8
}
```

### 2. API Layer
```python
@router.post("/analyze")
async def analyze_content(request: AnalysisRequest, service = Depends(...)):
    # Valida request (Pydantic)
    # Llama a application service
```

### 3. Application Layer
```python
async def analyze_content(self, request: AnalysisRequest):
    # Check cache
    # Execute domain service
    # Save to repository
    # Cache result
```

### 4. Domain Layer
```python
async def analyze_content_domain(content: str):
    # Pure business logic
    # No dependencies externas
    return ContentAnalysis(...)
```

### 5. Infrastructure Layer
```python
class RedisCacheAdapter:
    # Technical implementation
    # Redis connection
    # Serialization
```

## Migración desde Estructura Antigua

### Antes (Monolítico)
```python
# routers.py - todo mezclado
@router.post("/analyze")
async def analyze(...):
    # Lógica de negocio
    # Cache directo
    # DB directo
```

### Después (Modular)
```python
# api/routes/analysis.py
@router.post("/analyze")
async def analyze(request, service = Depends(...)):
    return await service.analyze_content(request)

# application/services.py
async def analyze_content(...):
    # Orchestration
    
# domain/services.py
async def analyze_content_domain(...):
    # Business logic
```

## Testing Modular

### Unit Tests - Domain
```python
def test_content_analysis_is_redundant():
    analysis = ContentAnalysis(redundancy_score=0.9)
    assert analysis.is_redundant(threshold=0.8) == True
```

### Integration Tests - Application
```python
async def test_analysis_service():
    mock_repo = Mock(IAnalysisRepository)
    mock_cache = Mock(ICacheService)
    service = AnalysisService(mock_repo, mock_cache)
    # Test orchestration
```

### E2E Tests - API
```python
async def test_analyze_endpoint(client):
    response = await client.post("/api/v1/analyze", json={...})
    assert response.status_code == 200
```

## Próximos Pasos

1. **Migrar más routers** a estructura modular
2. **Crear más adapters** (DatabaseRepository, KafkaMessaging)
3. **Agregar eventos de dominio** para event-driven architecture
4. **Implementar CQRS** si es necesario
5. **Service mesh** para inter-service communication

## Referencias

- Clean Architecture - Robert C. Martin
- Hexagonal Architecture - Alistair Cockburn
- Domain-Driven Design - Eric Evans
- FastAPI Dependency Injection
- Microservices Patterns
