# Arquitectura Mejorada - Content Redundancy Detector

## ✅ Arquitectura Implementada: Clean Architecture + Hexagonal Architecture

El sistema ha sido completamente refactorizado siguiendo principios de **Clean Architecture** y **Hexagonal Architecture (Ports & Adapters)** para máxima modularidad, testabilidad y mantenibilidad.

## 🏗️ Estructura de Capas

```
content_redundancy_detector/
│
├── domain/                      # 🎯 DOMAIN LAYER (Core)
│   ├── entities.py             # Entidades de negocio con comportamiento
│   ├── value_objects.py       # Objetos de valor inmutables
│   ├── interfaces.py           # Ports (contratos/interfaces)
│   └── events.py              # Eventos de dominio
│
├── application/                # 📋 APPLICATION LAYER (Orquestación)
│   ├── services.py            # Servicios de aplicación (use cases)
│   ├── dtos.py                # Data Transfer Objects
│   └── dependencies.py        # Inyección de dependencias FastAPI
│
├── infrastructure/             # 🔧 INFRASTRUCTURE LAYER (Adapters)
│   ├── adapters.py            # Implementaciones concretas de interfaces
│   ├── cache.py               # Implementaciones de cache
│   └── service_registry.py    # Registry pattern
│
├── api/                        # 🌐 API LAYER (Presentación)
│   ├── main.py                # FastAPI app
│   ├── middleware.py          # Middleware HTTP
│   ├── exception_handlers.py # Manejo de errores
│   └── routes/                # Routers por dominio
│       ├── analysis.py       # Endpoints de análisis
│       ├── health.py         # Health checks
│       └── ...
│
└── core/                       # ⚙️ CORE (Utilidades)
    ├── config.py              # Configuración
    └── logging_config.py      # Logging estructurado
```

## 🎯 Principios Aplicados

### 1. **Dependency Inversion Principle**
- Domain define interfaces (ports)
- Infrastructure implementa adapters
- Dependencias apuntan hacia adentro (hacia Domain)

### 2. **Separation of Concerns**
- **Domain**: Lógica de negocio pura, sin dependencias externas
- **Application**: Orquesta use cases, coordina capas
- **Infrastructure**: Implementaciones técnicas (Redis, DB, ML)
- **API**: Solo presenta HTTP, sin lógica de negocio

### 3. **Single Responsibility**
- Cada clase tiene una única razón para cambiar
- Servicios pequeños y enfocados
- Adapters específicos por tecnología

## 📦 Componentes por Capa

### Domain Layer (Entidades + Value Objects + Interfaces)

#### Entities (`domain/entities.py`)
```python
@dataclass
class ContentAnalysis:
    """Entidad con comportamiento de negocio"""
    content: str
    redundancy_score: float
    ...
    
    def is_redundant(self, threshold: float = 0.8) -> bool:
        """Lógica de negocio"""
        return self.redundancy_score >= threshold
```

#### Value Objects (`domain/value_objects.py`)
```python
@dataclass(frozen=True)
class AnalysisResult:
    """Objeto de valor inmutable"""
    content_hash: str
    redundancy_score: float
    ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialización"""
```

#### Interfaces/Ports (`domain/interfaces.py`)
```python
class IAnalysisRepository(ABC):
    """Port: Define contract"""
    @abstractmethod
    async def save_analysis(self, analysis: ContentAnalysis) -> None:
        pass
```

### Application Layer (Servicios + DTOs)

#### Application Services (`application/services.py`)
```python
class AnalysisService:
    """Orquesta lógica de dominio y coordina adapters"""
    
    def __init__(
        self,
        repository: IAnalysisRepository,      # Port
        cache_service: ICacheService,          # Port
        ml_service: IMLService,                # Port
        event_bus: IEventBus                   # Port
    ):
        # Todas las dependencias son interfaces (ports)
    
    async def analyze_content(self, request: AnalysisRequest) -> AnalysisResult:
        # 1. Check cache (infrastructure adapter)
        # 2. Execute domain logic
        # 3. Save to repository (infrastructure adapter)
        # 4. Publish event (infrastructure adapter)
```

#### DTOs (`application/dtos.py`)
```python
class AnalysisRequest(BaseModel):
    """DTO para API requests"""
    content: str
    threshold: Optional[float] = 0.8
```

### Infrastructure Layer (Adapters)

#### Repository Adapter (`infrastructure/adapters.py`)
```python
class InMemoryAnalysisRepository(IAnalysisRepository):
    """Adapter: Implementa interface de dominio"""
    
    async def save_analysis(self, analysis: ContentAnalysis) -> None:
        # Implementación técnica específica
        self._storage[analysis.content_hash] = analysis
```

#### Cache Adapter
```python
class RedisCacheAdapter(ICacheService):
    """Adapter: Redis con fallback a memoria"""
    
    async def get(self, key: str) -> Optional[Any]:
        # Implementación Redis
        # Fallback a memoria si Redis falla
```

### API Layer (Routers)

```python
@router.post("/analyze")
async def analyze_content(
    request: AnalysisRequest,                   # DTO
    service: AnalysisService = Depends(...)     # Inyección
) -> Dict[str, Any]:
    """Endpoint HTTP - solo presenta, no contiene lógica"""
    result = await service.analyze_content(request)
    return {"success": True, "data": result.to_dict()}
```

## 🔄 Flujo de Datos

```
HTTP Request
    ↓
API Router (api/routes/)
    ↓
Application Service (application/services.py)
    ↓
Domain Logic (domain/entities.py)
    ↓
Infrastructure Adapters (infrastructure/adapters.py)
    ↓
External Services (Redis, DB, ML)
```

## 💡 Ventajas de esta Arquitectura

### 1. **Testabilidad**
```python
# Test fácil con mocks de interfaces
mock_repo = Mock(IAnalysisRepository)
mock_cache = Mock(ICacheService)
service = AnalysisService(mock_repo, mock_cache, ...)
```

### 2. **Flexibilidad**
- Cambiar Redis → Memcached: Solo cambiar adapter
- Cambiar DB → MongoDB: Solo cambiar repository adapter
- Domain permanece intacto

### 3. **Mantenibilidad**
- Cambios localizados por capa
- Fácil de entender y navegar
- Código organizado por responsabilidad

### 4. **Escalabilidad**
- Servicios independientes
- Fácil de dividir en microservicios
- Componentes reutilizables

### 5. **Portabilidad**
- Domain no depende de frameworks
- Puede ejecutarse sin FastAPI
- Facilita migración a otros frameworks

## 🔌 Dependency Injection

### FastAPI Dependencies
```python
# application/dependencies.py
def get_analysis_service(
    repository: Annotated[IAnalysisRepository, Depends(get_analysis_repository)],
    cache_service: Annotated[ICacheService, Depends(get_cache_service)],
    ml_service: Annotated[IMLService, Depends(get_ml_service)],
    event_bus: Annotated[IEventBus, Depends(get_event_bus)]
) -> AnalysisService:
    """Inyección automática de dependencias"""
    return AnalysisService(repository, cache_service, ml_service, event_bus)
```

## 📊 Event-Driven Architecture

```python
# Domain Event
@dataclass
class AnalysisCompletedEvent(DomainEvent):
    content_hash: str
    redundancy_score: float

# Application Service
event = AnalysisCompletedEvent(...)
await event_bus.publish(event.event_type, event.to_dict())

# Infrastructure Adapter
class InMemoryEventBus(IEventBus):
    # Puede cambiarse por RabbitMQ/Kafka adapter
```

## 🧪 Testing Strategy

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
    service = AnalysisService(mock_repo, mock_cache, ...)
    result = await service.analyze_content(AnalysisRequest(...))
```

### E2E Tests - API
```python
async def test_analyze_endpoint(client):
    response = await client.post("/api/v1/analyze", json={
        "content": "test content..."
    })
    assert response.status_code == 200
```

## 🚀 Migración a Microservicios

Esta arquitectura facilita la migración a microservicios:

1. **Separar por dominio**: Cada dominio puede ser un microservicio
2. **Interfaces de comunicación**: Eventos o gRPC/REST APIs
3. **Bases de datos independientes**: Cada servicio su propia DB
4. **Deploy independiente**: Cada servicio se despliega por separado

## 📝 Próximos Pasos

1. ✅ Arquitectura Clean/Hexagonal implementada
2. ⏳ Implementar más adapters (PostgreSQL, RabbitMQ, Kafka)
3. ⏳ Agregar CQRS si es necesario
4. ⏳ Implementar Event Sourcing para auditoría
5. ⏳ Agregar OpenTelemetry para distributed tracing
6. ⏳ Configurar service mesh (Istio/Linkerd)

## 📚 Referencias

- **Clean Architecture** - Robert C. Martin
- **Hexagonal Architecture** - Alistair Cockburn
- **Domain-Driven Design** - Eric Evans
- **Microservices Patterns** - Chris Richardson

---

**Estado**: ✅ Arquitectura Clean/Hexagonal completamente implementada
**Próximo**: Implementar adapters adicionales y optimizaciones






