# Arquitectura Modular V7.1 - Hexagonal Architecture

## Resumen

Arquitectura completamente modular implementando **Hexagonal Architecture (Ports & Adapters)** y **Clean Architecture** para máxima separación de responsabilidades y testabilidad.

## Capas de la Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│                    API Layer (Controllers)                │
│              Thin HTTP handlers, delegates to use cases   │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│              Application Layer (Use Cases)                │
│         Orchestrates domain logic, one use case = one op  │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                 Domain Layer (Core)                       │
│    Pure business logic, entities, interfaces (ports)      │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│            Infrastructure Layer (Adapters)                │
│    Implements domain interfaces, external services         │
└───────────────────────────────────────────────────────────┘
```

## Estructura de Directorios

```
dermatology_ai/
├── core/
│   ├── domain/              # Domain Layer (Inner)
│   │   ├── entities.py      # Business entities
│   │   ├── interfaces.py    # Ports (contracts)
│   │   └── value_objects.py # Value objects
│   │
│   ├── application/         # Application Layer
│   │   └── use_cases.py    # Use cases
│   │
│   ├── infrastructure/      # Infrastructure Layer (Outer)
│   │   ├── repositories.py  # Repository implementations
│   │   └── adapters.py     # Service adapters
│   │
│   ├── plugin_system.py    # Plugin system
│   ├── module_loader.py    # Dynamic module loading
│   ├── service_factory.py  # Dependency injection
│   └── composition_root.py # Dependency wiring
│
├── api/
│   └── controllers/        # API Layer
│       ├── analysis_controller.py
│       └── recommendation_controller.py
│
├── services/               # Legacy services (being migrated)
├── utils/                  # Utilities
└── plugins/                # Plugin directory
```

## Principios Aplicados

### 1. Dependency Inversion Principle
- Domain layer no depende de infrastructure
- Infrastructure implementa interfaces del domain
- Dependencias apuntan hacia adentro

### 2. Separation of Concerns
- Cada capa tiene responsabilidad única
- Domain: Lógica de negocio pura
- Application: Casos de uso
- Infrastructure: Detalles técnicos
- API: HTTP handling

### 3. Interface Segregation
- Interfaces específicas y pequeñas
- Cada interfaz tiene un propósito claro
- Fácil de mockear para testing

### 4. Single Responsibility
- Cada clase tiene una responsabilidad
- Use cases representan una operación
- Repositories solo manejan persistencia

## Flujo de Ejecución

### Ejemplo: Analizar Imagen

1. **HTTP Request** → `AnalysisController.analyze_image()`
2. **Controller** → `AnalyzeImageUseCase.execute()`
3. **Use Case** → Crea `Analysis` entity (domain)
4. **Use Case** → `IAnalysisRepository.create()` (interface)
5. **Repository** → `AnalysisRepository.create()` (implementation)
6. **Repository** → `IDatabaseAdapter.insert()` (adapter)
7. **Database** → Persiste datos

## Componentes Clave

### Domain Layer

#### Entities
```python
@dataclass
class Analysis:
    id: str
    user_id: str
    metrics: Optional[SkinMetrics] = None
    status: AnalysisStatus = AnalysisStatus.PENDING
    
    def mark_completed(self, metrics: SkinMetrics):
        """Business logic in entity"""
        self.metrics = metrics
        self.status = AnalysisStatus.COMPLETED
```

#### Interfaces (Ports)
```python
class IAnalysisRepository(ABC):
    @abstractmethod
    async def create(self, analysis: Analysis) -> Analysis:
        pass
```

### Application Layer

#### Use Cases
```python
class AnalyzeImageUseCase:
    def __init__(
        self,
        analysis_repository: IAnalysisRepository,
        image_processor: IImageProcessor
    ):
        # Dependencies injected via interfaces
    
    async def execute(self, user_id: str, image_data: bytes) -> Analysis:
        # Orchestrates domain logic
```

### Infrastructure Layer

#### Repository Implementation (Adapter)
```python
class AnalysisRepository(IAnalysisRepository):
    def __init__(self, database: IDatabaseAdapter):
        self.database = database
    
    async def create(self, analysis: Analysis) -> Analysis:
        # Converts entity to database format
        data = self._to_dict(analysis)
        await self.database.insert("analyses", data)
        return analysis
```

### API Layer

#### Controller
```python
class AnalysisController:
    def __init__(self, analyze_use_case: AnalyzeImageUseCase):
        self.analyze_use_case = analyze_use_case
    
    @router.post("/analyze")
    async def analyze_image(self, file: UploadFile):
        # Thin controller, delegates to use case
        analysis = await self.analyze_use_case.execute(
            user_id=current_user["sub"],
            image_data=await file.read()
        )
        return {"analysis_id": analysis.id}
```

## Composition Root

El `CompositionRoot` es responsable de:
- Wire up todas las dependencias
- Crear instancias de servicios
- Resolver dependencias automáticamente
- Mantener domain layer puro

```python
composition_root = get_composition_root()
await composition_root.initialize(config)

# Get use cases with dependencies resolved
analyze_use_case = await composition_root.get_analyze_image_use_case()
```

## Testing Strategy

### Domain Layer Tests
- Test puro, sin mocks
- Test lógica de negocio
- No dependencias externas

```python
def test_analysis_mark_completed():
    analysis = Analysis(id="1", user_id="user1")
    metrics = SkinMetrics(overall_score=80.0, ...)
    analysis.mark_completed(metrics)
    assert analysis.status == AnalysisStatus.COMPLETED
```

### Use Case Tests
- Mock interfaces
- Test orquestación
- Test casos de error

```python
async def test_analyze_image_use_case():
    mock_repo = Mock(IAnalysisRepository)
    mock_processor = Mock(IImageProcessor)
    
    use_case = AnalyzeImageUseCase(mock_repo, mock_processor)
    result = await use_case.execute("user1", b"image_data")
    
    assert result.status == AnalysisStatus.COMPLETED
    mock_repo.create.assert_called_once()
```

### Repository Tests
- Mock database adapter
- Test conversión entity ↔ database
- Test queries

```python
async def test_analysis_repository():
    mock_db = Mock(IDatabaseAdapter)
    repo = AnalysisRepository(mock_db)
    
    analysis = Analysis(id="1", user_id="user1")
    await repo.create(analysis)
    
    mock_db.insert.assert_called_once()
```

## Ventajas de la Arquitectura

### 1. Testabilidad
- ✅ Domain testeable sin infraestructura
- ✅ Use cases testeables con mocks
- ✅ Repositories intercambiables

### 2. Mantenibilidad
- ✅ Cambios en infraestructura no afectan domain
- ✅ Lógica de negocio aislada
- ✅ Fácil de entender

### 3. Flexibilidad
- ✅ Cambiar database sin afectar lógica
- ✅ Intercambiar servicios externos
- ✅ Agregar nuevas implementaciones

### 4. Escalabilidad
- ✅ Cada capa puede escalarse independientemente
- ✅ Fácil migrar a microservicios
- ✅ Componentes reutilizables

## Migración desde V7.0

### Paso 1: Identificar Entidades
```python
# Extraer lógica de negocio a entities
# Mover de services/ a core/domain/entities.py
```

### Paso 2: Definir Interfaces
```python
# Crear interfaces en core/domain/interfaces.py
# Separar qué (interface) de cómo (implementation)
```

### Paso 3: Crear Use Cases
```python
# Extraer lógica de controllers a use cases
# Un use case = una operación de negocio
```

### Paso 4: Implementar Repositories
```python
# Crear implementaciones en core/infrastructure/repositories.py
# Implementar interfaces del domain
```

### Paso 5: Refactorizar Controllers
```python
# Hacer controllers delgados
# Delegar a use cases
```

## Mejores Prácticas

### 1. Domain Layer
- ✅ Solo lógica de negocio
- ✅ Sin dependencias de infraestructura
- ✅ Entities con comportamiento
- ✅ Value objects inmutables

### 2. Application Layer
- ✅ Un use case = una operación
- ✅ Orquesta domain logic
- ✅ No contiene lógica de negocio
- ✅ Coordina entre servicios

### 3. Infrastructure Layer
- ✅ Implementa interfaces del domain
- ✅ Maneja detalles técnicos
- ✅ Conversión entity ↔ database
- ✅ Adaptadores para servicios externos

### 4. API Layer
- ✅ Controllers delgados
- ✅ Solo HTTP handling
- ✅ Validación de entrada
- ✅ Delegación a use cases

## Conclusión

La arquitectura V7.1 proporciona:

- ✅ **Máxima Modularidad**: Hexagonal Architecture
- ✅ **Testabilidad**: Domain puro, fácil de testear
- ✅ **Flexibilidad**: Interfaces intercambiables
- ✅ **Mantenibilidad**: Separación clara de responsabilidades
- ✅ **Escalabilidad**: Componentes independientes
- ✅ **Clean Code**: Principios SOLID aplicados

El sistema está ahora completamente modular con arquitectura hexagonal (Clean Architecture) implementada.















