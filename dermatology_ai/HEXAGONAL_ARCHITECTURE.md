# Arquitectura Hexagonal (Ports & Adapters) - V7.1.0

## Resumen

Implementación de Arquitectura Hexagonal (Clean Architecture) para máxima modularidad y separación de responsabilidades.

## Capas de la Arquitectura

### 1. Domain Layer (`core/domain/`)

**Responsabilidad**: Lógica de negocio pura, sin dependencias de infraestructura.

#### Componentes:
- **Entities** (`entities.py`): Objetos de dominio con lógica de negocio
  - `Analysis` - Entidad principal de análisis
  - `User` - Entidad de usuario
  - `Product` - Entidad de producto
  - `SkinMetrics` - Value object para métricas
  - `Condition` - Value object para condiciones

- **Interfaces** (`interfaces.py`): Contratos (Ports)
  - `IAnalysisRepository` - Contrato para persistencia
  - `IUserRepository` - Contrato para usuarios
  - `IProductRepository` - Contrato para productos
  - `IImageProcessor` - Contrato para procesamiento
  - `IRecommendationService` - Contrato para recomendaciones
  - `ICacheService` - Contrato para cache
  - `IEventPublisher` - Contrato para eventos

- **Value Objects** (`value_objects.py`): Objetos inmutables
  - `SkinMetrics` - Métricas de piel
  - `Recommendation` - Recomendación

### 2. Application Layer (`core/application/`)

**Responsabilidad**: Casos de uso y orquestación de lógica de dominio.

#### Componentes:
- **Use Cases** (`use_cases.py`): Casos de uso específicos
  - `AnalyzeImageUseCase` - Analizar imagen
  - `GetRecommendationsUseCase` - Obtener recomendaciones
  - `GetAnalysisHistoryUseCase` - Obtener historial

### 3. Infrastructure Layer (`core/infrastructure/`)

**Responsabilidad**: Implementaciones de interfaces (Adapters).

#### Componentes:
- **Repositories** (`repositories.py`): Implementaciones de repositorios
  - `AnalysisRepository` - Implementa `IAnalysisRepository`
  - `UserRepository` - Implementa `IUserRepository`
  - `ProductRepository` - Implementa `IProductRepository`

- **Adapters** (`adapters.py`): Adaptadores para servicios externos
  - `ImageProcessorAdapter` - Adapta servicio de procesamiento
  - `CacheAdapter` - Adapta servicio de cache
  - `EventPublisherAdapter` - Adapta message broker

### 4. API Layer (`api/controllers/`)

**Responsabilidad**: Handlers HTTP, capa delgada que delega a use cases.

#### Componentes:
- **Controllers** (`*.py`): Controladores HTTP
  - `AnalysisController` - Endpoints de análisis
  - `RecommendationController` - Endpoints de recomendaciones

## Flujo de Datos

```
HTTP Request
    ↓
Controller (API Layer)
    ↓
Use Case (Application Layer)
    ↓
Domain Entity (Domain Layer)
    ↓
Repository Interface (Domain Layer)
    ↓
Repository Implementation (Infrastructure Layer)
    ↓
Database (External)
```

## Principios Aplicados

### 1. Dependency Inversion
- Domain layer no depende de infrastructure
- Infrastructure implementa interfaces del domain
- Dependencias apuntan hacia adentro

### 2. Separation of Concerns
- Cada capa tiene responsabilidad única
- Domain: Lógica de negocio
- Application: Casos de uso
- Infrastructure: Detalles técnicos
- API: HTTP handling

### 3. Interface Segregation
- Interfaces específicas y pequeñas
- Cada interfaz tiene un propósito claro
- Fácil de mockear para testing

### 4. Single Responsibility
- Cada clase tiene una responsabilidad
- Use cases representan una operación de negocio
- Repositories solo manejan persistencia

## Ejemplo de Uso

### 1. Definir Entidad de Dominio

```python
# core/domain/entities.py
@dataclass
class Analysis:
    id: str
    user_id: str
    metrics: Optional[SkinMetrics] = None
    status: AnalysisStatus = AnalysisStatus.PENDING
    
    def mark_completed(self, metrics: SkinMetrics):
        """Business logic"""
        self.metrics = metrics
        self.status = AnalysisStatus.COMPLETED
```

### 2. Definir Interfaz (Port)

```python
# core/domain/interfaces.py
class IAnalysisRepository(ABC):
    @abstractmethod
    async def create(self, analysis: Analysis) -> Analysis:
        pass
```

### 3. Implementar Repositorio (Adapter)

```python
# core/infrastructure/repositories.py
class AnalysisRepository(IAnalysisRepository):
    def __init__(self, database: IDatabaseAdapter):
        self.database = database
    
    async def create(self, analysis: Analysis) -> Analysis:
        # Convert entity to database format
        data = self._to_dict(analysis)
        await self.database.insert("analyses", data)
        return analysis
```

### 4. Crear Use Case

```python
# core/application/use_cases.py
class AnalyzeImageUseCase:
    def __init__(
        self,
        analysis_repository: IAnalysisRepository,
        image_processor: IImageProcessor
    ):
        self.analysis_repository = analysis_repository
        self.image_processor = image_processor
    
    async def execute(self, user_id: str, image_data: bytes) -> Analysis:
        # Business logic orchestration
        analysis = Analysis(id=..., user_id=user_id)
        await self.analysis_repository.create(analysis)
        # Process image...
        return analysis
```

### 5. Crear Controller

```python
# api/controllers/analysis_controller.py
class AnalysisController:
    def __init__(self, analyze_use_case: AnalyzeImageUseCase):
        self.analyze_use_case = analyze_use_case
    
    @router.post("/analyze")
    async def analyze_image(self, file: UploadFile):
        image_data = await file.read()
        analysis = await self.analyze_use_case.execute(
            user_id=current_user["sub"],
            image_data=image_data
        )
        return {"analysis_id": analysis.id}
```

## Ventajas

### 1. Testabilidad
- Domain layer testeable sin infraestructura
- Use cases testeables con mocks
- Repositories intercambiables

### 2. Mantenibilidad
- Cambios en infraestructura no afectan domain
- Lógica de negocio aislada
- Fácil de entender

### 3. Flexibilidad
- Cambiar database sin afectar lógica
- Intercambiar servicios externos
- Agregar nuevas implementaciones

### 4. Escalabilidad
- Cada capa puede escalarse independientemente
- Fácil migrar a microservicios
- Componentes reutilizables

## Testing

### Domain Layer Tests
```python
def test_analysis_mark_completed():
    analysis = Analysis(id="1", user_id="user1")
    metrics = SkinMetrics(overall_score=80.0, ...)
    analysis.mark_completed(metrics)
    assert analysis.status == AnalysisStatus.COMPLETED
```

### Use Case Tests
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
```python
async def test_analysis_repository():
    mock_db = Mock(IDatabaseAdapter)
    repo = AnalysisRepository(mock_db)
    
    analysis = Analysis(id="1", user_id="user1")
    await repo.create(analysis)
    
    mock_db.insert.assert_called_once()
```

## Migración desde V7.0

1. **Identificar entidades de dominio**
   - Extraer lógica de negocio
   - Crear entities puras

2. **Definir interfaces**
   - Crear contratos para servicios
   - Separar qué de cómo

3. **Implementar adapters**
   - Crear implementaciones de interfaces
   - Mantener domain puro

4. **Crear use cases**
   - Extraer lógica de controllers
   - Orquestar domain logic

5. **Refactorizar controllers**
   - Hacer controllers delgados
   - Delegar a use cases

## Conclusión

La arquitectura hexagonal proporciona:

- ✅ **Máxima Modularidad**: Capas completamente separadas
- ✅ **Testabilidad**: Domain testeable sin infraestructura
- ✅ **Flexibilidad**: Fácil cambiar implementaciones
- ✅ **Mantenibilidad**: Lógica de negocio aislada
- ✅ **Escalabilidad**: Componentes independientes

El sistema está ahora completamente modular con arquitectura hexagonal (Clean Architecture).















