# Arquitectura Rápida - Music Analyzer AI

## 🏗️ Visión General

Music Analyzer AI sigue una **arquitectura en capas** con **Clean Architecture** y **Domain-Driven Design** (DDD).

## 📐 Estructura de Capas

```
┌─────────────────────────────────────┐
│         API Layer (FastAPI)          │  ← Endpoints HTTP
├─────────────────────────────────────┤
│      Application Layer (Use Cases)  │  ← Lógica de negocio
├─────────────────────────────────────┤
│        Domain Layer (Interfaces)    │  ← Contratos y entidades
├─────────────────────────────────────┤
│   Infrastructure Layer (Services)    │  ← Implementaciones
└─────────────────────────────────────┘
```

## 📁 Estructura del Proyecto

```
music_analyzer_ai/
├── api/                          # Capa de API
│   ├── v1/                       # API v1 (nueva arquitectura)
│   │   ├── controllers/         # Controladores
│   │   ├── schemas/             # Esquemas Pydantic
│   │   └── routes.py            # Router principal
│   ├── routes/                   # Routers modulares (legacy)
│   ├── dependencies.py           # Dependencias FastAPI
│   ├── factories.py              # Factory functions
│   └── music_api.py              # API legacy
│
├── application/                  # Capa de Aplicación
│   └── use_cases/                # Casos de uso
│       ├── analysis/            # Análisis musical
│       └── recommendations/     # Recomendaciones
│
├── domain/                       # Capa de Dominio
│   └── interfaces/               # Interfaces/Contratos
│       ├── spotify.py            # ISpotifyService
│       ├── analysis.py           # IAnalysisService
│       └── repositories.py       # IRepository
│
├── infrastructure/               # Capa de Infraestructura
│   ├── repositories/             # Implementaciones de repositorios
│   ├── services/                 # Servicios concretos
│   └── adapters/                 # Adaptadores externos
│
├── core/                         # Core del sistema
│   └── di/                       # Dependency Injection
│
├── services/                     # Servicios de negocio
│   ├── spotify_service.py        # Servicio de Spotify
│   └── music_coach.py            # Coaching musical
│
└── config/                       # Configuración
    ├── settings.py               # Configuración
    └── di_setup.py               # Setup de DI
```

## 🔄 Flujo de Datos

### Ejemplo: Analizar una Canción

```
1. Cliente → POST /v1/music/analyze
   ↓
2. Controller → Valida request con Pydantic
   ↓
3. Use Case → Ejecuta lógica de negocio
   ↓
4. Repository → Obtiene datos de Spotify
   ↓
5. Service → Procesa y analiza
   ↓
6. Use Case → Retorna resultado
   ↓
7. Controller → Formatea respuesta
   ↓
8. Cliente ← JSON response
```

## 🎯 Patrones Arquitectónicos

### 1. Dependency Injection (DI)

Todos los servicios se inyectan mediante DI:

```python
# config/di_setup.py
register_service("spotify_service", SpotifyService, singleton=True)

# En controllers
@router.post("/analyze")
async def analyze(
    use_case: AnalyzeTrackUseCase = Depends(get_analyze_track_use_case)
):
    result = await use_case.execute(...)
```

### 2. Use Cases Pattern

La lógica de negocio está en casos de uso:

```python
# application/use_cases/analysis/analyze_track.py
class AnalyzeTrackUseCase:
    def __init__(
        self,
        spotify_service: ISpotifyService,
        analysis_service: IAnalysisService
    ):
        self.spotify_service = spotify_service
        self.analysis_service = analysis_service
    
    async def execute(self, track_id: str):
        # Lógica de negocio aquí
        track = await self.spotify_service.get_track(track_id)
        analysis = await self.analysis_service.analyze(track)
        return analysis
```

### 3. Repository Pattern

Abstrae el acceso a datos:

```python
# domain/interfaces/repositories.py
class ITrackRepository(ABC):
    @abstractmethod
    async def get_by_id(self, track_id: str) -> Track:
        pass

# infrastructure/repositories/spotify_track_repository.py
class SpotifyTrackRepository(ITrackRepository):
    async def get_by_id(self, track_id: str) -> Track:
        # Implementación con Spotify API
        pass
```

### 4. Factory Pattern

Crea servicios de forma centralizada:

```python
# api/factories.py
def get_spotify_service() -> ISpotifyService:
    return get_service("spotify_service")
```

## 🔌 Interfaces Principales

### ISpotifyService

```python
class ISpotifyService(ABC):
    @abstractmethod
    async def search_tracks(self, query: str, limit: int) -> List[Track]:
        pass
    
    @abstractmethod
    async def get_track(self, track_id: str) -> Track:
        pass
```

### IAnalysisService

```python
class IAnalysisService(ABC):
    @abstractmethod
    async def analyze_track(self, track: Track) -> Analysis:
        pass
```

### IRepository

```python
class ITrackRepository(ABC):
    @abstractmethod
    async def get_by_id(self, track_id: str) -> Track:
        pass
    
    @abstractmethod
    async def search(self, query: str) -> List[Track]:
        pass
```

## 🧩 Componentes Principales

### 1. API Layer

**Responsabilidad**: Manejar requests HTTP y validación

- **Controllers**: Manejan requests/responses
- **Schemas**: Validación con Pydantic
- **Routes**: Definición de endpoints

### 2. Application Layer

**Responsabilidad**: Lógica de negocio

- **Use Cases**: Casos de uso específicos
- **DTOs**: Data Transfer Objects

### 3. Domain Layer

**Responsabilidad**: Contratos y entidades

- **Interfaces**: Contratos de servicios
- **Entities**: Entidades de dominio

### 4. Infrastructure Layer

**Responsabilidad**: Implementaciones concretas

- **Repositories**: Acceso a datos
- **Services**: Servicios externos
- **Adapters**: Adaptadores para APIs externas

## 🔧 Configuración de DI

### Registrar un Servicio

```python
# config/di_setup.py
from core.di.container import register_service

register_service(
    "my_service",
    MyService,
    singleton=True,
    dependencies=["other_service"]
)
```

### Usar un Servicio

```python
# En controllers
from ...dependencies import get_my_service

@router.get("/endpoint")
async def my_endpoint(
    service: MyService = Depends(get_my_service)
):
    result = service.do_something()
    return result
```

## 📊 Ventajas de esta Arquitectura

1. **Testabilidad**: Fácil de mockear interfaces
2. **Mantenibilidad**: Separación clara de responsabilidades
3. **Escalabilidad**: Fácil agregar nuevas features
4. **Flexibilidad**: Cambiar implementaciones sin afectar otras capas

## 🚀 Migración de Legacy a Nueva Arquitectura

### Antes (Legacy)

```python
@router.post("/analyze")
async def analyze_track(track_id: str):
    spotify = SpotifyService()  # Instanciación directa
    analyzer = MusicAnalyzer()  # Instanciación directa
    
    track = spotify.get_track(track_id)
    analysis = analyzer.analyze(track)
    return analysis
```

### Después (Nueva Arquitectura)

```python
@router.post("/analyze")
async def analyze_track(
    request: AnalyzeTrackRequest,
    use_case: AnalyzeTrackUseCase = Depends(get_analyze_track_use_case)
):
    result = await use_case.execute(request.track_id)
    return build_analysis_response(result)
```

## 📚 Recursos Adicionales

- **Clean Architecture**: [../ARCHITECTURE_IMPROVEMENTS.md](../ARCHITECTURE_IMPROVEMENTS.md)
- **Refactoring Guide**: [../REFACTORING_GUIDE.md](../REFACTORING_GUIDE.md)
- **Use Cases**: [../USE_CASES_COMPLETE.md](../USE_CASES_COMPLETE.md)

---

**Última actualización**: 2025  
**Versión**: 2.21.0






