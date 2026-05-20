# Use Cases - Completados ✅

## Resumen

Se ha completado el **Paso 3** del plan de mejoras arquitectónicas: **Crear Primer Use Case de Ejemplo**.

## Estructura Creada

```
application/
├── __init__.py
├── exceptions.py                    # Excepciones de aplicación
├── dto/                             # Data Transfer Objects
│   ├── __init__.py
│   ├── analysis.py                 # DTOs de análisis
│   ├── recommendations.py          # DTOs de recomendaciones
│   └── coaching.py                 # DTOs de coaching
└── use_cases/                       # Casos de uso
    ├── __init__.py
    ├── analysis/
    │   ├── __init__.py
    │   ├── analyze_track.py        # Analizar track
    │   └── search_tracks.py        # Buscar tracks
    └── recommendations/
        ├── __init__.py
        ├── get_recommendations.py  # Obtener recomendaciones
        └── generate_playlist.py     # Generar playlist
```

## Use Cases Implementados

### 1. AnalyzeTrackUseCase (`analysis/analyze_track.py`)

**Propósito**: Orquestar el análisis completo de un track musical.

**Dependencias**:
- `ISpotifyService` - Para obtener datos de Spotify
- `ITrackRepository` - Para validar y obtener tracks
- `IAnalysisService` - Para realizar el análisis
- `ICoachingService` - Opcional, para coaching

**Método principal**:
```python
async def execute(
    track_id: str,
    include_coaching: bool = False
) -> AnalysisResultDTO
```

**Flujo**:
1. Valida que el track existe
2. Obtiene datos completos de Spotify
3. Realiza análisis musical
4. Opcionalmente genera coaching
5. Retorna DTO con resultados

**Excepciones**:
- `TrackNotFoundException` - Si el track no existe
- `AnalysisException` - Si el análisis falla

### 2. SearchTracksUseCase (`analysis/search_tracks.py`)

**Propósito**: Orquestar la búsqueda de tracks.

**Dependencias**:
- `ITrackRepository` - Para buscar tracks

**Método principal**:
```python
async def execute(
    query: str,
    limit: int = 10,
    offset: int = 0
) -> Dict[str, Any]
```

**Flujo**:
1. Valida query y parámetros
2. Busca tracks
3. Convierte a DTOs
4. Retorna resultados formateados

### 3. GetRecommendationsUseCase (`recommendations/get_recommendations.py`)

**Propósito**: Obtener recomendaciones de tracks.

**Dependencias**:
- `ITrackRepository` - Para validar track
- `IRecommendationService` - Para generar recomendaciones

**Método principal**:
```python
async def execute(
    track_id: str,
    limit: int = 20,
    method: str = "similarity",
    mood: Optional[str] = None,
    genre: Optional[str] = None
) -> List[RecommendationDTO]
```

**Métodos soportados**:
- `similarity` - Tracks similares
- `mood` - Basado en mood
- `genre` - Basado en género

### 4. GeneratePlaylistUseCase (`recommendations/generate_playlist.py`)

**Propósito**: Generar una playlist basada en criterios.

**Dependencias**:
- `IRecommendationService` - Para generar playlist

**Método principal**:
```python
async def execute(
    criteria: Dict[str, Any],
    length: int = 20
) -> PlaylistDTO
```

## DTOs Creados

### Analysis DTOs

**TrackAnalysisDTO**
- `track_id`, `track_name`, `artists`
- `album`, `duration_ms`, `preview_url`, `popularity`

**AnalysisResultDTO**
- `track_id`, `track_name`, `artists`
- `analysis` - Resultado completo del análisis
- `coaching` - Opcional, datos de coaching
- `to_dict()` - Método para convertir a diccionario

### Recommendation DTOs

**RecommendationDTO**
- `track_id`, `track_name`, `artists`
- `similarity_score`, `reason`
- `to_dict()` - Método para convertir a diccionario

**PlaylistDTO**
- `tracks` - Lista de RecommendationDTOs
- `total_tracks`, `criteria`
- `to_dict()` - Método para convertir a diccionario

### Coaching DTOs

**LearningStepDTO**
- `step`, `title`, `description`, `duration`
- `exercises` - Lista opcional de ejercicios

**ExerciseDTO**
- `name`, `description`, `difficulty`, `duration`
- `focus_areas` - Áreas de enfoque

**CoachingDTO**
- `overview`, `learning_path`, `practice_exercises`
- `performance_tips`, `difficulty_level`

## Excepciones Creadas

- `UseCaseException` - Base para todas las excepciones
- `TrackNotFoundException` - Track no encontrado
- `InvalidTrackIDException` - ID inválido
- `AnalysisException` - Error en análisis
- `RecommendationException` - Error en recomendaciones
- `CoachingException` - Error en coaching

## Ejemplo de Uso

### En un Controller

```python
from fastapi import Depends
from application.use_cases.analysis import AnalyzeTrackUseCase
from api.dependencies import get_analyze_track_use_case

@router.post("/analyze")
async def analyze_track(
    track_id: str,
    include_coaching: bool = False,
    use_case: AnalyzeTrackUseCase = Depends(get_analyze_track_use_case)
):
    try:
        result = await use_case.execute(track_id, include_coaching)
        return result.to_dict()
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AnalysisException as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### En Tests

```python
from unittest.mock import Mock, AsyncMock
from application.use_cases.analysis import AnalyzeTrackUseCase

async def test_analyze_track():
    # Crear mocks
    mock_spotify = Mock()
    mock_spotify.get_track = AsyncMock(return_value={"id": "123", "name": "Test"})
    mock_spotify.get_track_full_analysis = AsyncMock(return_value={
        "track_info": {},
        "audio_features": {},
        "audio_analysis": {}
    })
    
    mock_repo = Mock()
    mock_repo.get_by_id = AsyncMock(return_value={"id": "123"})
    
    mock_analyzer = Mock()
    mock_analyzer.analyze_track = AsyncMock(return_value={"analysis": "data"})
    
    # Crear use case
    use_case = AnalyzeTrackUseCase(
        spotify_service=mock_spotify,
        track_repository=mock_repo,
        analysis_service=mock_analyzer
    )
    
    # Ejecutar
    result = await use_case.execute("123")
    
    # Verificar
    assert result.track_id == "123"
    assert result.analysis is not None
```

## Beneficios

### 1. Separación de Responsabilidades
- La lógica de negocio está en use cases
- Los controllers solo orquestan y formatean respuestas

### 2. Testabilidad
- Fácil de testear con mocks
- Tests unitarios rápidos y aislados

### 3. Reutilización
- Los use cases pueden usarse desde diferentes puntos de entrada
- API, CLI, workers, etc.

### 4. Mantenibilidad
- Lógica centralizada y clara
- Fácil de entender y modificar

## Próximos Pasos

1. **Paso 4**: Implementar repositorios usando las interfaces
2. **Paso 5**: Migrar servicios existentes para implementar interfaces
3. **Paso 6**: Crear más use cases (coaching, export, etc.)
4. **Paso 7**: Integrar use cases en controllers

## Notas Importantes

- ✅ Todos los use cases son asíncronos
- ✅ Todos usan interfaces de dominio (no implementaciones concretas)
- ✅ Todos retornan DTOs (no entidades de dominio)
- ✅ Manejo de errores consistente
- ✅ Logging apropiado

---

**Estado**: ✅ Completado  
**Fecha**: 2024  
**Próximo**: Implementar repositorios




