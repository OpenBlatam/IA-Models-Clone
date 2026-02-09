# Ejemplos de Implementación - Mejoras Arquitectónicas

Este documento contiene ejemplos concretos de cómo implementar las mejoras arquitectónicas propuestas.

## 1. Sistema de DI Mejorado

### `core/di/container.py`

```python
"""
Container de inyección de dependencias mejorado
"""
from typing import Dict, Any, Optional, Type, Callable, TypeVar, Generic
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceContainer:
    """Container mejorado con soporte para scopes y decoradores"""
    
    def __init__(self):
        self._services: Dict[str, Type] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._scoped: Dict[str, Any] = {}
        self._dependencies: Dict[str, list] = {}
    
    def register(
        self,
        service_name: str,
        service_class: Type[T],
        singleton: bool = True,
        factory: Optional[Callable] = None,
        dependencies: Optional[list] = None
    ) -> None:
        """Registra un servicio con sus dependencias"""
        if factory:
            self._factories[service_name] = factory
        else:
            self._services[service_name] = service_class
        
        if dependencies:
            self._dependencies[service_name] = dependencies
        
        if singleton:
            self._singletons[service_name] = None  # Placeholder
    
    def resolve(self, service_name: str, scope: Optional[str] = None) -> Any:
        """Resuelve una dependencia"""
        # Check scoped instances
        if scope and service_name in self._scoped:
            return self._scoped[service_name]
        
        # Check singletons
        if service_name in self._singletons:
            if self._singletons[service_name] is None:
                self._singletons[service_name] = self._create_instance(service_name)
            return self._singletons[service_name]
        
        # Create new instance
        return self._create_instance(service_name)
    
    def _create_instance(self, service_name: str) -> Any:
        """Crea una instancia del servicio resolviendo dependencias"""
        if service_name in self._factories:
            factory = self._factories[service_name]
            deps = self._resolve_dependencies(service_name)
            return factory(**deps)
        
        if service_name in self._services:
            service_class = self._services[service_name]
            deps = self._resolve_dependencies(service_name)
            return service_class(**deps)
        
        raise ValueError(f"Service '{service_name}' not registered")
    
    def _resolve_dependencies(self, service_name: str) -> Dict[str, Any]:
        """Resuelve las dependencias de un servicio"""
        deps = {}
        if service_name in self._dependencies:
            for dep_name in self._dependencies[service_name]:
                deps[dep_name] = self.resolve(dep_name)
        return deps


# Global container
_container = ServiceContainer()


def get_container() -> ServiceContainer:
    """Obtiene el container global"""
    return _container


def register_service(
    service_name: str,
    service_class: Type[T],
    singleton: bool = True,
    dependencies: Optional[list] = None
) -> None:
    """Helper para registrar servicios"""
    _container.register(service_name, service_class, singleton, None, dependencies)


def resolve(service_name: str) -> Any:
    """Helper para resolver servicios"""
    return _container.resolve(service_name)
```

## 2. Interfaces de Dominio

### `domain/interfaces/analysis.py`

```python
"""
Interfaces para servicios de análisis
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class TrackAnalysis:
    """Value object para análisis de track"""
    track_id: str
    key_signature: str
    tempo: float
    energy: float
    danceability: float
    # ... más campos


class IAnalysisService(ABC):
    """Interface para servicio de análisis"""
    
    @abstractmethod
    async def analyze_track(self, track_id: str) -> TrackAnalysis:
        """Analiza un track musical"""
        pass
    
    @abstractmethod
    async def analyze_tracks_batch(self, track_ids: List[str]) -> List[TrackAnalysis]:
        """Analiza múltiples tracks"""
        pass


class IHarmonicAnalyzer(ABC):
    """Interface para análisis armónico"""
    
    @abstractmethod
    async def analyze_harmony(self, track_id: str) -> Dict[str, Any]:
        """Analiza la armonía de un track"""
        pass


class IStructureAnalyzer(ABC):
    """Interface para análisis de estructura"""
    
    @abstractmethod
    async def analyze_structure(self, track_id: str) -> Dict[str, Any]:
        """Analiza la estructura de un track"""
        pass
```

### `domain/interfaces/repositories.py`

```python
"""
Interfaces para repositorios
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from domain.entities.track import Track


class ITrackRepository(ABC):
    """Interface para repositorio de tracks"""
    
    @abstractmethod
    async def get_by_id(self, track_id: str) -> Optional[Track]:
        """Obtiene un track por ID"""
        pass
    
    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[Track]:
        """Busca tracks"""
        pass
    
    @abstractmethod
    async def get_audio_features(self, track_id: str) -> Dict[str, Any]:
        """Obtiene características de audio"""
        pass
```

## 3. Use Cases

### `application/use_cases/analysis/analyze_track.py`

```python
"""
Caso de uso: Analizar un track
"""
from typing import Dict, Any
from domain.interfaces.repositories import ITrackRepository
from domain.interfaces.analysis import IAnalysisService
from application.dto.analysis import AnalysisResultDTO
from application.exceptions import TrackNotFoundException


class AnalyzeTrackUseCase:
    """Caso de uso para analizar un track"""
    
    def __init__(
        self,
        track_repository: ITrackRepository,
        analysis_service: IAnalysisService
    ):
        self.track_repository = track_repository
        self.analysis_service = analysis_service
    
    async def execute(self, track_id: str, include_coaching: bool = False) -> AnalysisResultDTO:
        """
        Ejecuta el análisis de un track
        
        Args:
            track_id: ID del track a analizar
            include_coaching: Si incluir coaching en el resultado
        
        Returns:
            AnalysisResultDTO con el resultado del análisis
        
        Raises:
            TrackNotFoundException: Si el track no existe
        """
        # 1. Validar que el track existe
        track = await self.track_repository.get_by_id(track_id)
        if not track:
            raise TrackNotFoundException(f"Track {track_id} not found")
        
        # 2. Realizar análisis
        analysis = await self.analysis_service.analyze_track(track_id)
        
        # 3. Construir DTO de resultado
        result = AnalysisResultDTO(
            track_id=track_id,
            track_name=track.name,
            artists=track.artists,
            analysis=analysis,
            coaching=None  # Se puede agregar después si se requiere
        )
        
        # 4. Agregar coaching si se solicita
        if include_coaching:
            # Lógica para agregar coaching
            pass
        
        return result
```

### `application/use_cases/recommendations/get_recommendations.py`

```python
"""
Caso de uso: Obtener recomendaciones
"""
from typing import List
from domain.interfaces.repositories import ITrackRepository
from domain.interfaces.recommendations import IRecommendationService
from application.dto.recommendations import RecommendationDTO


class GetRecommendationsUseCase:
    """Caso de uso para obtener recomendaciones"""
    
    def __init__(
        self,
        track_repository: ITrackRepository,
        recommendation_service: IRecommendationService
    ):
        self.track_repository = track_repository
        self.recommendation_service = recommendation_service
    
    async def execute(
        self,
        track_id: str,
        limit: int = 20,
        method: str = "similarity"
    ) -> List[RecommendationDTO]:
        """
        Obtiene recomendaciones para un track
        
        Args:
            track_id: ID del track base
            limit: Número máximo de recomendaciones
            method: Método de recomendación (similarity, mood, genre)
        
        Returns:
            Lista de recomendaciones
        """
        # Validar track
        track = await self.track_repository.get_by_id(track_id)
        if not track:
            raise TrackNotFoundException(f"Track {track_id} not found")
        
        # Obtener recomendaciones según el método
        if method == "similarity":
            recommendations = await self.recommendation_service.get_similar_tracks(
                track_id, limit
            )
        elif method == "mood":
            recommendations = await self.recommendation_service.get_mood_based_recommendations(
                track_id, limit
            )
        else:
            recommendations = await self.recommendation_service.get_genre_based_recommendations(
                track_id, limit
            )
        
        # Convertir a DTOs
        return [
            RecommendationDTO(
                track_id=rec.track_id,
                track_name=rec.track_name,
                artists=rec.artists,
                similarity_score=rec.similarity_score,
                reason=rec.reason
            )
            for rec in recommendations
        ]
```

## 4. Implementaciones de Infraestructura

### `infrastructure/repositories/spotify_track_repository.py`

```python
"""
Implementación de repositorio usando Spotify API
"""
from typing import List, Optional, Dict, Any
from domain.interfaces.repositories import ITrackRepository
from domain.entities.track import Track
from infrastructure.external.spotify.spotify_client import SpotifyClient


class SpotifyTrackRepository(ITrackRepository):
    """Repositorio de tracks usando Spotify"""
    
    def __init__(self, spotify_client: SpotifyClient):
        self.client = spotify_client
    
    async def get_by_id(self, track_id: str) -> Optional[Track]:
        """Obtiene un track por ID desde Spotify"""
        try:
            data = await self.client.get_track(track_id)
            return self._map_to_track(data)
        except Exception:
            return None
    
    async def search(self, query: str, limit: int = 10) -> List[Track]:
        """Busca tracks en Spotify"""
        results = await self.client.search(query, limit=limit)
        return [self._map_to_track(item) for item in results.get('tracks', {}).get('items', [])]
    
    async def get_audio_features(self, track_id: str) -> Dict[str, Any]:
        """Obtiene características de audio desde Spotify"""
        return await self.client.get_audio_features(track_id)
    
    def _map_to_track(self, data: Dict[str, Any]) -> Track:
        """Mapea datos de Spotify a entidad Track"""
        return Track(
            id=data['id'],
            name=data['name'],
            artists=[artist['name'] for artist in data.get('artists', [])],
            album=data.get('album', {}).get('name'),
            duration_ms=data.get('duration_ms', 0),
            preview_url=data.get('preview_url')
        )
```

### `infrastructure/services/analysis_service.py`

```python
"""
Implementación del servicio de análisis
"""
from domain.interfaces.analysis import IAnalysisService, TrackAnalysis
from domain.interfaces.repositories import ITrackRepository
from infrastructure.cache.cache_manager import CacheManager


class AnalysisService(IAnalysisService):
    """Servicio de análisis implementado"""
    
    def __init__(
        self,
        track_repository: ITrackRepository,
        cache_manager: CacheManager
    ):
        self.track_repository = track_repository
        self.cache = cache_manager
    
    async def analyze_track(self, track_id: str) -> TrackAnalysis:
        """Analiza un track con cache"""
        # Verificar cache
        cache_key = f"analysis:{track_id}"
        cached = await self.cache.get(cache_key)
        if cached:
            return TrackAnalysis(**cached)
        
        # Obtener datos
        audio_features = await self.track_repository.get_audio_features(track_id)
        track = await self.track_repository.get_by_id(track_id)
        
        # Realizar análisis
        analysis = TrackAnalysis(
            track_id=track_id,
            key_signature=self._get_key_signature(audio_features),
            tempo=audio_features.get('tempo', 0),
            energy=audio_features.get('energy', 0),
            danceability=audio_features.get('danceability', 0)
        )
        
        # Guardar en cache
        await self.cache.set(cache_key, analysis.__dict__, ttl=3600)
        
        return analysis
    
    async def analyze_tracks_batch(self, track_ids: List[str]) -> List[TrackAnalysis]:
        """Analiza múltiples tracks en paralelo"""
        import asyncio
        tasks = [self.analyze_track(track_id) for track_id in track_ids]
        return await asyncio.gather(*tasks)
    
    def _get_key_signature(self, features: Dict[str, Any]) -> str:
        """Convierte key y mode a key signature"""
        key = features.get('key', 0)
        mode = features.get('mode', 0)
        # Lógica de conversión
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        modes = ['Minor', 'Major']
        return f"{keys[key]} {modes[mode]}"
```

## 5. Controllers Refactorizados

### `api/v1/controllers/analysis_controller.py`

```python
"""
Controller para endpoints de análisis
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from api.v1.schemas.analysis import (
    TrackAnalysisRequest,
    TrackAnalysisResponse
)
from api.v1.dependencies import get_analyze_track_use_case
from application.use_cases.analysis import AnalyzeTrackUseCase
from application.exceptions import TrackNotFoundException


router = APIRouter(prefix="/analysis", tags=["Analysis"])


@router.post("/", response_model=TrackAnalysisResponse)
async def analyze_track(
    request: TrackAnalysisRequest,
    use_case: AnalyzeTrackUseCase = Depends(get_analyze_track_use_case)
):
    """
    Analiza un track musical
    
    - **track_id**: ID del track a analizar
    - **include_coaching**: Si incluir coaching en la respuesta
    """
    try:
        result = await use_case.execute(
            track_id=request.track_id,
            include_coaching=request.include_coaching
        )
        
        return TrackAnalysisResponse(
            success=True,
            track_id=result.track_id,
            track_name=result.track_name,
            artists=result.artists,
            analysis=result.analysis.__dict__,
            coaching=result.coaching
        )
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing track: {str(e)}")
```

### `api/v1/dependencies.py`

```python
"""
Dependencias de FastAPI para inyección
"""
from fastapi import Depends
from core.di import get_container
from application.use_cases.analysis import AnalyzeTrackUseCase
from application.use_cases.recommendations import GetRecommendationsUseCase


def get_analyze_track_use_case() -> AnalyzeTrackUseCase:
    """Obtiene el caso de uso de análisis"""
    container = get_container()
    return container.resolve("analyze_track_use_case")


def get_recommendations_use_case() -> GetRecommendationsUseCase:
    """Obtiene el caso de uso de recomendaciones"""
    container = get_container()
    return container.resolve("recommendations_use_case")
```

## 6. Configuración de DI

### `config/di_setup.py`

```python
"""
Configuración de inyección de dependencias
"""
from core.di import register_service, get_container
from infrastructure.repositories.spotify_track_repository import SpotifyTrackRepository
from infrastructure.services.analysis_service import AnalysisService
from infrastructure.external.spotify.spotify_client import SpotifyClient
from infrastructure.cache.cache_manager import CacheManager
from application.use_cases.analysis import AnalyzeTrackUseCase
from application.use_cases.recommendations import GetRecommendationsUseCase


def setup_dependencies():
    """Configura todas las dependencias"""
    container = get_container()
    
    # Infrastructure Layer
    register_service("spotify_client", SpotifyClient, singleton=True)
    register_service("cache_manager", CacheManager, singleton=True)
    
    register_service(
        "track_repository",
        SpotifyTrackRepository,
        singleton=True,
        dependencies=["spotify_client"]
    )
    
    register_service(
        "analysis_service",
        AnalysisService,
        singleton=True,
        dependencies=["track_repository", "cache_manager"]
    )
    
    # Application Layer
    register_service(
        "analyze_track_use_case",
        AnalyzeTrackUseCase,
        singleton=True,
        dependencies=["track_repository", "analysis_service"]
    )
    
    register_service(
        "recommendations_use_case",
        GetRecommendationsUseCase,
        singleton=True,
        dependencies=["track_repository", "recommendation_service"]
    )
    
    logger.info("Dependencies configured successfully")
```

## 7. Tests Mejorados

### `tests/application/use_cases/test_analyze_track.py`

```python
"""
Tests para el caso de uso de análisis
"""
import pytest
from unittest.mock import Mock, AsyncMock
from application.use_cases.analysis import AnalyzeTrackUseCase
from application.exceptions import TrackNotFoundException
from domain.entities.track import Track


@pytest.fixture
def mock_track_repository():
    """Mock del repositorio de tracks"""
    repo = Mock()
    repo.get_by_id = AsyncMock(return_value=Track(
        id="123",
        name="Test Track",
        artists=["Test Artist"]
    ))
    return repo


@pytest.fixture
def mock_analysis_service():
    """Mock del servicio de análisis"""
    service = Mock()
    service.analyze_track = AsyncMock(return_value=Mock(
        key_signature="C Major",
        tempo=120.0
    ))
    return service


@pytest.fixture
def use_case(mock_track_repository, mock_analysis_service):
    """Caso de uso con mocks"""
    return AnalyzeTrackUseCase(
        track_repository=mock_track_repository,
        analysis_service=mock_analysis_service
    )


@pytest.mark.asyncio
async def test_analyze_track_success(use_case, mock_track_repository, mock_analysis_service):
    """Test de análisis exitoso"""
    result = await use_case.execute("123")
    
    assert result.track_id == "123"
    assert result.track_name == "Test Track"
    mock_track_repository.get_by_id.assert_called_once_with("123")
    mock_analysis_service.analyze_track.assert_called_once_with("123")


@pytest.mark.asyncio
async def test_analyze_track_not_found(use_case, mock_track_repository):
    """Test cuando el track no existe"""
    mock_track_repository.get_by_id = AsyncMock(return_value=None)
    
    with pytest.raises(TrackNotFoundException):
        await use_case.execute("999")
```

## 8. Migración Gradual

### Estrategia de Feature Flags

```python
# config/feature_flags.py
FEATURE_FLAGS = {
    "use_new_architecture": os.getenv("USE_NEW_ARCHITECTURE", "false").lower() == "true"
}

# api/v1/routes/analysis.py
from config.feature_flags import FEATURE_FLAGS

@router.post("/analyze")
async def analyze_track(request: TrackAnalysisRequest):
    if FEATURE_FLAGS["use_new_architecture"]:
        # Nueva arquitectura
        use_case = Depends(get_analyze_track_use_case)
        return await use_case.execute(request.track_id)
    else:
        # Arquitectura antigua (compatibilidad)
        return await legacy_analyze_track(request)
```

---

Estos ejemplos muestran cómo implementar las mejoras arquitectónicas propuestas de manera práctica y gradual.




